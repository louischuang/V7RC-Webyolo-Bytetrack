export type BrowserLlmStatus = "idle" | "checking" | "loading" | "ready" | "generating" | "error";

export type BrowserLlmMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type BrowserLlmConfig = {
  modelId: string;
  modelUrl: string;
  modelLibUrl: string;
  maxNewTokens: number;
  temperature: number;
};

export type BrowserLlmGeneration = {
  text: string;
  diagnostics: string[];
};

type WebLlmModule = typeof import("@mlc-ai/web-llm");
type MlcEngine = Awaited<ReturnType<WebLlmModule["CreateMLCEngine"]>>;
type AppConfig = import("@mlc-ai/web-llm").AppConfig;
type ChatChunk = {
  choices?: Array<{
    delta?: {
      content?: string;
    };
    finish_reason?: string | null;
  }>;
};
type ChatResponse = {
  choices?: Array<{
    message?: {
      content?: string | null;
    };
    finish_reason?: string | null;
  }>;
};
type WebGpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<{
      features: Set<string>;
      info?: {
        vendor?: string;
      };
    } | null>;
  };
};

export type LoadProgress = {
  text: string;
  progress?: number;
};

export async function getWebGpuStatus() {
  const webGpuNavigator = navigator as WebGpuNavigator;

  if (!webGpuNavigator.gpu) {
    return {
      ok: false,
      detail: "WebGPU is not exposed by this browser.",
    };
  }

  const adapter = await webGpuNavigator.gpu.requestAdapter();
  if (!adapter) {
    return {
      ok: false,
      detail: "No WebGPU adapter is available.",
    };
  }

  const features = [...adapter.features].sort();
  const hasShaderF16 = adapter.features.has("shader-f16");

  return {
    ok: hasShaderF16,
    detail: hasShaderF16
      ? `WebGPU ready: ${adapter.info?.vendor || "GPU"} / shader-f16`
      : `WebGPU adapter found, but shader-f16 is missing. Features: ${features.join(", ") || "none"}`,
  };
}

export class BrowserLlm {
  private engine: MlcEngine | null = null;

  constructor(private readonly config: BrowserLlmConfig) {}

  async load(onProgress: (progress: LoadProgress) => void) {
    const { CreateMLCEngine } = await import("@mlc-ai/web-llm");
    const modelUrl = toAbsoluteUrl(this.config.modelUrl);
    const modelLibUrl = toAbsoluteUrl(this.config.modelLibUrl);
    const appConfig: AppConfig = {
      model_list: [
        {
          model: modelUrl,
          model_id: this.config.modelId,
          model_lib: modelLibUrl,
          required_features: ["shader-f16"],
          overrides: {
            sliding_window_size: -1,
          },
        },
      ],
      cacheBackend: "indexeddb",
    };

    this.engine = await CreateMLCEngine(this.config.modelId, {
      appConfig,
      initProgressCallback: (progress: { text?: string; progress?: number }) => {
        onProgress({
          text: progress.text ?? "Loading model...",
          progress: progress.progress,
        });
      },
    });
  }

  async generate(messages: BrowserLlmMessage[]) {
    if (!this.engine) {
      throw new Error("Gemma model is not loaded.");
    }

    const diagnostics = [
      `model=${this.config.modelId}`,
      `messages=${messages.length}`,
      `max_tokens=${this.config.maxNewTokens}`,
    ];

    const chatResponse = (await this.engine.chat.completions.create({
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxNewTokens,
      stream: false,
    })) as ChatResponse;
    const chatText = chatResponse.choices?.[0]?.message?.content ?? "";
    diagnostics.push(
      `chat_nonstream len=${chatText.length} finish=${chatResponse.choices?.[0]?.finish_reason ?? "unknown"}`,
    );
    if (chatText.trim()) {
      return { text: chatText.trim(), diagnostics };
    }

    const chatStreamText = await this.generateFromChatStream(messages, diagnostics);
    if (chatStreamText.trim()) {
      return { text: chatStreamText.trim(), diagnostics };
    }

    const lastMessage = await this.engine.getMessage();
    diagnostics.push(`engine_getMessage len=${lastMessage.length}`);
    if (lastMessage.trim()) {
      return { text: lastMessage.trim(), diagnostics };
    }

    const rawStopText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_stop",
      stop: ["<turn|>"],
    });
    if (rawStopText.trim()) {
      return { text: rawStopText.trim(), diagnostics };
    }

    const rawNoStopText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_nostop",
    });
    if (rawNoStopText.trim()) {
      return { text: rawNoStopText.trim(), diagnostics };
    }

    const rawIgnoreEosText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_ignore_eos",
      ignoreEos: true,
      maxTokens: Math.min(64, this.config.maxNewTokens),
    });

    return { text: rawIgnoreEosText.trim(), diagnostics };
  }

  private async generateFromChatStream(messages: BrowserLlmMessage[], diagnostics: string[]) {
    if (!this.engine) {
      throw new Error("Gemma model is not loaded.");
    }

    const chunks = await this.engine.chat.completions.create({
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxNewTokens,
      stream: true,
    });

    let response = "";
    let chunkCount = 0;
    const finishReasons = new Set<string>();
    if (isAsyncIterable(chunks)) {
      for await (const chunk of chunks as AsyncIterable<ChatChunk>) {
        chunkCount += 1;
        response += chunk.choices?.[0]?.delta?.content ?? "";
        const finishReason = chunk.choices?.[0]?.finish_reason;
        if (finishReason) {
          finishReasons.add(finishReason);
        }
      }
    }

    diagnostics.push(
      `chat_stream chunks=${chunkCount} len=${response.length} finish=${[...finishReasons].join(",") || "none"}`,
    );
    return response.trim();
  }

  private async generateFromRawPrompt(
    messages: BrowserLlmMessage[],
    diagnostics: string[],
    options: { label: string; stop?: string[]; ignoreEos?: boolean; maxTokens?: number },
  ) {
    if (!this.engine) {
      throw new Error("Gemma model is not loaded.");
    }

    const prompt = toGemmaPrompt(messages);
    const chunks = await this.engine.completions.create({
      prompt,
      temperature: this.config.temperature,
      max_tokens: options.maxTokens ?? this.config.maxNewTokens,
      stream: true,
      stop: options.stop,
      ignore_eos: options.ignoreEos,
    });

    let response = "";
    let chunkCount = 0;
    const finishReasons = new Set<string>();
    if (isAsyncIterable(chunks)) {
      for await (const chunk of chunks as AsyncIterable<CompletionChunk>) {
        chunkCount += 1;
        response += chunk.choices?.[0]?.text ?? "";
        const finishReason = chunk.choices?.[0]?.finish_reason;
        if (finishReason) {
          finishReasons.add(finishReason);
        }
      }
    }

    const cleaned = cleanupGemmaResponse(response);
    diagnostics.push(
      `${options.label} chunks=${chunkCount} rawLen=${response.length} cleanLen=${cleaned.length} finish=${
        [...finishReasons].join(",") || "none"
      }`,
    );
    return cleaned;
  }
}

function isAsyncIterable(value: unknown): value is AsyncIterable<unknown> {
  return typeof value === "object" && value !== null && Symbol.asyncIterator in value;
}

function toAbsoluteUrl(url: string) {
  return new URL(url, window.location.origin).href;
}

type CompletionChunk = {
  choices?: Array<{
    text?: string;
    finish_reason?: string | null;
  }>;
};

function toGemmaPrompt(messages: BrowserLlmMessage[]) {
  const system = messages.find((message) => message.role === "system")?.content.trim();
  const turns = messages
    .filter((message) => message.role !== "system")
    .map((message) => {
      const role = message.role === "assistant" ? "model" : "user";
      return `<|turn>${role}\n${message.content.trim()}<turn|>\n`;
    })
    .join("");

  return `<bos>${system ? `${system}\n` : ""}${turns}<|turn>model\n`;
}

function cleanupGemmaResponse(response: string) {
  return response.replace(/<turn\|>[\s\S]*$/u, "").trim();
}
