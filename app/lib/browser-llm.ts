export type BrowserLlmStatus = "idle" | "checking" | "loading" | "ready" | "generating" | "error";

export type BrowserLlmMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type BrowserLlmConfig = {
  runtime: "webllm" | "transformers";
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
type WorkerResponse =
  | { id: number; type: "progress"; progress: LoadProgress }
  | { id: number; type: "loaded" }
  | { id: number; type: "generated"; result: BrowserLlmGeneration }
  | { id: number; type: "error"; error: string };
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
  private worker: Worker | null = null;
  private workerRequestId = 0;
  private workerRequests = new Map<
    number,
    {
      resolve: (value: unknown) => void;
      reject: (reason?: unknown) => void;
      onProgress?: (progress: LoadProgress) => void;
    }
  >();

  constructor(private readonly config: BrowserLlmConfig) {}

  async load(onProgress: (progress: LoadProgress) => void) {
    if (this.config.runtime === "transformers") {
      await this.loadTransformers(onProgress);
      return;
    }

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
            context_window_size: -1,
            attention_sink_size: 0,
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

  async generate(messages: BrowserLlmMessage[], imageDataUrl?: string) {
    if (this.config.runtime === "transformers") {
      return this.generateWithTransformers(messages, imageDataUrl);
    }

    if (!this.engine) {
      throw new Error("Gemma model is not loaded.");
    }

    await this.engine.resetChat(false, this.config.modelId);

    const diagnostics = [
      `model=${this.config.modelId}`,
      `messages=${messages.length}`,
      `max_tokens=${this.config.maxNewTokens}`,
      "resetChat=true",
    ];
    const candidates: string[] = [];

    const chatResponse = (await this.engine.chat.completions.create({
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxNewTokens,
      extra_body: {
        enable_thinking: false,
      },
      stream: false,
    })) as ChatResponse;
    const chatText = chatResponse.choices?.[0]?.message?.content ?? "";
    diagnostics.push(
      `chat_nonstream len=${chatText.length} finish=${chatResponse.choices?.[0]?.finish_reason ?? "unknown"}`,
    );
    const chatCandidate = cleanupGemmaResponse(chatText);
    if (isUsableCandidate(chatCandidate)) {
      return { text: chatCandidate, diagnostics };
    }
    if (chatCandidate) {
      candidates.push(chatCandidate);
    }

    const chatStreamText = cleanupGemmaResponse(await this.generateFromChatStream(messages, diagnostics));
    if (isUsableCandidate(chatStreamText)) {
      return { text: chatStreamText, diagnostics };
    }
    if (chatStreamText) {
      candidates.push(chatStreamText);
    }

    const lastMessage = await this.engine.getMessage();
    diagnostics.push(`engine_getMessage len=${lastMessage.length}`);
    const lastMessageCandidate = cleanupGemmaResponse(lastMessage);
    if (isUsableCandidate(lastMessageCandidate)) {
      return { text: lastMessageCandidate, diagnostics };
    }
    if (lastMessageCandidate) {
      candidates.push(lastMessageCandidate);
    }

    const rawStopText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_stop",
      stop: ["<turn|>"],
    });
    if (isUsableCandidate(rawStopText)) {
      return { text: rawStopText, diagnostics };
    }
    if (rawStopText) {
      candidates.push(rawStopText);
    }

    const rawNoStopText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_nostop",
    });
    if (isUsableCandidate(rawNoStopText)) {
      return { text: rawNoStopText, diagnostics };
    }
    if (rawNoStopText) {
      candidates.push(rawNoStopText);
    }

    const rawIgnoreEosText = await this.generateFromRawPrompt(messages, diagnostics, {
      label: "raw_stream_ignore_eos",
      ignoreEos: true,
      maxTokens: Math.min(64, this.config.maxNewTokens),
    });
    if (rawIgnoreEosText) {
      candidates.push(rawIgnoreEosText);
    }

    return { text: chooseBestCandidate(candidates), diagnostics };
  }

  private async loadTransformers(onProgress: (progress: LoadProgress) => void) {
    const modelId = this.config.modelUrl || this.config.modelId;
    this.ensureWorker();
    await this.postWorkerRequest(
      {
        type: "load",
        modelId,
        maxNewTokens: this.config.maxNewTokens,
        temperature: this.config.temperature,
      },
      onProgress,
    );
  }

  private async generateWithTransformers(messages: BrowserLlmMessage[], imageDataUrl?: string): Promise<BrowserLlmGeneration> {
    this.ensureWorker();
    return (await this.postWorkerRequest({
      type: "generate",
      messages,
      imageDataUrl,
      maxNewTokens: this.config.maxNewTokens,
      temperature: this.config.temperature,
    })) as BrowserLlmGeneration;
  }

  private async generateFromChatStream(messages: BrowserLlmMessage[], diagnostics: string[]) {
    if (!this.engine) {
      throw new Error("Gemma model is not loaded.");
    }

    const chunks = await this.engine.chat.completions.create({
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxNewTokens,
      extra_body: {
        enable_thinking: false,
      },
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
      } preview=${previewText(response)}`,
    );
    return cleaned;
  }

  private handleWorkerMessage(event: MessageEvent<WorkerResponse>) {
    const request = this.workerRequests.get(event.data.id);
    if (!request) {
      return;
    }

    if (event.data.type === "progress") {
      request.onProgress?.(event.data.progress);
      return;
    }

    this.workerRequests.delete(event.data.id);
    if (event.data.type === "error") {
      request.reject(new Error(event.data.error));
      return;
    }

    if (event.data.type === "generated") {
      request.resolve(event.data.result);
      return;
    }

    request.resolve(undefined);
  }

  private ensureWorker() {
    if (this.worker) {
      return;
    }

    this.worker = new Worker(new URL("./transformers-gemma-worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
    this.worker.onerror = (event: ErrorEvent) => {
      for (const request of this.workerRequests.values()) {
        request.reject(new Error(event.message || "Transformers.js worker failed."));
      }
      this.workerRequests.clear();
    };
  }

  private postWorkerRequest(payload: Record<string, unknown>, onProgress?: (progress: LoadProgress) => void) {
    this.ensureWorker();
    const id = ++this.workerRequestId;

    return new Promise((resolve, reject) => {
      this.workerRequests.set(id, { resolve, reject, onProgress });
      this.worker?.postMessage({ id, ...payload });
    });
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

  return `<bos>${system ? `<|turn>system\n${system}<turn|>\n` : ""}${turns}<|turn>model\n<|channel>thought\n<channel|>`;
}

function cleanupGemmaResponse(response: string) {
  return response
    .replace(/<turn\|>[\s\S]*$/u, "")
    .replace(/<think>[\s\S]*?<\/think>/gu, "")
    .replace(/<\|channel>thought[\s\S]*?<channel\|>/gu, "")
    .replace(/<eos>/gu, "")
    .trim();
}

function decodeTransformersOutput(processor: any, outputs: any, inputs: any) {
  try {
    const inputLength = inputs.input_ids?.dims?.at?.(-1);
    if (typeof inputLength === "number" && typeof outputs?.slice === "function") {
      const generated = outputs.slice(null, [inputLength, null]);
      return processor.batch_decode(generated, { skip_special_tokens: true })?.[0] ?? "";
    }
  } catch {
    // Fall back to streamed text below.
  }

  try {
    return processor.batch_decode(outputs, { skip_special_tokens: true })?.[0] ?? "";
  } catch {
    return "";
  }
}

function isUsableCandidate(response: string) {
  return response.trim().length >= 16;
}

function chooseBestCandidate(candidates: string[]) {
  return [...candidates].sort((left, right) => right.length - left.length)[0] ?? "";
}

function previewText(response: string) {
  return JSON.stringify(response.slice(0, 160));
}
