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

type WebLlmModule = typeof import("@mlc-ai/web-llm");
type MlcEngine = Awaited<ReturnType<WebLlmModule["CreateMLCEngine"]>>;
type AppConfig = import("@mlc-ai/web-llm").AppConfig;
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
    const appConfig: AppConfig = {
      model_list: [
        {
          model: this.config.modelUrl,
          model_id: this.config.modelId,
          model_lib: this.config.modelLibUrl,
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

    const reply = await this.engine.chat.completions.create({
      messages,
      temperature: this.config.temperature,
      max_tokens: this.config.maxNewTokens,
    });

    const content = reply.choices[0]?.message.content;
    if (typeof content === "string") {
      return content;
    }

    return "";
  }
}
