import { AutoProcessor, Gemma4ForConditionalGeneration, TextStreamer, env, load_image } from "@huggingface/transformers";
import type { BrowserLlmMessage } from "./browser-llm";

type WorkerRequest =
  | {
      id: number;
      type: "load";
      modelId: string;
      device: "wasm" | "webgpu";
      maxNewTokens: number;
      temperature: number;
    }
  | {
      id: number;
      type: "generate";
      messages: BrowserLlmMessage[];
      imageDataUrl?: string;
      device: "wasm" | "webgpu";
      maxNewTokens: number;
      temperature: number;
    };

let processor: any = null;
let model: any = null;

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  void handleMessage(event.data);
};

async function handleMessage(request: WorkerRequest) {
  try {
    if (request.type === "load") {
      await loadModel(request);
      self.postMessage({ id: request.id, type: "loaded" });
      return;
    }

    const result = await generateText(request);
    self.postMessage({ id: request.id, type: "generated", result });
  } catch (error) {
    self.postMessage({
      id: request.id,
      type: "error",
      error: error instanceof Error ? error.message : "Unknown Transformers.js worker error.",
    });
  }
}

async function loadModel(request: Extract<WorkerRequest, { type: "load" }>) {
  env.allowLocalModels = true;
  env.allowRemoteModels = true;
  env.useBrowserCache = true;

  const progress_callback = (progress: { status?: string; file?: string; progress?: number }) => {
    self.postMessage({
      id: request.id,
      type: "progress",
      progress: {
        text: progress.file ? `${progress.status ?? "Loading"} ${progress.file}` : progress.status ?? "Loading model...",
        progress: typeof progress.progress === "number" ? progress.progress / 100 : undefined,
      },
    });
  };

  processor = await AutoProcessor.from_pretrained(request.modelId, {
    progress_callback,
  });
  model = await Gemma4ForConditionalGeneration.from_pretrained(request.modelId, {
    dtype: "q4f16",
    device: request.device,
    progress_callback,
  });
}

async function generateText(request: Extract<WorkerRequest, { type: "generate" }>) {
  if (!processor || !model) {
    throw new Error("Gemma ONNX model is not loaded.");
  }

  const hfMessages = request.messages.map((message, index) => {
    const isLastUserMessage = message.role === "user" && index === request.messages.length - 1;

    if (request.imageDataUrl && isLastUserMessage) {
      return {
        role: message.role,
        content: [
          { type: "image" },
          { type: "text", text: message.content },
        ],
      };
    }

    return {
      role: message.role,
      content: message.content,
    };
  });
  const diagnostics = [
    "runtime=transformers-worker",
    `device=${request.device}`,
    `messages=${request.messages.length}`,
    `image=${request.imageDataUrl ? "true" : "false"}`,
    `max_new_tokens=${request.maxNewTokens}`,
  ];
  const prompt = processor.apply_chat_template(hfMessages, {
    enable_thinking: false,
    add_generation_prompt: true,
  });
  diagnostics.push(`prompt_len=${String(prompt).length}`);

  const image = request.imageDataUrl ? await load_image(request.imageDataUrl) : null;
  const inputs = await processor(prompt, image, null, {
    add_special_tokens: false,
  });
  let streamed = "";
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false,
    callback_function: (text: string) => {
      streamed += text;
    },
  });

  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: request.maxNewTokens,
    do_sample: request.temperature > 0,
    temperature: request.temperature,
    top_p: 0.95,
    streamer,
  });
  const decoded = decodeOutput(outputs, inputs);
  diagnostics.push(`streamed_len=${streamed.length}`);
  diagnostics.push(`decoded_len=${decoded.length}`);

  return {
    text: cleanupGemmaResponse(decoded || streamed),
    diagnostics,
  };
}

function decodeOutput(outputs: any, inputs: any) {
  try {
    const inputLength = inputs.input_ids?.dims?.at?.(-1);
    if (typeof inputLength === "number" && typeof outputs?.slice === "function") {
      return processor.batch_decode(outputs.slice(null, [inputLength, null]), { skip_special_tokens: true })?.[0] ?? "";
    }
  } catch {
    // Use streamed text if tensor slicing is not available.
  }

  try {
    return processor.batch_decode(outputs, { skip_special_tokens: true })?.[0] ?? "";
  } catch {
    return "";
  }
}

function cleanupGemmaResponse(response: string) {
  return response
    .replace(/<turn\|>[\s\S]*$/u, "")
    .replace(/<think>[\s\S]*?<\/think>/gu, "")
    .replace(/<\|channel>thought[\s\S]*?<channel\|>/gu, "")
    .replace(/<eos>/gu, "")
    .trim();
}
