export type LaneSegmentationBackend = "webgpu" | "wasm";

export type LaneSegmentationConfig = {
  inputSize: number;
  modelUrl: string;
  provider: string;
  threshold: number;
};

export type SegmentationLanePath = Array<{ x: number; y: number }>;

export type LaneSegmentationResult = {
  confidence: number;
  paths: SegmentationLanePath[];
};

type OrtModule = typeof import("onnxruntime-web");
type InferenceSession = Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>;

type TensorLike = {
  data: unknown;
  dims: readonly number[];
};

export class LaneSegmentationModel {
  private constructor(
    private readonly ort: OrtModule,
    private readonly session: InferenceSession,
    private readonly config: LaneSegmentationConfig,
    readonly backend: LaneSegmentationBackend,
  ) {}

  static async create(config: LaneSegmentationConfig): Promise<LaneSegmentationModel> {
    const providers = parseProviders(config.provider);

    try {
      const ort = await import("onnxruntime-web/webgpu");
      const session = await ort.InferenceSession.create(config.modelUrl, {
        executionProviders: providers.includes("webgpu") ? ["webgpu", "wasm"] : ["wasm"],
        graphOptimizationLevel: "all",
      });
      return new LaneSegmentationModel(ort, session, config, providers.includes("webgpu") ? "webgpu" : "wasm");
    } catch (webgpuError) {
      console.warn("Lane segmentation WebGPU load failed, falling back to WASM.", webgpuError);
      const ort = await import("onnxruntime-web");
      const session = await ort.InferenceSession.create(config.modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return new LaneSegmentationModel(ort, session, config, "wasm");
    }
  }

  async segment(source: CanvasImageSource): Promise<LaneSegmentationResult> {
    const frame = preprocessSource(source, this.config.inputSize);
    const inputName = this.session.inputNames[0];
    const inputTensor = new this.ort.Tensor("float32", frame, [
      1,
      3,
      this.config.inputSize,
      this.config.inputSize,
    ]);
    const outputs = await this.session.run({ [inputName]: inputTensor });
    const output = outputs[this.session.outputNames[0]] as TensorLike | undefined;

    if (!output || !(output.data instanceof Float32Array)) {
      return { confidence: 0, paths: [] };
    }

    return decodeSegmentationMask(output.data, output.dims, this.config.threshold);
  }
}

function parseProviders(provider: string) {
  const providers = provider
    .split(",")
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);

  return providers.length > 0 ? providers : ["webgpu", "wasm"];
}

function preprocessSource(source: CanvasImageSource, inputSize: number) {
  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create lane segmentation preprocessing context.");
  }

  context.fillStyle = "rgb(0, 0, 0)";
  context.fillRect(0, 0, inputSize, inputSize);
  context.drawImage(source, 0, 0, inputSize, inputSize);

  const pixels = context.getImageData(0, 0, inputSize, inputSize).data;
  const planeSize = inputSize * inputSize;
  const data = new Float32Array(planeSize * 3);

  for (let i = 0, pixel = 0; i < pixels.length; i += 4, pixel += 1) {
    data[pixel] = pixels[i] / 255;
    data[pixel + planeSize] = pixels[i + 1] / 255;
    data[pixel + planeSize * 2] = pixels[i + 2] / 255;
  }

  return data;
}

function decodeSegmentationMask(data: Float32Array, dims: readonly number[], threshold: number): LaneSegmentationResult {
  const shape = inferMaskShape(dims, data.length);
  if (!shape) {
    return { confidence: 0, paths: [] };
  }

  const paths: SegmentationLanePath[] = [];
  let hits = 0;
  const columnCount = 18;

  for (let column = 0; column < columnCount; column += 1) {
    const points: SegmentationLanePath = [];
    const startX = Math.floor((column / columnCount) * shape.width);
    const endX = Math.min(shape.width - 1, Math.ceil(((column + 1) / columnCount) * shape.width));

    for (let y = Math.floor(shape.height * 0.05); y < Math.floor(shape.height * 0.98); y += 3) {
      let bestX = -1;
      let bestScore = threshold;

      for (let x = startX; x <= endX; x += 1) {
        const score = readMaskValue(data, shape, x, y);
        if (score > bestScore) {
          bestScore = score;
          bestX = x;
        }
      }

      if (bestX >= 0) {
        hits += 1;
        points.push({ x: bestX / Math.max(1, shape.width - 1), y: y / Math.max(1, shape.height - 1) });
      }
    }

    if (points.length >= 6) {
      paths.push(points);
    }
  }

  const expectedHits = Math.max(1, Math.floor(shape.height / 3) * 2.8);
  return {
    confidence: Math.max(0, Math.min(1, hits / expectedHits)),
    paths,
  };
}

function inferMaskShape(dims: readonly number[], dataLength: number) {
  if (dims.length === 4) {
    const [, channels, height, width] = dims;
    return { channels: Math.max(1, channels), height, width };
  }

  if (dims.length === 3) {
    const [a, b, c] = dims;
    return a <= 4 ? { channels: a, height: b, width: c } : { channels: 1, height: a, width: b };
  }

  const side = Math.sqrt(dataLength);
  if (Number.isInteger(side)) {
    return { channels: 1, height: side, width: side };
  }

  return null;
}

function readMaskValue(
  data: Float32Array,
  shape: {
    channels: number;
    height: number;
    width: number;
  },
  x: number,
  y: number,
) {
  const laneChannel = shape.channels > 1 ? shape.channels - 1 : 0;
  const offset = laneChannel * shape.width * shape.height + y * shape.width + x;
  return data[offset] ?? 0;
}
