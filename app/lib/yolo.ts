import { COCO_LABELS } from "./coco";

export type Detection = {
  id: string;
  classId: number;
  label: string;
  confidence: number;
  box: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export type DetectorBackend = "webgpu" | "wasm";

export type DetectorConfig = {
  modelUrl: string;
  inputSize: number;
  confidenceThreshold: number;
  iouThreshold: number;
};

type OrtModule = typeof import("onnxruntime-web");
type InferenceSession = Awaited<ReturnType<OrtModule["InferenceSession"]["create"]>>;

type PreprocessResult = {
  data: Float32Array;
  scale: number;
  padX: number;
  padY: number;
  sourceWidth: number;
  sourceHeight: number;
};

export class YoloDetector {
  private constructor(
    private readonly ort: OrtModule,
    private readonly session: InferenceSession,
    private readonly config: DetectorConfig,
    readonly backend: DetectorBackend,
  ) {}

  static async create(config: DetectorConfig): Promise<YoloDetector> {
    try {
      const ort = await import("onnxruntime-web/webgpu");
      const session = await ort.InferenceSession.create(config.modelUrl, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });
      return new YoloDetector(ort, session, config, "webgpu");
    } catch (webgpuError) {
      console.warn("YOLO WebGPU load failed, falling back to WASM.", webgpuError);
      const ort = await import("onnxruntime-web");
      const session = await ort.InferenceSession.create(config.modelUrl, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
      return new YoloDetector(ort, session, config, "wasm");
    }
  }

  async detect(video: HTMLVideoElement): Promise<Detection[]> {
    const frame = preprocessVideoFrame(video, this.config.inputSize);
    const inputName = this.session.inputNames[0];
    const inputTensor = new this.ort.Tensor("float32", frame.data, [
      1,
      3,
      this.config.inputSize,
      this.config.inputSize,
    ]);

    const outputs = await this.session.run({ [inputName]: inputTensor });
    const outputName = this.session.outputNames[0];
    const output = outputs[outputName];

    if (!output || !(output.data instanceof Float32Array)) {
      return [];
    }

    return decodeDetections(
      output.data,
      output.dims,
      frame,
      this.config.confidenceThreshold,
      this.config.iouThreshold,
    );
  }
}

function preprocessVideoFrame(video: HTMLVideoElement, inputSize: number): PreprocessResult {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  const scale = Math.min(inputSize / sourceWidth, inputSize / sourceHeight);
  const drawWidth = Math.round(sourceWidth * scale);
  const drawHeight = Math.round(sourceHeight * scale);
  const padX = Math.floor((inputSize - drawWidth) / 2);
  const padY = Math.floor((inputSize - drawHeight) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = inputSize;
  canvas.height = inputSize;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Could not create preprocessing canvas context.");
  }

  context.fillStyle = "rgb(114, 114, 114)";
  context.fillRect(0, 0, inputSize, inputSize);
  context.drawImage(video, padX, padY, drawWidth, drawHeight);

  const pixels = context.getImageData(0, 0, inputSize, inputSize).data;
  const planeSize = inputSize * inputSize;
  const data = new Float32Array(planeSize * 3);

  for (let i = 0, pixel = 0; i < pixels.length; i += 4, pixel += 1) {
    data[pixel] = pixels[i] / 255;
    data[pixel + planeSize] = pixels[i + 1] / 255;
    data[pixel + planeSize * 2] = pixels[i + 2] / 255;
  }

  return { data, scale, padX, padY, sourceWidth, sourceHeight };
}

function decodeDetections(
  data: Float32Array,
  dims: readonly number[],
  frame: PreprocessResult,
  confidenceThreshold: number,
  iouThreshold: number,
): Detection[] {
  const candidates = decodeYoloTensor(data, dims, frame, confidenceThreshold);
  return nonMaxSuppression(candidates, iouThreshold).map((detection, index) => ({
    ...detection,
    id: `D${index + 1}`,
  }));
}

function decodeYoloTensor(
  data: Float32Array,
  dims: readonly number[],
  frame: PreprocessResult,
  confidenceThreshold: number,
): Detection[] {
  if (dims.length !== 3) {
    return decodeFlatDetections(data, frame, confidenceThreshold);
  }

  const [, dimA, dimB] = dims;
  const transposed = dimA < dimB;
  const boxCount = transposed ? dimB : dimA;
  const featureCount = transposed ? dimA : dimB;
  const detections: Detection[] = [];

  for (let boxIndex = 0; boxIndex < boxCount; boxIndex += 1) {
    const read = (featureIndex: number) =>
      transposed ? data[featureIndex * boxCount + boxIndex] : data[boxIndex * featureCount + featureIndex];

    if (featureCount < 6) {
      continue;
    }

    const cx = read(0);
    const cy = read(1);
    const width = read(2);
    const height = read(3);
    const classOffset = featureCount === 6 ? 5 : 4;
    const objectness = featureCount === 6 ? read(4) : 1;

    let classId = 0;
    let classScore = 0;
    for (let featureIndex = classOffset; featureIndex < featureCount; featureIndex += 1) {
      const score = read(featureIndex);
      if (score > classScore) {
        classScore = score;
        classId = featureIndex - classOffset;
      }
    }

    const confidence = objectness * classScore;
    if (confidence < confidenceThreshold) {
      continue;
    }

    const x1 = (cx - width / 2 - frame.padX) / frame.scale;
    const y1 = (cy - height / 2 - frame.padY) / frame.scale;
    const x2 = (cx + width / 2 - frame.padX) / frame.scale;
    const y2 = (cy + height / 2 - frame.padY) / frame.scale;
    detections.push(toDetection(classId, confidence, x1, y1, x2, y2, frame));
  }

  return detections;
}

function decodeFlatDetections(
  data: Float32Array,
  frame: PreprocessResult,
  confidenceThreshold: number,
): Detection[] {
  const stride = 6;
  const detections: Detection[] = [];

  for (let offset = 0; offset + stride <= data.length; offset += stride) {
    const confidence = data[offset + 4];
    if (confidence < confidenceThreshold) {
      continue;
    }

    detections.push(
      toDetection(
        Math.max(0, Math.round(data[offset + 5])),
        confidence,
        (data[offset] - frame.padX) / frame.scale,
        (data[offset + 1] - frame.padY) / frame.scale,
        (data[offset + 2] - frame.padX) / frame.scale,
        (data[offset + 3] - frame.padY) / frame.scale,
        frame,
      ),
    );
  }

  return detections;
}

function toDetection(
  classId: number,
  confidence: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  frame: PreprocessResult,
): Detection {
  const x = clamp(Math.min(x1, x2), 0, frame.sourceWidth);
  const y = clamp(Math.min(y1, y2), 0, frame.sourceHeight);
  const right = clamp(Math.max(x1, x2), 0, frame.sourceWidth);
  const bottom = clamp(Math.max(y1, y2), 0, frame.sourceHeight);

  return {
    id: "",
    classId,
    label: COCO_LABELS[classId] ?? `class ${classId}`,
    confidence,
    box: {
      x,
      y,
      width: Math.max(0, right - x),
      height: Math.max(0, bottom - y),
    },
  };
}

function nonMaxSuppression(detections: Detection[], iouThreshold: number) {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const selected: Detection[] = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
    if (!current) {
      break;
    }

    selected.push(current);

    for (let index = sorted.length - 1; index >= 0; index -= 1) {
      if (current.classId === sorted[index].classId && intersectionOverUnion(current, sorted[index]) > iouThreshold) {
        sorted.splice(index, 1);
      }
    }
  }

  return selected;
}

function intersectionOverUnion(a: Detection, b: Detection) {
  const ax2 = a.box.x + a.box.width;
  const ay2 = a.box.y + a.box.height;
  const bx2 = b.box.x + b.box.width;
  const by2 = b.box.y + b.box.height;
  const x1 = Math.max(a.box.x, b.box.x);
  const y1 = Math.max(a.box.y, b.box.y);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.box.width * a.box.height + b.box.width * b.box.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
