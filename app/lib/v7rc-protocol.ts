export const v7rcBleUuids = {
  service: "6e400001-b5a3-f393-e0a9-e50e24dcca9e",
  rx: "6e400002-b5a3-f393-e0a9-e50e24dcca9e",
  tx: "6e400003-b5a3-f393-e0a9-e50e24dcca9e",
} as const;

export type V7rcCommand = "HEX" | "DEG" | "SRV" | "SR2" | "SRT" | "CMD";
export type V7rcDriveMode = "vehicle" | "mecanum" | "tank";

export type V7rcRobotIntent = {
  linear: number;
  turn: number;
  strafe: number;
  speedScale: number;
  arm: {
    base: number;
    shoulder: number;
    elbow: number;
    wrist: number;
    gripper: number;
  };
  autonomy: boolean;
  neutral: boolean;
  emergencyStop: boolean;
};

export type V7rcChannelConfig = {
  neutralUs: number;
  minUs: number;
  maxUs: number;
  deadband: number;
};

export type V7rcChannelPreview = {
  index: number;
  logical: string;
  normalized: number;
  byteValue?: number;
  pwmUs: number;
};

const encoder = new TextEncoder();

const defaultChannelConfig: V7rcChannelConfig = {
  deadband: 0.03,
  maxUs: 2000,
  minUs: 1000,
  neutralUs: 1500,
};

const logicalChannels = [
  "drive throttle",
  "steering / yaw",
  "strafe / lateral",
  "speed scale / mode",
  "arm base yaw",
  "arm shoulder",
  "arm elbow",
  "wrist",
  "gripper",
  "autonomy enable",
  "neutral / brake",
  "emergency stop",
  "reserved 12",
  "reserved 13",
  "reserved 14",
  "reserved 15",
] as const;

export function createNeutralIntent(): V7rcRobotIntent {
  return {
    arm: {
      base: 0,
      elbow: 0,
      gripper: 0,
      shoulder: 0,
      wrist: 0,
    },
    autonomy: false,
    emergencyStop: false,
    linear: 0,
    neutral: true,
    speedScale: 0,
    strafe: 0,
    turn: 0,
  };
}

export function encodeHexFrame(channels: number[]) {
  if (channels.length !== 16) {
    throw new Error("HEX command requires exactly 16 channel bytes.");
  }

  const frame = new Uint8Array(20);
  frame.set(encoder.encode("HEX"), 0);
  channels.forEach((value, index) => {
    frame[index + 3] = clampByte(value);
  });
  frame[19] = "#".charCodeAt(0);
  return frame;
}

export function encodeDegFrame(degrees: number[]) {
  if (degrees.length !== 16) {
    throw new Error("DEG command requires exactly 16 channel values.");
  }

  const frame = new Uint8Array(20);
  frame.set(encoder.encode("DEG"), 0);
  degrees.forEach((value, index) => {
    frame[index + 3] = clampByte(Math.round(clamp(value, -90, 90) + 127));
  });
  frame[19] = "#".charCodeAt(0);
  return frame;
}

export function encodePwmTextFrame(command: "SRV" | "SR2" | "SRT", pwmUs: number[]) {
  if (pwmUs.length !== 4) {
    throw new Error(`${command} command requires exactly 4 PWM values.`);
  }

  return encodeAsciiFrame(`${command}${pwmUs.map((value) => formatPwm(value)).join("")}#`);
}

export function encodeSrtFrame(pwmUs: number[]) {
  return encodePwmTextFrame("SRT", pwmUs);
}

export function encodeCmdFrame(command: string) {
  const trimmed = command.slice(0, 16);
  return encodeAsciiFrame(`CMD${trimmed.padEnd(16, " ")}#`);
}

export function intentToHexChannels(intent: V7rcRobotIntent, config: V7rcChannelConfig = defaultChannelConfig) {
  const speedScale = clamp(intent.speedScale, 0, 1);
  const motionScale = intent.emergencyStop || intent.neutral || !intent.autonomy ? 0 : speedScale;
  const normalized = [
    intent.linear * motionScale,
    intent.turn * motionScale,
    intent.strafe * motionScale,
    speedScale,
    intent.arm.base,
    intent.arm.shoulder,
    intent.arm.elbow,
    intent.arm.wrist,
    intent.arm.gripper,
    intent.autonomy ? 1 : 0,
    intent.neutral ? 1 : 0,
    intent.emergencyStop ? 1 : 0,
    0,
    0,
    0,
    0,
  ];

  return normalized.map((value, index) => normalizedToByte(value, configForChannel(index, config)));
}

export function previewHexChannels(intent: V7rcRobotIntent, config: V7rcChannelConfig = defaultChannelConfig): V7rcChannelPreview[] {
  const bytes = intentToHexChannels(intent, config);

  return bytes.map((byteValue, index) => ({
    byteValue,
    index,
    logical: logicalChannels[index],
    normalized: byteToNormalized(byteValue, configForChannel(index, config)),
    pwmUs: byteValue * 10,
  }));
}

export function intentToSrtPwm(
  intent: V7rcRobotIntent,
  mode: V7rcDriveMode,
  config: V7rcChannelConfig = defaultChannelConfig,
) {
  const speedScale = clamp(intent.speedScale, 0, 1);
  const motionScale = intent.emergencyStop || intent.neutral || !intent.autonomy ? 0 : speedScale;
  const throttle = intent.linear * motionScale;
  const steering = intent.turn * motionScale;
  const strafe = intent.strafe * motionScale;

  if (mode === "mecanum") {
    return [
      normalizedToPwm(strafe, config),
      normalizedToPwm(throttle, config),
      normalizedToPwm(steering, config),
      config.neutralUs,
    ];
  }

  return [
    normalizedToPwm(steering, config),
    normalizedToPwm(throttle, config),
    config.neutralUs,
    config.neutralUs,
  ];
}

export function previewSrtChannels(
  intent: V7rcRobotIntent,
  mode: V7rcDriveMode,
  config: V7rcChannelConfig = defaultChannelConfig,
): V7rcChannelPreview[] {
  const pwmValues = intentToSrtPwm(intent, mode, config);
  const labels = srtLogicalChannelsForMode(mode);

  return pwmValues.map((pwmUs, index) => ({
    index,
    logical: labels[index],
    normalized: pwmToNormalized(pwmUs, config),
    pwmUs,
  }));
}

export function frameToDebugString(frame: Uint8Array) {
  return Array.from(frame)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join(" ");
}

function encodeAsciiFrame(value: string) {
  const frame = encoder.encode(value);
  if (frame.length > 20) {
    throw new Error("V7RC command must be 20 bytes or less.");
  }
  if (!value.endsWith("#")) {
    throw new Error("V7RC command must end with #.");
  }
  return frame;
}

function formatPwm(value: number) {
  return Math.round(clamp(value, 500, 2500)).toString().padStart(4, "0");
}

function normalizedToByte(value: number, config: V7rcChannelConfig) {
  return clampByte(Math.round(normalizedToPwm(value, config) / 10));
}

function byteToNormalized(value: number, config: V7rcChannelConfig) {
  const pwmUs = clampByte(value) * 10;
  if (pwmUs === config.neutralUs) {
    return 0;
  }

  const span = pwmUs > config.neutralUs ? config.maxUs - config.neutralUs : config.neutralUs - config.minUs;
  return Number(((pwmUs - config.neutralUs) / span).toFixed(2));
}

function normalizedToPwm(value: number, config: V7rcChannelConfig) {
  const clamped = Math.abs(value) < config.deadband ? 0 : clamp(value, -1, 1);
  const span = clamped >= 0 ? config.maxUs - config.neutralUs : config.neutralUs - config.minUs;
  return Math.round(config.neutralUs + clamped * span);
}

function pwmToNormalized(value: number, config: V7rcChannelConfig) {
  const pwmUs = Math.round(clamp(value, config.minUs, config.maxUs));
  if (pwmUs === config.neutralUs) {
    return 0;
  }

  const span = pwmUs > config.neutralUs ? config.maxUs - config.neutralUs : config.neutralUs - config.minUs;
  return Number(((pwmUs - config.neutralUs) / span).toFixed(2));
}

function srtLogicalChannelsForMode(mode: V7rcDriveMode) {
  if (mode === "mecanum") {
    return ["strafe left/right", "throttle forward/reverse", "steering / yaw", "reserved"];
  }

  if (mode === "tank") {
    return ["turn right/left", "throttle forward/reverse", "reserved", "reserved"];
  }

  return ["steering wheel", "throttle forward/reverse", "reserved", "reserved"];
}

function configForChannel(index: number, config: V7rcChannelConfig): V7rcChannelConfig {
  if (index === 3 || index === 9 || index === 10 || index === 11) {
    return {
      ...config,
      minUs: 1000,
      neutralUs: 1000,
    };
  }

  return config;
}

function clampByte(value: number) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function clamp(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) {
    return 0;
  }

  return Math.max(min, Math.min(max, value));
}
