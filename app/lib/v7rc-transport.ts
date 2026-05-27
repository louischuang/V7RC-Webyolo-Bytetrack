import { v7rcBleUuids } from "./v7rc-protocol";

type BluetoothRemoteGATTCharacteristicLike = {
  properties?: {
    notify?: boolean;
    write?: boolean;
    writeWithoutResponse?: boolean;
  };
  startNotifications?: () => Promise<BluetoothRemoteGATTCharacteristicLike>;
  writeValue?: (value: BufferSource) => Promise<void>;
  writeValueWithoutResponse?: (value: BufferSource) => Promise<void>;
  addEventListener?: (type: string, listener: EventListener) => void;
};

type BluetoothRemoteGATTServiceLike = {
  getCharacteristic: (uuid: string) => Promise<BluetoothRemoteGATTCharacteristicLike>;
};

type BluetoothRemoteGATTServerLike = {
  connected: boolean;
  connect: () => Promise<BluetoothRemoteGATTServerLike>;
  disconnect: () => void;
  getPrimaryService: (uuid: string) => Promise<BluetoothRemoteGATTServiceLike>;
};

type BluetoothDeviceLike = EventTarget & {
  gatt?: BluetoothRemoteGATTServerLike;
  name?: string;
};

type BluetoothNavigatorLike = Navigator & {
  bluetooth?: {
    requestDevice: (options: {
      filters: Array<{ services: string[] }>;
      optionalServices?: string[];
    }) => Promise<BluetoothDeviceLike>;
  };
};

export type V7rcTransportStatus = {
  connected: boolean;
  deviceName: string;
  lastPacket: string;
  lastMessage: string;
  mode: "mock" | "web-bluetooth";
};

export type V7rcTransport = {
  connect(): Promise<V7rcTransportStatus>;
  disconnect(): Promise<V7rcTransportStatus>;
  write(frame: Uint8Array): Promise<V7rcTransportStatus>;
  getStatus(): V7rcTransportStatus;
};

export function canUseWebBluetooth() {
  if (typeof navigator === "undefined") {
    return false;
  }

  return Boolean((navigator as BluetoothNavigatorLike).bluetooth);
}

export function createMockV7rcTransport(): V7rcTransport {
  let status: V7rcTransportStatus = {
    connected: false,
    deviceName: "Mock V7RC",
    lastMessage: "Mock transport is idle.",
    lastPacket: "",
    mode: "mock",
  };

  return {
    async connect() {
      status = { ...status, connected: true, lastMessage: "Mock robot connected." };
      return status;
    },
    async disconnect() {
      status = { ...status, connected: false, lastMessage: "Mock robot disconnected." };
      return status;
    },
    getStatus() {
      return status;
    },
    async write(frame) {
      if (!status.connected) {
        throw new Error("Mock robot is not connected.");
      }

      status = {
        ...status,
        lastMessage: `Mock wrote ${frame.length} bytes.`,
        lastPacket: packetToHex(frame),
      };
      return status;
    },
  };
}

export function createWebBluetoothV7rcTransport(onMessage?: (message: string) => void): V7rcTransport {
  let device: BluetoothDeviceLike | null = null;
  let server: BluetoothRemoteGATTServerLike | null = null;
  let rx: BluetoothRemoteGATTCharacteristicLike | null = null;
  let tx: BluetoothRemoteGATTCharacteristicLike | null = null;
  let status: V7rcTransportStatus = {
    connected: false,
    deviceName: "V7RC Robot",
    lastMessage: "Bluetooth transport is idle.",
    lastPacket: "",
    mode: "web-bluetooth",
  };

  const update = (next: Partial<V7rcTransportStatus>) => {
    status = { ...status, ...next };
    return status;
  };

  return {
    async connect() {
      const bluetooth = (navigator as BluetoothNavigatorLike).bluetooth;
      if (!bluetooth) {
        throw new Error("Web Bluetooth is not available in this browser.");
      }

      device = await bluetooth.requestDevice({
        filters: [{ services: [v7rcBleUuids.service] }],
        optionalServices: [v7rcBleUuids.service],
      });

      if (!device.gatt) {
        throw new Error("Selected device does not expose GATT.");
      }

      device.addEventListener("gattserverdisconnected", () => {
        rx = null;
        tx = null;
        server = null;
        update({ connected: false, lastMessage: "Robot disconnected." });
      });

      server = await device.gatt.connect();
      const service = await server.getPrimaryService(v7rcBleUuids.service);
      rx = await service.getCharacteristic(v7rcBleUuids.rx);
      tx = await service.getCharacteristic(v7rcBleUuids.tx).catch(() => null);

      if (tx?.properties?.notify && tx.startNotifications) {
        const notifier = await tx.startNotifications();
        notifier.addEventListener?.("characteristicvaluechanged", (event) => {
          const value = (event.target as { value?: DataView } | null)?.value;
          const message = value
            ? new TextDecoder().decode(new Uint8Array(value.buffer, value.byteOffset, value.byteLength))
            : "";
          if (message) {
            onMessage?.(message);
            update({ lastMessage: message });
          }
        });
      }

      return update({
        connected: true,
        deviceName: device.name || "V7RC Robot",
        lastMessage: "Robot connected.",
      });
    },
    async disconnect() {
      server?.disconnect();
      rx = null;
      tx = null;
      server = null;
      return update({ connected: false, lastMessage: "Robot disconnected." });
    },
    getStatus() {
      return status;
    },
    async write(frame) {
      if (!server?.connected || !rx) {
        throw new Error("Robot is not connected.");
      }

      const packet = toArrayBuffer(frame);
      if (rx.properties?.writeWithoutResponse && rx.writeValueWithoutResponse) {
        await rx.writeValueWithoutResponse(packet);
      } else if (rx.writeValue) {
        await rx.writeValue(packet);
      } else {
        throw new Error("RX characteristic does not support writes.");
      }

      return update({
        lastMessage: `Wrote ${frame.length} bytes.`,
        lastPacket: packetToHex(frame),
      });
    },
  };
}

function packetToHex(frame: Uint8Array) {
  return Array.from(frame)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join(" ");
}

function toArrayBuffer(frame: Uint8Array) {
  return frame.buffer.slice(frame.byteOffset, frame.byteOffset + frame.byteLength) as ArrayBuffer;
}
