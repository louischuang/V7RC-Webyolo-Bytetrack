import type { Detection } from "./yolo";

export type TrackState = "tracked" | "lost";

export type Track = Detection & {
  id: string;
  trackId: number;
  state: TrackState;
  age: number;
  hits: number;
  missed: number;
  lastSeenFrame: number;
};

export type ByteTrackConfig = {
  highThreshold: number;
  lowThreshold: number;
  matchThreshold: number;
  bufferFrames: number;
};

type Match = {
  trackIndex: number;
  detectionIndex: number;
  score: number;
};

const defaultConfig: ByteTrackConfig = {
  highThreshold: 0.6,
  lowThreshold: 0.1,
  matchThreshold: 0.8,
  bufferFrames: 30,
};

export class ByteTracker {
  private tracks: Track[] = [];
  private nextTrackId = 1;
  private frameId = 0;

  constructor(private readonly config: ByteTrackConfig = defaultConfig) {}

  reset() {
    this.tracks = [];
    this.nextTrackId = 1;
    this.frameId = 0;
  }

  update(detections: Detection[]): Track[] {
    this.frameId += 1;

    const highDetections = detections.filter((detection) => detection.confidence >= this.config.highThreshold);
    const lowDetections = detections.filter(
      (detection) =>
        detection.confidence >= this.config.lowThreshold && detection.confidence < this.config.highThreshold,
    );

    const activeTracks = this.tracks.filter((track) => track.state === "tracked");
    const highResult = associate(activeTracks, highDetections, this.config.matchThreshold);
    const matchedTrackIds = new Set<number>();
    const matchedHighDetections = new Set<number>();

    for (const match of highResult.matches) {
      const track = activeTracks[match.trackIndex];
      const detection = highDetections[match.detectionIndex];
      updateTrack(track, detection, this.frameId);
      matchedTrackIds.add(track.trackId);
      matchedHighDetections.add(match.detectionIndex);
    }

    const unmatchedActiveTracks = highResult.unmatchedTrackIndexes.map((index) => activeTracks[index]);
    const lowResult = associate(unmatchedActiveTracks, lowDetections, this.config.matchThreshold);

    for (const match of lowResult.matches) {
      const track = unmatchedActiveTracks[match.trackIndex];
      const detection = lowDetections[match.detectionIndex];
      updateTrack(track, detection, this.frameId);
      matchedTrackIds.add(track.trackId);
    }

    for (const track of activeTracks) {
      if (!matchedTrackIds.has(track.trackId)) {
        track.state = "lost";
        track.missed += 1;
        track.age += 1;
      }
    }

    const unmatchedHighDetections = highDetections.filter((_, index) => !matchedHighDetections.has(index));
    for (const detection of unmatchedHighDetections) {
      this.tracks.push(this.createTrack(detection));
    }

    for (const track of this.tracks) {
      if (track.state === "lost") {
        track.missed = this.frameId - track.lastSeenFrame;
      }
    }

    this.tracks = this.tracks.filter((track) => track.missed <= this.config.bufferFrames);

    return this.tracks
      .filter((track) => track.state === "tracked")
      .sort((a, b) => b.lastSeenFrame - a.lastSeenFrame || b.confidence - a.confidence);
  }

  private createTrack(detection: Detection): Track {
    const trackId = this.nextTrackId;
    this.nextTrackId += 1;

    return {
      ...detection,
      id: `T${trackId}`,
      trackId,
      state: "tracked",
      age: 1,
      hits: 1,
      missed: 0,
      lastSeenFrame: this.frameId,
    };
  }
}

function updateTrack(track: Track, detection: Detection, frameId: number) {
  track.classId = detection.classId;
  track.label = detection.label;
  track.confidence = detection.confidence;
  track.box = detection.box;
  track.state = "tracked";
  track.age += 1;
  track.hits += 1;
  track.missed = 0;
  track.lastSeenFrame = frameId;
}

function associate(tracks: Track[], detections: Detection[], matchThreshold: number) {
  const matches: Match[] = [];
  const unmatchedTrackIndexes = new Set(tracks.map((_, index) => index));
  const unmatchedDetectionIndexes = new Set(detections.map((_, index) => index));
  const minimumIou = Math.max(0, Math.min(1, 1 - matchThreshold));

  const candidates: Match[] = [];
  for (let trackIndex = 0; trackIndex < tracks.length; trackIndex += 1) {
    for (let detectionIndex = 0; detectionIndex < detections.length; detectionIndex += 1) {
      if (tracks[trackIndex].classId !== detections[detectionIndex].classId) {
        continue;
      }

      const score = intersectionOverUnion(tracks[trackIndex], detections[detectionIndex]);
      if (score >= minimumIou) {
        candidates.push({ trackIndex, detectionIndex, score });
      }
    }
  }

  candidates.sort((a, b) => b.score - a.score);

  for (const candidate of candidates) {
    if (!unmatchedTrackIndexes.has(candidate.trackIndex) || !unmatchedDetectionIndexes.has(candidate.detectionIndex)) {
      continue;
    }

    matches.push(candidate);
    unmatchedTrackIndexes.delete(candidate.trackIndex);
    unmatchedDetectionIndexes.delete(candidate.detectionIndex);
  }

  return {
    matches,
    unmatchedTrackIndexes: [...unmatchedTrackIndexes],
    unmatchedDetectionIndexes: [...unmatchedDetectionIndexes],
  };
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
