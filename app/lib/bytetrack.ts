import type { Detection } from "./yolo";

type Box = Detection["box"];

export type TrackState = "tracked" | "lost";

export type Track = Detection & {
  id: string;
  trackId: number;
  state: TrackState;
  age: number;
  hits: number;
  missed: number;
  lastSeenFrame: number;
  velocity: Box;
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

    const candidateTracks = this.tracks.filter((track) => track.missed <= this.config.bufferFrames);
    const highResult = associate(candidateTracks, highDetections, this.config.matchThreshold);
    const matchedTrackIds = new Set<number>();
    const matchedHighDetections = new Set<number>();

    for (const match of highResult.matches) {
      const track = candidateTracks[match.trackIndex];
      const detection = highDetections[match.detectionIndex];
      updateTrack(track, detection, this.frameId);
      matchedTrackIds.add(track.trackId);
      matchedHighDetections.add(match.detectionIndex);
    }

    const unmatchedCandidateTracks = highResult.unmatchedTrackIndexes.map((index) => candidateTracks[index]);
    const lowResult = associate(unmatchedCandidateTracks, lowDetections, this.config.matchThreshold);

    for (const match of lowResult.matches) {
      const track = unmatchedCandidateTracks[match.trackIndex];
      const detection = lowDetections[match.detectionIndex];
      updateTrack(track, detection, this.frameId);
      matchedTrackIds.add(track.trackId);
    }

    for (const track of candidateTracks) {
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
      velocity: {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
      },
    };
  }
}

function updateTrack(track: Track, detection: Detection, frameId: number) {
  const nextVelocity = {
    x: detection.box.x - track.box.x,
    y: detection.box.y - track.box.y,
    width: detection.box.width - track.box.width,
    height: detection.box.height - track.box.height,
  };

  track.classId = detection.classId;
  track.label = detection.label;
  track.confidence = detection.confidence;
  track.box = detection.box;
  track.velocity = smoothVelocity(track.velocity, nextVelocity);
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

      const track = tracks[trackIndex];
      const detection = detections[detectionIndex];
      const predictedBox = predictBox(track);
      const iou = intersectionOverUnionBoxes(predictedBox, detection.box);
      const score = matchingScore(predictedBox, detection.box);

      if (iou >= minimumIou || score >= 0.45) {
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

function smoothVelocity(previous: Box, next: Box): Box {
  const alpha = 0.75;
  return {
    x: previous.x * alpha + next.x * (1 - alpha),
    y: previous.y * alpha + next.y * (1 - alpha),
    width: previous.width * alpha + next.width * (1 - alpha),
    height: previous.height * alpha + next.height * (1 - alpha),
  };
}

function predictBox(track: Track): Box {
  const missedFrames = Math.max(1, track.missed + 1);
  return {
    x: track.box.x + track.velocity.x * missedFrames,
    y: track.box.y + track.velocity.y * missedFrames,
    width: Math.max(1, track.box.width + track.velocity.width * missedFrames),
    height: Math.max(1, track.box.height + track.velocity.height * missedFrames),
  };
}

function matchingScore(predictedBox: Box, detectionBox: Box) {
  const iou = intersectionOverUnionBoxes(predictedBox, detectionBox);
  const distanceScore = centerDistanceScore(predictedBox, detectionBox);
  return iou * 0.7 + distanceScore * 0.3;
}

function centerDistanceScore(a: Box, b: Box) {
  const ax = a.x + a.width / 2;
  const ay = a.y + a.height / 2;
  const bx = b.x + b.width / 2;
  const by = b.y + b.height / 2;
  const distance = Math.hypot(ax - bx, ay - by);
  const scale = Math.max(1, Math.hypot(Math.max(a.width, b.width), Math.max(a.height, b.height)));
  return Math.max(0, 1 - distance / scale);
}

function intersectionOverUnionBoxes(a: Box, b: Box) {
  const ax2 = a.x + a.width;
  const ay2 = a.y + a.height;
  const bx2 = b.x + b.width;
  const by2 = b.y + b.height;
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(ax2, bx2);
  const y2 = Math.min(ay2, by2);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - intersection;
  return union <= 0 ? 0 : intersection / union;
}
