import {
  FEATURE_DIM,
  HAND_DIM,
  HAND_LANDMARKS,
  POSE_DIM,
  POSE_LANDMARKS,
} from './constants';
import type { FrameLandmarks, Point } from './types';

/**
 * Flatten one frame of landmarks into the 258-float feature vector the model
 * expects. Missing parts are zero-filled, which is how the reference MediaPipe
 * Holistic pipelines handle an undetected hand — the model then learns that an
 * all-zero hand block means "not visible" rather than "at the origin".
 *
 * Layout: [pose x33x4 | left hand x21x3 | right hand x21x3] = 132 + 63 + 63 = 258
 */
export function extractKeypoints(frame: FrameLandmarks): Float32Array {
  const out = new Float32Array(FEATURE_DIM);
  let i = 0;

  if (frame.pose) {
    for (let p = 0; p < POSE_LANDMARKS; p++) {
      const lm = frame.pose[p];
      if (lm) {
        out[i] = lm.x;
        out[i + 1] = lm.y;
        out[i + 2] = lm.z;
        out[i + 3] = lm.visibility ?? 0;
      }
      i += 4;
    }
  } else {
    i += POSE_DIM;
  }

  for (const hand of [frame.left, frame.right]) {
    if (hand) {
      for (let p = 0; p < HAND_LANDMARKS; p++) {
        const lm = hand[p];
        if (lm) {
          out[i] = lm.x;
          out[i + 1] = lm.y;
          out[i + 2] = lm.z;
        }
        i += 3;
      }
    } else {
      i += HAND_DIM;
    }
  }

  return out;
}

/**
 * Mean absolute displacement between two frames' hand blocks. Used for
 * pause-based segmentation (research gap G3) and for the "still / signing"
 * indicator — cheap enough to run every frame.
 */
export function handMotion(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = POSE_DIM; i < FEATURE_DIM; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  return sum / (FEATURE_DIM - POSE_DIM);
}

/** Connection pairs for drawing the hand skeleton overlay. */
export const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

/** Upper-body pose connections only — the lower body carries no ISL information. */
export const POSE_CONNECTIONS: Array<[number, number]> = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
];

interface DrawOptions {
  width: number;
  height: number;
  mirrored?: boolean;
}

/** Draw the landmark skeleton onto the overlay canvas. */
export function drawFrame(
  ctx: CanvasRenderingContext2D,
  frame: FrameLandmarks,
  { width, height, mirrored = true }: DrawOptions,
): void {
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  if (mirrored) {
    ctx.translate(width, 0);
    ctx.scale(-1, 1);
  }

  const project = (p: Point) => [p.x * width, p.y * height] as const;

  const drawSet = (
    points: Point[] | null,
    connections: Array<[number, number]>,
    colour: string,
    dot: number,
  ) => {
    if (!points) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const [a, b] of connections) {
      const pa = points[a];
      const pb = points[b];
      if (!pa || !pb) continue;
      if ((pa.visibility ?? 1) < 0.5 || (pb.visibility ?? 1) < 0.5) continue;
      const [ax, ay] = project(pa);
      const [bx, by] = project(pb);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();

    ctx.fillStyle = colour;
    for (const p of points) {
      if (!p || (p.visibility ?? 1) < 0.5) continue;
      const [x, y] = project(p);
      ctx.beginPath();
      ctx.arc(x, y, dot, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  drawSet(frame.pose, POSE_CONNECTIONS, 'rgba(148, 163, 184, 0.75)', 2.5);
  drawSet(frame.left, HAND_CONNECTIONS, '#38bdf8', 3);
  drawSet(frame.right, HAND_CONNECTIONS, '#4ade80', 3);

  ctx.restore();
}
