export interface Point {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

/** One frame of extracted landmarks, before flattening to the 258-float vector. */
export interface FrameLandmarks {
  pose: Point[] | null;
  left: Point[] | null;
  right: Point[] | null;
}

/** Response contract for POST /predict (backend/app.py, Pranav). */
export interface PredictResponse {
  prediction: string;
  status: 'success' | 'error';
  /**
   * Softmax score for `prediction`. Not in the kickoff snippet, but required —
   * without it the low-confidence feedback loop cannot exist. See
   * TASK_DISTRIBUTION.md §10, open point 2.
   */
  confidence?: number;
  /** Optional server-side per-stage timings, in milliseconds (research gap G6). */
  timings?: { inference_ms?: number; total_ms?: number };
}

/** A sign that passed the stability check and was committed to the transcript. */
export interface CommittedSign {
  id: string;
  label: string;
  raw: string;
  confidence: number;
  at: number;
  corrected?: string;
}

/** Per-stage latency, measured client-side. Gap G6 is a primary research target. */
export interface Metrics {
  fps: number;
  landmarkMs: number;
  inferenceMs: number;
  endToEndMs: number;
  framesSeen: number;
  predictions: number;
}

export type EngineState =
  | 'idle'
  | 'loading-models'
  | 'ready'
  | 'running'
  | 'error';

export type BackendMode = 'live' | 'mock';

/** A user correction on a low-confidence prediction, queued for the data loop. */
export interface CorrectionRecord {
  predicted: string;
  corrected: string;
  confidence: number;
  at: number;
}
