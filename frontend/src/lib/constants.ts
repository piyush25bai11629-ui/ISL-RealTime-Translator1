/**
 * Project-wide constants.
 *
 * These are a contract shared with the ML and backend branches — they must match
 * `src/model.py` (Piyush) and `backend/app.py` (Pranav). See TASK_DISTRIBUTION.md §4.
 */

/** Frames per sign sample. Matches `input_shape=(30, 258)` in src/model.py. */
export const FRAME_WINDOW = 30;

/**
 * Landmark features per frame.
 *   132 pose  = 33 landmarks x (x, y, z, visibility)
 *    63 left  = 21 landmarks x (x, y, z)
 *    63 right = 21 landmarks x (x, y, z)
 *   ---
 *   258 total
 *
 * NOTE: this excludes face landmarks. Adding them (research gap G4, non-manual
 * markers) changes this dimension and every cached feature file — coordinate it
 * with the ML branch before changing.
 */
export const FEATURE_DIM = 258;
export const POSE_DIM = 132;
export const HAND_DIM = 63;

export const POSE_LANDMARKS = 33;
export const HAND_LANDMARKS = 21;

/** Demo vocabulary. Matches `actions` in src/data_collection.py and Dense(3) in src/model.py. */
export const ACTIONS = ['hello', 'thanks', 'iloveyou'] as const;
export type Action = (typeof ACTIONS)[number];

/** Human-readable labels for the caption line. */
export const ACTION_LABELS: Record<string, string> = {
  hello: 'Hello',
  thanks: 'Thank you',
  iloveyou: 'I love you',
};

/**
 * Below this softmax score a prediction is treated as uncertain: it is not
 * committed to the sentence, and the correction bar is offered instead.
 * This is the client half of the feedback loop on slide 11 of the research deck.
 */
export const CONFIDENCE_THRESHOLD = 0.7;

/** Consecutive agreeing predictions required before a sign is committed. */
export const STABILITY_FRAMES = 3;

/** Milliseconds of stillness that ends a sign, i.e. pause-based segmentation (gap G3). */
export const PAUSE_SEGMENT_MS = 1200;

/** Backend base URL. Override with VITE_API_BASE in a .env file. */
export const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

/** MediaPipe model bundles, served from Google's CDN. */
export const WASM_PATH =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
export const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
export const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
