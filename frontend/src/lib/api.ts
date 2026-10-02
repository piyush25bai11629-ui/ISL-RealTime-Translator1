import { ACTIONS, API_BASE, FEATURE_DIM, FRAME_WINDOW } from './constants';
import type { PredictResponse } from './types';

/**
 * POST a (30, 258) landmark window to the backend.
 *
 * Request body matches `FrameData` in backend/app.py:
 *   { "landmarks": number[][] }
 *
 * The response is expected to carry `confidence` in addition to the fields in
 * the kickoff snippet — see TASK_DISTRIBUTION.md §10, open point 2.
 */
export async function predict(
  window: Float32Array[],
  signal?: AbortSignal,
): Promise<PredictResponse> {
  const landmarks = window.map((f) => Array.from(f));

  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ landmarks }),
    signal,
  });

  if (!res.ok) {
    throw new Error(`Backend returned ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as PredictResponse;
}

/** Cheap liveness probe so the UI can show whether the backend is reachable. */
export async function ping(timeoutMs = 1500): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/docs`, { signal: ctrl.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Offline stand-in for POST /predict.
 *
 * The backend is Pranav's branch and the trained weights are Piyush's, so the
 * frontend has to be demonstrable before either lands. This scores the window
 * with a crude motion heuristic instead of a network call. It is NOT a model
 * and produces no research claim — the UI labels it MOCK wherever it is used.
 */
export function mockPredict(window: Float32Array[]): PredictResponse {
  const frames = window.slice(-FRAME_WINDOW);

  let motion = 0;
  for (let f = 1; f < frames.length; f++) {
    for (let i = 132; i < FEATURE_DIM; i++) {
      motion += Math.abs(frames[f][i] - frames[f - 1][i]);
    }
  }
  motion /= Math.max(1, (frames.length - 1) * (FEATURE_DIM - 132));

  // Rough vertical centre of the hand blocks: high hands vs low hands.
  let height = 0;
  let counted = 0;
  const last = frames[frames.length - 1];
  for (let i = 132 + 1; i < FEATURE_DIM; i += 3) {
    if (last[i] !== 0) {
      height += last[i];
      counted++;
    }
  }
  height = counted ? height / counted : 0.5;

  let index: number;
  if (motion > 0.012) index = 0;
  else if (height < 0.45) index = 2;
  else index = 1;

  const confidence = Math.min(0.99, 0.62 + Math.min(motion * 18, 0.3));

  return {
    prediction: ACTIONS[index],
    status: 'success',
    confidence,
    timings: { inference_ms: 0 },
  };
}

/**
 * Queue a user correction on a low-confidence prediction.
 *
 * Slide 11 of the research deck argues that logging corrections is how the
 * project generates labelled data for regional sign variation. The endpoint
 * does not exist yet, so corrections are buffered in localStorage and can be
 * exported from the UI.
 */
const CORRECTION_KEY = 'isl.corrections.v1';

export function saveCorrection(record: unknown): void {
  try {
    const existing = JSON.parse(localStorage.getItem(CORRECTION_KEY) ?? '[]');
    existing.push(record);
    localStorage.setItem(CORRECTION_KEY, JSON.stringify(existing));
  } catch {
    // Storage disabled or full — corrections are a nice-to-have, never fatal.
  }
}

export function loadCorrections(): unknown[] {
  try {
    return JSON.parse(localStorage.getItem(CORRECTION_KEY) ?? '[]');
  } catch {
    return [];
  }
}

export function clearCorrections(): void {
  try {
    localStorage.removeItem(CORRECTION_KEY);
  } catch {
    /* no-op */
  }
}
