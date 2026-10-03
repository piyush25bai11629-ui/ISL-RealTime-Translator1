import { useCallback, useRef, useState } from 'react';
import type { Metrics } from '../lib/types';

const EMPTY: Metrics = {
  fps: 0,
  landmarkMs: 0,
  inferenceMs: 0,
  endToEndMs: 0,
  framesSeen: 0,
  predictions: 0,
};

/** Exponential moving average — smooths per-frame jitter without hiding trends. */
function ema(previous: number, sample: number, alpha = 0.15): number {
  return previous === 0 ? sample : previous * (1 - alpha) + sample * alpha;
}

/**
 * Client-side latency instrumentation.
 *
 * Research gap G6 ("real-time is asserted far more often than it is measured")
 * is a PRIMARY target of this project, which makes this hook a deliverable
 * rather than a debugging aid. Per-stage milliseconds and end-to-end FPS are
 * measured here and displayed in the UI, following the reporting style of
 * MediaPipe Hands (Zhang et al., 2020).
 */
export function useMetrics() {
  const [metrics, setMetrics] = useState<Metrics>(EMPTY);
  const ref = useRef<Metrics>(EMPTY);
  const lastFrameAt = useRef<number>(0);
  const lastPublish = useRef<number>(0);

  const publish = useCallback((now: number) => {
    // Repaint the readout ~4x a second; updating every frame would cost more
    // than the pipeline it is measuring.
    if (now - lastPublish.current > 250) {
      lastPublish.current = now;
      setMetrics({ ...ref.current });
    }
  }, []);

  const recordFrame = useCallback(
    (landmarkMs: number) => {
      const now = performance.now();
      const prev = lastFrameAt.current;
      lastFrameAt.current = now;

      const next = ref.current;
      next.framesSeen += 1;
      next.landmarkMs = ema(next.landmarkMs, landmarkMs);
      if (prev) {
        const delta = now - prev;
        if (delta > 0) next.fps = ema(next.fps, 1000 / delta);
      }
      publish(now);
    },
    [publish],
  );

  const recordInference = useCallback(
    (inferenceMs: number) => {
      const next = ref.current;
      next.predictions += 1;
      next.inferenceMs = ema(next.inferenceMs, inferenceMs, 0.25);
      next.endToEndMs = next.landmarkMs + next.inferenceMs;
      publish(performance.now());
    },
    [publish],
  );

  const reset = useCallback(() => {
    ref.current = { ...EMPTY };
    lastFrameAt.current = 0;
    setMetrics({ ...EMPTY });
  }, []);

  return { metrics, recordFrame, recordInference, reset };
}
