import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ACTION_LABELS,
  CONFIDENCE_THRESHOLD,
  FRAME_WINDOW,
  PAUSE_SEGMENT_MS,
  STABILITY_FRAMES,
} from '../lib/constants';
import { drawFrame, extractKeypoints, handMotion } from '../lib/landmarks';
import { mockPredict, predict, saveCorrection } from '../lib/api';
import type {
  BackendMode,
  CommittedSign,
  CorrectionRecord,
  EngineState,
  FrameLandmarks,
  PredictResponse,
} from '../lib/types';
import { useLandmarker } from './useLandmarker';
import { useMetrics } from './useMetrics';

interface Options {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  backendMode: BackendMode;
  onCommit?: (sign: CommittedSign) => void;
}

/** Motion below this counts as stillness, which ends the current sign. */
const STILLNESS = 0.0015;

/**
 * The real-time inference loop.
 *
 * Per frame: extract landmarks -> push the 258-float vector into a 30-frame
 * ring buffer -> once full, request a prediction -> apply a stability filter ->
 * commit the sign on a pause. Every stage is timed (research gap G6).
 */
export function useRecognition({
  videoRef,
  canvasRef,
  backendMode,
  onCommit,
}: Options) {
  const { ready: modelsReady, error: modelError, detect } = useLandmarker();
  const { metrics, recordFrame, recordInference, reset: resetMetrics } = useMetrics();

  const [state, setState] = useState<EngineState>('loading-models');
  const [running, setRunning] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [confidence, setConfidence] = useState(0);
  const [bufferFill, setBufferFill] = useState(0);
  const [moving, setMoving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingCorrection, setPendingCorrection] =
    useState<{ predicted: string; confidence: number } | null>(null);

  const bufferRef = useRef<Float32Array[]>([]);
  const rafRef = useRef<number>(0);
  const inFlightRef = useRef(false);
  const lastVectorRef = useRef<Float32Array | null>(null);
  const lastMotionAt = useRef<number>(0);
  const streakRef = useRef<{ label: string | null; count: number }>({
    label: null,
    count: 0,
  });
  const committedRef = useRef<string | null>(null);
  const runningRef = useRef(false);
  const modeRef = useRef<BackendMode>(backendMode);

  useEffect(() => {
    modeRef.current = backendMode;
  }, [backendMode]);

  useEffect(() => {
    if (modelError) {
      setState('error');
      setError(modelError);
    } else if (modelsReady) {
      setState((s) => (s === 'loading-models' ? 'ready' : s));
    }
  }, [modelsReady, modelError]);

  const commit = useCallback(
    (label: string, score: number) => {
      if (committedRef.current === label) return;
      committedRef.current = label;

      const sign: CommittedSign = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        raw: label,
        label: ACTION_LABELS[label] ?? label,
        confidence: score,
        at: Date.now(),
      };
      onCommit?.(sign);
    },
    [onCommit],
  );

  const runPrediction = useCallback(async () => {
    if (inFlightRef.current) return;
    const window = bufferRef.current;
    if (window.length < FRAME_WINDOW) return;

    inFlightRef.current = true;
    const started = performance.now();

    try {
      let result: PredictResponse;
      if (modeRef.current === 'mock') {
        result = mockPredict(window);
      } else {
        result = await predict(window);
      }
      recordInference(performance.now() - started);

      const score = result.confidence ?? 0;
      setPrediction(result.prediction);
      setConfidence(score);

      const streak = streakRef.current;
      if (streak.label === result.prediction) streak.count += 1;
      else streakRef.current = { label: result.prediction, count: 1 };

      if (
        streakRef.current.count >= STABILITY_FRAMES &&
        score >= CONFIDENCE_THRESHOLD
      ) {
        commit(result.prediction, score);
        setPendingCorrection(null);
      } else if (
        streakRef.current.count >= STABILITY_FRAMES &&
        score > 0 &&
        score < CONFIDENCE_THRESHOLD
      ) {
        // Low confidence: do not commit. Offer a correction instead — this is
        // the client half of the feedback loop on slide 11 of the research deck.
        setPendingCorrection({ predicted: result.prediction, confidence: score });
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message} — switch to Mock mode to demo without the backend.`
          : 'Prediction failed.',
      );
      setRunning(false);
      runningRef.current = false;
      setState('error');
    } finally {
      inFlightRef.current = false;
    }
  }, [commit, recordInference]);

  const tick = useCallback(() => {
    if (!runningRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video && video.readyState >= 2) {
      const t0 = performance.now();
      let frame: FrameLandmarks | null = null;
      try {
        frame = detect(video, t0);
      } catch {
        // A dropped frame is not worth tearing the loop down for.
      }
      const landmarkMs = performance.now() - t0;

      if (frame) {
        recordFrame(landmarkMs);

        if (canvas) {
          const ctx = canvas.getContext('2d');
          if (ctx) {
            if (canvas.width !== video.videoWidth) {
              canvas.width = video.videoWidth;
              canvas.height = video.videoHeight;
            }
            drawFrame(ctx, frame, {
              width: canvas.width,
              height: canvas.height,
            });
          }
        }

        const vector = extractKeypoints(frame);
        const buffer = bufferRef.current;
        buffer.push(vector);
        if (buffer.length > FRAME_WINDOW) buffer.shift();
        setBufferFill(buffer.length);

        // Pause-based segmentation (research gap G3, engineering approximation).
        const previous = lastVectorRef.current;
        if (previous) {
          const motion = handMotion(previous, vector);
          const isMoving = motion > STILLNESS;
          setMoving(isMoving);
          if (isMoving) {
            lastMotionAt.current = performance.now();
          } else if (
            lastMotionAt.current &&
            performance.now() - lastMotionAt.current > PAUSE_SEGMENT_MS
          ) {
            // Sustained stillness closes the current sign, so the next
            // occurrence of the same label is treated as a new word.
            committedRef.current = null;
            streakRef.current = { label: null, count: 0 };
          }
        }
        lastVectorRef.current = vector;

        if (buffer.length === FRAME_WINDOW) void runPrediction();
      }
    }

    rafRef.current = requestAnimationFrame(tick);
  }, [canvasRef, detect, recordFrame, runPrediction, videoRef]);

  const start = useCallback(() => {
    if (runningRef.current || !modelsReady) return;
    setError(null);
    runningRef.current = true;
    setRunning(true);
    setState('running');
    lastMotionAt.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
  }, [modelsReady, tick]);

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    setState(modelsReady ? 'ready' : 'loading-models');
    cancelAnimationFrame(rafRef.current);
    setMoving(false);
  }, [modelsReady]);

  const resetBuffer = useCallback(() => {
    bufferRef.current = [];
    lastVectorRef.current = null;
    streakRef.current = { label: null, count: 0 };
    committedRef.current = null;
    setBufferFill(0);
    setPrediction(null);
    setConfidence(0);
    setPendingCorrection(null);
    resetMetrics();
  }, [resetMetrics]);

  /** Record a user correction on a low-confidence prediction. */
  const submitCorrection = useCallback(
    (corrected: string) => {
      const pending = pendingCorrection;
      if (!pending) return;

      const record: CorrectionRecord = {
        predicted: pending.predicted,
        corrected,
        confidence: pending.confidence,
        at: Date.now(),
      };
      saveCorrection(record);
      commit(corrected, pending.confidence);
      setPendingCorrection(null);
    },
    [commit, pendingCorrection],
  );

  const dismissCorrection = useCallback(() => setPendingCorrection(null), []);

  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  return {
    state,
    running,
    error,
    prediction,
    confidence,
    bufferFill,
    moving,
    metrics,
    modelsReady,
    pendingCorrection,
    start,
    stop,
    resetBuffer,
    submitCorrection,
    dismissCorrection,
  };
}
