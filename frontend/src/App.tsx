import { useCallback, useEffect, useRef, useState } from 'react';
import type Webcam from 'react-webcam';

import { Header } from './components/Header';
import { WebcamPanel } from './components/WebcamPanel';
import { TranslationBox } from './components/TranslationBox';
import { SentenceBuilder } from './components/SentenceBuilder';
import { MetricsPanel } from './components/MetricsPanel';
import { CorrectionBar } from './components/CorrectionBar';
import { ControlsBar } from './components/ControlsBar';

import { useRecognition } from './hooks/useRecognition';
import { useTTS } from './hooks/useTTS';
import { clearCorrections, loadCorrections, ping } from './lib/api';
import type { BackendMode, CommittedSign } from './lib/types';

export default function App() {
  const webcamRef = useRef<Webcam>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Default to mock so the UI is demonstrable before the backend and the
  // trained weights land on their own branches.
  const [backendMode, setBackendMode] = useState<BackendMode>('mock');
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [signs, setSigns] = useState<CommittedSign[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [correctionCount, setCorrectionCount] = useState(
    () => loadCorrections().length,
  );

  const tts = useTTS();
  const ttsRef = useRef(tts);
  ttsRef.current = tts;

  const handleCommit = useCallback((sign: CommittedSign) => {
    setSigns((prev) => [...prev, sign]);
    ttsRef.current.speak(sign.label);
  }, []);

  const {
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
  } = useRecognition({
    videoRef,
    canvasRef,
    backendMode,
    onCommit: handleCommit,
  });

  // react-webcam attaches the <video> element asynchronously, so poll briefly
  // until MediaPipe has something to read frames from.
  useEffect(() => {
    const id = window.setInterval(() => {
      videoRef.current = webcamRef.current?.video ?? null;
      if (videoRef.current) window.clearInterval(id);
    }, 200);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (backendMode !== 'live') return;
    let cancelled = false;
    void ping().then((ok) => {
      if (!cancelled) setBackendReachable(ok);
    });
    return () => {
      cancelled = true;
    };
  }, [backendMode]);

  const handleReset = useCallback(() => {
    resetBuffer();
    setSigns([]);
    ttsRef.current.stop();
  }, [resetBuffer]);

  const handleSubmitCorrection = useCallback(
    (corrected: string) => {
      submitCorrection(corrected);
      setCorrectionCount(loadCorrections().length);
    },
    [submitCorrection],
  );

  const handleExportCorrections = useCallback(() => {
    const data = loadCorrections();
    if (data.length === 0) return;

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `isl-corrections-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);

    clearCorrections();
    setCorrectionCount(0);
  }, []);

  const handleSpeak = useCallback(() => {
    const sentence = signs.map((s) => s.label).join(' ');
    if (sentence) ttsRef.current.speak(sentence, true);
  }, [signs]);

  const latest = signs[signs.length - 1];

  return (
    <div className="flex min-h-full flex-col">
      <Header
        state={state}
        backendMode={backendMode}
        onModeChange={setBackendMode}
        backendReachable={backendReachable}
      />

      <main className="mx-auto w-full max-w-7xl flex-1 px-5 py-6">
        {backendMode === 'mock' && (
          <p className="mb-5 rounded-lg border border-warn/50 bg-warn/10 px-4 py-2.5 text-xs text-warn">
            <span className="font-bold">MOCK MODE</span> — predictions come from
            a motion heuristic, not the trained model. Switch to Live backend
            once <code className="font-mono">POST /predict</code> is running on
            port 8000.
          </p>
        )}

        {(error || cameraError) && (
          <p
            role="alert"
            className="mb-5 rounded-lg border border-bad/50 bg-bad/10 px-4 py-2.5 text-xs text-bad"
          >
            {error ?? cameraError}
          </p>
        )}

        <div className="grid gap-5 lg:grid-cols-[1.4fr_1fr]">
          <div className="flex flex-col gap-5">
            <WebcamPanel
              ref={webcamRef}
              canvasRef={canvasRef}
              running={running}
              moving={moving}
              bufferFill={bufferFill}
              showOverlay={showOverlay}
              onUserMediaError={setCameraError}
            />

            <ControlsBar
              running={running}
              modelsReady={modelsReady}
              showOverlay={showOverlay}
              ttsSupported={tts.supported}
              ttsEnabled={tts.enabled}
              correctionCount={correctionCount}
              onStart={start}
              onStop={stop}
              onReset={handleReset}
              onToggleOverlay={setShowOverlay}
              onToggleTts={tts.setEnabled}
              onExportCorrections={handleExportCorrections}
            />
          </div>

          <div className="flex flex-col gap-5">
            <TranslationBox
              prediction={prediction}
              confidence={confidence}
              running={running}
            />

            <CorrectionBar
              pending={pendingCorrection}
              onSubmit={handleSubmitCorrection}
              onDismiss={dismissCorrection}
            />

            <SentenceBuilder
              signs={signs}
              onClear={() => setSigns([])}
              onSpeak={handleSpeak}
              ttsSupported={tts.supported}
              speaking={tts.speaking}
            />

            <MetricsPanel metrics={metrics} />
          </div>
        </div>
      </main>

      <footer className="border-t border-line px-5 py-4">
        <p className="mx-auto max-w-7xl text-[11px] text-muted">
          Isolated sign recognition over a 30-frame window of 258 MediaPipe
          landmark features. Output is a sequence of recognised signs, not a
          grammatical translation — sentence-level ISL corpora remain scarce
          (research gap G3).
        </p>
      </footer>

      {/* Screen-reader announcement channel for newly committed signs. */}
      <p className="sr-only" role="status" aria-live="polite">
        {latest ? `Recognised ${latest.label}` : ''}
      </p>
    </div>
  );
}
