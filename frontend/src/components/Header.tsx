import type { BackendMode, EngineState } from '../lib/types';

interface Props {
  state: EngineState;
  backendMode: BackendMode;
  onModeChange: (mode: BackendMode) => void;
  backendReachable: boolean | null;
}

const STATE_COPY: Record<EngineState, { text: string; dot: string }> = {
  idle: { text: 'Idle', dot: 'bg-muted' },
  'loading-models': { text: 'Loading models', dot: 'bg-warn animate-pulseRing' },
  ready: { text: 'Ready', dot: 'bg-accent' },
  running: { text: 'Recognising', dot: 'bg-ok animate-pulseRing' },
  error: { text: 'Error', dot: 'bg-bad' },
};

export function Header({
  state,
  backendMode,
  onModeChange,
  backendReachable,
}: Props) {
  const status = STATE_COPY[state];

  return (
    <header className="border-b border-line">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-4 px-5 py-4">
        <div className="flex items-center gap-3">
          <div
            aria-hidden="true"
            className="flex h-10 w-10 items-center justify-center rounded-lg border-2 border-accent text-lg font-bold text-accent"
          >
            ISL
          </div>
          <div>
            <h1 className="text-lg font-bold leading-tight">
              ISL Real-Time Translator
            </h1>
            <p className="text-xs text-muted">
              MediaPipe landmarks · CNN + LSTM · single RGB webcam
            </p>
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-3">
          <div
            className="flex items-center gap-2 rounded-full border border-line px-3 py-1.5"
            role="status"
            aria-live="polite"
          >
            <span
              aria-hidden="true"
              className={`h-2 w-2 rounded-full ${status.dot}`}
            />
            <span className="text-xs font-semibold">{status.text}</span>
          </div>

          <fieldset className="flex items-center gap-1 rounded-lg border border-line p-1">
            <legend className="sr-only">Prediction source</legend>
            {(['live', 'mock'] as BackendMode[]).map((mode) => {
              const active = backendMode === mode;
              const label = mode === 'live' ? 'Live backend' : 'Mock';
              return (
                <button
                  key={mode}
                  type="button"
                  onClick={() => onModeChange(mode)}
                  aria-pressed={active}
                  className={`rounded px-3 py-1.5 text-xs font-semibold transition-colors ${
                    active
                      ? 'bg-accent text-surface'
                      : 'text-muted hover:text-text'
                  }`}
                >
                  {label}
                </button>
              );
            })}
          </fieldset>

          {backendMode === 'live' && backendReachable === false && (
            <p className="text-xs font-semibold text-warn">
              Backend unreachable on :8000
            </p>
          )}
        </div>
      </div>
    </header>
  );
}
