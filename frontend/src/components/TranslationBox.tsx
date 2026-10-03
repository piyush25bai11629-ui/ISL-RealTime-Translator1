import { ACTION_LABELS, CONFIDENCE_THRESHOLD } from '../lib/constants';

interface Props {
  prediction: string | null;
  confidence: number;
  running: boolean;
}

export function TranslationBox({ prediction, confidence, running }: Props) {
  const label = prediction ? (ACTION_LABELS[prediction] ?? prediction) : null;
  const pct = Math.round(confidence * 100);
  const confident = confidence >= CONFIDENCE_THRESHOLD;

  return (
    <section aria-label="Current prediction">
      <div className="mb-2 flex items-baseline justify-between">
        <h2 className="stat-label">Current sign</h2>
        {prediction && (
          <span
            className={`font-mono text-xs font-semibold ${
              confident ? 'text-ok' : 'text-warn'
            }`}
          >
            {pct}% confident
          </span>
        )}
      </div>

      <div className="translation-box min-h-[104px]">
        {label ? (
          <div className="animate-riseIn">
            <p className="text-3xl font-bold leading-tight">{label}</p>
            <div
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Prediction confidence"
              className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-surface"
            >
              <div
                className={`h-full rounded-full transition-[width] duration-200 ${
                  confident ? 'bg-ok' : 'bg-warn'
                }`}
                style={{ width: `${pct}%` }}
              />
            </div>
            {!confident && (
              <p className="mt-2 text-xs text-warn">
                Below the {Math.round(CONFIDENCE_THRESHOLD * 100)}% threshold —
                not added to the sentence.
              </p>
            )}
          </div>
        ) : (
          <p className="text-base text-muted">
            {running
              ? 'Watching for a sign…'
              : 'No prediction yet. Start the camera to begin.'}
          </p>
        )}
      </div>
    </section>
  );
}
