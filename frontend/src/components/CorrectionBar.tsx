import { ACTIONS, ACTION_LABELS } from '../lib/constants';

interface Props {
  pending: { predicted: string; confidence: number } | null;
  onSubmit: (corrected: string) => void;
  onDismiss: () => void;
}

/**
 * The client half of the feedback loop.
 *
 * When the model is unsure, the user says what the sign actually was. The
 * correction is stored locally and can be exported for retraining — this is
 * the mechanism the research deck proposes for closing the signer-dependence
 * gap (G2) with data from real users rather than more lab recordings.
 */
export function CorrectionBar({ pending, onSubmit, onDismiss }: Props) {
  if (!pending) return null;

  const predictedLabel = ACTION_LABELS[pending.predicted] ?? pending.predicted;

  return (
    <section
      className="panel animate-riseIn border-warn/60 p-4"
      role="alertdialog"
      aria-label="Low confidence prediction"
    >
      <p className="text-sm">
        Not sure about{' '}
        <span className="font-semibold text-warn">{predictedLabel}</span> (
        {Math.round(pending.confidence * 100)}%). What did you sign?
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        {ACTIONS.map((action) => (
          <button
            key={action}
            type="button"
            className="btn-ghost !px-3 !py-1.5 !text-xs"
            onClick={() => onSubmit(action)}
          >
            {ACTION_LABELS[action] ?? action}
          </button>
        ))}
        <button
          type="button"
          className="btn !px-3 !py-1.5 !text-xs text-muted hover:text-text"
          onClick={onDismiss}
        >
          Skip
        </button>
      </div>
    </section>
  );
}
