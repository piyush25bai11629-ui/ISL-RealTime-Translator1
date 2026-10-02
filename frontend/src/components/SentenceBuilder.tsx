import type { CommittedSign } from '../lib/types';

interface Props {
  signs: CommittedSign[];
  onClear: () => void;
  onSpeak: () => void;
  ttsSupported: boolean;
  speaking: boolean;
}

/**
 * Accumulated signs, joined into a sentence.
 *
 * Sentence-level ISL data is scarce (research gap G3), so this is deliberately
 * a concatenation of isolated signs rather than a grammatical translation. The
 * distinction matters and is stated in the UI so a demo is not mistaken for
 * continuous sign language translation.
 */
export function SentenceBuilder({
  signs,
  onClear,
  onSpeak,
  ttsSupported,
  speaking,
}: Props) {
  const sentence = signs.map((s) => s.label).join(' ');

  return (
    <section className="panel p-5" aria-label="Translated sentence">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="stat-label">Sentence</h2>
        <div className="flex gap-2">
          <button
            type="button"
            className="btn-ghost !px-3 !py-1.5 !text-xs"
            onClick={onSpeak}
            disabled={!ttsSupported || signs.length === 0}
            title={
              ttsSupported
                ? 'Read the sentence aloud'
                : 'Speech synthesis is unavailable in this browser'
            }
          >
            {speaking ? 'Speaking…' : 'Speak'}
          </button>
          <button
            type="button"
            className="btn-ghost !px-3 !py-1.5 !text-xs"
            onClick={onClear}
            disabled={signs.length === 0}
          >
            Clear
          </button>
        </div>
      </div>

      <p
        className="min-h-[3rem] text-xl leading-relaxed"
        aria-live="polite"
        aria-atomic="true"
      >
        {sentence || (
          <span className="text-base text-muted">
            Signs you make will collect here.
          </span>
        )}
      </p>

      {signs.length > 0 && (
        <ul className="mt-4 flex flex-wrap gap-2" aria-label="Individual signs">
          {signs.map((sign) => (
            <li
              key={sign.id}
              className="animate-riseIn rounded-full border border-line px-3 py-1 text-xs"
            >
              <span className="font-semibold">{sign.label}</span>
              <span className="ml-2 font-mono text-muted">
                {Math.round(sign.confidence * 100)}%
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
