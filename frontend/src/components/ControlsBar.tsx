interface Props {
  running: boolean;
  modelsReady: boolean;
  showOverlay: boolean;
  ttsSupported: boolean;
  ttsEnabled: boolean;
  correctionCount: number;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
  onToggleOverlay: (next: boolean) => void;
  onToggleTts: (next: boolean) => void;
  onExportCorrections: () => void;
}

function Toggle({
  label,
  checked,
  disabled,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <label
      className={`flex items-center gap-2 text-xs font-semibold ${
        disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer'
      }`}
    >
      <input
        type="checkbox"
        className="h-4 w-4 accent-accent"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

export function ControlsBar({
  running,
  modelsReady,
  showOverlay,
  ttsSupported,
  ttsEnabled,
  correctionCount,
  onStart,
  onStop,
  onReset,
  onToggleOverlay,
  onToggleTts,
  onExportCorrections,
}: Props) {
  return (
    <section
      className="panel flex flex-wrap items-center gap-4 p-4"
      aria-label="Controls"
    >
      <div className="flex gap-2">
        {running ? (
          <button type="button" className="btn-ghost" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button
            type="button"
            className="btn-primary"
            onClick={onStart}
            disabled={!modelsReady}
          >
            {modelsReady ? 'Start' : 'Loading models…'}
          </button>
        )}
        <button type="button" className="btn-ghost" onClick={onReset}>
          Reset
        </button>
      </div>

      <div className="flex flex-wrap items-center gap-5">
        <Toggle
          label="Landmark overlay"
          checked={showOverlay}
          onChange={onToggleOverlay}
        />
        <Toggle
          label="Speak predictions"
          checked={ttsEnabled}
          disabled={!ttsSupported}
          onChange={onToggleTts}
        />
      </div>

      <button
        type="button"
        className="btn-ghost ml-auto !px-3 !py-1.5 !text-xs"
        onClick={onExportCorrections}
        disabled={correctionCount === 0}
        title="Download logged corrections as JSON for retraining"
      >
        Export corrections ({correctionCount})
      </button>
    </section>
  );
}
