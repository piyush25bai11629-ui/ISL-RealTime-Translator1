import type { Metrics } from '../lib/types';

interface Props {
  metrics: Metrics;
}

function Stat({
  label,
  value,
  unit,
  hint,
}: {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
}) {
  return (
    <div>
      <p className="stat-label">{label}</p>
      <p className="mt-0.5 font-mono text-xl font-semibold">
        {value}
        {unit && <span className="ml-1 text-sm text-muted">{unit}</span>}
      </p>
      {hint && <p className="mt-0.5 text-[11px] text-muted">{hint}</p>}
    </div>
  );
}

/**
 * Latency readout.
 *
 * Research gap G6: papers assert "real-time" far more often than they measure
 * it. Showing per-stage milliseconds in the product makes the claim falsifiable
 * and gives the evaluation workstream a live number to cite.
 */
export function MetricsPanel({ metrics }: Props) {
  const fmt = (n: number) => (n > 0 ? n.toFixed(1) : '—');

  return (
    <section className="panel p-5" aria-label="Performance metrics">
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="stat-label">Latency (G6)</h2>
        <span className="text-[11px] text-muted">measured, not assumed</span>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Stat label="Throughput" value={fmt(metrics.fps)} unit="fps" />
        <Stat
          label="End to end"
          value={fmt(metrics.endToEndMs)}
          unit="ms"
          hint={metrics.endToEndMs > 0 && metrics.endToEndMs < 100 ? 'within real-time budget' : undefined}
        />
        <Stat label="Landmarks" value={fmt(metrics.landmarkMs)} unit="ms" />
        <Stat label="Inference" value={fmt(metrics.inferenceMs)} unit="ms" />
      </div>

      <dl className="mt-4 flex gap-6 border-t border-line pt-3 text-xs text-muted">
        <div className="flex gap-2">
          <dt>Frames</dt>
          <dd className="font-mono text-text">{metrics.framesSeen}</dd>
        </div>
        <div className="flex gap-2">
          <dt>Predictions</dt>
          <dd className="font-mono text-text">{metrics.predictions}</dd>
        </div>
      </dl>
    </section>
  );
}
