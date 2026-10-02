import { forwardRef } from 'react';
import Webcam from 'react-webcam';
import { FRAME_WINDOW } from '../lib/constants';

interface Props {
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  running: boolean;
  moving: boolean;
  bufferFill: number;
  showOverlay: boolean;
  onUserMediaError: (message: string) => void;
}

/**
 * Webcam feed with the landmark skeleton drawn on a canvas above it.
 *
 * The preview is mirrored (scale-x-[-1]) because users expect a mirror; the
 * overlay applies the same flip in drawFrame(), and the feature vector itself
 * is left unmirrored so it matches the training data.
 */
export const WebcamPanel = forwardRef<Webcam, Props>(function WebcamPanel(
  { canvasRef, running, moving, bufferFill, showOverlay, onUserMediaError },
  ref,
) {
  const fillPct = Math.round((bufferFill / FRAME_WINDOW) * 100);

  return (
    <section className="panel overflow-hidden" aria-label="Camera feed">
      <div className="relative aspect-video w-full bg-black">
        <Webcam
          ref={ref}
          audio={false}
          mirrored
          className="absolute inset-0 h-full w-full object-cover"
          videoConstraints={{
            width: 640,
            height: 480,
            facingMode: 'user',
          }}
          onUserMediaError={(err) =>
            onUserMediaError(
              typeof err === 'string'
                ? err
                : 'Camera access was blocked. Allow camera permission and reload.',
            )
          }
        />

        <canvas
          ref={canvasRef}
          aria-hidden="true"
          className={`absolute inset-0 h-full w-full scale-x-[-1] object-cover transition-opacity ${
            showOverlay ? 'opacity-100' : 'opacity-0'
          }`}
        />

        {/* Recording / stillness indicator */}
        <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-surface/85 px-3 py-1.5 backdrop-blur">
          <span
            aria-hidden="true"
            className={`h-2 w-2 rounded-full ${
              running
                ? moving
                  ? 'bg-ok animate-pulseRing'
                  : 'bg-warn'
                : 'bg-muted'
            }`}
          />
          <span className="text-xs font-semibold">
            {running ? (moving ? 'Signing' : 'Still') : 'Paused'}
          </span>
        </div>

        {!running && (
          <div className="absolute inset-0 flex items-center justify-center bg-surface/70 backdrop-blur-sm">
            <p className="max-w-xs px-6 text-center text-sm text-muted">
              Press <span className="font-semibold text-text">Start</span> to
              begin. Keep your head and both hands in frame.
            </p>
          </div>
        )}
      </div>

      {/* Rolling window fill: the model needs 30 frames before it can predict. */}
      <div className="border-t border-line px-4 py-3">
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="stat-label">Frame buffer</span>
          <span className="font-mono text-xs text-muted">
            {bufferFill} / {FRAME_WINDOW}
          </span>
        </div>
        <div
          role="progressbar"
          aria-valuenow={bufferFill}
          aria-valuemin={0}
          aria-valuemax={FRAME_WINDOW}
          aria-label="Frames buffered before the next prediction"
          className="h-1.5 w-full overflow-hidden rounded-full bg-surface"
        >
          <div
            className="h-full rounded-full bg-accent transition-[width] duration-150"
            style={{ width: `${fillPct}%` }}
          />
        </div>
      </div>
    </section>
  );
});
