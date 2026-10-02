import { useCallback, useEffect, useRef, useState } from 'react';
import {
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import {
  HAND_MODEL_URL,
  POSE_MODEL_URL,
  WASM_PATH,
} from '../lib/constants';
import type { FrameLandmarks } from '../lib/types';

interface LandmarkerState {
  ready: boolean;
  error: string | null;
  detect: (video: HTMLVideoElement, timestamp: number) => FrameLandmarks | null;
}

/**
 * Loads MediaPipe Pose + Hand landmarkers and exposes a per-frame detect().
 *
 * The research deck specifies MediaPipe Holistic. `@mediapipe/tasks-vision`,
 * the maintained package, ships Pose and Hand landmarkers as separate tasks
 * rather than one Holistic graph, so they are composed here. The resulting
 * feature vector is identical in layout and dimension (258).
 *
 * Zhang et al. (2020) report 16.1 ms on a Pixel 3 for the hand pipeline; the
 * measured browser cost is surfaced in the metrics panel rather than assumed.
 */
export function useLandmarker(): LandmarkerState {
  const poseRef = useRef<PoseLandmarker | null>(null);
  const handRef = useRef<HandLandmarker | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

        const [pose, hands] = await Promise.all([
          PoseLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numPoses: 1,
          }),
          HandLandmarker.createFromOptions(vision, {
            baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numHands: 2,
          }),
        ]);

        if (cancelled) {
          pose.close();
          hands.close();
          return;
        }

        poseRef.current = pose;
        handRef.current = hands;
        setReady(true);
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : 'Could not load the MediaPipe models.',
          );
        }
      }
    })();

    return () => {
      cancelled = true;
      poseRef.current?.close();
      handRef.current?.close();
      poseRef.current = null;
      handRef.current = null;
    };
  }, []);

  const detect = useCallback(
    (video: HTMLVideoElement, timestamp: number): FrameLandmarks | null => {
      const pose = poseRef.current;
      const hands = handRef.current;
      if (!pose || !hands) return null;

      const poseResult = pose.detectForVideo(video, timestamp);
      const handResult = hands.detectForVideo(video, timestamp);

      const frame: FrameLandmarks = { pose: null, left: null, right: null };

      if (poseResult.landmarks?.length) {
        frame.pose = poseResult.landmarks[0].map((p) => ({
          x: p.x,
          y: p.y,
          z: p.z,
          visibility: p.visibility ?? 1,
        }));
      }

      handResult.landmarks?.forEach((points, i) => {
        // Handedness is reported from the camera's point of view. The preview is
        // mirrored for the user, so "Left" here is the signer's right hand. The
        // slot only has to stay consistent between training and inference.
        const label = handResult.handedness?.[i]?.[0]?.categoryName ?? 'Left';
        const mapped = points.map((p) => ({ x: p.x, y: p.y, z: p.z }));
        if (label === 'Left') frame.left = mapped;
        else frame.right = mapped;
      });

      return frame;
    },
    [],
  );

  return { ready, error, detect };
}
