"""
ISL-RealTime-Translator1
src/data_collection.py

PART 2 — Landmark Capture & Keypoint Extraction
--------------------------------------------------
This continues the pipeline from Part 1 (folder creation). It opens the
webcam, runs MediaPipe Holistic on every frame, draws the detected
landmarks for visual feedback, extracts pose + face + both-hand keypoints
into a single flat NumPy vector, and saves that vector to disk as a
.npy file inside the folders Part 1 already created:

    data/processed/<action>/<sequence>/<frame_num>.npy

Controls while running:
    q  -> quit early
    (there is a short "STARTING COLLECTION" pause at the start of every
     new sequence so you have time to reset your hand position)

Requires: opencv-python, mediapipe, numpy
    pip install opencv-python mediapipe numpy
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# 1. CONFIG — must match Part 1 exactly, or the .npy files will be written
#    to folders that don't exist / won't match your training script.
# --------------------------------------------------------------------------
DATA_PATH = os.path.join('data', 'processed')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30          # videos per action
sequence_length = 30       # frames per video ("30-frame sign loops")

mp_holistic = mp.solutions.holistic       # Holistic model (pose+face+hands)
mp_drawing = mp.solutions.drawing_utils   # Drawing helpers


# --------------------------------------------------------------------------
# 2. MEDIAPIPE HELPERS
# --------------------------------------------------------------------------
def mediapipe_detection(image, model):
    """Run one frame through the MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    image.flags.writeable = False                    # perf: mark read-only
    results = model.process(image)                   # inference
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    # RGB -> BGR for display
    return image, results


def draw_styled_landmarks(image, results):
    """Draw pose, face, and hand landmarks with distinct colors."""
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    """
    Flatten pose (33 landmarks x 4), face (468 x 3), left hand (21 x 3),
    and right hand (21 x 3) into one 1D vector. If a landmark set wasn't
    detected in a frame, fill it with zeros so every frame has the same
    fixed length (required for LSTM training later).
    """
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)

    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])


# --------------------------------------------------------------------------
# 3. MAIN COLLECTION LOOP
# --------------------------------------------------------------------------
def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam. Check the camera index/permissions.")

    # set_mediapipe_model_complexity: 1 = balanced speed/accuracy
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:

        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    if not ret:
                        print("Failed to grab frame from webcam.")
                        break

                    # Run detection + draw landmarks
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Give a pause at the start of each new sequence
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)   # 2s breather before recording starts
                    else:
                        cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}',
                                    (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    # Extract keypoints and save to the folder Part 1 created
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Quit early with 'q'
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    collect_data()