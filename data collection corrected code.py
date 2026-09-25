import os
import cv2
import numpy as np
import mediapipe as mp

# ==========================================
# 1. Configuration & Actions Setup
# ==========================================
# Target gestures for Indian Sign Language (ISL)
ACTIONS = np.array(["Hi", "Thank You", "I Love You", "Namaste", "Yes", "No"])

# Dataset paths
DATA_PATH = os.path.join("ISL_Data")

# 30 videos per gesture, 30 frames per video
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30

# Create folder structure for dataset collection
for action in ACTIONS:
    for sequence in range(NO_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# MediaPipe Solutions
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# 2. Helper Functions
# ==========================================
def mediapipe_detection(image, model):
    """Processes frame with MediaPipe Holistic model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """Draws styled pose and hand landmarks on the image."""
    # Pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    # Left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    # Right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )


def extract_keypoints(results):
    """Extracts coordinates for pose, left hand, and right hand landmarks."""
    pose = (
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    lh = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, lh, rh])


# ==========================================
# 3. Main Data Collection Routine
# ==========================================
def collect_data():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    try:
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            # --- Standby Loop: Wait for user to trigger recording ---
            print("Webcam initialized. Press 's' to start recording or 'q' to quit.")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cv2.putText(
                    frame,
                    "Press 's' to START recording | 'q' to QUIT",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("ISL Data Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    break
                elif key == ord("q"):
                    return

            # --- Recording Loop across Gestures ---
            for action in ACTIONS:
                for sequence in range(NO_SEQUENCES):
                    for frame_num in range(SEQUENCE_LENGTH):

                        ret, frame = cap.read()
                        if not ret:
                            print("Camera feed lost.")
                            return

                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)

                        # Pause before each sequence to let the user get ready
                        if frame_num == 0:
                            cv2.putText(
                                image,
                                "GET READY...",
                                (180, 200),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                (0, 0, 255),
                                3,
                                cv2.LINE_AA,
                            )
                            cv2.putText(
                                image,
                                f"Recording '{action}' | Video #{sequence + 1}/{NO_SEQUENCES}",
                                (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow("ISL Data Collector", image)
                            cv2.waitKey(2000)

                            # Flush buffer frames accumulated during the 2s pause
                            for _ in range(5):
                                cap.read()
                        else:
                            cv2.putText(
                                image,
                                f"Recording '{action}' | Video #{sequence + 1}/{NO_SEQUENCES}",
                                (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )
                            cv2.imshow("ISL Data Collector", image)

                        # Save extracted keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                        np.save(npy_path, keypoints)

                        # Quit anytime by pressing 'q'
                        if cv2.waitKey(10) & 0xFF == ord("q"):
                            print("Data collection stopped early by user.")
                            return

            print("Collection successfully completed!")

    finally:
        # Guarantee camera release and window closure on completion or error
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_data()