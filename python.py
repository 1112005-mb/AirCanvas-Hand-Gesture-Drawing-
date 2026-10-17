import cv2
import mediapipe as mp
import pyautogui
import time


def get_hand_position(landmarks, frame_width, frame_height, mp_hands):
    """Calculates wrist (X, Y) pixel coordinates from detected hand landmarks."""
    wrist_x = landmarks.landmark[mp_hands.HandLandmark.WRIST].x
    wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
    return int(wrist_x * frame_width), int(wrist_y * frame_height)


def run_subway_surfers_controller():
    """
    Gesture-based control system mapped to directional keys (Up, Down, Left, Right).
    Ideal for runner games such as Subway Surfers or Temple Run.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    last_x_position = None
    last_y_position = None
    action_state = "neutral"

    cap = cv2.VideoCapture(0)
    print("\n[INFO] Initializing Subway Surfers Gesture Control...")
    print("[INFO] Waiting 5 seconds: switch to your game window now!")
    for sec in range(5, 0, -1):
        print(f"Starting in {sec}...")
        time.sleep(1)

    print("Active! Perform hand gestures in front of the camera. Press 'q' in the camera window to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror view
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    current_x, current_y = get_hand_position(hand_landmarks, frame_width, frame_height, mp_hands)

                    if last_x_position is not None and last_y_position is not None:
                        delta_x = current_x - last_x_position
                        delta_y = last_y_position - current_y

                        jump_threshold = 20
                        slide_threshold = 20
                        horizontal_threshold = 20

                        # Jump and Slide Gestures
                        if delta_y > jump_threshold and action_state != "jump":
                            print("Jump detected! 🚀 [Up Arrow]")
                            pyautogui.press('up')
                            action_state = "jump"
                        elif delta_y < -slide_threshold and action_state != "slide":
                            print("Slide detected! 🏂 [Down Arrow]")
                            pyautogui.press('down')
                            action_state = "slide"
                        # Horizontal Movement Gestures
                        elif delta_x > horizontal_threshold and action_state != "right":
                            print("Move Right detected! 👉 [Right Arrow]")
                            pyautogui.press('right')
                            action_state = "right"
                        elif delta_x < -horizontal_threshold and action_state != "left":
                            print("Move Left detected! 👈 [Left Arrow]")
                            pyautogui.press('left')
                            action_state = "left"
                        elif abs(delta_x) < 10 and abs(delta_y) < 10:
                            action_state = "neutral"

                    last_x_position = current_x
                    last_y_position = current_y

            cv2.imshow("Subway Surfers Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Program terminated successfully.")


if __name__ == "__main__":
    run_subway_surfers_controller()

