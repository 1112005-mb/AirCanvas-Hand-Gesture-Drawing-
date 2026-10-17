"""
Hand Gesture Recognition & Drawing System
Author: Maansi Bharti
Project: Hand Gesture Drawing System (Air Canvas) with Multimodal Game Control

Unified Entry Point:
  • Default: Launches the Hand Gesture Drawing System (Air Canvas)
  • CLI Flag: --app vehicle (Runs the Two-Hand Gas/Brake Controller)
  • CLI Flag: --app subway (Runs the Subway Surfers Gesture Controller)
"""

import sys
import argparse
import cv2
import mediapipe as mp
import pyautogui
import air_canvas
from python import run_subway_surfers_controller


def run_vehicle_controller():
    """
    Two-Hand Vehicle Controller:
      - Right Hand: Accelerate / Gas ('Right Arrow')
      - Left Hand: Decelerate / Brake ('Left Arrow')
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    left_key_pressed = False
    right_key_pressed = False

    print("\n[INFO] Running Vehicle Gesture Controller...")
    print("  • Show Right Hand -> Press Gas (Right Arrow)")
    print("  • Show Left Hand  -> Press Brake (Left Arrow)")
    print("  • Press 'q' in the camera window to quit.\n")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            is_left_hand_visible = False
            is_right_hand_visible = False

            if result.multi_hand_landmarks and result.multi_handedness:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    handedness_info = result.multi_handedness[i]
                    hand_label = handedness_info.classification[0].label

                    if hand_label == "Left":
                        is_left_hand_visible = True
                    elif hand_label == "Right":
                        is_right_hand_visible = True

            # Right Hand (Gas)
            if is_right_hand_visible and not right_key_pressed:
                print("Right Hand Detected: Pressing Gas (Right Arrow)")
                pyautogui.keyDown('right')
                right_key_pressed = True
            elif not is_right_hand_visible and right_key_pressed:
                print("Right Hand Gone: Releasing Gas (Right Arrow)")
                pyautogui.keyUp('right')
                right_key_pressed = False

            # Left Hand (Brake)
            if is_left_hand_visible and not left_key_pressed:
                print("Left Hand Detected: Pressing Brake (Left Arrow)")
                pyautogui.keyDown('left')
                left_key_pressed = True
            elif not is_left_hand_visible and left_key_pressed:
                print("Left Hand Gone: Releasing Brake (Left Arrow)")
                pyautogui.keyUp('left')
                left_key_pressed = False

            cv2.imshow("Vehicle Gesture Controller (Gas/Brake)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if right_key_pressed:
            pyautogui.keyUp('right')
        if left_key_pressed:
            pyautogui.keyUp('left')
        cap.release()
        cv2.destroyAllWindows()
        print("Vehicle controller exited cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Hand Gesture Recognition and Drawing System")
    parser.add_argument(
        "--app",
        type=str,
        default="canvas",
        choices=["canvas", "vehicle", "subway"],
        help="Application mode to launch: 'canvas' (default), 'vehicle', or 'subway'"
    )
    args = parser.parse_args()

    if args.app == "canvas":
        air_canvas.main()
    elif args.app == "vehicle":
        run_vehicle_controller()
    elif args.app == "subway":
        run_subway_surfers_controller()


if __name__ == "__main__":
    main()