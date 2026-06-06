import cv2
import mediapipe as mp
import pyautogui 

# --- MediaPipe Hands Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# --- New: State variables to track if keys are pressed ---
# This prevents sending a key press command on every single frame
left_key_pressed = False
right_key_pressed = False
# ---------------------------------------------------------

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # --- New: Flags to check which hands are visible in the current frame ---
    is_left_hand_visible = False
    is_right_hand_visible = False
    # ----------------------------------------------------------------------

    # Check if any hands and their handedness are detected
    if result.multi_hand_landmarks and result.multi_handedness:
        # Loop through all detected hands
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Draw the landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the handedness (Left or Right)
            handedness_info = result.multi_handedness[i]
            hand_label = handedness_info.classification[0].label

            # Update flags based on which hand is detected
            if hand_label == "Left":
                is_left_hand_visible = True
            elif hand_label == "Right":
                is_right_hand_visible = True

    # --- New: Control Logic based on hand visibility ---

    # --- Right Hand (Gas) ---
    # If the right hand is visible AND the key isn't already pressed down
    if is_right_hand_visible and not right_key_pressed:
        print("Right Hand Detected: Pressing Gas (Right Arrow)")
        pyautogui.keyDown('right') # Hold the right arrow key down
        right_key_pressed = True
    # If the right hand is NOT visible AND the key was previously pressed
    elif not is_right_hand_visible and right_key_pressed:
        print("Right Hand Gone: Releasing Gas (Right Arrow)")
        pyautogui.keyUp('right') # Release the right arrow key
        right_key_pressed = False

    # --- Left Hand (Brake) ---
    # If the left hand is visible AND the key isn't already pressed down
    if is_left_hand_visible and not left_key_pressed:
        print("Left Hand Detected: Pressing Brake (Left Arrow)")
        pyautogui.keyDown('left') # Hold the left arrow key down
        left_key_pressed = True
    # If the left hand is NOT visible AND the key was previously pressed
    elif not is_left_hand_visible and left_key_pressed:
        print("Left Hand Gone: Releasing Brake (Left Arrow)")
        pyautogui.keyUp('left') # Release the left arrow key
        left_key_pressed = False

    # ----------------------------------------------------

    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- New: Release any pressed keys before exiting ---
if right_key_pressed:
    pyautogui.keyUp('right')
if left_key_pressed:
    pyautogui.keyUp('left')
# -------------------------------------------------

cap.release()
cv2.destroyAllWindows()