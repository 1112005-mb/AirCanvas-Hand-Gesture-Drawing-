"""
Air Canvas - Real-Time Hand Gesture Drawing Application.
Author: Maansi Bharti
Resume Project: Hand Gesture Drawing System (Python, OpenCV, MediaPipe)

Key Capabilities:
- Real-time 30+ FPS hand landmark tracking via MediaPipe Hands
- Exponential Moving Average (EMA) spatial smoothing to remove finger jitter
- Adaptive illumination enhancement via LAB-space CLAHE
- Dual-mode gesture control:
    * Index finger: Drawing mode
    * Index + Middle fingers: Selection / Palette interaction mode
- Multi-color palette (Blue, Green, Red, Yellow, Purple, Eraser)
- Dynamic brush sizing and Canvas snapshot export (press 's')
"""

import cv2
import numpy as np
import time
import os
from hands import HandTracker, SpatialSmoother, LightingEnhancer


class ColorPalette:
    CLEAR = (0, 0, 0)
    BLUE = (255, 120, 0)
    GREEN = (0, 220, 0)
    RED = (0, 50, 255)
    YELLOW = (0, 230, 255)
    PURPLE = (220, 50, 180)
    ERASER = (0, 0, 0)


def build_palette_header(width, current_color, brush_thickness, is_eraser):
    """Generates an aesthetic top navigation header with color selectors and status."""
    header = np.zeros((100, width, 3), dtype=np.uint8)
    header[:] = (35, 35, 38)  # Sleek dark background

    buttons = [
        {"name": "CLEAR", "color": (50, 50, 60), "text_color": (255, 255, 255)},
        {"name": "BLUE", "color": ColorPalette.BLUE, "text_color": (255, 255, 255)},
        {"name": "GREEN", "color": ColorPalette.GREEN, "text_color": (20, 20, 20)},
        {"name": "RED", "color": ColorPalette.RED, "text_color": (255, 255, 255)},
        {"name": "YELLOW", "color": ColorPalette.YELLOW, "text_color": (20, 20, 20)},
        {"name": "PURPLE", "color": ColorPalette.PURPLE, "text_color": (255, 255, 255)},
        {"name": "ERASER", "color": (120, 120, 130), "text_color": (255, 255, 255)},
    ]

    btn_width = width // len(buttons)

    for i, btn in enumerate(buttons):
        x1 = i * btn_width + 4
        x2 = (i + 1) * btn_width - 4
        y1, y2 = 12, 88

        # Highlight currently active tool
        active = False
        if btn["name"] == "ERASER" and is_eraser:
            active = True
        elif btn["name"] != "CLEAR" and btn["name"] != "ERASER" and not is_eraser and btn["color"] == current_color:
            active = True

        bg_col = btn["color"]
        cv2.rectangle(header, (x1, y1), (x2, y2), bg_col, -1, cv2.LINE_AA)

        if active:
            cv2.rectangle(header, (x1, y1), (x2, y2), (255, 255, 255), 3, cv2.LINE_AA)

        # Label text
        text_size = cv2.getTextSize(btn["name"], cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(header, btn["name"], (text_x, text_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, btn["text_color"], 1, cv2.LINE_AA)

    # Bottom dividing line
    cv2.line(header, (0, 99), (width, 99), (70, 70, 80), 2)
    return header, buttons, btn_width


def main():
    # Setup Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Core Engines
    tracker = HandTracker(detection_con=0.75, track_con=0.75)
    smoother = SpatialSmoother(min_alpha=0.18, max_alpha=0.75, velocity_threshold=35.0)
    enhancer = LightingEnhancer(clip_limit=2.5)

    # Runtime State
    draw_color = ColorPalette.BLUE
    brush_thickness = 8
    eraser_thickness = 45
    is_eraser = False
    enable_lighting_pipeline = False
    canvas = None
    prev_smoothed_pt = None

    print("\n" + "=" * 60)
    print("  🎨 HAND GESTURE DRAWING SYSTEM (Air Canvas)")
    print("=" * 60)
    print(" Controls:")
    print("  • Index Finger UP       -> Draw on Canvas")
    print("  • Index + Middle UP    -> Selection / Hover Mode")
    print("  • '[' / ']'             -> Decrease / Increase Brush Size")
    print("  • 'L' Key               -> Toggle Adaptive Lighting Filter (CLAHE)")
    print("  • 'C' Key               -> Clear Canvas")
    print("  • 'S' Key               -> Save Canvas to Disk")
    print("  • 'Q' Key               -> Quit Application")
    print("=" * 60 + "\n")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture webcam frame.")
            break

        # Flip horizontally for natural intuitive mirror interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Initialize canvas if not ready
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Apply Lighting Enhancement pipeline if active
        processed_frame = enhancer.enhance(frame) if enable_lighting_pipeline else frame

        # Run Hand Tracking
        processed_frame = tracker.find_hands(processed_frame, draw=False)
        landmarks = tracker.get_positions(processed_frame)
        fingers = tracker.fingers_up()
        fps = tracker.calculate_fps()

        active_smoothed_pt = None

        if len(landmarks) >= 21:
            index_x, index_y = landmarks[8][1], landmarks[8][2]
            middle_x, middle_y = landmarks[12][1], landmarks[12][2]

            # Selection / Tool Hover Mode (Index & Middle fingers open)
            if fingers[1] == 1 and fingers[2] == 1:
                smoother.reset()
                prev_smoothed_pt = None

                # Visual indicator for selection mode
                cv2.circle(frame, (index_x, index_y), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (middle_x, middle_y), 12, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (index_x, index_y), (middle_x, middle_y), (255, 255, 255), 2, cv2.LINE_AA)

                # Check if interacting with Top Navigation Palette
                if index_y < 100:
                    btn_width = w // 7
                    col_index = index_x // btn_width

                    if col_index == 0:  # CLEAR
                        canvas = np.zeros((h, w, 3), dtype=np.uint8)
                    elif col_index == 1:  # BLUE
                        draw_color = ColorPalette.BLUE
                        is_eraser = False
                    elif col_index == 2:  # GREEN
                        draw_color = ColorPalette.GREEN
                        is_eraser = False
                    elif col_index == 3:  # RED
                        draw_color = ColorPalette.RED
                        is_eraser = False
                    elif col_index == 4:  # YELLOW
                        draw_color = ColorPalette.YELLOW
                        is_eraser = False
                    elif col_index == 5:  # PURPLE
                        draw_color = ColorPalette.PURPLE
                        is_eraser = False
                    elif col_index == 6:  # ERASER
                        is_eraser = True

            # Drawing Mode (Only Index finger open)
            elif fingers[1] == 1 and fingers[2] == 0:
                # Apply Spatial Smoothing filter to fingertip trajectory
                active_smoothed_pt = smoother.smooth((index_x, index_y))

                curr_thickness = eraser_thickness if is_eraser else brush_thickness
                active_color = (0, 0, 0) if is_eraser else draw_color

                # Reticle indicator
                cv2.circle(frame, active_smoothed_pt, curr_thickness // 2,
                           (200, 200, 200) if is_eraser else active_color, -1, cv2.LINE_AA)

                if prev_smoothed_pt is not None:
                    # Draw continuous anti-aliased path onto virtual canvas
                    cv2.line(canvas, prev_smoothed_pt, active_smoothed_pt,
                             active_color, curr_thickness, cv2.LINE_AA)

                prev_smoothed_pt = active_smoothed_pt

            else:
                smoother.reset()
                prev_smoothed_pt = None
        else:
            smoother.reset()
            prev_smoothed_pt = None

        # Blend Canvas with Live Video Feed
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, inv_mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY_INV)
        inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)

        # Isolate background and add vibrant canvas strokes
        frame_bg = cv2.bitwise_and(frame, inv_mask)
        combined_display = cv2.bitwise_or(frame_bg, canvas)

        # Overlay Aesthetic Navigation Header
        header_img, _, _ = build_palette_header(w, draw_color, brush_thickness, is_eraser)
        combined_display[0:100, 0:w] = header_img

        # Diagnostics HUD (Top Right & Bottom Left)
        # FPS Counter (verifying 30+ FPS claim)
        fps_color = (0, 255, 0) if fps >= 30 else (0, 165, 255)
        cv2.putText(combined_display, f"FPS: {fps}", (w - 140, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, fps_color, 2, cv2.LINE_AA)

        # Lighting Pipeline Status
        light_txt = "LIGHTING FILTER: [ON]" if enable_lighting_pipeline else "LIGHTING FILTER: [OFF] (Press 'L')"
        light_col = (0, 255, 180) if enable_lighting_pipeline else (160, 160, 160)
        cv2.putText(combined_display, light_txt, (20, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, light_col, 1, cv2.LINE_AA)

        # Active Tool & Brush Size Indicator
        curr_tool = "ERASER" if is_eraser else "BRUSH"
        cv2.putText(combined_display, f"TOOL: {curr_tool} (Size: {brush_thickness}px)", (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        cv2.imshow("Air Canvas - Hand Gesture Drawing System", combined_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            print("Canvas cleared.")
        elif key == ord('l'):
            enable_lighting_pipeline = not enable_lighting_pipeline
            print(f"Adaptive lighting pipeline: {'Enabled' if enable_lighting_pipeline else 'Disabled'}")
        elif key == ord(']'):
            brush_thickness = min(40, brush_thickness + 2)
            print(f"Brush size: {brush_thickness}")
        elif key == ord('['):
            brush_thickness = max(2, brush_thickness - 2)
            print(f"Brush size: {brush_thickness}")
        elif key == ord('s'):
            filename = f"canvas_export_{int(time.time())}.png"
            cv2.imwrite(filename, canvas)
            print(f"Canvas saved successfully as '{filename}'!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
