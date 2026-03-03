import os
import time
from datetime import datetime

import cv2
import numpy as np

from motion_control import init_motion, shutdown_motion, rotate_camera, set_filter


# ==========================
# USER CONFIG
# ==========================
# Fixed settings
EXPOSURE = 500
camera_motor_pos = 7700

# Filter positions
bright_filter_positions = [1, 2, 3, 4]   # test these
dark_filter_pos = 5                     # opaque

ROOT_DIR = "/home/sciglob/HSI_code/hsi/hsi_captures/live"
MOTOR_PORT = "/dev/ttyUSB0"
FW_PORT = "/dev/ttyACM0"

# Capture / SNR
N_DARK = 10   # dark frames average
N_AVG  = 10   # bright frames average

# Camera
CAM_INDEX = 0
WIDTH = 1280
HEIGHT = 800

# Delay/timing (tune if needed)
SETTLE_AFTER_FILTER_S = 0.8
SETTLE_AFTER_EXPOSURE_S = 0.3
SETTLE_AFTER_MOTOR_S = 0.8


# ==========================
# Helpers
# ==========================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def safe_imwrite(path: str, img: np.ndarray) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")

def capture_gray(cap: cv2.VideoCapture) -> np.ndarray:
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from camera.")
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

def average_frames(cap: cv2.VideoCapture, n: int) -> np.ndarray:
    acc = None
    for _ in range(n):
        img = capture_gray(cap).astype(np.float32)
        acc = img if acc is None else (acc + img)
        time.sleep(0.01)
    avg = acc / float(n)
    return np.clip(avg, 0, 255).astype(np.uint8)

def set_manual_exposure(cap: cv2.VideoCapture, exposure_value: float) -> None:
    # Common on Jetson/V4L2 UVC: 0.75 manual, 0.25 auto
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_value))

def flush_frames(cap: cv2.VideoCapture, n: int = 3) -> None:
    for _ in range(n):
        cap.read()
        time.sleep(0.01)


# ==========================
# Main
# ==========================
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_DIR, f"filterwheel_test_{ts}")
    ensure_dir(out_dir)
    print(f"[INFO] Saving results to: {out_dir}")

    init_motion(MOTOR_PORT, FW_PORT)

    try:
        print(f"[INFO] Moving motor to {camera_motor_pos} ...")
        rotate_camera(camera_motor_pos)
        time.sleep(SETTLE_AFTER_MOTOR_S)

        # Open camera
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Check CAM_INDEX or connection.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        flush_frames(cap, n=5)

        # Set fixed exposure once
        print(f"[INFO] Setting exposure -> {EXPOSURE}")
        set_manual_exposure(cap, EXPOSURE)
        time.sleep(SETTLE_AFTER_EXPOSURE_S)
        flush_frames(cap, n=3)

        for fw_pos in bright_filter_positions:
            print(f"\n[TEST] Bright filter position = {fw_pos}")

            # ---- Bright average at this filter position ----
            print(f"[INFO] Set filter wheel -> {fw_pos}")
            set_filter(fw_pos)
            time.sleep(SETTLE_AFTER_FILTER_S)
            flush_frames(cap, n=3)

            bright_avg = average_frames(cap, N_AVG)
            bright_name = f"bright_fw{fw_pos}_exp{EXPOSURE}_motor{camera_motor_pos}.png"
            safe_imwrite(os.path.join(out_dir, bright_name), bright_avg)
            print(f"[SAVE] {bright_name}  (avg of {N_AVG})")

            # ---- Dark average (always fw=5) ----
            print(f"[INFO] Set filter wheel -> DARK (pos {dark_filter_pos})")
            set_filter(dark_filter_pos)
            time.sleep(SETTLE_AFTER_FILTER_S)
            flush_frames(cap, n=3)

            dark_avg = average_frames(cap, N_DARK)
            dark_name = f"dark_fw{dark_filter_pos}_exp{EXPOSURE}_motor{camera_motor_pos}_after_fw{fw_pos}.png"
            safe_imwrite(os.path.join(out_dir, dark_name), dark_avg)
            print(f"[SAVE] {dark_name}  (avg of {N_DARK})")

        cap.release()
        print("\n[DONE] Filter wheel position test complete.")

    finally:
        shutdown_motion()


if __name__ == "__main__":
    main()
