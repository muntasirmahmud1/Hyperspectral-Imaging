import os
import time
import cv2
import numpy as np
from datetime import datetime
import platform

from motion_control import init_motion, shutdown_motion, rotate_camera, set_filter


# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = "./hsi_captures/live"

# Motion
MOTOR_PORT = "/dev/ttyUSB0"
FW_PORT = "/dev/ttyACM0"

START_POS = 0
END_POS   =  1000          # inclusive
STEP      = 1 #5

bright_filter_pos = 1
dark_filter_pos = 5

# Capture / SNR
N_DARK = 5                 # dark frames at filter=5
N_AVG  = 5               # frames to average at each motor position (filter=1)


SETTLE_AFTER_MOVE_SEC = 0.50   # wait after motor move
SETTLE_AFTER_FILTER_SEC = 1 # wait after filter move
DISCARD_FRAMES_AFTER_MOVE = 1  # drop a couple frames to flush pipeline
CAPTURE_GAP_SEC = 0.3         # small gap between frames used for averaging

# Camera
CAM_INDEX = 0
WIDTH = 1280
HEIGHT = 720


# =========================================================
# Camera helper
# =========================================================
class HSICamera:
    def __init__(self, cam_index=0, width=1280, height=720):
        system = platform.system().lower()
        backend = cv2.CAP_DSHOW if "windows" in system else cv2.CAP_V4L2

        print(f"Opening camera index {cam_index} with backend {backend}")
        self.cap = cv2.VideoCapture(cam_index, backend)

        if not self.cap.isOpened():
            print("Failed with chosen backend, trying default backend...")
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera. Check index or connection.")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        # Reduce lag / buffering (may be ignored by some drivers, but harmless)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Manual exposure mode (best effort; depends on camera driver)
        if "windows" in system:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)     # many DirectShow drivers: 1=manual
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # many V4L2 drivers: 0.75=manual

        # Keep your preferred defaults (change if needed)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)

        self.system = system

    def read_gray(self) -> np.ndarray:
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# =========================================================
# Capture helpers
# =========================================================
def flush_frames(cam: HSICamera, n: int):
    for _ in range(max(0, int(n))):
        _ = cam.read_gray()


def capture_average(cam: HSICamera, n: int, gap_sec: float = 0.0) -> np.ndarray:
    """
    Capture n grayscale frames and return their averaged image (uint8).
    """
    acc = None
    for i in range(int(n)):
        img = cam.read_gray().astype(np.float32)
        if acc is None:
            acc = img
        else:
            acc += img
        if gap_sec > 0:
            time.sleep(gap_sec)

    avg = acc / float(n)
    avg = np.clip(avg, 0, 255).astype(np.uint8)
    return avg


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


# =========================================================
# Main auto-scan routine
# =========================================================
def main():
    # Create output folder: auto_scan_YYYYmmdd_HHMMSS
    ts_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT_DIR, f"auto_scan_{ts_folder}")
    safe_makedirs(out_dir)
    print(f"Saving auto scan to: {out_dir}")

    cam = None
    try:
        # Init camera + motion
        cam = HSICamera(cam_index=CAM_INDEX, width=WIDTH, height=HEIGHT)
        init_motion(MOTOR_PORT, FW_PORT)

        # 1) Reset motor + filter, then go to START_POS
        print("Resetting filter wheel -> 1 (open) and motor -> 0")
        set_filter(bright_filter_pos)
        time.sleep(SETTLE_AFTER_FILTER_SEC)

        rotate_camera(0)
        time.sleep(SETTLE_AFTER_MOVE_SEC)
        flush_frames(cam, DISCARD_FRAMES_AFTER_MOVE)

        print(f"Moving motor to START_POS = {START_POS}")
        rotate_camera(START_POS)
        time.sleep(SETTLE_AFTER_MOVE_SEC)
        flush_frames(cam, DISCARD_FRAMES_AFTER_MOVE)

        # 2) Dark frames at filter=5 (opaque)
        print(f"Setting filter wheel -> 5 (opaque), capturing {N_DARK} dark frames (avg)")
        set_filter(dark_filter_pos)
        time.sleep(SETTLE_AFTER_FILTER_SEC)

        flush_frames(cam, DISCARD_FRAMES_AFTER_MOVE)
        dark_avg = capture_average(cam, N_DARK, gap_sec=CAPTURE_GAP_SEC)

        dark_name = datetime.now().strftime("dark_avg_%Y%m%d_%H%M%S_%f")[:-3] + ".png"
        dark_path = os.path.join(out_dir, dark_name)
        cv2.imwrite(dark_path, dark_avg)
        print(f"Saved dark average: {dark_path}")

        # 3) Filter open for scan
        print("Setting filter wheel -> 1 (open) for scan")
        set_filter(bright_filter_pos)
        time.sleep(SETTLE_AFTER_FILTER_SEC)

        # 4) Scan positions, each position capture N_AVG frames and save avg
        print(f"Scanning motor positions {START_POS} -> {END_POS} (step={STEP}), avg={N_AVG} frames/pos")
        pos = START_POS
        frame_count = 0

        while pos <= END_POS:
            # Move motor
            rotate_camera(pos)
            time.sleep(SETTLE_AFTER_MOVE_SEC)

            # 5) Prevent motion blur / stale frames
            flush_frames(cam, DISCARD_FRAMES_AFTER_MOVE)

            # Capture and average
            img_avg = capture_average(cam, N_AVG, gap_sec=CAPTURE_GAP_SEC)

            # Optional: dark-correct before saving (uncomment if you want)
            # img_dc = img_avg.astype(np.float32) - dark_avg.astype(np.float32)
            # img_dc[img_dc < 0] = 0
            # img_avg = img_dc.astype(np.uint8)

            # Save with timestamp + motor position
            ts_img = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"frame_pos_{pos:05d}_{ts_img}.png"
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, img_avg)

            frame_count += 1
            if frame_count % 10 == 0 or pos == END_POS:
                print(f"  Saved {frame_count} frames, latest pos={pos} -> {fpath}")

            pos += STEP

        print("Auto scan complete ✅")

    finally:
        try:
            shutdown_motion()
        except Exception:
            pass
        if cam is not None:
            cam.release()


if __name__ == "__main__":
    main()
