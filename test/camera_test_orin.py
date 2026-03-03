# import cv2
# import numpy as np
# import os
# import platform

# # =========================================================
# # Helper: GPU availability
# # =========================================================
# def gpu_available():
#     try:
#         return cv2.cuda.getCudaEnabledDeviceCount() > 0
#     except Exception:
#         return False

# GPU = gpu_available()
# print(f"GPU available: {GPU}")

# # =========================================================
# # Camera class
# # =========================================================
# class HSICamera:
#     def __init__(self, cam_index=0):
#         system = platform.system().lower()

#         # Choose backend depending on OS
#         if "windows" in system:
#             backend = cv2.CAP_DSHOW
#         else:
#             # Jetson / Linux: use V4L2 or default
#             backend = cv2.CAP_V4L2  # you can try cv2.CAP_ANY if needed

#         print(f"Opening camera index {cam_index} with backend {backend}")
#         self.cap = cv2.VideoCapture(cam_index, backend)

#         if not self.cap.isOpened():
#             # Try again with default backend as a fallback
#             print("Failed with chosen backend, trying default backend...")
#             self.cap.release()
#             self.cap = cv2.VideoCapture(cam_index)

#         if not self.cap.isOpened():
#             raise RuntimeError(
#                 f"Cannot open camera at index {cam_index}. "
#                 "Check /dev/video* and permissions."
#             )

#         self.show_properties()

#     def show_properties(self):
#         print("\n=== Current Camera Properties ===")
#         props = {
#             "FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
#             "FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
#             "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
#             "CONTRAST": cv2.CAP_PROP_CONTRAST,
#             "SATURATION": cv2.CAP_PROP_SATURATION,
#             "GAIN": cv2.CAP_PROP_GAIN,
#             "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
#         }
#         for name, code in props.items():
#             print(f"{name:15s}: {self.cap.get(code)}")
#         print("==================================\n")

#     def set_exposure(self, value):
#         """Set exposure. On Linux/Jetson, auto-exposure control is different from Windows."""
#         # These flags are backend-dependent; on Jetson you sometimes need 1 or 0.25, etc.
#         # You can experiment with different values if this doesn't work.
#         self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # disable auto, may vary by driver
#         success = self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
#         print(f"Exposure set to {value}, success={success}")

#     def set_gain(self, value):
#         success = self.cap.set(cv2.CAP_PROP_GAIN, float(value))
#         print(f"Gain set to {value}, success={success}")

#     def capture(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             raise RuntimeError("Failed to capture frame.")
#         return frame

#     def release(self):
#         self.cap.release()
#         cv2.destroyAllWindows()


# # =========================================================
# # Main control loop
# # =========================================================
# if __name__ == "__main__":
#     # If your test script showed the camera is at index 1, change cam_index=1
#     cam = HSICamera(cam_index=0)

#     print("Press 'e'/'E' to decrease/increase exposure")
#     print("Press 'g'/'G' to decrease/increase gain")
#     print("Press 'q' to quit")

#     exposure = cam.cap.get(cv2.CAP_PROP_EXPOSURE)
#     gain = cam.cap.get(cv2.CAP_PROP_GAIN)

#     while True:
#         frame = cam.capture()

#         if GPU:
#             gpu_frame = cv2.cuda_GpuMat()
#             gpu_frame.upload(frame)
#             gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
#             display_frame = gpu_gray.download()
#         else:
#             display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         cv2.imshow("HSI Camera Live", display_frame)

#         key = cv2.waitKey(10) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('e'):  # decrease exposure
#             exposure -= 1
#             cam.set_exposure(exposure)
#         elif key == ord('E'):  # increase exposure
#             exposure += 1
#             cam.set_exposure(exposure)
#         elif key == ord('g'):  # decrease gain
#             gain -= 1
#             cam.set_gain(gain)
#         elif key == ord('G'):  # increase gain
#             gain += 1
#             cam.set_gain(gain)

#     cam.release()





import cv2
import subprocess
import platform
import re

CAM_INDEX = 0


def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True)
    except Exception as e:
        print(f"Command failed: {cmd}\n{e}")
        return ""


def test_with_v4l2():
    print("\n=== Using v4l2-ctl (Jetson / Linux) ===")
    info = run_cmd(f"v4l2-ctl --all -d /dev/video{CAM_INDEX}")

    if not info:
        print("Could not read v4l2 info.")
        return

    print(info)

    # -------- parse exposure --------
    exp_match = re.search(r"Exposure.*:.*min=(\d+)\s+max=(\d+)\s+step=(\d+).*value=(\d+)", info)
    if exp_match:
        mn, mx, step, val = map(int, exp_match.groups())
        print(f"\nExposure range   : {mn} to {mx}")
        print(f"Exposure step    : {step}")
        print(f"Current exposure : {val}")
    else:
        print("\nExposure control not found")

    # -------- parse gain --------
    gain_match = re.search(r"Gain.*:.*min=(\d+)\s+max=(\d+)\s+step=(\d+).*value=(\d+)", info)
    if gain_match:
        mn, mx, step, val = map(int, gain_match.groups())
        print(f"\nGain range       : {mn} to {mx}")
        print(f"Gain step        : {step}")
        print(f"Current gain     : {val}")
    else:
        print("\nGain control not found")

    # -------- list formats & frame sizes --------
    print("\n=== Supported formats & frame sizes ===")
    run_cmd(f"v4l2-ctl --list-formats-ext -d /dev/video{CAM_INDEX}")


def test_with_opencv():
    print("\n=== Using OpenCV probing ===")
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Try some common resolutions
    common_sizes = [
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1920, 1080),
    ]

    print("\nTesting supported frame sizes:")

    for w, h in common_sizes:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        rw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        rh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        ok = (int(rw) == w and int(rh) == h)
        print(f"Requested {w}x{h} -> Got {int(rw)}x{int(rh)}  ({'OK' if ok else 'NOT SUPPORTED'})")

    print("\nProbing exposure…")
    base = cap.get(cv2.CAP_PROP_EXPOSURE)
    print(f"Current exposure: {base}")

    for delta in [-1000, -100, -10, 10, 100, 1000]:
        val = base + delta
        cap.set(cv2.CAP_PROP_EXPOSURE, val)
        actual = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"Requested {val} -> driver set {actual}")

    print("\nProbing gain…")
    base = cap.get(cv2.CAP_PROP_GAIN)
    print(f"Current gain: {base}")

    for delta in [-10, -1, 1, 5, 10]:
        val = base + delta
        cap.set(cv2.CAP_PROP_GAIN, val)
        actual = cap.get(cv2.CAP_PROP_GAIN)
        print(f"Requested {val} -> driver set {actual}")

    cap.release()


if __name__ == "__main__":
    system = platform.system().lower()

    if "linux" in system:
        test_with_v4l2()

    test_with_opencv()
