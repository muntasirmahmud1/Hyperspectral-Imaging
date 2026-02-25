import platform
import subprocess
import cv2


def v4l2_set_controls(dev: str, controls: dict) -> None:
    """
    Set multiple V4L2 controls using v4l2-ctl.
    Example: v4l2_set_controls("/dev/video0", {"exposure_auto": 1, "exposure_absolute": 500})
    """
    args = ["v4l2-ctl", "-d", dev]
    for k, v in controls.items():
        args += ["-c", f"{k}={v}"]
    subprocess.run(args, check=False)

def set_uvc_color_manual(dev="/dev/video2", exposure=700, gain=0):
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_auto=1"], check=False)
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_auto_priority=0"], check=False)
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"exposure_absolute={int(exposure)}"], check=False)
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"gain={int(gain)}"], check=False)

# =========================================================
# ELP CAMERA CLASSES
# =========================================================
class ELPCamera:
    """
    HSI camera wrapper:
      - Windows: uses CAP_DSHOW and OpenCV exposure/gain
      - Linux: uses CAP_V4L2 for capture, but sets exposure/gain via v4l2-ctl for stability
    """
    def __init__(self, cam_index: int, dev: str, width: int, height: int,
                 exposure_start: int, gain_start: int):
        self.system = platform.system().lower()
        self.dev = dev
        self.width = width
        self.height = height

        backend = cv2.CAP_DSHOW if "windows" in self.system else cv2.CAP_V4L2
        print(f"[HSI] Opening camera index {cam_index} with backend {backend}")
        self.cap = cv2.VideoCapture(cam_index, backend)

        if not self.cap.isOpened():
            print("[HSI] Failed with chosen backend, trying default backend...")
            self.cap.release()
            self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            raise RuntimeError("[HSI] Cannot open camera. Check index or connection.")

        # Basic properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Put in manual mode
        if "windows" in self.system:
            # 1 manual, 3 auto (varies by driver)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            # Some drivers accept negative exposure values (log scale); keep if you want:
            # self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
            self.cap.set(cv2.CAP_PROP_GAIN, float(gain_start))
            self.set_exposure(exposure_start)
        else:
            # Linux: force manual via v4l2 controls (UVC)
            self.set_manual_mode(exposure_start, gain_start)

        self.show_properties()

    def show_properties(self) -> None:
        print("\n=== [HSI] Current OpenCV Camera Properties ===")
        props = {
            "FRAME_WIDTH": cv2.CAP_PROP_FRAME_WIDTH,
            "FRAME_HEIGHT": cv2.CAP_PROP_FRAME_HEIGHT,
            "BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
            "CONTRAST": cv2.CAP_PROP_CONTRAST,
            "SATURATION": cv2.CAP_PROP_SATURATION,
            "GAIN": cv2.CAP_PROP_GAIN,
            "EXPOSURE": cv2.CAP_PROP_EXPOSURE,
        }
        for name, code in props.items():
            print(f"{name:15s}: {self.cap.get(code)}")
        print("=============================================\n")

    def set_manual_mode(self, exposure: int, gain: int) -> None:
        """Linux: make sure camera is in manual exposure mode + set exposure/gain."""
        if "windows" in self.system:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            self.cap.set(cv2.CAP_PROP_GAIN, float(gain))
        else:
            exposure = int(max(1, min(500, exposure)))
            gain = int(max(0, min(100, gain)))
            v4l2_set_controls(self.dev, {
                "exposure_auto": 1,                # Manual Mode
                "exposure_auto_priority": 0,
                "exposure_absolute": exposure,
                "gain": gain
            })

    def set_exposure(self, value: int) -> int:
        if "windows" in self.system:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
            actual = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            print(f"[HSI] Exposure requested={value}, actual={actual}")
            return actual
        else:
            value = int(max(1, min(500, value)))
            v4l2_set_controls(self.dev, {"exposure_auto": 1, "exposure_absolute": value})
            print(f"[HSI] exposure_absolute set to {value}")
            return value

    def set_gain(self, value: int) -> int:
        if "windows" in self.system:
            success = self.cap.set(cv2.CAP_PROP_GAIN, float(value))
            actual = self.cap.get(cv2.CAP_PROP_GAIN)
            print(f"[HSI] Gain requested={value}, actual={actual}, success={success}")
            return actual
        else:
            value = int(max(0, min(100, value)))
            v4l2_set_controls(self.dev, {"gain": value})
            print(f"[HSI] gain set to {value}")
            return value

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("[HSI] Failed to capture frame.")
        return frame

    def release(self):
        self.cap.release()