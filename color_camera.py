import platform
import time
import cv2


class ColorCamera:
    """
    Robust color camera wrapper.
    - If the camera isn't available, it keeps running (returns None frames).
    - It periodically retries opening.
    - It never raises in normal operation.
    """

    def __init__(
        self,
        src=2,                 # can be int index OR "/dev/video2"
        width=640,
        height=480,
        retry_sec=2.0,
        backend_linux=cv2.CAP_V4L2,
        backend_windows=cv2.CAP_DSHOW,
        name="[COLOR]",
        open_timeout_sec=0.0,  # optional: delay after open (rarely needed)
    ):
        self.system = platform.system().lower()
        self.name = name

        self.src = src
        self.width = int(width)
        self.height = int(height)

        self.backend = backend_windows if "windows" in self.system else backend_linux

        self.retry_sec = float(retry_sec)
        self.next_retry_ts = 0.0

        self.cap = None
        self.ok = False
        self.open_timeout_sec = float(open_timeout_sec)

        self._try_open(force=True)

    def _try_open(self, force=False):
        now = time.time()
        if not force and now < self.next_retry_ts:
            return

        self.next_retry_ts = now + self.retry_sec

        # close any existing capture
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        print(f"{self.name} Opening camera src={self.src} backend={self.backend}")
        cap = cv2.VideoCapture(self.src, self.backend)

        if not cap.isOpened():
            print(f"{self.name} Failed with chosen backend, trying default backend...")
            cap.release()
            cap = cv2.VideoCapture(self.src)

        if not cap.isOpened():
            print(f"{self.name} Not available (will retry every {self.retry_sec:.1f}s)")
            self.ok = False
            return

        # Configure
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm-up delay (optional)
        if self.open_timeout_sec > 0:
            time.sleep(self.open_timeout_sec)

        self.cap = cap
        self.ok = True
        print(f"{self.name} OK: opened")

    def read(self):
        """
        Returns:
            frame (np.ndarray BGR) or None if not available.
        """
        if not self.ok or self.cap is None:
            self._try_open(force=False)
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print(f"{self.name} Read failed. Marking camera offline (will retry).")
            self.ok = False
            self._try_open(force=False)
            return None

        return frame

    def is_ok(self):
        return bool(self.ok)

    def release(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.ok = False