import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================
img_dir = r"/home/sciglob/HSI_code/hsi/hsi_captures/live/scan_20260109_094543"
pattern = os.path.join(img_dir, "frame_*.png")

N_DARK = 3              # <-- how many first frames are dark frames (set 1 if only one dark)
DO_SMILE = True         # <-- True to apply smile correction to all frames
SHIFT_MATRIX_PATH = r"/home/sciglob/HSI_code/hsi/hsi_captures/smile_shift_matrix.npy"


# Spectral range to collapse (optional)
USE_BAND = False
X_START, X_END = 0, None   # if USE_BAND=True, set X_END to an int (e.g., 900)

# =========================================================
# Smile correction helper
# =========================================================
def apply_shift_matrix(gray_img: np.ndarray, shift_matrix: np.ndarray) -> np.ndarray:
    """
    Apply per-row horizontal shift (smile correction).
    gray_img: (H, W) uint8/float32
    shift_matrix: (H, W) or (H,) or (H, Wsmall) depending on how you saved it.
                 This implementation supports (H,) or (H, W).
    Returns corrected image with same shape as input.
    """
    img = gray_img
    H, W = img.shape

    sm = shift_matrix
    if sm.ndim == 1:
        # (H,) -> broadcast to (H, W)
        if sm.shape[0] != H:
            raise ValueError(f"shift_matrix has shape {sm.shape}, expected ({H},) or ({H},{W})")
        sm = np.repeat(sm[:, None], W, axis=1)
    elif sm.ndim == 2:
        if sm.shape[0] != H:
            raise ValueError(f"shift_matrix has shape {sm.shape}, expected first dim = {H}")
        # If second dim not W, try to broadcast/clip
        if sm.shape[1] != W:
            # Common case: saved as (H, W) -> OK
            # If saved as (H, W_crop), you must crop image to match before correction.
            raise ValueError(f"shift_matrix width={sm.shape[1]} does not match image width={W}")
    else:
        raise ValueError("shift_matrix must be 1D or 2D")

    # Build source x coordinates for each row
    x = np.arange(W)[None, :]                  # (1, W)
    src_x = x + sm                             # (H, W)
    src_x = np.clip(src_x, 0, W - 1)

    # Sample using integer indexing (fast)
    src_x_int = src_x.astype(np.int32)
    y = np.arange(H)[:, None]                  # (H, 1)
    corrected = img[y, src_x_int]              # (H, W)
    return corrected


# =========================================================
# 1) Load files
# =========================================================
img_files = sorted(glob.glob(pattern))
print(f"Found {len(img_files)} image(s).")
if len(img_files) < (N_DARK + 1):
    raise RuntimeError(f"Need at least N_DARK({N_DARK}) + 1 scan image.")

# Load smile shift matrix if needed
shift_matrix = None
if DO_SMILE:
    if not os.path.exists(SHIFT_MATRIX_PATH):
        raise FileNotFoundError(f"Shift matrix not found: {SHIFT_MATRIX_PATH}")
    shift_matrix = np.load(SHIFT_MATRIX_PATH)
    print("Loaded shift matrix:", shift_matrix.shape)

# =========================================================
# 2) Read images as grayscale (+ optional smile correction)
# =========================================================
images = []
for f in img_files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {f}")

    if DO_SMILE:
        img = apply_shift_matrix(img, shift_matrix)

    images.append(img)

images = np.array(images)  # (N, H, W)
print("Images shape (N, H, W):", images.shape)

# =========================================================
# 3) Dark = average first N_DARK frames
# =========================================================
dark_stack = images[:N_DARK].astype(np.float32)   # (N_DARK, H, W)
dark = np.mean(dark_stack, axis=0)                # (H, W)

scan_imgs = images[N_DARK:].astype(np.float32)    # remaining scans

# Dark correction
scan_imgs_dc = scan_imgs - dark
scan_imgs_dc[scan_imgs_dc < 0] = 0

N_scans, H, W = scan_imgs_dc.shape
print(f"Dark frames used: {N_DARK}")
print(f"Scan images: {N_scans}, Height: {H}, Width: {W}")

# =========================================================
# 4) Collapse spectral axis to get spatial profile per scan
# =========================================================
if USE_BAND:
    x_end = W if X_END is None else int(X_END)
    x_start = int(X_START)
else:
    x_start, x_end = 0, W

spatial_profiles = np.mean(scan_imgs_dc[:, :, x_start:x_end], axis=2)  # (N_scans, H)
image_recon = spatial_profiles.T                                       # (H, N_scans)

print("Reconstructed image shape (y, scan_step):", image_recon.shape)

# =========================================================
# 5) Plot
# =========================================================
plt.figure(figsize=(7, 7))
plt.imshow(image_recon, aspect='auto', cmap='gray', origin='upper')
plt.colorbar(label="Intensity (a.u.)")
plt.xlabel("Scan step (image index)")
plt.ylabel("Spatial pixel (slit direction)")
title = f"Reconstructed object (dark avg={N_DARK})"
if DO_SMILE:
    title += " + smile corrected"
plt.title(title)
plt.tight_layout()
plt.show()

