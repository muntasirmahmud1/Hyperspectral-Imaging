import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIG (edit these)
# =========================================================
SCAN_DIR = r"/home/sciglob/HSI_code/hsi/hsi_captures/live/auto_scan_20260129_123616" #auto_scan_20260120_120734
SHIFT_MATRIX_PATH = r"/home/sciglob/HSI_code/hsi/hsi_captures/smile_shift_matrix.npy"

APPLY_SMILE_CORR = True
APPLY_DARK_CORR  = True

# Which frames to plot (0-based indices into the sorted frame list)
# Examples:
#   PLOT_IDXS = list(range(0, 10))          # first 10
#   PLOT_IDXS = [0, 5, 10, 50]              # custom
#   PLOT_IDXS = list(range(20, 31))         # frames 21..31
PLOT_IDXS = list(range(0, 500))
# PLOT_IDXS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] 

# How to collapse spatial dimension to get 1D spectrum
#   "mean" or "sum"
SPATIAL_REDUCE = "mean"
# SPATIAL_REDUCE = "sum"

# Optional: use only a subset of spatial rows (can help SNR / target region)
# Set to None to use full height.
ROW_START = None  # e.g., 200
ROW_END   = None  # e.g., 600


# =========================================================
# Dispersion polynomial: wavelength(p)
# lambda = a*p^2 + b*p + c
# Given: -2.442113e-06·p^2 + 8.461860e-02·p + 3.896111e+02
# =========================================================
A = -2.442113e-06
B =  8.461860e-02
C =  3.896111e+02


# =========================================================
# Smile correction via cv2.remap (fast)
# =========================================================
def apply_shift_matrix_gray(gray_u8: np.ndarray, shift_matrix: np.ndarray, sign: int = +1) -> np.ndarray:
    """
    Apply smile correction using shift matrix with cv2.remap.
    sign=+1 uses map_x = x + shift
    sign=-1 uses map_x = x - shift  (flip if correction bends the wrong way)
    """
    if gray_u8.ndim != 2:
        raise ValueError("apply_shift_matrix_gray expects 2D grayscale image")

    H, W = gray_u8.shape
    sm = np.asarray(shift_matrix)

    if sm.ndim == 1:
        if sm.shape[0] != H:
            raise ValueError(f"shift_matrix shape {sm.shape} incompatible with image height {H}")
        sm = sm[:, None].astype(np.float32) * np.ones((1, W), dtype=np.float32)
    elif sm.ndim == 2:
        if sm.shape != (H, W):
            raise ValueError(f"shift_matrix shape {sm.shape} incompatible with image shape {(H, W)}")
        sm = sm.astype(np.float32)
    else:
        raise ValueError(f"Unsupported shift_matrix ndim={sm.ndim}")

    x = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    y = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)

    map_x = x + (sign * sm)
    map_y = y

    corrected = cv2.remap(
        gray_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return corrected


# =========================================================
# Load helpers
# =========================================================
def find_dark_avg(scan_dir: str):
    dark_files = sorted(glob.glob(os.path.join(scan_dir, "dark_avg_*.png")))
    if not dark_files:
        return None, None
    dark_path = dark_files[-1]
    dark = cv2.imread(dark_path, cv2.IMREAD_GRAYSCALE)
    if dark is None:
        raise RuntimeError(f"Failed to read dark image: {dark_path}")
    return dark.astype(np.float32), dark_path


def load_scan_frames(scan_dir: str):
    frame_files = sorted(glob.glob(os.path.join(scan_dir, "frame_pos_*.png")))
    if not frame_files:
        raise RuntimeError(f"No scan frames found in: {scan_dir}")
    return frame_files


def pixel_to_wavelength(pix: np.ndarray) -> np.ndarray:
    pix = pix.astype(np.float64)
    return (A * pix * pix) + (B * pix) + C


def spatial_reduce_to_spectrum(img_f32: np.ndarray, row_start=None, row_end=None, mode="mean") -> np.ndarray:
    """
    Convert (H,W) image -> (W,) spectrum by summing/averaging over rows.
    """
    H, W = img_f32.shape
    rs = 0 if row_start is None else int(row_start)
    re = H if row_end is None else int(row_end)
    rs = max(0, min(rs, H - 1))
    re = max(rs + 1, min(re, H))

    roi = img_f32[rs:re, :]  # (rows, W)
    if mode == "sum":
        return np.sum(roi, axis=0)
    return np.mean(roi, axis=0)


# =========================================================
# Main
# =========================================================
def main():
    print(f"Scan dir: {SCAN_DIR}")

    # Load frame list
    frame_files = load_scan_frames(SCAN_DIR)
    print(f"Found {len(frame_files)} scan frames")

    # Dark
    dark, dark_path = find_dark_avg(SCAN_DIR)
    if APPLY_DARK_CORR:
        if dark is None:
            raise RuntimeError(
                "APPLY_DARK_CORR=True but no dark_avg_*.png found in scan folder."
            )
        print(f"Loaded dark avg: {dark_path}")
    else:
        print("Dark correction disabled.")

    # Shift matrix
    shift_matrix = None
    if APPLY_SMILE_CORR:
        if not os.path.exists(SHIFT_MATRIX_PATH):
            raise RuntimeError(f"Shift matrix not found: {SHIFT_MATRIX_PATH}")
        shift_matrix = np.load(SHIFT_MATRIX_PATH)
        print(f"Loaded shift matrix shape={shift_matrix.shape}")

    # Wavelength axis from pixel indices
    # Read first frame to get width
    test_img = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        raise RuntimeError(f"Failed to read: {frame_files[0]}")
    H, W = test_img.shape
    pix = np.arange(W, dtype=np.float64)
    wl = pixel_to_wavelength(pix)

    # Validate plot indices
    plot_idxs = [i for i in PLOT_IDXS if 0 <= i < len(frame_files)]
    if not plot_idxs:
        raise ValueError("No valid indices in PLOT_IDXS for the available frame count.")
    print(f"Plotting frames (0-based): {plot_idxs}")

    # Plot
    plt.figure(figsize=(10, 6))

    for i in plot_idxs:
        fp = frame_files[i]
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping unreadable: {fp}")
            continue

        # Smile correction
        if APPLY_SMILE_CORR:
            img = apply_shift_matrix_gray(img, shift_matrix, sign=+1)  # flip sign to -1 if needed

        img_f = img.astype(np.float32)

        # Dark correction
        if APPLY_DARK_CORR:
            img_f = img_f - dark
            img_f[img_f < 0] = 0

        # Collapse spatial -> spectrum
        spec = spatial_reduce_to_spectrum(
            img_f,
            row_start=ROW_START,
            row_end=ROW_END,
            mode=SPATIAL_REDUCE
        )

        # Label includes frame index + motor pos parsed from filename if available
        base = os.path.basename(fp)
        label = f"frame {i}"
        # Try parsing motor pos from: frame_pos_03830_YYYY....png
        try:
            parts = base.split("_")
            if "pos" in parts:
                pidx = parts.index("pos")
                motor_pos = int(parts[pidx + 1])
                label = f"pos {motor_pos} (idx {i})"
            elif base.startswith("frame_pos_"):
                motor_pos = int(base.split("_")[2])
                label = f"pos {motor_pos} (idx {i})"
        except Exception:
            pass

        plt.plot(wl-1.2, spec, linewidth=1, label=label)

    plt.xlabel("Wavelength (nm)")
    ylabel = "Counts" if SPATIAL_REDUCE == "sum" else "Counts"
    plt.ylabel(ylabel)
    title_bits = []
    if APPLY_SMILE_CORR:
        title_bits.append("smile-corrected")
    if APPLY_DARK_CORR:
        title_bits.append("dark-corrected")
    title_bits.append(f"rows={ROW_START}:{ROW_END}" if (ROW_START is not None or ROW_END is not None) else "rows=all")
    plt.title("Spectra (" + ", ".join(title_bits) + ")")

    plt.grid(True, linestyle="--", alpha=0.7)
    # plt.legend(fontsize=8, ncol=2)
    plt.xticks(np.arange(390, 495, 10))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
