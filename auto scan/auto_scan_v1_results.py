import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
SCAN_DIR = r"/home/sciglob/HSI_code/hsi/hsi_captures/live/auto_scan_20260203_114519"
SHIFT_MATRIX_PATH = r"/home/sciglob/HSI_code/hsi/hsi_captures/smile_shift_matrix.npy"

# Reconstruction band (spectral axis)
X_START = 0
X_END   = None   # None => use full width

# X_START = 200
# X_END   = 300

# Use corrected frames for reconstruction
APPLY_SMILE_CORR = True #False
APPLY_DARK_CORR  = True

# If you want to save outputs
SAVE_RECON_PNG = True


# =========================================================
# Smile correction via cv2.remap (fast)
# =========================================================
def apply_shift_matrix_gray(gray_u8: np.ndarray, shift_matrix: np.ndarray) -> np.ndarray:
    """
    Applies smile correction to a grayscale image using a precomputed shift_matrix.
    Supported shift_matrix shapes:
      - (H, W): per-pixel horizontal shift for each row/col
      - (H,): per-row constant horizontal shift
    Positive shift means sampling from the right (i.e., moves content left), depending on how matrix was built.
    """
    if gray_u8.ndim != 2:
        raise ValueError("apply_shift_matrix_gray expects a 2D grayscale image")

    H, W = gray_u8.shape

    sm = np.asarray(shift_matrix)
    if sm.ndim == 1:
        if sm.shape[0] != H:
            raise ValueError(f"shift_matrix shape {sm.shape} incompatible with image height {H}")
        # broadcast to (H, W)
        sm = sm[:, None].astype(np.float32) * np.ones((1, W), dtype=np.float32)
    elif sm.ndim == 2:
        if sm.shape != (H, W):
            raise ValueError(f"shift_matrix shape {sm.shape} incompatible with image shape {(H, W)}")
        sm = sm.astype(np.float32)
    else:
        raise ValueError(f"Unsupported shift_matrix ndim={sm.ndim}")

    # Build remap grids
    x = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)   # (H, W)
    y = np.arange(H, dtype=np.float32)[:, None].repeat(W, axis=1)   # (H, W)

    map_x = x + sm
    map_y = y

    # Remap
    # BORDER_CONSTANT will fill out-of-range with 0
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
# Load scan + dark
# =========================================================
def find_dark_avg(scan_dir: str):
    dark_files = sorted(glob.glob(os.path.join(scan_dir, "dark_avg_*.png")))
    if not dark_files:
        return None, None
    dark_path = dark_files[-1]  # take latest if multiple
    dark = cv2.imread(dark_path, cv2.IMREAD_GRAYSCALE)
    if dark is None:
        raise RuntimeError(f"Failed to read dark image: {dark_path}")
    return dark.astype(np.float32), dark_path


def load_scan_frames(scan_dir: str):
    frame_files = sorted(glob.glob(os.path.join(scan_dir, "frame_pos_*.png")))
    if not frame_files:
        raise RuntimeError(f"No scan frames found in: {scan_dir}")
    frames = []
    for fp in frame_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {fp}")
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (N, H, W)
    return frames, frame_files


# =========================================================
# Main
# =========================================================
def main():
    print(f"Scan dir: {SCAN_DIR}")

    # Load shift matrix
    shift_matrix = None
    if APPLY_SMILE_CORR:
        if not os.path.exists(SHIFT_MATRIX_PATH):
            raise RuntimeError(f"Shift matrix not found: {SHIFT_MATRIX_PATH}")
        shift_matrix = np.load(SHIFT_MATRIX_PATH)
        print(f"Loaded shift matrix: {SHIFT_MATRIX_PATH} shape={shift_matrix.shape}")

    # Load dark avg
    dark, dark_path = find_dark_avg(SCAN_DIR)
    if APPLY_DARK_CORR:
        if dark is None:
            raise RuntimeError(
                "APPLY_DARK_CORR=True but no dark_avg_*.png found in scan folder. "
                "Make sure your auto-scan saved dark_avg_....png in the same folder."
            )
        print(f"Loaded dark avg: {dark_path}")
    else:
        print("Dark correction disabled.")

    # Load scan frames
    scan_u8, frame_files = load_scan_frames(SCAN_DIR)
    N, H, W = scan_u8.shape
    print(f"Loaded {N} scan frames: shape={scan_u8.shape}")

    # Band selection
    xs = int(X_START)
    xe = W if X_END is None else int(X_END)
    xe = max(xs + 1, min(xe, W))
    print(f"Using spectral band: x=[{xs}:{xe}] (width {xe-xs})")

    # Process frames -> spatial profiles
    spatial_profiles = np.zeros((N, H), dtype=np.float32)

    for i in range(N):
        img = scan_u8[i]

        # Smile correction
        if APPLY_SMILE_CORR:
            img = apply_shift_matrix_gray(img, shift_matrix)

        img_f = img.astype(np.float32)

        # Dark correction
        if APPLY_DARK_CORR:
            img_f = img_f - dark
            img_f[img_f < 0] = 0

        # Collapse spectral axis -> (H,)
        spatial_profiles[i] = np.mean(img_f[:, xs:xe], axis=1)

        if (i + 1) % 25 == 0 or i == N - 1:
            print(f"Processed {i+1}/{N}")

    # Reconstructed spatial image: (H, N)
    recon = spatial_profiles.T  # rows=y, cols=scan step
    # recon = np.hstack((recon[:, 2001:3002], recon[:, 0:2001]))

    # Plot
    plt.figure(figsize=(4, 3))
    plt.imshow(recon, aspect='auto', cmap='gray', origin='upper')
    # plt.imshow(recon, aspect='auto', cmap='turbo', origin='upper')

    plt.colorbar(label="Intensity (dark-corrected, a.u.)")
    plt.xlabel("Scan step (frame index)")
    plt.ylabel("Spatial pixel (slit direction)")
    title_bits = []
    if APPLY_SMILE_CORR:
        title_bits.append("smile-corrected")
    if APPLY_DARK_CORR:
        title_bits.append("dark-corrected")
    title = "Reconstruction (" + ", ".join(title_bits) + ")"
    plt.title(title)
    plt.tight_layout()
    plt.show()

    # Save recon image (normalized to 8-bit for viewing)
    if SAVE_RECON_PNG:
        recon_norm = recon - recon.min()
        denom = (recon_norm.max() - recon_norm.min())
        if denom <= 0:
            denom = 1.0
        recon_u8 = (255.0 * (recon_norm / denom)).astype(np.uint8)

        out_path = os.path.join(SCAN_DIR, "recon_smile_dark_corrected.png")
        cv2.imwrite(out_path, recon_u8)
        print(f"Saved recon image: {out_path}")


if __name__ == "__main__":
    main()
