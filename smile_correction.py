import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import least_squares
from scipy.ndimage import map_coordinates
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from PIL import Image

def locate_emission_lines(image, row_range, prominence=0.7):
    """
    Locate emission lines in the image for given row range.
    Args:
        image (numpy array): The hyperspectral image (2D array).
        row_range (tuple): The range of rows to process (start_row, end_row).
        prominence (float): Minimum prominence for peak detection.
    Returns:
        emission_points (list): List of emission line coordinates [(x1, y1), (x2, y2), ...].
    """
    emission_points = []
    for y in range(row_range[0], row_range[1]):
        row_data = image[y, :]
        peaks, _ = find_peaks(row_data, prominence=prominence)
        emission_points.extend([(x, y) for x in peaks])
    return emission_points

def group_emission_lines(emission_points, eps=10, min_samples=5, y_scale=0.25):
    if len(emission_points) == 0:
        return []
    points = np.array(emission_points, dtype=float)
    scaled = points.copy()
    scaled[:, 1] *= y_scale
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled)

    clusters = []
    for lab in set(clustering.labels_):
        if lab != -1:
            clusters.append(points[clustering.labels_ == lab])
    return clusters


def fit_circular_arc(points):
    """
    Fit a circular arc to a set of points using least-squares fitting.
    Args:
        points (list): List of (x, y) points.
    Returns:
        (center_x, center_y, radius): Parameters of the fitted circle.
    """
    def residuals(params, x, y):
        cx, cy, r = params
        return np.sqrt((x - cx)**2 + (y - cy)**2) - r

    x, y = zip(*points)
    x, y = np.array(x), np.array(y)

    # Initial guess for the center and radius
    center_guess = (np.mean(x), np.mean(y))
    radius_guess = np.mean(np.sqrt((x - center_guess[0])**2 + (y - center_guess[1])**2))
    params_guess = (*center_guess, radius_guess)

    result = least_squares(residuals, params_guess, args=(x, y))
    return result.x


def signed_shift_from_circle(image_height, center_x, center_y, radius, cluster):
    """
    Compute signed horizontal shift per row using the physically correct arc branch.
    """
    y = np.arange(image_height, dtype=float)
    dy = y - center_y
    shift = np.zeros_like(y)

    valid = np.abs(dy) <= radius
    dx = np.sqrt(np.maximum(0.0, radius**2 - dy[valid]**2))

    # Two possible arc branches
    x_arc_right = center_x + dx
    x_arc_left  = center_x - dx

    # Use actual cluster points to choose the correct branch
    cluster = np.asarray(cluster)
    y_cl = cluster[:, 1].astype(int)
    x_cl = cluster[:, 0]

    # Interpolate arc x-values at cluster y locations
    x_r = np.interp(y_cl, y[valid], x_arc_right)
    x_l = np.interp(y_cl, y[valid], x_arc_left)

    # Choose the branch that best fits the data
    err_right = np.mean(np.abs(x_cl - x_r))
    err_left  = np.mean(np.abs(x_cl - x_l))

    if err_left < err_right:
        x_arc = x_arc_left
    else:
        x_arc = x_arc_right

    # Reference row = median y of detected cluster
    y_ref = int(np.median(y_cl))
    y_ref = np.clip(y_ref, 0, image_height - 1)

    x_ref = np.interp(y_ref, y[valid], x_arc)
    shift[valid] = x_arc - x_ref

    return shift

def construct_shift_matrix_multi_lines(image, clusters):
    H, W = image.shape
    shifts = []

    if len(clusters) == 0:
        raise ValueError("No emission-line clusters found. Tune prominence/eps/min_samples.")

    for cluster in clusters:
        cx, cy, r = fit_circular_arc(cluster)
        # shift_y = signed_shift_from_circle(H, cx, cy, r)
        shift_y = signed_shift_from_circle(H, cx, cy, r, cluster)
        shifts.append(shift_y)

    shifts = np.vstack(shifts)

    # median is very robust to bad lines
    shift_per_row = np.median(shifts, axis=0)

    return shift_per_row[:, None] * np.ones((1, W))


def generate_arc(center_x, center_y, radius, y_points, branch="+"):
    val = np.sqrt(np.maximum(0, radius**2 - (y_points - center_y)**2))
    return center_x + val if branch == "+" else center_x - val

def apply_shift_matrix(image, shift_matrix):
    """
    Apply the shift matrix to warp the image rows.
    Args:
        image (numpy array): The original hyperspectral image.
        shift_matrix (numpy array): The shift matrix.
    Returns:
        corrected_image (numpy array): The corrected image.
    """
    corrected_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        # x_indices = np.arange(image.shape[1]) - shift_matrix[y, :]
        x_indices = np.arange(image.shape[1]) + shift_matrix[y, :]
        x_indices = np.clip(x_indices, 0, image.shape[1] - 1)  # Ensure valid indices
        corrected_image[y, :] = map_coordinates(image[y, :], [x_indices], order=1, mode='nearest')
    return corrected_image

def normalize(data):
    n = (data - min(data)) / (max(data)- min(data))
    return n


# === Load image ===
image = plt.imread(r".\resolulation_newCam_Jan_06_2026\frame_102624_424912.png")

# Ensure float grayscale
if image.ndim == 3:
    image = image[:, :, 0]

image = image.astype(float)

# Normalize (important for peak detection consistency)
if image.max() > 1:
    image /= image.max()

H, W = image.shape
print("Image shape:", image.shape)

# === Detect emission line points ===
row_range = (0, H)

emission_points = np.array(locate_emission_lines(image, row_range=row_range,prominence=0.5))

print("Total emission points detected:", len(emission_points))


# === Group points into individual emission lines ===
clusters = group_emission_lines(
    emission_points,
    eps=15,
    min_samples=20,
    y_scale=0.25
)

print("Number of emission lines found:", len(clusters))

#########################################################################
emission_points = np.array(emission_points)
# Locate emission lines
x_detected, y_detected = emission_points[:, 0], emission_points[:, 1]

# # Step 1: Fit circular arc to detected emission lines
center_x, center_y, radius = fit_circular_arc(emission_points)

# # Generate fitted arc
y_arc = np.linspace(min(y_detected), max(y_detected), 1000)
x_arc = generate_arc(center_x, center_y, radius, y_arc)

############################################################################

# === Compute the shift matrix ===
shift_matrix = construct_shift_matrix_multi_lines(image, clusters)

# === Apply correction ===
corrected_image = apply_shift_matrix(image, shift_matrix)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray", aspect="auto")
# plt.plot(x_detected, y_detected, 'ro', markersize=2, label="Detected Emission Line")
# plt.plot(x_arc, y_arc, 'b-', linewidth=2, label="Fitted Circular Arc")
plt.title("Original")
plt.xlabel("Spectral (λ)")
plt.ylabel("Spatial (Y)")
plt.xticks(np.arange(0, 1280, 100))
plt.grid(color='r', which='major', axis='both', alpha = 0.5)

plt.subplot(1, 2, 2)
plt.imshow(corrected_image, cmap="gray", aspect="auto")
plt.title("Corrected (Smile Removed)")
plt.xlabel("Spectral (λ)")
plt.ylabel("Spatial (X)")
plt.xticks(np.arange(0, 1280, 100))
plt.grid(color='r', which='major', axis='both', alpha = 0.5)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.imshow(shift_matrix, cmap='jet', aspect='auto')
plt.colorbar(label="Shift Value")
plt.title("Frame Correction Matrix")
plt.xlabel("Spectral Dimension (λ)")
plt.ylabel("Spatial Dimension (X)")
plt.grid()
plt.show()