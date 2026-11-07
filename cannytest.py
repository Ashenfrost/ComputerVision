import sys
from datetime import datetime
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np


# ============================================================
# LOAD IMAGE
# ============================================================

# Run with an image path to test another image:
#   python cannytest.py 20.1.jpg
# Add "--show" to also open the figures in windows.
args = [a for a in sys.argv[1:] if not a.startswith("--")]
image_path = args[0] if args else '30.2.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# ============================================================
# 1. BLUR COMPARISON
# ============================================================

Gaussian = cv2.GaussianBlur(
    image,
    (7, 7),
    0
)

median = cv2.medianBlur(
    image,
    13
)

bilateral = cv2.bilateralFilter(
    image,
    15,
    150,
    150
)


blur_images = [
    (
        "Original Image",
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Gaussian Blurred Image",
        cv2.cvtColor(
            Gaussian,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Median Blurred Image",
        cv2.cvtColor(
            median,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Bilateral Blurred Image",
        cv2.cvtColor(
            bilateral,
            cv2.COLOR_BGR2RGB
        )
    ),
]


fig1, axes1 = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

for ax, (title, img) in zip(
    axes1.flatten(),
    blur_images
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

fig1.tight_layout()


# ============================================================
# 2. CANNY EDGE DETECTION
# ============================================================

gray_original = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

gray_gaussian = cv2.cvtColor(
    Gaussian,
    cv2.COLOR_BGR2GRAY
)

gray_median = cv2.cvtColor(
    median,
    cv2.COLOR_BGR2GRAY
)

gray_bilateral = cv2.cvtColor(
    bilateral,
    cv2.COLOR_BGR2GRAY
)


canny_original = cv2.Canny(
    gray_original,
    50,
    150
)

canny_gaussian = cv2.Canny(
    gray_gaussian,
    50,
    150
)

canny_median = cv2.Canny(
    gray_median,
    50,
    150
)

canny_bilateral = cv2.Canny(
    gray_bilateral,
    50,
    150
)


canny_images = [
    (
        "Canny - Original",
        canny_original
    ),
    (
        "Canny - Gaussian",
        canny_gaussian
    ),
    (
        "Canny - Median",
        canny_median
    ),
    (
        "Canny - Bilateral",
        canny_bilateral
    ),
]


fig2, axes2 = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

for ax, (title, img) in zip(
    axes2.flatten(),
    canny_images
):
    ax.imshow(
        img,
        cmap='gray'
    )
    ax.set_title(title)
    ax.axis('off')

fig2.tight_layout()


# ============================================================
# 3. CONTOURS FROM CANNY
# ============================================================

contours_original, _ = cv2.findContours(
    canny_original,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

contours_gaussian, _ = cv2.findContours(
    canny_gaussian,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

contours_median, _ = cv2.findContours(
    canny_median,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

contours_bilateral, _ = cv2.findContours(
    canny_bilateral,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)


contour_original = image.copy()
contour_gaussian = Gaussian.copy()
contour_median = median.copy()
contour_bilateral = bilateral.copy()


cv2.drawContours(
    contour_original,
    contours_original,
    -1,
    (0, 255, 0),
    2
)

cv2.drawContours(
    contour_gaussian,
    contours_gaussian,
    -1,
    (0, 255, 0),
    2
)

cv2.drawContours(
    contour_median,
    contours_median,
    -1,
    (0, 255, 0),
    2
)

cv2.drawContours(
    contour_bilateral,
    contours_bilateral,
    -1,
    (0, 255, 0),
    2
)


contour_images = [
    (
        "Contours - Original",
        cv2.cvtColor(
            contour_original,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Contours - Gaussian",
        cv2.cvtColor(
            contour_gaussian,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Contours - Median",
        cv2.cvtColor(
            contour_median,
            cv2.COLOR_BGR2RGB
        )
    ),
    (
        "Contours - Bilateral",
        cv2.cvtColor(
            contour_bilateral,
            cv2.COLOR_BGR2RGB
        )
    ),
]


fig3, axes3 = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

for ax, (title, img) in zip(
    axes3.flatten(),
    contour_images
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

fig3.tight_layout()


# ============================================================
# 4. HOUGH CIRCLE DETECTION
# ============================================================

hough_images = []


for name, blurred in [
    ("Hough - Original", image),
    ("Hough - Gaussian", Gaussian),
    ("Hough - Median", median),
    ("Hough - Bilateral", bilateral),
]:

    gray_blurred = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray_blurred,
        50,
        150
    )

    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=60,
        param1=200,
        param2=35,
        minRadius=30,
        maxRadius=220
    )

    detected = blurred.copy()

    if circles is not None:

        circles = np.round(
            circles[0, :]
        ).astype("int")

        for (x, y, r) in circles:

            cv2.circle(
                detected,
                (x, y),
                r,
                (0, 255, 0),
                2
            )

            cv2.circle(
                detected,
                (x, y),
                2,
                (0, 0, 255),
                3
            )

    hough_images.append(
        (
            name,
            cv2.cvtColor(
                detected,
                cv2.COLOR_BGR2RGB
            )
        )
    )


fig4, axes4 = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

for ax, (title, img) in zip(
    axes4.flatten(),
    hough_images
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

fig4.tight_layout()


# ============================================================
# 5. BLOB DETECTION
# ============================================================

blob_images = []


for name, blurred in [
    ("Blob - Original", image),
    ("Blob - Gaussian", Gaussian),
    ("Blob - Median", median),
    ("Blob - Bilateral", bilateral),
]:

    gray_blurred = cv2.cvtColor(
        blurred,
        cv2.COLOR_BGR2GRAY
    )

    gray_blurred = cv2.GaussianBlur(
        gray_blurred,
        (9, 9),
        0
    )

    thresholded = cv2.adaptiveThreshold(
        gray_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 200000

    params.filterByCircularity = True
    params.minCircularity = 0.05

    params.filterByConvexity = True
    params.minConvexity = 0.05

    params.filterByInertia = True
    params.minInertiaRatio = 0.05

    params.minDistBetweenBlobs = 10

    detector = cv2.SimpleBlobDetector_create(
        params
    )

    keypoints = detector.detect(
        thresholded
    )

    detected = blurred.copy()

    for kp in keypoints:

        x, y = map(
            int,
            kp.pt
        )

        r = int(
            kp.size / 2
        )

        cv2.circle(
            detected,
            (x, y),
            r,
            (0, 255, 0),
            2
        )

        cv2.circle(
            detected,
            (x, y),
            2,
            (0, 0, 255),
            3
        )

    blob_images.append(
        (
            name,
            cv2.cvtColor(
                detected,
                cv2.COLOR_BGR2RGB
            )
        )
    )


fig5, axes5 = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)

for ax, (title, img) in zip(
    axes5.flatten(),
    blob_images
):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

fig5.tight_layout()


# ============================================================
# 6. HSV + RED CONTOURS - RESTRICTED
# ============================================================

hsv = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2HSV
)


# ------------------------------------------------------------
# RED HSV RANGE
# ------------------------------------------------------------

# More restrictive red range
lower_red1 = np.array([
    0,
    100,
    70
])

upper_red1 = np.array([
    8,
    255,
    255
])


lower_red2 = np.array([
    172,
    100,
    70
])

upper_red2 = np.array([
    180,
    255,
    255
])


mask1 = cv2.inRange(
    hsv,
    lower_red1,
    upper_red1
)

mask2 = cv2.inRange(
    hsv,
    lower_red2,
    upper_red2
)


red_mask = mask1 | mask2


# ------------------------------------------------------------
# MORPHOLOGICAL FILTERING
# ------------------------------------------------------------

kernel = np.ones(
    (5, 5),
    np.uint8
)

# Close small gaps in the red ring
red_mask = cv2.morphologyEx(
    red_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# Remove small isolated red regions
red_mask = cv2.morphologyEx(
    red_mask,
    cv2.MORPH_OPEN,
    kernel
)


# ------------------------------------------------------------
# FIND RED CONTOURS
# ------------------------------------------------------------

red_contours, _ = cv2.findContours(
    red_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


hsv_detected = image.copy()


# ------------------------------------------------------------
# FILTER CONTOURS
# ------------------------------------------------------------

for contour in red_contours:

    area = cv2.contourArea(
        contour
    )

    # Ignore small red areas
    if area < 300:
        continue


    # Bounding box
    x, y, w, h = cv2.boundingRect(
        contour
    )


    # Ignore extremely small bounding boxes
    if w < 20 or h < 20:
        continue


    # Aspect ratio
    aspect_ratio = w / float(h)


    # The sign should be approximately square
    if aspect_ratio < 0.65 or aspect_ratio > 1.35:
        continue


    # --------------------------------------------------------
    # CIRCULARITY
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        continue


    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )


    # Require a reasonably round shape
    if circularity < 0.45:
        continue


    # --------------------------------------------------------
    # DRAW DETECTED SIGN
    # --------------------------------------------------------

    cv2.drawContours(
        hsv_detected,
        [contour],
        -1,
        (0, 255, 0),
        3
    )


    cv2.rectangle(
        hsv_detected,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )


    # Display information
    label = f"Area: {int(area)}"

    cv2.putText(
        hsv_detected,
        label,
        (x, max(y - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

fig6, axes6 = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


axes6[0].imshow(
    red_mask,
    cmap='gray'
)

axes6[0].set_title(
    "HSV - Restricted Red Mask"
)

axes6[0].axis('off')


axes6[1].imshow(
    cv2.cvtColor(
        hsv_detected,
        cv2.COLOR_BGR2RGB
    )
)

axes6[1].set_title(
    "HSV + Contours + Shape Filtering"
)

axes6[1].axis('off')


fig6.tight_layout()



# ============================================================
# 7. HSV CONTOURS + SHAPE ANALYSIS
# ============================================================

shape_detected = image.copy()

# Use the red HSV mask from section 6
contours, _ = cv2.findContours(
    red_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)


for contour in contours:

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area = cv2.contourArea(contour)

    if area < 100:
        continue


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(contour)

    if w < 15 or h < 15:
        continue


    # --------------------------------------------------------
    # Aspect ratio
    # --------------------------------------------------------

    aspect_ratio = w / float(h)

    # Sign should be approximately square
    if aspect_ratio < 0.5 or aspect_ratio > 1.5:
        continue


    # --------------------------------------------------------
    # Perimeter
    # --------------------------------------------------------

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        continue


    # --------------------------------------------------------
    # Circularity
    # --------------------------------------------------------

    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )


    # --------------------------------------------------------
    # Polygon approximation
    # --------------------------------------------------------

    epsilon = 0.02 * perimeter

    approx = cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )


    # --------------------------------------------------------
    # Determine shape
    # --------------------------------------------------------

    if circularity > 0.55:
        shape = "Circle"

    elif len(approx) == 3:
        shape = "Triangle"

    elif len(approx) == 4:
        shape = "Rectangle"

    else:
        shape = "Other"


    # --------------------------------------------------------
    # Draw candidate
    # --------------------------------------------------------

    cv2.drawContours(
        shape_detected,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    cv2.rectangle(
        shape_detected,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )


    # --------------------------------------------------------
    # Display information
    # --------------------------------------------------------

    label = (
        f"{shape} "
        f"A={int(area)} "
        f"C={circularity:.2f}"
    )

    cv2.putText(
        shape_detected,
        label,
        (x, max(y - 10, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# DISPLAY
# ============================================================

fig7, axes7 = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# Show original
axes7[0].imshow(
    cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
)

axes7[0].set_title(
    "Original"
)

axes7[0].axis('off')


# Show shape analysis
axes7[1].imshow(
    cv2.cvtColor(
        shape_detected,
        cv2.COLOR_BGR2RGB
    )
)

axes7[1].set_title(
    "HSV Contours + Shape Analysis"
)

axes7[1].axis('off')


fig7.tight_layout()


# ============================================================
# 8. KEYPOINT DETECTORS
# ============================================================

detector_factories = [
    ("ORB", cv2.ORB_create),
    ("SIFT", cv2.SIFT_create),
]

def draw_keypoints(
    source_image,
    detector_factory
):
    detector = detector_factory()
    gray_source = cv2.cvtColor(
        source_image,
        cv2.COLOR_BGR2GRAY
    )
    keypoints = detector.detect(gray_source, None)
    detected = cv2.drawKeypoints(
        source_image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    return detected, len(keypoints)


detector_inputs = [
    ("Original", image),
    ("Gaussian", Gaussian),
    ("Median", median),
    ("Bilateral", bilateral),
]


# Compare ORB and SIFT on the original image.
fig8, axes8 = plt.subplots(
    1,
    2,
    figsize=(12, 6)
)

for axis, (detector_name, detector_factory) in zip(
    axes8,
    detector_factories
):
    detected, keypoint_count = draw_keypoints(
        image,
        detector_factory
    )
    axis.imshow(
        cv2.cvtColor(
            detected,
            cv2.COLOR_BGR2RGB
        )
    )
    axis.set_title(
        f"{detector_name} - Original ({keypoint_count} keypoints)"
    )
    axis.axis('off')

fig8.tight_layout()


# Compare each detector across all blur types.
for figure_number, (detector_name, detector_factory) in enumerate(
    detector_factories,
    start=9
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    for axis, (blur_name, blurred_image) in zip(
        axes.flatten(),
        detector_inputs
    ):
        detected, keypoint_count = draw_keypoints(
            blurred_image,
            detector_factory
        )
        axis.imshow(
            cv2.cvtColor(
                detected,
                cv2.COLOR_BGR2RGB
            )
        )
        axis.set_title(
            f"{detector_name} - {blur_name} ({keypoint_count} keypoints)"
        )
        axis.axis('off')

    figure.suptitle(
        f"{detector_name} Keypoints Across Blur Types"
    )
    figure.tight_layout()


# ============================================================
# 9. FLANN MATCHING AGAINST 30.1
# ============================================================

comparison_paths = [
    f"30.{number}.jpg"
    for number in range(1, 6)
]

comparison_images = []

for comparison_path in comparison_paths:
    comparison_image = cv2.imread(comparison_path)

    if comparison_image is None:
        raise FileNotFoundError(
            f"Could not load comparison image: {comparison_path}"
        )

    comparison_images.append(
        comparison_image
    )


sift_matcher = cv2.SIFT_create()

reference_keypoints, reference_descriptors = sift_matcher.detectAndCompute(
    comparison_images[0],
    None
)

flann_index_params = {
    "algorithm": 1,
    "trees": 5,
}

flann_search_params = {
    "checks": 50,
}

flann_matcher = cv2.FlannBasedMatcher(
    flann_index_params,
    flann_search_params
)

match_images = []

for comparison_path, comparison_image in zip(
    comparison_paths,
    comparison_images
):
    comparison_keypoints, comparison_descriptors = (
        sift_matcher.detectAndCompute(
            comparison_image,
            None
        )
    )

    good_matches = []

    if (
        reference_descriptors is not None
        and comparison_descriptors is not None
    ):
        matches = flann_matcher.knnMatch(
            reference_descriptors,
            comparison_descriptors,
            k=2
        )

        for first_match, second_match in matches:
            if first_match.distance < 0.7 * second_match.distance:
                good_matches.append(first_match)

    good_matches = sorted(
        good_matches,
        key=lambda match: match.distance
    )

    match_image = cv2.drawMatches(
        comparison_images[0],
        reference_keypoints,
        comparison_image,
        comparison_keypoints,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    match_images.append(
        (
            comparison_path,
            match_image,
            len(good_matches)
        )
    )


fig10, axes10 = plt.subplots(
    2,
    3,
    figsize=(18, 10)
)

for axis, (comparison_path, match_image, match_count) in zip(
    axes10.flatten(),
    match_images
):
    axis.imshow(
        cv2.cvtColor(
            match_image,
            cv2.COLOR_BGR2RGB
        )
    )
    axis.set_title(
        f"30.1 vs {comparison_path} ({match_count} good matches)"
    )
    axis.axis('off')

for axis in axes10.flatten()[len(match_images):]:
    axis.axis('off')

fig10.suptitle(
    "FLANN SIFT Keypoint Matches Against 30.1"
)
fig10.tight_layout()


# ============================================================
# SAVE ALL FIGURES
# ============================================================

# Every run gets its own folder, so old outputs are never overwritten:
#   results/cannytest_<image>_<time>/
#       figures/        one PNG per comparison figure (for the report)
#       single_images/  every panel on its own, at original resolution
RUN_ID = f"cannytest_{Path(image_path).stem}_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR = Path("results") / RUN_ID
(OUT_DIR / "figures").mkdir(parents=True)
(OUT_DIR / "single_images").mkdir()

# Same order as the figures are created above
figure_names = [
    "01_blur_comparison",
    "02_canny",
    "03_contours",
    "04_hough_circles",
    "05_blob_detection",
    "06_hsv_red_mask",
    "07_hsv_shape_analysis",
    "08_orb_vs_sift",
    "09_orb_blur_types",
    "10_sift_blur_types",
    "11_flann_matching",
]
assert len(figure_names) == len(plt.get_fignums()), "update figure_names when adding/removing figures"

for name, number in zip(figure_names, plt.get_fignums()):
    figure = plt.figure(number)
    figure.savefig(OUT_DIR / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")

    # Save each panel without axes/title, using the panel title as filename
    for axis in figure.axes:
        if not axis.images:
            continue
        panel = axis.images[0]
        title = axis.get_title() or "panel"
        safe_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
        plt.imsave(
            OUT_DIR / "single_images" / f"{name}__{safe_title}.png",
            panel.get_array(),
            cmap=panel.get_cmap() if panel.get_array().ndim == 2 else None
        )

print(f"Saved {len(figure_names)} figures to {OUT_DIR}")

if "--show" in sys.argv:
    plt.show()
