import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Every run gets its own folder, so old outputs are never overwritten
OUT_DIR = Path("results") / f"colortest_{datetime.now():%Y%m%d_%H%M%S}"
OUT_DIR.mkdir(parents=True)

def detect_yellow_color(image_path):
    """
    This function takes an image file path, detects yellow color areas using HSV color space,
    and returns the image with bounding boxes drawn around the detected yellow areas.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for yellow color
    yellow_lower = np.array([20, 100, 100])  # Lower bound for yellow color
    yellow_upper = np.array([30, 255, 255])  # Upper bound for yellow color

    # Create a mask to isolate yellow color
    mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)

    # Find contours of the yellow areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw bounding boxes on
    output_image = image.copy()

    # Loop through each contour and draw a bounding box around the detected yellow areas
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green bounding box

    # Convert the output image to RGB for display with matplotlib
    output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    # Display the original and processed images side by side
    plt.figure(figsize=(12, 6))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # Display processed image with yellow color detection
    plt.subplot(1, 2, 2)
    plt.imshow(output_image_rgb)
    plt.title("Yellow Color Detection with Bounding Boxes")
    plt.axis('off')

    # Save the figure, the mask and the result as separate images
    name = Path(image_path).stem
    plt.savefig(OUT_DIR / f"{name}_figure.png", dpi=150, bbox_inches="tight")
    cv2.imwrite(str(OUT_DIR / f"{name}_mask.png"), mask)
    cv2.imwrite(str(OUT_DIR / f"{name}_detected.png"), output_image)
    boxes = sum(cv2.contourArea(c) > 500 for c in contours)
    (OUT_DIR / "summary.txt").write_text(
        f"{image_path}: HSV yellow {yellow_lower.tolist()}-{yellow_upper.tolist()}, "
        f"{len(contours)} contours, {boxes} with area > 500\n", encoding="utf-8")
    print(f"Saved to {OUT_DIR} ({boxes} yellow areas)")

    # Run with "--show" to also open the figure in a window
    if "--show" in sys.argv:
        plt.show()

    return output_image

# Example usage
image_path = 'Yellow.png'  # Replace with your image path
output_image = detect_yellow_color(image_path)