"""Stage 7 (geometric part): classify a contour's 2D outline.

Important limitation: this classifies the silhouette the camera SEES.
- Upright peg, camera looking straight down -> the top face -> true cross-section.
- Peg lying on its side -> a long rectangle (or a parallelogram/trapezoid),
  regardless of whether it is a hexagonal, square or triangular prism.
So the node also reports the colour->shape mapping from the config, and labels
which source each shape came from.
"""
import math

import cv2


def classify_shape(contour, epsilon_frac: float = 0.03):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if area <= 0 or peri <= 0:
        return 'unknown', 'none', {}

    circularity = 4.0 * math.pi * area / (peri * peri)       # 1.0 = perfect circle
    approx = cv2.approxPolyDP(contour, epsilon_frac * peri, True)
    n = len(approx)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    aspect = max(rw, rh) / max(min(rw, rh), 1e-6)

    feats = {'vertices': n, 'circularity': round(circularity, 3),
             'solidity': round(solidity, 3), 'aspect': round(aspect, 2)}

    if n == 3:
        label = 'triangle'
    elif n == 4:
        label = 'square' if aspect < 1.25 else 'rectangle'
    elif n == 6 and aspect < 1.3:
        label = 'hexagon'
    elif n >= 8 and circularity > 0.80 and aspect < 1.25:
        label = 'circle'
    elif n in (5, 7) and circularity > 0.80 and aspect < 1.3:
        label = 'hexagon'           # noisy hexagon outline often gives 5 or 7
    else:
        label = 'unknown'

    confidence = 'high' if (solidity > 0.92 and label not in ('unknown', 'rectangle')) else 'low'
    return label, confidence, feats
