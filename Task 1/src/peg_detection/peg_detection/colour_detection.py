"""Stage A: HSV colour segmentation -> one clean blob per peg colour.

Pipeline per colour:
  BGR -> (blur) -> HSV -> inRange -> morphological open/close -> external
  contours -> filter by area and fill ratio -> keep largest N.

OpenCV HSV ranges: H 0-179 (degrees / 2), S 0-255, V 0-255.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class ColourSpec:
    name: str
    hsv_lower: np.ndarray
    hsv_upper: np.ndarray
    draw_bgr: tuple = (255, 255, 255)
    expected_shape: str = ''


@dataclass
class DetectionParams:
    blur_kernel: int = 5          # Gaussian blur before HSV (odd, 0/1 = off)
    morph_kernel: int = 5         # open/close kernel size in px (0 = off)
    min_area_px: float = 300.0    # reject specks
    max_area_px: float = 60000.0  # reject huge regions (e.g. background)
    min_fill_ratio: float = 0.6   # mask pixels / contour area; rejects rings/outlines
    max_per_colour: int = 1       # one peg per colour in this task
    overlap_iou: float = 0.5      # cross-colour duplicate suppression threshold


@dataclass
class Blob:
    colour: str
    contour: np.ndarray
    mask: np.ndarray              # full-frame uint8, 255 inside this blob
    centroid: tuple               # (u, v) float pixels
    area_px: float
    fill_ratio: float
    bbox: tuple                   # (x, y, w, h)
    extra: dict = field(default_factory=dict)


def build_mask(hsv: np.ndarray, spec: ColourSpec, morph_kernel: int) -> np.ndarray:
    mask = cv2.inRange(hsv, spec.hsv_lower, spec.hsv_upper)
    if morph_kernel and morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)    # remove specks
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)   # fill small holes
    return mask


def find_blobs(mask: np.ndarray, colour: str, p: DetectionParams) -> List[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Blob] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < p.min_area_px or area > p.max_area_px:
            continue
        blob_mask = np.zeros_like(mask)
        cv2.drawContours(blob_mask, [c], -1, 255, thickness=cv2.FILLED)
        filled = cv2.countNonZero(cv2.bitwise_and(blob_mask, mask))
        fill_ratio = filled / max(area, 1.0)
        if fill_ratio < p.min_fill_ratio:
            continue          # e.g. the red socket outlines: big contour, thin line
        m = cv2.moments(c)
        if m['m00'] == 0:
            continue
        u, v = m['m10'] / m['m00'], m['m01'] / m['m00']
        blobs.append(Blob(colour, c, blob_mask, (u, v), area, fill_ratio,
                          cv2.boundingRect(c)))
    blobs.sort(key=lambda b: b.area_px, reverse=True)
    return blobs[: max(int(p.max_per_colour), 0)]


def _bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _mask_overlap(a: Blob, b: Blob) -> float:
    """Shared pixels / pixels of the smaller blob (0 = disjoint, 1 = one inside the other).

    Uses the real filled shapes, not bounding boxes: two pegs lying diagonally
    next to each other have heavily overlapping boxes but no shared pixels.
    """
    if _bbox_iou(a.bbox, b.bbox) == 0.0:
        return 0.0                                   # cheap early exit
    inter = cv2.countNonZero(cv2.bitwise_and(a.mask, b.mask))
    smaller = min(cv2.countNonZero(a.mask), cv2.countNonZero(b.mask))
    return inter / smaller if smaller else 0.0


def suppress_cross_colour_duplicates(blobs: List[Blob], overlap: float) -> List[Blob]:
    """If two colours claim the same pixels (e.g. orange/yellow edge), keep the larger blob."""
    kept: List[Blob] = []
    for b in sorted(blobs, key=lambda x: x.area_px, reverse=True):
        if all(_mask_overlap(b, k) < overlap for k in kept):
            kept.append(b)
    return kept


def detect_colours(bgr: np.ndarray, specs: List[ColourSpec], p: DetectionParams):
    """Return (blobs_by_colour, masks_by_colour)."""
    img = bgr
    if p.blur_kernel and p.blur_kernel > 1:
        k = p.blur_kernel if p.blur_kernel % 2 == 1 else p.blur_kernel + 1
        img = cv2.GaussianBlur(bgr, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks: Dict[str, np.ndarray] = {}
    all_blobs: List[Blob] = []
    for spec in specs:
        mask = build_mask(hsv, spec, p.morph_kernel)
        masks[spec.name] = mask
        all_blobs.extend(find_blobs(mask, spec.name, p))

    kept = suppress_cross_colour_duplicates(all_blobs, p.overlap_iou)
    by_colour: Dict[str, List[Blob]] = {s.name: [] for s in specs}
    for b in kept:
        by_colour[b.colour].append(b)
    return by_colour, masks


def hsv_at(bgr: np.ndarray, u: int, v: int, half: int = 2) -> Optional[tuple]:
    """Median HSV in a small window — handy for tuning thresholds."""
    h, w = bgr.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return None
    patch = bgr[max(0, v - half): v + half + 1, max(0, u - half): u + half + 1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return tuple(int(x) for x in np.median(hsv, axis=0))
