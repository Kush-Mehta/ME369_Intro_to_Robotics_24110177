"""Socket detection: red outlines on the shape board.

Robust to imperfect outlines (gaps, glare, rounded corners):

1. Red mask (two hue ranges because red wraps around H=0).
2. Group nearby red pixels (small gaps joined) -> convex hull of each group.
3. Accept a hull as a socket when its OUTLINE is mostly covered by red
   (coverage) and its INTERIOR is mostly not red. This rejects solid blobs
   (e.g. the orange peg's top) and lone arcs, but tolerates small breaks.
4. Circle fallback: a badly broken circle is found with a Hough circle fit.
5. Shape = nearest template using fill ratios against the minimum enclosing
   circle / rectangle / triangle, which survive rounded corners and the
   camera tilt far better than counting polygon vertices.
6. Each shape is used once: best one-to-one assignment over all sockets.

Position = hull centroid (the area centroid, which is where the peg's axis
goes - for a triangle this is 1/3 of the height from the flat side, not the
midpoint). Depth = median over the ring + board band around it, because the
hole itself shows the foam below the board.
"""
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

# Ideal features per shape:
#   area / fitted ellipse, area / min area rect, area / min enclosing triangle,
#   ellipse-fit residual (mean |r-1| of the outline vs its best-fit ellipse).
# All four barely change under camera tilt (a tilted circle is still a perfect
# ellipse); the residual is what separates circle (0.00) from hexagon (0.039).
TEMPLATES = {
    'circle':   (1.000, 0.785, 0.607, 0.003),
    'hexagon':  (0.984, 0.752, 0.667, 0.039),
    'square':   (0.909, 1.000, 0.500, 0.098),
    'triangle': (0.689, 0.500, 1.000, 0.199),
}
FEATURE_WEIGHTS = np.array([1.0, 1.0, 1.0, 3.0])


def _ellipse_residual(outline_pts: np.ndarray) -> float:
    pts = outline_pts.reshape(-1, 2).astype(np.float32)
    if len(pts) < 5:
        return 1.0
    (cx, cy), (ax1, ax2), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2))
    if ax1 <= 0 or ax2 <= 0:
        return 1.0
    a = math.radians(ang)
    c, s = math.cos(a), math.sin(a)
    d = pts - np.array([cx, cy], dtype=np.float32)
    x = d[:, 0] * c + d[:, 1] * s
    y = -d[:, 0] * s + d[:, 1] * c
    r = np.sqrt((x / (ax1 / 2)) ** 2 + (y / (ax2 / 2)) ** 2)
    return float(np.mean(np.abs(r - 1.0)))


@dataclass
class SocketParams:
    hsv_ranges: Sequence[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    blur_kernel: int = 3
    morph_kernel: int = 3            # CLOSE only: thin rings must not be eroded away
    group_px: int = 6                # red pieces closer than ~2x this are one group
    min_hole_area_px: float = 400.0  # hull area limits
    max_hole_area_px: float = 40000.0
    min_coverage: float = 0.75       # fraction of hull outline that has red nearby
    coverage_tol_px: float = 4.0
    max_interior_red: float = 0.10   # fraction of the hull's inside that is red
    surface_dilate_px: int = 6       # board pixels around the ring used for depth
    circle_fallback: bool = True
    circle_min_radius_px: int = 15
    circle_max_radius_px: int = 120
    circle_min_coverage: float = 0.45
    max_sockets: int = 4


@dataclass
class Socket:
    shape: str
    confidence: str
    features: dict
    hole_contour: np.ndarray         # hull of the ring (drawn on the image)
    outer_contour: np.ndarray
    centroid: tuple                  # (u, v)
    hole_area_px: float
    surface_mask: np.ndarray         # uint8 full-frame


def red_mask(bgr: np.ndarray, p: SocketParams) -> np.ndarray:
    img = bgr
    if p.blur_kernel and p.blur_kernel > 1:
        k = p.blur_kernel if p.blur_kernel % 2 == 1 else p.blur_kernel + 1
        img = cv2.GaussianBlur(bgr, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p.hsv_ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    if p.morph_kernel and p.morph_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.morph_kernel, p.morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _ellipse(r: int):
    r = max(int(r), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _template_costs(contour) -> Dict[str, float]:
    pts = contour.reshape(-1, 1, 2).astype(np.float32)
    area = cv2.contourArea(pts)
    dense = _dense_outline(contour)
    (_, _), (ea, eb), _ = cv2.fitEllipse(dense.reshape(-1, 1, 2).astype(np.float32))
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    tri_area, _ = cv2.minEnclosingTriangle(pts)
    f = np.array([area / (math.pi * ea * eb / 4.0) if ea * eb > 0 else 0.0,
                  area / (w * h) if w * h > 0 else 0.0,
                  area / tri_area if tri_area > 0 else 0.0,
                  _ellipse_residual(dense)])
    return {name: float(np.linalg.norm(FEATURE_WEIGHTS * (f - np.array(t))))
            for name, t in TEMPLATES.items()}, f


def _outline_coverage(outline_pts: np.ndarray, dist_to_red: np.ndarray, tol: float) -> float:
    h, w = dist_to_red.shape
    xs = np.clip(outline_pts[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(outline_pts[:, 1].round().astype(int), 0, h - 1)
    return float(np.mean(dist_to_red[ys, xs] <= tol)) if len(xs) else 0.0


def _dense_outline(contour, n: int = 360) -> np.ndarray:
    """Evenly resample a closed contour to n points."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return pts[:1]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


def _make_socket(hull, mask, p, coverage, source) -> dict:
    hull_mask = np.zeros_like(mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
    area = float(cv2.countNonZero(hull_mask))
    m = cv2.moments(hull_mask, binaryImage=True)
    u, v = m['m10'] / m['m00'], m['m01'] / m['m00']
    perimeter = cv2.arcLength(hull, True)
    ring_px = cv2.countNonZero(cv2.bitwise_and(mask, hull_mask))
    thickness = max(2, int(round(ring_px / max(perimeter, 1.0))))
    surface = cv2.dilate(hull_mask, _ellipse(p.surface_dilate_px))
    inner = cv2.erode(hull_mask, _ellipse(thickness + 2))
    surface[inner > 0] = 0                       # board top + ring, not the foam in the hole
    costs, f = _template_costs(hull)
    return {'hull': hull, 'centroid': (u, v), 'area': area, 'surface': surface,
            'costs': costs, 'fills': f, 'coverage': coverage, 'source': source,
            'thickness': thickness}


def _ring_candidates(mask: np.ndarray, p: SocketParams) -> List[dict]:
    dist_to_red = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    grouped = cv2.dilate(mask, _ellipse(p.group_px))
    n, labels = cv2.connectedComponents(grouped)
    out = []
    for lab in range(1, n):
        comp = np.where((labels == lab) & (mask > 0), 255, 0).astype(np.uint8)
        pts = cv2.findNonZero(comp)
        if pts is None or len(pts) < 30:
            continue
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        if not (p.min_hole_area_px <= hull_area <= p.max_hole_area_px):
            continue
        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        e = max(6, int(0.15 * math.sqrt(hull_area)))
        inner = cv2.erode(hull_mask, _ellipse(e))
        n_inner = cv2.countNonZero(inner)
        if n_inner == 0:
            continue
        interior_red = cv2.countNonZero(cv2.bitwise_and(comp, inner)) / n_inner
        if interior_red > p.max_interior_red:
            continue                              # solid blob (e.g. orange peg top)
        cov = _outline_coverage(_dense_outline(hull), dist_to_red, p.coverage_tol_px)
        if cov < p.min_coverage:
            continue                              # lone arc / noise, not a closed-ish outline
        out.append(_make_socket(hull, mask, p, cov, 'ring'))
    return out


def _circle_fallback(mask: np.ndarray, p: SocketParams, taken: List[dict]) -> List[dict]:
    dist_to_red = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 3)
    blur = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=50,
                               param2=12, minRadius=p.circle_min_radius_px,
                               maxRadius=p.circle_max_radius_px)
    if circles is None:
        return []
    best = None
    for x, y, r in circles[0]:
        if any(cv2.pointPolygonTest(t['hull'], (float(x), float(y)), False) >= 0 for t in taken):
            continue                              # already explained by another socket
        ring = np.column_stack([x + r * np.cos(np.linspace(0, 2 * np.pi, 360, endpoint=False)),
                                y + r * np.sin(np.linspace(0, 2 * np.pi, 360, endpoint=False))])
        cov = _outline_coverage(ring, dist_to_red, p.coverage_tol_px)
        inner = np.zeros_like(mask)
        cv2.circle(inner, (int(round(x)), int(round(y))), int(0.7 * r), 255, -1)
        n_inner = cv2.countNonZero(inner)
        interior_red = cv2.countNonZero(cv2.bitwise_and(mask, inner)) / max(n_inner, 1)
        if cov >= p.circle_min_coverage and interior_red <= p.max_interior_red:
            if best is None or cov > best[3]:
                best = (x, y, r, cov)
    if best is None:
        return []
    x, y, r, cov = best
    hull = cv2.ellipse2Poly((int(round(x)), int(round(y))), (int(round(r)), int(round(r))),
                            0, 0, 360, 5).reshape(-1, 1, 2)
    c = _make_socket(hull, mask, p, cov, 'hough-circle')
    c['costs'] = {k: (0.0 if k == 'circle' else 1.0) for k in TEMPLATES}
    return [c]


def _assign_shapes(cands: List[dict]) -> List[Tuple[dict, str, float]]:
    """Best one-to-one shape assignment (each shape used at most once)."""
    names = list(TEMPLATES)
    best, best_cost = None, float('inf')
    k = min(len(cands), len(names))
    for perm in itertools.permutations(names, k):
        cost = sum(c['costs'][s] for c, s in zip(cands[:k], perm))
        if cost < best_cost:
            best, best_cost = perm, cost
    return [(c, s, c['costs'][s]) for c, s in zip(cands[:k], best or ())]


def detect_sockets(bgr: np.ndarray, p: SocketParams):
    """Return (sockets sorted top-to-bottom then left-to-right, red mask)."""
    mask = red_mask(bgr, p)
    cands = _ring_candidates(mask, p)
    # Too many: keep the best-fitting (lowest min template cost, then highest coverage).
    cands.sort(key=lambda c: (min(c['costs'].values()), -c['coverage']))
    cands = cands[: p.max_sockets]
    if p.circle_fallback and len(cands) < p.max_sockets:
        cands += _circle_fallback(mask, p, cands)   # only searches outside known sockets

    sockets: List[Socket] = []
    for c, shape, cost in _assign_shapes(cands):
        others = sorted(v for s, v in c['costs'].items() if s != shape)
        margin = (others[0] - cost) if others else 1.0
        # High = clearly the best template AND a reasonable fit in absolute terms.
        conf = 'high' if (margin >= 0.03 and cost <= 0.25) else 'low'
        feats = {'fit_cost': round(cost, 3), 'margin': round(margin, 3),
                 'coverage': round(c['coverage'], 2), 'source': c['source'],
                 'fills(circ,rect,tri,ellres)': tuple(round(float(x), 3) for x in c['fills'])}
        sockets.append(Socket(shape, conf, feats, c['hull'], c['hull'], c['centroid'],
                              c['area'], c['surface']))
    sockets.sort(key=lambda s: (round(s.centroid[1] / 40), s.centroid[0]))
    return sockets, mask
