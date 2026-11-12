"""Stages B and C: robust depth lookup and pinhole back-projection.

Pinhole model (camera optical frame: +X right, +Y down, +Z out of the lens):
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    Z = depth
In words: (u - cx) is how many pixels the point is right of the image centre.
Dividing by fx turns pixels into "metres per metre of distance"; multiplying
by Z gives metres. Same for Y with rows.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    frame_id: str

    @classmethod
    def from_camera_info(cls, msg):
        k = msg.k
        return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]),
                   int(msg.width), int(msg.height), msg.header.frame_id)

    def as_matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


@dataclass
class DepthResult:
    valid: bool
    z_m: float = float('nan')
    iqr_m: float = float('nan')     # spread of depths used; large = unreliable
    n_valid: int = 0
    n_total: int = 0
    reason: str = ''

    @property
    def valid_fraction(self) -> float:
        return self.n_valid / self.n_total if self.n_total else 0.0


@dataclass
class DepthParams:
    min_m: float = 0.15       # D435 cannot measure closer than ~0.2-0.3 m reliably
    max_m: float = 2.0        # ignore far background
    erode_px: int = 3         # shrink region away from edges (edges = mixed depth)
    min_valid_px: int = 30    # need at least this many good pixels
    min_valid_fraction: float = 0.2


def robust_depth(depth_m: np.ndarray, region_mask: np.ndarray, p: DepthParams) -> DepthResult:
    """Median of valid depths inside the object's (eroded) region."""
    region = region_mask
    if p.erode_px and p.erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.erode_px + 1, 2 * p.erode_px + 1))
        eroded = cv2.erode(region_mask, k)
        if cv2.countNonZero(eroded) >= p.min_valid_px:
            region = eroded            # fall back to unshrunk region for tiny blobs

    vals = depth_m[region > 0]
    n_total = int(vals.size)
    good = vals[np.isfinite(vals) & (vals > p.min_m) & (vals < p.max_m)]
    n_valid = int(good.size)

    if n_total == 0:
        return DepthResult(False, reason='empty region')
    if n_valid < p.min_valid_px or n_valid / n_total < p.min_valid_fraction:
        return DepthResult(False, n_valid=n_valid, n_total=n_total,
                           reason=f'only {n_valid}/{n_total} valid depth px')
    q25, z, q75 = np.percentile(good, [25, 50, 75])
    return DepthResult(True, float(z), float(q75 - q25), n_valid, n_total)


def deproject(u: float, v: float, z: float, intr: Intrinsics) -> np.ndarray:
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    return np.array([x, y, z], dtype=np.float64)


def depth_at_patch(depth_m: np.ndarray, u: float, v: float, half: int,
                   p: DepthParams) -> Optional[float]:
    """Median depth in a (2*half+1)^2 window — used for the marker centre check."""
    h, w = depth_m.shape
    ui, vi = int(round(u)), int(round(v))
    patch = depth_m[max(0, vi - half): vi + half + 1, max(0, ui - half): ui + half + 1]
    good = patch[np.isfinite(patch) & (patch > p.min_m) & (patch < p.max_m)]
    return float(np.median(good)) if good.size >= 5 else None
