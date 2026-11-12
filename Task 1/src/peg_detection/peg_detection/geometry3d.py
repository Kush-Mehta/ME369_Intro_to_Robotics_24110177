"""3D geometry for the peg pipeline: camera -> robot transform, peg top-face centre,
socket centre on the board plane, marker-pose averaging and the frame-calibration fit.

Why this module exists
----------------------
The camera looks at the scene from the side. A peg's colour blob is its top face PLUS
the side faces the camera can see, so the 2D centre of the blob is NOT over the peg's
axis, and the median depth of the blob is neither the top nor the base. The old code
back-projected that one pixel -> a point shifted toward the camera and partway down the
peg. Flat things (sockets, the marker) did not have this problem.

Fix: do the geometry in 3D, in the robot frame (z up):
  peg    : back-project EVERY blob pixel with its own depth, keep the points within a few
           mm of the highest one (= the top face, whatever the camera angle), and take the
           area centroid of their x-y outline. -> centre of the top face, z = top height.
  socket : fit the board plane from the board pixels around the ring, intersect the camera
           rays of the socket outline with that plane, and take the polygon's area centroid
           on the plane. (Averaging pixels first is slightly wrong under perspective.)

Frames
------
camera optical frame (c): +X right, +Y down, +Z out of the lens (RealSense colour optical).
marker frame (m)        : from the ArUco pose, p_c = R_cm p_m + t_cm.
drawn frame (d)         : the hand-drawn axes on the foam: p_d = Rz(marker_yaw) p_m + marker_pos.
robot frame (r)         : ROBOTIS `link1` (what the IK / arm uses): p_r = Rz(robot_yaw) p_d + robot_off.
Everything is collapsed into one rigid transform  p_r = A @ p_c + b.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import math

import numpy as np


# --------------------------------------------------------------------- transforms
def rot_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class RigidTransform:
    """p_out = A @ p_in + b  (A is a proper rotation)."""
    A: np.ndarray
    b: np.ndarray

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return p @ self.A.T + self.b

    def inverse_apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        return (p - self.b) @ self.A

    @property
    def origin(self) -> np.ndarray:
        """Camera centre expressed in the output frame."""
        return self.b.copy()


def camera_to_robot(R_cm: np.ndarray, t_cm: np.ndarray, marker_yaw_rad: float,
                    marker_pos: Sequence[float], robot_yaw_rad: float,
                    robot_offset: Sequence[float]) -> RigidTransform:
    """Compose camera -> marker -> drawn axes -> robot (link1) into one transform."""
    Rm, Rr = rot_z(marker_yaw_rad), rot_z(robot_yaw_rad)
    A = Rr @ Rm @ R_cm.T
    b = Rr @ (np.asarray(marker_pos, float) - Rm @ R_cm.T @ np.asarray(t_cm, float)) \
        + np.asarray(robot_offset, float)
    return RigidTransform(A, b)


def average_poses(Rs: List[np.ndarray], ts: List[np.ndarray],
                  outlier_m: float = 0.005) -> Tuple[np.ndarray, np.ndarray, int]:
    """Average marker poses (camera is fixed): drop translation outliers, mean t,
    and the rotation closest (Frobenius) to the mean rotation matrix."""
    T = np.array(ts, dtype=np.float64)
    med = np.median(T, axis=0)
    keep = np.linalg.norm(T - med, axis=1) <= outlier_m
    if keep.sum() < max(3, len(ts) // 3):
        keep = np.ones(len(ts), bool)
    M = np.sum([R for R, k in zip(Rs, keep) if k], axis=0)
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
    return R, T[keep].mean(axis=0), int(keep.sum())


@dataclass
class MarkerRefinement:
    ok: bool
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    normal_change_deg: float = float('nan')   # PnP normal vs depth-plane normal
    side_m: float = float('nan')              # marker side measured on the depth plane
    plane_rms_mm: float = float('nan')
    n_px: int = 0
    reason: str = ''


def _fit_plane_svd(P: np.ndarray):
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    return n, float(n @ c)


def refine_marker_with_depth(R: np.ndarray, t: np.ndarray, corners: np.ndarray,
                             depth_m: np.ndarray, intr, radius_factor: float = 2.5,
                             band_m: float = 0.015, max_normal_change_deg: float = 10.0,
                             min_m: float = 0.15, max_m: float = 2.0) -> MarkerRefinement:
    """Make the marker frame consistent with the depth data.

    A small marker seen obliquely gives a noisy PnP pose (mostly its distance and tilt).
    The table around the marker is flat, so fit that plane from thousands of depth
    pixels, intersect the 4 corner rays with it, and rebuild the frame from the 3D corners.
    The PnP pose is only used to pick the right pixels and check the result.
    """
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    cu, cv_ = corners.mean(axis=0)
    side_px = np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))
    r = radius_factor * side_px
    h_, w_ = depth_m.shape
    v0, v1 = int(max(0, cv_ - r)), int(min(h_, cv_ + r + 1))
    u0, u1 = int(max(0, cu - r)), int(min(w_, cu + r + 1))
    vv, uu = np.mgrid[v0:v1, u0:u1]
    sel = (uu - cu) ** 2 + (vv - cv_) ** 2 <= r * r
    us, vs = uu[sel].astype(np.float64), vv[sel].astype(np.float64)
    z = depth_m[vv[sel], uu[sel]].astype(np.float64)
    ok = np.isfinite(z) & (z > min_m) & (z < max_m)
    P = pixel_rays(us[ok], vs[ok], intr) * z[ok][:, None]
    n0 = R[:, 2]
    d0 = float(n0 @ t)
    P = P[np.abs(P @ n0 - d0) < band_m]           # near the PnP plane (drops robot base etc.)
    if len(P) < 200:
        return MarkerRefinement(False, n_px=len(P), reason=f'only {len(P)} plane px')
    keep = np.ones(len(P), bool)
    for _ in range(3):
        n, d = _fit_plane_svd(P[keep])
        res = P @ n - d
        mad = np.median(np.abs(res[keep])) * 1.4826 + 2e-4
        keep = np.abs(res) < 3.0 * mad
    n, d = _fit_plane_svd(P[keep])
    if n @ n0 < 0:
        n, d = -n, -d
    change = float(np.degrees(np.arccos(np.clip(n @ n0, -1.0, 1.0))))
    if change > max_normal_change_deg:
        return MarkerRefinement(False, normal_change_deg=change, n_px=int(keep.sum()),
                                reason=f'depth plane {change:.1f} deg from marker normal')
    rays = pixel_rays(corners[:, 0], corners[:, 1], intr)
    C3 = rays * (d / (rays @ n))[:, None]          # TL, TR, BR, BL on the plane
    x = (C3[1] - C3[0]) + (C3[2] - C3[3])
    x -= (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Rr = np.column_stack([x, y, n])
    side = float(np.mean(np.linalg.norm(C3 - np.roll(C3, 1, axis=0), axis=1)))
    rms = float(np.sqrt(np.mean((P[keep] @ n - d) ** 2)) * 1000.0)
    return MarkerRefinement(True, Rr, C3.mean(axis=0), change, side, rms, int(keep.sum()))


def _frame_from_plane(n, d, corners, intr):
    rays = pixel_rays(corners[:, 0], corners[:, 1], intr)
    C3 = rays * (d / (rays @ n))[:, None]          # TL, TR, BR, BL on the plane
    x = (C3[1] - C3[0]) + (C3[2] - C3[3])
    x -= (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    side = float(np.mean(np.linalg.norm(C3 - np.roll(C3, 1, axis=0), axis=1)))
    return np.column_stack([x, y, n]), C3.mean(axis=0), side


def refine_marker_with_table(R: np.ndarray, t: np.ndarray, corners: np.ndarray,
                             depth_m: np.ndarray, intr, stride: int = 4, thresh_m: float = 0.006,
                             max_tilt_deg: float = 35.0, iters: int = 400,
                             min_m: float = 0.15, max_m: float = 2.0,
                             seed: int = 0) -> MarkerRefinement:
    """Like refine_marker_with_depth, but the plane is the WHOLE table/mat seen by the camera
    (RANSAC over the full depth image), not a small patch around the marker. A tiny marker's
    PnP tilt can be 10-15 deg off; any tilt makes heights grow with distance from the marker.
    The table plane from ~10 000 depth pixels fixes the tilt; the marker only gives the origin
    (its centre ray hits the plane) and the x direction (its corner rays)."""
    corners = np.asarray(corners, np.float64).reshape(4, 2)
    h_, w_ = depth_m.shape
    vv, uu = np.mgrid[0:h_:stride, 0:w_:stride]
    z = depth_m[vv, uu].astype(np.float64).ravel()
    us, vs = uu.ravel().astype(np.float64), vv.ravel().astype(np.float64)
    ok = np.isfinite(z) & (z > min_m) & (z < max_m)
    P = pixel_rays(us[ok], vs[ok], intr) * z[ok][:, None]
    if len(P) < 500:
        return MarkerRefinement(False, n_px=len(P), reason=f'only {len(P)} depth px')
    n0 = R[:, 2] / np.linalg.norm(R[:, 2])
    cos_max = math.cos(math.radians(max_tilt_deg))
    # the plane must also pass near the marker centre (it is lying on the table)
    cray = pixel_rays(np.array([corners[:, 0].mean()]), np.array([corners[:, 1].mean()]), intr)[0]
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ n0 < 0:
            n = -n
        if n @ n0 < cos_max:
            continue
        d = float(n @ a)
        cn = float(cray @ n)
        if abs(cn) < 1e-6 or not (min_m < d / cn < max_m * 1.5):   # marker ray must hit it ahead
            continue
        inl = np.abs(P @ n - d) < thresh_m
        k = int(inl.sum())
        if k > best_n:
            best, best_n = inl, k
    if best is None or best_n < 300:
        return MarkerRefinement(False, n_px=best_n, reason='no table plane found')
    keep = best
    for _ in range(3):
        n, d = _fit_plane_svd(P[keep])
        if n @ n0 < 0:
            n, d = -n, -d
        res = P @ n - d
        mad = np.median(np.abs(res[keep])) * 1.4826 + 2e-4
        keep = np.abs(res) < max(3.0 * mad, 0.003)
    n, d = _fit_plane_svd(P[keep])
    if n @ n0 < 0:
        n, d = -n, -d
    change = float(np.degrees(np.arccos(np.clip(n @ n0, -1.0, 1.0))))
    Rr, tr, side = _frame_from_plane(n, d, corners, intr)
    rms = float(np.sqrt(np.mean((P[keep] @ n - d) ** 2)) * 1000.0)
    return MarkerRefinement(True, Rr, tr, change, side, rms, int(keep.sum()))


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ------------------------------------------------------------------ pixel <-> 3D
def pixel_rays(us: np.ndarray, vs: np.ndarray, intr) -> np.ndarray:
    """Camera-frame ray directions with z = 1 (so a point is depth * ray)."""
    return np.column_stack([(us - intr.cx) / intr.fx, (vs - intr.cy) / intr.fy,
                            np.ones_like(us, dtype=np.float64)])


def project(p_cam: np.ndarray, intr) -> Tuple[float, float]:
    x, y, z = p_cam
    return float(intr.fx * x / z + intr.cx), float(intr.fy * y / z + intr.cy)


def polygon_centroid_xy(xy: np.ndarray) -> Tuple[np.ndarray, float]:
    """Area centroid of a closed polygon (vertices in order). Returns (centroid, area)."""
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return xy.mean(axis=0), 0.0
    cx = ((x + x1) * cross).sum() / (6.0 * a)
    cy = ((y + y1) * cross).sum() / (6.0 * a)
    return np.array([cx, cy]), abs(a)


def _region_points(depth_m, mask, erode_px, min_m, max_m):
    region = mask
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        er = cv2.erode(mask, k)
        if cv2.countNonZero(er) >= 30:
            region = er
    vs, us = np.nonzero(region)
    z = depth_m[vs, us].astype(np.float64)
    ok = np.isfinite(z) & (z > min_m) & (z < max_m)
    return us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok], int(len(z))


# ------------------------------------------------------------------- peg top face
@dataclass
class PegTopParams:
    erode_px: int = 2
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    top_percentile: float = 95.0   # robust "highest point" (ignores a few spikes)
    top_band_m: float = 0.006      # points within this of the top = top face
    min_top_px: int = 25
    min_valid_px: int = 40


@dataclass
class PegTop:
    valid: bool
    center: Optional[np.ndarray] = None   # robot frame [x, y, z_top]
    uv: Optional[Tuple[float, float]] = None   # top-face centre in the image
    top_outline_xy: Optional[np.ndarray] = None
    n_valid: int = 0
    n_top: int = 0
    top_spread_mm: float = float('nan')   # z spread of the top points (flatness check)
    reason: str = ''


def peg_top_center(depth_m: np.ndarray, mask: np.ndarray, intr, T: RigidTransform,
                   p: PegTopParams) -> PegTop:
    us, vs, z, n_all = _region_points(depth_m, mask, p.erode_px, p.min_depth_m, p.max_depth_m)
    if len(z) < p.min_valid_px:
        return PegTop(False, n_valid=len(z), reason=f'only {len(z)}/{n_all} valid depth px')
    P = T.apply(pixel_rays(us, vs, intr) * z[:, None])          # robot frame
    z_top = np.percentile(P[:, 2], p.top_percentile)
    top = P[P[:, 2] >= z_top - p.top_band_m]
    top = top[top[:, 2] <= z_top + p.top_band_m]                # drop spikes above
    if len(top) < p.min_top_px:
        return PegTop(False, n_valid=len(z), n_top=len(top),
                      reason=f'top face too small ({len(top)} px)')
    xy = top[:, :2]
    med = np.median(xy, axis=0)
    d = np.linalg.norm(xy - med, axis=1)
    xy = xy[d <= max(2.5 * np.median(d), 0.004)]                # flying-pixel outliers
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) >= 3:
        c_xy, area = polygon_centroid_xy(hull)
        if area < 1e-6:
            c_xy = np.median(xy, axis=0)
    else:
        c_xy = np.median(xy, axis=0)
    z_c = float(np.median(top[:, 2]))
    center = np.array([c_xy[0], c_xy[1], z_c])
    uv = project(T.inverse_apply(center), intr)
    return PegTop(True, center, uv, hull, len(z), len(top),
                  float(np.ptp(top[:, 2]) * 1000.0))


def silhouette_point(depth_m, mask, centroid_uv, intr, T: Optional[RigidTransform],
                     erode_px=3, min_m=0.15, max_m=2.0):
    """The OLD method (blob centroid + median depth), kept only for comparison."""
    _, _, z, _ = _region_points(depth_m, mask, erode_px, min_m, max_m)
    if len(z) < 10:
        return None
    u, v = centroid_uv
    p = pixel_rays(np.array([u]), np.array([v]), intr)[0] * np.median(z)
    return T.apply(p) if T is not None else p


# ------------------------------------------------------------- socket on the board
@dataclass
class SocketCenter:
    valid: bool
    center: Optional[np.ndarray] = None    # robot frame, on the board surface
    plane: Optional[np.ndarray] = None     # z = a x + b y + c  -> (a, b, c)
    tilt_deg: float = float('nan')
    n_surface: int = 0
    reason: str = ''
    outline_xy: Optional[np.ndarray] = None   # hole outline on the board plane (robot xy)


def fit_plane_z(P: np.ndarray, iters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Least-squares plane z = a x + b y + c with MAD outlier rejection."""
    keep = np.ones(len(P), bool)
    coef = np.array([0.0, 0.0, float(np.median(P[:, 2]))])
    for _ in range(iters + 1):
        Q = P[keep]
        if len(Q) < 10:
            break
        M = np.column_stack([Q[:, 0], Q[:, 1], np.ones(len(Q))])
        coef, *_ = np.linalg.lstsq(M, Q[:, 2], rcond=None)
        r = P[:, 2] - (coef[0] * P[:, 0] + coef[1] * P[:, 1] + coef[2])
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-4
        keep = np.abs(r) < 3.0 * 1.4826 * mad + 0.001
    return coef, keep


def socket_center_on_plane(depth_m, surface_mask, outline_px: np.ndarray, intr,
                           T: RigidTransform, min_m=0.15, max_m=2.0,
                           min_surface_px=30) -> SocketCenter:
    us, vs, z, _ = _region_points(depth_m, surface_mask, 0, min_m, max_m)
    if len(z) < min_surface_px:
        return SocketCenter(False, n_surface=len(z), reason=f'only {len(z)} board px')
    P = T.apply(pixel_rays(us, vs, intr) * z[:, None])
    (a, b_, c), _ = fit_plane_z(P)
    tilt = float(np.degrees(np.arctan(np.hypot(a, b_))))
    if tilt > 15.0:          # a local fit this steep is noise: use a level plane
        a = b_ = 0.0
        c = float(np.median(P[:, 2]))
    # Intersect the outline's camera rays with the plane (robot frame).
    rays = pixel_rays(outline_px[:, 0].astype(np.float64), outline_px[:, 1].astype(np.float64),
                      intr) @ T.A.T
    o = T.origin
    denom = rays[:, 2] - a * rays[:, 0] - b_ * rays[:, 1]
    s = (c + a * o[0] + b_ * o[1] - o[2]) / np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    Q = o + rays * s[:, None]
    c_xy, area = polygon_centroid_xy(Q[:, :2])
    if area <= 0:
        return SocketCenter(False, n_surface=len(z), reason='degenerate outline')
    zc = a * c_xy[0] + b_ * c_xy[1] + c
    return SocketCenter(True, np.array([c_xy[0], c_xy[1], zc]), np.array([a, b_, c]),
                        tilt, len(z), outline_xy=Q[:, :2])


def dense_outline(contour, n: int = 180) -> np.ndarray:
    """Evenly resample a closed pixel contour to n points (keeps the vertex order)."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return pts[:1]
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


# ----------------------------------------------------------- temporal smoothing
@dataclass
class MedianFilter:
    """Median of the last n positions; restarts when the object jumps (moved)."""
    n: int = 7
    jump_m: float = 0.02
    buf: list = field(default_factory=list)

    def update(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)
        if self.buf and np.linalg.norm(p - np.median(self.buf, axis=0)) > self.jump_m:
            self.buf = []
        self.buf = (self.buf + [p])[-self.n:]
        return np.median(self.buf, axis=0)


# ------------------------------------------------------ frame calibration (yaw + xyz)
def fit_yaw_translation(p: np.ndarray, q: np.ndarray):
    """Find yaw (about z) and translation d with  q ~= Rz(yaw) p + d.
    p: vision points, q: robot (FK) points, both (N,3). N = 1 -> translation only."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    pc, qc = p.mean(axis=0), q.mean(axis=0)
    if len(p) >= 2:
        P, Q = p[:, :2] - pc[:2], q[:, :2] - qc[:2]
        yaw = float(np.arctan2(np.sum(P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]),
                               np.sum(P[:, 0] * Q[:, 0] + P[:, 1] * Q[:, 1])))
    else:
        yaw = 0.0
    R = rot_z(yaw)
    d = qc - R @ pc
    resid = q - (p @ R.T + d)
    return yaw, d, resid


# ------------------------------------------------------------------- orientation
SYMMETRY = {'square': 4, 'hexagon': 6, 'triangle': 3}


def polygon_yaw_deg(xy: np.ndarray, n: int, samples: int = 360) -> Tuple[float, float]:
    """Direction of a VERTEX of an n-fold symmetric outline (robot xy, any order of points
    along the boundary), in degrees in [0, 360/n). Returns (yaw_deg, strength 0..1).
    Uses the n-th angular harmonic of the boundary radius: vertices = radius maxima."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return float('nan'), 0.0
    hull = cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    if len(hull) < 3:
        return float('nan'), 0.0
    c, _ = polygon_centroid_xy(hull)
    seg = np.roll(hull, -1, axis=0) - hull
    L = np.linalg.norm(seg, axis=1)
    t = np.linspace(0, L.sum(), samples, endpoint=False)
    cum = np.concatenate([[0], np.cumsum(L)])
    k = np.clip(np.searchsorted(cum, t, side='right') - 1, 0, len(hull) - 1)
    f = (t - cum[k]) / np.where(L[k] > 0, L[k], 1)
    pts = hull[k] + seg[k] * f[:, None] - c
    r = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    m = np.sum((r - r.mean()) * np.exp(1j * n * ang))
    strength = float(abs(m) / (np.sum(np.abs(r - r.mean())) + 1e-12))
    yaw = (np.degrees(np.angle(m)) / n) % (360.0 / n)
    return float(yaw), strength


def circle_center_xy(xy: np.ndarray, iters: int = 5) -> Optional[np.ndarray]:
    """Centre of a round outline (robot xy): algebraic circle fit, then drop the points
    furthest from the circle (ragged depth edges) and refit."""
    P = np.asarray(xy, float)
    if len(P) < 6:
        return None
    keep = np.ones(len(P), bool)
    c = None
    for _ in range(iters):
        Q = P[keep]
        A = np.column_stack([2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))])
        b = (Q ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = sol[:2]
        r = np.sqrt(max(sol[2] + c @ c, 1e-12))
        res = np.abs(np.linalg.norm(P - c, axis=1) - r)
        lim = max(np.percentile(res[keep], 75) * 2.0, 0.001)
        new = res < lim
        if new.sum() < 6 or (new == keep).all():
            break
        keep = new
    return c


APOTHEM_PER_SIDE = {4: 0.5, 6: math.sqrt(3) / 2, 3: 1 / (2 * math.sqrt(3))}   # apothem / side


def polygon_center_xy(xy: np.ndarray, n: int, side_m: float, iters: int = 40
                      ) -> Optional[Tuple[np.ndarray, float]]:
    """Fit a regular n-gon of KNOWN side length to a ragged top-face outline (robot xy).
    Robust (trimmed) least squares over centre + rotation: every outline point should lie on
    the polygon edge it is closest to; points far off (bulges from the side faces, depth
    noise) are dropped. Returns (centre_xy, vertex_yaw_deg) or None."""
    P = np.asarray(xy, float)
    if len(P) < 8 or n not in APOTHEM_PER_SIDE:
        return None
    a = APOTHEM_PER_SIDE[n] * side_m
    hull = cv2.convexHull(P.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    c, _ = polygon_centroid_xy(hull) if len(hull) >= 3 else (P.mean(0), 0)
    c = np.asarray(c, float)
    yaw_v, _ = polygon_yaw_deg(P, n)
    if not np.isfinite(yaw_v):
        return None
    th = math.radians(yaw_v + 180.0 / n)            # first FACE normal direction
    step = 2 * math.pi / n
    keep = np.ones(len(P), bool)
    for it in range(iters):
        ang = th + np.arange(n) * step
        N = np.column_stack([np.cos(ang), np.sin(ang)])
        D = (P - c) @ N.T                            # signed distance along each normal
        k = D.argmax(1)
        nk = N[k]
        r = D[np.arange(len(P)), k] - a              # distance to that edge line
        tk = np.column_stack([-nk[:, 1], nk[:, 0]])
        J = np.column_stack([-nk, ((P - c) * tk).sum(1)])
        if it >= 2:                                  # trim after the first coarse steps:
            lim = max(np.percentile(np.abs(r), 60), 1e-3)   # keep the best-fitting 60 %
            keep = np.abs(r) <= lim
            if keep.sum() < 8:
                keep = np.ones(len(P), bool)
        dx, *_ = np.linalg.lstsq(J[keep], -r[keep], rcond=None)
        c = c + dx[:2]
        th = th + float(np.clip(dx[2], -0.2, 0.2))
        if np.linalg.norm(dx[:2]) < 1e-5 and abs(dx[2]) < 1e-4:
            break
    yaw_out = (math.degrees(th) - 180.0 / n) % (360.0 / n)
    return c, yaw_out
