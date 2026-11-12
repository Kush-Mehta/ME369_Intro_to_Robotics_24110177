"""Rigid fit used by the robot-frame step (same maths as peg_detection.geometry3d)."""
import numpy as np


def rot_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fit_yaw_translation(p, q):
    """Yaw about z + translation with  q ~= Rz(yaw) p + d  (least squares). p, q: (N,3)."""
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
    return yaw, d, q - (p @ R.T + d)
