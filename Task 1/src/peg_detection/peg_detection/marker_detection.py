"""Robot-base reference via an ArUco marker.

Why a marker: the Manipulator X base is black, like its cables, gripper and
shadows, so colour segmentation cannot find it reliably. A printed ArUco
marker gives a unique ID, an exact pixel location AND a full 6-DoF pose
(position + orientation) from the four corners and the known side length.

Marker frame (OpenCV convention), as seen when looking at the printed marker:
    +X to the right along the marker, +Y up along the marker,
    +Z out of the paper (towards the camera if lying flat facing up).
"""
from typing import List, Tuple

import cv2
import numpy as np


def _dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f'Unknown ArUco dictionary {name}')
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_markers(gray: np.ndarray, dict_name: str) -> List[Tuple[int, np.ndarray]]:
    """Return [(id, corners(4,2) float32 in order TL, TR, BR, BL)]."""
    dictionary = _dictionary(dict_name)
    if hasattr(cv2.aruco, 'ArucoDetector'):             # OpenCV >= 4.7
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
    else:                                               # older contrib API
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2).astype(np.float32)) for i, c in zip(ids.flatten(), corners)]


def estimate_pose(corners: np.ndarray, length_m: float, K: np.ndarray, dist: np.ndarray):
    """Marker pose in the camera frame: p_cam = R @ p_marker + t."""
    h = length_m / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners.reshape(4, 1, 2), K, dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Reprojection error in pixels: a quick quality number.
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - corners.reshape(4, 2), axis=1)))
    return R, tvec.reshape(3), rvec.reshape(3), err


def camera_to_marker(p_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Express a camera-frame point in the marker frame: p_m = R^T (p_c - t)."""
    return R.T @ (np.asarray(p_cam, dtype=np.float64) - t)


def rotation_to_quaternion(R: np.ndarray):
    """3x3 rotation matrix -> (x, y, z, w)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (m[2, 1] - m[1, 2]) / s
        y, z = (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x = (m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s
        y, z = (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return tuple(q / np.linalg.norm(q))
