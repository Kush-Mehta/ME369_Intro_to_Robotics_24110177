"""
utils.py -- rotation helper functions for HW1 Part 2.

This module is provided COMPLETE and working -- you do not need to
modify it. Use it from 01_rotation_sandbox.py and
02_verify_skew_properties.py so you can focus on the actual
assignment logic rather than re-deriving basic rotation utilities.

Conventions:
  - Rotation matrices R are 3x3 numpy arrays, R in SO(3).
  - MuJoCo stores orientation as a wxyz quaternion in qpos[3:7] for
    a freejoint (the FIRST 3 entries of qpos are the xyz position).
  - hat(w) returns the 3x3 skew-symmetric matrix of w, matching the
    \\hat{omega} notation used in lecture and in HW1 Problem 5.
"""

import numpy as np


def hat(w):
    """Skew-symmetric matrix of a 3-vector w, i.e. w^ such that
    w^ v = w x v for all v. This is the same operator written
    \\hat{omega} in the '05 More Rotations' notes and in HW1."""
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array([
        [0.0,    -w[2],  w[1]],
        [w[2],    0.0,  -w[0]],
        [-w[1],  w[0],   0.0],
    ])


def vee(W):
    """Inverse of hat(): recover the 3-vector w from a skew-symmetric
    3x3 matrix W."""
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def Rx(theta):
    """Elementary rotation about x by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(theta):
    """Elementary rotation about y by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(theta):
    """Elementary rotation about z by theta radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


ELEMENTARY_ROTATIONS = {"x": Rx, "y": Ry, "z": Rz}


def quat_to_R(q):
    """MuJoCo wxyz quaternion -> 3x3 rotation matrix."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def R_to_quat(R):
    """3x3 rotation matrix -> MuJoCo wxyz quaternion."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def set_body_orientation(data, R, qpos_addr=3):
    """Write rotation matrix R into a freejoint's qpos quaternion slot.
    qpos_addr=3 is correct for a body whose freejoint is the first (and
    only) joint, as in asymmetric_body.xml (qpos[0:3]=xyz, qpos[3:7]=quat).
    """
    data.qpos[qpos_addr:qpos_addr + 4] = R_to_quat(R)


def get_body_orientation(data, qpos_addr=3):
    """Read the current rotation matrix R back out of qpos."""
    return quat_to_R(data.qpos[qpos_addr:qpos_addr + 4])


def is_close_to_identity(M, tol=1e-9):
    """Convenience check used when verifying identities in Task 2."""
    return np.max(np.abs(M - np.eye(M.shape[0]))) < tol
