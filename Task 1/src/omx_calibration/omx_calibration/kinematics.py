"""OpenMANIPULATOR-X kinematics matching the ROBOTIS URDF that the arm stack runs
(open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x.urdf, ROS 2 Jazzy):

  link1 -> joint1 : xyz (0.012, 0, 0)       axis z   (DXL ID 11)
  link2 -> joint2 : xyz (0, 0, 0.0595)      axis y   (DXL ID 12)
  link3 -> joint3 : xyz (0.024, 0, 0.128)   axis y   (DXL ID 13)
  link4 -> joint4 : xyz (0.124, 0, 0)       axis y   (DXL ID 14)
  link5 -> end_effector_link : xyz (0.126, 0, 0)  fixed
Frame = link1 (= world in the ROBOTIS launch), metres / radians.

Pitch phi = q2 + q3 + q4: 0 = gripper horizontal, +pi/2 = pointing straight DOWN.
The calibration wizard checks this model against the live TF before anything uses it.
"""
import math

import numpy as np

J1 = np.array([0.0, 0.0, 0.0])   # origin = base ROTATION AXIS (ROBOTIS link1 is 12 mm behind it)
LINK1_TO_AXIS = 0.012
J2_Z = 0.0595
L3 = np.array([0.024, 0.0, 0.128])
L4 = 0.124
EE = 0.126
# joint1 (ID 11) runs in Extended Position (multi-turn) mode on this setup (scripts/patch_robotis_base_multiturn.sh): the board is
# BEHIND the robot, right on the +-180 deg point, so the base must be free to pass it.
# IK always picks the base angle closest to the current one (see near()).
LIMITS = np.array([[-6 * math.pi, 6 * math.pi], [-1.5, 1.5], [-1.5, 1.4], [-1.7, 1.97]])
DOWN = math.pi / 2


def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def near(a, ref):
    """The angle equal to a (mod 2*pi) that is closest to ref - no long way round."""
    return ref + wrap(a - ref)


def frames(q):
    q = np.asarray(q, float)
    R = np.eye(3)
    p = J1.copy()
    o, a = [p.copy()], [R[:, 2].copy()]
    R = R @ _rz(q[0])
    p = p + R @ np.array([0, 0, J2_Z])
    o.append(p.copy()); a.append(R[:, 1].copy())
    R = R @ _ry(q[1])
    p = p + R @ L3
    o.append(p.copy()); a.append(R[:, 1].copy())
    R = R @ _ry(q[2])
    p = p + R @ np.array([L4, 0, 0])
    o.append(p.copy()); a.append(R[:, 1].copy())
    R = R @ _ry(q[3])
    return o, a, p + R @ np.array([EE, 0, 0])


def fk(q):
    """-> (end_effector_link position [m], pitch [rad])"""
    _, _, ee = frames(q)
    return ee, q[1] + q[2] + q[3]


def jacobian(q):
    o, a, ee = frames(q)
    J = np.zeros((4, 4))
    for i in range(4):
        J[:3, i] = np.cross(a[i], ee - o[i])
    J[3] = [0, 1, 1, 1]
    return J


def in_limits(q, tol=1e-6):
    q = np.asarray(q)
    return bool(np.all(q >= LIMITS[:, 0] - tol) and np.all(q <= LIMITS[:, 1] + tol))


def ik(p, pitch=DOWN, q0=None, tol=2e-4, max_iter=200, pitch_weight=0.1):
    """Damped-least-squares IK for position + pitch, joint limits by clamping, with restarts.
    Returns (q, ok). q0 = seed (use the current/commanded joints for smooth motion)."""
    p = np.asarray(p, float)
    yaw = math.atan2(p[1], p[0] - J1[0])
    if q0 is not None:
        yaw = near(yaw, float(q0[0]))
    seeds = ([np.asarray(q0, float)] if q0 is not None else []) + [
        np.array([yaw, 0.0, 0.0, 0.0]), np.array([yaw, -0.5, 0.5, 1.0]),
        np.array([yaw, 0.6, -0.3, 1.2])]
    best = None
    for s in seeds:
        q = np.clip(s, LIMITS[:, 0], LIMITS[:, 1])
        for _ in range(max_iter):
            x, phi = fk(q)
            e = np.concatenate([p - x, [pitch_weight * wrap(pitch - phi)]])
            if np.linalg.norm(e[:3]) < tol and abs(e[3]) / pitch_weight < math.radians(0.3):
                if q0 is not None:
                    q = q.copy()
                    q[0] = near(q[0], float(q0[0]))
                return q, True
            J = jacobian(q)
            J[3] *= pitch_weight
            smin = np.linalg.svd(J, compute_uv=False)[-1]
            lam = 1e-3 if smin > 0.02 else 0.04 * math.sqrt(1 - (smin / 0.02) ** 2) + 1e-3
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(4), e)
            m = np.max(np.abs(dq))
            if m > 0.2:
                dq *= 0.2 / m
            q = np.clip(q + dq, LIMITS[:, 0], LIMITS[:, 1])
        err = np.linalg.norm(fk(q)[0] - p)
        if best is None or err < best[1]:
            best = (q, err)
    return best[0], False
