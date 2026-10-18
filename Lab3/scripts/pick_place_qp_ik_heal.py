#!/usr/bin/env python3
"""Pick and Place using Quadratic-Programming IK (joint-limit + step-limit
constrained) for Franka Panda and HEAL arm.

This is a drop-in variant of the original DLS-IK script. EVERYTHING outside
`solve_ik()` is byte-for-byte unchanged from the working version you posted.
Only the IK solver internals are replaced: instead of a fixed-damping
pseudo-inverse update, each iteration solves a small QP

    minimize_{dq}   || J dq - e ||^2  +  reg * || dq ||^2
    subject to       lb <= dq <= ub

where the box bounds `lb`/`ub` simultaneously enforce:
  (a) joint position limits:  q + dq within [joint_min, joint_max]
  (b) a per-step velocity/step limit (keeps the local linearization valid)

This removes the need for a hand-tuned fixed damping constant: near a
singularity the least-squares term `||J dq - e||^2 + reg||dq||^2` stays
well-posed (reg always keeps H = J^T J + reg*I positive definite), and the
box constraints stop any joint from being driven to/through its limit in a
single step, which is one of the common causes of divergence/failure you'd
see as "singularities" near joint-range edges.

Requirements (not in the original script):
    pip install qpsolvers[osqp]
"""

import csv
import math
import os
import re
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mujoco
import mujoco.viewer
import numpy as np

try:
    from PIL import Image as _PIL_Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import qpsolvers
    from qpsolvers import solve_qp
except ImportError as exc:
    raise ImportError(
        "This script requires the 'qpsolvers' package with a QP backend. "
        "Install with:\n    pip install qpsolvers[osqp]\n"
    ) from exc

# qpsolvers reports which backends are actually usable in THIS environment.
# "osqp" is preferred, but if the osqp extra didn't install correctly (very
# common -- `pip install qpsolvers[osqp]` sometimes resolves without an
# actual working osqp backend depending on platform/wheel availability),
# silently requesting solver="osqp" would make EVERY solve_qp() call raise
# on the very first IK iteration of the very first episode -- which looks
# exactly like "the arm never moves, 0% success" because the failure
# happens before any smooth_move() is ever reached.
_AVAILABLE_QP_SOLVERS = list(qpsolvers.available_solvers)
if not _AVAILABLE_QP_SOLVERS:
    raise RuntimeError(
        "qpsolvers is installed but reports NO available QP solver backends "
        "(qpsolvers.available_solvers is empty). Install one, e.g.:\n"
        "    pip install qpsolvers[osqp]\n"
        "or\n"
        "    pip install osqp\n"
        "then re-run. (If osqp still doesn't show up, try `pip install "
        "qpsolvers[quadprog]` as a fallback backend.)"
    )
QP_SOLVER = "osqp" if "osqp" in _AVAILABLE_QP_SOLVERS else _AVAILABLE_QP_SOLVERS[0]
print(f"[qpsolvers] available backends: {_AVAILABLE_QP_SOLVERS} -> using '{QP_SOLVER}'")

# ─────────────────────────────────────────────────────────────────────────────
# TARGET ROBOT SELECTION
# ─────────────────────────────────────────────────────────────────────────────
ROBOT_TO_RUN = "heal"  # Change to "heal" to run the HEAL arm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATACLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RobotConfig:
    name: str
    xml_path: str
    arm_joints: List[str]
    arm_actuators: List[str]
    gripper_actuators: List[str]
    gripper_open: float
    gripper_close: float
    home_angles: List[float]
    tray_body_name: str
    needs_temp_scene: bool
    # Optional per-robot extras. Defaults (None) keep the original Franka behaviour.
    jaw_axis_local: Optional[tuple] = None   # gripper closing axis in the TCP frame
    sample_bounds: Optional[dict] = None     # cube/tray sampling limits (smaller workspaces)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & SETUP
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DESC_DIR = REPO_ROOT / "robot_descriptions"
GRIPPER_XML = DESC_DIR / "robotiq_2f85_v4" / "2f85.xml"   # Robotiq 2F-85 mounted on the HEAL arm

# TCP offset (m) along the Robotiq approach axis, measured from its mounting plane:
# the finger pads span z = 0.121 .. 0.159 m, so 0.145 sits between the pad tips.
HEAL_TCP_Z = 0.145

TCP_SITE         = "tcp"
CUBE_BODY        = "cube"
CUBE_JOINT       = "cube_joint"
TABLE_SURFACE_Z  = 0.20
MIN_SEPARATION   = 0.32
APPROACH_OFFSET_Z= 0.12
GRASP_OFFSET_Z   = 0.010
LIFT_OFFSET_Z    = 0.15
COLLISION_MARGIN = 0.002
SCREENSHOT_DIR   = Path("screenshots")

N_EPISODES = 100
SEED = 42

TRAY_HALF_X = 0.075
TRAY_HALF_Y = 0.095
TRAY_BASE_HALF_Z = 0.006
TRAY_WALL_HALF_Z = 0.025
TRAY_WALL_THICK = 0.008

# ─── Constraint-violation tracking (for fair cross-method comparison) ───────
# Same values are used, unchanged, in pick_place_dls_ik.py and
# pick_place_mink_ik.py, so violation counts across the three CSVs are
# directly comparable. This is logging only -- it does not affect the
# IK solve or the motion in any way.
MAX_JOINT_VEL_RADPS = 2.175   # generic conservative per-joint velocity cap (rad/s)
JOINT_LIMIT_EPS     = 1e-3    # "at the limit" tolerance for logging purposes

ROBOT_CONFIGS = {
    "franka": RobotConfig(
        name="Franka Panda",
        xml_path="robot_descriptions/franka_panda_with_table.xml",
        arm_joints=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        arm_actuators=["actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7"],
        gripper_actuators=["actuator8"],
        gripper_open=255.0,
        gripper_close=0.0,
        home_angles=[0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854],
        tray_body_name="tray",
        needs_temp_scene=False
    ),
    "heal": RobotConfig(
        name="HEAL Arm",
        xml_path="robot_descriptions/arm_gripper_with_table.xml",
        arm_joints=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        arm_actuators=["turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3"],
        # Robotiq 2F-85 (attached with prefix "gripper/"): ONE tendon-driven actuator,
        # ctrl 0 = fully open, 255 = fully closed (the Franka's is the other way round).
        gripper_actuators=["gripper/fingers_actuator"],
        gripper_open=0.0,
        gripper_close=255.0,
        home_angles=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        tray_body_name="random_tray",
        needs_temp_scene=True,
        # Fingers close along the gripper's x axis. At the HEAL home pose that axis is
        # ~61 deg from world x (not axis-aligned), so the yaw sent to the IK has to be
        # corrected for it or the jaws close diagonally on the cube.
        jaw_axis_local=(1.0, 0.0, 0.0),
        # HEAL reaches ~0.66 m (grasp height) / ~0.58 m (lift & drop height) top-down;
        # keep cube + tray where a top-down pick AND place is physically reachable.
        sample_bounds={"x": (-0.28, 0.28), "y": (0.32, 0.75), "max_radius": 0.63},
    )
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES & LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def take_screenshot(model, data, ep: int, stage: str, robot_name: str) -> str:
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    prefix = robot_name.replace(" ", "_").lower()
    fname = SCREENSHOT_DIR / f"{prefix}_ep{ep:03d}_{stage}_{timestamp}.png"
    try:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = 180.0
        camera.elevation = -30.0
        camera.distance = 1.8
        camera.lookat[:] = [0.0, 0.4, 0.2]

        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        if hasattr(renderer, 'close'): renderer.close()

        if _HAS_PIL:
            _PIL_Image.fromarray(pixels).save(str(fname))
        else:
            ppm_path = str(fname).replace(".png", ".ppm")
            with open(ppm_path, "wb") as f:
                f.write(f"P6\n640 480\n255\n".encode())
                f.write(pixels.tobytes())
            fname = Path(ppm_path)
        print(f"  📸 Front-view screenshot saved: {fname}")
        return str(fname)
    except Exception as exc:
        print(f"  ⚠ Screenshot failed: {exc}")
        return ""

def rot_to_yaw_deg(R: np.ndarray) -> float:
    R_mat = np.asarray(R).reshape(3, 3)
    return math.degrees(math.atan2(R_mat[1, 0], R_mat[0, 0]))

def yaw_to_rot_matrix(yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)

def wrap_gripper_yaw(yaw_rad: float) -> float:
    """Wrap a yaw angle into (-pi/2, pi/2]. A parallel-jaw gripper's grasp
    pose repeats every 180 deg -- grasping at yaw and yaw+180 deg is the
    same physical gripper orientation to a symmetric gripper/cube -- so the
    wrist should always be commanded to the nearer of the two equivalent
    orientations. Without this, for a raw object yaw near +-90 to 180 deg,
    IK is asked to reach a needlessly large (sometimes kinematically
    unreachable, e.g. past a wrist joint's range limit) wrist rotation for
    a target that looks identical to one well within easy reach. This was
    the actual cause of orientation getting permanently stuck regardless of
    solver tuning -- the target itself, not the solver, was unreachable."""
    # Cube has 90-deg symmetry -> wrap into [-pi/4, pi/4) so the wrist never
    # needs more than 45 deg of yaw rotation.
    return (yaw_rad + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0

def rotation_error_vector(current_rotation, target_rotation):
    r = target_rotation @ current_rotation.T
    cosine = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]], dtype=float)
    if angle < 1e-8: return 0.5 * vee
    if abs(math.sin(angle)) > 1e-7: return (angle / (2.0 * math.sin(angle))) * vee
    eigvals, eigvecs = np.linalg.eig(r)
    axis = np.real(eigvecs[:, int(np.argmin(np.abs(eigvals - 1.0)))])
    norm = np.linalg.norm(axis)
    return np.zeros(3) if norm < 1e-10 else angle * axis / norm

def get_min_clearance(data, site_id: int, tray_id: int) -> float:
    """Calculates clearance to the table surface and the tray ONLY.
       Completely eliminates the cube from the distance metric."""
    tcp_pos = data.site_xpos[site_id]
    tray_pos = data.xpos[tray_id]

    dist_table_z = max(0.0, tcp_pos[2] - TABLE_SURFACE_Z)
    dist_tray = float(np.linalg.norm(tcp_pos - tray_pos))
    clearance_tray = max(0.0, dist_tray - 0.12)

    return min(dist_table_z, clearance_tray)

def count_joint_limit_violations(data, arm_qpos_addrs, joint_limits, eps=JOINT_LIMIT_EPS):
    """Counts how many arm joints are currently at/beyond their position limits.
       Logging only -- does not alter control or the IK solve."""
    q = np.array([data.qpos[a] for a in arm_qpos_addrs])
    below = q <= (joint_limits[:, 0] + eps)
    above = q >= (joint_limits[:, 1] - eps)
    return int(np.count_nonzero(below | above))

def count_velocity_violations(data, arm_dof_addrs, max_vel=MAX_JOINT_VEL_RADPS):
    """Counts how many arm joints currently exceed the velocity cap.
       Logging only -- does not alter control or the IK solve."""
    qv = np.array([data.qvel[d] for d in arm_dof_addrs])
    return int(np.count_nonzero(np.abs(qv) > max_vel))

def check_collisions(model, data, margin: float = COLLISION_MARGIN):
    mujoco.mj_forward(model, data)
    collisions = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < margin:
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"

            if ("tray" in g1 and "table" in g2) or ("table" in g1 and "tray" in g2): continue
            if ("cube" in g1 and "table" in g2) or ("table" in g1 and "cube" in g2): continue
            if ("cube" in g1 and "tray" in g2) or ("tray" in g1 and "cube" in g2): continue

            is_cube = ("cube" in g1) or ("cube" in g2)
            is_robot_gripper = any(x in g1 or x in g2 for x in ["finger", "hand", "pad", "panda", "driver", "geom_"])
            if is_cube and is_robot_gripper: continue

            collisions.append(f"{g1}↔{g2}")
    return len(collisions) > 0, list(set(collisions))

def check_pick_success(data, cube_id: int, site_id: int):
    cube_pos = data.xpos[cube_id].copy()
    tcp_pos = data.site_xpos[site_id].copy()
    horiz_dist = np.linalg.norm(cube_pos[:2] - tcp_pos[:2])
    if horiz_dist > 0.06: return False, f"Cube too far from TCP ({horiz_dist:.3f}m)"
    if cube_pos[2] - TABLE_SURFACE_Z < 0.02: return False, f"Cube not lifted off table (z={cube_pos[2]:.3f}m)"
    return True, "ok"

def check_place_success(data, cube_id: int, tray_id: int):
    cube_pos = data.xpos[cube_id].copy()
    tray_pos = data.xpos[tray_id].copy()
    xy_dist = np.linalg.norm(cube_pos[:2] - tray_pos[:2])
    if xy_dist > 0.12: return False, f"Cube placed outside tray ({xy_dist:.3f}m)"
    if cube_pos[2] < 0.18: return False, f"Cube fell off table (z={cube_pos[2]:.3f}m)"
    return True, "ok"

# ─────────────────────────────────────────────────────────────────────────────
# IK SOLVER (QP, joint-limit + step-limit constrained) & MOTION
# ─────────────────────────────────────────────────────────────────────────────
MAX_STEP_RAD = 0.16     # per-iteration joint step cap, same magnitude as the
                          # original DLS script's `largest_step` clamp
QP_REG       = 1e-4      # Tikhonov regularizer on dq; keeps H = J^T J + reg*I
                          # positive definite even when J loses rank
                          # (i.e. at/near a kinematic singularity)

# Mathematical model ported from a working reference implementation: the
# raw task error is magnitude-clamped BEFORE it is weighted and handed to
# the QP, matching the reference's max_err clamp. This is the real fix for
# large initial errors producing a badly-scaled single-shot QP solution --
# the step cap afterward (unchanged, see docstring below) is then rarely
# needed as anything but a safety net.
MAX_ERR_POS = 0.05
MAX_ERR_ROT = 0.3

def solve_ik(
    model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits,
    seed_angles, target_position, target_rotation, orientation_weight=0.8, max_iterations=650,
    max_err_pos=MAX_ERR_POS, max_err_rot=MAX_ERR_ROT
):
    # NOTE on orientation_weight: CSV logs showed position converging to a
    # few mm while rotation stayed stuck for hundreds of iterations. Root
    # cause: shrinking the orientation rows of J/error by a small weight
    # (previously 0.15) before the QP solve makes position dominate the
    # objective, starving orientation of correction. Raised to 0.8 (near
    # parity with position) so both converge on a comparable timescale.
    """Drop-in replacement for the DLS solve_ik(), same signature/return
    contract: (q, success, pos_err_norm, rot_err_norm, iters_used).

    Each iteration solves:
        minimize_dq   || J dq - e ||^2 + QP_REG * || dq ||^2
        subject to    lb <= dq <= ub        (lb/ub = joint-limit distance)

    then, same as the DLS version, if the solution's largest component
    exceeds MAX_STEP_RAD it is rescaled by a single uniform scalar (not
    re-solved with a tighter elementwise box). This matters: an elementwise
    per-joint step cap baked into the QP's constraints would make the QP
    *re-optimize* under those caps rather than just shrink its unconstrained
    answer -- and because orientation is weighted much more lightly than
    position in the combined objective (orientation_weight^2 << 1), a
    re-optimized solution under tight elementwise caps tends to spend nearly
    all of the allowed joint motion reducing position error and almost none
    on orientation, stalling orientation convergence even as position
    converges fine. A uniform scalar rescale instead preserves the same
    position/orientation balance as the unconstrained QP solution, exactly
    like the DLS version's step clamp -- and because lb<=0<=ub always holds
    here, scaling dq by any factor in [0,1] keeps q+dq inside the joint
    limits that were already enforced by the QP, so joint limits are still
    always respected after the rescale.
    """
    original_qpos = data.qpos.copy()
    q = np.asarray(seed_angles, dtype=float).copy()
    q = np.clip(q, joint_limits[:, 0] + 1e-4, joint_limits[:, 1] - 1e-4)
    jacp, jacr = np.zeros((3, model.nv), dtype=float), np.zeros((3, model.nv), dtype=float)
    n = len(arm_dof_addrs)
    iters_used = 0
    pe_norm = re_norm = float("inf")

    try:
        for iters_used in range(1, max_iterations + 1):
            for addr, angle in zip(arm_qpos_addrs, q): data.qpos[addr] = angle
            mujoco.mj_forward(model, data)

            pos_err = target_position - data.site_xpos[site_id]
            rot_err = rotation_error_vector(data.site_xmat[site_id].reshape(3, 3), target_rotation)
            pe_norm, re_norm = float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

            if pe_norm < 0.004 and re_norm < 0.09:
                return q.copy(), True, pe_norm, re_norm, iters_used

            # Clamp the RAW (unweighted) error magnitude before building the
            # weighted task error fed into the QP -- see note above.
            if pe_norm > max_err_pos:
                pos_err = pos_err * (max_err_pos / pe_norm)
            if re_norm > max_err_rot:
                rot_err = rot_err * (max_err_rot / re_norm)

            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            j = np.vstack((jacp[:, arm_dof_addrs], orientation_weight * jacr[:, arm_dof_addrs]))
            error = np.concatenate((pos_err, orientation_weight * rot_err))

            # QP formulation: minimize 0.5 dq^T P dq + c^T dq
            #   with ||J dq - e||^2 + reg||dq||^2
            #      = dq^T (J^T J + reg I) dq - 2 e^T J dq + const
            #   => matching 0.5 dq^T P dq + c^T dq term-by-term:
            #      P = 2 (J^T J + reg I),  c = -2 J^T e
            P = 2.0 * (j.T @ j + QP_REG * np.eye(n))
            P = 0.5 * (P + P.T)  # symmetrize for numerical safety
            c = -2.0 * (j.T @ error)

            # Box constraints enforce ONLY the joint-position limits here
            # (the per-step cap is applied afterward as a uniform rescale,
            # not folded into these bounds -- see the docstring above for
            # why mixing the two inside the QP distorts the solution).
            lb = joint_limits[:, 0] - q
            ub = joint_limits[:, 1] - q
            # Guard against lb > ub from floating point edge cases at the
            # very boundary of a joint limit.
            lb = np.minimum(lb, -1e-8)
            ub = np.maximum(ub,  1e-8)

            dq = solve_qp(P, c, lb=lb, ub=ub, solver=QP_SOLVER)
            if dq is None:
                # QP infeasible this iteration (shouldn't normally happen
                # with box-only constraints) -- fall back to a zero step so
                # the loop can still terminate via the iteration budget
                # rather than crashing.
                dq = np.zeros(n)
            else:
                dq = np.asarray(dq, dtype=float)
                # Uniform (direction-preserving) step-size cap, same role as
                # the DLS version's clamp. Safe w.r.t. the joint limits just
                # enforced above: lb<=0<=ub always, so for any q+dq already
                # inside [lb,ub], scaling dq by a factor in [0,1] keeps it
                # inside [lb,ub] too.
                largest_step = float(np.max(np.abs(dq)))
                if largest_step > MAX_STEP_RAD:
                    dq *= MAX_STEP_RAD / largest_step

            q = np.clip(q + dq, joint_limits[:, 0] + 1e-4, joint_limits[:, 1] - 1e-4)

        accepted = pe_norm < 0.010 and re_norm < 0.22
        return q.copy(), accepted, pe_norm, re_norm, iters_used
    finally:
        data.qpos[:] = original_qpos
        mujoco.mj_forward(model, data)

def smooth_move(model, data, viewer, actuator_ids, targets, duration: float, log_dict: dict, site_id: int, tray_id: int,
                 arm_qpos_addrs=None, arm_dof_addrs=None, joint_limits=None) -> bool:
    duration = duration / 2.0
    starts = np.array([data.ctrl[aid] for aid in actuator_ids])
    steps  = max(1, int(duration / model.opt.timestep))
    for step in range(steps):
        if not viewer.is_running(): return False
        t = (step + 1) / steps
        alpha = 0.5 * (1.0 - math.cos(math.pi * t))
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = starts[i] + alpha * (targets[i] - starts[i])
        mujoco.mj_step(model, data)

        dist = get_min_clearance(data, site_id, tray_id)
        if dist < log_dict["min_clearance_m"]:
            log_dict["min_clearance_m"] = dist

        # Constraint-violation tracking (logging only; only performed when
        # arm joint info is passed in, i.e. for arm moves, not gripper moves).
        if arm_qpos_addrs is not None:
            log_dict["joint_limit_violations"] += count_joint_limit_violations(data, arm_qpos_addrs, joint_limits)
            log_dict["velocity_limit_violations"] += count_velocity_violations(data, arm_dof_addrs)

        viewer.sync()
    return True

# ─────────────────────────────────────────────────────────────────────────────
# SCENE GENERATION & RANDOMIZATION
# ─────────────────────────────────────────────────────────────────────────────
def sub_once_text(text, old, new, what):
    """str.replace(old, new, 1) that fails loudly if the expected anchor is missing."""
    if text.count(old) < 1:
        raise RuntimeError(f"Could not find {what} ('{old.strip()}') in the scene XML")
    return text.replace(old, new, 1)

def _absolutize_mesh_paths(xml: str, base_dir: Path) -> str:
    """Rewrite every <mesh file="..."> to an absolute path and drop the <compiler>
    meshdir. This lets two MJCF files that use different meshdirs (HEAL arm:
    heal_meshes/, Robotiq: assets/) be combined in memory, no matter how a given
    MuJoCo version treats a child model's meshdir inside attach()."""
    m = re.search(r'\bmeshdir="([^"]*)"', xml)
    mesh_dir = (base_dir / m.group(1)) if m else base_dir
    xml = re.sub(r'\s+meshdir="[^"]*"', "", xml, count=1)
    return re.sub(
        r'(<mesh\b[^>]*?\bfile=")([^"]+)(")',
        lambda mo: mo.group(1) + (mesh_dir / mo.group(2)).as_posix() + mo.group(3),
        xml,
    )

def build_temporary_scene(source_xml: Path):
    """HEAL arm + Robotiq 2F-85 + table + random tray, assembled and compiled
    IN MEMORY (MjSpec). Returns (model, None); the second value only exists so
    main() stays unchanged -- no temp XML is written any more, so nothing can be
    left behind in robot_descriptions/ if a run is interrupted.

    Why this replaces the old text-patching: arm_gripper_with_table.xml contains
    NO gripper (it is attached at the 'right_center' site, exactly like
    load_heal_robot.py does), so the old code that patched left_driver_joint /
    right_driver_joint into the XML produced actuators pointing at joints that
    do not exist ("unknown transmission target 'left_driver_joint'")."""
    xml = source_xml.read_text(encoding="utf-8")

    # --- scene extras that are plain text edits of the arm file -----------------
    xml = sub_once_text(xml, "  </asset>",
        '    <material name="random_tray_mat" rgba="0.30 0.35 0.40 1" specular="0.3" shininess="0.4"/>\n  </asset>',
        "asset block")

    # The 2F-85 4-bar linkage is tuned for elliptic friction cones + impratio 10
    # (its own file sets this, but attach() ignores a child's <option>).
    xml = sub_once_text(xml, "<option ", '<option cone="elliptic" impratio="10" ', "option tag")

    # TCP between the finger pads, on the gripper approach (z) axis. The Robotiq
    # base frame coincides with the 'right_center' site frame (that is where it
    # is attached), so a site at (0,0,HEAL_TCP_Z) in this body IS the pad centre.
    # NOTE: the 'grasp_tcp' site inside 2f85.xml (z=0.0657) is at the finger-
    # driver pivots, ~7.5 cm short of the pads, so it is NOT used.
    xml, n_site = re.subn(
        r'(<site\s+name="right_center"[^>]*/>)',
        lambda mo: mo.group(1) + f'\n                  <site name="grasp_tcp" pos="0 0 {HEAL_TCP_Z}" size="0.006" rgba="0.1 1 0.2 1" group="1"/>',
        xml, count=1)
    if n_site != 1:
        raise RuntimeError("Could not find site 'right_center' in " + str(source_xml))

    tray_xml = f"""
    <body name="random_tray" pos="0 0 {TABLE_SURFACE_Z + TRAY_BASE_HALF_Z:.8f}">
      <geom name="random_tray_floor" type="box" size="{TRAY_HALF_X} {TRAY_HALF_Y} {TRAY_BASE_HALF_Z}" material="random_tray_mat" friction="0.8 0.1 0.1"/>
      <geom name="random_tray_wall_y_pos" type="box" size="{TRAY_HALF_X} {TRAY_WALL_THICK} {TRAY_WALL_HALF_Z}" pos="0 {TRAY_HALF_Y - TRAY_WALL_THICK:.6f} {TRAY_BASE_HALF_Z + TRAY_WALL_HALF_Z:.6f}" material="random_tray_mat"/>
      <geom name="random_tray_wall_y_neg" type="box" size="{TRAY_HALF_X} {TRAY_WALL_THICK} {TRAY_WALL_HALF_Z}" pos="0 {-TRAY_HALF_Y + TRAY_WALL_THICK:.6f} {TRAY_BASE_HALF_Z + TRAY_WALL_HALF_Z:.6f}" material="random_tray_mat"/>
      <geom name="random_tray_wall_x_pos" type="box" size="{TRAY_WALL_THICK} {TRAY_HALF_Y} {TRAY_WALL_HALF_Z}" pos="{TRAY_HALF_X - TRAY_WALL_THICK:.6f} 0 {TRAY_BASE_HALF_Z + TRAY_WALL_HALF_Z:.6f}" material="random_tray_mat"/>
      <geom name="random_tray_wall_x_neg" type="box" size="{TRAY_WALL_THICK} {TRAY_HALF_Y} {TRAY_WALL_HALF_Z}" pos="{-TRAY_HALF_X + TRAY_WALL_THICK:.6f} 0 {TRAY_BASE_HALF_Z + TRAY_WALL_HALF_Z:.6f}" material="random_tray_mat"/>
    </body>"""
    xml = sub_once_text(xml, "</worldbody>", tray_xml + "\n  </worldbody>", "worldbody close")

    # --- build the two specs ------------------------------------------------------
    arm_src  = xml
    grip_src = GRIPPER_XML.read_text(encoding="utf-8")
    arm  = mujoco.MjSpec.from_string(_absolutize_mesh_paths(arm_src,  source_xml.parent))
    grip = mujoco.MjSpec.from_string(_absolutize_mesh_paths(grip_src, GRIPPER_XML.parent))

    # Gravity compensation on every arm link and every gripper body (not the cube).
    # The arm is driven by position servos (kp 400-800) and carries ~25 kg of
    # links: without this the shoulder/elbow sag ~2-3 cm and tilt the tool ~3 deg
    # under load, so the fingertips would be driven into the table at the grasp.
    for text, spec, skip in ((arm_src, arm, {"cube"}), (grip_src, grip, set())):
        for name in re.findall(r'<body\s+name="([^"]+)"', text):
            if name in skip:
                continue
            body = spec.body(name)
            if body is not None:
                body.gravcomp = 1.0

    # Mount the Robotiq on the arm's flange site (same call as load_heal_robot.py).
    # Everything from the gripper gets the "gripper/" prefix, e.g.
    # "gripper/fingers_actuator", "gripper/left_driver_joint".
    arm.attach(grip, prefix="gripper/", site=arm.site("right_center"))
    return arm.compile(), None

def randomize_episode_positions(model, data, tray_id: int, cube_joint_id: int, rng: np.random.Generator, prev_cube_xy=None, prev_tray_xy=None, bounds=None):
    cube_xy = tray_xy = None
    for _ in range(20_000):
        if bounds is None:   # original full-table ranges (Franka) -- unchanged
            tray_candidate = np.array([rng.uniform(-0.27, 0.27), rng.uniform(0.34, 0.76)])
            cube_candidate = np.array([rng.uniform(-0.28, 0.28), rng.uniform(0.32, 0.75)])
        else:                # robot with a smaller reachable workspace (HEAL)
            tray_candidate = np.array([rng.uniform(*bounds["x"]), rng.uniform(*bounds["y"])])
            cube_candidate = np.array([rng.uniform(*bounds["x"]), rng.uniform(*bounds["y"])])
            if max(np.linalg.norm(tray_candidate), np.linalg.norm(cube_candidate)) > bounds["max_radius"]: continue
        if np.linalg.norm(cube_candidate - tray_candidate) < MIN_SEPARATION: continue
        if prev_cube_xy is not None and np.linalg.norm(cube_candidate - prev_cube_xy) < 0.12: continue
        if prev_tray_xy is not None and np.linalg.norm(tray_candidate - prev_tray_xy) < 0.25: continue
        cube_xy, tray_xy = cube_candidate, tray_candidate
        break

    if cube_xy is None: raise RuntimeError("Could not sample non-overlapping poses.")
    cube_yaw, tray_yaw = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(-math.pi, math.pi))

    model.body_pos[tray_id]  = np.array([tray_xy[0], tray_xy[1], TABLE_SURFACE_Z])
    model.body_quat[tray_id] = np.array([math.cos(tray_yaw / 2), 0.0, 0.0, math.sin(tray_yaw / 2)])

    adr, dof = model.jnt_qposadr[cube_joint_id], model.jnt_dofadr[cube_joint_id]
    data.qpos[adr:adr + 3] = [cube_xy[0], cube_xy[1], TABLE_SURFACE_Z + 0.02]
    data.qpos[adr + 3:adr + 7] = [math.cos(cube_yaw / 2), 0.0, 0.0, math.sin(cube_yaw / 2)]
    data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, data)
    return cube_xy.copy(), cube_yaw, tray_xy.copy(), tray_yaw

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if ROBOT_TO_RUN not in ROBOT_CONFIGS:
        sys.exit(f"[ERROR] Invalid ROBOT_TO_RUN: {ROBOT_TO_RUN}")

    cfg = ROBOT_CONFIGS[ROBOT_TO_RUN]
    rng = np.random.default_rng(SEED)

    xml_full_path = REPO_ROOT / cfg.xml_path
    temp_file_to_delete = None

    if cfg.needs_temp_scene:
        model, temp_file_to_delete = build_temporary_scene(xml_full_path)
        site_name_to_find = "grasp_tcp"
    else:
        model = mujoco.MjModel.from_xml_path(str(xml_full_path))
        site_name_to_find = TCP_SITE

    data = mujoco.MjData(model)

    def req_id(obj_type, name):
        idx = mujoco.mj_name2id(model, obj_type, name)
        if idx < 0: raise RuntimeError(f"Missing '{name}' in model")
        return idx

    site_id       = req_id(mujoco.mjtObj.mjOBJ_SITE, site_name_to_find)
    cube_id       = req_id(mujoco.mjtObj.mjOBJ_BODY, CUBE_BODY)
    tray_id       = req_id(mujoco.mjtObj.mjOBJ_BODY, cfg.tray_body_name)
    cube_joint_id = req_id(mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT)

    arm_jnt_ids    = [req_id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in cfg.arm_joints]
    arm_qpos_addrs = [int(model.jnt_qposadr[jid]) for jid in arm_jnt_ids]
    arm_dof_addrs  = [int(model.jnt_dofadr[jid]) for jid in arm_jnt_ids]
    joint_limits   = np.array([model.jnt_range[jid] for jid in arm_jnt_ids], dtype=float)

    arm_act_ids    = [req_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in cfg.arm_actuators]
    grip_act_ids   = [req_id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in cfg.gripper_actuators]

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_log_path = Path(f"{ROBOT_TO_RUN}_ik_log_qp_{timestamp_str}.csv")
    logs = []

    print("=" * 60)
    print(f"{cfg.name} QP-IK (joint-limit constrained) Pick & Place - {N_EPISODES} Episodes (2x Speed)")
    print("=" * 60)

    prev_cube_xy = prev_tray_xy = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.sync()
        for _ in range(int(0.5 / model.opt.timestep)):
            if not viewer.is_running(): return
            mujoco.mj_step(model, data); viewer.sync()

        for ep in range(1, N_EPISODES + 1):
            if not viewer.is_running(): break

            log = {
                "episode": ep,
                "robot": cfg.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cube_x": float("nan"), "cube_y": float("nan"), "cube_z": float("nan"), "cube_yaw_deg": float("nan"),
                "tray_x": float("nan"), "tray_y": float("nan"), "tray_z": float("nan"), "tray_yaw_deg": float("nan"),
                "end_tcp_x": float("nan"), "end_tcp_y": float("nan"), "end_tcp_z": float("nan"), "end_tcp_yaw_deg": float("nan"),
                "success": False, "fail_stage": "", "fail_reason": "", "screenshot": "",
                "solve_time_s": 0.0, "ik_iterations": 0, "min_clearance_m": float("inf"),
                "joint_limit_violations": 0, "velocity_limit_violations": 0
            }

            try:
                # 1. Reset joints & open gripper
                for addr, aid, angle in zip(arm_qpos_addrs, arm_act_ids, cfg.home_angles):
                    data.qpos[addr] = data.ctrl[aid] = angle
                for aid in grip_act_ids:
                    data.ctrl[aid] = cfg.gripper_open
                data.qvel[:] = 0.0

                # 2. Randomize poses
                prev_cube_xy, cube_yaw, prev_tray_xy, tray_yaw = randomize_episode_positions(
                    model, data, tray_id, cube_joint_id, rng, prev_cube_xy, prev_tray_xy, bounds=cfg.sample_bounds
                )

                for _ in range(int(0.5 / model.opt.timestep)):
                    if not viewer.is_running(): break
                    mujoco.mj_step(model, data); viewer.sync()

                mujoco.mj_forward(model, data)
                base_target_rot = data.site_xmat[site_id].reshape(3, 3).copy()
                cube_pos, tray_pos = data.xpos[cube_id].copy(), data.xpos[tray_id].copy()

                log["min_clearance_m"] = get_min_clearance(data, site_id, tray_id)

                log.update({
                    "cube_x": cube_pos[0], "cube_y": cube_pos[1], "cube_z": cube_pos[2], "cube_yaw_deg": math.degrees(cube_yaw),
                    "tray_x": tray_pos[0], "tray_y": tray_pos[1], "tray_z": tray_pos[2], "tray_yaw_deg": math.degrees(tray_yaw),
                })

                print(f"\n--- EPISODE {ep}/{N_EPISODES} ---")
                print(f"  Cube: x={cube_pos[0]:.3f} y={cube_pos[1]:.3f} yaw={log['cube_yaw_deg']:.1f}°")
                print(f"  Tray: x={tray_pos[0]:.3f} y={tray_pos[1]:.3f} yaw={log['tray_yaw_deg']:.1f}°")

                # Wrap to the gripper's 180-deg-symmetric equivalent yaw before
                # building the target orientation -- see wrap_gripper_yaw().
                # Yaw of the gripper's closing axis at the home pose (0 for Franka, where
                # cfg.jaw_axis_local is None -> identical to the original behaviour).
                jaw_home_yaw = 0.0
                if cfg.jaw_axis_local is not None:
                    jaw_world = base_target_rot @ np.asarray(cfg.jaw_axis_local, dtype=float)
                    jaw_home_yaw = math.atan2(jaw_world[1], jaw_world[0])
                grasp_rot = yaw_to_rot_matrix(wrap_gripper_yaw(cube_yaw - jaw_home_yaw)) @ base_target_rot
                drop_rot  = grasp_rot  # tray yaw is irrelevant for a top-down cube drop

                approach_pos = cube_pos + np.array([0.0, 0.0, APPROACH_OFFSET_Z])
                grasp_pos    = cube_pos + np.array([0.0, 0.0, GRASP_OFFSET_Z])
                lift_pos     = cube_pos + np.array([0.0, 0.0, LIFT_OFFSET_Z])
                drop_pos     = tray_pos + np.array([0.0, 0.0, 0.15])

                failed = False

                def record_failure(stage: str, reason: str):
                    nonlocal failed
                    failed = True
                    print(f"  ✗ [{stage}] Failed: {reason}")
                    log["fail_stage"], log["fail_reason"] = stage, reason
                    tcp = data.site_xpos[site_id].copy()
                    log["end_tcp_x"], log["end_tcp_y"], log["end_tcp_z"] = tcp
                    log["end_tcp_yaw_deg"] = rot_to_yaw_deg(data.site_xmat[site_id])
                    log["screenshot"] = take_screenshot(model, data, ep, stage, cfg.name)

                # ── 1. IK Approach ─────────────────────────────────────────────
                print("  [IK] Solving approach pose...")
                t0 = time.time()
                q_app, ok_app, pe, re, iters = solve_ik(
                    model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, cfg.home_angles, approach_pos, grasp_rot
                )
                if not ok_app:
                    # Retry with orientation_weight RAISED to full parity (1.0),
                    # not lowered -- see note above on why lowering it further
                    # made the already-starved orientation task worse.
                    q_app, ok_app, pe, re, iters2 = solve_ik(
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, cfg.home_angles, approach_pos, grasp_rot, orientation_weight=1.0
                    )
                    iters += iters2

                log["solve_time_s"] += (time.time() - t0)
                log["ik_iterations"] += iters

                if not ok_app: record_failure("Approach_IK", f"IK infeasible (pos_err={pe:.4f}m, rot_err={re:.4f}rad)")

                if not failed:
                    print("  → Moving to approach pose...")
                    if not smooth_move(model, data, viewer, arm_act_ids, q_app, 2.0, log, site_id, tray_id,
                                        arm_qpos_addrs, arm_dof_addrs, joint_limits): break
                    col, col_desc = check_collisions(model, data)
                    if col: record_failure("Approach_Move", f"Collision detected: {', '.join(col_desc)}")

                # ── 2. IK Grasp ────────────────────────────────────────────────
                if not failed:
                    print("  [IK] Solving grasp pose...")
                    t0 = time.time()
                    q_gr, ok_gr, pe, re, iters = solve_ik(
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_app, grasp_pos, grasp_rot
                    )
                    log["solve_time_s"] += (time.time() - t0)
                    log["ik_iterations"] += iters
                    if not ok_gr: record_failure("Grasp_IK", f"IK infeasible (pos_err={pe:.4f}m, rot_err={re:.4f}rad)")

                if not failed:
                    print("  → Lowering to cube...")
                    smooth_move(model, data, viewer, arm_act_ids, q_gr, 1.5, log, site_id, tray_id,
                                arm_qpos_addrs, arm_dof_addrs, joint_limits)
                    col, col_desc = check_collisions(model, data)
                    if col: record_failure("Lower_Move", f"Collision detected: {', '.join(col_desc)}")

                # ── 3. Close Gripper & Lift ───────────────────────────────────
                if not failed:
                    print("  → Closing gripper...")
                    target_grip = [cfg.gripper_close] * len(grip_act_ids)
                    smooth_move(model, data, viewer, grip_act_ids, target_grip, 1.0, log, site_id, tray_id)
                    for _ in range(50):
                        if not viewer.is_running(): break
                        mujoco.mj_step(model, data); viewer.sync()

                    print("  → Lifting cube...")
                    t0 = time.time()
                    q_lift, ok_lift, pe, re, iters = solve_ik(
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_gr, lift_pos, grasp_rot
                    )
                    log["solve_time_s"] += (time.time() - t0)
                    log["ik_iterations"] += iters

                    smooth_move(model, data, viewer, arm_act_ids, q_lift, 1.5, log, site_id, tray_id,
                                arm_qpos_addrs, arm_dof_addrs, joint_limits)
                    mujoco.mj_forward(model, data)

                    pick_ok, pick_msg = check_pick_success(data, cube_id, site_id)
                    if not pick_ok: record_failure("Pick_Verify", pick_msg)

                # ── 4. IK Drop ────────────────────────────────────────────────
                if not failed:
                    print("  [IK] Solving drop pose...")
                    t0 = time.time()
                    q_drop, ok_drop, pe, re, iters = solve_ik(
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_lift, drop_pos, drop_rot
                    )
                    log["solve_time_s"] += (time.time() - t0)
                    log["ik_iterations"] += iters

                    if not ok_drop: record_failure("Drop_IK", f"IK infeasible (pos_err={pe:.4f}m, rot_err={re:.4f}rad)")

                if not failed:
                    print("  → Moving to tray...")
                    smooth_move(model, data, viewer, arm_act_ids, q_drop, 2.25, log, site_id, tray_id,
                                arm_qpos_addrs, arm_dof_addrs, joint_limits)
                    col, col_desc = check_collisions(model, data)
                    if col: record_failure("Transit_Move", f"Collision detected: {', '.join(col_desc)}")

                if not failed:
                    print("  → Opening gripper (dropping)...")
                    target_grip = [cfg.gripper_open] * len(grip_act_ids)
                    smooth_move(model, data, viewer, grip_act_ids, target_grip, 0.5, log, site_id, tray_id)

                    for _ in range(int(0.75 / model.opt.timestep)):
                        if not viewer.is_running(): break
                        mujoco.mj_step(model, data); viewer.sync()

                    mujoco.mj_forward(model, data)
                    place_ok, place_msg = check_place_success(data, cube_id, tray_id)
                    if not place_ok:
                        record_failure("Drop_Verify", place_msg)
                    else:
                        log["success"] = True
                        print("  ✓ Episode Successful!")
                        log["end_tcp_x"], log["end_tcp_y"], log["end_tcp_z"] = data.site_xpos[site_id].copy()
                        log["end_tcp_yaw_deg"] = rot_to_yaw_deg(data.site_xmat[site_id])

            except Exception as exc:
                print(f"  ✗ [Unhandled Exception] Aborting episode {ep}: {exc}")
                traceback.print_exc()
                log["fail_stage"], log["fail_reason"] = "UnhandledException", str(exc)
                log["screenshot"] = take_screenshot(model, data, ep, "exception", cfg.name)

            logs.append(log)

            for _ in range(int(0.75 / model.opt.timestep)):
                if not viewer.is_running(): break
                mujoco.mj_step(model, data); viewer.sync()

    # ── Write Log Summary & Cleanup ───────────────────────────────────────
    if logs:
        with open(csv_log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)

        success_count = sum(1 for l in logs if l["success"])
        mean_time = np.mean([l["solve_time_s"] for l in logs])
        mean_jl_viol = np.mean([l["joint_limit_violations"] for l in logs])
        mean_vel_viol = np.mean([l["velocity_limit_violations"] for l in logs])

        print("\n" + "=" * 60)
        print(f"All episodes complete. Success Rate: {success_count}/{len(logs)} ({100*success_count/len(logs):.1f}%)")
        print(f"Mean Planning Time (Solve IK) per episode: {mean_time:.4f} s")
        print(f"Mean joint-limit violations per episode: {mean_jl_viol:.2f}")
        print(f"Mean velocity-limit violations per episode: {mean_vel_viol:.2f}")
        print(f"Detailed CSV Log: {csv_log_path}")

    if temp_file_to_delete and os.path.exists(temp_file_to_delete):
        try: os.remove(temp_file_to_delete)
        except OSError: pass

if __name__ == "__main__":
    main()
