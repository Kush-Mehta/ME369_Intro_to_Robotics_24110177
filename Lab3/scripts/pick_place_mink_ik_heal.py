#!/usr/bin/env python3
"""Pick and Place using mink-based IK for Franka Panda and HEAL arm.

This is a drop-in variant of the original DLS-IK script. EVERYTHING outside
`solve_ik()` (scene generation, randomization, collision checking, motion,
logging, main loop) is byte-for-byte unchanged from the working version you
posted. Only the IK solver internals are replaced, using the `mink`
differential-IK library (https://github.com/kevinzakka/mink), which wraps a
QP-based velocity IK with task costs + configuration/velocity limits and is
built specifically to avoid the blow-ups that a fixed-damping DLS update
suffers near singularities.

Requirements (not in the original script):
    pip install mink qpsolvers[osqp]

Notes on why this helps your ~50% singularity failure rate:
- mink solves a QP at every step (minimize weighted task velocity error,
  subject to joint position/velocity limits) instead of a raw pseudo-inverse
  update, so there is no need for a hand-tuned fixed damping constant that
  either under-regularizes (blow-up near singularities) or over-regularizes
  (stalls far from them).
- Velocity limits are enforced as hard QP constraints (not a post-hoc clamp
  + rescale of the whole step), so individual joints don't get thrown to
  their limits while others stall.
- Because only the arm DOFs are exposed to the task (all other DOFs are
  zeroed before integration), the solve is well-conditioned for exactly the
  7/6-DOF chain you care about.
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
    import mink
except ImportError as exc:
    raise ImportError(
        "This script requires the 'mink' package. Install with:\n"
        "    pip install mink qpsolvers[osqp]\n"
    ) from exc

# mink's solve_ik() delegates to qpsolvers internally. If the osqp backend
# didn't actually get installed correctly (`pip install mink qpsolvers[osqp]`
# can resolve without a working osqp wheel on some platforms), requesting
# solver="osqp" makes EVERY call raise on the very first IK iteration of the
# very first episode -- which looks exactly like "the arm never moves, 0%
# success", because the failure happens before smooth_move() is ever reached.
try:
    import qpsolvers
    _AVAILABLE_QP_SOLVERS = list(qpsolvers.available_solvers)
except ImportError:
    _AVAILABLE_QP_SOLVERS = []
if not _AVAILABLE_QP_SOLVERS:
    raise RuntimeError(
        "No usable qpsolvers backend was found (qpsolvers.available_solvers "
        "is empty), but mink requires one. Install one, e.g.:\n"
        "    pip install qpsolvers[osqp]\n"
        "or\n"
        "    pip install osqp\n"
        "then re-run."
    )
MINK_QP_SOLVER = "osqp" if "osqp" in _AVAILABLE_QP_SOLVERS else _AVAILABLE_QP_SOLVERS[0]
print(f"[qpsolvers] available backends: {_AVAILABLE_QP_SOLVERS} -> using '{MINK_QP_SOLVER}'")

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
# pick_place_qp_ik.py, so violation counts across the three CSVs are
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
        # Use Franka's original coordinate ranges, but cap both target radii
        # to an empirically selected 0.63 m for approximately 10% IK failures.
        sample_bounds={"max_radius": 0.63},
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
# IK SOLVER (mink-based) & MOTION
# ─────────────────────────────────────────────────────────────────────────────
# One mink.Configuration / task pair is reused across calls, keyed by model id,
# to avoid rebuilding them on every single solve_ik() invocation.
_MINK_CACHE = {}

def _get_mink_objects(model, arm_dof_addrs):
    """Builds (and caches) the mink Configuration + limits for this model.
    Only called once per model instance.

    NOTE: the FrameTask is deliberately NOT cached here -- its
    orientation_cost must track the caller's `orientation_weight` (see
    solve_ik below), so it is built fresh on every solve_ik() call instead.
    Caching it with a fixed cost would silently ignore the fallback retry
    that lowers orientation_weight when the first IK attempt fails."""
    key = id(model)
    if key in _MINK_CACHE:
        return _MINK_CACHE[key]

    configuration = mink.Configuration(model)

    # Joint (position) limits from the MJCF, same source of truth the
    # original DLS solver used (`model.jnt_range`).
    limits = [mink.ConfigurationLimit(model=model)]

    # Mask of which generalized velocities belong to the controlled arm
    # chain. Any DOF not in this set (e.g. gripper fingers, free joints)
    # is zeroed out before integration so it never moves during IK solves,
    # mirroring the original script's "only the 7/6 arm joints are driven"
    # behavior.
    nv_mask = np.zeros(model.nv, dtype=bool)
    nv_mask[list(arm_dof_addrs)] = True

    bundle = {"configuration": configuration, "limits": limits, "nv_mask": nv_mask}
    _MINK_CACHE[key] = bundle
    return bundle

def solve_ik(
    model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits,
    seed_angles, target_position, target_rotation, orientation_weight=0.8, max_iterations=650,
    home_angles=None, k_null=3e-4
):
    """Drop-in replacement for the DLS solve_ik(), same signature/return
    contract: (q, success, pos_err_norm, rot_err_norm, iters_used).

    Internally runs mink's QP-based differential IK instead of a fixed-
    damping pseudo-inverse update.

    NOTE on orientation_weight: CSV logs across all 3 solvers showed the
    same pattern -- position converging fine while rotation stayed stuck.
    Root cause: a low orientation_weight (previously 0.15) as the
    FrameTask's orientation_cost makes mink's QP treat orientation as
    nearly irrelevant next to position_cost=1.0. Raised to 0.8 for parity.

    NOTE on k_null: previously 0.5, comparable to position_cost=1.0. Unlike
    the DLS/QP variants' analytic null-space projector N=(I-J^+J), mink's
    PostureTask is NOT guaranteed to stay confined to the task's null
    space -- it is resolved as a second weighted objective in the same QP,
    so at a high cost it actively competed with the FrameTask and pulled
    the arm away from the target (pos_err regressed from ~0.003m to
    0.17-0.37m, pegged at the iteration budget, once this was added).
    Lowered to 0.03 -- small enough to only break ties in the redundant
    DOF without fighting the primary task.

    A mink.PostureTask with a low cost is added alongside the main
    FrameTask when home_angles is given. This is mink's built-in null-space
    mechanism -- it only has leverage over the redundant (7th) DOF left
    unconstrained by the primary 6D pose task, so it pulls the elbow toward
    a reference posture without perturbing the end-effector task, exactly
    like the null-space projection N=(I-J^+J) used in the DLS/QP variants.
    Without it, mink is free to converge to whatever elbow configuration is
    nearest the seed, including elbow-locked/wrist-flip postures near
    singularities.
    """
    original_qpos = data.qpos.copy()
    site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id)

    bundle = _get_mink_objects(model, arm_dof_addrs)
    configuration = bundle["configuration"]
    limits = bundle["limits"]
    nv_mask = bundle["nv_mask"]

    # Built fresh each call so orientation_weight actually takes effect --
    # this is what makes the fallback retry (lower orientation_weight when
    # the first attempt fails) behave differently from the first attempt,
    # same as it does in the DLS version.
    task = mink.FrameTask(
        frame_name=site_name,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=float(orientation_weight),
        lm_damping=1.0,
    )

    # Seed the configuration's full qpos from current data, then overwrite
    # just the arm joints with the seed angles (matches original behavior
    # of only perturbing the arm chain before solving).
    q_full = data.qpos.copy()
    for addr, angle in zip(arm_qpos_addrs, np.asarray(seed_angles, dtype=float)):
        q_full[addr] = angle
    configuration.update(q_full)

    target_se3 = mink.SE3.from_rotation_and_translation(
        mink.SO3.from_matrix(np.asarray(target_rotation, dtype=float)),
        np.asarray(target_position, dtype=float),
    )
    task.set_target(target_se3)

    tasks = [task]
    if home_angles is not None:
        posture_task = mink.PostureTask(model=model, cost=k_null)
        # Full-qpos posture target: keep every DOF at its current solve-time
        # value except the arm joints, which get pulled toward home_angles
        # (low cost, so this never fights the primary FrameTask -- it only
        # acts in whatever null space direction is left over).
        posture_target = q_full.copy()
        for addr, angle in zip(arm_qpos_addrs, np.asarray(home_angles, dtype=float)):
            posture_target[addr] = angle
        posture_task.set_target(posture_target)
        tasks.append(posture_task)

    dt = 0.02          # integration step for the velocity IK loop (s)
    # The quantity that actually matters here is max_vel * dt, the per-
    # iteration joint-space step cap -- it needs to match the DLS version's
    # 0.16 rad/iteration clamp for a fair, equal-iteration-budget comparison.
    # (max_vel alone is not a real robot speed limit; it only exists to
    # bound this velocity IK's per-iteration integration step.)
    max_vel = 0.16 / dt   # = 8.0 rad/s, giving a 0.16 rad/iteration cap
    pe_norm = re_norm = float("inf")
    iters_used = 0

    try:
        for iters_used in range(1, max_iterations + 1):
            vel = mink.solve_ik(
                configuration,
                # Posture task is only a soft tie-breaker for the redundant
                # DOF. It is dropped for the last 40% of the iteration budget
                # so it can never leave a steady-state position/orientation
                # offset (it acts as a spring toward home and otherwise
                # biases the converged pose).
                tasks=tasks if iters_used <= int(0.6 * max_iterations) else [task],
                dt=dt,
                solver=MINK_QP_SOLVER,
                limits=limits,
                damping=1e-6,   # tiny Tikhonov term purely for QP numerical
                                  # conditioning; the real singularity handling
                                  # comes from the QP's own formulation + limits
            )
            vel = np.asarray(vel, dtype=float)
            vel[~nv_mask] = 0.0

            # Uniform (direction-preserving) velocity cap. An elementwise
            # clip here would distort the balance between the position and
            # orientation task components whenever any single joint's
            # velocity exceeds max_vel -- mirroring the same distortion
            # found in the QP-based variant's step cap. A uniform scalar
            # rescale keeps the solve's chosen direction intact.
            largest_vel = float(np.max(np.abs(vel))) if vel.size else 0.0
            if largest_vel > max_vel:
                vel *= max_vel / largest_vel

            configuration.integrate_inplace(vel, dt)

            # Sync mink's internal qpos back into `data` so we can read the
            # actual site pose via MuJoCo's own forward kinematics (keeps the
            # error computation identical in meaning to the DLS version).
            data.qpos[:] = configuration.q
            mujoco.mj_forward(model, data)

            pos_err = np.asarray(target_position, dtype=float) - data.site_xpos[site_id]
            rot_err = rotation_error_vector(data.site_xmat[site_id].reshape(3, 3), target_rotation)
            pe_norm = float(np.linalg.norm(pos_err))
            re_norm = float(np.linalg.norm(rot_err))

            if pe_norm < 0.004 and re_norm < 0.09:
                q_arm = np.array([data.qpos[a] for a in arm_qpos_addrs], dtype=float)
                q_arm = np.clip(q_arm, joint_limits[:, 0] + 1e-4, joint_limits[:, 1] - 1e-4)
                return q_arm, True, pe_norm, re_norm, iters_used

        accepted = pe_norm < 0.010 and re_norm < 0.22
        q_arm = np.array([data.qpos[a] for a in arm_qpos_addrs], dtype=float)
        q_arm = np.clip(q_arm, joint_limits[:, 0] + 1e-4, joint_limits[:, 1] - 1e-4)
        return q_arm, accepted, pe_norm, re_norm, iters_used
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
        if bounds is None or "x" not in bounds or "y" not in bounds:
            # Preserve Franka's exact cube and tray coordinate ranges.
            tray_candidate = np.array([rng.uniform(-0.27, 0.27), rng.uniform(0.34, 0.76)])
            cube_candidate = np.array([rng.uniform(-0.28, 0.28), rng.uniform(0.32, 0.75)])
        else:
            # Optional box bounds for a restricted-workspace experiment.
            tray_candidate = np.array([rng.uniform(*bounds["x"]), rng.uniform(*bounds["y"])])
            cube_candidate = np.array([rng.uniform(*bounds["x"]), rng.uniform(*bounds["y"])])

        # Optional radial cap. With bounds={"max_radius": ...}, the original
        # Franka ranges above are retained and only farther target pairs are rejected.
        if bounds is not None and "max_radius" in bounds:
            if max(np.linalg.norm(tray_candidate), np.linalg.norm(cube_candidate)) > bounds["max_radius"]:
                continue

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
    csv_log_path = Path(f"{ROBOT_TO_RUN}_ik_log_mink_{timestamp_str}.csv")
    logs = []

    print("=" * 60)
    print(f"{cfg.name} mink QP-IK Pick & Place - {N_EPISODES} Episodes (2x Speed)")
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
                    model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, cfg.home_angles, approach_pos, grasp_rot,
                    home_angles=cfg.home_angles
                )
                if not ok_app:
                    # Retry with orientation_weight RAISED to full parity (1.0),
                    # not lowered -- see note above on why lowering it further
                    # made the already-starved orientation task worse.
                    q_app, ok_app, pe, re, iters2 = solve_ik(
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, cfg.home_angles, approach_pos, grasp_rot, orientation_weight=1.0,
                        home_angles=cfg.home_angles
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
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_app, grasp_pos, grasp_rot,
                        home_angles=cfg.home_angles
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
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_gr, lift_pos, grasp_rot,
                        home_angles=cfg.home_angles
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
                        model, data, site_id, arm_qpos_addrs, arm_dof_addrs, joint_limits, q_lift, drop_pos, drop_rot,
                        home_angles=cfg.home_angles
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
