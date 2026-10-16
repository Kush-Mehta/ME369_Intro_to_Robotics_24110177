"""
launch_table_scene.py
=====================
Loads arm_gripper_with_table.xml, places the cube at a RANDOM position on
the table surface, then opens the MuJoCo passive viewer.

Arm movement is smooth: the script linearly interpolates data.ctrl[] (the
position-servo setpoints) over a configurable duration — no Python PID, no
actuator tuning needed.  MuJoCo's own position-servo actuators handle the
physics; we just tell them where to go, step by step.

Usage:
    python3 launch_table_scene.py
    python3 launch_table_scene.py --seed 42        # reproducible cube position
    python3 launch_table_scene.py --demo            # runs a smooth arm demo

Table geometry (matches arm_gripper_with_table.xml)
────────────────────────────────────────────────────
  Orientation   : long axis along Y  (90° flip from original)
  Table body pos: x=0.00  y=0.55  (world frame)
  Half-sizes    : x=0.30  y=0.40  z=0.02  (top slab)
  Leg height    : 0.16 m  (legs are 0.08 m half-height)
  Top surface z : 0.18 m  (= 0.16 + 0.02)
  Cube half     : 0.02 m
  Cube centre z : 0.20 m  (= 0.18 + 0.02)

  Valid cube X range : [0.00 - 0.30 + 0.02 + MARGIN, 0.00 + 0.30 - 0.02 - MARGIN]
  Valid cube Y range : [0.55 - 0.40 + 0.02 + MARGIN, 0.55 + 0.40 - 0.02 - MARGIN]
"""

import os
import sys
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seed",  type=int,  default=None,
                    help="Random seed for reproducible cube placement")
parser.add_argument("--demo",  action="store_true",
                    help="Run a smooth arm movement demo after launch")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
XML_PATH   = os.path.join(REPO_ROOT, "robot_descriptions",
                           "arm_gripper_with_table.xml")

if not os.path.exists(XML_PATH):
    sys.exit(f"[ERROR] XML not found: {XML_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# TABLE GEOMETRY — keep in sync with arm_gripper_with_table.xml
# Single source of truth: edit these values, nowhere else.
# ─────────────────────────────────────────────────────────────────────────────
TABLE_BODY_X   =  0.00   # table body pos x in world
TABLE_BODY_Y   =  0.55   # table body pos y in world
TABLE_HALF_WID =  0.30   # half-size along X (short axis)
TABLE_HALF_LEN =  0.40   # half-size along Y (long axis, 90° flip)
TABLE_TOP_Z    =  0.18   # z of table top surface
CUBE_HALF      =  0.02   # half-size of the cube

MARGIN = 0.03  # keep cube this far from edges

x_lo = TABLE_BODY_X - TABLE_HALF_WID + CUBE_HALF + MARGIN
x_hi = TABLE_BODY_X + TABLE_HALF_WID - CUBE_HALF - MARGIN
y_lo = TABLE_BODY_Y - TABLE_HALF_LEN + CUBE_HALF + MARGIN
y_hi = TABLE_BODY_Y + TABLE_HALF_LEN - CUBE_HALF - MARGIN
cube_z = TABLE_TOP_Z + CUBE_HALF   # 0.20

cube_x = float(rng.uniform(x_lo, x_hi))
cube_y = float(rng.uniform(y_lo, y_hi))
cube_yaw = float(rng.uniform(-np.pi, np.pi))  # random rotation about Z

print(f"XML      : {XML_PATH}")
print(f"Cube pos : x={cube_x:.3f}  y={cube_y:.3f}  z={cube_z:.3f}  "
      f"yaw={np.degrees(cube_yaw):.1f} deg")
print(f"X range  : [{x_lo:.3f}, {x_hi:.3f}]")
print(f"Y range  : [{y_lo:.3f}, {y_hi:.3f}]")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL + PLACE CUBE
# ─────────────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# Locate cube free-joint
cube_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
if cube_body_id < 0:
    sys.exit("[ERROR] Body 'cube' not found in model.")

cube_joint_id = model.body_jntadr[cube_body_id]
if cube_joint_id < 0 or model.jnt_type[cube_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
    sys.exit("[ERROR] Body 'cube' does not have a freejoint.")

qpos_addr = model.jnt_qposadr[cube_joint_id]

# Free-joint layout: [x, y, z,  qw, qx, qy, qz]
# Orientation: upright, with a random yaw about Z (quat = [cos(y/2),0,0,sin(y/2)])
data.qpos[qpos_addr + 0] = cube_x
data.qpos[qpos_addr + 1] = cube_y
data.qpos[qpos_addr + 2] = cube_z
data.qpos[qpos_addr + 3] = np.cos(cube_yaw / 2.0)
data.qpos[qpos_addr + 4] = 0.0
data.qpos[qpos_addr + 5] = 0.0
data.qpos[qpos_addr + 6] = np.sin(cube_yaw / 2.0)

mujoco.mj_forward(model, data)

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTH MOVEMENT HELPER
# ─────────────────────────────────────────────────────────────────────────────
# The position-servo actuators in the XML accept a target angle in data.ctrl[i].
# We smoothly interpolate ctrl from the current position to the target over
# `duration` seconds.  MuJoCo drives the joints — no Python PID needed.

ARM_JOINT_NAMES  = ["joint_1", "joint_2", "joint_3",
                     "joint_4", "joint_5", "joint_6"]
ARM_ACTUATOR_NAMES = ["turret", "shoulder", "elbow",
                       "wrist_1", "wrist_2", "wrist_3"]

# Pre-compute actuator ctrl indices
actuator_ids = []
for name in ARM_ACTUATOR_NAMES:
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid < 0:
        print(f"[WARNING] Actuator '{name}' not found — skipping smooth move")
        actuator_ids = []
        break
    actuator_ids.append(aid)


def smooth_move(viewer, target_angles_rad, duration=2.0):
    """
    Smoothly move arm to target joint angles over `duration` seconds.
    Uses cosine easing so it starts/ends gently.
    
    target_angles_rad : list/array of 6 floats (radians), one per arm joint
    duration          : seconds of real-time motion
    """
    if not actuator_ids:
        return

    start_ctrl = np.array([data.ctrl[i] for i in actuator_ids], dtype=float)
    target      = np.array(target_angles_rad, dtype=float)

    dt        = model.opt.timestep          # simulation timestep (0.002 s)
    steps     = max(1, int(duration / dt))  # total sim steps

    for step in range(steps):
        if not viewer.is_running():
            return
        # Cosine easing: smooth start and end
        t = step / steps
        alpha = 0.5 * (1.0 - np.cos(np.pi * t))   # 0 → 1, smooth

        interp = start_ctrl + alpha * (target - start_ctrl)
        for i, aid in enumerate(actuator_ids):
            data.ctrl[aid] = interp[i]

        mujoco.mj_step(model, data)
        viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCH VIEWER
# ─────────────────────────────────────────────────────────────────────────────
print("\nOpening MuJoCo viewer …  (close window or Ctrl+C to exit)")
print("Controls: Ctrl+drag=grab  Scroll=zoom  Right-click=context")
if args.demo:
    print("Demo mode: arm will perform a smooth choreographed movement.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.sync()

    if args.demo:
        # ── Smooth arm demo ──────────────────────────────────────────────
        # Poses defined as [j1, j2, j3, j4, j5, j6] in radians
        HOME      = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
        REACH_OUT = [0.0,  0.5, -0.5,  0.0,  0.3,  0.0]
        SCAN_LEFT = [1.0,  0.4, -0.4,  0.2,  0.2,  0.5]
        SCAN_RIGHT= [-1.0, 0.4, -0.4, -0.2,  0.2, -0.5]
        HOVER     = [0.3,  0.8, -0.6,  0.0,  0.5,  0.0]

        print("[DEMO] Moving to reach-out pose …")
        smooth_move(viewer, REACH_OUT, duration=2.5)
        time.sleep(0.3)

        print("[DEMO] Scanning left …")
        smooth_move(viewer, SCAN_LEFT, duration=2.0)
        time.sleep(0.3)

        print("[DEMO] Scanning right …")
        smooth_move(viewer, SCAN_RIGHT, duration=3.0)
        time.sleep(0.3)

        print("[DEMO] Hovering over table …")
        smooth_move(viewer, HOVER, duration=2.5)
        time.sleep(0.3)

        print("[DEMO] Returning to home …")
        smooth_move(viewer, HOME, duration=2.5)
        print("[DEMO] Done — viewer remains open.\n")

    # Keep viewer alive
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
