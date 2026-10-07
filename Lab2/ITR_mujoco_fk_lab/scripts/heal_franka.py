"""
Forward Kinematics lab:  Addverb HEAL (6 DOF)  or  Franka Panda (7 DOF)

Set ROBOT = 'h'  to open HEAL
Set ROBOT = 'f'  to open Franka

Run from the scripts/ folder:
    python3 fk_heal_franka.py

What you will see
─────────────────
• A MuJoCo window with the robot.
• Right panel → "Control" → one slider per joint.  Move a slider → arm moves.
• Franka: actuator8 slider opens/closes the gripper (0 = closed, 255 = open).
• Top-left overlay:
      MuJoCo  x  y  z   (simulator's end-effector position)
      DH FK   x  y  z   (our DH chain result)
      Error              (distance between them, in mm)
• GREEN sphere = MuJoCo end-effector
• RED   sphere = our DH result
  (They should sit on top of each other if DH is correct.)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHOOSE YOUR ROBOT HERE
#   'h'  →  Addverb HEAL  (6 DOF)
#   'f'  →  Franka Panda  (7 DOF)
# ─────────────────────────────────────────────────────────────────────────────
ROBOT = 'f'

# ─────────────────────────────────────────────────────────────────────────────
import os, time
import numpy as np
import mujoco
import mujoco.viewer

# ─────────────────────────────────────────────────────────────────────────────
# Paths  –  script lives in  .../scripts/ ,  xml files in  .../robot_descriptions/
# ─────────────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(HERE, "..", "robot_descriptions")
if not os.path.isdir(ROBOT_DIR):
    ROBOT_DIR = os.path.join(HERE, "robot_descriptions")

# ─────────────────────────────────────────────────────────────────────────────
# DH TABLES
# Row = one joint:  (theta_offset, d, a, alpha)
# Standard DH convention:  T = Rz(θ+offset) · Tz(d) · Tx(a) · Rx(alpha)
# ─────────────────────────────────────────────────────────────────────────────
PI = np.pi

#          theta_off        d          a       alpha
HEAL_DH = [
    ( PI,           0.1498,   0.0,  1.570792),   # joint 1 – base yaw
    ( PI/2,         0.0875,   0.3,  PI      ),   # joint 2 – shoulder
    ( 0.000796,     0.08737,  0.0,  1.57    ),   # joint 3 – elbow
    ( PI,           0.37736,  0.0,  0.50951 ),   # joint 4 – wrist 1
    (-1.571593,     0.000101, 0.0,  1.570792),   # joint 5 – wrist 2
    ( 0.0,         -0.1227,   0.0,  0.0     ),   # joint 6 – wrist 3
]
HEAL_BASE = [0.0, 0.0, 0.171]

#            theta_off  d        a        alpha
FRANKA_DH = [
    (0.0,  0.333,   0.0,     -PI/2),   # joint 1
    (0.0,  0.0,     0.0,      PI/2),   # joint 2
    (0.0,  0.316,   0.0825,   PI/2),   # joint 3
    (0.0,  0.0,    -0.0825,  -PI/2),   # joint 4
    (0.0,  0.384,   0.0,      PI/2),   # joint 5
    (0.0,  0.0,     0.088,    PI/2),   # joint 6
    (0.0,  0.107,   0.0,      0.0 ),   # joint 7 (flange / hand)
]
FRANKA_BASE = [-0.3, 0.0, 0.8]   # link0 pos from panda.xml

# ─────────────────────────────────────────────────────────────────────────────
# Robot configs
# ─────────────────────────────────────────────────────────────────────────────
ROBOTS = {
    'h': {
        "name"        : "HEAL (6 DOF)",
        "xml"         : "single_arm_heal_effort_actuation_rs_mj.xml",
        "ee_body"     : "end_effector",
        "dh"          : HEAL_DH,
        "base"        : HEAL_BASE,
        "arm_n"       : 6,     # number of arm joints (= number of DH rows)
        "has_gripper" : False,
    },
    'f': {
        "name"        : "FRANKA (7 DOF)",
        "xml"         : os.path.join("franka", "scene_fixed.xml"),
        "ee_body"     : "hand",
        "dh"          : FRANKA_DH,
        "base"        : FRANKA_BASE,
        "arm_n"       : 7,     # joints 1-7 are the arm; actuator8 is the gripper
        "has_gripper" : True,
        # qpos indices of the two finger joints (from panda.xml: finger_joint1, finger_joint2)
        "finger_qpos" : [7, 8],
        # gripper actuator index (0-based)
        "gripper_act" : 7,
        # gripper gain: ctrl value * gain = finger displacement in metres
        # from panda.xml: gainprm="0.01568627451"
        "gripper_gain": 0.01568627451,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – 4×4 DH matrix for one joint
# ─────────────────────────────────────────────────────────────────────────────
def dh_matrix(theta, d, a, alpha):
    """
    Standard DH transformation matrix.

    theta = joint angle + offset (rotation about previous z-axis)
    d     = link offset          (translation along previous z-axis)
    a     = link length          (translation along new x-axis)
    alpha = link twist           (rotation about new x-axis)

    Expanding  Rz(θ)·Tz(d)·Tx(a)·Rx(α)  gives:
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,  -st*ca,   st*sa,  a*ct],
        [st,   ct*ca,  -ct*sa,  a*st],
        [0.0,     sa,      ca,     d],
        [0.0,    0.0,     0.0,   1.0],
    ])

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – chain all joint matrices → end-effector (x, y, z)
# ─────────────────────────────────────────────────────────────────────────────
def forward_kinematics(dh_table, base_xyz, joint_angles):
    """
    Multiply one 4×4 matrix per joint (like following directions step by step).
    Returns the (x, y, z) position of the end effector.
    """
    T = np.eye(4)
    T[0, 3], T[1, 3], T[2, 3] = base_xyz   # place chain at the robot base

    for (theta_off, d, a, alpha), q in zip(dh_table, joint_angles):
        T = T @ dh_matrix(q + theta_off, d, a, alpha)

    return T[:3, 3]   # last column = position

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_robot(cfg):
    """Load XML, fix slider ranges so every joint has a usable range."""
    model = mujoco.MjModel.from_xml_path(os.path.join(ROBOT_DIR, cfg["xml"]))
    data  = mujoco.MjData(model)
    n     = cfg["arm_n"]

    # Copy each arm joint's range onto its actuator (HEAL has no ctrlrange set)
    for i in range(n):
        model.actuator_ctrllimited[i] = 1
        model.actuator_ctrlrange[i]   = model.jnt_range[i]

    return model, data


def draw_sphere(scene, slot, xyz, radius, rgba):
    """Draw a coloured sphere in the 3-D view."""
    mujoco.mjv_initGeom(
        scene.geoms[slot],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [radius, 0.0, 0.0],
        xyz,
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )


def print_overlay(cfg, mujoco_xyz, dh_xyz, error_mm, f1=0.0, f2=0.0):
    """
    Print a live-updating overlay to the terminal using ANSI escape codes.
    This is the fallback when mjr_overlay / set_texts are unavailable.
    Moves cursor to top-left and rewrites the block every frame.
    """
    lines = [
        f"\033[1m{cfg['name']}\033[0m",
        "",
        f"  MuJoCo   x={mujoco_xyz[0]:+.4f}   y={mujoco_xyz[1]:+.4f}   z={mujoco_xyz[2]:+.4f}",
        f"  DH FK    x={dh_xyz[0]:+.4f}   y={dh_xyz[1]:+.4f}   z={dh_xyz[2]:+.4f}",
        f"  Error    {error_mm:.4f} mm",
    ]
    if cfg["has_gripper"]:
        lines += ["", f"  Gripper  {f1*1000:.1f} mm  /  {f2*1000:.1f} mm"]

    # \033[H = move cursor home (top-left), \033[J = clear to end of screen
    print("\033[H\033[J" + "\n".join(lines), end="", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if ROBOT not in ROBOTS:
        print("ERROR: set ROBOT = 'h' (HEAL) or ROBOT = 'f' (Franka) at the top of the file.")
        return

    cfg         = ROBOTS[ROBOT]
    model, data = load_robot(cfg)
    n           = cfg["arm_n"]

    print(f"Launching {cfg['name']} …")
    print("→ Right panel → Control → drag sliders to move joints")
    if cfg["has_gripper"]:
        print("→ actuator8 slider = gripper  (0 = closed,  255 = open)")
    print("→ Top-left overlay: MuJoCo vs DH position + error")
    print("→ GREEN sphere = MuJoCo end-effector   RED sphere = DH result")

    # Shared state written each physics tick, read by the render callback.
    state = {
        "mujoco_xyz" : np.zeros(3),
        "dh_xyz"     : np.zeros(3),
        "error_mm"   : 0.0,
        "f1"         : 0.0,
        "f2"         : 0.0,
    }

    # ── Render callback: called by the viewer on every frame ──────────────
    # mjr_overlay draws text directly onto the OpenGL framebuffer – this is
    # the only approach that reliably works in the passive viewer across all
    # MuJoCo 3.x builds (set_texts has an unstable API).
    def render_callback(model, data):
        mj_xyz = state["mujoco_xyz"]
        dk_xyz = state["dh_xyz"]
        err    = state["error_mm"]

        left_col = (
            f"{cfg['name']}\n"
            f"\n"
            f"MuJoCo\n"
            f"DH FK\n"
            f"Error"
        )
        right_col = (
            f"\n"
            f"\n"
            f"x={mj_xyz[0]:+.4f}  y={mj_xyz[1]:+.4f}  z={mj_xyz[2]:+.4f}\n"
            f"x={dk_xyz[0]:+.4f}  y={dk_xyz[1]:+.4f}  z={dk_xyz[2]:+.4f}\n"
            f"{err:.4f} mm"
        )
        if cfg["has_gripper"]:
            left_col  += f"\n\nGripper (each finger)"
            right_col += f"\n\n{state['f1']*1000:.1f} mm  /  {state['f2']*1000:.1f} mm"

        # mjr_overlay needs the current OpenGL viewport + context.
        # The passive viewer exposes them via viewer.ctx and viewer.viewport
        # when called from inside the render callback.
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewer.viewport,
            left_col,
            right_col,
            viewer.ctx,
        )

    # ── Launch viewer – try render_callback first (MuJoCo >= 3.1),        ──
    # ── fall back to plain launch_passive (older builds).                  ──
    # ── WAYLAND FIX: force X11 backend so launch_passive doesn't freeze.   ──
    os.environ.setdefault("MUJOCO_GL", "glx")        # use GLX/X11, not EGL/Wayland
    os.environ.setdefault("DISPLAY", ":0")            # ensure X display is set

    try:
        viewer = mujoco.viewer.launch_passive(
            model, data,
            render_callback=render_callback,
        )
        use_terminal_overlay = False
    except TypeError:
        # Old MuJoCo: launch_passive does not accept render_callback.
        # Fall back to plain passive viewer + terminal overlay.
        viewer = mujoco.viewer.launch_passive(model, data)
        use_terminal_overlay = True

    # Start with zero user geoms; we reset and redraw every frame.
    viewer.user_scn.ngeom = 0

    while viewer.is_running():

        # ── Arm joints: read slider, apply directly as joint angle ─────────
        # We DON'T run physics for the arm  (no mj_step for joints 0-6).
        # Reason: mj_step would let gravity pull the arm down and the pose
        # would not match the slider exactly, making the comparison confusing.
        arm_angles = data.ctrl[:n].copy()
        data.qpos[:n] = arm_angles

        # ── Gripper: needs physics (it's force-driven via a tendon) ─────────
        # The Franka gripper is NOT position-controlled directly.
        # actuator8 applies a force on the "split" tendon.
        # Running mj_step moves the fingers to wherever that force pushes them.
        if cfg["has_gripper"]:
            # Run a few physics steps so the gripper catches up to the slider.
            # The arm joints are re-applied after each step so they don't drift.
            for _ in range(5):
                data.qpos[:n] = data.ctrl[:n].copy()   # freeze arm in place
                mujoco.mj_step(model, data)             # physics: moves gripper
            # Read back actual finger positions and show them
            fq = cfg["finger_qpos"]
            state["f1"] = data.qpos[fq[0]]   # metres, 0 = closed, 0.04 = open
            state["f2"] = data.qpos[fq[1]]
        else:
            mujoco.mj_forward(model, data)

        # ── MuJoCo's end-effector position ─────────────────────────────────
        mujoco_xyz = data.body(cfg["ee_body"]).xpos.copy()

        # ── Our DH chain result ─────────────────────────────────────────────
        dh_xyz = forward_kinematics(cfg["dh"], cfg["base"], arm_angles)

        # ── Error ──────────────────────────────────────────────────────────
        error_mm = np.linalg.norm(mujoco_xyz - dh_xyz) * 1000.0

        # ── Update shared state (render_callback reads this) ───────────────
        state["mujoco_xyz"] = mujoco_xyz
        state["dh_xyz"]     = dh_xyz
        state["error_mm"]   = error_mm

        # ── Terminal overlay (old MuJoCo fallback) ─────────────────────────
        if use_terminal_overlay:
            print_overlay(cfg, mujoco_xyz, dh_xyz, error_mm,
                          state["f1"], state["f2"])

        # ── Marker spheres ─────────────────────────────────────────────────
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            draw_sphere(viewer.user_scn, 0, mujoco_xyz, 0.015, [0, 1, 0, 0.8])
            draw_sphere(viewer.user_scn, 1, dh_xyz,     0.025, [1, 0, 0, 0.4])
            viewer.user_scn.ngeom = 2
        viewer.sync()

        time.sleep(0.02)


if __name__ == "__main__":
    main()
