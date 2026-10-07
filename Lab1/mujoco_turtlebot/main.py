import pathlib
import time
import numpy as np
import mujoco
import mujoco.viewer

# Path to the scene file
MODEL_PATH = pathlib.Path("robotis_tb3/scene_turtlebot3_waffle_pi.xml")

# TurtleBot3 Waffle Pi Kinematic Parameters
WHEEL_RADIUS = 0.033  # meters
TRACK_WIDTH = 0.287   # meters

# Target Velocity Commands
CMD_LIN_VEL = 0.22    # m/s
CMD_ANG_VEL = 1.20    # rad/s

# Global velocity state
target_lin_vel = 0.0
target_ang_vel = 0.0

# GLFW Keycodes
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262


def key_callback(keycode: int):
    global target_lin_vel, target_ang_vel

    char = chr(keycode).lower() if 0 <= keycode < 128 else ""

    if keycode == KEY_UP or char == "i":
        target_lin_vel = CMD_LIN_VEL
        target_ang_vel = 0.0
    elif keycode == KEY_DOWN or char == "k":
        target_lin_vel = -CMD_LIN_VEL
        target_ang_vel = 0.0
    elif keycode == KEY_LEFT or char == "j":
        target_ang_vel = CMD_ANG_VEL
    elif keycode == KEY_RIGHT or char == "l":
        target_ang_vel = -CMD_ANG_VEL
    elif keycode == 32 or char in ("space", "x", "s"):
        target_lin_vel = 0.0
        target_ang_vel = 0.0


def main():
    global target_lin_vel, target_ang_vel

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    base_body_id = model.body("base").id

    # SETTLE CONTACTS: Step simulation 500 times so wheels settle on floor
    mujoco.mj_resetData(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start_time = time.time()

        while viewer.is_running():
            # Correct Differential Drive Kinematics (Forward = Positive)
            v_left = target_lin_vel - (target_ang_vel * TRACK_WIDTH / 2.0)
            v_right = target_lin_vel + (target_ang_vel * TRACK_WIDTH / 2.0)

            data.ctrl[0] = v_left / WHEEL_RADIUS
            data.ctrl[1] = v_right / WHEEL_RADIUS

            sim_target = time.time() - start_time
            while data.time < sim_target:
                mujoco.mj_step(model, data)

            # Extract 3x3 Rotation Matrix R and position
            R = data.xmat[base_body_id].reshape(3, 3).copy()
            pos = data.xpos[base_body_id]

            # Threshold micro-vibrations (< 0.005) to eliminate display jitter
            R[np.abs(R) < 0.005] = 0.0

            viewer.user_scn.ngeom = 0

            # Clean formatting
            lines = [
                "Rotation Matrix R:",
                f"| {R[0,0]:+0.2f}  {R[0,1]:+0.2f}  {R[0,2]:+0.2f} |",
                f"| {R[1,0]:+0.2f}  {R[1,1]:+0.2f}  {R[1,2]:+0.2f} |",
                f"| {R[2,0]:+0.2f}  {R[2,1]:+0.2f}  {R[2,2]:+0.2f} |",
                f"v: {target_lin_vel:+.2f} m/s | w: {target_ang_vel:+.2f} rad/s",
            ]

            base_z = pos[2] + 0.65
            for idx, line_text in enumerate(lines):
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    size=np.zeros(3),
                    pos=[pos[0], pos[1], base_z - (idx * 0.08)],
                    mat=np.eye(3).flatten(),
                    rgba=[0.2, 1.0, 0.4, 1.0],
                )
                geom.label = line_text
                viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.016)


if __name__ == "__main__":
    main()