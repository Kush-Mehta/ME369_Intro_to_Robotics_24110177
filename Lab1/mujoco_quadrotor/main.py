import pathlib
import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = pathlib.Path("skydio_x2/scene.xml")

# Flight Speed Parameters
PITCH_MAX = 0.08       # Pitch tilt limit (~4.5 degrees)
ROLL_MAX = 0.08        # Roll tilt limit (~4.5 degrees)
YAW_RATE_MAX = 0.30    # Yaw rate limit (rad/s)
CLIMB_SPEED = 0.35     # Continuous climb/descend speed (m/s)

# Control Target States
target_pitch = 0.0
target_roll = 0.0
target_yaw_rate = 0.0
target_z = 0.5
climb_rate = 0.0
stop_requested = False
reset_requested = False

# GLFW Special Keycodes
KEY_ESC = 256
KEY_ENTER = 257
KEY_UP = 265
KEY_DOWN = 264
KEY_LEFT = 263
KEY_RIGHT = 262


def key_callback(keycode: int):
    """Callback using strictly unmapped GLFW keys (Comma/Period for Yaw)."""
    global target_pitch, target_roll, target_yaw_rate, target_z, climb_rate, stop_requested, reset_requested

    char = chr(keycode) if 0 <= keycode < 128 else ""

    # 1. ESC KEY: INSTANT RESET
    if keycode == KEY_ESC:
        reset_requested = True
        return

    # 2. ARROW KEYS: Pitch & Roll
    if keycode == KEY_UP:
        target_pitch = -PITCH_MAX
    elif keycode == KEY_DOWN:
        target_pitch = PITCH_MAX
    elif keycode == KEY_LEFT:
        target_roll = -ROLL_MAX
    elif keycode == KEY_RIGHT:
        target_roll = ROLL_MAX

    # 3. ALTITUDE CONTROL: '+' / '-'
    elif char in ("=", "+"):
        climb_rate = CLIMB_SPEED
    elif char in ("-", "_"):
        climb_rate = -CLIMB_SPEED

    # 4. YAW CONTROL: Comma (,) / Period (.)
    elif char in (",", "<"):
        target_yaw_rate = YAW_RATE_MAX   # Rotate Left
    elif char in (".", ">"):
        target_yaw_rate = -YAW_RATE_MAX  # Rotate Right

    # 5. ENTER: LOCK HOVER & LEVEL OUT
    elif keycode == KEY_ENTER:
        climb_rate = 0.0
        target_pitch = 0.0
        target_roll = 0.0
        target_yaw_rate = 0.0
        stop_requested = True


def main():
    global target_pitch, target_roll, target_yaw_rate, target_z, climb_rate, stop_requested, reset_requested

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH.resolve()}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    base_body_id = model.body("x2").id
    total_mass = model.body(base_body_id).subtreemass[0]
    num_actuators = model.nu
    hover_thrust = (total_mass * 9.81) / num_actuators

    # Actuator Mixer Matrix
    mixer = np.zeros((num_actuators, 4))
    for i in range(num_actuators):
        site_id = model.actuator_trnid[i, 0]
        site_pos = model.site_pos[site_id]
        yaw_gear = model.actuator_gear[i, 5]

        mixer[i, 0] = 1.0
        mixer[i, 1] = -np.sign(site_pos[0]) * 0.5
        mixer[i, 2] =  np.sign(site_pos[1]) * 0.5
        mixer[i, 3] =  np.sign(yaw_gear) * 0.2

    def reset_simulation():
        """Resets physics and control targets to starting state."""
        global target_pitch, target_roll, target_yaw_rate, target_z, climb_rate
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.5
        target_pitch = 0.0
        target_roll = 0.0
        target_yaw_rate = 0.0
        target_z = 0.5
        climb_rate = 0.0
        for _ in range(100):
            data.ctrl[:num_actuators] = hover_thrust
            mujoco.mj_step(model, data)

    reset_simulation()

    # Controller Gains
    kp_att, kd_att = 3.0, 0.40
    kv_trans = 0.25               # Velocity damping gain (brakes horizontal drift)
    kp_yaw, kd_yaw = 1.0, 0.10
    kp_z,   kd_z   = 14.0, 4.5

    dt = 0.016

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        start_time = time.time()

        while viewer.is_running():
            # Process ESC Reset Request
            if reset_requested:
                reset_simulation()
                reset_requested = False
                start_time = time.time()

            pos = data.xpos[base_body_id]
            vel = data.qvel[:6]
            R = data.xmat[base_body_id].reshape(3, 3).copy()
            omega = vel[3:6]

            # Convert world velocities to body frame for horizontal braking
            v_body_x = R[0, 0] * vel[0] + R[1, 0] * vel[1] + R[2, 0] * vel[2]
            v_body_y = R[0, 1] * vel[0] + R[1, 1] * vel[1] + R[2, 1] * vel[2]

            # Lock target height on Enter
            if stop_requested:
                target_z = pos[2]
                stop_requested = False

            # Update target height during active climb/descent
            if climb_rate != 0.0:
                target_z = max(0.1, target_z + climb_rate * dt)

            roll_curr = np.arctan2(R[2, 1], R[2, 2])
            pitch_curr = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))

            # Altitude PD Control
            z_error = target_z - pos[2]
            u_z = kp_z * z_error - kd_z * vel[2]

            # Attitude Control + Velocity Damping
            u_pitch = np.clip(-kp_att * (pitch_curr - target_pitch) - kd_att * omega[1] - kv_trans * v_body_x, -0.5, 0.5)
            u_roll  = np.clip(-kp_att * (roll_curr - target_roll)   - kd_att * omega[0] + kv_trans * v_body_y, -0.5, 0.5)
            u_yaw   = np.clip(-kp_yaw * (-target_yaw_rate)          - kd_yaw * omega[2], -0.2, 0.2)

            total_thrust = hover_thrust + (u_z / num_actuators)
            cmd_vector = np.array([total_thrust, u_pitch, u_roll, u_yaw])

            data.ctrl[:num_actuators] = np.clip(mixer @ cmd_vector, 0.0, 15.0)

            sim_target = time.time() - start_time
            while data.time < sim_target:
                mujoco.mj_step(model, data)

            # Suppress micro floating-point noise
            R[np.abs(R) < 0.005] = 0.0

            # Minimal HUD (Current Position, Angles, Rotation Matrix R)
            viewer.user_scn.ngeom = 0
            lines = [
                f"Pos: X={pos[0]:+.2f} Y={pos[1]:+.2f} Z={pos[2]:+.2f}",
                f"Pitch: {pitch_curr:+.2f} | Roll: {roll_curr:+.2f}",
                "Rotation Matrix R:",
                f"| {R[0,0]:+0.2f}  {R[0,1]:+0.2f}  {R[0,2]:+0.2f} |",
                f"| {R[1,0]:+0.2f}  {R[1,1]:+0.2f}  {R[1,2]:+0.2f} |",
                f"| {R[2,0]:+0.2f}  {R[2,1]:+0.2f}  {R[2,2]:+0.2f} |",
            ]

            base_z = pos[2] + 0.40
            for idx, line_text in enumerate(lines):
                geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    type=mujoco.mjtGeom.mjGEOM_LABEL,
                    size=np.zeros(3),
                    pos=[pos[0], pos[1], base_z - (idx * 0.06)],
                    mat=np.eye(3).flatten(),
                    rgba=[0.1, 0.9, 1.0, 1.0],
                )
                geom.label = line_text
                viewer.user_scn.ngeom += 1

            viewer.sync()
            time.sleep(0.016)


if __name__ == "__main__":
    main()