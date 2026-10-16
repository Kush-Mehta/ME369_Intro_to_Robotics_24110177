#!/usr/bin/env python3
"""
Simple MuJoCo demo: load ONE HEAL arm and drive a couple of joints with constant
joint velocities using your JointVelocityController (D-controller).

Folder structure (relative to this script):
  ../../robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml
  ../../utils/mj_velocity_control/mj_velocity_ctrl.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# --- Resolve paths (portable across machines as long as repo structure is same) ---
SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(REPO_ROOT)

# Your controller (already implemented by you)
from utils.mj_velocity_control.mj_velocity_ctrl import JointVelocityController

def main():
    # Path to a single HEAL arm (effort-actuated, MuJoCo-ready)
    robot_xml = os.path.join(REPO_ROOT, "robot_descriptions", "single_arm_heal_effort_actuation_rs_mj.xml")

    # --- Load model & data ---
    # For one robot, you can directly load the XML; no need for MjSpec composition.
    model = mujoco.MjModel.from_xml_path(robot_xml)
    data  = mujoco.MjData(model)

    # Run a forward pass to populate bias terms etc.
    mujoco.mj_forward(model, data)

    # --- Controller: simple D controller on joint velocities ---
    controller = JointVelocityController(model, data, kd=2.0)

    # Constant velocity targets for a couple of actuators (edit indices as you like)
    v_target = np.zeros(model.nu)
    # Example: spin base and elbow a bit
    if model.nu >= 1: v_target[0] =  0.2    # actuator 0
    if model.nu >= 4: v_target[3] = -0.15    # actuator 3
    controller.set_velocity_target(v_target)

    # Register control callback so MuJoCo calls it every step
    mujoco.set_mjcb_control(controller.control_callback)

    # --- Launch viewer and simulate ---
    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_time = time.time()
        while viewer.is_running():
            # Step the simulation at (approximately) real time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Step at a fixed number of sub-steps per frame for stability
            for _ in range(5):
                mujoco.mj_step(model, data)

            viewer.sync()

if __name__ == "__main__":
    main()
