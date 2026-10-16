#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# This script constructs a composite MuJoCo model by attaching a Robotiq 2F-85 
# gripper to a single-arm HEAL robot using MJCF files. The combined model is 
# then visualized using MuJoCo's passive viewer, with gravity compensation applied.
#
# The robot and gripper XML files are loaded using **relative paths** resolved 
#    from the script location, making this script portable and independent of the 
#    machine-specific absolute paths.
#
# Gravity compensation is handled using the feedforward bias force term
#    `qfrc_bias`, ensuring the robot stays in equilibrium during passive viewing.
#    Gripper actuators are skipped in the gravity-comp loop so they can be driven
#    explicitly without being overwritten.
#
# The gripper will automatically open and close every 2 seconds as an example
#    of time-based control using the simulation clock (`data.time`).
#
# To run this, ensure the following directory structure exists:
#     └── robot_descriptions/
#         ├── single_arm_heal_effort_actuation_rs_mj.xml
#         └── robotiq_2f85_v4/2f85.xml
# ------------------------------------------------------------------------------

import os
import mujoco
import mujoco.viewer
import numpy as np

def gripper_config():
    """
    Returns a list of gripper configs, each with:
      - actuator_id : the controller index driving the fingers
      - open_cmd     : the command value to open the gripper
      - close_cmd    : the command value to close the gripper
    You can add multiple grippers by extending this list.
    """
    return [
        {"actuator_id": 6, "open_cmd": 0.0,   "close_cmd": 255.0},
        # e.g. for a second gripper:
        # {"actuator_id": 7, "open_cmd": 0.0, "close_cmd": 200.0},
    ]


def construct_heal_with_gripper():
    """
    Loads the HEAL arm and the Robotiq 2F-85 gripper MJCF files, attaches the
    gripper at the 'right_center' site of the arm, and compiles the combined model.
    """
    # Resolve paths relative to this script location
    script_dir = os.path.dirname(__file__)
    repo_root  = os.path.abspath(os.path.join(script_dir, '..'))
    desc_dir   = os.path.join(repo_root, 'robot_descriptions')

    arm_path  = os.path.join(desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
    grip_path = os.path.join(desc_dir, 'robotiq_2f85_v4', '2f85.xml')

    arm  = mujoco.MjSpec.from_file(arm_path)
    grip = mujoco.MjSpec.from_file(grip_path)

    # Attach the gripper spec to the arm spec
    arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

    return arm.compile()


def compensate_gravity(model, data):
    """
    Applies gravity and Coriolis compensation to all actuators except those
    driving grippers. Uses `data.qfrc_bias` calculated by mj_forward().
    """
    skip = {g["actuator_id"] for g in GRIPPERS}
    for aid in range(model.nu):
        if aid in skip:
            continue
        joint_dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[joint_dof]


# build the list of gripper configs once
GRIPPERS = gripper_config()

if __name__ == '__main__':
    # Construct the combined HEAL + gripper model and data
    model = construct_heal_with_gripper()
    data  = mujoco.MjData(model)

    # Precompute bias forces for the initial pose
    mujoco.mj_forward(model, data)

    # Launch MuJoCo passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 1) Apply gravity compensation to all non-gripper actuators
            compensate_gravity(model, data)

            # 2) Use simulation time to toggle open/close every 2 seconds:
            #    - data.time % 4.0 goes 0→4 repeatedly
            #    - first 2s: open, next 2s: close
            if (data.time % 4.0) < 2.0:
                cmd_key = "open_cmd"
            else:
                cmd_key = "close_cmd"

            # 3) Send the chosen command to each gripper actuator
            for g in GRIPPERS:
                data.ctrl[g["actuator_id"]] = g[cmd_key]

            # 4) Step the simulation forward and sync the viewer
            mujoco.mj_step(model, data)
            viewer.sync()
