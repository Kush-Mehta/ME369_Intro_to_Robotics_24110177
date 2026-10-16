# ------------------------------------------------------------------------------
# This script constructs a composite MuJoCo model by attaching a Robotiq 2F-85 
# gripper to a single-arm HEAL robot using MJCF files. The combined model is 
# then visualized using MuJoCo's passive viewer, with gravity compensation applied.
#
# The robot and gripper XML files are loaded using **relative paths** resolved 
# from the script location, making this script portable and independent of the 
# machine-specific absolute paths. You can run it on **any PC** as long as the 
# folder structure within the repository remains consistent.
#
# Gravity compensation is handled using the feedforward bias force term
# `qfrc_bias`, ensuring the robot stays in equilibrium during passive viewing.
# So you wouldn't see the robot falling in simulation since the gravity compensation toreques are applied.
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

def construct_heal_with_gripper():
    """
    Constructs a MuJoCo model by attaching a Robotiq 2F-85 gripper to the
    HEAL single-arm robot. The attachment is done at the 'right_center' site.
    """
    # Resolve paths relative to this script location
    script_dir = os.path.dirname(__file__)
    repo_root  = os.path.abspath(os.path.join(script_dir, '..'))
    desc_dir   = os.path.join(repo_root, 'robot_descriptions')

    # Full paths to HEAL arm and Robotiq gripper MJCF XML files
    arm_path  = os.path.join(desc_dir, 'single_arm_heal_effort_actuation_rs_mj.xml')
    grip_path = os.path.join(desc_dir, 'robotiq_2f85_v4', '2f85.xml')

    # Load the MJCF specs for the arm and the gripper
    arm  = mujoco.MjSpec.from_file(arm_path)
    grip = mujoco.MjSpec.from_file(grip_path)

    # Attach the gripper to the HEAL arm at the specified site with a prefix
    arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

    # Compile and return the final combined model
    return arm.compile()

def compensate_gravity(model, data):
    """
    Applies feedforward control to each actuator to compensate for gravity and Coriolis effects.
    This uses the bias force vector `data.qfrc_bias`, which is populated by `mj_forward()`.
    """
    for aid in range(model.nu):
        joint_dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[joint_dof]

if __name__ == '__main__':
    # Build the combined HEAL + gripper model
    model = construct_heal_with_gripper()
    data  = mujoco.MjData(model)
    print(data.qpos)

    # Compute bias forces for the initial configuration
    mujoco.mj_forward(model, data)

    # Launch the passive MuJoCo viewer to visualize simulation without user control
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Continuously run the simulation with gravity compensation until the viewer is closed
        while viewer.is_running():
            compensate_gravity(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
