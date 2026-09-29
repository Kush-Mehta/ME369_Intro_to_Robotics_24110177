#!/usr/bin/env python3
import mujoco
import mujoco.viewer

# Update the XML path to your file
_xml_path = "/home/pikapika/Debojit_WS/RL-Based-Dual-Arm-Manipulation/Dual_Arm_Manipulation/robot_descriptions/robotiq_2f85_v4/mjx_2f85.xml"

if __name__ == "__main__":
    # Load the model and create the simulation data
    model = mujoco.MjModel.from_xml_path(_xml_path)
    data = mujoco.MjData(model)

    # Launch the MuJoCo viewer in passive mode
    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        # Run the simulation loop
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
