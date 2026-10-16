import numpy as np

class JointVelocityController:
    """
    A derivative-only (D) controller for all actuators,
    while applying gravity compensation to all joints.
    """
    def __init__(self, model, data, kd=50.0, gripper_ids=None):
        """
        :param model: The MuJoCo MjModel
        :param data: The MuJoCo MjData
        :param kd: Derivative gain (scalar or array). If scalar, apply the same gain to all actuators.
        """
        
        self.model = model
        self.data = data

        # Map from actuators to DOF indices
        self.joint_ids = self.model.actuator_trnid[:, 0]   # Joint IDs controlled by actuators
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]

        self.num_actuators = self.model.nu  # Number of actuators

        # Store KD as scalar or array
        if np.isscalar(kd):
            self.kd = np.full(self.num_actuators, kd, dtype=float)
        else:
            self.kd = np.array(kd, dtype=float)
            assert len(self.kd) == self.num_actuators, \
                   "kd array must match the number of actuators"

        # Which actuator indices to leave alone (grippers)
        self.gripper_ids = set(gripper_ids) if gripper_ids is not None else set()
        
        # Default velocity targets
        self.v_targets = np.zeros(self.num_actuators, dtype=float)

        # Trajectory function (optional)
        self.trajectory_function = None

    def set_velocity_target(self, v_des):
        """
        Set a constant velocity target (if not using a trajectory).
        :param v_des: List or array of desired velocities, matching the number of actuators.
        """
        v_des = np.array(v_des, dtype=float)
        assert len(v_des) == self.num_actuators, \
               "v_des array must match the number of actuators"
        self.v_targets = v_des

    def set_velocity_trajectory(self, trajectory_function):
        """
        Set a trajectory function to dynamically update velocity targets.
        :param trajectory_function: A function that takes time `t` as input and returns velocity targets.
        """
        self.trajectory_function = trajectory_function

    def control_callback(self, model, data):
        """
        Applies gravity compensation and D-control to all non-gripper joints,
        and returns the applied torques for logging.
        """
        applied_torques = np.zeros(self.num_actuators)

        # Update target velocities from trajectory, if provided
        if self.trajectory_function:
            self.v_targets = self.trajectory_function(data.time)

        # Clear all control signals (for safety)
        for i in range(self.num_actuators):
            if i not in self.gripper_ids:
                data.ctrl[i] = 0.0

        # Apply control law: gravity + D term
        for i in range(self.num_actuators):
            if i in self.gripper_ids:
                continue

            dof_i = self.dof_indices[i]
            gc = data.qfrc_bias[dof_i]  # gravity compensation
            v_actual = data.qvel[dof_i]
            v_target = self.v_targets[i]
            torque_d = -self.kd[i] * (v_actual - v_target)

            tau = gc + torque_d
            data.ctrl[i] = tau
            applied_torques[i] = tau

        return applied_torques

class JointPositionController:
    """
    A proportional (P) controller for MuJoCo actuators,
    with gravity compensation based on qfrc_bias.
    """
    def __init__(self, model, data, kp=5.0, gripper_ids=None):
        """
        :param model: The MuJoCo MjModel
        :param data: The MuJoCo MjData
        :param kp: Proportional gain (scalar or array). If scalar, same gain is used for all actuators.
        """
        self.model = model
        self.data = data

        # Map actuator to corresponding DOF (joint velocity index)
        self.joint_ids = self.model.actuator_trnid[:, 0]  # Joint IDs actuated
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]  # DOF indices in qpos/qvel

        self.num_actuators = self.model.nu

        # Store KP gains
        if np.isscalar(kp):
            self.kp = np.full(self.num_actuators, kp, dtype=float)
        else:
            self.kp = np.array(kp, dtype=float)
            assert len(self.kp) == self.num_actuators, \
                "kp array must match the number of actuators"

        # Which actuator indices to leave alone (grippers)
        self.gripper_ids = set(gripper_ids) if gripper_ids is not None else set()
        # Default position targets

        # Optional time-varying trajectory
        self.trajectory_function = None

    def set_position_target(self, q_des):
        """
        Set constant position targets.
        :param q_des: List or array of desired joint positions (one per actuator)
        """
        q_des = np.array(q_des, dtype=float)
        assert len(q_des) == self.num_actuators, \
            "q_des array must match the number of actuators"
        self.q_targets = q_des

    def set_position_trajectory(self, trajectory_function):
        """
        Set a function that returns time-varying joint positions.
        :param trajectory_function: Callable with time input, returns position targets.
        """
        self.trajectory_function = trajectory_function

    def control_callback(self, model, data):
        """
        Compute control torques: gravity compensation + P-control for joint position error.
        """
        if self.trajectory_function:
            self.q_targets = self.trajectory_function(data.time)

        # clear any old commands
        data.ctrl[:] = 0

        for i in range(self.num_actuators):
            if i in self.gripper_ids:
                continue
            dof_i    = self.dof_indices[i]
            q_actual = data.qpos[self.joint_ids[i]]
            q_target = self.q_targets[i]
            error    = q_target - q_actual
            # gravity + P-term
            data.ctrl[i] = data.qfrc_bias[dof_i] + self.kp[i] * error
