"""Move to OUR home pose the short way round - safe from any pose.

    ros2 run omx_calibration go_home                                  # open gripper, lift 5 cm, home
    ros2 run omx_calibration go_home --ros-args -p keep_gripper:=true # don't open the gripper

1. opens the gripper (so a peg it is stuck on is let go),
2. lifts the gripper straight up 5 cm (so it doesn't drag anything sideways),
3. joint move to home, base the short way round.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import kinematics as K
from . import store
from .wizard import ARM, G_OPEN, PARK_DEFAULT

try:
    from control_msgs.action import GripperCommand
except ImportError:  # pragma: no cover
    GripperCommand = None


class GoHome(Node):
    def __init__(self):
        super().__init__('omx_go_home')
        self.declare_parameter('time_s', 5.0)
        self.declare_parameter('keep_gripper', False)
        self.declare_parameter('lift_m', 0.05)
        self.q = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.grip_ac = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd') \
            if GripperCommand else None

    def _js(self, m):
        d = dict(zip(m.name, m.position))
        if all(j in d for j in ARM):
            self.q = np.array([d[j] for j in ARM])

    def spin_for(self, secs):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def send(self, q, T):
        msg = JointTrajectory()
        msg.joint_names = ARM
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(T)
        pt.time_from_start.nanosec = int((T - int(T)) * 1e9)
        msg.points = [pt]
        self.pub.publish(msg)
        self.spin_for(T + 0.8)


def main():
    rclpy.init()
    n = GoHome()
    n.spin_for(0.5)
    t0 = time.monotonic()
    while n.q is None and time.monotonic() - t0 < 5.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if n.q is None:
        print('no /joint_states - is the arm bringup running?')
        return
    print(f'now  [rad]: {np.round(n.q, 3).tolist()}')

    # 1. let go of anything
    if not n.get_parameter('keep_gripper').value and n.grip_ac is not None:
        if n.grip_ac.wait_for_server(timeout_sec=2.0):
            g = GripperCommand.Goal()
            g.command.position = float(G_OPEN)
            n.grip_ac.send_goal_async(g)
            print('gripper: opening')
            n.spin_for(1.5)
        else:
            print('gripper action not available - not opening')

    # 2. straight up first (keep the current gripper pitch)
    p, phi = K.fk(n.q)
    lift = float(n.get_parameter('lift_m').value)
    q_up, ok = K.ik(p + np.array([0.0, 0.0, lift]), phi, q0=n.q)
    if ok and np.max(np.abs(q_up - n.q)) < 1.0:
        print(f'lifting {lift*100:.0f} cm straight up')
        n.send(q_up, 1.5)
    else:
        print('cannot lift straight up from here - going home directly')

    # 3. home, base the short way round
    home = np.array(store.load().get('poses', {}).get('park_joints', PARK_DEFAULT), float)
    home[0] = K.near(home[0], float(n.q[0]))
    T = max(float(n.get_parameter('time_s').value), float(np.max(np.abs(home - n.q))) / 0.4)
    print(f'home [rad]: {np.round(home, 3).tolist()}   ({T:.1f} s)')
    n.send(home, T)
    print(f'reached [rad]: {np.round(n.q, 3).tolist()}  '
          f'(max error {math.degrees(float(np.max(np.abs(n.q - home)))):.1f} deg)')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
