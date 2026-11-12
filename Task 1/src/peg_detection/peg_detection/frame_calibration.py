"""Find the transform from the hand-drawn foam axes to the arm's own frame (ROBOTIS link1).

    ros2 run peg_detection frame_calibration --ros-args -p colour:=green

Needs running: the peg detector (camera pose frozen) and the ROBOTIS arm bringup
(robot_state_publisher -> TF link1 -> end_effector_link).

For each pair (4+ pairs, spread over the workspace):
  1. place a peg somewhere, keep the arm away  -> [v]  (vision: peg top centre, 1 s average)
  2. bring the gripper (pointing straight down, fingers closed) so the middle of the
     fingertips rests on the centre of that peg's top -> [r]  (robot: TF tip position)
     Don't knock the peg; step 1 must be done before the gripper hides the peg.
Then [f]: fits yaw + x/y/z offset, applies it to /peg_detector live and prints the YAML
to paste into config/detection_params.yaml (robot_frame: ...).

Use the SAME gripper orientation here as when grasping: the fit then absorbs any offset
between `end_effector_link` and the real fingertips, so IK targets line up.

Commands:  v | r | r x y z (manual, metres) | f | t (test: tip vs peg now) | u | l | c <colour> | q
"""
import json
import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from .geometry3d import fit_yaw_translation, rot_z


class FrameCalibration(Node):
    def __init__(self):
        super().__init__('frame_calibration')
        self.declare_parameter('colour', 'green')
        self.declare_parameter('detector', '/peg_detector')
        self.declare_parameter('robot_frame', 'link1')
        self.declare_parameter('tip_frame', 'end_effector_link')
        self.declare_parameter('average_s', 1.0)
        self.colour = self.get_parameter('colour').value
        det = self.get_parameter('detector').value
        self.hist = {}            # colour -> deque[(t, xyz)]
        self.lock = threading.Lock()
        self.create_subscription(String, f'{det}/scene', self._on_scene, 10)
        self.get_cli = self.create_client(GetParameters, f'{det}/get_parameters')
        self.set_cli = self.create_client(SetParameters, f'{det}/set_parameters')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pairs = []           # (p_drawn (3,), q_robot (3,))
        self.pending_v = None

    def _on_scene(self, msg):
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        now = time.monotonic()
        with self.lock:
            for name, p in s.get('pegs', {}).items():
                self.hist.setdefault(name, deque(maxlen=300)).append(
                    (now, np.array([p['x'], p['y'], p['z_top']])))

    # -------------------------------------------------------------- helpers
    def _call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'service {cli.srv_name} not available (is the detector running?)')
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f'{cli.srv_name} timed out')
            time.sleep(0.02)
        return fut.result()

    def current_robot_params(self):
        req = GetParameters.Request()
        req.names = ['robot_frame.yaw_deg', 'robot_frame.offset_m']
        vals = self._call(self.get_cli, req).values
        return float(vals[0].double_value), np.array(list(vals[1].double_array_value), float)

    def apply_params(self, yaw_deg, off):
        p1 = Parameter(name='robot_frame.yaw_deg',
                       value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                            double_value=float(yaw_deg)))
        p2 = Parameter(name='robot_frame.offset_m',
                       value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                            double_array_value=[float(x) for x in off]))
        req = SetParameters.Request()
        req.parameters = [p1, p2]
        res = self._call(self.set_cli, req).results
        return all(r.successful for r in res)

    def vision_average(self, colour):
        win = float(self.get_parameter('average_s').value)
        time.sleep(win)
        now = time.monotonic()
        with self.lock:
            pts = [p for t, p in self.hist.get(colour, []) if now - t <= win]
        if len(pts) < 3:
            return None, None
        P = np.array(pts)
        return np.median(P, axis=0), float(np.max(np.std(P, axis=0)) * 1000)

    def tip(self):
        tr = self.tf_buffer.lookup_transform(self.get_parameter('robot_frame').value,
                                             self.get_parameter('tip_frame').value, Time())
        t = tr.transform.translation
        return np.array([t.x, t.y, t.z])

    # ----------------------------------------------------------- commands
    def cmd_v(self):
        p, sd = self.vision_average(self.colour)
        if p is None:
            print(f'  no {self.colour} peg in /peg_detector/scene during the last second '
                  '(camera pose frozen? peg visible?)')
            return
        yaw0, off0 = self.current_robot_params()
        p_drawn = rot_z(-math.radians(yaw0)) @ (p - off0)     # undo the current robot_frame
        self.pending_v = (p, p_drawn)
        print(f'  vision {self.colour}: x={p[0]*100:+.1f} y={p[1]*100:+.1f} z_top={p[2]*100:+.1f} cm '
              f'(jitter {sd:.1f} mm)  -> now put the gripper tip on it and press r')

    def cmd_r(self, manual=None):
        if self.pending_v is None:
            print('  record the vision point first (v)')
            return
        try:
            q = np.array(manual, float) if manual else self.tip()
        except Exception as e:  # noqa: BLE001
            print(f'  TF {self.get_parameter("robot_frame").value} -> '
                  f'{self.get_parameter("tip_frame").value} not available: {e}\n'
                  '  (is the ROBOTIS bringup running?)  or type: r x y z')
            return
        p, p_drawn = self.pending_v
        self.pairs.append((p_drawn, q))
        self.pending_v = None
        print(f'  robot tip: x={q[0]*100:+.1f} y={q[1]*100:+.1f} z={q[2]*100:+.1f} cm | '
              f'difference to vision {np.linalg.norm(q - p)*1000:.0f} mm | pairs: {len(self.pairs)}')

    def cmd_f(self):
        if len(self.pairs) < 2:
            print('  need at least 2 pairs (4+ spread over the workspace is better)')
            return
        P = np.array([a for a, _ in self.pairs])
        Q = np.array([b for _, b in self.pairs])
        yaw, d, resid = fit_yaw_translation(P, Q)
        err = np.linalg.norm(resid, axis=1) * 1000
        for i, e in enumerate(err):
            print(f'  pair {i+1}: residual {e:5.1f} mm  (xy {np.linalg.norm(resid[i,:2])*1000:4.1f}, '
                  f'z {resid[i,2]*1000:+5.1f})')
        print(f'  RMS {np.sqrt(np.mean(err**2)):.1f} mm, max {err.max():.1f} mm '
              '(> ~5 mm: a bad pair, a moved peg, or the camera pose is off)')
        yaw_deg = math.degrees(yaw)
        ok = self.apply_params(yaw_deg, d)
        print(f'  applied to detector: {ok}')
        print('\n  Paste into config/detection_params.yaml:\n'
              '    robot_frame:\n'
              f'      yaw_deg: {yaw_deg:.3f}\n'
              f'      offset_m: [{d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f}]\n')

    def cmd_t(self):
        p, sd = self.vision_average(self.colour)
        try:
            q = self.tip()
        except Exception as e:  # noqa: BLE001
            print(f'  no TF: {e}')
            return
        if p is None:
            print(f'  tip at {np.round(q*100, 1)} cm; {self.colour} peg not visible')
            return
        dlt = (q - p) * 1000
        print(f'  tip - peg top (mm): dx {dlt[0]:+.1f} dy {dlt[1]:+.1f} dz {dlt[2]:+.1f} '
              f'| xy distance {np.linalg.norm(dlt[:2]):.1f} mm')


def main(args=None):
    rclpy.init(args=args)
    node = FrameCalibration()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    print(__doc__)
    try:
        while rclpy.ok():
            try:
                line = input(f'[{node.colour}, {len(node.pairs)} pairs] > ').strip().split()
            except EOFError:
                break
            if not line:
                continue
            c, args_ = line[0].lower(), line[1:]
            try:
                if c == 'v':
                    node.cmd_v()
                elif c == 'r':
                    node.cmd_r([float(a) for a in args_] if len(args_) == 3 else None)
                elif c == 'f':
                    node.cmd_f()
                elif c == 't':
                    node.cmd_t()
                elif c == 'u':
                    if node.pairs:
                        node.pairs.pop()
                    node.pending_v = None
                    print(f'  removed; {len(node.pairs)} pairs')
                elif c == 'l':
                    for i, (a, b) in enumerate(node.pairs):
                        print(f'  {i+1}: drawn {np.round(a*100, 1)}  robot {np.round(b*100, 1)} cm')
                elif c == 'c' and args_:
                    node.colour = args_[0]
                    print(f'  colour = {node.colour}')
                elif c == 'q':
                    break
                else:
                    print('  commands: v, r, r x y z, f, t, u, l, c <colour>, q')
            except RuntimeError as e:
                print(f'  {e}')
    except KeyboardInterrupt:
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
