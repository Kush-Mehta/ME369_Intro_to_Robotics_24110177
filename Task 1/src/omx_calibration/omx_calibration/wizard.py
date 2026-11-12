"""Calibration wizard: one menu, every calibration step in order, all results in
~/omx_calibration/calibration.yaml.

    Terminal 1:  ros2 launch omx_calibration bringup.launch.py
    Terminal 2:  ros2 run omx_calibration wizard

Steps (each saves its section when done; any step can be re-run):
  1 preflight      hardware/software checks (USB port, latency timer, brltty, camera, ROS topics)
  2 arm            our kinematics vs the live TF, joint-zero check, base joint wrap check
  3 camera         marker size, camera-pose quality (jitter, tilt, scale), freeze
  4 measurements   ruler values (board, sockets, pegs) vs what the camera measures
  5 gripper        open / grip / closed values
  6 robot_frame    camera -> arm frame: touch 6 peg tops with the gripper, fit, apply
  7 verification   touch 3 NEW points: the real accuracy
  8 poses          park pose and safe travel height for the pick-and-place
  9 summary        status of everything + report.md
"""
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import kinematics as K
from . import preflight, store
from .fitting import fit_yaw_translation, rot_z

try:
    from control_msgs.action import GripperCommand
except ImportError:  # pragma: no cover
    GripperCommand = None

ARM = ['joint1', 'joint2', 'joint3', 'joint4']
COLOURS = ['green', 'orange', 'blue', 'yellow']
# Home / park = the pose recorded on the lab setup (gripper over the QR code side).
# joint1 = 180 deg; the actual command is always the turn closest to the current base angle.
PARK_DEFAULT = [3.157, -0.960, -0.109, 1.094]   # raw reading -9.409 rad = 3.157 + 2 full turns
G_MIN, G_MAX = -0.010, 0.019                  # gripper_left_joint range (m)
G_OPEN, G_CLOSED = 0.019, -0.010              # ROBOTIS: bigger = more open


# =============================================================== ROS interface
class Robot(Node):
    def __init__(self):
        super().__init__('omx_calibration_wizard')
        self.declare_parameter('detector', '/peg_detector')
        self.declare_parameter('robot_frame', 'link1')
        self.declare_parameter('tip_frame', 'end_effector_link')
        self.declare_parameter('z_floor_m', -0.005)       # jog never goes below this (link1 z)
        det = self.get_parameter('detector').value
        self.q = None
        self.grip = None
        self.scene = None
        self.status = None
        self.hist = {}
        self.lock = threading.Lock()
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self.create_subscription(String, f'{det}/scene', self._on_scene, 10)
        self.create_subscription(String, f'{det}/status', self._on_status, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.grip_ac = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd') \
            if GripperCommand else None
        self.get_cli = self.create_client(GetParameters, f'{det}/get_parameters')
        self.set_cli = self.create_client(SetParameters, f'{det}/set_parameters')
        self.recal_cli = self.create_client(Trigger, f'{det}/recalibrate')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.q_cmd = None

    # ---- subscriptions
    def _on_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        if all(j in d for j in ARM):
            self.q = np.array([d[j] for j in ARM])
        if 'gripper_left_joint' in d:
            self.grip = d['gripper_left_joint']

    def _on_scene(self, msg):
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        now = time.monotonic()
        with self.lock:
            self.scene = (now, s)
            for name, p in s.get('pegs', {}).items():
                self.hist.setdefault(name, deque(maxlen=400)).append(
                    (now, np.array([p['x'], p['y'], p['z_top']])))

    def _on_status(self, msg):
        try:
            self.status = (time.monotonic(), json.loads(msg.data))
        except ValueError:
            pass

    # ---- helpers
    def fresh_status(self, max_age=2.0):
        s = self.status
        return s[1] if s and time.monotonic() - s[0] < max_age else None

    def fresh_scene(self, max_age=2.0):
        s = self.scene
        return s[1] if s and time.monotonic() - s[0] < max_age else None

    def tip(self):
        tr = self.tf_buffer.lookup_transform(self.get_parameter('robot_frame').value,
                                             self.get_parameter('tip_frame').value, Time())
        t = tr.transform.translation
        return np.array([t.x - K.LINK1_TO_AXIS, t.y, t.z])   # TF link1 -> axis frame

    def send_joints(self, q, duration):
        q = [float(v) for v in q]
        if self.q is not None:
            q[0] = K.near(q[0], float(self.q[0]))   # base: never the long way round
        msg = JointTrajectory()
        msg.joint_names = ARM
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        msg.points = [pt]
        self.traj_pub.publish(msg)
        self.q_cmd = np.array(q, float)

    def move(self, q, speed=0.4, min_t=1.5):
        if self.q is None:
            raise RuntimeError('no /joint_states')
        q = np.array(q, float)
        q[0] = K.near(q[0], float(self.q[0]))
        T = max(min_t, float(np.max(np.abs(q - self.q))) / speed)
        self.send_joints(q, T)
        time.sleep(T + 0.4)
        return float(np.max(np.abs(self.q - q)))

    def gripper(self, pos, wait=1.5):
        if self.grip_ac is None:
            print('  control_msgs not installed: gripper commands unavailable')
            return
        if not self.grip_ac.wait_for_server(timeout_sec=2.0):
            print('  gripper action /gripper_controller/gripper_cmd not available')
            return
        goal = GripperCommand.Goal()
        goal.command.position = float(np.clip(pos, G_MIN, G_MAX))
        self.grip_ac.send_goal_async(goal)
        time.sleep(wait)

    def _call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{cli.srv_name} not available - is the peg detector running?')
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > timeout:
                raise RuntimeError(f'{cli.srv_name} timed out')
            time.sleep(0.02)
        return fut.result()

    def get_robot_frame(self):
        req = GetParameters.Request()
        req.names = ['robot_frame.yaw_deg', 'robot_frame.offset_m']
        v = self._call(self.get_cli, req).values
        return float(v[0].double_value), np.array(list(v[1].double_array_value), float)

    def set_params(self, values: dict):
        req = SetParameters.Request()
        for name, val in values.items():
            p = Parameter(name=name)
            if isinstance(val, (list, tuple, np.ndarray)):
                p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                         double_array_value=[float(x) for x in val])
            else:
                p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(val))
            req.parameters.append(p)
        return all(r.successful for r in self._call(self.set_cli, req).results)

    def recalibrate(self):
        return self._call(self.recal_cli, Trigger.Request()).message

    def vision_peg(self, colour, secs=1.0):
        """Median peg top (current robot frame) over `secs`, jitter in mm."""
        time.sleep(secs)
        now = time.monotonic()
        with self.lock:
            pts = [p for t, p in self.hist.get(colour, []) if now - t <= secs]
        if len(pts) < 3:
            return None, None
        P = np.array(pts)
        return np.median(P, axis=0), float(np.max(np.std(P, axis=0)) * 1000)

    def wait_frozen(self, timeout=20.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s = self.fresh_status()
            if s and s.get('state') == 'frozen':
                return s
            time.sleep(0.2)
        return None


# =============================================================== terminal helpers
def ask(prompt, default=None):
    d = f' [{default}]' if default not in (None, '') else ''
    try:
        r = input(f'{prompt}{d}: ').strip()
    except EOFError:
        r = ''
    return r if r else ('' if default is None else str(default))


def ask_float(prompt, default=None):
    while True:
        r = ask(prompt, default)
        if r == '':
            return None
        try:
            return float(r)
        except ValueError:
            print('  please type a number')


def yes(prompt, default='y'):
    return ask(f'{prompt} (y/n)', default).lower().startswith('y')


def pause(msg='Press Enter to continue'):
    ask(msg)


def header(title):
    print('\n' + '=' * 70 + f'\n {title}\n' + '=' * 70)


def verdict(ok, text):
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return ok


class Keys:
    """Single key presses without Enter (Linux terminal)."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def get(self, timeout=0.1):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None


# =============================================================== Cartesian jog
JOG_HELP = ('  w/s: +x/-x   a/d: +y/-y   r/f: up/down   1/2/3: step 1/5/20 mm\n'
            '  o/p: gripper open/close   Enter: RECORD   x: cancel\n'
            '  (x, y in the arm frame link1; the gripper stays pointing straight down)')


def jog(robot: Robot, title, allow_arm=True):
    """Keyboard jog of the gripper tip, pointing down. Returns True on Enter, False on cancel."""
    if robot.q is None:
        print('  no /joint_states - is the arm bringup running?')
        return False
    print(f'\n  >> {title}')
    print(JOG_HELP)
    q = robot.q.copy()
    p, phi = K.fk(q)
    if allow_arm and abs(phi - K.DOWN) > math.radians(4):
        print(f'  The gripper is at pitch {math.degrees(phi):.0f} deg; jogging keeps it pointing down.')
        if not yes('  Rotate the wrist to point down at the current position now?'):
            return False
        q2, ok = K.ik(p, K.DOWN, q0=q)
        if not ok:
            p = p + np.array([0, 0, 0.05])
            q2, ok = K.ik(p, K.DOWN, q0=q)
        if not ok:
            print('  cannot point down here: move the arm (teleop / park) closer to the workspace first')
            return False
        robot.move(q2)
        q = q2
    p = K.fk(q)[0]
    q_cmd = q.copy()
    step = 0.005
    floor = float(robot.get_parameter('z_floor_m').value)
    keymap = {'w': (1, 0, 0), 's': (-1, 0, 0), 'a': (0, 1, 0), 'd': (0, -1, 0),
              'r': (0, 0, 1), 'f': (0, 0, -1)}
    g = robot.grip if robot.grip is not None else 0.0
    last_print = ''
    with Keys() as keys:
        while True:
            line = (f'\r  tip x {p[0]*100:+6.1f}  y {p[1]*100:+6.1f}  z {p[2]*100:+6.1f} cm | '
                    f'step {step*1000:.0f} mm | gripper {g*1000:+.1f} mm | base {math.degrees(q_cmd[0]):+.0f} deg  ')
            if line != last_print:
                sys.stdout.write(line)
                sys.stdout.flush()
                last_print = line
            k = keys.get()
            if k is None:
                continue
            if k in ('\n', '\r'):
                print()
                return True
            if k in ('x', '\x1b'):
                print('\n  cancelled')
                return False
            if k in '123':
                step = {'1': 0.001, '2': 0.005, '3': 0.020}[k]
            elif k in 'op':
                g = float(np.clip(g + (0.002 if k == 'o' else -0.002), G_MIN, G_MAX))   # o = open, p = close
                threading.Thread(target=robot.gripper, args=(g, 0.0), daemon=True).start()
            elif allow_arm and k in keymap:
                p_new = p + step * np.array(keymap[k], float)
                if p_new[2] < floor:
                    sys.stdout.write(f'\n  z floor {floor*100:.1f} cm reached\n')
                    continue
                q_new, ok = K.ik(p_new, K.DOWN, q0=q_cmd)
                if not ok or abs(q_new[0] - q_cmd[0]) > 0.5:
                    sys.stdout.write('\n  not reachable with the gripper pointing down '
                                     '(or the base joint would wrap past +-180 deg)\n')
                    continue
                dt = max(0.25, float(np.max(np.abs(q_new - q_cmd))) / 0.5)
                robot.send_joints(q_new, dt)
                q_cmd, p = q_new, p_new


def lift(robot: Robot, dz=0.05):
    """Straight up by dz with the gripper pointing down (after touching a peg)."""
    if robot.q is None:
        return
    p, _ = K.fk(robot.q)
    q, ok = K.ik(p + np.array([0, 0, dz]), K.DOWN, q0=robot.q)
    if ok:
        robot.move(q, min_t=1.0)


def park(robot: Robot):
    d = store.load()
    q = d.get('poses', {}).get('park_joints', PARK_DEFAULT)
    print(f'  moving to park pose {np.round(np.degrees(q), 0).tolist()} deg ...')
    robot.move(q)


# =============================================================== steps
def step_preflight(robot: Robot):
    header('1. PREFLIGHT - hardware and software')
    res = preflight.checks()
    ok = preflight.print_checks(res)
    print('  ROS topics (waiting up to 5 s):')
    t0 = time.monotonic()
    while time.monotonic() - t0 < 5 and (robot.q is None or robot.fresh_status() is None):
        time.sleep(0.2)
    ok &= verdict(robot.q is not None, '/joint_states from the arm (ROBOTIS bringup running)')
    try:
        robot.tip()
        tf_ok = True
    except Exception:  # noqa: BLE001
        tf_ok = False
    ok &= verdict(tf_ok, 'TF link1 -> end_effector_link (robot_state_publisher)')
    ok &= verdict(robot.fresh_status() is not None, '/peg_detector/status (vision running)')
    store.update('preflight', {'checks': [{'name': n, 'ok': o, 'detail': d} for n, o, d, _ in res]}, ok)
    print('\n  ' + ('All good.' if ok else 'Fix the FAIL lines, then re-run step 1.'))


def step_arm(robot: Robot):
    header('2. ARM - model, zero pose, base joint')
    if robot.q is None:
        print('  no /joint_states: start the arm bringup first')
        return
    data, ok = {}, True
    # a) our kinematics vs the live TF (validates the IK model used by the final code)
    try:
        err = np.linalg.norm(K.fk(robot.q)[0] - robot.tip()) * 1000
        data['fk_vs_tf_mm'] = err
        ok &= verdict(err < 1.0, f'kinematics model matches TF at this pose ({err:.2f} mm)')
    except Exception as e:  # noqa: BLE001
        print(f'  TF not available: {e}')
        ok = False
    # b) zero pose
    print('\n  Zero-pose check: all joints to 0. The arm will stand up and point forward (+x).')
    print('  Make sure the space ABOVE and IN FRONT of the arm (+x side) is clear.')
    if yes('  Move to zero now?'):
        q_before = robot.q.copy()
        e = robot.move([0.0, 0.0, 0.0, 0.0])
        data['zero_tracking_deg'] = math.degrees(e)
        verdict(e < math.radians(1.5), f'arm reached the zero command (max joint error {math.degrees(e):.2f} deg)')
        try:
            err = np.linalg.norm(K.fk(robot.q)[0] - robot.tip()) * 1000
            data['fk_vs_tf_zero_mm'] = err
        except Exception:  # noqa: BLE001
            pass
        print('\n  LOOK at the arm (set-square / phone level app):')
        print('   - upper arm (between motors 12 and 13) VERTICAL')
        print('   - forearm (13 -> 14) and gripper (14 -> tip) HORIZONTAL, in one straight line')
        print('   - seen from above, the whole arm points along the +x direction of link1')
        z_ok = yes('  Is that what you see (each within ~2 deg)?')
        data['zero_pose_visual_ok'] = z_ok
        if not z_ok:
            data['zero_pose_notes'] = ask('  Which joint, which way, roughly how many degrees?')
            print('  -> A joint offset makes every computed position wrong in a pose-dependent way;\n'
                  '     the camera fit cannot remove it. Fix the horn / zero before going on (README 2.2).')
        ok &= z_ok
        if yes('  Move back to where it was?'):
            robot.move(q_before)
    # c) base joint wrap: where is the workspace in joint1?
    print('\n  Base joint check: jog/teleop the gripper over the MIDDLE of the peg area,')
    print('  then over the board end nearest +y and the end nearest -y.')
    angles = []
    for where in ('the middle of the peg area', 'the +y end of the board', 'the -y end of the board'):
        if jog(robot, f'move over {where}, Enter'):
            angles.append(math.degrees(robot.q[0]))
            print(f'  base joint = {angles[-1]:+.0f} deg')
    if angles:
        data['base_angles_deg'] = angles
        lo, hi = min(angles), max(angles)
        near_wrap = False                 # base is multi-turn: +-180 deg is no longer a limit
        wrap_ok = (hi - lo) < 270         # angles are continuous; only a huge spread is a problem
        data['base_wrap_ok'] = wrap_ok
        ok &= verdict(wrap_ok, f'workspace spread on the base joint is reasonable '
                      f'(range {lo:+.0f} .. {hi:+.0f} deg)')
        if near_wrap and wrap_ok:
            print('  WARNING: the workspace is close to +-180 deg on the base joint; keep everything on one side.')
        if not wrap_ok:
            print('  -> Moving between those areas would swing the base the LONG way round (almost 360 deg).\n'
                  '     Fix (README 2.3): rotate the setup so the work area is in front (+x) of the arm,\n'
                  '     or set Homing Offset = 2048 on motor ID 11 in Dynamixel Wizard (then redo steps 2, 6, 7).')
    store.update('arm', data, ok)


def step_camera(robot: Robot):
    header('3. CAMERA - marker size and camera pose')
    print('  Before this step: camera streaming >= 10 min (warm), marker sheet taped flat,\n'
          '  camera / board / sheet fixed. Arm away from the marker.')
    s = robot.wait_frozen(5.0)
    if s is None:
        print('  camera pose not frozen yet ...')
        if robot.fresh_status() is None:
            print('  no /peg_detector/status: is the vision running?')
            return
        if yes('  Ask the detector to re-measure the camera pose (arm away from the marker)?'):
            print('  ' + robot.recalibrate())
        s = robot.wait_frozen(20.0)
        if s is None:
            print('  still not frozen: is marker 0 fully visible in annotated_image?')
            return
    while True:
        length = s['marker_length_m']
        print(f'\n  length_m in use: {length*1000:.1f} mm')
        r = ask_float('  RULER: side of the black square of marker 0 in mm (Enter = keep)', '')
        if r is not None and abs(r / 1000 - length) > 0.0002:
            robot.set_params({'marker.length_m': r / 1000})
            store.write_detector_overrides(length_m=r / 1000)
            print('  ' + robot.recalibrate())
            time.sleep(1.0)
            s = robot.wait_frozen(25.0)
            if s is None:
                print('  not frozen after re-measuring - check the marker visibility')
                return
            continue
        break
    time.sleep(3.0)                       # let the live/frozen comparison settle
    s = robot.fresh_status() or s
    fz, mk = s.get('frozen') or {}, s.get('marker') or {}
    data = {'length_m': s['marker_length_m'], 'frozen_frames': fz.get('n'),
            'jitter_mm': fz.get('jitter_mm'), 'tilt_correction_deg': mk.get('tilt_correction_deg'),
            'marker_side_from_depth_mm': mk.get('side_mm'), 'plane_rms_mm': mk.get('plane_rms_mm'),
            'drift_mm': s.get('drift_mm'), 'socket_z_m': s.get('socket_z_drawn', {})}
    ok = True
    ok &= verdict((fz.get('jitter_mm') or 99) < 1.0, f"pose jitter while freezing: {fz.get('jitter_mm', float('nan')):.2f} mm (< 1)")
    if mk.get('tilt_correction_deg') is not None:
        ok &= verdict(mk['tilt_correction_deg'] < 3.0,
                      f"tilt correction from the depth plane: {mk['tilt_correction_deg']:.2f} deg (< 3)")
        ratio = mk['side_mm'] / (s['marker_length_m'] * 1000)
        ok &= verdict(abs(ratio - 1) < 0.03, f"marker size seen by depth {mk['side_mm']:.1f} mm vs ruler "
                      f"{s['marker_length_m']*1000:.1f} mm ({(ratio-1)*100:+.1f} %, within 3 %)")
    else:
        print('  marker not visible right now (arm in the way?) - move the arm away and re-run for the full check')
        ok = False
    if s.get('drift_mm') is not None:
        ok &= verdict(s['drift_mm'] < 3.0, f"live marker vs frozen pose: {s['drift_mm']:.1f} mm (< 3)")
    zs = list((s.get('socket_z_drawn') or {}).values())
    if len(zs) >= 2:
        spread = (max(zs) - min(zs)) * 1000
        data['socket_z_spread_mm'] = spread
        ok &= verdict(spread < 2.0, f'the {len(zs)} sockets are at the same height (spread {spread:.1f} mm < 2)')
    if not ok:
        print('\n  If size/heights fail: RealSense depth needs its own calibration -> README 6.1\n'
              '  (realsense-viewer: On-Chip Calibration + Tare), then re-run this step.')
    store.update('camera', data, ok)


def _drawn(p, yaw_deg, off):
    return rot_z(-math.radians(yaw_deg)) @ (np.asarray(p) - off)


def step_measurements(robot: Robot):
    header('4. MEASUREMENTS - ruler values vs camera')
    s = robot.fresh_scene()
    if not s or s.get('calibration') != 'frozen':
        print('  vision not ready (camera pose not frozen) - do step 3 first')
        return
    yaw, off = robot.get_robot_frame()
    print('  Measure with a ruler / caliper, in mm. Enter to skip a value.')
    d = store.load().get('measurements', {})
    board = ask_float('  board thickness (foam to board top)', d.get('board_thickness_mm'))
    depth = ask_float('  socket depth (board top to bottom of the hole)', d.get('socket_depth_mm'))
    on = ask('  pegs stand on the foam (f) or on the board (b)?', d.get('pegs_stand_on', 'f'))
    pegs = {}
    for c in COLOURS:
        old = d.get('pegs', {}).get(c, {})
        h = ask_float(f'  {c:6s} peg height', old.get('height_mm'))
        w = ask_float(f'  {c:6s} peg width (across flats / diameter)', old.get('width_mm'))
        pegs[c] = {'height_mm': h, 'width_mm': w}
    s = robot.fresh_scene() or s
    base = (board or 0.0) if on.startswith('b') else 0.0
    ok = True
    print('\n  camera vs ruler (heights above the foam, in the drawn frame):')
    socket_z = [(_drawn([v['x'], v['y'], v['z']], yaw, off)[2]) * 1000 for v in s.get('sockets', {}).values()]
    if board is not None and socket_z:
        e = float(np.mean(socket_z)) - board
        ok &= verdict(abs(e) < 2.0, f'socket Z {np.mean(socket_z):.1f} mm vs board {board:.1f} mm ({e:+.1f}, < 2)')
    for c in COLOURS:
        p = s.get('pegs', {}).get(c)
        h = pegs[c]['height_mm']
        if p is None or h is None:
            continue
        z = _drawn([p['x'], p['y'], p['z_top']], yaw, off)[2] * 1000
        e = z - (h + base)
        pegs[c]['camera_top_mm'] = z
        ok &= verdict(abs(e) < 3.0, f'{c:6s} peg top {z:.1f} mm vs ruler {h + base:.1f} mm ({e:+.1f}, < 3)')
    if not ok:
        print('  -> All off by the same %: depth scale (README 6.1). Only one peg off: its top face is\n'
              '     shiny / not found (check annotated_image: white dot + outline on the top face).')
    store.update('measurements', {'board_thickness_mm': board, 'socket_depth_mm': depth,
                                  'pegs_stand_on': 'board' if on.startswith('b') else 'foam',
                                  'pegs': pegs}, ok)


def step_gripper(robot: Robot):
    header('5. GRIPPER - open / grip / closed values')
    if robot.grip is None:
        print('  no gripper_left_joint in /joint_states')
        return
    vals = {}
    print('  Only o/p (gripper) do anything in this step.')
    if jog(robot, 'open the gripper fully with o, Enter', allow_arm=False):
        vals['open_m'] = robot.grip
    print('  Hold a peg between the fingers (by hand), close with p until it is held firmly.')
    for c in COLOURS:
        if jog(robot, f'{c} peg: close until it just grips, Enter (x = skip)', allow_arm=False):
            vals.setdefault('grip_m', {})[c] = robot.grip
        robot.gripper(vals.get('open_m', G_OPEN))
    if jog(robot, 'close fully (no peg) with p, Enter', allow_arm=False):
        vals['closed_m'] = robot.grip
    robot.gripper(vals.get('open_m', G_OPEN))
    ok = 'open_m' in vals and 'grip_m' in vals
    for k, v in vals.items():
        print(f'  {k}: {v}')
    store.update('gripper', vals, ok)


def _touch_pair(robot: Robot, colour, k, n, park_between=True):
    """Vision point (before) + touched tip (after). Returns (p_link1_current, q_tip, joint1) or None."""
    print(f'\n  --- point {k}/{n} ---')
    if park_between and yes('  Move the arm to the park pose first (so it does not hide the peg)?'):
        park(robot)
    pause(f'  Place the {colour} peg at a NEW spot (spread them out), hands away, then Enter')
    p, sd = robot.vision_peg(colour)
    if p is None:
        print(f'  {colour} peg not seen (frozen? visible in annotated_image?) - skipped')
        return None
    print(f'  vision: x {p[0]*100:+.1f}  y {p[1]*100:+.1f}  z_top {p[2]*100:+.1f} cm  (jitter {sd:.1f} mm)')
    if sd > 2.0:
        print('  WARNING: jittery - lighting / reflections?')
    if not jog(robot, 'jog the fingertips (closed, pointing down) to just TOUCH the centre of the peg top, Enter'):
        return None
    try:
        q = robot.tip()
    except Exception as e:  # noqa: BLE001
        print(f'  no TF: {e}')
        return None
    j1 = float(robot.q[0])
    print(f'  robot tip: x {q[0]*100:+.1f}  y {q[1]*100:+.1f}  z {q[2]*100:+.1f} cm   '
          f'(vision - tip = {np.linalg.norm(p - q)*1000:.0f} mm)')
    lift(robot)
    return p, q, j1


def step_robot_frame(robot: Robot):
    header('6. ROBOT FRAME - camera -> arm (link1)')
    s = robot.fresh_scene()
    if not s or s.get('calibration') != 'frozen':
        print('  vision not ready (camera pose not frozen) - do step 3 first')
        return
    colour = ask('  Peg colour to use', 'green')
    n = int(ask_float('  Number of points (6 recommended, min 3)', 6))
    print('\n  For each point: put the peg somewhere new, the wizard records where the camera sees it,\n'
          '  then you jog the closed gripper (pointing down) until the fingertips TOUCH the centre of\n'
          '  the peg top. Spread the points: near the board, near the base, both far sides.')
    robot.gripper(G_CLOSED)                      # fingers closed = a well-defined tip
    pairs = []
    for k in range(1, n + 1):
        yaw0, off0 = robot.get_robot_frame()
        r = _touch_pair(robot, colour, k, n)
        if r is None:
            continue
        p, q, j1 = r
        pairs.append({'drawn': _drawn(p, yaw0, off0), 'robot': q, 'joint1': j1})
    if len(pairs) < 3:
        print('  fewer than 3 good points - not fitted')
        return
    while True:
        P = np.array([x['drawn'] for x in pairs])
        Q = np.array([x['robot'] for x in pairs])
        yaw, d, resid = fit_yaw_translation(P, Q)
        err = np.linalg.norm(resid, axis=1) * 1000
        print('\n  residuals after the fit:')
        for i, e in enumerate(err):
            print(f'   point {i+1}: {e:5.1f} mm')
        rms = float(np.sqrt(np.mean(err ** 2)))
        print(f'  RMS {rms:.1f} mm   max {err.max():.1f} mm')
        worst = int(np.argmax(err))
        if len(pairs) > 4 and err[worst] > 2.5 * np.median(err) and err[worst] > 5 and \
                yes(f'  Point {worst+1} looks bad (peg moved / tip off-centre). Drop it and refit?'):
            pairs.pop(worst)
            continue
        break
    ok = verdict(rms < 4.0 and err.max() < 7.0, f'fit quality: RMS {rms:.1f} mm (< 4), max {err.max():.1f} mm (< 7)')
    yaw_deg = math.degrees(yaw)
    robot.set_params({'robot_frame.yaw_deg': yaw_deg, 'robot_frame.offset_m': d})
    f = store.write_detector_overrides(robot_yaw_deg=yaw_deg, robot_offset_m=d)
    print(f'  applied to the running detector and saved to {f}')
    print(f'  yaw {yaw_deg:+.2f} deg, offset [{d[0]*1000:+.1f}, {d[1]*1000:+.1f}, {d[2]*1000:+.1f}] mm')
    j1s = [math.degrees(x['joint1']) for x in pairs]
    if min(j1s) < -90 and max(j1s) > 90:
        print('  WARNING: base joint used both +170 and -170 deg sides -> see step 2 / README 2.3')
    store.update('robot_frame', {'yaw_deg': yaw_deg, 'offset_m': d, 'rms_mm': rms,
                                 'max_mm': float(err.max()), 'colour': colour,
                                 'pairs': [{'drawn_m': x['drawn'], 'robot_m': x['robot']} for x in pairs]},
                 ok)
    if ok:
        print('  Next: step 7 (verification on NEW points).')


def step_verification(robot: Robot):
    header('7. VERIFICATION - accuracy on NEW points')
    s = robot.fresh_scene()
    if not s or s.get('calibration') != 'frozen':
        print('  vision not ready - do steps 3 and 6 first')
        return
    if not store.load().get('robot_frame', {}).get('passed'):
        print('  note: step 6 has not passed yet')
    colour = ask('  Peg colour', 'green')
    n = int(ask_float('  Number of new points', 3))
    robot.gripper(G_CLOSED)
    rows = []
    for k in range(1, n + 1):
        r = _touch_pair(robot, colour, k, n)
        if r is None:
            continue
        p, q, _ = r
        dlt = (q - p) * 1000
        rows.append({'vision_m': p, 'tip_m': q, 'dx_mm': dlt[0], 'dy_mm': dlt[1], 'dz_mm': dlt[2],
                     'xy_mm': float(np.hypot(dlt[0], dlt[1]))})
        print(f'  error: dx {dlt[0]:+.1f}  dy {dlt[1]:+.1f}  dz {dlt[2]:+.1f} mm   xy {rows[-1]["xy_mm"]:.1f} mm')
    if not rows:
        return
    xy = np.array([r['xy_mm'] for r in rows])
    dz = np.array([r['dz_mm'] for r in rows])
    ok = verdict(xy.max() < 5.0 and np.abs(dz).max() < 5.0,
                 f'xy error mean {xy.mean():.1f} / max {xy.max():.1f} mm, z error max {np.abs(dz).max():.1f} mm (< 5)')
    if not ok:
        print('  -> see README 6.2')
    store.update('verification', {'points': rows, 'xy_mean_mm': float(xy.mean()),
                                  'xy_max_mm': float(xy.max()), 'z_max_abs_mm': float(np.abs(dz).max())}, ok)


def step_poses(robot: Robot):
    header('8. POSES - park pose and travel height for the pick-and-place')
    data = dict(store.load().get('poses', {}))
    print('  PARK: a pose that keeps the arm out of the camera view of pegs, sockets and marker.')
    if jog(robot, 'jog to a good park pose (or keep the current one), Enter'):
        data['park_joints'] = robot.q.tolist()
    print('  TRAVEL HEIGHT: tip height for moving between peg and socket - clear of the tallest peg + 3 cm.')
    if jog(robot, 'jog the (pointing-down) tip to the travel height over the pegs, Enter'):
        data['travel_z_m'] = float(K.fk(robot.q)[0][2])
    ok = 'park_joints' in data and 'travel_z_m' in data
    for k, v in data.items():
        print(f'  {k}: {np.round(v, 4).tolist() if isinstance(v, list) else round(v, 4)}')
    store.update('poses', data, ok)


def step_summary(robot: Robot):
    header('9. SUMMARY')
    d = store.load()
    for i, st in enumerate(store.STEPS, 1):
        print(f'  {i}. {st:<13} {store.status_line(d, st)}')
    rf, vf = d.get('robot_frame', {}), d.get('verification', {})
    lines = ['# OpenMANIPULATOR-X calibration report', '',
             f"Date: {d.get('meta', {}).get('updated', '')}  Host: {d.get('meta', {}).get('host', '')}", '',
             '| Step | Result |', '|---|---|']
    lines += [f'| {st} | {store.status_line(d, st)} |' for st in store.STEPS]
    lines += ['', '## Key numbers', '']
    if d.get('camera'):
        c = d['camera']
        lines.append(f"- Camera pose jitter {c.get('jitter_mm')} mm, tilt correction "
                     f"{c.get('tilt_correction_deg')} deg, marker by depth {c.get('marker_side_from_depth_mm')} mm "
                     f"(ruler {round(c.get('length_m', 0) * 1000, 1)} mm)")
    if rf:
        lines.append(f"- Camera -> arm fit: {len(rf.get('pairs', []))} points, RMS {rf.get('rms_mm')} mm, "
                     f"yaw {rf.get('yaw_deg')} deg, offset {rf.get('offset_m')} m")
    if vf:
        lines.append(f"- **Verification on new points: xy mean {vf.get('xy_mean_mm')} mm, max "
                     f"{vf.get('xy_max_mm')} mm, z max {vf.get('z_max_abs_mm')} mm**")
    path = os.path.join(store.CALIB_DIR, 'report.md')
    os.makedirs(store.CALIB_DIR, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n  calibration file: {store.CALIB_FILE}\n  report:           {path}')
    if all(d.get(s, {}).get('passed') for s in store.STEPS):
        print('\n  EVERYTHING PASSED - the pick-and-place code can use calibration.yaml + /peg_detector/scene.')


STEP_FUNCS = [step_preflight, step_arm, step_camera, step_measurements, step_gripper,
              step_robot_frame, step_verification, step_poses, step_summary]
NAMES = store.STEPS + ['summary']


def menu(robot: Robot):
    while True:
        d = store.load()
        print('\n' + '-' * 70 + '\n OMX CALIBRATION WIZARD     (file: ' + store.CALIB_FILE + ')\n' + '-' * 70)
        for i, n in enumerate(NAMES, 1):
            st = store.status_line(d, n) if n != 'summary' else ''
            print(f'  {i}. {n:<13} {st}')
        print('  a. run steps 1-9 in order   p. park the arm   q. quit')
        c = ask('choice').lower()
        try:
            if c == 'q':
                return
            if c == 'p':
                park(robot)
            elif c == 'a':
                for fn in STEP_FUNCS:
                    fn(robot)
                    if fn is not step_summary and not yes('\nContinue with the next step?'):
                        break
            elif c.isdigit() and 1 <= int(c) <= len(STEP_FUNCS):
                STEP_FUNCS[int(c) - 1](robot)
        except RuntimeError as e:
            print(f'  ERROR: {e}')
        except KeyboardInterrupt:
            print('\n  step interrupted')


def main(args=None):
    rclpy.init(args=args)
    robot = Robot()
    ex = MultiThreadedExecutor()
    ex.add_node(robot)
    th = threading.Thread(target=ex.spin, daemon=True)
    th.start()
    try:
        menu(robot)
    finally:
        ex.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
