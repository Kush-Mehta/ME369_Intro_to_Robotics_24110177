"""Pick pegs and put them in their sockets (cylinder, square, hexagon, triangle).

    ros2 run omx_calibration pick_place                        # green cylinder
    ros2 run omx_calibration pick_place --colour green --dry-run
    ros2 run omx_calibration pick_place --colour blue          # square
    ros2 run omx_calibration pick_place --colour all           # green, blue, orange, yellow
    ros2 run omx_calibration pick_place --colour yellow --check   # only print the alignment analysis

Needs: arm bringup + peg detector running (camera pose FROZEN), arm at home (go_home).
Enter before each move (--no-step = run straight through). q or Ctrl+C = stop, arm holds.

Geometry facts this code is built on (4-DOF arm, no wrist roll):
  * The gripper's yaw is fixed by the base angle: its fingers always close along the
    direction perpendicular to the line base -> gripper.
  * Closing parallel fingers on a square / hexagon SQUEEZES it straight (two flat faces on
    the fingers), whatever its starting angle. A triangle ends in one of TWO ways
    (a flat face on the left or on the right finger); which one follows from its start angle.
  * After that, the peg's angle at a socket is fixed by where the socket is, so the
    leftover misalignment is predicted BEFORE moving (vision measures the socket angle).
  * ONE gripper pitch for the whole move -> the peg stays vertical in the tilted gripper.
"""
import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import kinematics as K
from . import store
from .wizard import ARM, G_CLOSED, G_OPEN, PARK_DEFAULT, Robot

JOINT_SPEED = 0.5        # rad/s while following a path (max over joints)
STEP_M = 0.005           # Cartesian step between IK solutions

# colour -> (shape, symmetry n (0 = round), size: square/hex/triangle SIDE or cylinder DIAMETER, m)
PEGS = {'green': ('circle', 0, 0.040), 'blue': ('square', 4, 0.040),
        'orange': ('hexagon', 6, 0.020), 'yellow': ('triangle', 3, 0.030)}
ORDER = ['green', 'blue', 'orange', 'yellow']
# Hand-tuned on the lab setup: shift of the PLACE target from the detected socket centre, m (dx, dy)
SOCKET_OFFSET = {'green': (0.0, -0.005), 'orange': (0.0, -0.005),
                 'yellow': (-0.003, 0.005), 'blue': (-0.00425, -0.007)}


def get_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--colour', default='green', help="green / blue / orange / yellow / all")
    a.add_argument('--peg-height', type=float, default=0.100, help='m (all pegs: 0.100)')
    a.add_argument('--grip-depth', type=float, default=-0.04,
                   help='how far BELOW the peg top the aimed tool point goes, m (tuned on the lab arm: -0.04)')
    a.add_argument('--tool-length', type=float, default=0.085,
                   help='motor-14 axis -> aimed tool point, m (tuned on the lab arm: 0.085; ROBOTIS tip = 0.126)')
    a.add_argument('--approach', type=float, default=0.060, help='hover height above the peg top, m')
    a.add_argument('--lift-clear', type=float, default=0.030,
                   help='peg bottom height above the socket surface while moving, m')
    a.add_argument('--release-gap', type=float, default=0.005,
                   help='peg bottom height above the socket surface when released, m')
    a.add_argument('--min-pitch', type=float, default=30.0, help='deg; 90 = straight down')
    a.add_argument('--gripper-gap', type=float, default=0.0,
                   help='MEASURED inside gap between the open fingers, m (0 = do not check)')
    a.add_argument('--max-misalign', type=float, default=12.0,
                   help='deg; refuse to insert a peg predicted to be more than this off its socket')
    a.add_argument('--force', action='store_true', help='insert even if the misalignment check fails')
    a.add_argument('--check', action='store_true', help='only print vision + alignment analysis')
    a.add_argument('--dry-run', action='store_true',
                   help='gripper opens, hovers 3 cm above the peg top and over the socket, never closes')
    a.add_argument('--no-step', action='store_true', help='do not wait for Enter before each move')
    return a.parse_args(remove_ros_args(sys.argv)[1:])


# ------------------------------------------------------------------ geometry of the grasp
def wrap(a, period):
    return (a + period / 2) % period - period / 2


def circ_mean(deg, period):
    z = np.exp(1j * np.radians(np.asarray(deg)) * 360.0 / period)
    return float(np.degrees(np.angle(z.mean())) * period / 360.0) % period


def circumradius(n, size):
    return {4: size / math.sqrt(2), 6: size, 3: size / math.sqrt(3)}[n]


def width_across(n, R, psi_deg):
    """Width of the peg along the finger-closing direction. psi = a vertex direction measured
    from the base->peg line; the fingers close at 90 deg to that line."""
    v = np.radians(psi_deg + np.arange(n) * 360.0 / n - 90.0)
    p = R * np.cos(v)
    return float(p.max() - p.min())


def squeeze(n, R, psi0):
    """Angle the peg rotates to when the fingers squeeze it (local minimum of the width)."""
    psi, step = float(psi0), 0.25
    for _ in range(2000):
        w0, wl, wr = width_across(n, R, psi), width_across(n, R, psi - step), width_across(n, R, psi + step)
        if wl < w0 and wl <= wr:
            psi -= step
        elif wr < w0:
            psi += step
        else:
            break
    return psi % (360.0 / n)


def base_angle_deg(p):
    return math.degrees(math.atan2(p[1], p[0]))


# ------------------------------------------------------------------ vision
def read_targets(robot, colour, secs=1.5):
    pegs, socks, pyaw, syaw = [], [], [], []
    target, last = None, None
    t_end = time.monotonic() + secs
    while time.monotonic() < t_end:
        sc = robot.scene
        if sc is not None and sc is not last and time.monotonic() - sc[0] < 1.0:
            last = sc
            s = sc[1]
            p = s.get('pegs', {}).get(colour)
            if p:
                target = p.get('target_socket') or target
                pegs.append([p['x'], p['y'], p['z_top']])
                if 'yaw_deg' in p:
                    pyaw.append(p['yaw_deg'])
                so = s.get('sockets', {}).get(target) if target else None
                if so:
                    socks.append([so['x'], so['y'], so['z']])
                    if 'yaw_deg' in so:
                        syaw.append(so['yaw_deg'])
        time.sleep(0.05)
    if not pegs:
        return None, 'peg not seen (is it in view? camera pose FROZEN?)'
    if not socks:
        return None, f'socket "{target}" not seen'
    P, S = np.array(pegs), np.array(socks)
    jitter = max(P.std(0)[:2].max(), S.std(0)[:2].max()) * 1000
    return {'peg': np.median(P, 0), 'sock': np.median(S, 0), 'socket': target,
            'peg_yaw': pyaw, 'sock_yaw': syaw}, f'{len(P)} frames, jitter {jitter:.1f} mm'


def alignment(colour, v, a):
    """Predict the squeeze result and the misalignment at the socket. Returns (ok, lines)."""
    shape, n, size = PEGS[colour]
    if n == 0:
        return True, ['    round peg: no alignment needed']
    period = 360.0 / n
    if len(v['peg_yaw']) < 3 or len(v['sock_yaw']) < 3:
        return False, ['    peg or socket angle not measured (yaw missing in /peg_detector/scene - '
                       'updated peg_detection installed?)']
    R = circumradius(n, size)
    peg_yaw, sock_yaw = circ_mean(v['peg_yaw'], period), circ_mean(v['sock_yaw'], period)
    phi_p, phi_s = base_angle_deg(v['peg']), base_angle_deg(v['sock'])
    psi0 = (peg_yaw - phi_p) % period                       # vertex angle relative to base->peg line
    w0 = width_across(n, R, psi0)
    psi = squeeze(n, R, psi0)
    w = width_across(n, R, psi)
    err = wrap(sock_yaw - (phi_s + psi), period)
    L = [f'    peg angle {peg_yaw:5.1f} deg, socket angle {sock_yaw:5.1f} deg (vertex direction, mod {period:.0f})',
         f'    width between the fingers: {w0*1000:.1f} mm as it stands -> {w*1000:.1f} mm after the squeeze',
         f'    predicted misalignment at the socket: {err:+.1f} deg']
    ok = True
    if a.gripper_gap > 0 and w0 > a.gripper_gap - 0.003:
        L.append(f'    !! the open fingers ({a.gripper_gap*1000:.0f} mm) are too narrow for the peg at this '
                 f'angle ({w0*1000:.0f} mm) - turn the peg by hand a bit')
        ok = False
    if n == 3:
        other = (psi + 60.0) % period
        err2 = wrap(sock_yaw - (phi_s + other), period)
        L.append(f'    (triangle: the other squeeze outcome would give {err2:+.1f} deg)')
        if abs(err) > a.max_misalign and abs(err2) <= a.max_misalign:
            L.append('    !! this triangle will squeeze into the WRONG of its two ways. The arm cannot turn it\n'
                     '       (no wrist roll). Turn it by hand by about 60 deg and run again.')
    if abs(err) > a.max_misalign:
        L.append(f'    !! more than --max-misalign {a.max_misalign:.0f} deg: it will probably not go in'
                 + (' (--force given, trying anyway)' if a.force else ' (use --force to try anyway)'))
        ok = ok and a.force
    return ok, L


# ------------------------------------------------------------------ planning
def line(q_start, p_a, p_b, pitch):
    """IK every STEP_M along a straight line; None if any point fails."""
    n = max(1, int(math.ceil(np.linalg.norm(p_b - p_a) / STEP_M)))
    q, out = q_start, []
    for i in range(1, n + 1):
        p = p_a + (p_b - p_a) * i / n
        q2, ok = K.ik(p, pitch, q0=q)
        if not ok or np.max(np.abs(q2 - q)) > 0.35:
            return None
        out.append(q2)
        q = q2
    return out


def retreat(q_start, p_a, p_b, pitch):
    """Straight line (after release): the pitch may flatten step by step where needed."""
    n = max(1, int(math.ceil(np.linalg.norm(p_b - p_a) / STEP_M)))
    q, out, ph = q_start, [], pitch
    for i in range(1, n + 1):
        p = p_a + (p_b - p_a) * i / n
        while ph > math.radians(10):
            q2, ok = K.ik(p, ph, q0=q)
            if ok and np.max(np.abs(q2 - q)) <= 0.35:
                break
            ph -= math.radians(2)
        else:
            return None
        out.append(q2)
        q = q2
    return out


def plan(q_home, wps, pitch):
    """wps: list of (name, point, kind): 'joint' (free move), 'line' (fixed pitch), 'retreat'."""
    q, p_prev, segs = q_home, None, []
    for name, p, kind in wps:
        if kind == 'joint':
            q2, ok = K.ik(p, pitch, q0=q)
            if not ok:
                return None
            segs.append((name, [q2]))
            q = q2
        else:
            path = (retreat if kind == 'retreat' else line)(q, p_prev, p, pitch)
            if path is None:
                return None
            segs.append((name, path))
            q = path[-1]
        p_prev = p
    return segs


# ------------------------------------------------------------------ execution
def run_path(robot, path, label, speed=JOINT_SPEED):
    q_now = robot.q.copy()
    msg = JointTrajectory()
    msg.joint_names = ARM
    t, prev = 0.0, q_now
    for q in path:
        q = np.array(q, float)
        q[0] = K.near(q[0], float(prev[0]))
        dt = max(0.08, float(np.max(np.abs(q - prev))) / speed)
        if len(path) == 1:
            dt = max(dt, 2.0)
        t += dt
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.sec = int(t)
        pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        msg.points.append(pt)
        prev = q
    robot.traj_pub.publish(msg)
    time.sleep(t + 0.3)
    t_end = time.monotonic() + 3.0                     # let it settle
    while time.monotonic() < t_end:
        if np.degrees(np.abs(robot.q - prev)).max() < 1.5:
            break
        time.sleep(0.1)
    e = np.degrees(np.abs(robot.q - prev))
    tip = K.fk(robot.q)[0]
    print(f'    {label}: {t:.1f} s, joint error [deg] j1 {e[0]:.1f} j2 {e[1]:.1f} j3 {e[2]:.1f} j4 {e[3]:.1f}, '
          f'tip x {tip[0]*100:+.1f} y {tip[1]*100:+.1f} z {tip[2]*100:+.1f} cm')
    if e.max() > 3.0:
        print(f'    WARNING: joint {int(e.argmax()) + 1} is {e.max():.1f} deg off (load / gravity?)')
    return e.max() < 8.0


def wait_enter(step, text):
    if step:
        r = input(f'  -> {text}   [Enter = go, q = abort] ')
        if r.strip().lower() == 'q':
            raise KeyboardInterrupt


def go_home(robot):
    home = np.array(store.load().get('poses', {}).get('park_joints', PARK_DEFAULT), float)
    home[0] = K.near(home[0], float(robot.q[0]))
    run_path(robot, [home], 'home')


def task(robot, a, colour):
    print(f'\n=== pick & place: {colour} ({PEGS[colour][0]}) {"(DRY RUN)" if a.dry_run else ""} ===')
    v, info = read_targets(robot, colour)
    if v is None:
        print('  vision:', info)
        return False
    peg, sock = v['peg'], v['sock'].copy()
    dx, dy = SOCKET_OFFSET.get(colour, (0.0, 0.0))
    sock[0] += dx
    sock[1] += dy
    print(f'  vision ({info}):')
    print(f'    {colour} peg top  x {peg[0]*100:+.1f}  y {peg[1]*100:+.1f}  z {peg[2]*100:+.1f} cm')
    print(f'    {v["socket"]:8s} socket x {sock[0]*100:+.1f}  y {sock[1]*100:+.1f}  z {sock[2]*100:+.1f} cm')
    r_peg = math.hypot(peg[0], peg[1])
    if r_peg > 0.32 or not (0.03 <= peg[2] <= 0.20):
        print(f'  !! this {colour} detection is not a peg on the table (distance {r_peg*100:.0f} cm, '
              f'top z {peg[2]*100:.0f} cm). Is something else {colour} in the camera view?')
        return False
    ok, lines = alignment(colour, v, a)
    print('  alignment:')
    print('\n'.join(lines))
    if a.check:
        return True
    if not ok:
        print('  -> not attempting this peg')
        return False

    peg_h = a.peg_height
    held = peg_h - a.grip_depth                        # pad centre height above the peg bottom
    top = peg[2]
    grip_z = top - a.grip_depth
    lift_z = max(sock[2] + a.lift_clear + held, top + 0.01)
    rel_z = sock[2] + a.release_gap + held
    if a.dry_run:                                      # stay clear of the peg and the socket
        grip_z = top + 0.03
        lift_z = max(lift_z, grip_z)
        rel_z = lift_z
    print(f'  peg height {peg_h*100:.1f} cm, gripped {a.grip_depth*100:.1f} cm below its top')
    above = lambda xy, z: np.array([xy[0], xy[1], z])  # noqa: E731
    wps = [
        ('above peg', above(peg, top + a.approach), 'joint'),
        ('down to grip', above(peg, grip_z), 'line'),
        ('lift', above(peg, lift_z), 'line'),
        ('over socket', above(sock, lift_z), 'line'),
        ('down to release', above(sock, rel_z), 'line'),
        ('up', above(sock, rel_z + 0.04), 'retreat'),
    ]
    if min(p[2] for _, p, _ in wps) < 0.005:
        print('  a waypoint is below the board - check --peg-height / --grip-depth')
        return False
    segs, pitch = None, None
    for pd in np.arange(90.0, a.min_pitch - 0.1, -1.0):
        segs = plan(robot.q.copy(), wps, math.radians(pd))
        if segs:
            pitch = pd
            break
    if not segs:
        print(f'  NOT REACHABLE with any gripper pitch >= {a.min_pitch:.0f} deg. '
              'Try a smaller --grip-depth or --lift-clear.')
        return False
    print(f'  plan OK: gripper pitch {pitch:.0f} deg (90 = straight down), same for the whole move')
    for name, p, _ in wps:
        print(f'    {name:16s} x {p[0]*100:+6.1f}  y {p[1]*100:+6.1f}  z {p[2]*100:+6.1f} cm')

    step = not a.no_step
    wait_enter(step, 'OPEN gripper')
    robot.gripper(G_OPEN, wait=2.0)
    if robot.grip is not None:
        print(f'    gripper at {robot.grip*1000:+.1f} mm (open = {G_OPEN*1000:+.1f})')
        if abs(robot.grip - G_OPEN) > 0.004:
            print('    gripper did NOT open - stopping')
            return False
    for name, path in segs:
        if name == 'lift' and not a.dry_run:
            wait_enter(step, 'CLOSE gripper on the peg (squeezes it straight)')
            robot.gripper(G_CLOSED, wait=2.0)
            if robot.grip is not None:
                print(f'    gripper at {robot.grip*1000:+.1f} mm (fully closed = {G_CLOSED*1000:+.1f}; '
                      'well above that = holding the peg)')
        if name == 'up' and not a.dry_run:
            wait_enter(step, 'OPEN gripper (release peg)')
            robot.gripper(G_OPEN, wait=1.5)
        wait_enter(step, name)
        if not run_path(robot, path, name):
            print('    joint error > 8 deg after settling - stopping (blocked? collision?)')
            return False
    wait_enter(step, 'home')
    go_home(robot)
    return True


def hold(robot):
    """Stop now: replace the running trajectory by 'stay where you are'."""
    if robot.q is None:
        return
    msg = JointTrajectory()
    msg.joint_names = ARM
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in robot.q]
    pt.time_from_start.nanosec = 200_000_000
    msg.points = [pt]
    for _ in range(3):
        robot.traj_pub.publish(msg)
        time.sleep(0.05)


def main():
    a = get_args()
    try:                                   # keep ROS alive on Ctrl+C so we can send the stop
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    except (ImportError, TypeError):
        rclpy.init()
    robot = Robot()
    ex = MultiThreadedExecutor()
    ex.add_node(robot)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        t0 = time.monotonic()
        while robot.q is None and time.monotonic() - t0 < 5:
            time.sleep(0.1)
        if robot.q is None:
            print('no /joint_states - is the arm bringup running?')
            return
        K.EE = a.tool_length                           # IK aims the finger-pad centre
        print(f'tool length {K.EE*100:.1f} cm (motor-14 axis -> finger-pad centre)')
        colours = ORDER if a.colour == 'all' else [a.colour]
        results = {}
        for c in colours:
            if c not in PEGS:
                print(f'unknown colour {c}')
                continue
            results[c] = task(robot, a, c)
            if a.check:
                continue
            if not results[c] and len(colours) > 1 and robot.q is not None:
                print('  (skipping to the next peg after going home)')
                go_home(robot)
        print('\nresult:', ', '.join(f'{c} {"OK" if r else "--"}' for c, r in results.items()))
    except KeyboardInterrupt:
        hold(robot)
        print('\n  STOPPED (Ctrl+C) - the arm holds where it is. Run go_home when clear.')
    finally:
        ex.shutdown()
        robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
