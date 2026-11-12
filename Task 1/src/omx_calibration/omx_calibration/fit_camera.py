"""Correct the camera -> robot coordinates from points you MEASURED by hand.

    ros2 run omx_calibration fit_camera

Put a peg at a spot, measure its true position (x, y from the base rotation axis, z = its top
height, same units/axes as the detector prints), and type:

    green -14 -6.3 10        colour  x  y  z   (cm)  -> the script averages 2 s of detections
    fit                      compute the correction from all points so far, show errors
    apply                    send it to the running detector + save it in the config
    list / undo / quit

Points needed:  1 = offset only,  2 = offset + rotation,  3+ spread out = also TILT
(the tilt is what makes heights grow with distance). 5-6 points, near/far/left/right, is best.
The correction is tied to the camera pose, so after `apply` the frozen camera pose is reused
on the next start (marker.load_saved: true). Moved the camera? Recalibrate and refit.
"""
import json
import os
import re
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_msgs.msg import String

IDENT = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


class Fit(Node):
    def __init__(self):
        super().__init__('omx_fit_camera')
        self.scene = None
        self.create_subscription(String, '/peg_detector/scene', self._scene, 10)
        self.get_cli = self.create_client(GetParameters, '/peg_detector/get_parameters')
        self.set_cli = self.create_client(SetParameters, '/peg_detector/set_parameters')

    def _scene(self, m):
        try:
            self.scene = (time.monotonic(), json.loads(m.data))
        except ValueError:
            pass

    def call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError('peg detector parameter service not available - is the launch running?')
        fut = cli.call_async(req)
        t0 = time.monotonic()
        while not fut.done() and time.monotonic() - t0 < timeout:
            time.sleep(0.02)
        if not fut.done():
            raise RuntimeError('parameter service timed out')
        return fut.result()

    def get_correction(self):
        req = GetParameters.Request(names=['robot_frame.correction'])
        v = self.call(self.get_cli, req).values[0]
        c = list(v.double_array_value)
        return c if len(c) == 12 else list(IDENT)

    def set_correction(self, c):
        p = Parameter(name='robot_frame.correction',
                      value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                           double_array_value=[float(x) for x in c]))
        r = self.call(self.set_cli, SetParameters.Request(parameters=[p]))
        if not r.results[0].successful:
            raise RuntimeError(r.results[0].reason)

    def measure(self, colour, secs=2.0):
        pts, last = [], None
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            sc = self.scene
            if sc is not None and sc is not last and time.monotonic() - sc[0] < 1.0:
                last = sc
                p = sc[1].get('pegs', {}).get(colour)
                if p:
                    pts.append([p['x'], p['y'], p['z_top']])
            time.sleep(0.03)
        if len(pts) < 5:
            return None, 0.0
        P = np.array(pts)
        return np.median(P, 0), float(P.std(0).max() * 1000)


# ------------------------------------------------------------------ fitting
def fit_rigid(M, T):
    """Rigid transform (R, t) with R @ M_i + t ~= T_i. 1 pt: translation; 2 pts or collinear:
    rotation about z + translation; 3+ spread points: full 3-D rotation (incl. tilt)."""
    M, T = np.asarray(M, float), np.asarray(T, float)
    n = len(M)
    if n == 1:
        return np.eye(3), T[0] - M[0], 'offset only (1 point)'
    cm, ct = M.mean(0), T.mean(0)
    A, B = M - cm, T - ct
    spread = np.linalg.svd(A[:, :2], compute_uv=False) if n >= 2 else [0, 0]
    planar_ok = n >= 3 and spread[-1] > 0.02 * np.sqrt(n)        # points not on one line
    if planar_ok:
        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T
        how = f'rotation (incl. tilt) + offset ({n} points)'
    else:
        H = A[:, :2].T @ B[:, :2]
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R2 = Vt.T @ D @ U.T
        R = np.eye(3)
        R[:2, :2] = R2
        how = f'rotation about z + offset ({n} points{"; on one line -> no tilt" if n >= 3 else ""})'
    t = ct - R @ cm
    return R, t, how


def angles(R):
    tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw, tilt


def save_config(c):
    """Write the correction into detection_params.yaml (source + installed) and reuse the pose."""
    paths = []
    try:
        from ament_index_python.packages import get_package_share_directory
        paths.append(os.path.join(get_package_share_directory('peg_detection'), 'config',
                                  'detection_params.yaml'))
    except Exception:  # noqa: BLE001
        pass
    paths.append(os.path.expanduser('~/Documents/omx_calib_ws/src/peg_detection/config/detection_params.yaml'))
    done = set()
    line = '      correction: [' + ', '.join(f'{x:.6f}' for x in c) + ']   # fit_camera\n'
    for p in paths:
        if not os.path.exists(p):
            continue
        real = os.path.realpath(p)
        if real in done:
            continue
        s = open(real).read()
        if re.search(r'^      correction: \[.*$\n', s, re.M):
            s = re.sub(r'^      correction: \[.*$\n', line, s, count=1, flags=re.M)
        else:
            s = re.sub(r'(^      offset_m: \[.*$\n)', r'\1' + line.replace('\\', '\\\\'), s, count=1, flags=re.M)
        s = re.sub(r'^(      load_saved: )false', r'\1true ', s, count=1, flags=re.M)
        open(real, 'w').write(s)
        done.add(real)
        print(f'  saved in {real}')
    if not done:
        print('  !! config file not found - correction is active now but NOT saved')


def table(M, T, R=None, t=None):
    for i, (m, tr) in enumerate(zip(M, T)):
        p = m if R is None else R @ m + t
        e = (p - tr) * 100
        print(f'   {i+1}: true {np.round(tr*100,1)}  camera {np.round(p*100,1)}  error x {e[0]:+.1f} '
              f'y {e[1]:+.1f} z {e[2]:+.1f} cm  ({np.linalg.norm(e):.1f})')


def main():
    rclpy.init()
    node = Fit()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    M, T, names = [], [], []
    fitted = None
    print(__doc__)
    try:
        while True:
            cmd = input('fit_camera> ').strip()
            if not cmd:
                continue
            w = cmd.split()
            k = w[0].lower()
            if k in ('q', 'quit', 'exit'):
                break
            if k == 'list':
                table(M, T)
                continue
            if k == 'undo':
                if M:
                    M.pop(); T.pop(); names.pop()
                    print(f'  removed; {len(M)} points left')
                continue
            if k == 'fit':
                if not M:
                    print('  no points yet')
                    continue
                R, t, how = fit_rigid(M, T)
                yaw, tilt = angles(R)
                print(f'  fit: {how}  -> rotate {yaw:+.2f} deg about z, tilt {tilt:.2f} deg, '
                      f'shift {np.round(t*100, 2)} cm')
                print('  BEFORE:'); table(M, T)
                print('  AFTER:'); table(M, T, R, t)
                fitted = (R, t)
                print("  'apply' to use it")
                continue
            if k == 'apply':
                if fitted is None:
                    print("  run 'fit' first")
                    continue
                R, t = fitted
                c0 = np.array(node.get_correction(), float)
                R0, t0 = c0[:9].reshape(3, 3), c0[9:]
                Rn, tn = R @ R0, R @ t0 + t
                c = list(Rn.ravel()) + list(tn)
                node.set_correction(c)
                print('  applied to the running detector')
                save_config(c)
                M = [R @ m + t for m in M]                  # stored readings now corrected
                fitted = None
                print('  check a peg again (type its colour + true xyz) - it should now match')
                continue
            if k == 'reset':
                node.set_correction(IDENT)
                save_config(IDENT)
                M, T, names, fitted = [], [], [], None
                print('  correction reset to identity (points cleared)')
                continue
            if len(w) == 4:
                try:
                    tr = np.array([float(v) for v in w[1:]]) / 100.0
                except ValueError:
                    print('  usage: colour x y z  (cm)')
                    continue
                m, jit = node.measure(k)
                if m is None:
                    print(f'  {k} not seen (in view? camera pose FROZEN?)')
                    continue
                M.append(m); T.append(tr); names.append(k)
                e = (m - tr) * 100
                print(f'  point {len(M)}: camera {np.round(m*100,1)} cm (jitter {jit:.1f} mm), '
                      f'error x {e[0]:+.1f} y {e[1]:+.1f} z {e[2]:+.1f} cm')
                continue
            print('  commands: <colour> x y z | fit | apply | list | undo | reset | quit')
    except (KeyboardInterrupt, EOFError):
        print()
    except RuntimeError as e:
        print('  ERROR:', e)
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
