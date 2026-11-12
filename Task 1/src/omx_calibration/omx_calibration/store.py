"""One calibration file for everything: ~/omx_calibration/calibration.yaml

The wizard writes each section when its step passes; the peg detector reads its own part
through peg_detector_overrides.yaml; the pick-and-place code reads calibration.yaml.
"""
import os
import socket
import time

import yaml

CALIB_DIR = os.path.expanduser(os.environ.get('OMX_CALIB_DIR', '~/omx_calibration'))
CALIB_FILE = os.path.join(CALIB_DIR, 'calibration.yaml')
OVERRIDES_FILE = os.path.join(CALIB_DIR, 'peg_detector_overrides.yaml')
CAMERA_POSE_FILE = os.path.join(CALIB_DIR, 'camera_pose.yaml')

STEPS = ['preflight', 'arm', 'camera', 'measurements', 'gripper', 'robot_frame',
         'verification', 'poses']


def _clean(x):
    """numpy -> plain python, rounded, so the YAML stays readable."""
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            x = x.tolist()
        elif isinstance(x, np.generic):
            x = x.item()
    except ImportError:
        pass
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float):
        return round(x, 6)
    return x


def load() -> dict:
    try:
        with open(CALIB_FILE) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def save(d: dict) -> str:
    os.makedirs(CALIB_DIR, exist_ok=True)
    d.setdefault('meta', {})
    d['meta'].setdefault('created', time.strftime('%Y-%m-%d %H:%M:%S'))
    d['meta']['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    d['meta']['host'] = socket.gethostname()
    tmp = CALIB_FILE + '.tmp'
    with open(tmp, 'w') as f:
        yaml.safe_dump(_clean(d), f, sort_keys=False)
    os.replace(tmp, CALIB_FILE)
    return CALIB_FILE


def update(section: str, data: dict, passed: bool) -> dict:
    d = load()
    data = dict(data)
    data['passed'] = bool(passed)
    data['date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    d[section] = data
    save(d)
    return d


def status_line(d: dict, step: str) -> str:
    s = d.get(step)
    if not s:
        return 'not done'
    return f"{'PASS' if s.get('passed') else 'FAIL'}  ({s.get('date', '')})"


def write_detector_overrides(length_m=None, robot_yaw_deg=None, robot_offset_m=None) -> str:
    """Parameters the peg detector loads on top of its config (see detection.launch.py)."""
    try:
        with open(OVERRIDES_FILE) as f:
            cur = yaml.safe_load(f) or {}
    except FileNotFoundError:
        cur = {}
    p = cur.setdefault('peg_detector', {}).setdefault('ros__parameters', {})
    p.setdefault('marker', {})['save_file'] = CAMERA_POSE_FILE
    if length_m is not None:
        p['marker']['length_m'] = float(length_m)
    if robot_yaw_deg is not None:
        p.setdefault('robot_frame', {})['yaw_deg'] = float(robot_yaw_deg)
    if robot_offset_m is not None:
        p.setdefault('robot_frame', {})['offset_m'] = [float(v) for v in robot_offset_m]
    os.makedirs(CALIB_DIR, exist_ok=True)
    with open(OVERRIDES_FILE, 'w') as f:
        f.write('# Written by the omx_calibration wizard - loaded by peg_detection/detection.launch.py\n')
        yaml.safe_dump(_clean(cur), f, sort_keys=False)
    return OVERRIDES_FILE
