"""Hardware / software checks before calibrating. Run standalone:

    ros2 run omx_calibration preflight

Each check comes from a problem we actually hit (see README, "Lessons").
"""
import glob
import grp
import os
import shutil
import subprocess


def _run(cmd, timeout=8):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def checks():
    """-> list of (name, ok: True/False/None(=warning), detail, fix)"""
    out = []
    distro = os.environ.get('ROS_DISTRO', '')
    out.append(('ROS 2 distro is Jazzy', distro == 'jazzy', f'ROS_DISTRO={distro or "unset"}',
                'source /opt/ros/jazzy/setup.bash (arm + vision must be on the same distro)'))

    ports = sorted(glob.glob('/dev/ttyUSB*'))
    out.append(('U2D2 serial port present', bool(ports), ', '.join(ports) or 'no /dev/ttyUSB*',
                'replug U2D2; `sudo dmesg | tail` - "error -71" = bad cable/USB port, try another'))
    for p in ports:
        lt = f'/sys/bus/usb-serial/devices/{os.path.basename(p)}/latency_timer'
        try:
            v = open(lt).read().strip()
        except OSError:
            v = '?'
        out.append((f'{p} latency timer = 1 ms', v == '1', f'latency_timer={v}',
                    'ros2 run open_manipulator_bringup om_create_udev_rules, then replug '
                    '(16 ms = "no status packet" errors, arm ignores commands)'))

    rc, txt = _run(['dpkg', '-s', 'brltty'])
    out.append(('brltty not installed', rc != 0, 'installed' if rc == 0 else 'not installed',
                'sudo apt remove brltty   (it grabs USB-serial adapters)'))

    try:
        groups = [g.gr_name for g in grp.getgrall() if os.environ.get('USER', '') in g.gr_mem]
        ok = 'dialout' in groups or os.geteuid() == 0
    except Exception:  # noqa: BLE001
        ok, groups = None, []
    out.append(('user in dialout group', ok, 'yes' if ok else 'no',
                'sudo usermod -aG dialout $USER, then log out/in'))

    rc, txt = _run(['pgrep', '-fil', 'dynamixel.?wizard'])
    out.append(('Dynamixel Wizard closed', rc != 0, 'running!' if rc == 0 else 'not running',
                'close Dynamixel Wizard (two programs on one port corrupt packets)'))

    if shutil.which('rs-enumerate-devices'):
        rc, txt = _run(['rs-enumerate-devices', '-s'])
        found = 'D435' in txt
        out.append(('RealSense D435i connected', found, 'found' if found else 'not found',
                    'plug into a USB 3 port (blue), close realsense-viewer'))
        if found:
            rc, full = _run(['rs-enumerate-devices'], timeout=15)
            usb = ''
            for line in full.splitlines():
                if 'Usb Type Descriptor' in line:
                    usb = line.split()[-1]
                    break
            out.append(('RealSense on USB 3', usb.startswith('3') if usb else None,
                        f'USB {usb or "?"}', 'use a USB 3 port and the original cable'))
    else:
        out.append(('RealSense tools installed', False, 'rs-enumerate-devices missing',
                    'bash scripts/install_all.sh'))
    return out


def print_checks(results) -> bool:
    all_ok = True
    for name, ok, detail, fix in results:
        tag = {True: 'PASS', False: 'FAIL', None: 'WARN'}[ok]
        print(f'  [{tag}] {name:<34} {detail}')
        if ok is False:
            all_ok = False
            print(f'         fix: {fix}')
    return all_ok


def main():
    print('\nPreflight checks')
    ok = print_checks(checks())
    print('\nAll good.' if ok else '\nFix the FAIL lines, then run again.')


if __name__ == '__main__':
    main()
