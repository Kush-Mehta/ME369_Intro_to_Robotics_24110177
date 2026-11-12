# Vision-Guided Peg Sorting with the OpenMANIPULATOR-X

**Task attempted: Task 1, Peg Sorting.**

The ROBOTIS OpenMANIPULATOR-X picks four coloured pegs from the table and inserts each one into
its matching socket. An Intel RealSense D435i camera finds the pegs and sockets, the positions are
converted into the robot's base frame, and an inverse-kinematics planner moves the arm.

| Peg | Shape | Size | Socket |
|---|---|---|---|
| Green | Cylinder | Ø 40 mm | Circle |
| Blue | Square prism | side 40 mm | Square |
| Orange | Hexagonal prism | side 20 mm | Hexagon |
| Yellow | Triangular prism | side 30 mm | Triangle |

All pegs are 100 mm tall.

---

## Contents

1. [Package contents](#1-package-contents)
2. [What we built](#2-what-we-built)
3. [Learnings and limitations](#3-learnings-and-limitations)
4. [Setup from scratch](#4-setup-from-scratch)
5. [Running the system](#5-running-the-system)
6. [Troubleshooting](#6-troubleshooting)
7. [Reference](#7-reference)

---

## 1. Package contents

This archive is a complete ROS 2 workspace. It contains all three parts of the project:

```
ITR_Task1_PegSorting/
├── README.md                          this file
├── markers/                           printable ArUco marker
├── scripts/                           helper scripts
└── src/
    ├── omx_calibration/               (1) our package: IK, motion, pick-and-place, calibration tools
    ├── peg_detection/                 (1) our package: RealSense vision, peg/socket detection
    └── robotis/
        ├── open_manipulator/          (2) ROBOTIS arm stack (description, ros2_control bringup),
        │                                  with our modifications for this setup
        └── open_manipulator_msgs/     (3) ROBOTIS message definitions
```

| Part | Origin | What we changed |
|---|---|---|
| `omx_calibration` | Written by us | Entire package |
| `peg_detection` | Written by us | Entire package |
| `open_manipulator` | ROBOTIS (open source) | Base motor in multi-turn mode, extended joint 1 limits (Section 2.4) |
| `open_manipulator_msgs` | ROBOTIS (open source) | Unchanged |

---

## 2. What we built

### 2.1 System overview

```
 RealSense D435i ──► peg_detection ──► /peg_detector/scene (JSON, robot base frame)
  (colour + depth)    colour masks, depth top-face,             │
                      ArUco + table plane, shape fit            ▼
                                                     omx_calibration / pick_place
                                                     IK (damped least squares),
                                                     straight-line Cartesian paths
                                                                │
 OpenMANIPULATOR-X ◄── ros2_control (ROBOTIS) ◄── /arm_controller/joint_trajectory
  Dynamixel ID 11–15                               /gripper_controller/gripper_cmd
```

### 2.2 Approach

**1. Perception (`peg_detection`)**
- Each peg is segmented by colour (HSV masks) in the colour image. The aligned depth image is
  deprojected pixel by pixel into 3-D points.
- **Top-face extraction:** a tall peg seen at an angle shows its side as well as its top. The
  centroid of the coloured blob would therefore lie between the top and the base. Instead, only the
  points within a thin height band below the highest part are kept, so the result is the real top
  face.
- **Known-shape fit:** the top-face outline is ragged at depth edges. We fit the known geometry to
  it: a circle for the cylinder, and a regular polygon of known side length with robust trimmed
  least squares for the square, hexagon and triangle. This gives the centre and the rotation (yaw)
  of each peg. In tests with one side of the outline bulging by 6 mm, the centre error dropped from
  3.4 mm to 0.5 mm.
- **Sockets:** the hole outlines are intersected with the board plane, giving the centre and the
  rotation of each socket.
- Pegs are matched to their sockets by colour and shape. If several objects of one colour are
  visible, only the one at a plausible height and position is kept.

**2. Camera-to-robot calibration**
- A small ArUco marker (DICT_4X4_50, ID 4) placed at a measured position (38 mm from the base
  rotation axis) defines the board origin and its x direction.
- **Table-plane correction:** from a 40-pixel marker, the camera cannot judge the table's tilt
  precisely. A tilt of a few degrees made peg heights vary from 9 to 13 cm across the board. The
  tilt is therefore taken from a RANSAC plane fitted to the whole table in the depth image (about
  15,000 points), and the marker only provides the origin and the direction.
- The camera pose is averaged over 30 frames and then frozen, so the arm may cover the marker
  during operation.
- **Ground-truth fit (`fit_camera`):** pegs are placed at hand-measured positions. The tool finds the
  rigid transform (rotation including tilt, plus translation) that best maps the camera's readings
  onto the measured positions (Kabsch / least squares). It then applies the transform to the running
  detector and saves it.

**3. Kinematics and motion (`omx_calibration`)**
- Forward kinematics from the ROBOTIS URDF geometry. Inverse kinematics by **damped least squares**
  with adaptive damping, solving for position plus gripper pitch (4 DOF → 4 targets).
- **Constant-pitch planning:** the far sockets cannot be reached with the gripper pointing straight
  down. The planner searches for the steepest gripper pitch that reaches *every* waypoint, and keeps
  it for the whole pick-and-place. The peg is gripped and placed at the same tilt, so it stays
  vertical.
- Every move is a straight Cartesian line, solved by IK every 5 mm and sent as one joint trajectory.
- Sequence per peg: open → above peg → down → close → lift → over socket → down → release → up → home.
- **Alignment prediction:** before moving, the planner predicts how the gripper squeezes a peg
  straight and how far it will be rotated relative to its socket (Section 3.1). It reports this,
  and refuses insertions beyond a set limit unless `--force` is given.
- Safety: the planner checks the joint error after every segment, `q` or Ctrl+C stops the arm and
  holds it in place, and `go_home` recovers from any pose (open the gripper, lift, then home).

**4. Setup-specific engineering**
- The board lies behind the robot, at a base angle of about 180°. That is exactly where a standard
  Dynamixel base would have to turn almost a full revolution the other way. We switched the base
  motor to **multi-turn (Extended Position) mode**, extended the joint 1 limits, and made all motion
  code choose the **shortest equivalent base angle**. The arm therefore never swings the long way
  round.
- A custom home pose that keeps the arm clear of the camera's view of the board.
- Hand-measured corrections: the robot base frame sits 3.6 cm below the board surface, the grip point
  is offset from the ROBOTIS tool point, and small per-socket place offsets were tuned by
  observation.

### 2.3 Key features
- Fully automatic detection of the pegs and sockets, with each peg assigned to its socket
- Peg yaw and socket yaw estimation, with a squeeze-based alignment prediction
- Robust camera calibration: table-plane tilt plus ground-truth least-squares fit
- Single-pitch Cartesian pick-and-place with shortest-path base motion past ±180°
- Step mode (Enter before each move), dry-run mode (hover only), check mode (analysis only)
- Emergency stop with Ctrl+C or `q`, and a safe `go_home` from any pose
- Diagnostic overlay image showing the detections, the marker axes and coordinates

### 2.4 Technologies
| Area | Used |
|---|---|
| Middleware | ROS 2 Jazzy (Ubuntu 24.04), ros2_control, JointTrajectoryController, GripperActionController, TF2 |
| Robot | ROBOTIS OpenMANIPULATOR-X, Dynamixel XM430-W350 (Protocol 2.0, 1 Mbps), U2D2, `dynamixel_hardware_interface` |
| Camera | Intel RealSense D435i, `realsense2_camera` (aligned depth) |
| Vision | OpenCV 4.10 (HSV segmentation, contours, ArUco), NumPy (deprojection, RANSAC, SVD plane fits, circle and polygon fits) |
| Robotics maths | Forward kinematics, damped-least-squares IK, Kabsch rigid fitting, Cartesian interpolation |
| Language | Python 3.12 |

Changes to the ROBOTIS stack (in `src/robotis/open_manipulator/open_manipulator_description`):
- `ros2_control/open_manipulator_x_position.ros2_control.xacro`: ID 11 `Operating Mode` 3 → **4** (multi-turn)
- `urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro`: joint 1 limits ±π → **±6π**

---

## 3. Learnings and limitations

### 3.1 The square, hexagon and triangle cannot be aligned exactly: a degrees-of-freedom limit

Inserting a prismatic peg into its socket needs **five** constraints:

| Constraint | Count |
|---|---|
| Position of the peg (x, y, z) | 3 |
| Approach angle of the gripper (pitch: the peg must stay vertical) | 1 |
| Rotation of the peg about the vertical axis (yaw), to match the socket | 1 |
| **Total** | **5** |

The OpenMANIPULATOR-X has only **four** joints: base, shoulder, elbow, wrist pitch. It has **no
wrist roll**. The three pitch joints all rotate in the same vertical plane, and that plane always
points along the line from the base to the gripper. The gripper's yaw is therefore fixed at

&nbsp;&nbsp;&nbsp;&nbsp;**gripper yaw = base angle = atan2(y, x)** of the target point.

At a given socket, the gripper's yaw is fully determined by where the socket is, and it cannot be
set independently. With 4 DOF and 5 constraints, one constraint has to give, and it is the yaw.

```
  SIDE VIEW: why there is no yaw control          TOP VIEW: yaw is fixed by the socket's position

          wrist (pitch)                                     socket (fixed by the board)
            ○═══╗                                          ┌───────────┐
           ╱    ║ gripper                                  │           │
  elbow   ╱     ║ (fingers close                           │     +     │
 (pitch) ○      ║  sideways, out                           │           │
          ╲        of the page)                            └─────╱─────┘
           ╲                                                    ╱
            ○ shoulder (pitch)                                 ╱  arm line, angle φ = atan2(y, x)
            │                                                 ╱   = gripper yaw (no choice)
       ═════◉═════ base (yaw): the ONLY joint                ◉ base
                   that turns about the vertical

  Shoulder, elbow and wrist all rotate in the same vertical plane, so they set
  x, z and pitch. Only the base sets the direction of that plane, and that
  direction must point at the target. Nothing is left to turn the gripper.
```

What the peg looks like when it arrives over its socket (rotation exaggerated):

```
                        ╱╲
                 ┌─────╱──╲─────┐
                 │    ╱    ╲    │        ─── square socket: orientation θ (fixed by the board)
                 │   ╱      ╲   │
                 │  ╱   +    ╲  │        ╱╲  square peg: orientation φ + k·90°
                 │  ╲        ╱  │            (set by the gripper = by the socket's position)
                 │   ╲      ╱   │
                 │    ╲    ╱    │        Misalignment Δ = θ − φ  (mod 90° for a square)
                 └─────╲──╱─────┘        Corners hit the rim unless Δ ≈ 0.
                        ╲╱               On our board: Δ ≈ 5–10°.
```

What happens in practice:
- When the parallel fingers close on a square or hexagonal peg, they **squeeze it straight**, so its
  faces lie flat against the fingers whatever its starting angle. This makes the grip repeatable, but
  it also locks the peg's yaw to the gripper.
- Over the socket, the peg's yaw is then fixed by the socket's position. It matches the socket's own
  rotation only by chance. On our board the leftover error is about **5–10°** for the square and
  hexagon.
- The triangle can settle in one of **two** ways between the fingers. In one way it is a few degrees
  off; in the other it is 60° off and cannot go in.
- Putting the peg down and gripping it again does not help: the peg's yaw relative to the
  base-to-peg line stays the same through any grip and release.
- The cylinder is rotationally symmetric, so it is unaffected and inserts reliably.

The software predicts this misalignment before moving (`pick_place --check`), and the per-socket
place offsets reduce its effect, but they cannot remove it. **Fixes** would need hardware: a wrist
roll joint (a 5-DOF arm), a turntable under the board, or a compliant, chamfered socket. Pushing a
peg to turn it on the table is possible with this arm, but it is unreliable.

### 3.2 Collision with the neighbouring peg

The demo video shows a collision **while the arm turns after a peg has been placed**. The gripper is
still open, and an open gripper is wider than the gap between neighbouring sockets: the holes are
about 7.5 cm apart, and a peg standing in a neighbouring socket sticks up 10 cm. When the base turns
toward the next target or home, the open fingers sweep sideways at peg height and hit the
neighbouring peg.

```
  SIDE VIEW along the socket row, just after release (before the base turns)

                                 ┃            ┃   open fingers, only 4 cm above the
                                 ┃            ┃   release height
     ┌─────┐                     ┃   ┌────┐   ┃
     │     │  ◄═══ base turns ═══┃   │    │   ┃
     │peg  │      fingers sweep  ┃   │peg │   ┃
     │in   │      sideways       ┃   │just│   ┃
     │next │      at this height     │put │
     │hole │                         │in  │
  ═══╧═════╧═════════════════════════╧════╧═══════ board
        ◄──────── ≈ 7.5 cm (hole to hole) ────────►

  open-finger span + clearance  >  free gap to the neighbouring peg   →  collision
  lowest fingertip (≈ 8 cm high) < neighbouring peg top (10 cm)       →  they meet it
```

Causes:
- The socket spacing is smaller than the open gripper's width plus clearance.
- After release, the gripper only moves 4 cm up before the base turns, which is below the top of the
  neighbouring pegs.
- The joint-space move to home does not check for obstacles.

Possible fixes: close the gripper after leaving the peg and before turning; retreat vertically to
above the tallest peg (more than 10 cm above the board) before any base rotation; fill the sockets
in an order that keeps already-placed pegs out of the sweep path; add collision checking against
the detected pegs.

### 3.3 Other learnings
- **Calibration matters more than the IK.** Once the peg coordinates are right in the robot frame,
  the IK part is straightforward. Most of the work went into camera-to-robot accuracy: marker tilt,
  the top face versus the blob centroid, and ground-truth fitting.
- **A small fiducial gives a poor tilt.** A 40-pixel marker gives an accurate position but a noisy
  orientation. Using the whole table plane from the depth image fixed heights that varied by 4 cm
  across the board.
- **Reach shapes the plan.** The far sockets are near the edge of the workspace. The gripper cannot
  point straight down there, which led to the constant-pitch strategy.
- **The hardware state must be known.** Lessons from setting up the hardware:
  - Motor position counters that shift after configuration writes;
  - a gripper current limit set below the configured goal current;
  - a 12 V supply that was off;
  - the U2D2's USB latency and `brltty` conflicts.

  The rule we adopted: power-cycle before every session, and let only ROS write to the motors.
- **Base wrap-around.** A single-turn base at ±180° forces long swings. Multi-turn mode with
  shortest-angle selection removes the problem without changing the physical setup.
- **Hand-tuned values do not transfer.** The grip depth, tool length and socket offsets were tuned on
  our setup and must be re-checked on any other one.

---

## 4. Setup from scratch

### 4.1 Requirements

**Computer:** Ubuntu **24.04** (64-bit, native install), one **USB 3** port for the camera, one USB
port for the U2D2, internet during installation, about 10 GB of free disk space.

**Hardware:**
- OpenMANIPULATOR-X with Dynamixel XM430-W350:
  - IDs **11** base, **12** shoulder, **13** elbow, **14** wrist, **15** gripper;
  - 1 Mbps, Protocol 2.0.
- U2D2 + U2D2 Power Hub with a **12 V** supply.
- Intel RealSense **D435i** with a USB 3 cable.
- Peg board with 4 sockets, 4 pegs, and the ArUco marker from `markers/`:
  - DICT_4X4_50, **ID 4**;
  - printed so that the black square is **37.4 mm**.

### 4.2 Install ROS 2 Jazzy
Skip this if `/opt/ros/jazzy/setup.bash` already exists.
```bash
sudo apt update && sudo apt install -y software-properties-common curl
sudo add-apt-repository -y universe
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y ros-jazzy-desktop ros-dev-tools
```
If a step fails, follow the official guide:
<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debians.html>

Check: `source /opt/ros/jazzy/setup.bash && ros2 --help | head -3`

### 4.3 Unpack the workspace
```bash
cd ~
unzip -q ~/Downloads/ITR_Task1_PegSorting.zip      # creates ~/ITR_Task1_PegSorting
cd ~/ITR_Task1_PegSorting
ls src            # omx_calibration  peg_detection  robotis
```

### 4.4 Install the dependencies
**ROS packages:**
```bash
sudo apt install -y python3-pip python3-numpy python3-yaml python3-rosdep \
  python3-colcon-common-extensions \
  ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-control-msgs \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-rqt-image-view \
  ros-jazzy-tf2-ros ros-jazzy-message-filters ros-jazzy-std-srvs \
  ros-jazzy-dynamixel-sdk ros-jazzy-dynamixel-hardware-interface \
  ros-jazzy-realsense2-camera
[ -f /etc/ros/rosdep/sources.list.d/20-default.list ] || sudo rosdep init
rosdep update
cd ~/ITR_Task1_PegSorting
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -y --rosdistro jazzy
```

**RealSense USB permissions:**
```bash
sudo curl -sSL -o /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

**Python:** OpenCV ≥ 4.7 is required for the ArUco detector. Install it **without** dependencies, so
that the system NumPy used by ROS is kept:
```bash
python3 -m pip install --user --break-system-packages --no-deps opencv-python-headless==4.10.0.84
python3 -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__); assert hasattr(cv2.aruco, 'ArucoDetector')"
```
Expected: `OpenCV 4.10.0 NumPy 1.26.x`.

### 4.5 Configure the U2D2 (serial port)
```bash
# 1 ms USB latency + permissions for the U2D2 (FTDI)
echo 'KERNEL=="ttyUSB*", DRIVERS=="ftdi_sio", MODE="0666", ATTR{device/latency_timer}="1"' | \
  sudo tee /etc/udev/rules.d/99-omx-u2d2.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo apt purge -y brltty               # this service grabs FTDI adapters
sudo usermod -aG dialout $USER
```
**Log out and back in.** Then, with the U2D2 plugged in:
```bash
ls /dev/ttyUSB0                                          # exists
cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer    # 1
groups | grep -o dialout                                 # dialout
```

### 4.6 Build
```bash
cd ~/ITR_Task1_PegSorting
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
grep -qxF "source ~/ITR_Task1_PegSorting/install/setup.bash" ~/.bashrc || echo "source ~/ITR_Task1_PegSorting/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
ros2 pkg list | grep -E "omx_calibration|peg_detection|open_manipulator_bringup"
```
The build must finish without `failed` packages, and the last command must list all three packages.
After editing any file in `src/`, rebuild with
`colcon build --symlink-install && source install/setup.bash`.

### 4.7 Hardware setup
1. **Layout:**
   - The board, the sockets and the marker are **behind** the robot.
   - The marker's centre is **38 mm from the base rotation axis**, on the line from the robot toward
     the board.
   - The camera is fixed above the board. The board, the marker and all peg start positions must be
     in view.
   - No other objects of the four peg colours may be in view.
2. **Connect:** U2D2 → USB; Power Hub → **12 V on**; RealSense → **USB 3** port.
3. **Dynamixel settings** (check once in ROBOTIS Dynamixel Wizard 2.0 with torque off, then close
   the Wizard):

   | Item | ID 11–14 | ID 15 |
   |---|---|---|
   | Current Limit | 1193 | 1193 |
   | PWM Limit | 885 | 885 |
   | Homing Offset | 0 | 0 |
   | Max / Min Position Limit | 4095 / 0 | 4095 / 0 |

**Coordinate frame used everywhere:**
- origin on the base rotation axis, at board height;
- **+x** from the board toward the robot;
- **+y** to the left when looking along +x;
- **+z** up;
- units are m in the software and cm in printouts.

---

## 5. Running the system

### 5.1 Start-up (every session)
1. **Switch the 12 V supply off for 5 s, then on.** This resets the motors' position counters.
2. **Terminal 1:** arm, camera, detector and image viewer:
   ```bash
   ros2 launch omx_calibration bringup.launch.py
   ```
   The arm switches its torque on and holds its position. It does not move at start-up.
3. **Terminal 2:** move to the home pose:
   ```bash
   ros2 run omx_calibration go_home
   ```
4. **Check:**
   - the image window shows the outlined pegs and sockets;
   - the bottom line reads **`camera pose FROZEN`**;
   - the marker shows coloured axes.

Launch options: `vision:=false` (arm only), `arm:=false` (camera only), `viewer:=false`,
`port_name:=/dev/ttyUSB1`.

### 5.2 Functions and how to access them

| Function | Command |
|---|---|
| Start everything (arm + camera + vision + viewer) | `ros2 launch omx_calibration bringup.launch.py` |
| Move to home safely from any pose | `ros2 run omx_calibration go_home` |
| System check (ROS, serial port, camera) | `ros2 run omx_calibration preflight` |
| View the detections | `ros2 run rqt_image_view rqt_image_view /peg_detector/annotated_image` |
| Read peg and socket coordinates | `ros2 topic echo /peg_detector/scene --once` |
| Calibrate the camera to the robot | `ros2 run omx_calibration fit_camera` |
| Re-measure the camera pose after moving the camera | `ros2 service call /peg_detector/recalibrate std_srvs/srv/Trigger` |
| Analyse one peg (no motion) | `ros2 run omx_calibration pick_place --colour blue --check` |
| Rehearse without gripping | `ros2 run omx_calibration pick_place --colour green --dry-run` |
| Sort one peg | `ros2 run omx_calibration pick_place --colour green` (`blue`, `orange`, `yellow`) |
| Sort all four | `ros2 run omx_calibration pick_place --colour all` |
| Open / close the gripper by hand | `ros2 action send_goal /gripper_controller/gripper_cmd control_msgs/action/GripperCommand "{command: {position: 0.019}}"` (open) / `-0.010` (close) |

### 5.3 Camera calibration (once per setup, and again whenever the camera or the robot moves)
```bash
ros2 run omx_calibration fit_camera
```
```
fit_camera> reset                      # clear any previous correction
fit_camera> green -14 -6.3 10          # colour, then the MEASURED x y z of its top centre in cm
fit_camera> blue -11.5 10.5 10         # ... 4-6 points spread over the work area
fit_camera> fit                        # shows the error per point before and after
fit_camera> apply                      # activates and saves the correction
fit_camera> quit
```
After `fit`, every point should be within about 0.5 cm. Use `undo` to remove a mismeasured point.

### 5.4 Peg sorting
With the arm at home and the pegs standing upright in the camera's view:
```bash
ros2 run omx_calibration pick_place --colour all
```
- The program waits for **Enter** before each move. Use `--no-step` to run continuously.
- It prints the detected positions, the alignment prediction and the planned waypoints.
- **Stop at any time** with `q` at a prompt or **Ctrl+C**. The arm holds its position. Then clear the
  area and run `go_home`.

Options:

| Option | Meaning | Default |
|---|---|---|
| `--colour` | `green`, `blue`, `orange`, `yellow`, `all` | `green` |
| `--check` | analysis only, no motion | |
| `--dry-run` | hover 3 cm above the peg and over the socket, never grip | |
| `--no-step` | do not wait for Enter | |
| `--force` | insert even when the predicted misalignment exceeds the limit | |
| `--grip-depth` | grip point below the peg top, m (higher value = lower grip) | −0.04 |
| `--tool-length` | wrist axis to grip point, m | 0.085 |
| `--max-misalign` | maximum predicted yaw error accepted, degrees | 12 |

Per-socket place offsets (`SOCKET_OFFSET`, metres) and the other tuned values are at the top of
`src/omx_calibration/omx_calibration/pick_place.py`. Rebuild after editing them.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `ping ID 11 ... no status packet` | 12 V off, U2D2 unplugged, Dynamixel Wizard still open, or the latency/permission rules from 4.5 are missing |
| `/dev/ttyUSB0` missing, `error -71` in `dmesg` | Use another USB cable or port; make sure `brltty` is removed |
| `symbol lookup error ... fastcdr` | `sudo apt update && sudo apt full-upgrade -y`, then `rm -rf build install log` and rebuild |
| `Goal Current ... exceeds the limit` (ID 15) | Set the gripper's Current Limit to 1193 in Dynamixel Wizard |
| Gripper moves the wrong way, or reads outside −0.010…+0.020 | Power-cycle the 12 V and relaunch; never let another program write to the motors while ROS runs |
| `No RealSense devices were found` | Replug into USB 3; check the udev rule (4.4) |
| `MARKER: not visible` | Marker must be DICT_4X4_50 ID 4, uncovered, well lit; arm at home during start-up |
| Heights differ across the board | `ros2 param get /peg_detector marker.refine_mode` must return `table`; redo the calibration (5.3) |
| A peg is reported at an impossible position | Another object of that colour is in view. Remove it. |
| `NOT REACHABLE with any gripper pitch` | Peg outside the arm's reach: move it closer to the base |
| `joint error > 8 deg ... stopping` | The arm was blocked. Clear the path, run `go_home`, try `--dry-run` |
| Peg dragged instead of lifted | Grip lower on the peg: increase `--grip-depth` slightly (e.g. −0.035) |
| `go_home` does not move | Gripper caught on a peg or arm blocked. `go_home` opens the gripper and lifts first; clear the area and retry. |

---

## 7. Reference

**Topics**

| Topic | Content |
|---|---|
| `/peg_detector/scene` | JSON with pegs (`x`, `y`, `z_top`, `yaw_deg`, `target_socket`) and sockets (`x`, `y`, `z`, `yaw_deg`), in m, robot frame |
| `/peg_detector/annotated_image` | Camera image with detections |
| `/joint_states` | `joint1`–`joint4`, `gripper_left_joint` |
| `/arm_controller/joint_trajectory` | Arm commands |
| `/gripper_controller/gripper_cmd` | Gripper action (`+0.019` open, `−0.010` closed) |

**Main files**

| File | Purpose |
|---|---|
| `src/omx_calibration/omx_calibration/kinematics.py` | Forward kinematics, damped-least-squares IK, shortest base angle |
| `src/omx_calibration/omx_calibration/pick_place.py` | Planning, alignment prediction, pick-and-place sequence, tuned values |
| `src/omx_calibration/omx_calibration/fit_camera.py` | Ground-truth camera-to-robot fit |
| `src/omx_calibration/omx_calibration/go_home.py` | Safe homing |
| `src/omx_calibration/omx_calibration/wizard.py` | Robot interface, home pose, gripper values, guided calibration wizard |
| `src/omx_calibration/launch/bringup.launch.py` | Starts the arm, camera, detector and viewer |
| `src/peg_detection/peg_detection/peg_detector_node.py` | Detection node, camera pose, scene publisher |
| `src/peg_detection/peg_detection/geometry3d.py` | Deprojection, top-face extraction, table plane, circle and polygon fits |
| `src/peg_detection/config/detection_params.yaml` | Colours, marker, frame offsets, fitted correction |
| `~/omx_calibration/camera_pose.yaml` | Saved camera pose (created at runtime) |
