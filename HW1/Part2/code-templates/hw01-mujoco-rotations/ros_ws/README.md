# ros_ws -- HW1 Task 3 (optional/bonus)

A minimal ROS 2 (ament_python) workspace containing one package,
`hw01_tf_demo`, which broadcasts a fixed `space_frame` and a moving
`body_frame` as TF transforms so you can watch current-frame vs.
fixed-frame rotation composition live in `rviz2`.

This task is optional/bonus for HW1 -- Tasks 1 and 2 (MuJoCo only, see
`../README.md`) are the required core of Part 2. Do this one if you have
ROS 2 already set up, or want the practice.

## Build & run

```bash
# from ros_ws/
colcon build --symlink-install
source install/setup.bash
ros2 run hw01_tf_demo tf_broadcaster_node
```

In a second terminal:

```bash
source install/setup.bash
rviz2
# Add a "TF" display, set the Fixed Frame (top-left panel) to "space_frame"
```

While it's running, try toggling the composition rule live:

```bash
ros2 param set /hw01_tf_broadcaster compose_frame fixed
ros2 param set /hw01_tf_broadcaster compose_frame current
```

## What you need to implement

See the `TODO(student)` block in
`src/hw01_tf_demo/hw01_tf_demo/tf_broadcaster_node.py::on_timer`. It's the
same current-frame-vs-fixed-frame composition logic as
`compose_sequence()` in `../scripts/01_rotation_sandbox.py` -- reuse your
reasoning (and, if you want, literally reuse the code) rather than
re-deriving it from scratch.

## Notes

- This template targets ROS 2 (`rclpy`, `tf2_ros`, ament_python packaging).
  It was written and syntax-checked without a live ROS 2 install available,
  so double check package names/APIs against whatever ROS 2 distro you have
  (Humble/Iron/Jazzy) if something doesn't match.
- `R_to_quat_xyzw` intentionally uses ROS's XYZW quaternion order, which is
  different from the WXYZ order MuJoCo uses in `../scripts/utils.py` --
  worth noticing if you ever pass orientation data between the two.
