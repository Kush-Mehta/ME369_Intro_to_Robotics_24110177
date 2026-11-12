#!/usr/bin/env bash
# Put the BASE motor (ID 11, joint1) into Extended Position (multi-turn) mode in the
# ROBOTIS open_manipulator package, so it can pass +-180 deg (our board is behind the robot).
#
#   bash scripts/patch_robotis_base_multiturn.sh  <folder that contains open_manipulator_description>
#   e.g. bash scripts/patch_robotis_base_multiturn.sh ~/ros2_ws/src/open_manipulator
#        bash scripts/patch_robotis_base_multiturn.sh src/robotis/open_manipulator
# Then rebuild:  colcon build --symlink-install --packages-select open_manipulator_description
# Undo:          the script keeps .orig copies next to the two files.
set -euo pipefail
ROOT="${1:?give the open_manipulator folder}"
X="$ROOT/open_manipulator_description/ros2_control/open_manipulator_x_position.ros2_control.xacro"
U="$ROOT/open_manipulator_description/urdf/open_manipulator_x/open_manipulator_x_arm.urdf.xacro"
for f in "$X" "$U"; do [ -f "$f" ] || { echo "not found: $f"; exit 1; }; done
[ -f "$X.orig" ] || cp "$X" "$X.orig"
[ -f "$U.orig" ] || cp "$U" "$U.orig"

# 1) ID 11: Operating Mode 3 -> 4 (only inside the gpio block that has ID 11)
python3 - "$X" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
m = re.search(r'(<param name="ID">11</param>.*?<param name="Operating Mode">)(\d)(</param>)', s, re.S)
if not m: sys.exit('ID 11 block not found in ' + p)
s = s[:m.start(2)] + '4' + s[m.end(2):]
open(p, 'w').write(s); print('ID 11 Operating Mode ->', 4)
EOF

# 2) joint1 limits +-pi -> +-6pi (multi-turn)
python3 - "$U" <<'EOF'
import re, sys
p = sys.argv[1]; s = open(p).read()
m = re.search(r'(<joint name="\$\{prefix\}joint1".*?<limit[^>]*?lower=")([^"]*)("\s+upper=")([^"]*)(")', s, re.S)
if not m: sys.exit('joint1 limit not found in ' + p)
s = s[:m.start(2)] + '${-6*pi}' + s[m.end(2):m.start(4)] + '${6*pi}' + s[m.end(4):]
open(p, 'w').write(s); print('joint1 limits -> +-6*pi (multi-turn range)')
EOF
grep -n -A7 '"ID">11<' "$X" | grep "Operating Mode"
grep -n -A6 'joint1" type' "$U" | grep limit
echo "OK. Now rebuild open_manipulator_description and ALWAYS launch with init_position:=false,"
echo "then: ros2 run omx_calibration go_home"
