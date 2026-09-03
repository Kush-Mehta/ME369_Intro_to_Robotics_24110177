# HW1 Part 2 starter -- MuJoCo rotation sandbox

Starter code for ME 639 HW1, Part 2 (see `assignments/hw01-rotations/hw01.tex`
for the full task descriptions). This supersedes the earlier "build a web
app to play with rotations" homework mentioned in the 04-Rotations and
05-More-Rotations lecture notes -- same idea, now built on MuJoCo.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Sanity-check the model loads and renders:

```bash
cd scripts
python 01_rotation_sandbox.py
```

You should see a small red/blue/green "dart" (an asymmetric body, so you can
always tell how it's oriented) sitting in the MuJoCo viewer, with its body
frame axes drawn on it (red=x, green=y, blue=z) and a fixed space frame drawn
faintly at the origin.

## What's here

| Path | Status | Purpose |
|---|---|---|
| `model/asymmetric_body.xml` | complete | The MJCF scene: an asymmetric free body + body/space frame axis markers. |
| `scripts/utils.py` | complete | Rotation helpers (`hat`, `vee`, `Rx/Ry/Rz`, quaternion &lt;-&gt; rotation matrix, read/write body orientation). Use these, don't reinvent them. |
| `scripts/01_rotation_sandbox.py` | **starter -- has TODOs** | Task 1: current-frame vs. fixed-frame composition. |
| `scripts/02_verify_skew_properties.py` | **starter -- has TODOs** | Task 2: numerically verify the Problem 5 identities. |
| `ros_ws/` | **starter -- has TODOs** | Task 3 (optional/bonus): broadcast TF frames and animate in `rviz2`. See `ros_ws/README.md`. |

Every TODO is marked `TODO(student)` with a comment explaining what's
expected and why. Nothing here requires you to write simulation or
rendering boilerplate from scratch -- that part is done. What's asked of
you is the rotation reasoning: current-frame vs. fixed-frame composition,
and verifying the identities from Problem 5.

## AI use

Full AI use (including "vibe coding") is expected and encouraged for this
part, per the course AI Use Policy -- fill in the AI Use Note at the end of
`hw01.tex` with what you asked and how you used the output. The one place
AI is restricted anywhere in HW1 is Problem 5(b) in Part 1 (pen and paper),
which has nothing to do with this code.
