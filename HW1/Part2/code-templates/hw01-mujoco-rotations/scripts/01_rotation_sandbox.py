"""
01_rotation_sandbox.py -- HW1 Part 2, Task 1: does rotation order matter?

STARTER CODE. The model loading, viewer, and simulation loop below
are complete and working -- run this file as-is and you should see
the asymmetric dart sitting in the viewer. Your job is to fill in
the TODOs so that:

  1. The user can queue up a sequence of elemental rotations
     (about x, y, or z), each one EITHER about the current body
     frame OR about the fixed space frame (their choice).
  2. The dart's orientation updates to reflect that sequence.
  3. You can run the SAME sequence of angles twice -- once
     "current frame" and once "fixed frame" -- and see (and
     screen-record) that the final orientation is visibly
     different, exactly as you proved symbolically in HW1
     Problem 3 (Lynch & Park Ch.3 Ex.3.4-style reasoning) and
     the "Composition of Rotations" lecture derivation.

This is intentionally a plain script, not a GUI app -- editing the
`rotation_sequence` list below and re-running is a perfectly good
"sandbox." A slider UI is a nice-to-have, not a requirement. Use AI
freely here; document what you asked it for in your AI Use Note.
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "../model/asymmetric_body.xml"


# ---------------------------------------------------------------
# TODO(student): edit this sequence to try different experiments.
# Each entry is (axis, angle_radians, frame) where frame is either
# "current" (compose on the RIGHT: R_new = R_old @ R_step) or
# "fixed" (compose on the LEFT: R_new = R_step @ R_old).
# Compare the two frame choices for the SAME list of (axis, angle).
# ---------------------------------------------------------------
rotation_sequence = [
    #("z", np.deg2rad(90), "fixed"),   # TODO: try "fixed" here instead
    #("x", np.deg2rad(90), "fixed"),   # TODO: try "fixed" here instead
    ("z", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
    ("x", np.deg2rad(90), "current"),   # TODO: try "fixed" here instead
]


def compose_sequence(sequence):
    """TODO(student): implement this.

    Given a list of (axis, angle, frame) tuples, return the final
    3x3 rotation matrix R obtained by applying them in order,
    starting from R = identity.

    Hint: ELEMENTARY_ROTATIONS["x"](angle) gives you R_x(angle) etc.
    Hint: think about *why* current-frame composition is a right-
    multiply and fixed-frame composition is a left-multiply -- this
    is exactly the "Current Frame" vs. "Fixed Frame" distinction
    from the Composition-of-Rotations lecture. Don't just guess;
    check it against your Problem 3/4 reasoning.
    """
    R = np.eye(3)
    for axis, angle, frame in sequence:
        R_step = ELEMENTARY_ROTATIONS[axis](angle)
        # TODO: replace the next line with the correct composition
        if frame == "current":
            R = R @ R_step
        elif frame == "fixed":
            R = R_step @ R        
        # rule depending on `frame`.
        #raise NotImplementedError("Implement current- vs fixed-frame composition")
    return R


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    R_final = np.eye(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer open. Close the window to exit.")
        print(f"Applied sequence: {rotation_sequence}")

        # Give time to start the screen recording
        time.sleep(15)

        # -----------------------------------------------------------
        # Animate the sequence step by step.
        # Each elemental rotation is performed smoothly over 1 second.
        # -----------------------------------------------------------
        for axis, angle, frame in rotation_sequence:
            if not viewer.is_running():
                break

            R_start = R_final.copy()

            # Smoothly increase the angle from 0 to the final angle.
            start_time = time.time()
            duration = 1.0

            while viewer.is_running():
                elapsed = time.time() - start_time
                alpha = min(elapsed / duration, 1.0)

                current_angle = alpha * angle
                R_step = ELEMENTARY_ROTATIONS[axis](current_angle)

                if frame == "current":
                    R = R_start @ R_step
                elif frame == "fixed":
                    R = R_step @ R_start
                else:
                    raise ValueError(f"Unknown frame: {frame}")

                set_body_orientation(data, R)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if alpha >= 1.0:
                    break

                time.sleep(1 / 60)

            # Store the exact final orientation of this step
            R_step = ELEMENTARY_ROTATIONS[axis](angle)

            if frame == "current":
                R_final = R_start @ R_step
            else:
                R_final = R_step @ R_start

            # Hold the result for 1 second before the next rotation
            time.sleep(1)

        # Keep viewer open after the sequence finishes
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)

    # TODO(student, optional): instead of a static final pose,
    # animate the sequence step by step (e.g. interpolate each
    # elemental rotation over ~1 second) so the recording clearly
    # shows each step happening, not just the end result.


if __name__ == "__main__":
    main()

