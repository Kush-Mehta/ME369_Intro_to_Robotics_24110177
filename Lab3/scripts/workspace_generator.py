"""
workspace_generator.py
======================
Generates Task Workspace and Dexterous Workspace for:
  - HEAL robot  (6-DOF, reads joint limits from arm_gripper.xml)
  - Franka Panda (7-DOF, uses official joint limits + self-contained minimal XML)

Outputs (in scripts/ directory):
  heal_workspace.csv            heal_dexterous_workspace.csv
  franka_workspace.csv          franka_dexterous_workspace.csv

Usage:
  python3 workspace_generator.py              # both robots
  python3 workspace_generator.py --robot HEAL
  python3 workspace_generator.py --robot FRANKA
  python3 workspace_generator.py --samples 12 --yaw 8

Dexterous workspace definition (Tutorial 4):
  A TCP position is "dexterous" if the robot can reach it with
  ALL N_YAW evenly-spaced yaw angles feasible — i.e. arbitrary
  end-effector orientation about yaw is possible at that point.
"""

import os, csv, re, argparse, tempfile
from itertools import product

import mujoco
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--robot",   choices=["HEAL", "FRANKA", "BOTH"], default="BOTH")
parser.add_argument("--samples", type=int, default=8,
                    help="Samples per joint (default 8; use 12+ for publication quality)")
parser.add_argument("--yaw",     type=int, default=6,
                    help="Yaw samples for dexterous check (default 6)")
args = parser.parse_args()

N_SAMPLES = args.samples
N_YAW     = args.yaw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ─────────────────────────────────────────────────────────────────────────────
# ROBOT CONFIGS
# ─────────────────────────────────────────────────────────────────────────────
ROBOTS = {
    "HEAL": {
        # arm_gripper.xml — all 6 joints ±π (±3.14159 rad), read from XML.
        # workspace_generator uses limits_override=None so joint limits come
        # directly from the XML joint range="" attributes.  The XML and this
        # workspace analysis are therefore always in sync — the XML is the
        # single source of truth for HEAL joint limits.
        "xml": os.path.join(REPO_ROOT, "robot_descriptions", "arm_gripper.xml"),
        "joints": ["joint_1", "joint_2", "joint_3",
                   "joint_4", "joint_5", "joint_6"],
        "yaw_joint":        "joint_6",   # last DOF = wrist yaw
        "site":             "right_center",
        "limits_override":  None,        # read from XML (all joints ±π)
        "clean_mesh":       "c-a01-85-open",   # missing visual mesh to strip
    },
    "FRANKA": {
        # Self-contained minimal XML — no missing includes or meshes
        "xml": os.path.join(REPO_ROOT, "robot_descriptions", "franka_panda_minimal.xml"),
        "joints": ["joint1", "joint2", "joint3",
                   "joint4", "joint5", "joint6", "joint7"],
        "yaw_joint":        "joint7",    # wrist rotation joint
        "site":             "tcp",       # defined in franka_panda_minimal.xml
        # Official Franka Emika joint limits (deg → rad)
        # Source: Franka datasheet 2022, confirmed in MuJoCo Menagerie
        "limits_override": np.deg2rad([
            [-166,  166],   # joint1  A1
            [-101,  101],   # joint2  A2
            [-166,  166],   # joint3  A3
            [-176,   -4],   # joint4  A4
            [-166,  166],   # joint5  A5
            [  -1,  215],   # joint6  A6
            [-166,  166],   # joint7  A7
        ]),
        "clean_mesh": None,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def strip_mesh_geom(xml_file, mesh_name):
    """Remove one <geom> referencing a missing mesh so MuJoCo can load the file."""
    with open(xml_file, "r", encoding="utf-8") as f:
        text = f.read()
    text, n = re.subn(
        r'[ \t]*<geom\b[^>]*mesh="' + re.escape(mesh_name) + r'"[^>]*/>\s*',
        "", text, count=1
    )
    if n == 0:
        return xml_file, False
    fd, tmp = tempfile.mkstemp(suffix="_ws.xml", dir=os.path.dirname(xml_file))
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    return tmp, True


def get_joint_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if i < 0:
        raise RuntimeError(f"Joint '{name}' not found in model.")
    return i


def get_site_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if i < 0:
        raise RuntimeError(f"Site '{name}' not found in model.")
    return i


def get_tcp(model, data, qpos_addrs, q_vals, site_id):
    """Set joint angles, run forward kinematics, return TCP xyz."""
    for addr, val in zip(qpos_addrs, q_vals):
        data.qpos[addr] = val
    mujoco.mj_forward(model, data)
    return data.site_xpos[site_id].copy()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def generate(robot_name, cfg):
    print()
    print("=" * 65)
    print(f"  {robot_name} WORKSPACE GENERATOR")
    print("=" * 65)

    xml_file = cfg["xml"]
    if not os.path.exists(xml_file):
        print(f"[SKIP] XML not found:\n  {xml_file}")
        return

    # ── Strip missing visual mesh if needed ──────────────────────────────────
    tmp_xml = None
    load_xml = xml_file
    if cfg["clean_mesh"]:
        load_xml, made_tmp = strip_mesh_geom(xml_file, cfg["clean_mesh"])
        if made_tmp:
            tmp_xml = load_xml
            print(f"Note: Removed missing mesh '{cfg['clean_mesh']}' (visual only).")

    try:
        model = mujoco.MjModel.from_xml_path(load_xml)
        data  = mujoco.MjData(model)
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        if tmp_xml and os.path.exists(tmp_xml):
            os.remove(tmp_xml)
        return

    # ── Joints ───────────────────────────────────────────────────────────────
    joint_names = cfg["joints"]
    joint_ids   = [get_joint_id(model, n) for n in joint_names]
    qpos_addrs  = [model.jnt_qposadr[i] for i in joint_ids]
    n_joints    = len(joint_names)

    # ── Joint limits ─────────────────────────────────────────────────────────
    if cfg["limits_override"] is not None:
        limits = np.array(cfg["limits_override"], dtype=float)
        print("Using official joint limits (from user spec / Franka datasheet).")
        # Also write limits into the model so MuJoCo clamp is consistent
        for i, jid in enumerate(joint_ids):
            model.jnt_range[jid] = limits[i]
    else:
        limits = np.array([model.jnt_range[i] for i in joint_ids], dtype=float)
        print("Joint limits read from XML.")

    limits_deg = np.rad2deg(limits)
    print(f"\n  {'Joint':<22} {'Min (°)':>9} {'Max (°)':>9}")
    print("  " + "-" * 42)
    for name, (lo, hi) in zip(joint_names, limits_deg):
        print(f"  {name:<22} {lo:>9.1f} {hi:>9.1f}")

    # ── TCP site ─────────────────────────────────────────────────────────────
    site_id = get_site_id(model, cfg["site"])

    # ── Yaw joint index (within joint list) ──────────────────────────────────
    yaw_idx = joint_names.index(cfg["yaw_joint"])

    # ── Sample grids ─────────────────────────────────────────────────────────
    joint_values = [
        np.linspace(limits[i, 0], limits[i, 1], N_SAMPLES)
        for i in range(n_joints)
    ]
    yaw_values = np.linspace(limits[yaw_idx, 0], limits[yaw_idx, 1], N_YAW)

    total = N_SAMPLES ** n_joints
    print(f"\n  Samples / joint  : {N_SAMPLES}")
    print(f"  Total configs    : {total:,}")
    print(f"  Yaw samples      : {N_YAW}  (dexterous check)")

    # ── Output CSV paths ─────────────────────────────────────────────────────
    task_csv = os.path.join(SCRIPT_DIR, f"{robot_name.lower()}_workspace.csv")
    dext_csv = os.path.join(SCRIPT_DIR, f"{robot_name.lower()}_dexterous_workspace.csv")

    header = [f"theta{i+1}" for i in range(n_joints)] + ["x", "y", "z"]

    # ── [1/2] Task workspace ─────────────────────────────────────────────────
    print(f"\n[1/2] Generating task workspace ...")

    # unique_positions: rounded-key → base q for dexterous check
    unique_positions = {}

    with open(task_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for count, q in enumerate(product(*joint_values), 1):
            q = list(q)
            xyz = get_tcp(model, data, qpos_addrs, q, site_id)
            writer.writerow([*np.rad2deg(q), *xyz])

            key = tuple(np.round(xyz, 2))   # 1 cm grid for uniqueness
            if key not in unique_positions:
                unique_positions[key] = q

            if count % 20_000 == 0 or count == total:
                print(f"   {count:>10,} / {total:,}  ({100*count/total:.1f}%)")

    print(f"   Saved: {task_csv}")

    # ── [2/2] Dexterous workspace ─────────────────────────────────────────────
    # Definition (Tutorial 4):
    #   A TCP position is dexterous ↔ the robot can achieve it with
    #   *all* sampled yaw angles of the last joint.
    #
    # Implementation:
    #   For each unique TCP position p (from task workspace), take the
    #   joint config q that produced it. Sweep the yaw joint through
    #   N_YAW values across its full range while keeping all other
    #   joints fixed. If FK at every yaw still places the TCP within
    #   TOL of p (pure yaw doesn't translate TCP, only orientates it),
    #   then p is dexterous.
    #
    #   Note: pure wrist-yaw rotation keeps TCP position constant by
    #   definition — the check validates that the joint is within its
    #   own limits at each yaw sample, which the linspace guarantees.
    #   We therefore flag all task-workspace points as dexterous here
    #   unless a yaw value is out of the joint's own range. Because we
    #   sampled yaw within limits, the "real" dexterous subset is the
    #   intersection of task-workspace positions reachable at every yaw
    #   WITHOUT violating other-joint limits. We verify this via FK.

    print(f"\n[2/2] Computing dexterous workspace ...")

    TOL = 0.015   # 1.5 cm tolerance (covers FK floating-point noise)
    dext_count = 0
    n_unique = len(unique_positions)

    with open(dext_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "is_dexterous"])

        for idx, (key, q_base) in enumerate(unique_positions.items()):
            xyz_base = np.array(key)
            dexterous = True

            for yaw_val in yaw_values:
                q_test = list(q_base)
                q_test[yaw_idx] = yaw_val
                xyz_test = get_tcp(model, data, qpos_addrs, q_test, site_id)
                if np.linalg.norm(xyz_test - xyz_base) > TOL:
                    dexterous = False
                    break

            writer.writerow([*xyz_base, int(dexterous)])
            if dexterous:
                dext_count += 1

            if (idx + 1) % 5_000 == 0:
                print(f"   Checked {idx+1:,} / {n_unique:,} positions ...")

    print(f"   Dexterous positions : {dext_count:,} / {n_unique:,} "
          f"({100*dext_count/max(n_unique,1):.1f}%)")
    print(f"   Saved: {dext_csv}")

    # cleanup temp XML
    if tmp_xml and os.path.exists(tmp_xml):
        os.remove(tmp_xml)

    print(f"\n  {robot_name} done.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
robots_to_run = ["HEAL", "FRANKA"] if args.robot == "BOTH" else [args.robot]

for rname in robots_to_run:
    generate(rname, ROBOTS[rname])

print("\nAll done. Run plot_workspace.py to visualise.")
