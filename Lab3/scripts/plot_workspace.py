"""
plot_workspace.py
=================
Plots Task Workspace and Dexterous Workspace for HEAL and/or Franka.

Produces per-robot:
  1. 3-D point cloud  (task = blue, dexterous = orange)
  2. Top-down XY occupancy map with table feasibility overlay
  3. Side-view XZ occupancy map

Usage:
  python3 plot_workspace.py            # both robots
  python3 plot_workspace.py --robot HEAL
  python3 plot_workspace.py --robot FRANKA
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — remove if you want interactive
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=["HEAL", "FRANKA", "BOTH"], default="BOTH")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GRID_RES = 0.02   # 2 cm voxel for occupancy maps

# ─────────────────────────────────────────────────────────────────────────────
def load(robot):
    task_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_workspace.csv")
    dext_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_dexterous_workspace.csv")
    if not os.path.exists(task_csv):
        print(f"[SKIP] {task_csv} not found. Run workspace_generator.py first.")
        return None, None
    task = pd.read_csv(task_csv)[["x","y","z"]].values
    dext = None
    if os.path.exists(dext_csv):
        df = pd.read_csv(dext_csv)
        dext = df[df["is_dexterous"] == 1][["x","y","z"]].values
    return task, dext


def occupancy_map(pts, res=GRID_RES, axes=(0, 1)):
    """Project points onto a 2-D grid and return (grid, x_edges, y_edges)."""
    x = pts[:, axes[0]]
    y = pts[:, axes[1]]
    x_edges = np.arange(x.min() - res, x.max() + res, res)
    y_edges = np.arange(y.min() - res, y.max() + res, res)
    grid, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    return (grid > 0).astype(float), x_edges, y_edges


def plot_robot(robot):
    task, dext = load(robot)
    if task is None:
        return

    print(f"\nPlotting {robot}: {len(task):,} task pts", end="")
    if dext is not None:
        print(f", {len(dext):,} dexterous pts", end="")
    print()

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(f"{robot} Robot — Workspace Analysis", fontsize=14, fontweight="bold")

    # ── 1. 3-D scatter ───────────────────────────────────────────────────────
    ax3d = fig.add_subplot(131, projection="3d")

    ax3d.scatter(
        task[:,0], task[:,1], task[:,2],
        c=task[:,2], cmap="Blues", s=1, alpha=0.25, label="Task WS"
    )
    if dext is not None and len(dext):
        ax3d.scatter(
            dext[:,0], dext[:,1], dext[:,2],
            c="darkorange", s=3, alpha=0.5, label="Dexterous WS"
        )

    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D Point Cloud")
    ax3d.legend(markerscale=5, fontsize=8)

    # ── 2. Top-down XY occupancy ─────────────────────────────────────────────
    ax_xy = fig.add_subplot(132)
    occ_t, xe, ye = occupancy_map(task, axes=(0, 1))

    ax_xy.imshow(
        occ_t.T, origin="lower", cmap="Blues",
        extent=[xe[0], xe[-1], ye[0], ye[-1]],
        aspect="equal", alpha=0.7
    )

    if dext is not None and len(dext):
        occ_d, xed, yed = occupancy_map(dext, axes=(0, 1))
        ax_xy.imshow(
            occ_d.T, origin="lower", cmap="Oranges",
            extent=[xed[0], xed[-1], yed[0], yed[-1]],
            aspect="equal", alpha=0.6
        )

    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-down Occupancy (XY)\nBlue=Task  Orange=Dexterous")
    ax_xy.set_aspect("equal")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], color="steelblue",  linewidth=8, alpha=0.6, label="Task WS"),
        Line2D([0],[0], color="darkorange", linewidth=8, alpha=0.6, label="Dexterous WS"),
    ]
    ax_xy.legend(handles=legend_els, fontsize=7)

    # ── 3. Side-view XZ occupancy ────────────────────────────────────────────
    ax_xz = fig.add_subplot(133)
    occ_xz, xe2, ze2 = occupancy_map(task, axes=(0, 2))
    ax_xz.imshow(
        occ_xz.T, origin="lower", cmap="Blues",
        extent=[xe2[0], xe2[-1], ze2[0], ze2[-1]],
        aspect="auto", alpha=0.7
    )
    if dext is not None and len(dext):
        occ_dxz, xed2, zed2 = occupancy_map(dext, axes=(0, 2))
        ax_xz.imshow(
            occ_dxz.T, origin="lower", cmap="Oranges",
            extent=[xed2[0], xed2[-1], zed2[0], zed2[-1]],
            aspect="auto", alpha=0.6
        )

    ax_xz.set_xlabel("X (m)"); ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Side-view Occupancy (XZ)\nBlue=Task  Orange=Dexterous")

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, f"{robot.lower()}_workspace_analysis.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
robots_to_plot = (
    ["HEAL", "FRANKA"] if args.robot == "BOTH" else [args.robot]
)

for r in robots_to_plot:
    plot_robot(r)

print("\nPlotting complete.")
