import os
import argparse
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError:
    raise SystemExit(
        "plotly not found. Install with:  pip install plotly\n"
        "or:  pip install plotly --break-system-packages"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--robot",    choices=["HEAL", "FRANKA", "BOTH"], default="BOTH")
parser.add_argument("--max_pts",  type=int, default=50_000,
                    help="Downsample large point clouds to this many points "
                         "for faster rendering (default 50 000)")
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCHEME
# ─────────────────────────────────────────────────────────────────────────────
TCP_COLOUR  = "royalblue"
DEXT_COLOUR = "darkorange"
MARKER_SIZE_TASK = 2
MARKER_SIZE_DEXT = 3

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csvs(robot):
    """Return (task_xyz, dext_xyz_all, dext_xyz_yes) or None on missing files."""
    task_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_workspace.csv")
    dext_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_dexterous_workspace.csv")

    if not os.path.exists(task_csv) or os.path.getsize(task_csv) < 30:
        print(f"[SKIP] {os.path.basename(task_csv)} not found or empty.")
        print(f"       Run  python3 workspace_generator.py --robot {robot}  first.")
        return None

    task = pd.read_csv(task_csv)[["x", "y", "z"]].dropna().values

    dext_yes = None
    dext_all = None
    if os.path.exists(dext_csv) and os.path.getsize(dext_csv) > 30:
        df = pd.read_csv(dext_csv)
        dext_all = df[["x", "y", "z"]].dropna().values
        dext_yes = df[df["is_dexterous"] == 1][["x", "y", "z"]].dropna().values

    return task, dext_all, dext_yes


def downsample(pts, n):
    """Random downsample to at most n points."""
    if pts is None or len(pts) == 0:
        return pts
    if len(pts) <= n:
        return pts
    idx = np.random.choice(len(pts), n, replace=False)
    return pts[idx]


def axis_scene(title_z="Z (m)"):
    return dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title=title_z,
        xaxis=dict(backgroundcolor="rgb(230,230,250)", gridcolor="white",
                   showbackground=True, zerolinecolor="white"),
        yaxis=dict(backgroundcolor="rgb(230,250,230)", gridcolor="white",
                   showbackground=True, zerolinecolor="white"),
        zaxis=dict(backgroundcolor="rgb(250,230,230)", gridcolor="white",
                   showbackground=True, zerolinecolor="white"),
        aspectmode="data",
    )


def scatter3d(pts, name, colour, size, opacity=0.35):
    """Return a go.Scatter3d trace."""
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=dict(size=size, color=colour, opacity=opacity),
        name=name,
        hovertemplate="X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<extra></extra>",
    )


def compute_feasible_table(task, z_min=0.55, margin=0.05):
    """Return (x_lo, x_hi, y_lo, y_hi) of feasible table region or None."""
    above = task[task[:, 2] > z_min]
    if len(above) < 50:
        return None
    x_lo = above[:, 0].min() + margin
    x_hi = above[:, 0].max() - margin
    y_lo = above[:, 1].min() + margin
    y_hi = above[:, 1].max() - margin
    if x_hi <= x_lo or y_hi <= y_lo:
        return None
    return x_lo, x_hi, y_lo, y_hi


# ─────────────────────────────────────────────────────────────────────────────
# VIEW GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def get_robot_views(robot):
    result = load_csvs(robot)
    if result is None:
        return []

    task, dext_all, dext_yes = result
    N_MAX = args.max_pts

    task_d  = downsample(task,     N_MAX)
    dext_d  = downsample(dext_all, N_MAX)
    dext_yd = downsample(dext_yes, N_MAX)

    print(f"\n{'='*60}")
    print(f"  {robot}  |  task={len(task):,} pts", end="")
    if dext_yes is not None:
        frac = 100 * len(dext_yes) / max(len(dext_all), 1)
        print(f"  dexterous={len(dext_yes):,} ({frac:.1f}%)", end="")
    print()

    z_table = 0.55
    margin  = 0.05
    table_region = compute_feasible_table(task, z_min=z_table, margin=margin)
    if table_region:
        x_lo, x_hi, y_lo, y_hi = table_region
        print(f"  Feasible table (z>{z_table}m, {margin}m margin):")
        print(f"    X: {x_lo:.3f} → {x_hi:.3f}   L = {x_hi-x_lo:.3f} m")
        print(f"    Y: {y_lo:.3f} → {y_hi:.3f}   W = {y_hi-y_lo:.3f} m")

    views = []

    # 1. TCP Trace View
    tcp_trace = scatter3d(
        task_d, name="TCP Workspace",
        colour=task_d[:, 2],
        size=MARKER_SIZE_TASK, opacity=0.3
    )
    tcp_trace.marker.colorscale = "Blues"
    tcp_trace.marker.showscale = True
    tcp_trace.marker.colorbar = dict(title="Z (m)", len=0.6, x=1.0)
    
    tcp_title = (
        f"<b>{robot} — TCP / Task Workspace</b><br>"
        f"<sup>{len(task):,} configurations | "
        f"drag to rotate · scroll to zoom</sup>"
    )
    
    views.append({
        "label": f"{robot} TCP",
        "title": tcp_title,
        "traces": [tcp_trace]
    })

    # 2. Dexterous Trace View
    if dext_yd is not None and len(dext_yd) > 0:
        dxt_pct = 100 * len(dext_yes) / max(len(dext_all), 1)
        
        bg_trace = scatter3d(
            task_d, name="Task (bg)",
            colour=TCP_COLOUR, size=MARKER_SIZE_TASK, opacity=0.08
        )
        dext_trace = scatter3d(
            dext_yd, name="Dexterous",
            colour=DEXT_COLOUR, size=MARKER_SIZE_DEXT + 1, opacity=0.55
        )
        
        dext_title = (
            f"<b>{robot} — Dexterous Workspace</b><br>"
            f"<sup>{len(dext_yes):,} dexterous positions ({dxt_pct:.1f}%) | "
            f"drag to rotate · scroll to zoom</sup>"
        )
        
        views.append({
            "label": f"{robot} Dexterous",
            "title": dext_title,
            "traces": [bg_trace, dext_trace]
        })

    return views


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
robots_to_plot = ["HEAL", "FRANKA"] if args.robot == "BOTH" else [args.robot]

all_views = []
for r in robots_to_plot:
    all_views.extend(get_robot_views(r))

if not all_views:
    print("\nNo data to plot. Exiting.")
    exit(0)

fig = go.Figure()

# Add all traces to the figure and map their indices
trace_index = 0
view_ranges = []

for view in all_views:
    start_idx = trace_index
    for trace in view["traces"]:
        fig.add_trace(trace)
        trace_index += 1
    view_ranges.append((start_idx, trace_index))

# Set initial visibility (only first view is visible)
for i in range(trace_index):
    fig.data[i].visible = (i >= view_ranges[0][0] and i < view_ranges[0][1])

# Build dropdown menu buttons
buttons = []
for i, view in enumerate(all_views):
    visibility = [False] * trace_index
    start_idx, end_idx = view_ranges[i]
    for j in range(start_idx, end_idx):
        visibility[j] = True
        
    buttons.append(dict(
        label=view["label"],
        method="update",
        args=[
            {"visible": visibility},
            {"title": dict(text=view["title"], x=0.5, xanchor="center")}
        ]
    ))

fig.update_layout(
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.01,
        xanchor="left",
        y=1.1,
        yanchor="top",
        bgcolor="white",
        bordercolor="gray"
    )],
    title=dict(text=all_views[0]["title"], x=0.5, xanchor="center"),
    scene=axis_scene("Z (m)"),
    legend=dict(x=0.85, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=100, b=0),
    height=800,
)

out_html = os.path.join(SCRIPT_DIR, "combined_workspace_3d.html")
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"\nSaved combined interactive plot: {os.path.basename(out_html)}")
print("Open the HTML file in any browser and use the dropdown menu to toggle views.")