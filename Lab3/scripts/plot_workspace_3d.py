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
# COLOUR SCHEME - HIGH CONTRAST (RED & ORANGE)
# ─────────────────────────────────────────────────────────────────────────────
TCP_COLOUR  = "darkred"
DEXT_COLOUR = "orange"
MARKER_SIZE_TASK = 2
MARKER_SIZE_DEXT = 4

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_csvs(robot):
    """Return (task_xyz, dext_xyz_all, dext_xyz_yes) or None on missing/corrupt files."""
    task_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_workspace.csv")
    dext_csv = os.path.join(SCRIPT_DIR, f"{robot.lower()}_dexterous_workspace.csv")

    if not os.path.exists(task_csv) or os.path.getsize(task_csv) < 30:
        print(f"[SKIP] {os.path.basename(task_csv)} not found or empty.")
        return None

    try:
        task_df = pd.read_csv(task_csv)
        task_df.columns = task_df.columns.astype(str).str.strip().str.lower()
        
        if all(c in task_df.columns for c in ['x', 'y', 'z']):
            task = task_df[["x", "y", "z"]].dropna().values
        elif len(task_df.columns) >= 3:
            task = task_df.iloc[:, -3:].dropna().values
        else:
            print(f"[SKIP] {os.path.basename(task_csv)} has missing columns.")
            return None
            
        if task.shape[1] < 3:
            return None
            
    except Exception as e:
        print(f"[SKIP] Could not parse {task_csv}: {e}")
        return None

    dext_yes = None
    dext_all = None
    if os.path.exists(dext_csv) and os.path.getsize(dext_csv) > 30:
        try:
            dext_df = pd.read_csv(dext_csv)
            dext_df.columns = dext_df.columns.astype(str).str.strip().str.lower()
            
            if all(c in dext_df.columns for c in ['x', 'y', 'z']):
                dext_all = dext_df[["x", "y", "z"]].dropna().values
            elif len(dext_df.columns) >= 3:
                dext_all = dext_df.iloc[:, 0:3].dropna().values
            else:
                dext_all = np.array([])

            if "is_dexterous" in dext_df.columns:
                dext_yes = dext_df[dext_df["is_dexterous"] == 1][["x", "y", "z"]].dropna().values
            elif "dexterous" in dext_df.columns:
                dext_yes = dext_df[dext_df["dexterous"] == 1][["x", "y", "z"]].dropna().values
            elif len(dext_df.columns) >= 4:
                mask = dext_df.iloc[:, 3] == 1
                dext_yes = dext_df[mask].iloc[:, 0:3].dropna().values
            else:
                dext_yes = np.array([])
        except Exception:
            pass

    return task, dext_all, dext_yes


def downsample(pts, n):
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
        xaxis=dict(backgroundcolor="rgb(230,230,250)", gridcolor="white", showbackground=True),
        yaxis=dict(backgroundcolor="rgb(230,250,230)", gridcolor="white", showbackground=True),
        zaxis=dict(backgroundcolor="rgb(250,230,230)", gridcolor="white", showbackground=True),
        aspectmode="data",
    )


def scatter3d(pts, name, colour, size, opacity=0.35, colorscale=None, showscale=False):
    marker_dict = dict(size=size, color=colour, opacity=opacity)
    if colorscale:
        marker_dict["colorscale"] = colorscale
        marker_dict["showscale"] = showscale
        if showscale:
            marker_dict["colorbar"] = dict(title="Z (m)", len=0.6, x=1.0)
            
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode="markers",
        marker=marker_dict,
        name=name,
        hovertemplate="X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<extra></extra>",
    )


# ─────────────────────────────────────────────────────────────────────────────
# VIEW GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def get_robot_traces(robot):
    result = load_csvs(robot)
    if result is None:
        return None, None, 0, 0, 0

    task, dext_all, dext_yes = result
    N_MAX = args.max_pts

    task_d  = downsample(task, N_MAX)
    dext_yd = downsample(dext_yes, N_MAX)

    print(f"\n{'='*60}")
    print(f"  {robot}  |  task={len(task):,} pts", end="")
    if dext_yes is not None:
        frac = 100 * len(dext_yes) / max(len(dext_all), 1)
        print(f"  dexterous={len(dext_yes):,} ({frac:.1f}%)", end="")
    print()

    # 1. TCP Trace (Base layer - Solid Dark Red)
    tcp_trace = scatter3d(
        task_d, name=f"{robot} TCP",
        colour=TCP_COLOUR,
        size=MARKER_SIZE_TASK, opacity=0.4  # Slightly higher opacity so the dark red shows well
    )
    
    # 2. Dexterous Trace (Overlay - Solid Orange)
    if dext_yd is not None and len(dext_yd) > 0:
        dext_trace = scatter3d(
            dext_yd, name=f"{robot} Dexterous",
            colour=DEXT_COLOUR, size=MARKER_SIZE_DEXT, opacity=0.9
        )
    else:
        dext_trace = go.Scatter3d(x=[], y=[], z=[], name=f"{robot} No Dexterous")

    n_task = len(task)
    n_dext = len(dext_yes) if dext_yes is not None else 0
    n_all = max(len(dext_all), 1) if dext_all is not None else 1

    return tcp_trace, dext_trace, n_task, n_dext, n_all


# ─────────────────────────────────────────────────────────────────────────────
# COMPILE AND SAVE
# ─────────────────────────────────────────────────────────────────────────────
robots_to_plot = ["HEAL", "FRANKA"] if args.robot == "BOTH" else [args.robot]

fig = go.Figure()
buttons = []
trace_idx = 0
first_title = ""

total_traces = len(robots_to_plot) * 2

for r_idx, robot in enumerate(robots_to_plot):
    tcp_trace, dext_trace, n_task, n_dext, n_all = get_robot_traces(robot)
    if tcp_trace is None:
        continue
        
    fig.add_trace(tcp_trace)
    fig.add_trace(dext_trace)
    
    # Show only the first robot's Overlay by default
    fig.data[trace_idx].visible = (r_idx == 0)
    fig.data[trace_idx + 1].visible = (r_idx == 0)
    
    dxt_pct = 100 * n_dext / n_all
    
    title_tcp = f"<b>{robot} — TCP Workspace Only</b><br><sup>{n_task:,} configurations</sup>"
    title_dext = f"<b>{robot} — Dexterous Workspace Only</b><br><sup>{n_dext:,} dexterous positions</sup>"
    title_overlay = f"<b>{robot} — Workspace Overlay</b><br><sup>{n_dext:,} dexterous ({dxt_pct:.1f}%) overlaid on {n_task:,} TCP</sup>"
    
    if r_idx == 0:
        first_title = title_overlay

    # Visibility arrays for the 3 states
    vis_tcp_only = [False] * total_traces
    vis_tcp_only[trace_idx] = True
    
    vis_dext_only = [False] * total_traces
    vis_dext_only[trace_idx + 1] = True
    
    vis_overlay = [False] * total_traces
    vis_overlay[trace_idx] = True
    vis_overlay[trace_idx + 1] = True
    
    # 1. TCP Only Button
    buttons.append(dict(
        label=f"{robot}: TCP Only", method="update",
        args=[{"visible": vis_tcp_only}, {"title": dict(text=title_tcp, x=0.5, xanchor="center")}]
    ))
    # 2. Dexterous Only Button
    buttons.append(dict(
        label=f"{robot}: Dexterous Only", method="update",
        args=[{"visible": vis_dext_only}, {"title": dict(text=title_dext, x=0.5, xanchor="center")}]
    ))
    # 3. Overlay Button
    buttons.append(dict(
        label=f"{robot}: Overlay", method="update",
        args=[{"visible": vis_overlay}, {"title": dict(text=title_overlay, x=0.5, xanchor="center")}]
    ))
    
    trace_idx += 2

if not fig.data:
    print("\nNo data to plot. Exiting.")
    exit(0)

# Layout configuration to fit the 6 buttons neatly
fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        direction="right",
        active=2, # Default to the 3rd button (Overlay) of the first robot
        buttons=buttons,
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.5,
        xanchor="center",
        y=1.1,
        yanchor="bottom",
        bgcolor="white",
        bordercolor="gray",
        font=dict(size=11) # Slightly smaller font so buttons fit nicely
    )],
    title=dict(text=first_title, x=0.5, xanchor="center"),
    scene=axis_scene("Z (m)"),
    legend=dict(x=0.85, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
    margin=dict(l=0, r=0, t=120, b=0),
    height=800,
)

out_html = os.path.join(SCRIPT_DIR, "combined_workspace_3d.html")
fig.write_html(out_html, include_plotlyjs="cdn")
print(f"\nSaved combined interactive plot: {os.path.basename(out_html)}")
print("Open the HTML file in any browser and use the buttons at the top to toggle individual/overlay views.")