# IK Solver Benchmarking: HEAL and Franka

This repository contains the MuJoCo pick-and-place implementations used to compare three inverse-kinematics (IK) methods—QP, Mink, and Damped Least Squares (DLS)—on the HEAL arm and Franka Panda robot. It also includes scripts and output files for visualising the robots' reachable workspaces.

## Folder Structure

```text
Lab3/
├── scripts/
│   ├── pick_place_qp_ik_heal.py
│   ├── pick_place_qp_ik.py
│   ├── pick_place_mink_ik_heal.py
│   ├── pick_place_mink_ik.py
│   ├── pick_place_dls_ik_heal.py
│   ├── pick_place_dls_ik.py
│   ├── workspace_generator.py
│   └── plot_workspace_3d.py
│
├── screenshots/
│   └── screenshots of failed pick-and-place cases
│
└── Workspace_plotting_csvs_and_3d_html_file_For_workspace_visualization/
    ├── workspace data (.csv files)
    └── interactive 3D workspace visualisation (.html file)
```

### `scripts/`

This folder contains six pick-and-place scripts, covering the three IK methods on both robots:

- **HEAL arm:** filenames ending in `_heal.py`
- **Franka Panda:** corresponding filenames without the `_heal` suffix
- **QP:** `pick_place_qp_ik_...`
- **Mink:** `pick_place_mink_ik_...`
- **DLS:** `pick_place_dls_ik_...`

The folder also contains the workspace utilities:

- **`workspace_generator.py`** — generates sampled workspace data using the robot model and forward kinematics.
- **`plot_workspace_3d.py`** — plots the generated workspace data in 3D.

### `screenshots/`

This folder contains screenshots captured for failed pick-and-place cases, which can be used to inspect and document failure behaviour during the experiments.

### `Workspace_plotting_csvs_and_3d_html_file_For_workspace_visualization/`

This folder stores the workspace-visualisation outputs, including CSV files containing workspace data and an HTML file for the interactive 3D workspace view.

## IK Benchmark

The pick-and-place scripts are used to compare QP, Mink, and DLS on both HEAL and Franka. The accompanying report discusses the setup, workspace constraints, solver success rates, computation time, joint-limit behaviour, and failure analysis.
