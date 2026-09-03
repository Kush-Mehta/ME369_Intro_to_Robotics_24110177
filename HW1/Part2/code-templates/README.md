# code-templates/

Starter repositories for the implementation-heavy parts of assignments and
projects, one folder per assignment/project that needs one (e.g.
`hw01-mujoco-rotations/`).

## Convention

Each starter folder should:

- Be **runnable as-is** on a fresh checkout (after installing
  `requirements.txt`) -- a student should never be debugging your
  environment setup as a side quest to the actual assignment.
- Clearly separate **complete, working code** (e.g. `utils.py` in
  `hw01-mujoco-rotations/scripts/`) from **starter code with gaps**
  (marked `TODO(student): ...` with a comment explaining what's expected
  and why) -- so students spend their time on the concept the assignment
  is actually testing, not on boilerplate.
- Include its own `README.md` with setup and run instructions, and a
  short table mapping files to their status (complete vs. starter) and to
  the corresponding task in the assignment handout.
- Note in that same `README.md` what parts of the environment were not
  verifiable when the template was written (e.g. "written without a live
  ROS 2 install available -- check package/API names against your distro")
  so a student debugging an issue knows whether to suspect the template or
  their own setup first.
