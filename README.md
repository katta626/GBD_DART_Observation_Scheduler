# DART
A Locally Hosted Webpage for Data Aquisition and Data Archive centre for Diamond Array Radio Telescope at Gauribidnoor Radio Telescope.

## Setup

```bash
conda env create -f envirorment_d.yml
conda activate DART_env
cp dart_settings.local.toml.example dart_settings.local.toml   # then edit for this machine
streamlit run app1.py
```

On the DSP machine, run the trigger API (paths can be overridden with
`DART_LOG_DIR`, `DART_OBSERVATION_SCRIPT`, `DART_API_PORT`):

```bash
python dsp_flask/trigger_api.py
```

## Settings

- `dart_settings.toml`: shared defaults, tracked in git.
- `dart_settings.local.toml`: per-machine overrides (the real API IP, refresh rate,
  `enable_diagnostics`, the HDF5 archive directory, ...). It is git-ignored, so a deploy
  never overwrites it. Both files are re-read live.

## Branches and deployment

- `main`: all development. New features are added here (feature branch → PR → `main`).
- `prod`: what is running on the production system. It only receives tested code from `main`.

Machine-specific values belong in `dart_settings.local.toml`, not in code. A deploy then
needs no hand edits, and `prod` stays identical to `main`.

`main` and `prod` currently have unrelated git histories. Deploying for the first time:

1. On the production machine, back up `data_store.db` and `dart_settings.toml`.
2. Create `dart_settings.local.toml` with the production values (see the `.example`).
3. Bring the code from `main` into `prod`:
   `git merge main --allow-unrelated-histories`, keeping `main`'s version of the code files.
4. Restore `data_store.db` if the merge removed it (it is no longer tracked).
5. Restart the Streamlit app and the scheduler.

Later deploys are an ordinary `git merge main` into `prod`.
