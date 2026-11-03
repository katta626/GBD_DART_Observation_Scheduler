# DART
A Locally Hosted Webpage for Data Aquisition and Data Archive centre for Diamond Array Radio Telescope at Gauribidnoor Radio Telescope.

## Setup

```bash
conda env create -f environment.yml
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

Deploying (on the production machine, in `~/pulsar_streamlit/DART`):

1. Stop the app: Ctrl-C inside screen `scheduler`.
2. Back up the database: `cp data_store.db ~/pulsar_streamlit/data_store.db.bak_<date>`.
3. Tag for rollback: `git tag prod-before-<date>`.
4. `git fetch origin` and `git merge origin/main`.
5. Restart in screen `scheduler`:
   `/home/pulsar/miniconda3/envs/DART_env/bin/streamlit run app1.py`, then press the scheduler button.
6. `git push origin prod`.

Rollback: stop the app, `git reset --hard <tag>`, restart.

### One-time cleanup of the `prod` branch

`prod` still tracks about 1,100 runtime files that are now in `.gitignore` (the live
`data_store.db`, observation logs, HDF5 data, `versions/`, caches). Untrack them on the
production machine itself, after a deploy that brought in this `.gitignore`. `--cached`
only removes them from git; every file stays on disk:

```bash
git ls-files -ci --exclude-standard -z | xargs -0 git rm --cached -q
git commit -m "Stop tracking runtime files on prod"
git push origin prod
```

Do not make this commit anywhere else and pull it into the production machine: pulling a
commit that deletes tracked files deletes them from disk. After this cleanup, rolling back
to a tag made after it no longer touches the live database; rolling back to an older tag
still restores the old `data_store.db`, so copy the live one out first.
