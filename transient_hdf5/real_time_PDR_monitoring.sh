
#!/usr/bin/env bash
set -euo pipefail

# Keep the data beside this script, regardless of the machine or login name.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
local_dir="$script_dir/temp_data/transient_hdf5_data"

# Remote archive directory for the current observation. app1.py passes it from
# [hdf5_sync].remote_dir in the settings; the default keeps the script runnable alone.
remote_dir="${DART_HDF5_REMOTE_DIR:-/media/vela/GBD_HDF5ARCHIVE/23_07_2026_17_43}"

# DART app1.py now reads the HDF5 files directly. No PNG-plotting script or
# separate Streamlit server is started here.
exec "$script_dir/rsync_new_hdf5_files.sh" "$remote_dir" "$local_dir"

