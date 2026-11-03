#!/usr/bin/env bash
set -euo pipefail

source_dir="$1"
local_dir="$2"
remote_machine="${DART_HDF5_REMOTE_HOST:-vela@172.17.20.203}"
remote_source="$remote_machine:$source_dir"
files_to_keep=120
poll_seconds=10

mkdir -p "$local_dir"
remote_file_list="$local_dir/.remote_hdf5_files.txt"
trap 'rm -f "$remote_file_list"' EXIT

while true; do
    # Obtain only the newest files. rsync then transfers them directly into the
    # directory discovered by app1.py from this repository's location.
    ssh "$remote_machine" "find '$source_dir' -maxdepth 1 -type f -name '*.hdf5' -printf '%f\\n' | tail -n 12" \
        > "$remote_file_list"

    if [ -s "$remote_file_list" ]; then
        rsync -av --files-from="$remote_file_list" "$remote_source/" "$local_dir/"
    fi

    mapfile -t files_to_delete < <(
        find "$local_dir" -maxdepth 1 -type f -name '*.hdf5' -printf '%T@ %p\\n' \
            | sort -nr \
            | tail -n +$((files_to_keep + 1)) \
            | cut -d' ' -f2-
    )
    if [ "${#files_to_delete[@]}" -gt 0 ]; then
        rm -f -- "${files_to_delete[@]}"
    fi

    sleep "$poll_seconds"
done
