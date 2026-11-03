from datetime import datetime
import atexit
import base64
import glob
import html
import os
import re
import subprocess
import sys
import datetime as dtt
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
import requests

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from datastore import DataStore
from dart_config import (
    APP_TIMEZONE,
    ARCHIVE_RECENT_LIMIT,
    BACKGROUND_IMAGE_PATH,
    DB_PATH,
    FITS_PLOTS_DIR,
    LOG_DIR,
    OBSERVATORY_HEIGHT_M,
    OBSERVATORY_LATITUDE,
    OBSERVATORY_LONGITUDE,
    PAGE_ICON,
    PAGE_TITLE,
    PROJECT_ROOT,
    QUICK_OBSERVATION_COUNTDOWN_SECONDS,
    QUICK_OBSERVATION_NAME,
    SCHEDULER_SCRIPT_PATH,
    get_config_mtime,
    get_fragment_refresh_seconds,
    get_runtime_settings,
)
from pulsar_info import RA, count
from schedule_time import (
    OBSERVATION_TIME_FORMAT,
    calculate_duration_minutes,
    extract_clock_time,
    format_sidereal_time,
    normalize_observation_time,
)

db = DataStore(DB_PATH)
observing_location = EarthLocation(
    lat=OBSERVATORY_LATITUDE * u.deg,
    lon=OBSERVATORY_LONGITUDE * u.deg,
    height=OBSERVATORY_HEIGHT_M * u.m,
)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, initial_sidebar_state="expanded", layout="wide")


_APP_RUNTIME = {"proc": None}
HDF5_SYNC_SCRIPT = Path(PROJECT_ROOT) / "transient_hdf5" / "real_time_PDR_monitoring.sh"
HDF5_SYNC_LOG_PATH = Path(LOG_DIR) / "hdf5_sync.log"


def start_hdf5_sync_runner(runtime: dict) -> None:
    """Start the HDF5 rsync loop once for the DART server process."""
    sync_process = runtime.get("hdf5_sync_proc")
    if sync_process is not None and sync_process.poll() is None:
        return

    previous_log_handle = runtime.pop("hdf5_sync_log_handle", None)
    if previous_log_handle is not None:
        previous_log_handle.close()

    if not HDF5_SYNC_SCRIPT.is_file():
        runtime["hdf5_sync_error"] = f"Sync script not found: {HDF5_SYNC_SCRIPT}"
        return

    try:
        HDF5_SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(HDF5_SYNC_LOG_PATH, "a", buffering=1)
        hdf5_sync_settings = get_runtime_settings()["hdf5_sync"]
        runtime["hdf5_sync_proc"] = subprocess.Popen(
            ["bash", str(HDF5_SYNC_SCRIPT)],
            cwd=str(PROJECT_ROOT),
            env={
                **os.environ,
                "DART_HDF5_REMOTE_HOST": hdf5_sync_settings["remote_host"],
                "DART_HDF5_REMOTE_DIR": hdf5_sync_settings["remote_dir"],
            },
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        runtime["hdf5_sync_log_handle"] = log_handle
        runtime.pop("hdf5_sync_error", None)
    except OSError as exc:
        runtime["hdf5_sync_error"] = f"Unable to start HDF5 sync: {exc}"


@st.cache_resource(show_spinner=False)
def bootstrap_app_runtime() -> dict:
    for observation in db.get_all_observations():
        db.update_observation(observation["name"], status="Not Started", started_at=None)
    db.update_system_status("Log_Current", "")
    db.update_system_status("status_current", "❌")
    start_hdf5_sync_runner(_APP_RUNTIME)
    return _APP_RUNTIME


def get_runtime_state() -> dict:
    return bootstrap_app_runtime()


def cleanup() -> None:
    runtime = get_runtime_state()
    for process_key in ("proc", "hdf5_sync_proc"):
        proc = runtime.get(process_key)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait()
    log_handle = runtime.get("hdf5_sync_log_handle")
    if log_handle is not None:
        log_handle.close()


atexit.register(cleanup)


def initialize_app_state() -> None:
    defaults = {
        "edit_mode": False,
        "add_mode": False,
        "search_mode": False,
        "quick_duration": 10,
        "config_mtime": None,
        "config_refresh_interval_ms": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    get_runtime_state()


def get_all_time():
    utc_now = datetime.now(dtt.UTC)
    ist_now = utc_now.astimezone(APP_TIMEZONE).replace(microsecond=0)
    observing_time = Time(utc_now, scale="utc", location=observing_location)
    return observing_time.sidereal_time("mean"), ist_now


def get_current_lst_time() -> str:
    lst, _ = get_all_time()
    return format_sidereal_time(lst)


def sync_runtime_config() -> dict:
    runtime_settings = get_runtime_settings()
    refresh_interval_ms = runtime_settings["ui"]["refresh_interval_ms"]

    config_mtime = get_config_mtime()

    previous_refresh_interval_ms = st.session_state.get("config_refresh_interval_ms")
    previous_config_mtime = st.session_state.get("config_mtime")

    if previous_refresh_interval_ms is None and previous_config_mtime is None:
        st.session_state.config_refresh_interval_ms = refresh_interval_ms
        st.session_state.config_mtime = config_mtime
        return runtime_settings

    if previous_refresh_interval_ms != refresh_interval_ms or previous_config_mtime != config_mtime:
        st.session_state.config_refresh_interval_ms = refresh_interval_ms
        st.session_state.config_mtime = config_mtime
        st.rerun()

    return runtime_settings


def set_bg_hack(main_bg: str) -> None:
    if not os.path.exists(main_bg):
        return

    with open(main_bg, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_countdown(countdown_seconds) -> str:
    if countdown_seconds in (None, ""):
        return "Unknown"

    try:
        remaining = max(0, int(round(float(countdown_seconds))))
    except (TypeError, ValueError):
        return str(countdown_seconds)

    hours, remainder = divmod(remaining, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_ist_timestamp(timestamp_seconds: float) -> str:
    return datetime.fromtimestamp(timestamp_seconds, APP_TIMEZONE).strftime("%d-%m-%Y %H:%M:%S IST")


def parse_observation_datetime(value) -> datetime | None:
    normalized_value = normalize_observation_time(value)
    if not normalized_value:
        return None

    try:
        return datetime.strptime(normalized_value, OBSERVATION_TIME_FORMAT)
    except ValueError:
        return None


def format_elapsed_time(started_at_value) -> str:
    started_at = parse_observation_datetime(started_at_value)
    if started_at is None:
        return "00:00:00"

    now_local = datetime.now(APP_TIMEZONE).replace(tzinfo=None, microsecond=0)
    elapsed_seconds = max(0, int((now_local - started_at).total_seconds()))
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_file_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


# Observation files are named like J0534+2200_60973_26_10_2025_profile.fits,
# 1pps_61140_10_04_26_profile.png (two-digit year) or
# 1pps_61139.63_175MHz_..._Tscrunchd.fits (MJD only).
FILE_DATE_PATTERN = re.compile(r"_(\d{2})_(\d{2})_(\d{4}|\d{2})(?=[_.])")
FILE_MJD_PATTERN = re.compile(r"_(\d{5})(?:\.\d+)?_")
MJD_EPOCH = dtt.date(1858, 11, 17)
UNKNOWN_DATE_LABEL = "Unknown"


def parse_observation_date(file_name: str) -> dtt.date | None:
    """Observation date from dd_mm_yyyy / dd_mm_yy in the file name, else from its MJD."""
    match = FILE_DATE_PATTERN.search(file_name)
    if match is not None:
        day, month, year = (int(value) for value in match.groups())
        if year < 100:
            year += 2000
        try:
            return dtt.date(year, month, day)
        except ValueError:
            pass

    # MJD 58849 = 2020-01-01; smaller values (e.g. 40587, the Unix epoch) are broken timestamps.
    match = FILE_MJD_PATTERN.search(file_name)
    if match is not None and 58849 <= int(match.group(1)) <= 80000:
        return MJD_EPOCH + dtt.timedelta(days=int(match.group(1)))
    return None


def format_observation_date(observation_date: dtt.date | None) -> str:
    return observation_date.strftime("%d-%m-%Y") if observation_date else UNKNOWN_DATE_LABEL


def build_widget_key(prefix: str, value: str) -> str:
    safe_value = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return f"{prefix}_{safe_value}"


def inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .dart-info-card {
            background: linear-gradient(145deg, rgba(12, 28, 52, 0.92), rgba(28, 66, 98, 0.85));
            border: 1px solid rgba(145, 196, 255, 0.22);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            margin: 0.4rem 0 0.8rem 0;
            box-shadow: 0 16px 40px rgba(5, 16, 30, 0.22);
            color: #f4f8ff;
        }
        .dart-info-kicker {
            font-size: 0.72rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #8bd6ff;
            margin-bottom: 0.35rem;
        }
        .dart-info-title {
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.2rem;
        }
        .dart-info-meta {
            font-size: 0.92rem;
            color: rgba(244, 248, 255, 0.78);
            line-height: 1.45;
        }
        .dart-highlight {
            background: linear-gradient(135deg, rgba(240, 248, 255, 0.9), rgba(214, 236, 255, 0.88));
            border: 1px solid rgba(63, 120, 180, 0.2);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
            color: #17324d;
        }
        .dart-highlight-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .dart-highlight-meta {
            font-size: 0.95rem;
            color: #35526f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def archive_sort_key(record: dict) -> tuple:
    return (record["observation_date"] or dtt.date.min, record["created_ts"])


@st.cache_data(show_spinner=False, ttl=10)
def build_archive_records(base_dir: str) -> list[dict]:
    """One record per observation: a FITS file and/or its preview PNG with the same name."""
    if not os.path.exists(base_dir):
        return []

    observations: dict[tuple[str, str], dict] = {}
    pulsar_dirs = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    search_dirs = [(None, base_dir)] + [(pulsar, os.path.join(base_dir, pulsar)) for pulsar in pulsar_dirs]
    for pulsar, directory in search_dirs:
        for file_path in glob.glob(os.path.join(directory, "*.fits")) + glob.glob(os.path.join(directory, "*.png")):
            stem, extension = os.path.splitext(os.path.basename(file_path))
            # Loose files in the top folder are named <pulsar>_<mjd>_...
            record_pulsar = pulsar or stem.split("_")[0]
            entry = observations.setdefault((record_pulsar, stem), {"fits_path": None, "preview_path": None})
            entry["fits_path" if extension == ".fits" else "preview_path"] = file_path

    records = []
    for (pulsar, stem), entry in observations.items():
        main_path = entry["fits_path"] or entry["preview_path"]
        try:
            created_ts = os.path.getmtime(main_path)
            fits_size = os.path.getsize(entry["fits_path"]) if entry["fits_path"] else 0
        except OSError:
            continue
        observation_date = parse_observation_date(stem)
        records.append(
            {
                "pulsar": pulsar,
                "fits_path": entry["fits_path"],
                "file_name": os.path.basename(main_path),
                "observation_date": observation_date,
                "observation_date_label": format_observation_date(observation_date),
                "preview_path": entry["preview_path"],
                "created_ts": created_ts,
                "created_at": format_ist_timestamp(created_ts),
                "file_size_bytes": fits_size,
                "file_size": format_file_size(fits_size) if entry["fits_path"] else "—",
            }
        )

    records.sort(key=archive_sort_key, reverse=True)
    return records


# FITS files are large: keep at most two in memory, only the ones picked for download.
@st.cache_data(show_spinner=False, ttl=60, max_entries=2)
def load_binary_file(file_path: str) -> bytes:
    with open(file_path, "rb") as file_handle:
        return file_handle.read()


def get_active_observations(schedule: dict[str, dict]) -> list[dict]:
    active_records = []
    for name, details in schedule.items():
        if details.get("status") != "In Progress":
            continue

        started_at = details.get("started_at") or details.get("ra_start")
        active_records.append(
            {
                "name": name,
                "duration": details.get("duration"),
                "started_at": normalize_observation_time(started_at),
                "ra_start": normalize_observation_time(details.get("ra_start")),
            }
        )

    active_records.sort(
        key=lambda record: parse_observation_datetime(record["started_at"]) or datetime.min
    )
    return active_records


@st.cache_data(show_spinner=False, ttl=60)
def get_live_storage_stats(space_url: str, request_timeout_seconds: float) -> tuple[str, str, str]:
    try:
        response = requests.get(space_url, timeout=request_timeout_seconds)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return "Unavailable", "Unavailable", "Unavailable"

    total = str(data.get("total", "Unavailable"))
    used = str(data.get("used", "Unavailable"))
    free = str(data.get("free", "Unavailable"))
    return total, used, free


@st.fragment(run_every=get_fragment_refresh_seconds())
def render_time_panel() -> None:
    schedule = {obs["name"]: obs for obs in db.get_all_observations(order_by_countdown=True)}
    current_lst, ist_now = get_all_time()
    current_lst_str = format_sidereal_time(current_lst)
    runtime_settings = sync_runtime_config()
    total, used, free = get_live_storage_stats(
        runtime_settings["network"]["space_url"],
        runtime_settings["network"]["request_timeout_seconds"],
    )

    st.markdown(
        f"""
        <div class="dart-info-card">
            <div class="dart-info-kicker">Live Space Available</div>
            <div class="dart-info-title">Avaliable Space {html.escape(free)}</div>
            <div class="dart-info-meta">Used {html.escape(used)}</div>
            <div class="dart-info-meta">Total Space {html.escape(total)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="dart-info-card">
            <div class="dart-info-kicker">Clock</div>
            <div class="dart-info-title">IST :  {html.escape(ist_now.strftime("%H:%M:%S ; %d-%m-%Y IST"))}</div>
            <div class="dart-info-title">LST :  {html.escape(current_lst_str)}</div>
            <div class="dart-info-title">{html.escape(ist_now.strftime("%d-%m-%Y IST"))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    active_observations = get_active_observations(schedule)
    if not active_observations:
        st.markdown(
            """
            <div class="dart-info-card">
                <div class="dart-info-kicker">Observation Status</div>
                <div class="dart-info-title">Idle</div>
                <div class="dart-info-meta">No pulsar is currently marked as in progress.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for active in active_observations:
        elapsed = format_elapsed_time(active["started_at"])
        planned_duration = active.get("duration") or "N/A"
        st.markdown(
            f"""
            <div class="dart-info-card">
                <div class="dart-info-kicker">Currently Observing</div>
                <div class="dart-info-title">{html.escape(active["name"])}</div>
                <div class="dart-info-meta">Elapsed: {html.escape(elapsed)}</div>
                <div class="dart-info-meta">Started: {html.escape(active["started_at"])}</div>
                <div class="dart-info-meta">Planned duration: {html.escape(str(planned_duration))} min</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_schedule_dataframe(schedule: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for pulsar, details in schedule.items():
        started_at = normalize_observation_time(details.get("started_at", "")) if details.get("started_at") else "—"
        rows.append(
            {
                "Pulsar": pulsar,
                "Duration (minutes)": details.get("duration", "N/A"),
                "Status": details.get("status", "Not Started"),
                "RA Start": normalize_observation_time(details.get("ra_start", "")),
                "Count Down": format_countdown(details.get("count_down")),
                "Started": started_at if details.get("status") == "In Progress" else "—",
            }
        )

    return pd.DataFrame(rows)
    


@st.fragment(run_every=get_fragment_refresh_seconds())
def render_schedule_table():
    schedule = {
        obs["name"]: obs
        for obs in db.get_all_observations(order_by_countdown=True)
    }

    if schedule:
        st.dataframe(
            build_schedule_dataframe(schedule),
            width='stretch',
            hide_index=True,
        )
    else:
        st.info("No scheduled pulsars yet.")

def add_schedule_entry(
    name: str,
    duration: int,
    ra_start,
    countdown: float,
    *,
    status: str = "Not Started",
    started_at: str | None = None,
) -> None:
    db.add_or_update_observation(
        name=name,
        duration=int(duration),
        status=status,
        ra_start=normalize_observation_time(ra_start),
        count_down=float(countdown),
        started_at=normalize_observation_time(started_at) if started_at else None,
    )


def upsert_quick_observation(duration: int, ra_start, countdown: float, name=QUICK_OBSERVATION_NAME) -> str:
    existing = db.get_observation(name)
    action = "Updated" if existing else "Added"
    status = "Not Started"
    started_at = None

    if existing and existing.get("status") == "In Progress":
        status = "In Progress"
        started_at = existing.get("started_at")
    #name=QUICK_OBSERVATION_NAME
    add_schedule_entry(
        name=name,
        duration=duration,
        ra_start=ra_start,
        countdown=countdown,
        status=status,
        started_at=started_at,
    )
    return action


def render_quick_add() -> None:
    with st.popover("⏱️ add_now"):
        current_ra = get_current_lst_time()
        st.caption(f"Current RA / LST: {current_ra}")
        st.session_state.quick_duration = st.number_input(
            "Duration (minutes)",
            min_value=1,
            step=1,
            value=st.session_state.quick_duration,
            key="quick_add_duration",
        )

        if st.button("Confirm Add", key="quick_add_confirm"):
            action = upsert_quick_observation(
                duration=st.session_state.quick_duration,
                ra_start=current_ra,
                countdown=QUICK_OBSERVATION_COUNTDOWN_SECONDS,
            )
            st.success(f"{action} {QUICK_OBSERVATION_NAME} with current RA {current_ra}")
            st.rerun()


def render_add_observation(schedule: dict[str, dict]) -> None:
    with st.expander("Add New Pulsar", expanded=True):
        st.checkbox(
            "Activate Search Mode",
            help="Check this box to switch from Normal Mode to Search Mode.",
            key="search_mode",
        )

        if st.session_state.search_mode:
            st.write("Current Mode: Search Mode")
            st.caption("Use HH:MM or HH:MM:SS. The saved value is normalized automatically.")

            start_ra = st.text_input("RA Start", placeholder="HH:MM:SS", key="search_ra_start")
            end_ra = st.text_input("RA End", placeholder="HH:MM:SS", key="search_ra_end")
            name_ra = st.text_input("Name", key="name_ra")

            if st.button("Confirm Add", key="confirm_add_search"):
                name_ra = name_ra.strip() or QUICK_OBSERVATION_NAME
                if not re.fullmatch(r"[A-Za-z0-9+\-_.]{1,64}", name_ra):
                    st.error("Name may only contain letters, digits and + - _ .")
                    return
                try:
                    duration_minutes = calculate_duration_minutes(start_ra, end_ra)
                    countdown = count(extract_clock_time(start_ra))
                except ValueError as exc:
                    st.error(str(exc))
                    return

                action = upsert_quick_observation(
                    duration=duration_minutes,
                    ra_start=start_ra,
                    countdown=countdown,
                    name=name_ra
                )
                st.success(f"{action} pulsar '{name_ra}'.")
                st.session_state.add_mode = False
                st.rerun()

        else:
            st.write("Current Mode: Normal Mode")
            new_pulsar_name = st.text_input("New Pulsar Name", key="new_pulsar_name")
            new_duration = st.number_input(
                "New Duration (minutes)",
                min_value=1,
                step=1,
                key="new_pulsar_duration",
            )

            if st.button("Confirm Add", key="confirm_add_normal"):
                name = new_pulsar_name.strip()
                if not name:
                    st.error("Please enter a pulsar name.")
                    return
                if name in schedule:
                    st.error("Pulsar already exists. Use Edit instead.")
                    return

                try:
                    ra_start, countdown = RA(name)
                except Exception as exc:
                    st.error(f"Could not look up '{name}': {exc}")
                    return

                add_schedule_entry(
                    name=name,
                    duration=new_duration,
                    ra_start=ra_start,
                    countdown=max(0, countdown / 2),
                )
                st.success(f"Added pulsar '{name}'.")
                st.session_state.add_mode = False
                st.rerun()


def render_edit_observation(schedule: dict[str, dict]) -> None:
    with st.expander("✏️ Edit Pulsar", expanded=True):
        selected_pulsar = st.selectbox(
            "Select Pulsar to Edit",
            options=list(schedule.keys()),
            index=0,
            key="edit_selected_pulsar",
        )

        if not selected_pulsar:
            return

        details = schedule[selected_pulsar]
        duration = int(details.get("duration", 1) or 1)
        status = details.get("status", "Not Started")
        ra_start = normalize_observation_time(details.get("ra_start", ""))
        count_down = details.get("count_down", duration * 30)
        started_at = details.get("started_at")

        st.caption(f"RA Start: {ra_start}")

        new_duration = st.number_input(
            "Duration (minutes)",
            min_value=1,
            value=duration,
            step=1,
            key=f"edit_duration_{selected_pulsar}",
        )

        if st.button("Update Schedule", key=f"update_schedule_{selected_pulsar}"):
            db.add_or_update_observation(
                name=selected_pulsar,
                duration=int(new_duration),
                status=status,
                ra_start=ra_start,
                count_down=count_down,
                started_at=started_at,
            )
            st.success(f"Updated schedule for '{selected_pulsar}'.")
            st.session_state.edit_mode = False
            st.rerun()

        if st.button("Delete Schedule", type="secondary", key=f"delete_schedule_{selected_pulsar}"):
            db.delete_observation(selected_pulsar)
            st.warning(f"Deleted schedule for '{selected_pulsar}'.")
            st.session_state.edit_mode = False
            st.rerun()


def read_all_lines(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r") as file_handle:
        return [line.strip() for line in file_handle.readlines()]


def filter_lines(lines: list[str], keywords: list[str]) -> list[str]:
    if not keywords:
        return lines
    return [line for line in lines if any(keyword in line for keyword in keywords)]


def main_1() -> None:
    inject_custom_styles()
    set_bg_hack(BACKGROUND_IMAGE_PATH)
    runtime = get_runtime_state()

    schedule = {obs["name"]: obs for obs in db.get_all_observations(order_by_countdown=True)}
    with st.sidebar:
        st.header("SCHEDULER")
        render_time_panel()

        status_current = db.get_system_status("status_current") or "❌"
        sync_process = runtime.get("hdf5_sync_proc")
        sync_running = sync_process is not None and sync_process.poll() is None
        if sync_running:
            st.caption("HDF5 rsync: 🟢 running")
        else:
            st.warning("HDF5 rsync is not running.")
            if runtime.get("hdf5_sync_error"):
                st.caption(runtime["hdf5_sync_error"])
            if st.button("Restart HDF5 rsync", key="restart_hdf5_rsync"):
                start_hdf5_sync_runner(runtime)
                st.rerun()

        edit_col, add_col, quick_add_col, run_col = st.columns([2, 2, 4, 3])

        with edit_col:
            if st.button("✏️", key="open_edit_mode"):
                st.session_state.edit_mode = True

        with add_col:
            if st.button("➕ Add", key="open_add_mode"):
                st.session_state.add_mode = True

        with quick_add_col:
            render_quick_add()

        render_schedule_table()

        with run_col:
            proc = runtime.get("proc")
            obs_running = proc is not None and proc.poll() is None
            if st.button(status_current, type="primary", key="toggle_scheduler"):
                if obs_running:
                    if proc:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        runtime["proc"] = None
                        db.update_system_status("status_current", "DEAD ☠️")
                        st.warning("Stopped observation.")
                else:
                    runtime["proc"] = subprocess.Popen(
                        [sys.executable, SCHEDULER_SCRIPT_PATH],
                        cwd=str(PROJECT_ROOT),
                    )
                    db.update_system_status("status_current", "LIVE 🟢")
                    st.success("Started observation.")

                st.rerun()

        if st.session_state.add_mode:
            render_add_observation(schedule)

        if schedule and st.session_state.edit_mode:
            render_edit_observation(schedule)


def render_log_updates_tab() -> None:
    log_files = db.get_system_status("Log_Current") or []
    if isinstance(log_files, str):
        log_files = [log_files]

    filter_keywords = [
        "Waiting for",
        "Observation",
        "Pulsar:",
        "ACQ over",
        "Removing Trigger file from remote machine",
        "Observation stoped",
        "SLIP check",
        "Pulsar data reduction pipeline started ...",
        "---",
        "All done!",
    ]

    if not log_files:
        st.warning("No log files found in the current run.")
        return

    for log_file in log_files:
        file_path = os.path.join(LOG_DIR, log_file)
        with st.expander(f"Show Log Updates: {log_file}", expanded=False):
            filtered_lines = filter_lines(read_all_lines(file_path), filter_keywords)
            if filtered_lines:
                st.text("\n".join(filtered_lines))
            else:
                st.text("No matching logs yet.")


def format_archive_choice(record: dict) -> str:
    return f"{record['pulsar']} · {record['observation_date_label']} · {record['file_name']} ({record['file_size']})"


def render_download_picker(filtered_records: list[dict]) -> None:
    """One download button for the picked observation; only that FITS file is read."""
    downloadable = [record for record in filtered_records if record["fits_path"]]
    with st.container(border=True):
        st.markdown("**Download a FITS file**")
        if not downloadable:
            st.caption("None of the matching observations has a FITS file.")
            return
        records_by_path = {record["fits_path"]: record for record in downloadable}
        pick_col, button_col = st.columns([3, 1], vertical_alignment="bottom")
        with pick_col:
            chosen_path = st.selectbox(
                f"Observation ({len(downloadable)} with FITS in the current results)",
                options=list(records_by_path),
                format_func=lambda path: format_archive_choice(records_by_path[path]),
                key="archive_download_choice",
            )
        with button_col:
            st.download_button(
                "Download FITS",
                data=load_binary_file(chosen_path),
                file_name=os.path.basename(chosen_path),
                mime="application/fits",
                key="archive_download_button",
                type="primary",
                width='stretch',
            )


def render_archive_card_meta(record: dict) -> None:
    st.write(f"File: `{record['file_name']}`")
    st.write(f"Added (IST): {record['created_at']}")
    if record["fits_path"]:
        st.write(f"FITS Size: {record['file_size']}")
    else:
        st.caption("FITS file not available for this observation.")


def build_date_overview(archive_records: list[dict]) -> pd.DataFrame:
    """Per pulsar: how many observations and on which dates (newest first)."""
    dates_by_pulsar: dict[str, dict[str, int]] = {}
    for record in sorted(archive_records, key=archive_sort_key, reverse=True):
        pulsar_dates = dates_by_pulsar.setdefault(record["pulsar"], {})
        label = record["observation_date_label"]
        pulsar_dates[label] = pulsar_dates.get(label, 0) + 1

    rows = []
    for pulsar in sorted(dates_by_pulsar):
        pulsar_dates = dates_by_pulsar[pulsar]
        rows.append(
            {
                "Pulsar": pulsar,
                "Observations": sum(pulsar_dates.values()),
                "Latest": next(iter(pulsar_dates)),
                "Observation Dates": ", ".join(
                    label if count == 1 else f"{label} (x{count})" for label, count in pulsar_dates.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def render_archive_tab() -> None:
    st.markdown(
        """
        <div class="dart-highlight">
            <div class="dart-highlight-title">Observation Archive</div>
            <div class="dart-highlight-meta">
                Search, filter by pulsar and observation date, preview, and download FITS observations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    archive_records = build_archive_records(FITS_PLOTS_DIR)
    if not archive_records:
        st.info("No FITS archives are available yet.")
        return

    search_query = st.text_input(
        "Search observations",
        placeholder="Search by pulsar, file name, or observation date",
        key="archive_search_query",
    ).strip()

    available_pulsars = sorted({record["pulsar"] for record in archive_records})
    pulsar_col, date_col = st.columns([1, 1])

    with pulsar_col:
        selected_pulsars = st.multiselect(
            "Filter pulsars",
            options=available_pulsars,
            default=[],
            key="archive_pulsar_filter",
        )

    # Only offer the dates that exist for the chosen pulsars, newest first, with counts.
    date_counts: dict[str, int] = {}
    for record in archive_records:
        if selected_pulsars and record["pulsar"] not in selected_pulsars:
            continue
        label = record["observation_date_label"]
        date_counts[label] = date_counts.get(label, 0) + 1
    with date_col:
        selected_dates = st.multiselect(
            f"Observation dates ({len(date_counts)} available)",
            options=list(date_counts),
            default=[],
            format_func=lambda label: f"{label}  ·  {date_counts[label]} obs",
            placeholder="All dates",
            key="archive_date_filter",
        )

    preview_filter_col, fits_filter_col, sort_col = st.columns([1, 1, 1.4])
    with preview_filter_col:
        preview_only = st.checkbox(
            "Preview only",
            value=False,
            help="Only show observations with a preview PNG.",
            key="archive_preview_only",
        )
    with fits_filter_col:
        fits_only = st.checkbox(
            "FITS only",
            value=False,
            help="Only show observations with a downloadable FITS file.",
            key="archive_fits_only",
        )
    with sort_col:
        sort_mode = st.selectbox(
            "Sort by",
            options=["Newest", "Oldest", "Pulsar A-Z", "Largest FITS"],
            key="archive_sort_mode",
        )

    with st.expander(f"Available observation dates · {len(available_pulsars)} pulsars", expanded=False):
        st.dataframe(build_date_overview(archive_records), width='stretch', hide_index=True)

    normalized_query = search_query.lower()
    filtered_records = []
    for record in archive_records:
        searchable_text = " ".join(
            [
                record["pulsar"],
                record["file_name"],
                record["observation_date_label"],
                record["created_at"],
            ]
        ).lower()

        if normalized_query and normalized_query not in searchable_text:
            continue
        if selected_pulsars and record["pulsar"] not in selected_pulsars:
            continue
        if selected_dates and record["observation_date_label"] not in selected_dates:
            continue
        if preview_only and record["preview_path"] is None:
            continue
        if fits_only and record["fits_path"] is None:
            continue

        filtered_records.append(record)

    if sort_mode == "Oldest":
        filtered_records.sort(key=archive_sort_key)
    elif sort_mode == "Pulsar A-Z":
        filtered_records.sort(key=archive_sort_key, reverse=True)
        filtered_records.sort(key=lambda record: record["pulsar"])
    elif sort_mode == "Largest FITS":
        filtered_records.sort(key=lambda record: record["file_size_bytes"], reverse=True)
    else:
        filtered_records.sort(key=archive_sort_key, reverse=True)

    page_size = max(1, ARCHIVE_RECENT_LIMIT)
    page_count = max(1, -(-len(filtered_records) // page_size))

    total_col, matched_col, pages_col = st.columns(3)
    total_col.metric("Total observations", len(archive_records))
    matched_col.metric("Matches", len(filtered_records))
    pages_col.metric("Pages", page_count)

    if not filtered_records:
        st.warning("No archive entries matched the current search and filters.")
        return

    render_download_picker(filtered_records)

    archive_summary = pd.DataFrame(
        [
            {
                "Pulsar": record["pulsar"],
                "Observation Date": record["observation_date_label"],
                "File": record["file_name"],
                "Added (IST)": record["created_at"],
                "FITS Size": record["file_size"],
                "Preview": "Yes" if record["preview_path"] else "No",
            }
            for record in filtered_records
        ]
    )
    st.caption(f"All {len(filtered_records)} matching observations:")
    st.dataframe(archive_summary, width='stretch', hide_index=True, height=min(420, 38 + 35 * len(filtered_records)))

    page = 1
    if page_count > 1:
        page = st.selectbox(
            "Preview page",
            options=list(range(1, page_count + 1)),
            format_func=lambda number: (
                f"Page {number} of {page_count} · observations "
                f"{(number - 1) * page_size + 1}–{min(number * page_size, len(filtered_records))}"
            ),
            key="archive_preview_page",
        )
    page_records = filtered_records[(page - 1) * page_size : page * page_size]

    featured_record = page_records[0]
    with st.container(border=True):
        feature_meta_col, feature_preview_col = st.columns([1.8, 1.2])
        with feature_meta_col:
            st.subheader(f"Featured: {featured_record['pulsar']}")
            st.write(f"Observation Date: {featured_record['observation_date_label']}")
            render_archive_card_meta(featured_record)
        with feature_preview_col:
            if featured_record["preview_path"]:
                st.image(
                    featured_record["preview_path"],
                    caption=f"{featured_record['pulsar']} preview",
                    width='stretch',
                )
            else:
                st.info("Preview image not available.")

    remaining_records = page_records[1:]
    if not remaining_records:
        return

    card_columns = st.columns(2)
    for index, record in enumerate(remaining_records):
        with card_columns[index % 2]:
            with st.container(border=True):
                st.subheader(record["pulsar"])
                st.caption(f"Observed on {record['observation_date_label']}")
                render_archive_card_meta(record)
                if record["preview_path"]:
                    st.image(
                        record["preview_path"],
                        caption=f"{record['pulsar']} preview",
                        width='stretch',
                    )
                else:
                    st.info("Preview image not available.")


def render_diagnostic_tab() -> None:
    render_live_diagnostic_plots()


TRANSIENT_HDF5_DIR = Path(PROJECT_ROOT) / "transient_hdf5"
LIVE_HDF5_DIR = TRANSIENT_HDF5_DIR / "temp_data" / "transient_hdf5_data"
PHASE_CALIBRATION_FILE = "rf_path_phase.dat"


def hdf5_power_db(values: np.ndarray) -> np.ndarray:
    """The exact power conversion used by the original HDF5 plotting scripts."""
    return 10 * np.log10(np.maximum(np.abs(values) ** 2, np.finfo(float).tiny))


def get_live_hdf5_files() -> list[Path]:
    """Find the files synced beside this project, without machine-specific paths."""
    files_with_mtime = []
    for file_path in LIVE_HDF5_DIR.glob("*.hdf5"):
        try:
            files_with_mtime.append((file_path.stat().st_mtime, file_path))
        except OSError:
            # The sync process may prune an old file while this fragment refreshes.
            continue
    return [file_path for _, file_path in sorted(files_with_mtime)]


def load_hdf5_arrays(file_path: Path, dataset_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read precisely the arrays consumed by the original plotting code."""
    with h5py.File(file_path, "r", locking=False) as hdf5_file:
        missing = [name for name in dataset_names if name not in hdf5_file]
        if missing:
            raise KeyError(", ".join(missing))
        return {name: np.asarray(hdf5_file[name][:]) for name in dataset_names}


def get_phase_calibration_path() -> Path | None:
    """Locate a calibration file relative to the live data, when supplied."""
    candidates = (
        LIVE_HDF5_DIR / PHASE_CALIBRATION_FILE,
        LIVE_HDF5_DIR.parent / PHASE_CALIBRATION_FILE,
        TRANSIENT_HDF5_DIR / PHASE_CALIBRATION_FILE,
    )
    return next((path for path in candidates if path.is_file()), None)


def collect_last_hour_hdf5_data(
    hdf5_files: list[Path], end_timestamp: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace the old combined-HDF5 intermediate file with in-memory data."""
    cutoff = end_timestamp - 3600
    timestamp_parts: list[np.ndarray] = []
    correlation_parts: list[np.ndarray] = []
    pa_power_parts: list[np.ndarray] = []

    for file_path in reversed(hdf5_files):
        try:
            arrays = load_hdf5_arrays(file_path, ("timestamp", "corr_spec_avg", "p_sum_avg"))
        except (OSError, ValueError, KeyError):
            # Skip a file while the acquisition program is still writing it.
            continue

        file_timestamps = arrays["timestamp"]
        selected = file_timestamps >= cutoff
        if not np.any(selected):
            if file_timestamps.size and file_timestamps.max() < cutoff:
                break
            continue

        timestamp_parts.append(file_timestamps[selected])
        correlation_parts.append(arrays["corr_spec_avg"][selected])
        pa_power_parts.append(arrays["p_sum_avg"][selected])

    if not timestamp_parts:
        return np.array([]), np.array([]), np.array([])

    timestamps = np.concatenate(timestamp_parts)
    correlation = np.concatenate(correlation_parts)
    pa_power = np.concatenate(pa_power_parts)
    order = np.argsort(timestamps)
    return timestamps[order], correlation[order], pa_power[order]


def render_spectrogram(times: pd.DatetimeIndex, power: np.ndarray, frequency_axis: np.ndarray) -> None:
    """Render the original imshow(power.T) as a native Streamlit Vega chart."""
    spectrogram_data = pd.DataFrame(
        {
            "time": np.repeat(times.tz_localize(None), power.shape[1]),
            "frequency_mhz": np.tile(frequency_axis, power.shape[0]),
            "power_db": power.reshape(-1),
        }
    )
    st.vega_lite_chart(
        spectrogram_data,
        {
            "mark": "rect",
            "encoding": {
                "x": {"field": "time", "type": "temporal", "title": "Time"},
                "y": {"field": "frequency_mhz", "type": "quantitative", "title": "Frequency (MHz)"},
                "color": {"field": "power_db", "type": "quantitative", "title": "Power (dB)"},
                "tooltip": ["time", "frequency_mhz", "power_db"],
            },
        },
        height=360,
        width="stretch",
    )


def render_pearson_phase_plot(
    arrays: dict[str, np.ndarray], times: pd.DatetimeIndex
) -> None:
    """Recreate the five panels of Pearson_Corr_coeff.png from HDF5 arrays."""
    corr = arrays["corr_spec_avg"]
    acorr1 = arrays["acorr_spec_avg1"]
    acorr2 = arrays["acorr_spec_avg2"]
    # The legacy loop saves after every index, so its final PNG contains index 49.
    sample_index = min(49, corr.shape[0] - 1)
    channels = np.arange(corr.shape[1])
    denominator = np.sqrt(acorr1[sample_index] * acorr2[sample_index])
    pearson = np.divide(
        corr[sample_index],
        denominator,
        out=np.full(corr.shape[1], np.nan + 0j),
        where=denominator != 0,
    )

    st.caption(f"Spectrum {sample_index + 1} of {corr.shape[0]} · {times[sample_index]}")
    left, right = st.columns(2)
    with left:
        st.caption("Auto-correlation 1 power")
        st.line_chart(pd.DataFrame({"Power (dB)": hdf5_power_db(acorr1[sample_index])}, index=channels), height=220)
        st.caption("Cross-correlation power")
        st.line_chart(pd.DataFrame({"Power (dB)": hdf5_power_db(corr[sample_index])}, index=channels), height=220)
        st.caption("Phase")
        st.line_chart(pd.DataFrame({"Phase (degrees)": np.rad2deg(np.angle(corr[sample_index]))}, index=channels), height=220)
    with right:
        st.caption("Auto-correlation 2 power")
        st.line_chart(pd.DataFrame({"Power (dB)": hdf5_power_db(acorr2[sample_index])}, index=channels), height=220)
        st.caption("Pearson correlation")
        # Matplotlib's former plt.plot(complex_array) displayed the real component.
        st.line_chart(pd.DataFrame({"Pearson correlation": pearson.real}, index=channels), height=220)


def render_cross_correlation_plots(arrays: dict[str, np.ndarray], times: pd.DatetimeIndex) -> None:
    """Recreate cc_tseries_phase.png using the original variables and transforms."""
    corr = arrays["corr_spec_avg"]
    corr_channel_average = np.mean(corr, axis=1)
    phase = np.rad2deg(np.angle(corr_channel_average))
    frequency_axis = np.linspace(200 - 16.5 / 2, 200 + 16.5 / 2, corr.shape[1])

    st.caption("Real and imaginary channel-averaged cross-correlation")
    st.line_chart(
        pd.DataFrame(
            {"Re": corr_channel_average.real, "Im": corr_channel_average.imag}, index=times
        ),
        height=250,
    )
    st.caption("Cross-correlation spectrogram")
    render_spectrogram(times, hdf5_power_db(corr), frequency_axis)
    st.caption("Channel-averaged phase")
    st.line_chart(
        pd.DataFrame({"Phase (degrees)": phase, "Zero reference": np.zeros_like(phase)}, index=times),
        height=250,
    )


def render_one_hour_plot(hdf5_files: list[Path], end_timestamp: float) -> None:
    """Recreate one_hr_plot.png without creating a combined HDF5 file or PNG."""
    timestamps, corr, p_sum = collect_last_hour_hdf5_data(hdf5_files, end_timestamp)
    if not timestamps.size:
        st.info("No readable samples from the latest hour are available yet.")
        return

    calibration_path = get_phase_calibration_path()
    if calibration_path is not None:
        try:
            phase_input = np.loadtxt(calibration_path, dtype=float, delimiter=",")
            if phase_input.size != corr.shape[1]:
                raise ValueError(f"expected {corr.shape[1]} channels, found {phase_input.size}")
            corr = corr * (1j ** (phase_input / 90))
            st.caption(f"Phase calibration: `{calibration_path.name}`")
        except (OSError, ValueError) as exc:
            st.warning(f"Phase calibration was not applied: {exc}")
    else:
        st.caption("Phase calibration file not found; showing uncalibrated correlation data.")

    start_channel = min(50, corr.shape[1])
    stop_channel = min(200, corr.shape[1])
    if start_channel == stop_channel:
        st.info("The data does not contain the original channel range 50:200.")
        return

    corr_selected = corr[:, start_channel:stop_channel]
    pa_selected = p_sum[:, start_channel:stop_channel]
    corr_channel_average = np.mean(corr_selected, axis=1)
    pa_channel_average = np.mean(pa_selected, axis=1)
    phase = np.rad2deg(np.angle(corr_channel_average))
    power = hdf5_power_db(corr_selected)
    times = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(APP_TIMEZONE)
    frequency_axis = np.linspace(
        (200 - 16.5 / 2) + start_channel * (16.5 / corr.shape[1]),
        (200 - 16.5 / 2) + stop_channel * (16.5 / corr.shape[1]),
        stop_channel - start_channel,
    )

    first_column, second_column = st.columns(2)
    with first_column:
        st.caption("PA power")
        st.line_chart(pd.DataFrame({"PA power": pa_channel_average}, index=times), height=220)
        st.caption("Real and imaginary cross-correlation")
        st.line_chart(
            pd.DataFrame({"Re1": corr_channel_average.real, "Im1": corr_channel_average.imag}, index=times),
            height=220,
        )
    with second_column:
        st.caption("Cross-correlation power")
        st.line_chart(
            pd.DataFrame({"Correlation power": np.abs(corr_channel_average) ** 2}, index=times),
            height=220,
        )
        st.caption("Phase")
        st.line_chart(
            pd.DataFrame({"Phase (degrees)": phase, "Zero reference": np.zeros_like(phase)}, index=times),
            height=220,
        )
    st.caption("Frequency-sliced spectrogram (channels 50:200)")
    render_spectrogram(times, power, frequency_axis)


@st.fragment(run_every=get_fragment_refresh_seconds())
def render_live_diagnostic_plots() -> None:
    """Direct HDF5 dashboard; no PNG plots or second Streamlit application."""
    if h5py is None:
        st.error("HDF5 support is not installed. Install the project's `h5py` dependency and restart DART.")
        return

    hdf5_files = get_live_hdf5_files()
    if not hdf5_files:
        st.info(f"Waiting for live HDF5 files in `{LIVE_HDF5_DIR}`")
        return

    latest_file = hdf5_files[-1]
    try:
        arrays = load_hdf5_arrays(
            latest_file,
            ("corr_spec_avg", "acorr_spec_avg1", "acorr_spec_avg2", "timestamp"),
        )
    except (OSError, ValueError, KeyError) as exc:
        st.warning(f"Unable to read the latest live HDF5 file: {exc}")
        return

    timestamps = arrays["timestamp"]
    corr = arrays["corr_spec_avg"]
    if not timestamps.size or corr.ndim != 2:
        st.info("The latest HDF5 file does not yet contain spectra.")
        return

    times = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(APP_TIMEZONE)
    st.subheader("Live HDF5 diagnostic plots")
    st.caption(f"Local source: `{latest_file}` · {len(hdf5_files)} synced files available")

    pearson_tab, cross_tab, spectrogram_tab, hour_tab = st.tabs(
        ["Pearson / phase", "Cross-correlation", "Spectrogram", "Last hour"]
    )
    with pearson_tab:
        render_pearson_phase_plot(arrays, times)
    with cross_tab:
        render_cross_correlation_plots(arrays, times)
    with spectrogram_tab:
        frequency_axis = np.linspace(200 - 16.5 / 2, 200 + 16.5 / 2, corr.shape[1])
        render_spectrogram(times, hdf5_power_db(corr), frequency_axis)
    with hour_tab:
        render_one_hour_plot(hdf5_files, float(timestamps[-1]))


def main() -> None:
    
    tab1, tab2, tab3 = st.tabs(["LOG UPDATES 🫶🏻", "Pulsar Data Archive 🌀", "DIAGNOSTIC PLOTS 🧚‍♀️"])

    with tab1:
        render_log_updates_tab()

    with tab2:
        st.markdown("""
                    <style>

                    /* Apply to all tab blocks */
                    .stTabs div[data-testid="stVerticalBlock"] {
                        background-color: #f8e8ea;
                        background-color: #f2e3e5;
                        background-color: #eec9cf;
                        color: black;
                    }

                    /* Only Tab 2 */
                    .stTabs div[role="tabpanel"]:nth-of-type(2) div[data-testid="stVerticalBlock"] {
                        background-color: #f5f5f5 !important;
                        color: black !important;
                    }

                    </style>
                    """, unsafe_allow_html=True)
            
        render_archive_tab()

    with tab3:
        if get_runtime_settings()["ui"]["enable_diagnostics"]:
            render_diagnostic_tab()
        else:
            st.write("Under Construction")


if __name__ == "__main__":
    initialize_app_state()
    main_1()
    main()
