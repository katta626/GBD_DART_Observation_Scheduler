from __future__ import annotations

"""DART runtime configuration loader.

Edit `dart_settings.toml` for shared defaults. Machine-specific values (the real
API IP on the production system, a slower refresh, ...) go in
`dart_settings.local.toml`, which is git-ignored and overrides the shared file,
so deploying a new version from `main` never overwrites production settings.
The values in `_DEFAULT_CONFIG` are fallback defaults used only when a key is
present in neither TOML file.
"""

from copy import deepcopy
from pathlib import Path
import tomllib
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "dart_settings.toml"
LOCAL_CONFIG_PATH = PROJECT_ROOT / "dart_settings.local.toml"

_DEFAULT_CONFIG = {
    "paths": {
        "database": "data_store.db",
        "log_dir": "log_files",
        "fits_plots": "fits_plots",
        "background_image": "dart.png",
        "scheduler_script": "scheduler1.py",
    },
    "network": {
        "scheme": "http",
        "api_host": "172.17.20.XXX",
        "api_port": 6000,
        "request_timeout_seconds": 5,
        "flask_bind_host": "0.0.0.0",
    },
    "observatory": {
        "longitude": 77.437547,
        "latitude": 13.603839,
        "height_m": 713,
    },
    "ui": {
        "timezone": "Asia/Kolkata",
        "archive_recent_limit": 3,
        "refresh_interval_ms": 1000,
        "page_title": "DART",
        "page_icon": "☩",
        "quick_observation_name": "1pps",
        "quick_add_countdown_seconds": 2,
        "enable_diagnostics": True,
    },
    "scheduler": {
        "threshold_seconds": 10,
        "poll_interval_seconds": 1,
        "quick_trigger_countdown": 10,
        "standard_trigger_countdown": 300,
    },
    "observation": {
        "script": "",
    },
    "hdf5_sync": {
        "remote_host": "vela@172.17.20.203",
        "remote_dir": "/media/vela/GBD_HDF5ARCHIVE/23_07_2026_17_43",
    },
    "diagnostics": {
        "center_frequency_mhz": 200.0,
        "bandwidth_mhz": 16.5,
        "channel0_is_high": False,
        "hour_channel_start": 50,
        "hour_channel_stop": 200,
        "phase_calibration_file": "",
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config() -> dict:
    merged = deepcopy(_DEFAULT_CONFIG)
    for path in (CONFIG_PATH, LOCAL_CONFIG_PATH):
        if path.exists():
            with open(path, "rb") as config_file:
                merged = _deep_merge(merged, tomllib.load(config_file))
    return merged


def get_config_mtime() -> tuple[float | None, float | None]:
    """Modification times of both settings files, used to detect live edits."""
    mtimes = []
    for path in (CONFIG_PATH, LOCAL_CONFIG_PATH):
        try:
            mtimes.append(path.stat().st_mtime if path.exists() else None)
        except OSError:
            mtimes.append(None)
    return tuple(mtimes)


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def get_runtime_settings() -> dict:
    raw_config = _load_raw_config()

    resolved_paths = {
        "database": _resolve_path(raw_config["paths"]["database"]),
        "log_dir": _resolve_path(raw_config["paths"]["log_dir"]),
        "fits_plots": _resolve_path(raw_config["paths"]["fits_plots"]),
        "background_image": _resolve_path(raw_config["paths"]["background_image"]),
        "scheduler_script": _resolve_path(raw_config["paths"]["scheduler_script"]),
    }

    network = {
        "scheme": str(raw_config["network"]["scheme"]),
        "api_host": str(raw_config["network"]["api_host"]),
        "api_port": int(raw_config["network"]["api_port"]),
        "request_timeout_seconds": float(raw_config["network"]["request_timeout_seconds"]),
        "flask_bind_host": str(raw_config["network"]["flask_bind_host"]),
    }
    network["api_base_url"] = f'{network["scheme"]}://{network["api_host"]}:{network["api_port"]}'
    network["trigger_url"] = f'{network["api_base_url"]}/trigger'
    network["observation_over_url"] = f'{network["api_base_url"]}/observation-over'
    network["space_url"] = f'{network["api_base_url"]}/get-prms'
    network["log_url_template"] = f'{network["api_base_url"]}/get-log?filename={{filename}}'

    return {
        "paths": resolved_paths,
        "network": network,
        "observatory": {
            "longitude": float(raw_config["observatory"]["longitude"]),
            "latitude": float(raw_config["observatory"]["latitude"]),
            "height_m": float(raw_config["observatory"]["height_m"]),
        },
        "ui": {
            "timezone": str(raw_config["ui"]["timezone"]),
            "archive_recent_limit": int(raw_config["ui"]["archive_recent_limit"]),
            "refresh_interval_ms": int(raw_config["ui"]["refresh_interval_ms"]),
            "page_title": str(raw_config["ui"]["page_title"]),
            "page_icon": str(raw_config["ui"]["page_icon"]),
            "quick_observation_name": str(raw_config["ui"]["quick_observation_name"]),
            "quick_add_countdown_seconds": int(raw_config["ui"]["quick_add_countdown_seconds"]),
            "enable_diagnostics": bool(raw_config["ui"]["enable_diagnostics"]),
        },
        "scheduler": {
            "threshold_seconds": float(raw_config["scheduler"]["threshold_seconds"]),
            "poll_interval_seconds": float(raw_config["scheduler"]["poll_interval_seconds"]),
            "quick_trigger_countdown": int(raw_config["scheduler"]["quick_trigger_countdown"]),
            "standard_trigger_countdown": int(raw_config["scheduler"]["standard_trigger_countdown"]),
        },
        "observation": {
            "script": str(raw_config["observation"]["script"]).strip(),
        },
        "hdf5_sync": {
            "remote_host": str(raw_config["hdf5_sync"]["remote_host"]),
            "remote_dir": str(raw_config["hdf5_sync"]["remote_dir"]),
        },
        "diagnostics": {
            "center_frequency_mhz": float(raw_config["diagnostics"]["center_frequency_mhz"]),
            "bandwidth_mhz": float(raw_config["diagnostics"]["bandwidth_mhz"]),
            "channel0_is_high": bool(raw_config["diagnostics"]["channel0_is_high"]),
            "hour_channel_start": int(raw_config["diagnostics"]["hour_channel_start"]),
            "hour_channel_stop": int(raw_config["diagnostics"]["hour_channel_stop"]),
            "phase_calibration_file": str(raw_config["diagnostics"]["phase_calibration_file"]).strip(),
        },
    }


def get_fragment_refresh_seconds() -> float:
    refresh_interval_ms = get_runtime_settings()["ui"]["refresh_interval_ms"]
    return max(1.0, refresh_interval_ms / 1000)


_RAW_CONFIG = _load_raw_config()


DB_PATH = _resolve_path(_RAW_CONFIG["paths"]["database"])
LOG_DIR = _resolve_path(_RAW_CONFIG["paths"]["log_dir"])
FITS_PLOTS_DIR = _resolve_path(_RAW_CONFIG["paths"]["fits_plots"])
BACKGROUND_IMAGE_PATH = _resolve_path(_RAW_CONFIG["paths"]["background_image"])
SCHEDULER_SCRIPT_PATH = _resolve_path(_RAW_CONFIG["paths"]["scheduler_script"])

API_SCHEME = str(_RAW_CONFIG["network"]["scheme"])
API_HOST = str(_RAW_CONFIG["network"]["api_host"])
API_PORT = int(_RAW_CONFIG["network"]["api_port"])
REQUEST_TIMEOUT_SECONDS = float(_RAW_CONFIG["network"]["request_timeout_seconds"])
FLASK_BIND_HOST = str(_RAW_CONFIG["network"]["flask_bind_host"])
API_BASE_URL = f"{API_SCHEME}://{API_HOST}:{API_PORT}"
TRIGGER_URL = f"{API_BASE_URL}/trigger"
SPACE_URL = f"{API_BASE_URL}/get-prms"
LOG_URL_TEMPLATE = f"{API_BASE_URL}/get-log?filename={{filename}}"

OBSERVATORY_LONGITUDE = float(_RAW_CONFIG["observatory"]["longitude"])
OBSERVATORY_LATITUDE = float(_RAW_CONFIG["observatory"]["latitude"])
OBSERVATORY_HEIGHT_M = float(_RAW_CONFIG["observatory"]["height_m"])

TIMEZONE_NAME = str(_RAW_CONFIG["ui"]["timezone"])
APP_TIMEZONE = ZoneInfo(TIMEZONE_NAME)
ARCHIVE_RECENT_LIMIT = int(_RAW_CONFIG["ui"]["archive_recent_limit"])
APP_REFRESH_INTERVAL_MS = int(_RAW_CONFIG["ui"]["refresh_interval_ms"])
PAGE_TITLE = str(_RAW_CONFIG["ui"]["page_title"])
PAGE_ICON = str(_RAW_CONFIG["ui"]["page_icon"])
QUICK_OBSERVATION_NAME = str(_RAW_CONFIG["ui"]["quick_observation_name"])
QUICK_OBSERVATION_COUNTDOWN_SECONDS = int(_RAW_CONFIG["ui"]["quick_add_countdown_seconds"])

SCHEDULER_THRESHOLD_SECONDS = float(_RAW_CONFIG["scheduler"]["threshold_seconds"])
POLL_INTERVAL_SECONDS = float(_RAW_CONFIG["scheduler"]["poll_interval_seconds"])
QUICK_TRIGGER_COUNTDOWN = int(_RAW_CONFIG["scheduler"]["quick_trigger_countdown"])
STANDARD_TRIGGER_COUNTDOWN = int(_RAW_CONFIG["scheduler"]["standard_trigger_countdown"])

OBSERVATION_SCRIPT_PATH = str(_RAW_CONFIG["observation"]["script"]).strip()
if OBSERVATION_SCRIPT_PATH:
    OBSERVATION_SCRIPT_PATH = _resolve_path(OBSERVATION_SCRIPT_PATH)
