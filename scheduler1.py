import os
import time
import re
from datetime import datetime
from urllib.parse import quote

import requests

from datastore import DataStore
from dart_config import (
    APP_TIMEZONE,
    DB_PATH,
    LOG_DIR,
    QUICK_OBSERVATION_NAME,
    QUICK_TRIGGER_COUNTDOWN,
    SCHEDULER_THRESHOLD_SECONDS,
    STANDARD_TRIGGER_COUNTDOWN,
    get_runtime_settings,
)
from pulsar_info import RA, count
from schedule_time import extract_clock_time, log_date_from_observation_time, normalize_observation_time


db = DataStore(DB_PATH)
TRIGGER_RETRY_DELAY_SECONDS = 30
MAX_TRIGGER_RETRY_WINDOW_SECONDS = 120
_TRIGGER_RETRY_STATE: dict[str, dict[str, float]] = {}


def get_current_started_at() -> str:
    return normalize_observation_time(datetime.now(APP_TIMEZONE))


def read_all_lines(file_path: str) -> list[str]:
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as file_handle:
        return [line.strip() for line in file_handle.readlines()]


def fetch_log_file(log_name: str) -> str:
    runtime_settings = get_runtime_settings()
    log_url_template = runtime_settings["network"]["log_url_template"]
    request_timeout_seconds = runtime_settings["network"]["request_timeout_seconds"]
    os.makedirs(LOG_DIR, exist_ok=True)
    file_path = os.path.join(LOG_DIR, log_name)

    try:
        response = requests.get(log_url_template.format(filename=quote(log_name)), timeout=request_timeout_seconds)
        if response.status_code == 200:
            with open(file_path, "wb") as file_handle:
                file_handle.write(response.content)
    except requests.RequestException:
        pass

    return file_path


def sync_log_tracking(log_name: str, log_current: list[str]) -> None:
    if log_name not in log_current:
        log_current.append(log_name)
        db.update_system_status("Log_Current", log_current)


def handle_completed_observation(pulsar: str, log_name: str, log_current: list[str], ra_start: str) -> None:
    file_path = fetch_log_file(log_name)
    sync_log_tracking(log_name, log_current)

    lines = read_all_lines(file_path)
    if lines and "Observation Over." in lines[-1]:
        db.update_observation(pulsar, status="Not Started", ra_start=ra_start, started_at=None)
        clear_trigger_attempt(pulsar)
        runtime_settings = get_runtime_settings()
        request_timeout_seconds = runtime_settings["network"]["request_timeout_seconds"]
        observation_over_url = runtime_settings["network"]["observation_over_url"]

        requests.post(
            f"{observation_over_url}?pulsar_name={quote(pulsar)}",
            timeout=request_timeout_seconds,
        )

        if log_name in log_current:
            log_current.remove(log_name)
            db.update_system_status("Log_Current", log_current)


def get_retry_window_seconds(duration_minutes: int) -> int:
    # Retry only briefly so a missed trigger does not consume the whole observation window.
    scaled_window = int(duration_minutes * 60 * 0.2)
    return max(TRIGGER_RETRY_DELAY_SECONDS, min(scaled_window, MAX_TRIGGER_RETRY_WINDOW_SECONDS))


def is_retry_window_open(pulsar: str, duration_minutes: int) -> bool:
    state = _TRIGGER_RETRY_STATE.get(pulsar)
    if state is None:
        return True
    return (time.time() - state["first_attempt"]) <= get_retry_window_seconds(duration_minutes)


def can_retry_trigger(pulsar: str) -> bool:
    state = _TRIGGER_RETRY_STATE.get(pulsar)
    if state is None:
        return True
    return (time.time() - state["last_attempt"]) >= TRIGGER_RETRY_DELAY_SECONDS


def mark_trigger_attempt(pulsar: str) -> None:
    now = time.time()
    state = _TRIGGER_RETRY_STATE.get(pulsar)
    first_attempt = now if state is None else state["first_attempt"]
    _TRIGGER_RETRY_STATE[pulsar] = {"first_attempt": first_attempt, "last_attempt": now}


def clear_trigger_attempt(pulsar: str) -> None:
    _TRIGGER_RETRY_STATE.pop(pulsar, None)


def trigger_observation(payload: dict) -> bool:
    runtime_settings = get_runtime_settings()
    trigger_url = runtime_settings["network"]["trigger_url"]
    request_timeout_seconds = runtime_settings["network"]["request_timeout_seconds"]
    try:
        response = requests.post(trigger_url, json=payload, timeout=request_timeout_seconds)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def update_quick_observation(pulsar: str, data: dict, log_current: list[str]) -> None:
    status = data.get("status", "Not Started")
    duration = int(data.get("duration") or 0)
    #ra_start = normalize_observation_time(data.get("ra_start", ""))
    raw_ra_start = data.get("ra_start", "")

    if raw_ra_start:
    # 1. Normalize first
        normalized_ra_start = normalize_observation_time(raw_ra_start)

        try:
        # 2. Parse the normalized value
            parsed_ra_start = datetime.strptime(
                normalized_ra_start,
                "%d_%m_%Y %H:%M:%S"
            )

        # 3. Change ONLY the date
            parsed_ra_start = parsed_ra_start.replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            )

        # 4. Normalize again
            ra_start = normalize_observation_time(parsed_ra_start)

        except (ValueError, TypeError) as e:
            print(f"Could not update RA start date: {e}")
            ra_start = normalized_ra_start
    else:
        ra_start = ""


    started_at = normalize_observation_time(data.get("started_at", "")) if data.get("started_at") else None

    if status != "Not Started":
        if status == "In Progress" and not started_at:
            started_at = get_current_started_at()
            db.update_observation(pulsar, started_at=started_at)
        log_name = f"{pulsar}_{log_date_from_observation_time(ra_start)}_observation.log"
        handle_completed_observation(pulsar, log_name, log_current, ra_start)
        return

    try:
        updated_countdown = count(extract_clock_time(ra_start))
    except ValueError:
        return

    db.update_observation(pulsar, count_down=updated_countdown, ra_start=ra_start, started_at=None)

    if updated_countdown <= SCHEDULER_THRESHOLD_SECONDS:
        if status == "Not Started" and is_retry_window_open(pulsar, duration) and can_retry_trigger(pulsar):
            mark_trigger_attempt(pulsar)
            trigger_sent = trigger_observation(
                {"target": pulsar, "duration": duration * 2, "countdown": str(QUICK_TRIGGER_COUNTDOWN)}
            )
            if trigger_sent:
                db.update_observation(pulsar, status="In Progress", started_at=get_current_started_at())
                clear_trigger_attempt(pulsar)
    else:
        clear_trigger_attempt(pulsar)


def update_standard_observation(pulsar: str, data: dict, log_current: list[str]) -> None:
    status = data.get("status", "Not Started")
    duration = int(data.get("duration") or 0)
    ra_start = normalize_observation_time(data.get("ra_start", ""))
    started_at = normalize_observation_time(data.get("started_at", "")) if data.get("started_at") else None

    if status != "Not Started":
        if status == "In Progress" and not started_at:
            started_at = get_current_started_at()
            db.update_observation(pulsar, started_at=started_at)
        log_name = f"{pulsar}_{log_date_from_observation_time(ra_start)}_observation.log"
        handle_completed_observation(pulsar, log_name, log_current, ra_start)
        return

    try:
        next_ra_start, updated_countdown = RA(pulsar)
        updated_countdown -= (duration * 60) / 2
        normalized_ra_start = normalize_observation_time(next_ra_start)
        db.update_observation(pulsar, count_down=updated_countdown, ra_start=normalized_ra_start, started_at=None)
    except Exception:
        return

    if updated_countdown <= SCHEDULER_THRESHOLD_SECONDS:
        if is_retry_window_open(pulsar, duration) and can_retry_trigger(pulsar):
            mark_trigger_attempt(pulsar)
            trigger_sent = trigger_observation(
                {"target": pulsar, "duration": duration * 2, "countdown": str(STANDARD_TRIGGER_COUNTDOWN)}
            )
            if trigger_sent:
                db.update_observation(
                    pulsar,
                    status="In Progress",
                    ra_start=normalized_ra_start,
                    started_at=get_current_started_at(),
                )
                clear_trigger_attempt(pulsar)
    else:
        clear_trigger_attempt(pulsar)


def main() -> None:
    schedule = {obs["name"]: obs for obs in db.get_all_observations()}
    log_current = db.get_system_status("Log_Current") or []

    if isinstance(log_current, str):
        log_current = [log_current]

    for pulsar, data in schedule.items():
        countdown = data.get("count_down")
        if countdown is None:
            continue

        try:
            if re.match(r"^J\d{4}\+\d{4}", pulsar):
                update_standard_observation(pulsar, data, log_current)
            else:
                update_quick_observation(pulsar, data, log_current)
        except Exception as exc:
            print(f"Scheduler warning for {pulsar}: {exc}")


if __name__ == "__main__":
    while True:
        try:
            main()
        except Exception as exc:
            print(f"Scheduler loop warning: {exc}")
        time.sleep(get_runtime_settings()["scheduler"]["poll_interval_seconds"])
