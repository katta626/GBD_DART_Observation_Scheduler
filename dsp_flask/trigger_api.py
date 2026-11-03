from flask import Flask, request, jsonify, send_from_directory
import os
import re
import shlex
import shutil
import subprocess

# Override per machine with environment variables instead of editing this file.
log_files = os.environ.get("DART_LOG_DIR", "/home/dsp/PDR_acquire_setup/master_obs_log")
observation_script = os.environ.get(
    "DART_OBSERVATION_SCRIPT",
    "/home/dsp/PDR_acquire_setup/GBD_DART_lite_obs_script_V5_IQUV_FITS_09_09_2025.sh",
)

# Pulsar / search-mode names: letters, digits and + - _ . only (e.g. J0534+2200, 1pps).
TARGET_PATTERN = re.compile(r"^[A-Za-z0-9+\-_.]{1,64}$")

app = Flask(__name__)


def validate_target(value) -> str:
    target = str(value or "").strip()
    if not TARGET_PATTERN.fullmatch(target):
        raise ValueError(f"Invalid target name: {value!r}")
    return target


def validate_number(value, field: str) -> str:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"'{field}' must be a number") from None
    if number < 0:
        raise ValueError(f"'{field}' must not be negative")
    return str(number)


@app.route('/trigger', methods=['POST'])
def trigger_observation():
    data = request.get_json(silent=True) or {}
    print("Received parameters:", data)
    try:
        params = {
            "target": validate_target(data.get("target")),
            "duration": validate_number(data.get("duration"), "duration"),
            "countdown": validate_number(data.get("countdown"), "countdown"),
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    run_observation(params)
    return {"status": "observation started"}, 200


def run_observation(params):
    target = params["target"]
    # The command is typed into the screen session's shell, so every argument is quoted.
    obs_cmd = shlex.join([observation_script, params["duration"], target, params["countdown"]])

    subprocess.run(["screen", "-S", target, "-X", "quit"], check=False)
    subprocess.run(["screen", "-dmS", target], check=False)
    subprocess.run(["screen", "-S", target, "-X", "stuff", f"{obs_cmd}\r"], check=False)


@app.route('/get-log', methods=['GET'])
def get_log_file():
    filename = request.args.get("filename")
    if not filename:
        return jsonify({"error": "Filename parameter missing"}), 400

    # basename() stops requests such as ../../etc/passwd from leaving the log directory.
    safe_filename = os.path.basename(filename)
    if not safe_filename or not os.path.isfile(os.path.join(log_files, safe_filename)):
        return jsonify({"error": "Log file not found"}), 404
    return send_from_directory(log_files, safe_filename, as_attachment=True)


@app.route('/get-prms', methods=['GET'])
def get_prms():
    total, used, free = shutil.disk_usage(log_files)
    gb = 1024 ** 3
    return jsonify({
        "total": f"{total / gb:.2f} GB",
        "used": f"{used / gb:.2f} GB",
        "free": f"{free / gb:.2f} GB",
    }), 200


@app.route('/observation-over', methods=['POST'])
def observation_over():
    try:
        pulsar_name = validate_target(request.args.get("pulsar_name"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    subprocess.run(["screen", "-S", pulsar_name, "-X", "quit"], check=False)

    print(f"Observation over received for {pulsar_name}")
    return jsonify({"status": "received", "pulsar_name": pulsar_name}), 200


if __name__ == '__main__':
    app.run(host=os.environ.get("DART_BIND_HOST", "0.0.0.0"), port=int(os.environ.get("DART_API_PORT", "6000")))
