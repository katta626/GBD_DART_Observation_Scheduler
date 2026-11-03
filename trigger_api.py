"""Manual test client for the DSP Flask API (dsp_flask/trigger_api.py).

Running this file sends a REAL observation trigger, so it only runs when
executed directly, never on import.
"""
import requests

from dart_config import get_runtime_settings


def main() -> None:
    runtime_settings = get_runtime_settings()

    data = {
        "target": "J0437-4715",
        "duration": 300,
        "countdown": "100",
    }

    response = requests.post(
        runtime_settings["network"]["trigger_url"],
        json=data,
        timeout=runtime_settings["network"]["request_timeout_seconds"],
    )
    print("Trigger status:", response.status_code)

    filename = "1pps_09_09_2025_observation.log"
    url = runtime_settings["network"]["log_url_template"].format(filename=filename)

    response = requests.get(url, timeout=runtime_settings["network"]["request_timeout_seconds"])

    if response.status_code == 200:
        with open(filename, "wb") as file_handle:
            file_handle.write(response.content)
        print("Log downloaded successfully")
    else:
        try:
            print("Error:", response.json())
        except ValueError:
            print("Error:", response.text)


if __name__ == "__main__":
    main()
