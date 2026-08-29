import os
import hopsworks

from src.config import HOPSWORKS_API_KEY

# Name and version of our feature group
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

_project = None


def _ensure_tmp_folder_exists():
    """
    On Windows, the hopsworks library tries to use a '/tmp' path
    internally for storing connection certificates. Windows turns
    '/tmp' into 'C:\\tmp', which crashes with WinError 3 if that
    folder doesn't already exist. Creating it ahead of time fixes it.
    On Linux/Mac (like GitHub Actions), /tmp already exists, so this
    does nothing there.
    """
    if os.name == "nt":
        os.makedirs(r"C:\tmp", exist_ok=True)


def get_project():
    """
    Connect to the Hopsworks project.
    """

    global _project

    if _project is not None:
        return _project

    if not HOPSWORKS_API_KEY:
        print("[hopsworks] HOPSWORKS_API_KEY is not set.")
        return None

    _ensure_tmp_folder_exists()

    try:
        cert_folder = os.path.join(
            os.environ.get("TEMP", r"C:\tmp"),
            "hopsworks_certs"
        )

        os.makedirs(
            cert_folder,
            exist_ok=True
        )

        _project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY,
            cert_folder=cert_folder,
            engine="python"
        )

        print(
            "[hopsworks] Connected successfully."
        )

        return _project

    except Exception as e:
        print(
            f"[hopsworks] Login failed: {e}"
        )
        return None


def get_feature_group():

    project = get_project()

    if project is None:
        return None

    try:
        fs = project.get_feature_store()

        fg = fs.get_or_create_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION,
            description=(
                "AQI features for Sukkur, Karachi and Lahore."
            ),
            primary_key=[
                "city",
                "timestamp_unix"
            ],
            event_time="timestamp",
            online_enabled=False,
            time_travel_format="HUDI",
        )

        return fg

    except Exception as e:
        print(
            f"[hopsworks] Could not access "
            f"feature group: {e}"
        )
        return None


def get_model_registry():

    project = get_project()

    if project is None:
        return None

    try:
        return project.get_model_registry()

    except Exception as e:
        print(
            f"[hopsworks] Could not access "
            f"model registry: {e}"
        )
        return None