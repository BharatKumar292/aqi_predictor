"""
Hopsworks connection for the AQI project.

This file handles the connection to Hopsworks,
the feature store, and the model registry.
"""

import hopsworks

from src.config import HOPSWORKS_API_KEY

# Name and version of our feature group
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

_project = None

def get_project():
    """
    Connect to the Hopsworks project.
    The connection is saved so we do not log in
    again every time the function is called.
    """

    global _project

    if _project is not None:
        return _project

    if not HOPSWORKS_API_KEY:
        print( "[hopsworks] HOPSWORKS_API_KEY is not set.")
        return None

    try:
        _project = hopsworks.login(
            api_key_value=HOPSWORKS_API_KEY
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
    """
    Get the AQI feature group from Hopsworks.

    If the feature group does not exist,
    Hopsworks will create it.
    """

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
        )

        return fg

    except Exception as e:
        print(
            f"[hopsworks] Could not access "
            f"feature group: {e}"
        )
        return None

def get_model_registry():
    # Get the Hopsworks model registry.

    project = get_project()

    if project is None:
        return None

    try:
        return project.get_model_registry()

    except Exception as e:
        print(f"[hopsworks] Could not access "
            f"model registry: {e}"
        )
        return None
