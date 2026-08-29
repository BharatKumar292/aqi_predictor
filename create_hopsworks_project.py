import os
from dotenv import load_dotenv
import hopsworks

load_dotenv(".env")

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    print("HOPSWORKS_API_KEY is not set.")
    raise SystemExit(1)

print("Connecting to Hopsworks...")

connection = hopsworks.connection(
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=api_key,
    engine="python"
)

print("Connected to Hopsworks!")
print("Creating project...")

project = hopsworks.create_project(
    "pearls_aqi_predictor",
    description="Air Quality Index prediction using pollution and weather data."
)

print("Project created successfully!")
print(f"Project name: {project.name}")

connection.close()