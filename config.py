from dotenv import load_dotenv
import os
load_dotenv()

ENV = os.getenv("ENV")
MOUNT_PATH = os.getenv("MOUNT_PATH")
print(f"ENV: {ENV}")
print(f"MOUNT_PATH: {MOUNT_PATH}")

if ENV == "Local":
    Supported_Models = [
        "prompthero/openjourney",
    ]
else:
    Supported_Models = [
        "prompthero/openjourney",
        "stabilityai/stable-diffusion-xl-base-1.0"
    ]