import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from viam.module.module import Module

# Import to trigger model registration
from yolov8n_pose import yolov8


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
