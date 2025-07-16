import asyncio

from viam.module.module import Module
from yolov8 import yolov8n_pose


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
