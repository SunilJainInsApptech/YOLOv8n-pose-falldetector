import asyncio

from viam.module.module import Module
from yolov8n_pose import yolov8


if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())
