import asyncio

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

# Import the vision service class
from yolov8 import yolov8

# Register the model with the Viam registry
Registry.register_resource_creator(
    Vision.SUBTYPE,
    yolov8.MODEL,
    ResourceCreatorRegistration(yolov8.new, yolov8.validate_config),
)

if __name__ == "__main__":
    asyncio.run(Module.run_from_registry())