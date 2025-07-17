import asyncio
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.info("Starting module...")

from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

logger.info("Imports successful, importing yolov8...")

# Import the vision service class
from yolov8 import yolov8

logger.info(f"yolov8 imported successfully. MODEL: {yolov8.MODEL}")

# Register the model with the Viam registry using string-based subtype
logger.info("Registering model with Viam registry...")
Registry.register_resource_creator(
    "rdk:service:vision",
    yolov8.MODEL,
    ResourceCreatorRegistration(yolov8.new, yolov8.validate_config),
)
logger.info("Model registered successfully!")

if __name__ == "__main__":
    logger.info("Starting Module.run_from_registry()...")
    asyncio.run(Module.run_from_registry())