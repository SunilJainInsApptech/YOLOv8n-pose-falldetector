"""
This file registers the model with the Python SDK.
"""

from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.services.vision import Vision

from .yolov8n_pose import yolov8

Registry.register_resource_creator(
    Vision.SUBTYPE,
    yolov8.MODEL,
    ResourceCreatorRegistration(yolov8.new, yolov8.validate_config),
)
