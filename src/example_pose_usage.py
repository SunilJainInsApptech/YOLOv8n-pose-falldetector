"""
Example usage of YOLOv8n-pose model for fall detection with keypoints extraction.
"""

import asyncio
try:
    from yolov8n_pose import yolov8
    from viam.proto.app.robot import ComponentConfig
    from viam.resource.types import Model, ModelFamily
    from viam.utils import dict_to_struct
    VIAM_AVAILABLE = True
except ImportError:
    print("Viam SDK not available. This is just a demonstration of the keypoint structure.")
    VIAM_AVAILABLE = False

async def example_pose_detection():
    """
    Example of how to use the YOLOv8 pose detection service.
    """
    
    # Configure the service for pose detection
    config_dict = {
        "model_location": "yolov8n-pose.pt",  # Use pre-trained YOLOv8n-pose model
        "task": "pose"  # Specify pose estimation task
    }
    
    # Create component config
    config = ComponentConfig()
    config.name = "pose_detector"
    config.model = "viam-labs:vision:yolov8"
    config.attributes = dict_to_struct(config_dict)
    
    # Initialize the pose detection service
    pose_service = yolov8.new(config, {})
    
    print("YOLOv8n-pose service initialized successfully!")
    print("Available methods:")
    print("- get_pose_detections(image)")
    print("- get_pose_detections_from_camera(camera_name)")
    print("- analyze_pose_for_fall_detection(keypoints)")
    
    # Example of analyzing keypoints for fall detection
    example_keypoints = [
        {"name": "nose", "x": 320, "y": 400, "confidence": 0.9, "visible": True},
        {"name": "left_shoulder", "x": 300, "y": 350, "confidence": 0.8, "visible": True},
        {"name": "right_shoulder", "x": 340, "y": 350, "confidence": 0.8, "visible": True},
        {"name": "left_hip", "x": 310, "y": 380, "confidence": 0.7, "visible": True},
        {"name": "right_hip", "x": 330, "y": 380, "confidence": 0.7, "visible": True},
    ]
    
    fall_analysis = pose_service.analyze_pose_for_fall_detection(example_keypoints)
    print(f"\nFall analysis result: {fall_analysis}")

def print_keypoint_info():
    """
    Print information about COCO pose keypoints used by YOLOv8n-pose.
    """
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    print("YOLOv8n-pose uses COCO pose format with 17 keypoints:")
    for i, name in enumerate(keypoint_names):
        print(f"{i}: {name}")
    
    print("\nEach keypoint contains:")
    print("- x, y coordinates")
    print("- confidence score")
    print("- visibility flag (confidence > 0.5)")

if __name__ == "__main__":
    print("YOLOv8n-pose Fall Detection Example")
    print("===================================")
    
    print_keypoint_info()
    print("\n" + "="*50 + "\n")
    
    # Run the example
    asyncio.run(example_pose_detection())
