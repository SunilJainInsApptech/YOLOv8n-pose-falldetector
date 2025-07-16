# YOLOv8n-pose Fall Detection

This project implements a YOLOv8n-pose model for fall detection with keypoints extraction using the Viam robotics platform.

## Features

- **Pose Detection**: Extract 17 COCO keypoints from human poses
- **Fall Detection**: Analyze pose keypoints to detect potential falls
- **Keypoints Extraction**: Get detailed x,y coordinates and confidence scores for each keypoint
- **Camera Integration**: Works with Viam camera components

## Keypoints Structure

The YOLOv8n-pose model uses the COCO pose format with 17 keypoints:

```
0: nose          5: left_shoulder   10: left_wrist    15: left_knee
1: left_eye      6: right_shoulder  11: right_wrist   16: left_ankle
2: right_eye     7: left_elbow      12: left_hip      17: right_ankle  
3: left_ear      8: right_elbow     13: right_hip
4: right_ear     9: left_wrist      14: right_knee
```

Each keypoint contains:
- `x`, `y`: Pixel coordinates
- `confidence`: Detection confidence (0-1)
- `visible`: Boolean indicating if confidence > 0.5

## Usage

### Basic Pose Detection

```python
# Get pose detections from an image
pose_detections = await pose_service.get_pose_detections(image)

# Get pose detections from a camera
pose_detections = await pose_service.get_pose_detections_from_camera("camera_name")
```

### Fall Detection Analysis

```python
# Analyze keypoints for fall detection
for detection in pose_detections:
    if "keypoints" in detection:
        fall_analysis = pose_service.analyze_pose_for_fall_detection(detection["keypoints"])
        
        if fall_analysis["fall_detected"]:
            print(f"Fall detected with confidence: {fall_analysis['confidence']}")
            print(f"Indicators: {fall_analysis['indicators']}")
```

## Configuration

Configure the service to use YOLOv8n-pose:

```json
{
  "model_location": "yolov8n-pose.pt",
  "task": "pose"
}
```

## Fall Detection Algorithm

The fall detection analyzes several indicators:

1. **Horizontal Orientation**: Person lying down (shoulders and hips at similar height)
2. **Head Below Hips**: Unusual body position indicating a fall
3. **Multiple Limbs on Ground**: Several extremities touching the ground

Fall scores above 0.5 trigger a fall detection alert.

## Model Setup

1. The service will automatically download the YOLOv8n-pose model if not present
2. For custom models, specify the path in `model_location`
3. Set `task: "pose"` to enable pose estimation mode

## API Methods

### Core Methods
- `get_pose_detections(image)`: Get pose data from image
- `get_pose_detections_from_camera(camera_name)`: Get pose data from camera
- `extract_keypoints(keypoints, index)`: Extract keypoints from YOLO results
- `analyze_pose_for_fall_detection(keypoints)`: Analyze pose for fall detection

### Standard Vision Methods (Enhanced with Keypoints)
- `get_detections(image)`: Standard detections with keypoints if available
- `get_detections_from_camera(camera_name)`: Camera detections with keypoints

## Example Output

```python
{
    "confidence": 0.85,
    "class_name": "person",
    "x_min": 100, "y_min": 50, "x_max": 300, "y_max": 400,
    "keypoints": [
        {
            "name": "nose",
            "x": 200.5,
            "y": 80.3,
            "confidence": 0.92,
            "visible": True
        },
        # ... more keypoints
    ]
}
```

## Fall Detection Output

```python
{
    "fall_detected": True,
    "confidence": 0.75,
    "indicators": ["horizontal_orientation", "head_below_hips"],
    "keypoints_analyzed": 12
}
```
