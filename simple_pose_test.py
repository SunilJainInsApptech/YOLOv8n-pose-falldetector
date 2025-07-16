"""
Simple test script for YOLOv8n-pose keypoints extraction and fall detection.
No external dependencies required.
"""

def create_mock_keypoints_data():
    """Create mock keypoints data for testing."""
    # Simulate keypoints format: list of [x, y, confidence] for each keypoint
    
    # Person standing upright (normal pose)
    standing_person = [
        [320, 100, 0.9],  # nose
        [315, 95, 0.8],   # left_eye
        [325, 95, 0.8],   # right_eye
        [310, 95, 0.7],   # left_ear
        [330, 95, 0.7],   # right_ear
        [300, 150, 0.9],  # left_shoulder
        [340, 150, 0.9],  # right_shoulder
        [280, 200, 0.8],  # left_elbow
        [360, 200, 0.8],  # right_elbow
        [260, 250, 0.7],  # left_wrist
        [380, 250, 0.7],  # right_wrist
        [310, 280, 0.9],  # left_hip
        [330, 280, 0.9],  # right_hip
        [305, 380, 0.8],  # left_knee
        [335, 380, 0.8],  # right_knee
        [300, 480, 0.7],  # left_ankle
        [340, 480, 0.7],  # right_ankle
    ]
    
    # Person fallen down (horizontal pose)
    fallen_person = [
        [200, 300, 0.9],  # nose
        [195, 295, 0.8],  # left_eye
        [205, 295, 0.8],  # right_eye
        [190, 295, 0.7],  # left_ear
        [210, 295, 0.7],  # right_ear
        [150, 310, 0.9],  # left_shoulder
        [250, 310, 0.9],  # right_shoulder
        [100, 320, 0.8],  # left_elbow
        [300, 320, 0.8],  # right_elbow
        [50, 330, 0.7],   # left_wrist
        [350, 330, 0.7],  # right_wrist
        [160, 350, 0.9],  # left_hip
        [240, 350, 0.9],  # right_hip
        [120, 370, 0.8],  # left_knee
        [280, 370, 0.8],  # right_knee
        [80, 380, 0.7],   # left_ankle
        [320, 380, 0.7],  # right_ankle
    ]
    
    return [standing_person, fallen_person]

def extract_keypoints_from_data(keypoints_data, person_index=0):
    """Extract keypoints from mock data."""
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    if len(keypoints_data) <= person_index:
        return None
        
    # Get keypoints for the specific person
    kpts = keypoints_data[person_index]
    
    keypoints_list = []
    for i, name in enumerate(keypoint_names):
        if i < len(kpts):
            x, y, conf = kpts[i]
            keypoints_list.append({
                "name": name,
                "x": float(x),
                "y": float(y),
                "confidence": float(conf),
                "visible": conf > 0.5
            })
    
    return keypoints_list

def analyze_pose_for_fall_detection(keypoints):
    """Analyze pose keypoints to detect potential falls."""
    if not keypoints:
        return {"fall_detected": False, "confidence": 0.0, "reason": "No keypoints detected"}
    
    # Convert keypoints to dictionary for easier access
    kp_dict = {kp["name"]: kp for kp in keypoints if kp["visible"]}
    
    fall_indicators = []
    fall_score = 0.0
    
    # Check if person is lying down (horizontal orientation)
    if all(kp in kp_dict for kp in ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]):
        shoulder_center_y = (kp_dict["left_shoulder"]["y"] + kp_dict["right_shoulder"]["y"]) / 2
        hip_center_y = (kp_dict["left_hip"]["y"] + kp_dict["right_hip"]["y"]) / 2
        
        # If shoulders and hips are at similar height (horizontal position)
        y_diff = abs(shoulder_center_y - hip_center_y)
        if y_diff < 60:  # Increased threshold for better detection
            fall_indicators.append("horizontal_orientation")
            fall_score += 0.5  # Increased weight
            print(f"    DEBUG: Horizontal orientation detected (shoulder_y: {shoulder_center_y:.1f}, hip_y: {hip_center_y:.1f}, diff: {y_diff:.1f})")
    
    # Check if head is lower than hips
    if all(kp in kp_dict for kp in ["nose", "left_hip", "right_hip"]):
        nose_y = kp_dict["nose"]["y"]
        hip_center_y = (kp_dict["left_hip"]["y"] + kp_dict["right_hip"]["y"]) / 2
        
        if nose_y > hip_center_y:
            fall_indicators.append("head_below_hips")
            fall_score += 0.4  # Increased weight
            print(f"    DEBUG: Head below hips detected (nose_y: {nose_y:.1f}, hip_y: {hip_center_y:.1f})")
    
    # Check for body aspect ratio (width vs height)
    if all(kp in kp_dict for kp in ["nose", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"]):
        # Calculate body height and width
        min_y = min(kp_dict["nose"]["y"], kp_dict["left_shoulder"]["y"], kp_dict["right_shoulder"]["y"])
        max_y = max(kp_dict["left_ankle"]["y"], kp_dict["right_ankle"]["y"])
        body_height = max_y - min_y
        
        body_width = abs(kp_dict["right_shoulder"]["x"] - kp_dict["left_shoulder"]["x"])
        
        if body_height > 0 and body_width > 0:
            aspect_ratio = body_width / body_height
            if aspect_ratio > 0.8:  # More horizontal than vertical
                fall_indicators.append("horizontal_aspect_ratio")
                fall_score += 0.3
                print(f"    DEBUG: Horizontal aspect ratio detected (width/height: {aspect_ratio:.2f})")
    
    # Check for unusual limb positions
    limbs_on_ground = 0
    if keypoints:
        ground_threshold_y = max([kp["y"] for kp in keypoints if kp["visible"]], default=0) - 50
        
        for limb in ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]:
            if limb in kp_dict and kp_dict[limb]["y"] > ground_threshold_y:
                limbs_on_ground += 1
        
        if limbs_on_ground >= 3:
            fall_indicators.append("multiple_limbs_on_ground")
            fall_score += 0.3
            print(f"    DEBUG: Multiple limbs on ground detected ({limbs_on_ground} limbs)")
    
    fall_detected = fall_score > 0.5
    
    return {
        "fall_detected": fall_detected,
        "confidence": fall_score,
        "indicators": fall_indicators,
        "keypoints_analyzed": len(kp_dict)
    }

def main():
    """Main test function."""
    print("YOLOv8n-pose Keypoints Extraction and Fall Detection Test")
    print("="*60)
    
    # Create test data
    mock_data = create_mock_keypoints_data()
    print(f"Created mock data for {len(mock_data)} people")
    
    test_cases = [
        ("Standing Person", 0, "Should NOT detect fall"),
        ("Fallen Person", 1, "Should detect fall")
    ]
    
    for case_name, person_idx, expected in test_cases:
        print(f"\n{'-'*50}")
        print(f"Testing: {case_name} ({expected})")
        print(f"{'-'*50}")
        
        # Extract keypoints
        keypoints = extract_keypoints_from_data(mock_data, person_idx)
        
        if keypoints:
            print(f"Extracted {len(keypoints)} keypoints")
            
            # Show key body points
            key_points = ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
            print("Key body points:")
            for kp in keypoints:
                if kp["name"] in key_points:
                    print(f"  {kp['name']:>15}: ({kp['x']:>5.1f}, {kp['y']:>5.1f}) conf={kp['confidence']:.2f}")
            
            # Analyze for fall
            print("\nFall Detection Analysis:")
            fall_result = analyze_pose_for_fall_detection(keypoints)
            
            print(f"  Fall Detected: {fall_result['fall_detected']}")
            print(f"  Confidence: {fall_result['confidence']:.2f}")
            print(f"  Indicators: {fall_result['indicators']}")
            print(f"  Keypoints Analyzed: {fall_result['keypoints_analyzed']}")
            
            # Validation
            if case_name == "Standing Person" and not fall_result['fall_detected']:
                print("  ✓ CORRECT: No fall detected for standing person")
            elif case_name == "Fallen Person" and fall_result['fall_detected']:
                print("  ✓ CORRECT: Fall detected for fallen person")
            else:
                print("  ✗ INCORRECT: Unexpected result")
        else:
            print("  ✗ ERROR: No keypoints extracted")
    
    print(f"\n{'='*60}")
    print("Test completed!")
    print("\nThis demonstrates how the YOLOv8n-pose model will:")
    print("1. Extract 17 COCO keypoints from detected persons")
    print("2. Analyze pose geometry to detect falls")
    print("3. Provide confidence scores and indicators")

if __name__ == "__main__":
    main()
