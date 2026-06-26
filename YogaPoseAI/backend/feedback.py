def generate_feedback(angles):
    """
    Generate real-time pose correction feedback based on joint angles
    """
    feedback = []
    
    if not angles:
        return feedback
    
    # Elbow corrections
    if angles.get("left_elbow", 0) < 100:
        feedback.append("Bend your left elbow more")
    
    if angles.get("right_elbow", 0) < 100:
        feedback.append("Bend your right elbow more")
    
    if angles.get("left_elbow", 180) > 160:
        feedback.append("Straighten your left arm more")
    
    if angles.get("right_elbow", 180) > 160:
        feedback.append("Straighten your right arm more")
    
    # Knee corrections
    if angles.get("left_knee", 0) < 100:
        feedback.append("Bend your left knee more")
    
    if angles.get("right_knee", 0) < 100:
        feedback.append("Bend your right knee more")
    
    if angles.get("left_knee", 180) > 160:
        feedback.append("Straighten your left leg")
    
    if angles.get("right_knee", 180) > 160:
        feedback.append("Straighten your right leg")
    
    # Hip alignment
    if angles.get("left_hip", 0) < 80:
        feedback.append("Open your hips wider")
    
    # Shoulder corrections
    if angles.get("left_shoulder", 0) < 60:
        feedback.append("Roll your shoulders back - left side")
    
    if angles.get("right_shoulder", 0) < 60:
        feedback.append("Roll your shoulders back - right side")
    
    return feedback
