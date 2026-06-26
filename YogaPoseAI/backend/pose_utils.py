import numpy as np

def extract_landmarks(results):
    """
    Extract landmarks in format used for training: x0, y0, x1, y1, ..., x32, y32
    Model expects 80 features total
    """
    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark
    landmarks = []
    
    # Extract x, y for all 33 landmarks (66 features)
    for idx in range(33):
        if idx < len(lm):
            landmarks.append(lm[idx].x)
            landmarks.append(lm[idx].y)
        else:
            landmarks.append(0.0)
            landmarks.append(0.0)
    
    # Add angle features in the same order as training dataset
    angles = extract_angles(results)

    angle_order = [
        'left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
        'left_knee', 'right_knee', 'left_hip', 'right_hip',
        'left_ankle', 'right_ankle', 'torso_left', 'torso_right',
        'left_wrist', 'right_wrist'
    ]

    for k in angle_order:
        landmarks.append(float(angles.get(k, 0.0)))

    # Ensure exactly 80 features
    if len(landmarks) < 80:
        landmarks += [0.0] * (80 - len(landmarks))

    return np.array(landmarks[:80])


def calculate_angle(a, b, c):
    """
    Calculate angle in degrees between three points
    a, b, c are [x, y] or [x, y, z] coordinates
    b is the vertex
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees


def extract_angles(results):
    """
    Extract key joint angles from pose landmarks
    Returns dict with angle names as keys
    """
    if not results.pose_landmarks:
        return {}
    
    lm = results.pose_landmarks.landmark
    
    angles = {}
    
    # Shoulders
    angles["left_shoulder"] = calculate_angle(
        [lm[12].x, lm[12].y],
        [lm[11].x, lm[11].y],
        [lm[13].x, lm[13].y]
    )
    
    angles["right_shoulder"] = calculate_angle(
        [lm[11].x, lm[11].y],
        [lm[12].x, lm[12].y],
        [lm[14].x, lm[14].y]
    )
    
    # Elbows
    angles["left_elbow"] = calculate_angle(
        [lm[11].x, lm[11].y],
        [lm[13].x, lm[13].y],
        [lm[15].x, lm[15].y]
    )
    
    angles["right_elbow"] = calculate_angle(
        [lm[12].x, lm[12].y],
        [lm[14].x, lm[14].y],
        [lm[16].x, lm[16].y]
    )
    
    # Hips
    angles["left_hip"] = calculate_angle(
        [lm[12].x, lm[12].y],
        [lm[23].x, lm[23].y],
        [lm[25].x, lm[25].y]
    )
    
    angles["right_hip"] = calculate_angle(
        [lm[11].x, lm[11].y],
        [lm[24].x, lm[24].y],
        [lm[26].x, lm[26].y]
    )
    
    # Knees
    angles["left_knee"] = calculate_angle(
        [lm[23].x, lm[23].y],
        [lm[25].x, lm[25].y],
        [lm[27].x, lm[27].y]
    )
    
    angles["right_knee"] = calculate_angle(
        [lm[24].x, lm[24].y],
        [lm[26].x, lm[26].y],
        [lm[28].x, lm[28].y]
    )
    
    # Ankles
    try:
        angles["left_ankle"] = calculate_angle(
            [lm[25].x, lm[25].y],
            [lm[27].x, lm[27].y],
            [lm[23].x, lm[23].y]
        )
    except Exception:
        angles["left_ankle"] = 0.0

    try:
        angles["right_ankle"] = calculate_angle(
            [lm[26].x, lm[26].y],
            [lm[28].x, lm[28].y],
            [lm[24].x, lm[24].y]
        )
    except Exception:
        angles["right_ankle"] = 0.0

    # Wrists
    try:
        angles["left_wrist"] = calculate_angle(
            [lm[13].x, lm[13].y],
            [lm[15].x, lm[15].y],
            [lm[11].x, lm[11].y]
        )
    except Exception:
        angles["left_wrist"] = 0.0

    try:
        angles["right_wrist"] = calculate_angle(
            [lm[14].x, lm[14].y],
            [lm[16].x, lm[16].y],
            [lm[12].x, lm[12].y]
        )
    except Exception:
        angles["right_wrist"] = 0.0

    # Torso approximations (angle between shoulder-hip-hip)
    try:
        angles["torso_left"] = calculate_angle(
            [lm[11].x, lm[11].y],
            [lm[23].x, lm[23].y],
            [lm[24].x, lm[24].y]
        )
    except Exception:
        angles["torso_left"] = 0.0

    try:
        angles["torso_right"] = calculate_angle(
            [lm[12].x, lm[12].y],
            [lm[24].x, lm[24].y],
            [lm[23].x, lm[23].y]
        )
    except Exception:
        angles["torso_right"] = 0.0

    return angles