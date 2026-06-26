import cv2
import mediapipe as mp
import numpy as np
import torch
from backend.model_loader import model, scaler, label_map
from backend.pose_utils import extract_landmarks, extract_angles
from backend.feedback import generate_feedback

# -------------------------
# MediaPipe Setup
# -------------------------
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils


# -------------------------
# Frame Processing
# -------------------------
def process_frame(frame):

    try:

        if frame is None:
            return frame, "No Pose", 0.0, ["Frame is None"]

        if len(frame.shape) != 3:
            return frame, "Invalid Frame", 0.0, ["Bad frame shape"]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        landmarks = extract_landmarks(results)

        # No pose detected
        if landmarks is None or not results.pose_landmarks:
            return frame, "No Pose", 0.0, ["No person detected"]

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Visibility-based confidence
        lm = results.pose_landmarks.landmark
        confidence = float(np.mean([p.visibility for p in lm]))

        # ===== CLASSIFY POSE =====
        try:
            # Normalize landmarks
            landmarks_scaled = scaler.transform([landmarks])
            
            print(f"DEBUG: Landmarks shape: {landmarks.shape}, Scaled shape: {landmarks_scaled.shape}")
            print(f"DEBUG: Raw landmarks (first 10): {landmarks[:10]}")
            print(f"DEBUG: Scaled landmarks (first 10): {landmarks_scaled[0][:10]}")
            
            # Run model inference
            with torch.no_grad():
                landmarks_tensor = torch.FloatTensor(landmarks_scaled)
                output = model(landmarks_tensor)
                probabilities = torch.softmax(output, dim=1)
                pose_class = torch.argmax(output, dim=1).item()
                pose_confidence = probabilities[0][pose_class].item()
                pose_name = label_map.get(pose_class, "Unknown Pose")
                # Top-3 classes
                topk = torch.topk(probabilities, k=3, dim=1)
                top_indices = topk.indices[0].cpu().numpy().tolist()
                top_scores = topk.values[0].cpu().numpy().tolist()
                top_list = [f"{label_map.get(i,'?')}:{s:.2f}" for i, s in zip(top_indices, top_scores)]
                
                # Log ALL probabilities
                print(f"DEBUG: ALL PROBS: {[f'{label_map.get(i, \"?\")}:{probabilities[0][i]:.4f}' for i in range(len(label_map))]}")
            
            print(f"DEBUG: Pose: {pose_name}, Model Confidence: {pose_confidence:.2f}, Visibility Confidence: {confidence:.2f}, Top: {top_list}")
            
            # ===== EXTRACT ANGLES & GENERATE FEEDBACK =====
            try:
                angles = extract_angles(results)
                feedback = generate_feedback(angles)
                print(f"DEBUG: Angles: {angles}")
                print(f"DEBUG: Feedback: {feedback}")
            except Exception as angle_error:
                print("Angle Error:", angle_error)
                feedback = ["Could not calculate angles"]

            return (
                frame,
                pose_name,
                max(confidence, pose_confidence),
                feedback if feedback else ["Pose looks good!"]
            )

        except Exception as model_error:
            print("Model Error:", model_error)
            import traceback
            traceback.print_exc()
            return (
                frame,
                "Pose Detected",
                confidence,
                ["Pose detected but classification failed"]
            )

    except Exception as e:

        print("ERROR:", e)

        return (
            frame,
            "Error",
            0.0,
            [str(e)]
        )
