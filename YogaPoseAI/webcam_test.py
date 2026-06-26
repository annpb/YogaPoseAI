import cv2
import mediapipe as mp

# ================= MEDIAPIPE SETUP =================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ================= WEBCAM =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ ERROR: Webcam not opening")
    exit()

print("✅ Webcam started... Press 'q' to quit")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to get frame")
        break

    print("🔄 Running frame...")

    # Convert BGR → RGB (VERY IMPORTANT)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose
    results = pose.process(rgb)

    # Check pose detection
    if results.pose_landmarks:
        print("🧍 POSE DETECTED")

        # Draw skeleton
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    else:
        print("❌ No pose detected")

    # Show window
    cv2.imshow("Yoga Pose Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
