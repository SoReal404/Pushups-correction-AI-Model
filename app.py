import cv2
import mediapipe as mp
import joblib
import numpy as np
from collections import deque
import pyttsx3
import time
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 🔇 Suppress all warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# 📊 Feedback labels mapped to audio messages
feedback_map = {
    0: "Bad form. Keep your back straight.",
    1: "Great form! Keep going.",
    2: "Lower your hips. Engage your core.",
    3: "Raise your body. You're too low.",
    4: "Align your arms. Your elbows are too wide."
}

# 🧠 Load trained model
model = joblib.load("pushup_pose_model.pkl")

# 📐 Angle calculation for reps
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

# 🗣️ Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    print("🗣️", text)
    engine.say(text)
    engine.runAndWait()

# 🎯 Counters and trackers
counter = 0
stage = None
pred_history = deque(maxlen=5)
last_feedback = None
last_feedback_time = 0

# 🕵️ Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 📷 Start webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    speak("Push-up trainer ready. Begin when you're ready.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            # 🦾 Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            pose_row = []
            for lm in landmarks:
                pose_row += [lm.x, lm.y, lm.z]

            # 🧠 Predict form and smooth
            pred = model.predict([pose_row])[0]
            pred_history.append(pred)
            smoothed_pred = max(set(pred_history), key=pred_history.count)

            # 💪 Rep counting (right arm)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < 90:
                stage = "down"
            if angle > 160 and stage == "down":
                stage = "up"
                counter += 1
                speak(f"Great job. Rep number {counter}")

            # 📣 Feedback every 2.5 seconds only when form changes
            current_time = time.time()
            form_text = feedback_map.get(smoothed_pred, "Unknown form")
            if form_text != last_feedback and (current_time - last_feedback_time > 2.5):
                last_feedback_time = current_time
                last_feedback = form_text
                speak(form_text)

            # 🎨 Display counters and feedback
            cv2.putText(image, f'Reps: {counter}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(image, f'Form: {form_text}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        except Exception as e:
            # Fail silently
            pass

        # 🕸️ Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # 🖼️ Show output
        cv2.imshow('Push-up AI Trainer', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            speak(f"Session ended. You completed {counter} push-ups.")
            break

cap.release()
cv2.destroyAllWindows()
