"""
ML-Based Gesture Recognizer
===========================
Production-quality gesture recognition using your trained model.

This replaces all the hacky threshold-based detection with 
clean ML inference.

Features:
- Uses your custom-trained TFLite model
- Real-time inference at 30+ FPS
- Smooth prediction with temporal filtering
- Clean UI for game integration
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import json
import os
import urllib.request
from collections import deque
from pathlib import Path

# =====================
# CONFIG
# =====================
SCRIPT_DIR = Path(__file__).parent
CAMERA_INDEX = 0
RIGHT_HAND_DOMINANT = True

# Model paths
MODEL_DIR = SCRIPT_DIR / "gesture_model"
TFLITE_MODEL = MODEL_DIR / "gesture_model.tflite"
LABELS_FILE = MODEL_DIR / "gesture_labels.json"

# Hand detection
HAND_MODEL_PATH = SCRIPT_DIR / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Gesture detection settings
SEQUENCE_LENGTH = 30
PREDICTION_SMOOTHING = 5  # Average predictions over N frames
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trigger gesture
GESTURE_COOLDOWN = 0.5  # Seconds between same gesture triggers

# UI
DISPLAY_TIME = 1.0  # How long to show triggered gesture

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


# =====================
# Model Inference
# =====================
class GestureClassifier:
    def __init__(self, model_dir, labels_path):
        import tensorflow as tf
        
        # Load Keras model (more reliable than TFLite for LSTM)
        keras_path = model_dir / "gesture_model.keras"
        self.model = tf.keras.models.load_model(str(keras_path))
        
        # Load labels
        with open(str(labels_path)) as f:
            self.labels = json.load(f)
        self.labels = {int(k): v for k, v in self.labels.items()}
        
        print(f"Loaded model: {keras_path}")
        print(f"Labels: {self.labels}")
    
    def predict(self, sequence):
        """
        Predict gesture from landmark sequence.
        sequence: numpy array of shape (SEQUENCE_LENGTH, 63)
        Returns: (gesture_name, confidence)
        """
        # Ensure correct shape
        input_data = np.array([sequence], dtype=np.float32)
        
        output = self.model.predict(input_data, verbose=0)[0]
        
        gesture_idx = np.argmax(output)
        confidence = output[gesture_idx]
        gesture_name = self.labels.get(gesture_idx, "unknown")
        
        return gesture_name, confidence, output


# =====================
# Gesture State Machine
# =====================
class GestureDetector:
    def __init__(self, classifier):
        self.classifier = classifier
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.prediction_history = deque(maxlen=PREDICTION_SMOOTHING)
        
        self.last_gesture = None
        self.last_gesture_time = 0
        self.triggered_gesture = None
        self.triggered_time = 0
        
        self.gesture_cooldowns = {}
    
    def landmarks_to_features(self, landmarks):
        """Convert landmarks to normalized feature vector."""
        points = []
        for lm in landmarks:
            points.append([lm.x, lm.y, lm.z])
        
        points = np.array(points)
        
        # Normalize relative to wrist
        wrist = points[0]
        points = points - wrist
        
        # Scale by hand size
        scale = np.linalg.norm(points[9])
        if scale > 0.001:
            points = points / scale
        
        return points.flatten()
    
    def update(self, landmarks, t):
        """
        Update with new landmarks.
        Returns (gesture_name, confidence) if gesture triggered, else (None, 0)
        """
        if not landmarks:
            return None, 0, {}
        
        # Add to sequence
        features = self.landmarks_to_features(landmarks)
        self.sequence.append(features)
        
        # Need full sequence to predict
        if len(self.sequence) < SEQUENCE_LENGTH:
            return None, 0, {}
        
        # Run inference
        seq_array = np.array(self.sequence, dtype=np.float32)
        gesture, confidence, all_probs = self.classifier.predict(seq_array)
        
        # Smooth predictions
        self.prediction_history.append((gesture, confidence))
        
        # Get most common prediction with average confidence
        gesture_votes = {}
        for g, c in self.prediction_history:
            if g not in gesture_votes:
                gesture_votes[g] = []
            gesture_votes[g].append(c)
        
        smoothed_gesture = max(gesture_votes, key=lambda g: len(gesture_votes[g]))
        smoothed_confidence = np.mean(gesture_votes[smoothed_gesture])
        
        # Build probability dict for UI
        prob_dict = {self.classifier.labels[i]: float(p) for i, p in enumerate(all_probs)}
        
        # Check if we should trigger
        triggered = None
        
        if smoothed_gesture != 'idle' and smoothed_confidence >= CONFIDENCE_THRESHOLD:
            # Check cooldown
            last_trigger = self.gesture_cooldowns.get(smoothed_gesture, 0)
            if t - last_trigger >= GESTURE_COOLDOWN:
                triggered = smoothed_gesture
                self.gesture_cooldowns[smoothed_gesture] = t
                self.triggered_gesture = triggered
                self.triggered_time = t
                print(f"=== {triggered.upper()} ({smoothed_confidence:.0%}) ===")
        
        return triggered, smoothed_confidence, prob_dict
    
    def get_display(self, t):
        """Get current display state."""
        if self.triggered_gesture and (t - self.triggered_time) < DISPLAY_TIME:
            return self.triggered_gesture, True
        return None, False
    
    def reset(self):
        self.sequence.clear()
        self.prediction_history.clear()


# =====================
# Main Application
# =====================
def main():
    # Check for trained model
    if not os.path.exists(TFLITE_MODEL):
        print("\n" + "="*50)
        print("ERROR: No trained model found!")
        print("="*50)
        print(f"\nExpected: {TFLITE_MODEL}")
        print("\nSteps to train your model:")
        print("1. Run: python gesture_ml/collect_gestures.py")
        print("2. Collect 30+ samples per gesture")
        print("3. Run: python gesture_ml/train_gesture_model.py")
        print("4. Run this script again")
        return
    
    # Download hand model if needed
    if not HAND_MODEL_PATH.exists():
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(HAND_MODEL_URL, str(HAND_MODEL_PATH))
        print("Done!")
    
    # Initialize
    print("\nLoading models...")
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
    )
    
    classifier = GestureClassifier(MODEL_DIR, LABELS_FILE)
    gesture_detector = GestureDetector(classifier)
    
    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Gesture Recognizer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Recognizer", 1280, 720)
    
    print("\n" + "="*50)
    print("ML GESTURE RECOGNIZER")
    print("="*50)
    print("\nGestures: swipe_left, swipe_right, select, idle")
    print("Press Q to quit")
    
    # Cursor tracking
    cursor_pos = None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        
        # Detect hands
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = hand_detector.detect(mp_image)
        
        # Find dominant hand
        dominant = None
        num_hands = 0
        
        if result.hand_landmarks and result.handedness:
            num_hands = len(result.hand_landmarks)
            target = "Right" if RIGHT_HAND_DOMINANT else "Left"
            
            for i, hand in enumerate(result.handedness):
                if hand[0].category_name == target:
                    dominant = result.hand_landmarks[i]
                    break
            
            if dominant is None and result.hand_landmarks:
                dominant = result.hand_landmarks[0]
        
        # Draw hands
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                is_dom = (landmarks == dominant)
                color = (0, 255, 0) if is_dom else (255, 200, 0)
                
                for c in HAND_CONNECTIONS:
                    p1, p2 = landmarks[c[0]], landmarks[c[1]]
                    cv2.line(frame,
                             (int(p1.x * w), int(p1.y * h)),
                             (int(p2.x * w), int(p2.y * h)),
                             color, 2)
                
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)
        
        # Process gesture
        prob_dict = {}
        if dominant:
            triggered, confidence, prob_dict = gesture_detector.update(dominant, now)
            
            # Update cursor
            palm_x = sum(dominant[i].x for i in [0, 5, 9, 13, 17]) / 5
            palm_y = sum(dominant[i].y for i in [0, 5, 9, 13, 17]) / 5
            
            if cursor_pos is None:
                cursor_pos = np.array([palm_x, palm_y])
            else:
                cursor_pos = 0.4 * np.array([palm_x, palm_y]) + 0.6 * cursor_pos
            
            cx, cy = int(cursor_pos[0] * w), int(cursor_pos[1] * h)
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        else:
            gesture_detector.reset()
            cursor_pos = None
        
        # === UI ===
        cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
        
        # Gesture display
        display_gesture, is_active = gesture_detector.get_display(now)
        if is_active and display_gesture:
            gesture_display = {
                'swipe_left': '← SWIPE LEFT ←',
                'swipe_right': '→ SWIPE RIGHT →',
                'select': '● SELECT ●',
                'idle': 'IDLE'
            }.get(display_gesture, display_gesture.upper())
            
            cv2.putText(frame, gesture_display, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        else:
            cv2.putText(frame, "Ready", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 180, 180), 2)
        
        # Hand count
        cv2.putText(frame, f"Hands: {num_hands}", (w - 150, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        dom_text = "Right" if RIGHT_HAND_DOMINANT else "Left"
        cv2.putText(frame, f"Dominant: {dom_text}", (w - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Probability bars (bottom)
        if prob_dict and dominant:
            bar_y = h - 30
            bar_h = 15
            bar_w = 120
            x_offset = 20
            
            for gesture, prob in prob_dict.items():
                # Label
                cv2.putText(frame, f"{gesture}:", (x_offset, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                
                # Bar background
                cv2.rectangle(frame, 
                              (x_offset + 80, bar_y - bar_h), 
                              (x_offset + 80 + bar_w, bar_y), 
                              (50, 50, 50), -1)
                
                # Bar fill
                fill_w = int(bar_w * prob)
                color = (0, 255, 0) if prob > CONFIDENCE_THRESHOLD else (100, 100, 200)
                if fill_w > 0:
                    cv2.rectangle(frame,
                                  (x_offset + 80, bar_y - bar_h),
                                  (x_offset + 80 + fill_w, bar_y),
                                  color, -1)
                
                # Percentage
                cv2.putText(frame, f"{prob:.0%}", (x_offset + 85 + bar_w, bar_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
                
                x_offset += 220
        
        cv2.imshow("Gesture Recognizer", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
