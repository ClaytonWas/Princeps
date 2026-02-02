"""
Gesture Data Collection Tool
============================
Record examples of your gestures to train a custom ML model.

GESTURES TO COLLECT:
- swipe_left: Swipe your hand to the left
- swipe_right: Swipe your hand to the right  
- select: Pinch thumb and index finger
- idle: Just hold your hand still (neutral)

INSTRUCTIONS:
1. Run this script
2. Press the key for the gesture you want to record:
   - L = swipe_left
   - R = swipe_right
   - S = select
   - I = idle
3. Perform the gesture while recording (2 seconds per sample)
4. Collect at least 30 samples per gesture
5. Press Q to quit

Data is saved to: gesture_data/
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import json
import urllib.request
from datetime import datetime

# =====================
# CONFIG
# =====================
CAMERA_INDEX = 0
RECORD_DURATION = 2.0  # Seconds per gesture sample
FRAME_RATE = 30  # Target FPS
DATA_DIR = "gesture_data"
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

GESTURES = {
    'l': 'swipe_left',
    'r': 'swipe_right', 
    's': 'select',
    'i': 'idle',
}

# =====================
# Setup
# =====================
os.makedirs(DATA_DIR, exist_ok=True)
for gesture in GESTURES.values():
    os.makedirs(os.path.join(DATA_DIR, gesture), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done!")

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
)


def landmarks_to_array(landmarks):
    """Convert landmarks to normalized numpy array."""
    if not landmarks:
        return None
    
    # Get all 21 landmarks as (x, y, z)
    points = []
    for lm in landmarks:
        points.append([lm.x, lm.y, lm.z])
    
    points = np.array(points)
    
    # Normalize relative to wrist (landmark 0)
    wrist = points[0]
    points = points - wrist  # Center on wrist
    
    # Scale by hand size (distance from wrist to middle finger MCP)
    scale = np.linalg.norm(points[9])  # MCP of middle finger
    if scale > 0.001:
        points = points / scale
    
    return points.flatten().tolist()


def get_sample_counts():
    """Count existing samples per gesture."""
    counts = {}
    for gesture in GESTURES.values():
        path = os.path.join(DATA_DIR, gesture)
        counts[gesture] = len([f for f in os.listdir(path) if f.endswith('.json')])
    return counts


def save_sample(gesture_name, frames):
    """Save a recorded gesture sample."""
    path = os.path.join(DATA_DIR, gesture_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(path, f"{gesture_name}_{timestamp}.json")
    
    data = {
        "gesture": gesture_name,
        "timestamp": timestamp,
        "frames": frames,
        "num_frames": len(frames)
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    return filename


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Gesture Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Collection", 1280, 720)
    
    recording = False
    recording_gesture = None
    recording_start = 0
    recorded_frames = []
    
    print("\n" + "="*50)
    print("GESTURE DATA COLLECTION")
    print("="*50)
    print("\nKeys:")
    print("  L = Record swipe_left")
    print("  R = Record swipe_right")
    print("  S = Record select")
    print("  I = Record idle")
    print("  Q = Quit")
    print("\n" + "="*50)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        
        # Detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)
        
        landmarks = None
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            
            # Draw hand
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            for c in connections:
                p1, p2 = landmarks[c[0]], landmarks[c[1]]
                cv2.line(frame,
                         (int(p1.x * w), int(p1.y * h)),
                         (int(p2.x * w), int(p2.y * h)),
                         (0, 255, 0), 2)
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)
        
        # Recording logic
        if recording:
            elapsed = now - recording_start
            remaining = RECORD_DURATION - elapsed
            
            if landmarks:
                frame_data = landmarks_to_array(landmarks)
                recorded_frames.append({
                    "time": elapsed,
                    "landmarks": frame_data
                })
            
            if elapsed >= RECORD_DURATION:
                # Save the recording
                if len(recorded_frames) >= 10:
                    filename = save_sample(recording_gesture, recorded_frames)
                    print(f"  ✓ Saved {len(recorded_frames)} frames to {filename}")
                else:
                    print(f"  ✗ Not enough frames ({len(recorded_frames)}), discarded")
                
                recording = False
                recording_gesture = None
                recorded_frames = []
            else:
                # Show recording UI
                cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 150), -1)
                cv2.putText(frame, f"RECORDING: {recording_gesture}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {remaining:.1f}s", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Progress bar
                progress = elapsed / RECORD_DURATION
                bar_w = int((w - 40) * progress)
                cv2.rectangle(frame, (20, 85), (w - 20, 95), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, 85), (20 + bar_w, 95), (0, 255, 0), -1)
        else:
            # Show status UI
            cv2.rectangle(frame, (0, 0), (w, 100), (40, 40, 40), -1)
            cv2.putText(frame, "Ready - Press L/R/S/I to record gesture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Show sample counts
            counts = get_sample_counts()
            count_text = " | ".join([f"{g}: {c}" for g, c in counts.items()])
            cv2.putText(frame, count_text, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if not landmarks:
                cv2.putText(frame, "NO HAND DETECTED", (w//2 - 150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imshow("Gesture Collection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif not recording and chr(key).lower() in GESTURES:
            gesture = GESTURES[chr(key).lower()]
            print(f"\n▶ Recording {gesture}...")
            recording = True
            recording_gesture = gesture
            recording_start = now
            recorded_frames = []
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("COLLECTION COMPLETE")
    print("="*50)
    counts = get_sample_counts()
    for gesture, count in counts.items():
        status = "✓" if count >= 30 else "✗ (need 30+)"
        print(f"  {gesture}: {count} samples {status}")
    print("\nNext step: Run train_gesture_model.py")


if __name__ == "__main__":
    main()
