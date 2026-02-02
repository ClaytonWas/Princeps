"""
Kingdom Manager Hand Mechanic System
=====================================
Clean, minimal, hyper-efficient gesture detection.
Priority: RELIABLE swipe detection that never fails.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import urllib.request
import os
from collections import deque

# =====================
# CONFIG
# =====================
CAMERA_INDEX = 0
RIGHT_HAND_DOMINANT = True  # False = left hand dominant

# Swipe detection - VERY PERMISSIVE
SWIPE_HISTORY_SIZE = 40     # Frames to track
SWIPE_MIN_DISTANCE = 0.06   # Normalized horizontal distance (lowered!)
SWIPE_MIN_VELOCITY = 0.08   # Distance per second (lowered!)
SWIPE_COOLDOWN = 0.5        # Seconds between swipes
SWIPE_DISPLAY_TIME = 1.0    # How long to show swipe result
DEBUG_SWIPE = True          # Print swipe analysis to console

# Cursor / Tracking Stabilization
CURSOR_SMOOTHING = 0.35     # Lower = more stable (0.2-0.5 recommended)
JUMP_THRESHOLD = 0.08       # Max allowed movement per frame (reject outliers)
USE_PALM_CENTER = True      # Use palm center instead of fingertip (more stable)

# Hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# =====================
# Setup MediaPipe
# =====================
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done!")

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
)

# =====================
# Stable Position Tracker
# =====================
class StableTracker:
    """
    Provides stable, jump-free position tracking.
    Uses exponential smoothing + outlier rejection.
    """
    def __init__(self, smoothing=0.35, jump_threshold=0.08):
        self.smoothing = smoothing
        self.jump_threshold = jump_threshold
        self.position = None
        self.velocity = np.array([0.0, 0.0])
        self.last_time = None
        self.stable_x_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.stable_t_history = deque(maxlen=SWIPE_HISTORY_SIZE)
    
    def update(self, raw_x: float, raw_y: float, t: float) -> tuple:
        """
        Update with raw position, returns stabilized (x, y).
        Also updates internal history for swipe detection.
        """
        raw = np.array([raw_x, raw_y])
        
        if self.position is None:
            self.position = raw
            self.last_time = t
            return raw_x, raw_y
        
        dt = t - self.last_time
        self.last_time = t
        
        # Calculate movement
        delta = raw - self.position
        distance = np.linalg.norm(delta)
        
        # Reject outliers (sudden jumps = tracking errors)
        if distance > self.jump_threshold:
            # Likely a tracking glitch - use prediction instead
            # Move slightly toward the new position but don't jump
            self.position = self.position + delta * 0.1
        else:
            # Normal update with exponential smoothing
            self.position = self.smoothing * raw + (1 - self.smoothing) * self.position
            
            # Update velocity (for potential future use)
            if dt > 0:
                self.velocity = delta / dt
        
        # Store stabilized position for swipe detection
        self.stable_x_history.append(self.position[0])
        self.stable_t_history.append(t)
        
        return self.position[0], self.position[1]
    
    def get_swipe_data(self):
        """Get stabilized position history for swipe analysis."""
        return list(self.stable_x_history), list(self.stable_t_history)
    
    def reset(self):
        self.position = None
        self.velocity = np.array([0.0, 0.0])
        self.last_time = None
        self.stable_x_history.clear()
        self.stable_t_history.clear()


def get_palm_center(landmarks):
    """
    Get stable palm center position (average of wrist + base of fingers).
    Much more stable than fingertip tracking.
    """
    # Use wrist (0) and base of each finger (5, 9, 13, 17)
    indices = [0, 5, 9, 13, 17]
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)
    return x, y


# =====================
# Swipe Detector Class
# =====================
class SwipeDetector:
    def __init__(self, history_size=30):
        self.x_history = deque(maxlen=history_size)
        self.t_history = deque(maxlen=history_size)
        self.last_swipe_time = 0
        self.last_result = None
        self.last_result_time = 0
    
    def update(self, x: float, t: float) -> str | None:
        """
        Update with new x position and timestamp.
        Returns "LEFT", "RIGHT", or None.
        """
        self.x_history.append(x)
        self.t_history.append(t)
        
        return self._check_swipe(t)
    
    def _check_swipe(self, t: float) -> str | None:
        """Check if current state contains a valid swipe."""
        # Need at least 5 samples
        if len(self.x_history) < 5:
            return None
        
        # Cooldown check
        if t - self.last_swipe_time < SWIPE_COOLDOWN:
            return None
        
        # Analyze the motion
        result = self._analyze()
        
        if result:
            self.last_swipe_time = t
            self.last_result = result
            self.last_result_time = t
            self.x_history.clear()
            self.t_history.clear()
        
        return result
    
    def _analyze(self) -> str | None:
        """Check if current history contains a valid swipe."""
        x = np.array(self.x_history)
        t = np.array(self.t_history)
        
        # Simple approach: just look at total movement over recent history
        dx = x[-1] - x[0]
        dt = t[-1] - t[0]
        
        if dt < 0.05:
            if DEBUG_SWIPE:
                print(f"  [swipe] dt={dt:.3f} too short")
            return None
        
        abs_dx = abs(dx)
        velocity = abs_dx / dt if dt > 0 else 0
        
        if DEBUG_SWIPE:
            print(f"  [swipe] dx={dx:+.3f}, dt={dt:.2f}s, vel={velocity:.2f}, need dist>{SWIPE_MIN_DISTANCE:.2f} vel>{SWIPE_MIN_VELOCITY:.2f}")
        
        # Check thresholds
        if abs_dx < SWIPE_MIN_DISTANCE:
            return None
        
        if velocity < SWIPE_MIN_VELOCITY:
            return None
        
        # Success!
        direction = "RIGHT" if dx > 0 else "LEFT"
        if DEBUG_SWIPE:
            print(f"  [swipe] >>> DETECTED: {direction}")
        return direction
    
    def get_display(self, t: float) -> str:
        """Get current display string."""
        if self.last_result and (t - self.last_result_time) < SWIPE_DISPLAY_TIME:
            if self.last_result == "RIGHT":
                return "→ SWIPE RIGHT →"
            else:
                return "← SWIPE LEFT ←"
        return "Ready"
    
    def reset(self):
        """Clear history."""
        self.x_history.clear()
        self.t_history.clear()


# =====================
# Main Application
# =====================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Kingdom Manager", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kingdom Manager", 1280, 720)
    
    swipe = SwipeDetector(SWIPE_HISTORY_SIZE)
    tracker = StableTracker(CURSOR_SMOOTHING, JUMP_THRESHOLD)
    
    # SEPARATE raw tracking for swipes (no smoothing!)
    raw_x_history = deque(maxlen=SWIPE_HISTORY_SIZE)
    raw_t_history = deque(maxlen=SWIPE_HISTORY_SIZE)
    
    print("Kingdom Manager Hand Control")
    print("- Swipe left/right with your dominant hand")
    print("- Press Q to quit")
    
    # Track stability metrics
    jump_count = 0
    frame_count = 0
    last_debug_time = 0
    
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
        result = detector.detect(mp_image)
        
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
            
            # Fallback to first hand
            if dominant is None and result.hand_landmarks:
                dominant = result.hand_landmarks[0]
        
        # Draw all hands
        if result.hand_landmarks:
            for i, landmarks in enumerate(result.hand_landmarks):
                is_dom = (landmarks == dominant)
                color = (0, 255, 0) if is_dom else (255, 200, 0)
                
                for c in HAND_CONNECTIONS:
                    p1 = landmarks[c[0]]
                    p2 = landmarks[c[1]]
                    cv2.line(frame,
                             (int(p1.x * w), int(p1.y * h)),
                             (int(p2.x * w), int(p2.y * h)),
                             color, 2)
                
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, color, -1)
        
        # Process dominant hand
        if dominant:
            frame_count += 1
            
            # Get position - palm center is more stable than fingertip
            if USE_PALM_CENTER:
                raw_x, raw_y = get_palm_center(dominant)
            else:
                tip = dominant[8]
                raw_x, raw_y = tip.x, tip.y
            
            # Track RAW positions for swipe (no smoothing!)
            raw_x_history.append(raw_x)
            raw_t_history.append(now)
            
            # Apply stabilization for CURSOR only
            stable_x, stable_y = tracker.update(raw_x, raw_y, now)
            
            cx, cy = int(stable_x * w), int(stable_y * h)
            
            # Draw cursor
            cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            
            # Also draw raw position (small red dot) to show stabilization working
            rx, ry = int(raw_x * w), int(raw_y * h)
            cv2.circle(frame, (rx, ry), 3, (0, 0, 255), -1)
            
            # Check for swipe using RAW history (unsmoothed!)
            if len(raw_x_history) >= 5 and now - swipe.last_swipe_time > SWIPE_COOLDOWN:
                swipe.x_history = deque(raw_x_history, maxlen=SWIPE_HISTORY_SIZE)
                swipe.t_history = deque(raw_t_history, maxlen=SWIPE_HISTORY_SIZE)
                
                swipe_result = swipe._check_swipe(now)
                
                if swipe_result:
                    print(f"=== SWIPE {swipe_result}! ===")
                    raw_x_history.clear()
                    raw_t_history.clear()
        else:
            tracker.reset()
            swipe.reset()
            raw_x_history.clear()
            raw_t_history.clear()
        
        # === UI ===
        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 70), (20, 20, 20), -1)
        
        # Gesture display
        display = swipe.get_display(now)
        color = (0, 255, 255) if "SWIPE" in display else (200, 200, 200)
        cv2.putText(frame, display, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Hand count
        hand_text = f"Hands: {num_hands}"
        if num_hands == 2:
            hand_text += " (TWO HANDS UP)"
        cv2.putText(frame, hand_text, (w - 300, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Debug: show tracking info
        if len(raw_x_history) > 1:
            dx = raw_x_history[-1] - raw_x_history[0]
            dt = raw_t_history[-1] - raw_t_history[0] if len(raw_t_history) > 1 else 0
            vel = abs(dx) / dt if dt > 0 else 0
            debug_text = f"dx:{dx:+.3f} dt:{dt:.2f}s vel:{vel:.2f}"
            cv2.putText(frame, debug_text, (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Show progress bar toward swipe threshold
            progress = min(abs(dx) / SWIPE_MIN_DISTANCE, 1.0)
            bar_w = int(200 * progress)
            bar_color = (0, 255, 0) if progress >= 1.0 else (100, 100, 255)
            cv2.rectangle(frame, (20, h - 50), (20 + bar_w, h - 35), bar_color, -1)
            cv2.rectangle(frame, (20, h - 50), (220, h - 35), (100, 100, 100), 1)
            cv2.putText(frame, f"{int(progress*100)}%", (225, h - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        cv2.imshow("Kingdom Manager", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
