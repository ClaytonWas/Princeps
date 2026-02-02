"""
Kingdom Manager Hand Mechanic System
=====================================
Reliable swipe detection + pinch-to-select.
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

# Swipe detection (movement-based)
SWIPE_HISTORY_SIZE = 40
SWIPE_MIN_DISTANCE = 0.06
SWIPE_MIN_VELOCITY = 0.08
SWIPE_COOLDOWN = 0.5
SWIPE_DISPLAY_TIME = 1.0
DEBUG_SWIPE = False  # Set True for console debug

# Finger-direction swipe (index+middle finger angle)
FINGER_SWIPE_ENABLED = True
FINGER_SWIPE_RIGHT_ANGLE = 45    # Degrees from vertical for RIGHT (pointing up-right)
FINGER_SWIPE_LEFT_ANGLE = 135    # Degrees from vertical for LEFT (pointing sideways-left)
FINGER_SWIPE_TOLERANCE = 25      # Degrees tolerance for angle detection
FINGER_SWIPE_HOLD_TIME = 0.15    # Must hold pose for this long
FINGER_SWIPE_COOLDOWN = 0.6      # Seconds between finger swipes
FINGER_SWIPE_MAX_MOVEMENT = 0.025  # Max hand movement to allow tilt (prevents conflict with movement swipes)

# Select/Pinch detection
PINCH_THRESHOLD = 0.04       # Fingers must be THIS close
PINCH_RELEASE = 0.06         # Hysteresis for release
PINCH_HOLD_TIME = 0.12       # Must hold for this long
SELECT_COOLDOWN = 0.4
SELECT_DISPLAY_TIME = 0.6

# Cursor
CURSOR_SMOOTHING = 0.4
JUMP_THRESHOLD = 0.08
USE_PALM_CENTER = True

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
# Helper Functions
# =====================
def get_palm_center(landmarks):
    """Get stable palm center position."""
    indices = [0, 5, 9, 13, 17]
    x = sum(landmarks[i].x for i in indices) / len(indices)
    y = sum(landmarks[i].y for i in indices) / len(indices)
    return x, y


def get_pinch_distance(landmarks):
    """Get distance between thumb tip and index tip."""
    thumb = landmarks[4]
    index = landmarks[8]
    return np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)


def get_pinch_point(landmarks):
    """Get midpoint between thumb and index for select cursor."""
    thumb = landmarks[4]
    index = landmarks[8]
    return (thumb.x + index.x) / 2, (thumb.y + index.y) / 2


def get_finger_angle(landmarks, tip_idx, base_idx):
    """
    Get angle of a finger from vertical (0 = pointing up).
    Returns angle in degrees (0-180, where 90 = horizontal right, 180 = down).
    """
    tip = landmarks[tip_idx]
    base = landmarks[base_idx]
    
    # Vector from base to tip
    dx = tip.x - base.x
    dy = tip.y - base.y  # Note: y increases downward in image coords
    
    # Angle from vertical (up = 0 degrees)
    # atan2 gives angle from positive x-axis, we want from negative y-axis (up)
    angle_rad = np.arctan2(dx, -dy)  # -dy because y is inverted
    angle_deg = np.degrees(angle_rad)
    
    # Normalize to 0-360
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg


def get_combined_finger_angle(landmarks):
    """
    Get average angle of index and middle fingers.
    Uses MCP (base) to tip for each finger.
    """
    # Index: tip=8, MCP=5
    # Middle: tip=12, MCP=9
    index_angle = get_finger_angle(landmarks, 8, 5)
    middle_angle = get_finger_angle(landmarks, 12, 9)
    
    # Average the angles (handle wrap-around at 360)
    if abs(index_angle - middle_angle) > 180:
        if index_angle > middle_angle:
            middle_angle += 360
        else:
            index_angle += 360
    
    avg_angle = (index_angle + middle_angle) / 2
    if avg_angle >= 360:
        avg_angle -= 360
    
    return avg_angle, index_angle, middle_angle


# =====================
# Finger Direction Swipe Detector
# =====================
class FingerSwipeDetector:
    """
    Detects swipes based on finger pointing direction.
    - Fingers pointing up-right → SWIPE RIGHT
    - Fingers pointing sideways-left → SWIPE LEFT
    Only activates when hand is relatively still (not during movement swipes).
    """
    def __init__(self):
        self.current_direction = None
        self.direction_start_time = 0
        self.last_swipe_time = 0
        self.last_result = None
        self.last_result_time = 0
        self.position_history = deque(maxlen=10)
    
    def update(self, landmarks, t: float, hand_x: float) -> str | None:
        """Check finger angle and return swipe direction if held long enough."""
        # Track hand position to detect movement
        self.position_history.append((hand_x, t))
        
        # Check if hand is moving too much (doing a movement swipe)
        if len(self.position_history) >= 3:
            recent_x = [p[0] for p in self.position_history]
            movement = abs(max(recent_x) - min(recent_x))
            if movement > FINGER_SWIPE_MAX_MOVEMENT:
                # Hand is moving - disable tilt detection
                self.current_direction = None
                return None
        
        angle, idx_angle, mid_angle = get_combined_finger_angle(landmarks)
        
        # Determine if pointing in a swipe direction
        detected_direction = None
        
        # RIGHT: pointing up-right (around 45 degrees from vertical)
        if abs(angle - FINGER_SWIPE_RIGHT_ANGLE) < FINGER_SWIPE_TOLERANCE:
            detected_direction = "RIGHT"
        # LEFT: pointing sideways-left (around 135 degrees from vertical, or -45)
        elif abs(angle - FINGER_SWIPE_LEFT_ANGLE) < FINGER_SWIPE_TOLERANCE:
            detected_direction = "LEFT"
        # Also check wrapped angle for left (around 315 degrees = -45)
        elif abs(angle - (360 - FINGER_SWIPE_LEFT_ANGLE + 90)) < FINGER_SWIPE_TOLERANCE:
            detected_direction = "LEFT"
        
        # State machine
        if detected_direction != self.current_direction:
            self.current_direction = detected_direction
            self.direction_start_time = t
            return None
        
        # Check if held long enough
        if detected_direction and t - self.direction_start_time > FINGER_SWIPE_HOLD_TIME:
            if t - self.last_swipe_time > FINGER_SWIPE_COOLDOWN:
                self.last_swipe_time = t
                self.last_result = detected_direction
                self.last_result_time = t
                self.current_direction = None  # Reset to require new gesture
                self.position_history.clear()  # Clear movement history
                return detected_direction
        
        return None
    
    def get_display(self, t: float) -> tuple:
        """Returns (text, is_active)"""
        if self.last_result and (t - self.last_result_time) < SWIPE_DISPLAY_TIME:
            if self.last_result == "RIGHT":
                return "↗ TILT RIGHT ↗", True
            else:
                return "↖ TILT LEFT ↖", True
        return None, False
    
    def get_current_angle(self, landmarks) -> float:
        """Get current finger angle for debug display."""
        angle, _, _ = get_combined_finger_angle(landmarks)
        return angle
    
    def reset(self):
        self.current_direction = None
        self.position_history.clear()


# =====================
# Swipe Detector
# =====================
class SwipeDetector:
    def __init__(self):
        self.x_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.t_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.last_swipe_time = 0
        self.last_result = None
        self.last_result_time = 0
    
    def update(self, x: float, t: float) -> str | None:
        self.x_history.append(x)
        self.t_history.append(t)
        
        if len(self.x_history) < 5:
            return None
        if t - self.last_swipe_time < SWIPE_COOLDOWN:
            return None
        
        result = self._analyze()
        if result:
            self.last_swipe_time = t
            self.last_result = result
            self.last_result_time = t
            self.x_history.clear()
            self.t_history.clear()
        
        return result
    
    def _analyze(self) -> str | None:
        x = np.array(self.x_history)
        t = np.array(self.t_history)
        
        dx = x[-1] - x[0]
        dt = t[-1] - t[0]
        
        if dt < 0.05:
            return None
        
        abs_dx = abs(dx)
        velocity = abs_dx / dt
        
        if DEBUG_SWIPE:
            print(f"  dx={dx:+.3f}, vel={velocity:.2f}")
        
        if abs_dx < SWIPE_MIN_DISTANCE:
            return None
        if velocity < SWIPE_MIN_VELOCITY:
            return None
        
        return "RIGHT" if dx > 0 else "LEFT"
    
    def get_display(self, t: float) -> tuple:
        """Returns (text, is_active)"""
        if self.last_result and (t - self.last_result_time) < SWIPE_DISPLAY_TIME:
            if self.last_result == "RIGHT":
                return "→ SWIPE RIGHT →", True
            else:
                return "← SWIPE LEFT ←", True
        return None, False
    
    def get_progress(self) -> float:
        """Get progress toward swipe threshold (0-1)."""
        if len(self.x_history) < 2:
            return 0.0
        dx = abs(self.x_history[-1] - self.x_history[0])
        return min(dx / SWIPE_MIN_DISTANCE, 1.0)
    
    def reset(self):
        self.x_history.clear()
        self.t_history.clear()


# =====================
# Select Detector (Pinch)
# =====================
class SelectDetector:
    def __init__(self):
        self.is_pinching = False
        self.pinch_start_time = 0
        self.last_select_time = 0
        self.last_select_display_time = 0
        self.confirmed = False
    
    def update(self, landmarks, t: float) -> bool:
        """Returns True if select just triggered."""
        dist = get_pinch_distance(landmarks)
        
        # Hysteresis thresholds
        threshold = PINCH_RELEASE if self.is_pinching else PINCH_THRESHOLD
        
        if dist < threshold:
            if not self.is_pinching:
                self.pinch_start_time = t
                self.is_pinching = True
                self.confirmed = False
            elif not self.confirmed and t - self.pinch_start_time > PINCH_HOLD_TIME:
                if t - self.last_select_time > SELECT_COOLDOWN:
                    self.confirmed = True
                    self.last_select_time = t
                    self.last_select_display_time = t
                    return True
        else:
            self.is_pinching = False
            self.confirmed = False
        
        return False
    
    def get_display(self, t: float) -> tuple:
        """Returns (text, is_active)"""
        if t - self.last_select_display_time < SELECT_DISPLAY_TIME:
            return "● SELECT ●", True
        return None, False
    
    def get_pinch_progress(self, landmarks) -> float:
        """Get how close fingers are to pinching (0-1)."""
        dist = get_pinch_distance(landmarks)
        if dist > PINCH_RELEASE:
            return 0.0
        return 1.0 - (dist / PINCH_RELEASE)
    
    def reset(self):
        self.is_pinching = False
        self.confirmed = False


# =====================
# Stable Cursor Tracker
# =====================
class CursorTracker:
    def __init__(self):
        self.position = None
    
    def update(self, raw_x: float, raw_y: float) -> tuple:
        raw = np.array([raw_x, raw_y])
        
        if self.position is None:
            self.position = raw
            return raw_x, raw_y
        
        delta = raw - self.position
        distance = np.linalg.norm(delta)
        
        # Reject jumps
        if distance > JUMP_THRESHOLD:
            self.position = self.position + delta * 0.15
        else:
            self.position = CURSOR_SMOOTHING * raw + (1 - CURSOR_SMOOTHING) * self.position
        
        return self.position[0], self.position[1]
    
    def reset(self):
        self.position = None


# =====================
# Main Application
# =====================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow("Kingdom Manager", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Kingdom Manager", 1280, 720)
    
    swipe = SwipeDetector()
    select = SelectDetector()
    cursor = CursorTracker()
    finger_swipe = FingerSwipeDetector()
    
    # Raw position history for swipe (no smoothing)
    raw_x_history = deque(maxlen=SWIPE_HISTORY_SIZE)
    raw_t_history = deque(maxlen=SWIPE_HISTORY_SIZE)
    
    print("Kingdom Manager Hand Control")
    print("- Swipe left/right to dismiss/accept (movement)")
    print("- Tilt fingers up-right or sideways-left (casual swipe)")
    print("- Pinch thumb+index to select")
    print("- Press Q to quit")
    
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
        gesture_text = "Ready"
        gesture_active = False
        
        if dominant:
            # Get tracking position
            if USE_PALM_CENTER:
                raw_x, raw_y = get_palm_center(dominant)
            else:
                raw_x, raw_y = dominant[8].x, dominant[8].y
            
            # Track raw positions for swipe
            raw_x_history.append(raw_x)
            raw_t_history.append(now)
            
            # Smooth cursor for display
            stable_x, stable_y = cursor.update(raw_x, raw_y)
            cx, cy = int(stable_x * w), int(stable_y * h)
            
            # === SELECT DETECTION ===
            select_triggered = select.update(dominant, now)
            if select_triggered:
                print("=== SELECT ===")
            
            # === FINGER DIRECTION SWIPE ===
            finger_swipe_result = None
            if FINGER_SWIPE_ENABLED and not select.is_pinching:
                finger_swipe_result = finger_swipe.update(dominant, now, raw_x)
                if finger_swipe_result:
                    print(f"=== TILT {finger_swipe_result}! ===")
            
            # === MOVEMENT SWIPE DETECTION ===
            # Only detect swipe if NOT actively pinching
            swipe_result = None
            if not select.is_pinching:
                # Use raw history for swipe detection
                if len(raw_x_history) >= 5 and now - swipe.last_swipe_time > SWIPE_COOLDOWN:
                    swipe.x_history = deque(raw_x_history, maxlen=SWIPE_HISTORY_SIZE)
                    swipe.t_history = deque(raw_t_history, maxlen=SWIPE_HISTORY_SIZE)
                    swipe_result = swipe._analyze()
                    
                    if swipe_result:
                        swipe.last_swipe_time = now
                        swipe.last_result = swipe_result
                        swipe.last_result_time = now
                        print(f"=== SWIPE {swipe_result}! ===")
                        raw_x_history.clear()
                        raw_t_history.clear()
            
            # === CURSOR DRAWING ===
            if select.is_pinching:
                # Pinching cursor
                px, py = get_pinch_point(dominant)
                pcx, pcy = int(px * w), int(py * h)
                cv2.circle(frame, (pcx, pcy), 18, (0, 255, 255), 3)
                cv2.circle(frame, (pcx, pcy), 8, (0, 255, 255), -1)
                
                if select.confirmed:
                    cv2.putText(frame, "SELECTED!", (pcx + 25, pcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # Normal cursor
                cv2.circle(frame, (cx, cy), 12, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            
            # === GESTURE DISPLAY ===
            swipe_display, swipe_active = swipe.get_display(now)
            select_display, select_active = select.get_display(now)
            finger_swipe_display, finger_swipe_active = finger_swipe.get_display(now)
            
            if select_active:
                gesture_text = select_display
                gesture_active = True
            elif finger_swipe_active:
                gesture_text = finger_swipe_display
                gesture_active = True
            elif swipe_active:
                gesture_text = swipe_display
                gesture_active = True
        else:
            cursor.reset()
            swipe.reset()
            select.reset()
            finger_swipe.reset()
            raw_x_history.clear()
            raw_t_history.clear()
        
        # === UI ===
        cv2.rectangle(frame, (0, 0), (w, 80), (20, 20, 20), -1)
        
        # Gesture text
        text_color = (0, 255, 255) if gesture_active else (180, 180, 180)
        cv2.putText(frame, gesture_text, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 2)
        
        # Hand count
        hand_text = f"Hands: {num_hands}"
        if num_hands == 2:
            hand_text += " (TWO HANDS)"
        cv2.putText(frame, hand_text, (w - 280, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Dominant hand indicator
        dom_text = "Right Hand" if RIGHT_HAND_DOMINANT else "Left Hand"
        cv2.putText(frame, dom_text, (w - 280, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Progress bars at bottom
        if dominant:
            # Swipe progress
            swipe_prog = 0.0
            if len(raw_x_history) > 1:
                dx = abs(raw_x_history[-1] - raw_x_history[0])
                swipe_prog = min(dx / SWIPE_MIN_DISTANCE, 1.0)
            
            bar_y = h - 40
            cv2.putText(frame, "Swipe:", (20, bar_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            cv2.rectangle(frame, (80, bar_y - 8), (230, bar_y + 8), (50, 50, 50), -1)
            bar_w = int(150 * swipe_prog)
            bar_color = (0, 255, 128) if swipe_prog >= 1.0 else (80, 80, 200)
            if bar_w > 0:
                cv2.rectangle(frame, (80, bar_y - 8), (80 + bar_w, bar_y + 8), bar_color, -1)
            
            # Pinch progress
            pinch_prog = select.get_pinch_progress(dominant)
            cv2.putText(frame, "Select:", (260, bar_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            cv2.rectangle(frame, (330, bar_y - 8), (480, bar_y + 8), (50, 50, 50), -1)
            pinch_w = int(150 * pinch_prog)
            pinch_color = (0, 255, 255) if pinch_prog >= 0.8 else (80, 150, 180)
            if pinch_w > 0:
                cv2.rectangle(frame, (330, bar_y - 8), (330 + pinch_w, bar_y + 8), pinch_color, -1)
            
            # Finger angle indicator
            if FINGER_SWIPE_ENABLED:
                angle = finger_swipe.get_current_angle(dominant)
                # Draw angle arc indicator
                arc_x = w - 100
                arc_y = h - 80
                arc_r = 40
                
                # Draw reference arcs for left/right zones
                cv2.ellipse(frame, (arc_x, arc_y), (arc_r, arc_r), 0, 
                           -FINGER_SWIPE_RIGHT_ANGLE - FINGER_SWIPE_TOLERANCE,
                           -FINGER_SWIPE_RIGHT_ANGLE + FINGER_SWIPE_TOLERANCE, 
                           (0, 100, 0), 2)
                cv2.ellipse(frame, (arc_x, arc_y), (arc_r, arc_r), 0,
                           -FINGER_SWIPE_LEFT_ANGLE - FINGER_SWIPE_TOLERANCE,
                           -FINGER_SWIPE_LEFT_ANGLE + FINGER_SWIPE_TOLERANCE,
                           (0, 0, 100), 2)
                
                # Draw current angle line
                angle_rad = np.radians(-angle + 90)  # Convert to drawing coords
                end_x = int(arc_x + arc_r * np.cos(angle_rad))
                end_y = int(arc_y - arc_r * np.sin(angle_rad))
                
                # Color based on zone
                line_color = (150, 150, 150)
                if abs(angle - FINGER_SWIPE_RIGHT_ANGLE) < FINGER_SWIPE_TOLERANCE:
                    line_color = (0, 255, 0)
                elif abs(angle - FINGER_SWIPE_LEFT_ANGLE) < FINGER_SWIPE_TOLERANCE:
                    line_color = (255, 100, 100)
                
                cv2.line(frame, (arc_x, arc_y), (end_x, end_y), line_color, 3)
                cv2.putText(frame, f"{int(angle)}°", (arc_x - 15, arc_y + arc_r + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                cv2.putText(frame, "R", (arc_x + 35, arc_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
                cv2.putText(frame, "L", (arc_x - 50, arc_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 0, 0), 1)
        
        cv2.imshow("Kingdom Manager", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
