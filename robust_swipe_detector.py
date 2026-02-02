"""
Robust Swipe Detector
=====================
Combines the optimized hand tracking with swipe left/right detection.
Uses all 5 tracking fixes for stable, reliable swipe gestures.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
from collections import deque

# =====================
# CONFIG
# =====================
CAMERA_INDEX = 0

# Tracking config
HAND_TIMEOUT = 0.3          # seconds to hold last position
VELOCITY_SMOOTHING = 0.3    # blend factor for velocity updates
MAX_PREDICTION_FRAMES = 5   # max frames to predict through

# Swipe detection config
SWIPE_HISTORY_SIZE = 30     # frames of history
SWIPE_MIN_DISTANCE = 0.12   # minimum palm travel distance (normalized)
SWIPE_MIN_VELOCITY = 0.4    # minimum speed (units/sec)
SWIPE_COOLDOWN = 0.5        # seconds between swipes
SWIPE_DISPLAY_TIME = 0.8    # how long to show swipe feedback

# Pinch detection config
PINCH_THRESHOLD = 0.045     # distance to activate pinch
PINCH_RELEASE = 0.065       # distance to release pinch (hysteresis)
PINCH_HOLD_TIME = 0.1       # must hold pinch for this long to activate
SELECT_COOLDOWN = 0.3       # seconds between selections
SELECT_DISPLAY_TIME = 0.5   # how long to show selection feedback

# Model setup
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Display
WINDOW_NAME = "Robust Swipe Detector"


class RobustHandTracker:
    def __init__(self):
        # Download model if needed
        if not os.path.exists(MODEL_PATH):
            print("Downloading hand landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Done!")
        
        # Fix #1: Proper tracking mode config using Tasks API
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.7
            )
        )
        
        # Fix #2: Temporal persistence state
        self.last_landmarks = None
        self.last_seen_time = 0
        
        # Fix #3: Velocity prediction state
        self.prev_palm_x = None
        self.prev_palm_y = None
        self.prev_time = None
        self.velocity_x = 0
        self.velocity_y = 0
        self.frames_since_detection = 0
        
        # Fix #4: Palm position (computed from wrist + MCPs)
        self.palm_x = 0.5
        self.palm_y = 0.5
        
        # Fix #5: Handedness locking
        self.locked_hand_label = None
        self.first_detection = True
    
    def get_palm_center(self, landmarks):
        """Use wrist (0), index MCP (5), and pinky MCP (17)."""
        palm_x = (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3
        palm_y = (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
        return palm_x, palm_y
    
    def update_velocity(self, palm_x, palm_y, current_time):
        if self.prev_palm_x is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                raw_vx = (palm_x - self.prev_palm_x) / dt
                raw_vy = (palm_y - self.prev_palm_y) / dt
                self.velocity_x = (VELOCITY_SMOOTHING * raw_vx + 
                                   (1 - VELOCITY_SMOOTHING) * self.velocity_x)
                self.velocity_y = (VELOCITY_SMOOTHING * raw_vy + 
                                   (1 - VELOCITY_SMOOTHING) * self.velocity_y)
        
        self.prev_palm_x = palm_x
        self.prev_palm_y = palm_y
        self.prev_time = current_time
    
    def predict_position(self, dt):
        predicted_x = self.palm_x + self.velocity_x * dt
        predicted_y = self.palm_y + self.velocity_y * dt
        predicted_x = max(0, min(1, predicted_x))
        predicted_y = max(0, min(1, predicted_y))
        return predicted_x, predicted_y
    
    def process_frame(self, frame, timestamp_ms):
        """
        Returns: (palm_x, palm_y, landmarks, status, hand_label)
        - status: 'detected', 'persisted', 'predicted', 'lost'
        """
        current_time = time.time()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if results.hand_landmarks:
            landmarks = results.hand_landmarks[0]
            
            # Get handedness - invert because we flipped the frame before processing
            # MediaPipe sees flipped image, so Left→Right and Right→Left for the user
            if results.handedness:
                mp_label = results.handedness[0][0].category_name
                # Invert the label since frame was flipped before detection
                detected_label = "Right" if mp_label == "Left" else "Left"
                self.locked_hand_label = detected_label  # Always update, don't lock
            
            # Fix #4: Palm center
            palm_x, palm_y = self.get_palm_center(landmarks)
            
            # Fix #3: Update velocity
            self.update_velocity(palm_x, palm_y, current_time)
            
            self.palm_x = palm_x
            self.palm_y = palm_y
            self.last_landmarks = landmarks
            self.last_seen_time = current_time
            self.frames_since_detection = 0
            
            return palm_x, palm_y, landmarks, 'detected', self.locked_hand_label
        
        else:
            time_since_last = current_time - self.last_seen_time
            self.frames_since_detection += 1
            
            # Fix #2: Temporal persistence
            if time_since_last < HAND_TIMEOUT and self.last_landmarks is not None:
                # Fix #3: Predict position
                if self.frames_since_detection <= MAX_PREDICTION_FRAMES:
                    pred_x, pred_y = self.predict_position(time_since_last)
                    return pred_x, pred_y, self.last_landmarks, 'predicted', self.locked_hand_label
                else:
                    return self.palm_x, self.palm_y, self.last_landmarks, 'persisted', self.locked_hand_label
            
            return None, None, None, 'lost', None
    
    def reset_handedness(self):
        self.first_detection = True
        self.locked_hand_label = None


class PinchDetector:
    """
    Detects pinch gestures (thumb to index or middle finger).
    Returns activation point at the fingertip.
    """
    
    def __init__(self):
        self.is_pinching = False
        self.pinch_start_time = 0
        self.last_select_time = 0
        self.last_select_result = None
        self.last_select_result_time = 0
        self.active_finger = None  # 'index' or 'middle'
    
    def get_pinch_distance(self, landmarks, finger='index'):
        """Get distance between thumb tip and finger tip."""
        thumb = landmarks[4]  # Thumb tip
        if finger == 'index':
            tip = landmarks[8]   # Index tip
        else:
            tip = landmarks[12]  # Middle tip
        
        dx = thumb.x - tip.x
        dy = thumb.y - tip.y
        return np.sqrt(dx*dx + dy*dy)
    
    def get_pinch_point(self, landmarks, finger='index'):
        """Get midpoint between thumb and finger tip (activation point)."""
        thumb = landmarks[4]
        if finger == 'index':
            tip = landmarks[8]
        else:
            tip = landmarks[12]
        
        return (thumb.x + tip.x) / 2, (thumb.y + tip.y) / 2
    
    def get_fingertip(self, landmarks, finger='index'):
        """Get fingertip position."""
        if finger == 'index':
            tip = landmarks[8]
        else:
            tip = landmarks[12]
        return tip.x, tip.y
    
    def update(self, landmarks, t: float) -> tuple:
        """
        Check for pinch gesture.
        Returns: (is_selecting, select_x, select_y, finger_type, just_selected)
        - is_selecting: True if currently pinching
        - select_x, select_y: activation point (fingertip position)
        - finger_type: 'index' or 'middle'
        - just_selected: True on the frame selection was triggered
        """
        if landmarks is None:
            self.is_pinching = False
            self.active_finger = None
            return False, None, None, None, False
        
        # Check both fingers for pinch
        index_dist = self.get_pinch_distance(landmarks, 'index')
        middle_dist = self.get_pinch_distance(landmarks, 'middle')
        
        # Determine which finger is pinching (prefer the closer one)
        min_dist = min(index_dist, middle_dist)
        closer_finger = 'index' if index_dist <= middle_dist else 'middle'
        
        just_selected = False
        
        # State machine with hysteresis
        if not self.is_pinching:
            # Check for pinch start
            if min_dist < PINCH_THRESHOLD:
                if self.pinch_start_time == 0:
                    self.pinch_start_time = t
                    self.active_finger = closer_finger
                elif t - self.pinch_start_time > PINCH_HOLD_TIME:
                    # Pinch confirmed after hold time
                    self.is_pinching = True
                    if t - self.last_select_time > SELECT_COOLDOWN:
                        just_selected = True
                        self.last_select_time = t
                        self.last_select_result = self.active_finger
                        self.last_select_result_time = t
            else:
                self.pinch_start_time = 0
                self.active_finger = None
        else:
            # Check for pinch release (use higher threshold for hysteresis)
            if min_dist > PINCH_RELEASE:
                self.is_pinching = False
                self.pinch_start_time = 0
                self.active_finger = None
            else:
                # Update active finger if it changed
                self.active_finger = closer_finger
        
        if self.is_pinching and self.active_finger:
            tip_x, tip_y = self.get_fingertip(landmarks, self.active_finger)
            return True, tip_x, tip_y, self.active_finger, just_selected
        
        return False, None, None, None, False
    
    def get_display(self, t: float) -> tuple:
        """Returns (text, is_active)"""
        if self.last_select_result and (t - self.last_select_result_time) < SELECT_DISPLAY_TIME:
            finger_name = self.last_select_result.upper()
            return f"✓ SELECT ({finger_name})", True
        return None, False
    
    def reset(self):
        self.is_pinching = False
        self.pinch_start_time = 0
        self.active_finger = None


class SwipeDetector:
    """
    Detects horizontal swipe gestures based on palm movement.
    """
    
    def __init__(self):
        self.x_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.t_history = deque(maxlen=SWIPE_HISTORY_SIZE)
        self.last_swipe_time = 0
        self.last_result = None
        self.last_result_time = 0
    
    def update(self, x: float, t: float) -> str | None:
        """
        Feed in palm x-position and timestamp.
        Returns 'LEFT', 'RIGHT', or None.
        """
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
        
        if abs_dx < SWIPE_MIN_DISTANCE:
            return None
        if velocity < SWIPE_MIN_VELOCITY:
            return None
        
        return "RIGHT" if dx > 0 else "LEFT"
    
    def get_display(self, t: float) -> tuple:
        """Returns (text, color, is_active)"""
        if self.last_result and (t - self.last_result_time) < SWIPE_DISPLAY_TIME:
            if self.last_result == "RIGHT":
                return ">>> SWIPE RIGHT >>>", (0, 255, 0), True
            else:
                return "<<< SWIPE LEFT <<<", (0, 255, 255), True
        return None, None, False
    
    def get_progress(self) -> float:
        """Get progress toward swipe threshold (0-1)."""
        if len(self.x_history) < 2:
            return 0.0
        dx = abs(self.x_history[-1] - self.x_history[0])
        return min(dx / SWIPE_MIN_DISTANCE, 1.0)
    
    def get_direction_hint(self) -> str | None:
        """Get current movement direction for visual hint."""
        if len(self.x_history) < 3:
            return None
        dx = self.x_history[-1] - self.x_history[0]
        if abs(dx) > 0.02:
            return "RIGHT" if dx > 0 else "LEFT"
        return None
    
    def reset(self):
        self.x_history.clear()
        self.t_history.clear()


def draw_hand_skeleton(frame, landmarks, color=(0, 255, 0)):
    """Draw hand landmarks and connections."""
    h, w = frame.shape[:2]
    
    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    for start, end in CONNECTIONS:
        pt1 = landmarks[start]
        pt2 = landmarks[end]
        x1, y1 = int(pt1.x * w), int(pt1.y * h)
        x2, y2 = int(pt2.x * w), int(pt2.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, color, -1)
    
    # Highlight palm landmarks
    for idx in [0, 5, 17]:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 8, (255, 255, 0), 2)


def draw_swipe_indicator(frame, progress, direction_hint):
    """Draw swipe progress bar and direction arrows."""
    h, w = frame.shape[:2]
    
    # Progress bar at bottom
    bar_height = 20
    bar_y = h - 40
    bar_width = int(w * 0.6)
    bar_x = (w - bar_width) // 2
    
    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (50, 50, 50), -1)
    
    # Progress fill
    if progress > 0:
        fill_width = int(bar_width * progress)
        color = (0, 255, 0) if progress >= 1.0 else (0, 200, 200)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                      color, -1)
    
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (200, 200, 200), 2)
    
    # Direction arrows
    if direction_hint == "RIGHT":
        cv2.arrowedLine(frame, (w//2, bar_y - 30), (w//2 + 80, bar_y - 30), 
                        (0, 255, 255), 3, tipLength=0.4)
    elif direction_hint == "LEFT":
        cv2.arrowedLine(frame, (w//2, bar_y - 30), (w//2 - 80, bar_y - 30), 
                        (0, 255, 255), 3, tipLength=0.4)


def draw_swipe_result(frame, text, color):
    """Draw large swipe result text."""
    h, w = frame.shape[:2]
    
    # Calculate text size for centering
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (w - text_w) // 2
    y = h // 2
    
    # Background box
    padding = 20
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), 
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (x - padding, y - text_h - padding), 
                  (x + text_w + padding, y + baseline + padding), 
                  color, 3)
    
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def main():
    print("=" * 50)
    print("ROBUST SWIPE DETECTOR")
    print("=" * 50)
    print("\nSwipe your hand LEFT or RIGHT to trigger actions!")
    print(f"  - Min distance: {SWIPE_MIN_DISTANCE}")
    print(f"  - Min velocity: {SWIPE_MIN_VELOCITY}")
    print(f"  - Cooldown: {SWIPE_COOLDOWN}s")
    print("\nControls:")
    print("  R - Reset handedness lock")
    print("  Q - Quit")
    print("=" * 50)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    tracker = RobustHandTracker()
    swipe_detector = SwipeDetector()
    pinch_detector = PinchDetector()
    
    # Swipe counters
    left_count = 0
    right_count = 0
    select_count = 0
    
    # Frame tracking
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        timestamp_ms = int(frame_count * (1000 / 30))
        frame_count += 1
        current_time = time.time()
        
        # Track hand
        palm_x, palm_y, landmarks, status, hand_label = tracker.process_frame(frame, timestamp_ms)
        
        # Status colors
        STATUS_COLORS = {
            'detected': (0, 255, 0),
            'predicted': (0, 255, 255),
            'persisted': (0, 165, 255),
            'lost': (0, 0, 255)
        }
        color = STATUS_COLORS.get(status, (128, 128, 128))
        
        # Draw hand skeleton
        if landmarks is not None:
            draw_hand_skeleton(frame, landmarks, color)
        
        # Update pinch detection
        is_pinching, pinch_x, pinch_y, pinch_finger, just_selected = pinch_detector.update(landmarks, current_time)
        
        if just_selected:
            select_count += 1
            print(f"SELECT ({pinch_finger.upper()})! (Total: {select_count})")
        
        # Draw pinch activation point
        if is_pinching and pinch_x is not None:
            px, py = int(pinch_x * w), int(pinch_y * h)
            # Pulsing selection cursor
            pulse = int(5 * np.sin(current_time * 10) + 20)
            cv2.circle(frame, (px, py), pulse, (255, 0, 255), 3)
            cv2.circle(frame, (px, py), 8, (255, 255, 255), -1)
            cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
            
            # Draw line from thumb to finger
            thumb = landmarks[4]
            finger_idx = 8 if pinch_finger == 'index' else 12
            finger = landmarks[finger_idx]
            cv2.line(frame, (int(thumb.x * w), int(thumb.y * h)), 
                     (int(finger.x * w), int(finger.y * h)), (255, 0, 255), 2)
        
        # Update swipe detection
        swipe_result = None
        if palm_x is not None:
            swipe_result = swipe_detector.update(palm_x, current_time)
            
            if swipe_result:
                if swipe_result == "LEFT":
                    left_count += 1
                    print(f"SWIPE LEFT! (Total: {left_count})")
                else:
                    right_count += 1
                    print(f"SWIPE RIGHT! (Total: {right_count})")
            
            # Draw palm cursor (smaller if pinching)
            px, py = int(palm_x * w), int(palm_y * h)
            cursor_size = 10 if is_pinching else 15
            cv2.circle(frame, (px, py), cursor_size, color, 3)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), -1)
        else:
            swipe_detector.reset()
            pinch_detector.reset()
        
        # Draw swipe progress indicator
        progress = swipe_detector.get_progress()
        direction_hint = swipe_detector.get_direction_hint()
        draw_swipe_indicator(frame, progress, direction_hint)
        
        # Draw swipe result if active
        display_text, display_color, is_active = swipe_detector.get_display(current_time)
        if is_active:
            draw_swipe_result(frame, display_text, display_color)
        
        # Draw pinch/select result if active
        pinch_display, pinch_active = pinch_detector.get_display(current_time)
        if pinch_active and not is_active:  # Don't overlap with swipe display
            draw_swipe_result(frame, pinch_display, (255, 0, 255))
        
        # Status panel
        cv2.rectangle(frame, (10, 10), (200, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (200, 140), color, 2)
        
        cv2.putText(frame, f"Status: {status}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"Hand: {hand_label or 'None'}", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Left swipes: {left_count}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Right swipes: {right_count}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Selections: {select_count}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Pinch status indicator
        if is_pinching:
            cv2.putText(frame, f"PINCH: {pinch_finger.upper()}", (w - 150, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Controls hint
        cv2.putText(frame, "R=Reset  Q=Quit", (w - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(WINDOW_NAME, frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            tracker.reset_handedness()
            swipe_detector.reset()
            pinch_detector.reset()
            print("Reset handedness and swipe state")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Left swipes:  {left_count}")
    print(f"Right swipes: {right_count}")
    print(f"Selections:   {select_count}")
    print(f"Total actions: {left_count + right_count + select_count}")


if __name__ == "__main__":
    main()
