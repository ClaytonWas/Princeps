"""
Princeps Gesture Engine
=======================
Core gesture detection module for hand-controlled games.

Usage:
    from gesture_engine import GestureEngine, GestureState

    engine = GestureEngine()
    
    while engine.running:
        state = engine.update()
        
        if state.swipe_left:
            # Handle swipe left
        if state.pinch_started:
            # Handle pinch start
        
        # Draw your game using state.cursor_x, state.cursor_y
        engine.render_pip(your_game_frame)
        engine.show(your_game_frame)
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
from enum import Enum, auto


# =====================
# CONFIG
# =====================
class GestureConfig:
    """Configuration for gesture detection. Modify these to tune sensitivity."""
    
    # Camera
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Tracking - MUCH more forgiving
    HAND_TIMEOUT = 0.8           # Hold position longer when hand lost (was 0.3)
    VELOCITY_SMOOTHING = 0.15    # Smoother velocity (was 0.3, lower = smoother)
    MAX_PREDICTION_FRAMES = 15   # More prediction frames (was 5)
    POSITION_SMOOTHING = 0.4     # NEW: Smooth position changes (0 = no smoothing, 1 = frozen)
    
    # Detection confidence - LOWER = more detections
    MIN_DETECTION_CONFIDENCE = 0.3   # NEW: Lower = detect hands more easily (was 0.5)
    MIN_TRACKING_CONFIDENCE = 0.3    # NEW: Lower = track through occlusion better (was 0.7)
    
    # Jitter filtering
    JITTER_THRESHOLD = 0.008    # NEW: Ignore movements smaller than this
    
    # Swipe detection - MUCH easier to trigger
    SWIPE_HISTORY_SIZE = 20     # Shorter window (was 30)
    SWIPE_MIN_DISTANCE = 0.06   # Half the distance! (was 0.12)
    SWIPE_MIN_VELOCITY = 0.2    # Half the speed! (was 0.4)
    SWIPE_COOLDOWN = 0.4        # Slightly faster repeat (was 0.5)
    SWIPE_DIRECTIONAL_RATIO = 2.0  # NEW: X movement must be 2x Y movement
    
    # Pinch detection - more forgiving
    PINCH_THRESHOLD = 0.055     # Slightly larger (was 0.045)
    PINCH_RELEASE = 0.075       # More hysteresis (was 0.065)
    PINCH_HOLD_TIME = 0.04      # Faster response (was 0.05)
    PINCH_SMOOTHING = 0.3       # NEW: Smooth pinch distance readings
    
    # Model
    MODEL_PATH = "hand_landmarker.task"
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


# =====================
# GESTURE STATE
# =====================
@dataclass
class GestureState:
    """
    Immutable snapshot of the current gesture state.
    This is what your game loop receives each frame.
    """
    # Time
    timestamp: float = 0.0
    delta_time: float = 0.0
    frame_count: int = 0
    
    # Hand tracking
    hand_detected: bool = False
    hand_status: str = 'lost'  # 'detected', 'predicted', 'persisted', 'lost'
    hand_label: Optional[str] = None  # 'Left' or 'Right'
    
    # Cursor position (normalized 0-1)
    cursor_x: Optional[float] = None
    cursor_y: Optional[float] = None
    
    # Velocity (units per second, normalized)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    # Pinch state
    is_pinching: bool = False
    pinch_started: bool = False  # True only on the frame pinch began
    pinch_ended: bool = False    # True only on the frame pinch ended
    pinch_finger: Optional[str] = None  # 'index' or 'middle'
    
    # Swipe events (True only on the frame swipe was detected)
    swipe_left: bool = False
    swipe_right: bool = False
    swipe_direction: Optional[str] = None  # 'LEFT', 'RIGHT', or None
    
    # Raw landmarks (for advanced use)
    landmarks: Optional[Any] = None
    
    def cursor_pixel(self, width: int, height: int) -> tuple:
        """Convert normalized cursor to pixel coordinates."""
        if self.cursor_x is None:
            return None, None
        return int(self.cursor_x * width), int(self.cursor_y * height)


# =====================
# INTERNAL COMPONENTS
# =====================
class _OneEuroFilter:
    """
    One Euro Filter for super smooth cursor movement.
    Adapts smoothing based on speed - slow = smooth, fast = responsive.
    """
    
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
    
    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)
    
    def _exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev
    
    def filter(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        
        t_e = t - self.t_prev
        if t_e <= 0:
            return self.x_prev
        
        # Derivative
        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self._exponential_smoothing(a_d, dx, self.dx_prev)
        
        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        
        # Filter
        a = self._smoothing_factor(t_e, cutoff)
        x_hat = self._exponential_smoothing(a, x, self.x_prev)
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class _HandTracker:
    """Internal hand tracking component with heavy smoothing and prediction."""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        
        if not os.path.exists(config.MODEL_PATH):
            print("Downloading hand landmarker model...")
            urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
            print("Done!")
        
        # Use configurable confidence thresholds
        self.detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=config.MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
            )
        )
        
        self.last_landmarks = None
        self.last_seen_time = 0
        self.prev_time = None
        self.velocity_x = 0
        self.velocity_y = 0
        self.frames_since_detection = 0
        self.palm_x = 0.5
        self.palm_y = 0.5
        self.smoothed_x = 0.5
        self.smoothed_y = 0.5
        self.hand_label = None
        
        # One Euro Filters for super smooth tracking
        self.filter_x = _OneEuroFilter(min_cutoff=1.5, beta=0.01)
        self.filter_y = _OneEuroFilter(min_cutoff=1.5, beta=0.01)
        
        # Landmark history for temporal smoothing
        self.landmark_history = deque(maxlen=5)
        
        # Consecutive detection counter
        self.consecutive_detections = 0
        self.consecutive_losses = 0
        
        # RAW palm position (unsmoothed - for swipe detection)
        self.raw_palm_x = 0.5
        self.raw_palm_y = 0.5
    
    def get_palm_center(self, landmarks):
        """Get palm center from multiple landmarks for stability."""
        # Use more points for better stability
        # Wrist (0), Index MCP (5), Middle MCP (9), Ring MCP (13), Pinky MCP (17)
        points = [0, 5, 9, 13, 17]
        palm_x = sum(landmarks[i].x for i in points) / len(points)
        palm_y = sum(landmarks[i].y for i in points) / len(points)
        return palm_x, palm_y
    
    def _smooth_landmarks(self, landmarks):
        """Apply temporal smoothing to landmarks using history."""
        if len(self.landmark_history) == 0:
            return landmarks
        
        # Weight recent frames more heavily
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(self.landmark_history):]
        total_weight = sum(weights) + 0.4  # Current frame gets 0.4
        
        smoothed = []
        for i in range(len(landmarks)):
            x = landmarks[i].x * 0.4
            y = landmarks[i].y * 0.4
            z = landmarks[i].z * 0.4
            
            for j, hist_landmarks in enumerate(self.landmark_history):
                w = weights[j] if j < len(weights) else 0.1
                x += hist_landmarks[i].x * w
                y += hist_landmarks[i].y * w
                z += hist_landmarks[i].z * w
            
            # Create a simple object to hold smoothed values
            class SmoothLandmark:
                pass
            sl = SmoothLandmark()
            sl.x = x / total_weight
            sl.y = y / total_weight
            sl.z = z / total_weight
            smoothed.append(sl)
        
        return smoothed
    
    def update(self, frame, timestamp_ms) -> tuple:
        current_time = time.time()
        config = self.config
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if results.hand_landmarks:
            raw_landmarks = results.hand_landmarks[0]
            
            # Add to history and smooth
            self.landmark_history.append(raw_landmarks)
            landmarks = self._smooth_landmarks(raw_landmarks)
            
            if results.handedness:
                mp_label = results.handedness[0][0].category_name
                self.hand_label = "Right" if mp_label == "Left" else "Left"
            
            raw_palm_x, raw_palm_y = self.get_palm_center(landmarks)
            
            # Store RAW position for swipe detection (before any filtering)
            self.raw_palm_x = raw_palm_x
            self.raw_palm_y = raw_palm_y
            
            # Apply One Euro Filter for butter-smooth movement
            palm_x = self.filter_x.filter(raw_palm_x, current_time)
            palm_y = self.filter_y.filter(raw_palm_y, current_time)
            
            # Apply jitter threshold - ignore tiny movements
            if self.prev_time is not None:
                dx = abs(palm_x - self.smoothed_x)
                dy = abs(palm_y - self.smoothed_y)
                if dx < config.JITTER_THRESHOLD and dy < config.JITTER_THRESHOLD:
                    palm_x = self.smoothed_x
                    palm_y = self.smoothed_y
            
            # Update velocity with heavy smoothing
            if self.prev_time is not None:
                dt = current_time - self.prev_time
                if dt > 0:
                    raw_vx = (palm_x - self.smoothed_x) / dt
                    raw_vy = (palm_y - self.smoothed_y) / dt
                    self.velocity_x = (config.VELOCITY_SMOOTHING * raw_vx + 
                                       (1 - config.VELOCITY_SMOOTHING) * self.velocity_x)
                    self.velocity_y = (config.VELOCITY_SMOOTHING * raw_vy + 
                                       (1 - config.VELOCITY_SMOOTHING) * self.velocity_y)
            
            self.prev_time = current_time
            
            # Additional position smoothing
            alpha = 1.0 - config.POSITION_SMOOTHING
            self.smoothed_x = alpha * palm_x + config.POSITION_SMOOTHING * self.smoothed_x
            self.smoothed_y = alpha * palm_y + config.POSITION_SMOOTHING * self.smoothed_y
            
            self.palm_x = self.smoothed_x
            self.palm_y = self.smoothed_y
            self.last_landmarks = landmarks
            self.last_seen_time = current_time
            self.frames_since_detection = 0
            self.consecutive_detections += 1
            self.consecutive_losses = 0
            
            return self.smoothed_x, self.smoothed_y, landmarks, 'detected', self.hand_label
        
        else:
            time_since_last = current_time - self.last_seen_time
            self.frames_since_detection += 1
            self.consecutive_detections = 0
            self.consecutive_losses += 1
            
            # Much more forgiving timeout
            if time_since_last < config.HAND_TIMEOUT and self.last_landmarks is not None:
                if self.frames_since_detection <= config.MAX_PREDICTION_FRAMES:
                    # Predict with velocity, but decay velocity over time
                    decay = max(0, 1.0 - (self.frames_since_detection / config.MAX_PREDICTION_FRAMES))
                    pred_x = self.smoothed_x + self.velocity_x * time_since_last * decay
                    pred_y = self.smoothed_y + self.velocity_y * time_since_last * decay
                    pred_x = max(0.02, min(0.98, pred_x))  # Keep in bounds with margin
                    pred_y = max(0.02, min(0.98, pred_y))
                    
                    self.smoothed_x = pred_x
                    self.smoothed_y = pred_y
                    
                    return pred_x, pred_y, self.last_landmarks, 'predicted', self.hand_label
                else:
                    # Just hold position
                    return self.smoothed_x, self.smoothed_y, self.last_landmarks, 'persisted', self.hand_label
            
            # Only truly lost after extended timeout
            if self.consecutive_losses > 30:  # About 1 second
                self.filter_x.reset()
                self.filter_y.reset()
                self.landmark_history.clear()
            
            return None, None, None, 'lost', None


class _PinchDetector:
    """Internal pinch detection component with smoothing."""
    
    def __init__(self, config: GestureConfig):
        self.config = config
        self.is_pinching = False
        self.pinch_start_time = 0
        self.active_finger = None
        
        # Smoothed pinch distances
        self.smoothed_index_dist = 1.0
        self.smoothed_middle_dist = 1.0
    
    def get_pinch_distance(self, landmarks, finger='index'):
        thumb = landmarks[4]
        tip = landmarks[8] if finger == 'index' else landmarks[12]
        dx = thumb.x - tip.x
        dy = thumb.y - tip.y
        return np.sqrt(dx*dx + dy*dy)
    
    def update(self, landmarks, t: float) -> tuple:
        """Returns: (is_pinching, just_started, just_ended, finger)"""
        config = self.config
        
        if landmarks is None:
            was_pinching = self.is_pinching
            self.is_pinching = False
            self.active_finger = None
            # Don't reset smoothed distances immediately
            return False, False, was_pinching, None
        
        # Get raw distances
        raw_index = self.get_pinch_distance(landmarks, 'index')
        raw_middle = self.get_pinch_distance(landmarks, 'middle')
        
        # Smooth the distances to reduce jitter
        smooth = config.PINCH_SMOOTHING
        self.smoothed_index_dist = smooth * self.smoothed_index_dist + (1 - smooth) * raw_index
        self.smoothed_middle_dist = smooth * self.smoothed_middle_dist + (1 - smooth) * raw_middle
        
        index_dist = self.smoothed_index_dist
        middle_dist = self.smoothed_middle_dist
        
        min_dist = min(index_dist, middle_dist)
        closer_finger = 'index' if index_dist <= middle_dist else 'middle'
        
        was_pinching = self.is_pinching
        just_started = False
        just_ended = False
        
        if not self.is_pinching:
            if min_dist < config.PINCH_THRESHOLD:
                if self.pinch_start_time == 0:
                    self.pinch_start_time = t
                    self.active_finger = closer_finger
                elif t - self.pinch_start_time > config.PINCH_HOLD_TIME:
                    self.is_pinching = True
                    just_started = True
            else:
                self.pinch_start_time = 0
                self.active_finger = None
        else:
            if min_dist > config.PINCH_RELEASE:
                self.is_pinching = False
                self.pinch_start_time = 0
                self.active_finger = None
                just_ended = True
            else:
                self.active_finger = closer_finger
        
        return self.is_pinching, just_started, just_ended, self.active_finger
    
    def reset(self):
        self.is_pinching = False
        self.pinch_start_time = 0
        self.active_finger = None
        # Reset smoothed distances more gently
        self.smoothed_index_dist = 0.5
        self.smoothed_middle_dist = 0.5


class _SwipeDetector:
    """
    Improved swipe detection with:
    - Raw position tracking (bypasses smoothing)
    - Peak velocity detection
    - Directional filtering (must be mostly horizontal)
    - Multiple detection windows
    """
    
    def __init__(self, config: GestureConfig):
        self.config = config
        # Use raw positions for swipe (smoothing kills swipe detection!)
        self.x_history = deque(maxlen=config.SWIPE_HISTORY_SIZE)
        self.y_history = deque(maxlen=config.SWIPE_HISTORY_SIZE)
        self.t_history = deque(maxlen=config.SWIPE_HISTORY_SIZE)
        self.last_swipe_time = 0
        
        # Track peak velocity for better detection
        self.peak_velocity = 0
        self.peak_direction = None
    
    def update(self, x: float, t: float, y: float = 0.5, raw_x: float = None) -> Optional[str]:
        """
        Returns 'LEFT', 'RIGHT', or None.
        Pass raw_x if available (unsmoothed position) for better detection.
        """
        config = self.config
        
        # Prefer raw position if available
        use_x = raw_x if raw_x is not None else x
        
        self.x_history.append(use_x)
        self.y_history.append(y)
        self.t_history.append(t)
        
        if len(self.x_history) < 3:
            return None
        if t - self.last_swipe_time < config.SWIPE_COOLDOWN:
            return None
        
        # Try multiple detection windows (short, medium, long)
        windows = [5, 10, 15]
        
        for window in windows:
            if len(self.x_history) < window:
                continue
            
            result = self._check_window(window)
            if result:
                self.last_swipe_time = t
                self.x_history.clear()
                self.y_history.clear()
                self.t_history.clear()
                self.peak_velocity = 0
                return result
        
        return None
    
    def _check_window(self, window: int) -> Optional[str]:
        """Check a specific window size for swipe."""
        config = self.config
        
        x_arr = np.array(list(self.x_history)[-window:])
        y_arr = np.array(list(self.y_history)[-window:])
        t_arr = np.array(list(self.t_history)[-window:])
        
        dx = x_arr[-1] - x_arr[0]
        dy = y_arr[-1] - y_arr[0]
        dt = t_arr[-1] - t_arr[0]
        
        if dt < 0.03:  # Need at least 30ms
            return None
        
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        
        # Must be mostly horizontal (X movement >> Y movement)
        if abs_dy > 0.01 and abs_dx / max(abs_dy, 0.001) < config.SWIPE_DIRECTIONAL_RATIO:
            return None
        
        velocity = abs_dx / dt
        
        # Check thresholds
        if abs_dx < config.SWIPE_MIN_DISTANCE:
            return None
        if velocity < config.SWIPE_MIN_VELOCITY:
            return None
        
        # Additional check: make sure movement is consistent direction
        # (not back-and-forth)
        if len(x_arr) >= 5:
            mid = len(x_arr) // 2
            first_half = x_arr[mid] - x_arr[0]
            second_half = x_arr[-1] - x_arr[mid]
            # Both halves should move same direction
            if first_half * second_half < 0:
                return None
        
        return "RIGHT" if dx > 0 else "LEFT"
    
    def get_progress(self) -> tuple:
        """Get current swipe progress (0-1) and direction hint."""
        if len(self.x_history) < 3:
            return 0.0, None
        
        x_arr = np.array(self.x_history)
        dx = x_arr[-1] - x_arr[0]
        abs_dx = abs(dx)
        
        progress = min(abs_dx / self.config.SWIPE_MIN_DISTANCE, 1.0)
        direction = "RIGHT" if dx > 0.02 else ("LEFT" if dx < -0.02 else None)
        
        return progress, direction
    
    def reset(self):
        self.x_history.clear()
        self.y_history.clear()
        self.t_history.clear()
        self.peak_velocity = 0


# =====================
# GESTURE ENGINE
# =====================
class GestureEngine:
    """
    Main gesture detection engine with threaded camera/detection.
    
    This is the core class your game uses. It handles:
    - Camera capture (threaded)
    - Hand tracking (threaded) 
    - Gesture detection (pinch, swipe)
    - Frame timing
    
    The camera and MediaPipe detection run on a background thread,
    so your game loop isn't blocked by the ~30fps MediaPipe processing.
    """
    
    def __init__(self, config: GestureConfig = None, threaded: bool = True):
        self.config = config or GestureConfig()
        self._threaded = threaded
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.CAMERA_INDEX)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.config.CAMERA_INDEX}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
        
        # Initialize components
        self._tracker = _HandTracker(self.config)
        self._pinch = _PinchDetector(self.config)
        self._swipe = _SwipeDetector(self.config)
        
        # State
        self._frame_count = 0
        self._last_time = time.time()
        self._running = True
        self._current_frame = None
        self._landmarks = None
        self._status = 'lost'
        
        # Threaded tracking state (shared between threads)
        self._lock = threading.Lock()
        self._thread_frame = None
        self._thread_result = None  # (palm_x, palm_y, landmarks, status, hand_label, raw_x, raw_y)
        self._thread_running = False
        
        # Start background thread if enabled
        if self._threaded:
            self._thread_running = True
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            'swipe_left': [],
            'swipe_right': [],
            'pinch_start': [],
            'pinch_end': [],
            'hand_lost': [],
            'hand_found': [],
        }
    
    def _capture_loop(self):
        """Background thread: capture frames and run MediaPipe detection."""
        frame_count = 0
        while self._thread_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            timestamp_ms = int(frame_count * (1000 / 30))
            frame_count += 1
            
            # Run detection (this is the slow part - ~30fps)
            palm_x, palm_y, landmarks, status, hand_label = self._tracker.update(frame, timestamp_ms)
            raw_x = self._tracker.raw_palm_x
            raw_y = self._tracker.raw_palm_y
            
            # Store results thread-safely
            with self._lock:
                self._thread_frame = frame
                self._thread_result = (palm_x, palm_y, landmarks, status, hand_label, raw_x, raw_y)
    
    @property
    def running(self) -> bool:
        """Check if engine is still running."""
        return self._running
    
    @property
    def camera_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame (flipped)."""
        return self._current_frame
    
    def on(self, event: str, callback: Callable):
        """
        Register a callback for an event.
        
        Events:
            - 'swipe_left': Called when swipe left detected
            - 'swipe_right': Called when swipe right detected  
            - 'pinch_start': Called when pinch begins
            - 'pinch_end': Called when pinch ends
            - 'hand_lost': Called when hand tracking is lost
            - 'hand_found': Called when hand is first detected
        
        Example:
            engine.on('swipe_left', lambda state: print("Swiped left!"))
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, state: GestureState):
        """Emit an event to all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            callback(state)
    
    def update(self) -> GestureState:
        """
        Process one frame and return the current gesture state.
        
        Call this once per game loop iteration.
        Returns a GestureState with all current gesture information.
        """
        # Timing
        current_time = time.time()
        delta_time = current_time - self._last_time
        self._last_time = current_time
        self._frame_count += 1
        
        if self._threaded:
            # Get latest results from background thread (non-blocking)
            with self._lock:
                if self._thread_frame is not None:
                    self._current_frame = self._thread_frame.copy()
                if self._thread_result is not None:
                    palm_x, palm_y, landmarks, status, hand_label, raw_x, raw_y = self._thread_result
                else:
                    palm_x, palm_y, landmarks, status, hand_label, raw_x, raw_y = None, None, None, 'lost', None, 0.5, 0.5
        else:
            # Synchronous mode (original behavior)
            ret, frame = self.cap.read()
            if not ret:
                self._running = False
                return GestureState()
            
            frame = cv2.flip(frame, 1)
            self._current_frame = frame
            
            timestamp_ms = int(self._frame_count * (1000 / 30))
            palm_x, palm_y, landmarks, status, hand_label = self._tracker.update(frame, timestamp_ms)
            raw_x = self._tracker.raw_palm_x
            raw_y = self._tracker.raw_palm_y
        
        was_detected = self._status != 'lost'
        is_detected = status != 'lost'
        
        self._landmarks = landmarks
        self._status = status
        
        # Pinch detection
        is_pinching, pinch_started, pinch_ended, pinch_finger = self._pinch.update(landmarks, current_time)
        
        # Swipe detection - use RAW position for better swipe detection
        swipe_result = None
        if palm_x is not None:
            swipe_result = self._swipe.update(palm_x, current_time, y=palm_y, raw_x=raw_x)
        else:
            self._swipe.reset()
            self._pinch.reset()
        
        # Build state
        state = GestureState(
            timestamp=current_time,
            delta_time=delta_time,
            frame_count=self._frame_count,
            hand_detected=is_detected,
            hand_status=status,
            hand_label=hand_label,
            cursor_x=palm_x,
            cursor_y=palm_y,
            velocity_x=self._tracker.velocity_x if not self._threaded else 0.0,
            velocity_y=self._tracker.velocity_y if not self._threaded else 0.0,
            is_pinching=is_pinching,
            pinch_started=pinch_started,
            pinch_ended=pinch_ended,
            pinch_finger=pinch_finger,
            swipe_left=(swipe_result == 'LEFT'),
            swipe_right=(swipe_result == 'RIGHT'),
            swipe_direction=swipe_result,
            landmarks=landmarks,
        )
        
        # Emit events
        if swipe_result == 'LEFT':
            self._emit('swipe_left', state)
        elif swipe_result == 'RIGHT':
            self._emit('swipe_right', state)
        
        if pinch_started:
            self._emit('pinch_start', state)
        if pinch_ended:
            self._emit('pinch_end', state)
        
        if not was_detected and is_detected:
            self._emit('hand_found', state)
        elif was_detected and not is_detected:
            self._emit('hand_lost', state)
        
        return state
    
    def render_pip(self, game_frame: np.ndarray, 
                   pip_width: int = 200, pip_height: int = 150,
                   position: str = 'bottom-right', padding: int = 10,
                   show_skeleton: bool = True) -> np.ndarray:
        """
        Render the camera feed as a picture-in-picture overlay on your game frame.
        
        Args:
            game_frame: Your game's rendered frame
            pip_width: Width of PIP window
            pip_height: Height of PIP window
            position: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
            padding: Pixels from edge
            show_skeleton: Whether to draw hand skeleton on PIP
        
        Returns:
            The game_frame with PIP overlay (modifies in place)
        """
        if self._current_frame is None:
            return game_frame
        
        pip_frame = self._current_frame.copy()
        
        # Draw skeleton if requested
        if show_skeleton and self._landmarks is not None:
            self._draw_skeleton(pip_frame, self._landmarks, self._status)
        
        # Resize
        pip_resized = cv2.resize(pip_frame, (pip_width, pip_height))
        
        # Calculate position
        h, w = game_frame.shape[:2]
        if position == 'bottom-right':
            x = w - pip_width - padding
            y = h - pip_height - padding
        elif position == 'bottom-left':
            x = padding
            y = h - pip_height - padding
        elif position == 'top-right':
            x = w - pip_width - padding
            y = padding
        else:  # top-left
            x = padding
            y = padding
        
        # Draw border
        cv2.rectangle(game_frame, (x - 2, y - 2), 
                     (x + pip_width + 2, y + pip_height + 2), 
                     (80, 80, 80), 2)
        
        # Place PIP
        game_frame[y:y + pip_height, x:x + pip_width] = pip_resized
        
        return game_frame
    
    def _draw_skeleton(self, frame, landmarks, status):
        """Draw hand skeleton on frame."""
        h, w = frame.shape[:2]
        
        STATUS_COLORS = {
            'detected': (0, 255, 0),
            'predicted': (0, 255, 255),
            'persisted': (0, 165, 255),
            'lost': (0, 0, 255)
        }
        color = STATUS_COLORS.get(status, (128, 128, 128))
        
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
            cv2.circle(frame, (x, y), 3, color, -1)
    
    def show(self, frame: np.ndarray, window_name: str = "Princeps") -> bool:
        """
        Display a frame and handle window events.
        
        Returns False if user pressed Q or closed window.
        Note: Key handling (including ESC) is done by the game loop, not here.
        """
        cv2.imshow(window_name, frame)
        
        # Check if window was closed via X button
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            self._running = False
            return False
        
        return True
    
    def get_key(self) -> int:
        """Get the last key pressed (non-blocking). Returns -1 if none."""
        return cv2.waitKey(1) & 0xFF
    
    def close(self):
        """Clean up resources."""
        # Stop background thread
        if self._threaded:
            self._thread_running = False
            if hasattr(self, '_capture_thread'):
                self._capture_thread.join(timeout=1.0)
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# =====================
# UTILITY FUNCTIONS
# =====================
def draw_cursor(frame: np.ndarray, state: GestureState, 
                size: int = 15, pinch_color=(0, 255, 0), hover_color=(200, 200, 200)):
    """
    Draw a cursor on the frame based on gesture state.
    
    Args:
        frame: Frame to draw on
        state: Current GestureState
        size: Cursor size in pixels
        pinch_color: Color when pinching
        hover_color: Color when hovering
    """
    if state.cursor_x is None:
        return
    
    h, w = frame.shape[:2]
    cx, cy = int(state.cursor_x * w), int(state.cursor_y * h)
    
    if state.is_pinching:
        cv2.circle(frame, (cx, cy), size, pinch_color, -1)
        cv2.circle(frame, (cx, cy), size + 2, (255, 255, 255), 2)
    else:
        cv2.line(frame, (cx - size, cy), (cx + size, cy), hover_color, 2)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), hover_color, 2)
        cv2.circle(frame, (cx, cy), size // 2, hover_color, 2)


def point_in_rect(px: int, py: int, x: int, y: int, w: int, h: int) -> bool:
    """Check if a point is inside a rectangle."""
    return x <= px <= x + w and y <= py <= y + h


# =====================
# DEMO
# =====================
if __name__ == "__main__":
    print("=" * 50)
    print("GESTURE ENGINE DEMO")
    print("=" * 50)
    print("\nGestures:")
    print("  - Move hand to control cursor")
    print("  - Pinch to 'click'")
    print("  - Swipe left/right for actions")
    print("\nPress Q to quit")
    print("=" * 50)
    
    # Create engine
    with GestureEngine() as engine:
        # Register event callbacks (optional - you can also check state directly)
        engine.on('swipe_left', lambda s: print(">>> SWIPE LEFT!"))
        engine.on('swipe_right', lambda s: print(">>> SWIPE RIGHT!"))
        engine.on('pinch_start', lambda s: print(f">>> PINCH START ({s.pinch_finger})"))
        
        # Game loop
        while engine.running:
            # Get current gesture state
            state = engine.update()
            
            # Create a simple game frame
            game_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            game_frame[:] = (40, 40, 50)  # Dark background
            
            # Draw some UI
            cv2.putText(game_frame, "GESTURE ENGINE DEMO", (450, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            
            # Status display
            status_text = f"Hand: {state.hand_status} | Pinch: {state.is_pinching}"
            cv2.putText(game_frame, status_text, (50, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            # Draw cursor
            draw_cursor(game_frame, state)
            
            # Add PIP
            engine.render_pip(game_frame)
            
            # Show
            engine.show(game_frame)
    
    print("\nDemo finished!")
