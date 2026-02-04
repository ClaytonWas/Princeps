"""
Settings Module
================
Game settings and configuration management.

Features:
- Input mode switching (hand tracking / mouse)
- Gesture parameter tuning
- Resolution and display settings
- Keyboard bindings
"""

import cv2
import numpy as np
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto


# =====================================================
# SETTINGS DATA
# =====================================================

class InputMode(Enum):
    """Input method for cursor control."""
    HAND_TRACKING = "Hand Tracking"
    MOUSE = "Mouse"


@dataclass
class ControlSettings:
    """Settings for input controls."""
    # Input mode
    input_mode: str = "Hand Tracking"  # "Hand Tracking" or "Mouse"
    
    # Gesture sensitivity
    swipe_min_distance: float = 0.06      # Min distance for swipe (0.02 - 0.15)
    swipe_min_velocity: float = 0.2       # Min velocity for swipe (0.1 - 0.6)
    pinch_threshold: float = 0.055        # Distance for pinch detection (0.03 - 0.08)
    pinch_hold_time: float = 0.5          # Time to hold for selection (0.2 - 1.0)
    
    # Keyboard bindings (for mouse mode)
    key_swipe_left: int = ord('a')        # Default: A
    key_swipe_right: int = ord('d')       # Default: D
    key_select: int = ord(' ')            # Default: Space
    
    # Mouse settings
    mouse_smoothing: float = 0.3          # Mouse movement smoothing (0 - 0.8)


@dataclass
class GraphicsSettings:
    """Settings for display."""
    resolution: Tuple[int, int] = (1280, 720)
    fullscreen: bool = False
    show_pip: bool = True                 # Picture-in-picture camera view
    pip_size: Tuple[int, int] = (180, 135)
    max_fps: int = 60                     # 30-250, or 251 for uncapped


@dataclass
class GameSettings:
    """Complete game settings."""
    controls: ControlSettings = field(default_factory=ControlSettings)
    graphics: GraphicsSettings = field(default_factory=GraphicsSettings)
    
    def save(self, path: str = "settings.json"):
        """Save settings to file."""
        data = {
            'controls': asdict(self.controls),
            'graphics': {
                **asdict(self.graphics),
                'resolution': list(self.graphics.resolution),
                'pip_size': list(self.graphics.pip_size)
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str = "settings.json") -> 'GameSettings':
        """Load settings from file."""
        if not os.path.exists(path):
            return cls()
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            settings = cls()
            
            if 'controls' in data:
                for key, value in data['controls'].items():
                    if hasattr(settings.controls, key):
                        setattr(settings.controls, key, value)
            
            if 'graphics' in data:
                for key, value in data['graphics'].items():
                    if key == 'resolution':
                        settings.graphics.resolution = tuple(value)
                    elif key == 'pip_size':
                        settings.graphics.pip_size = tuple(value)
                    elif hasattr(settings.graphics, key):
                        setattr(settings.graphics, key, value)
            
            return settings
        except Exception as e:
            print(f"Error loading settings: {e}")
            return cls()


# =====================================================
# UI COMPONENTS FOR SETTINGS
# =====================================================

class SettingsSlider:
    """Interactive slider for settings."""
    
    def __init__(self, x: int, y: int, width: int, 
                 label: str, min_val: float, max_val: float, 
                 value: float, step: float = 0.01):
        self.x = x
        self.y = y
        self.width = width
        self.height = 25
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        
        self.is_hovered = False
        self.is_dragging = False
        self.hover_time = 0.0
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               is_pinching: bool, frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update slider. Returns True if value changed."""
        if cursor_x is None:
            self.is_hovered = False
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        # Check if hovering over slider track
        in_slider = (self.x <= px <= self.x + self.width and 
                     self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_slider
        
        if in_slider and is_pinching:
            # Calculate new value from cursor position
            relative_x = (px - self.x) / self.width
            relative_x = max(0, min(1, relative_x))
            
            new_value = self.min_val + relative_x * (self.max_val - self.min_val)
            new_value = round(new_value / self.step) * self.step
            
            if new_value != self.value:
                self.value = new_value
                return True
        
        return False
    
    def draw(self, frame: np.ndarray):
        """Draw the slider."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Label
        cv2.putText(frame, self.label, (self.x, self.y - 5), 
                   font, 0.4, (180, 180, 180), 1)
        
        # Track background
        track_color = (60, 60, 70) if not self.is_hovered else (70, 70, 85)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), track_color, -1)
        
        # Fill
        fill_ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(self.width * fill_ratio)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + fill_width, self.y + self.height), (80, 140, 180), -1)
        
        # Border
        border_color = (120, 160, 200) if self.is_hovered else (100, 100, 120)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 1)
        
        # Value text
        value_text = f"{self.value:.2f}"
        cv2.putText(frame, value_text, (self.x + self.width + 10, self.y + 18), 
                   font, 0.4, (200, 200, 200), 1)


class FPSSlider:
    """Slider for FPS setting with special handling for uncapped (251)."""
    
    def __init__(self, x: int, y: int, width: int, value: int = 60):
        self.x = x
        self.y = y
        self.width = width
        self.height = 25
        self.label = "Max FPS"
        self.min_val = 30
        self.max_val = 251  # 251 = Uncapped
        self.value = value
        
        self.is_hovered = False
        self.is_dragging = False
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               is_pinching: bool, frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update slider. Returns True if value changed."""
        if cursor_x is None:
            self.is_hovered = False
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        # Check if hovering over slider track
        in_slider = (self.x <= px <= self.x + self.width and 
                     self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_slider
        
        if in_slider and is_pinching:
            # Calculate new value from cursor position
            relative_x = (px - self.x) / self.width
            relative_x = max(0, min(1, relative_x))
            
            # Map to FPS value (30-251)
            raw_value = self.min_val + relative_x * (self.max_val - self.min_val)
            
            # Snap to increments of 10 (30, 40, 50, ..., 250, 251)
            if raw_value >= 249:
                new_value = 251  # Uncapped
            else:
                new_value = int(round(raw_value / 10) * 10)
                new_value = max(30, min(250, new_value))
            
            if new_value != self.value:
                self.value = new_value
                return True
        
        return False
    
    def draw(self, frame: np.ndarray):
        """Draw the slider."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Label
        cv2.putText(frame, self.label, (self.x, self.y - 5), 
                   font, 0.4, (180, 180, 180), 1)
        
        # Track background
        track_color = (60, 60, 70) if not self.is_hovered else (70, 70, 85)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), track_color, -1)
        
        # Fill
        fill_ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(self.width * fill_ratio)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + fill_width, self.y + self.height), (80, 140, 180), -1)
        
        # Border
        border_color = (120, 160, 200) if self.is_hovered else (100, 100, 120)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 1)
        
        # Value text
        if self.value >= 251:
            value_text = "Uncapped"
        else:
            value_text = f"{self.value}"
        cv2.putText(frame, value_text, (self.x + self.width + 10, self.y + 18), 
                   font, 0.4, (200, 200, 200), 1)


class SettingsButton:
    """Button for settings menu."""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: Tuple[int, int, int] = (60, 60, 70)):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        
        self.is_hovered = False
        self.hover_start = 0.0
        self.activation_time = 0.4
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               is_pinching: bool, frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update button. Returns True if activated."""
        if cursor_x is None:
            self.is_hovered = False
            self.hover_start = 0.0
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        in_btn = (self.x <= px <= self.x + self.width and 
                  self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_btn
        
        if in_btn and is_pinching:
            if self.hover_start == 0.0:
                self.hover_start = timestamp
            elif timestamp - self.hover_start >= self.activation_time:
                self.hover_start = timestamp + 0.3  # Cooldown
                return True
        else:
            if not is_pinching:
                self.hover_start = 0.0
        
        return False
    
    def get_progress(self, timestamp: float) -> float:
        """Get activation progress 0-1."""
        if self.hover_start == 0.0:
            return 0.0
        return min(1.0, (timestamp - self.hover_start) / self.activation_time)
    
    def draw(self, frame: np.ndarray, timestamp: float):
        """Draw the button."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background
        bg_color = self.color if not self.is_hovered else tuple(min(255, c + 20) for c in self.color)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), bg_color, -1)
        
        # Progress fill
        progress = self.get_progress(timestamp)
        if progress > 0:
            fill_width = int(self.width * progress)
            cv2.rectangle(frame, (self.x, self.y), 
                         (self.x + fill_width, self.y + self.height), (60, 150, 60), -1)
        
        # Border
        border_color = (150, 150, 150) if self.is_hovered else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 2)
        
        # Text
        (tw, th), _ = cv2.getTextSize(self.text, font, 0.55, 1)
        tx = self.x + (self.width - tw) // 2
        ty = self.y + (self.height + th) // 2
        cv2.putText(frame, self.text, (tx, ty), font, 0.55, (220, 220, 220), 1)


class SettingsToggle:
    """Toggle button for boolean settings."""
    
    def __init__(self, x: int, y: int, label: str, value: bool):
        self.x = x
        self.y = y
        self.width = 50
        self.height = 25
        self.label = label
        self.value = value
        
        self.is_hovered = False
        self.last_toggle = 0.0
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               is_pinching: bool, pinch_started: bool,
               frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update toggle. Returns True if value changed."""
        if cursor_x is None:
            self.is_hovered = False
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        in_toggle = (self.x <= px <= self.x + self.width and 
                     self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_toggle
        
        if in_toggle and pinch_started and timestamp - self.last_toggle > 0.3:
            self.value = not self.value
            self.last_toggle = timestamp
            return True
        
        return False
    
    def draw(self, frame: np.ndarray):
        """Draw the toggle."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Label
        cv2.putText(frame, self.label, (self.x, self.y - 5), 
                   font, 0.4, (180, 180, 180), 1)
        
        # Track
        track_color = (60, 120, 60) if self.value else (60, 60, 70)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), track_color, -1)
        
        # Knob
        knob_x = self.x + self.width - 12 if self.value else self.x + 2
        cv2.rectangle(frame, (knob_x, self.y + 2), 
                     (knob_x + 10, self.y + self.height - 2), (220, 220, 220), -1)
        
        # Border
        border_color = (150, 150, 150) if self.is_hovered else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 1)
        
        # State text
        state_text = "ON" if self.value else "OFF"
        cv2.putText(frame, state_text, (self.x + self.width + 10, self.y + 18), 
                   font, 0.4, (150, 200, 150) if self.value else (150, 150, 150), 1)


class ResolutionSelector:
    """Dropdown-style resolution selector."""
    
    RESOLUTIONS = [
        (1280, 720),
        (1366, 768),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
    ]
    
    def __init__(self, x: int, y: int, current: Tuple[int, int]):
        self.x = x
        self.y = y
        self.width = 150
        self.height = 30
        self.current = current
        self.current_idx = self._find_index(current)
        
        self.is_hovered = False
        self.left_hovered = False
        self.right_hovered = False
        self.last_change = 0.0
    
    def _find_index(self, res: Tuple[int, int]) -> int:
        for i, r in enumerate(self.RESOLUTIONS):
            if r == res:
                return i
        return 0
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               pinch_started: bool, frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update selector. Returns True if value changed."""
        if cursor_x is None:
            self.is_hovered = False
            self.left_hovered = False
            self.right_hovered = False
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        in_area = (self.x <= px <= self.x + self.width and 
                   self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_area
        self.left_hovered = in_area and px < self.x + 30
        self.right_hovered = in_area and px > self.x + self.width - 30
        
        if pinch_started and timestamp - self.last_change > 0.3:
            if self.left_hovered and self.current_idx > 0:
                self.current_idx -= 1
                self.current = self.RESOLUTIONS[self.current_idx]
                self.last_change = timestamp
                return True
            elif self.right_hovered and self.current_idx < len(self.RESOLUTIONS) - 1:
                self.current_idx += 1
                self.current = self.RESOLUTIONS[self.current_idx]
                self.last_change = timestamp
                return True
        
        return False
    
    def draw(self, frame: np.ndarray):
        """Draw the selector."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Label
        cv2.putText(frame, "Resolution", (self.x, self.y - 5), 
                   font, 0.4, (180, 180, 180), 1)
        
        # Background
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), (50, 50, 60), -1)
        
        # Left arrow
        arrow_color = (200, 200, 200) if self.left_hovered else (120, 120, 120)
        cv2.putText(frame, "<", (self.x + 8, self.y + 22), font, 0.6, arrow_color, 2)
        
        # Right arrow
        arrow_color = (200, 200, 200) if self.right_hovered else (120, 120, 120)
        cv2.putText(frame, ">", (self.x + self.width - 20, self.y + 22), font, 0.6, arrow_color, 2)
        
        # Current value
        res_text = f"{self.current[0]}x{self.current[1]}"
        (tw, _), _ = cv2.getTextSize(res_text, font, 0.45, 1)
        cv2.putText(frame, res_text, (self.x + (self.width - tw) // 2, self.y + 20), 
                   font, 0.45, (220, 220, 220), 1)
        
        # Border
        border_color = (150, 150, 150) if self.is_hovered else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 1)


class InputModeSelector:
    """Selector for input mode (Hand Tracking / Mouse)."""
    
    MODES = ["Hand Tracking", "Mouse"]
    
    def __init__(self, x: int, y: int, current: str):
        self.x = x
        self.y = y
        self.width = 180
        self.height = 30
        self.current = current
        self.current_idx = self.MODES.index(current) if current in self.MODES else 0
        
        self.is_hovered = False
        self.left_hovered = False
        self.right_hovered = False
        self.last_change = 0.0
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               pinch_started: bool, frame_w: int, frame_h: int, timestamp: float) -> bool:
        """Update selector. Returns True if value changed."""
        if cursor_x is None:
            self.is_hovered = False
            return False
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        in_area = (self.x <= px <= self.x + self.width and 
                   self.y <= py <= self.y + self.height)
        
        self.is_hovered = in_area
        self.left_hovered = in_area and px < self.x + 30
        self.right_hovered = in_area and px > self.x + self.width - 30
        
        if pinch_started and timestamp - self.last_change > 0.3:
            if self.left_hovered or self.right_hovered:
                self.current_idx = 1 - self.current_idx  # Toggle between 0 and 1
                self.current = self.MODES[self.current_idx]
                self.last_change = timestamp
                return True
        
        return False
    
    def draw(self, frame: np.ndarray):
        """Draw the selector."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Label
        cv2.putText(frame, "Input Mode", (self.x, self.y - 5), 
                   font, 0.4, (180, 180, 180), 1)
        
        # Background
        bg_color = (50, 70, 50) if self.current == "Hand Tracking" else (50, 50, 70)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), bg_color, -1)
        
        # Arrows
        arrow_color = (200, 200, 200) if self.left_hovered else (120, 120, 120)
        cv2.putText(frame, "<", (self.x + 8, self.y + 22), font, 0.6, arrow_color, 2)
        
        arrow_color = (200, 200, 200) if self.right_hovered else (120, 120, 120)
        cv2.putText(frame, ">", (self.x + self.width - 20, self.y + 22), font, 0.6, arrow_color, 2)
        
        # Current value
        (tw, _), _ = cv2.getTextSize(self.current, font, 0.45, 1)
        cv2.putText(frame, self.current, (self.x + (self.width - tw) // 2, self.y + 20), 
                   font, 0.45, (220, 220, 220), 1)
        
        # Border
        border_color = (150, 150, 150) if self.is_hovered else (100, 100, 100)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), border_color, 1)


# =====================================================
# GESTURE VISUALIZER
# =====================================================

class GestureVisualizer:
    """Visualizes gesture detection for the controls menu."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # History for visualization
        self.position_history: List[Tuple[float, float]] = []
        self.max_history = 30
        
        # Event display
        self.last_events: List[Tuple[float, str]] = []
        self.max_events = 5
    
    def add_event(self, timestamp: float, event: str):
        """Add an event to display."""
        self.last_events.append((timestamp, event))
        if len(self.last_events) > self.max_events:
            self.last_events.pop(0)
    
    def update(self, gesture_state: Any):
        """Update with current gesture state."""
        if gesture_state.cursor_x is not None:
            self.position_history.append((gesture_state.cursor_x, gesture_state.cursor_y))
            if len(self.position_history) > self.max_history:
                self.position_history.pop(0)
        
        # Track events
        if gesture_state.swipe_left:
            self.add_event(gesture_state.timestamp, "SWIPE LEFT")
        if gesture_state.swipe_right:
            self.add_event(gesture_state.timestamp, "SWIPE RIGHT")
        if gesture_state.pinch_started:
            self.add_event(gesture_state.timestamp, "PINCH START")
        if gesture_state.pinch_ended:
            self.add_event(gesture_state.timestamp, "PINCH END")
    
    def draw(self, frame: np.ndarray, gesture_state: Any, timestamp: float):
        """Draw the gesture visualizer."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Background panel
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), (30, 30, 40), -1)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), (80, 80, 100), 1)
        
        # Title
        cv2.putText(frame, "Gesture Monitor", (self.x + 10, self.y + 20), 
                   font, 0.5, (180, 180, 200), 1)
        
        # Tracking area
        track_x = self.x + 10
        track_y = self.y + 35
        track_w = self.width - 20
        track_h = 100
        
        cv2.rectangle(frame, (track_x, track_y), 
                     (track_x + track_w, track_y + track_h), (20, 20, 30), -1)
        
        # Draw position history (trail)
        if len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                alpha = i / len(self.position_history)
                color = (int(100 * alpha), int(180 * alpha), int(100 * alpha))
                
                p1 = self.position_history[i - 1]
                p2 = self.position_history[i]
                
                x1 = track_x + int(p1[0] * track_w)
                y1 = track_y + int(p1[1] * track_h)
                x2 = track_x + int(p2[0] * track_w)
                y2 = track_y + int(p2[1] * track_h)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Current position
        if gesture_state.cursor_x is not None:
            cx = track_x + int(gesture_state.cursor_x * track_w)
            cy = track_y + int(gesture_state.cursor_y * track_h)
            
            color = (100, 200, 100) if gesture_state.is_pinching else (200, 200, 200)
            cv2.circle(frame, (cx, cy), 8, color, -1)
            cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1)
        
        # Status info
        info_y = track_y + track_h + 25
        
        # Hand status
        status_color = {
            'detected': (100, 200, 100),
            'predicted': (100, 200, 200),
            'persisted': (100, 150, 200),
            'lost': (100, 100, 150)
        }.get(gesture_state.hand_status, (150, 150, 150))
        
        cv2.putText(frame, f"Hand: {gesture_state.hand_status}", (self.x + 10, info_y), 
                   font, 0.4, status_color, 1)
        
        # Pinch status
        pinch_text = "PINCHING" if gesture_state.is_pinching else "Open"
        pinch_color = (100, 200, 100) if gesture_state.is_pinching else (150, 150, 150)
        cv2.putText(frame, f"Pinch: {pinch_text}", (self.x + 120, info_y), 
                   font, 0.4, pinch_color, 1)
        
        # Velocity
        info_y += 20
        cv2.putText(frame, f"Vel X: {gesture_state.velocity_x:+.2f}", (self.x + 10, info_y), 
                   font, 0.35, (150, 150, 180), 1)
        cv2.putText(frame, f"Vel Y: {gesture_state.velocity_y:+.2f}", (self.x + 100, info_y), 
                   font, 0.35, (150, 150, 180), 1)
        
        # Recent events
        info_y += 25
        cv2.putText(frame, "Events:", (self.x + 10, info_y), font, 0.4, (180, 180, 180), 1)
        
        for i, (evt_time, evt_text) in enumerate(reversed(self.last_events[-3:])):
            age = timestamp - evt_time
            alpha = max(0.3, 1.0 - age / 2.0)
            color = tuple(int(c * alpha) for c in (150, 220, 150))
            cv2.putText(frame, f"  {evt_text}", (self.x + 10, info_y + 18 + i * 15), 
                       font, 0.35, color, 1)


# =====================================================
# KEY BINDING DISPLAY
# =====================================================

class KeyBindingDisplay:
    """Shows and allows editing of key bindings."""
    
    def __init__(self, x: int, y: int, settings: ControlSettings):
        self.x = x
        self.y = y
        self.settings = settings
        
        self.bindings = [
            ("Swipe Left", "key_swipe_left", settings.key_swipe_left),
            ("Swipe Right", "key_swipe_right", settings.key_swipe_right),
            ("Select/Pinch", "key_select", settings.key_select),
        ]
        
        self.waiting_for_key = None  # Index of binding waiting for input
    
    def get_key_name(self, key_code: int) -> str:
        """Get display name for a key code."""
        special_keys = {
            32: "SPACE",
            13: "ENTER",
            27: "ESC",
            9: "TAB",
            8: "BACKSPACE",
        }
        
        if key_code in special_keys:
            return special_keys[key_code]
        elif 97 <= key_code <= 122:  # a-z
            return chr(key_code).upper()
        elif 48 <= key_code <= 57:  # 0-9
            return chr(key_code)
        else:
            return f"[{key_code}]"
    
    def handle_key(self, key: int) -> bool:
        """Handle a key press. Returns True if a binding was set."""
        if self.waiting_for_key is not None and key != 27:  # ESC cancels
            attr_name = self.bindings[self.waiting_for_key][1]
            setattr(self.settings, attr_name, key)
            self.bindings[self.waiting_for_key] = (
                self.bindings[self.waiting_for_key][0],
                attr_name,
                key
            )
            self.waiting_for_key = None
            return True
        elif key == 27:
            self.waiting_for_key = None
        return False
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               pinch_started: bool, frame_w: int, frame_h: int) -> int:
        """Check if a binding button was clicked. Returns index or -1."""
        if cursor_x is None or not pinch_started:
            return -1
        
        px = int(cursor_x * frame_w)
        py = int(cursor_y * frame_h)
        
        for i, (label, attr, key) in enumerate(self.bindings):
            btn_y = self.y + 25 + i * 35
            if (self.x + 120 <= px <= self.x + 200 and 
                btn_y <= py <= btn_y + 25):
                self.waiting_for_key = i
                return i
        
        return -1
    
    def draw(self, frame: np.ndarray):
        """Draw the key binding editor."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, "Key Bindings (Mouse Mode)", (self.x, self.y), 
                   font, 0.45, (180, 180, 200), 1)
        
        for i, (label, attr, key) in enumerate(self.bindings):
            y = self.y + 25 + i * 35
            
            # Label
            cv2.putText(frame, label, (self.x, y + 18), font, 0.4, (160, 160, 160), 1)
            
            # Key button
            is_waiting = self.waiting_for_key == i
            btn_color = (80, 100, 80) if is_waiting else (50, 50, 60)
            cv2.rectangle(frame, (self.x + 120, y), (self.x + 200, y + 25), btn_color, -1)
            cv2.rectangle(frame, (self.x + 120, y), (self.x + 200, y + 25), (100, 100, 120), 1)
            
            key_text = "Press key..." if is_waiting else self.get_key_name(key)
            text_color = (200, 255, 200) if is_waiting else (200, 200, 200)
            cv2.putText(frame, key_text, (self.x + 130, y + 18), font, 0.4, text_color, 1)
