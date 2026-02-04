"""
Scene Manager Module
====================
Modular scene management with swipe navigation.

Features:
- Swipe left/right to switch between scenes
- Scene carousel with smooth transitions
- Extensible scene registration
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto


class TransitionDirection(Enum):
    """Direction of scene transition."""
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()


@dataclass
class SceneTransition:
    """Holds transition animation state."""
    active: bool = False
    direction: TransitionDirection = TransitionDirection.NONE
    progress: float = 0.0  # 0 to 1
    speed: float = 3.0  # Progress per second
    from_scene: Optional[str] = None
    to_scene: Optional[str] = None


class NavigableScene(ABC):
    """
    Base class for scenes that can be navigated via swipe.
    
    Override can_navigate_left/right to control when navigation is allowed.
    """
    
    def __init__(self):
        self.scene_manager: Optional['SceneCarouselManager'] = None
        self.shared_data: Dict[str, Any] = {}
        self.name: str = ""
    
    def on_enter(self, from_scene: Optional[str] = None, direction: TransitionDirection = TransitionDirection.NONE):
        """Called when entering this scene."""
        pass
    
    def on_exit(self, to_scene: Optional[str] = None, direction: TransitionDirection = TransitionDirection.NONE):
        """Called when leaving this scene."""
        pass
    
    @abstractmethod
    def update(self, gesture_state: Any, delta_time: float) -> Optional[str]:
        """
        Update scene logic.
        
        Returns:
            Scene name to force transition to, or None
        """
        pass
    
    @abstractmethod
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render the scene to the frame."""
        pass
    
    def can_navigate_left(self) -> bool:
        """Override to control when left navigation is allowed."""
        return True
    
    def can_navigate_right(self) -> bool:
        """Override to control when right navigation is allowed."""
        return True
    
    def get_indicator_label(self) -> str:
        """Override to provide a custom indicator label."""
        return self.name.upper()


class SceneCarouselManager:
    """
    Manages a carousel of scenes with swipe navigation.
    
    Scenes are arranged in a linear order. Swipe left goes to next scene,
    swipe right goes to previous scene.
    """
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        
        self.scenes: Dict[str, NavigableScene] = {}
        self.scene_order: List[str] = []
        self.current_scene: Optional[str] = None
        
        self.transition = SceneTransition()
        self.shared_data: Dict[str, Any] = {}
        
        # Visual settings
        self.show_indicators = True
        self.indicator_height = 40
    
    def add_scene(self, name: str, scene: NavigableScene, position: Optional[int] = None):
        """
        Add a scene to the carousel.
        
        Args:
            name: Unique scene identifier
            scene: The scene instance
            position: Optional position in order (default: append to end)
        """
        scene.scene_manager = self
        scene.shared_data = self.shared_data
        scene.name = name
        self.scenes[name] = scene
        
        if position is not None:
            self.scene_order.insert(position, name)
        else:
            self.scene_order.append(name)
    
    def set_scene(self, name: str):
        """Set the current scene without transition."""
        if name not in self.scenes:
            raise ValueError(f"Scene '{name}' not found")
        
        if self.current_scene:
            self.scenes[self.current_scene].on_exit(name, TransitionDirection.NONE)
        
        prev = self.current_scene
        self.current_scene = name
        self.scenes[name].on_enter(prev, TransitionDirection.NONE)
    
    def get_adjacent_scenes(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the scenes to the left and right of current."""
        if not self.current_scene or not self.scene_order:
            return None, None
        
        idx = self.scene_order.index(self.current_scene)
        
        left_scene = self.scene_order[idx - 1] if idx > 0 else None
        right_scene = self.scene_order[idx + 1] if idx < len(self.scene_order) - 1 else None
        
        return left_scene, right_scene
    
    def navigate_left(self) -> bool:
        """Navigate to the previous scene (swipe right gesture)."""
        left_scene, _ = self.get_adjacent_scenes()
        
        if left_scene and self.scenes[self.current_scene].can_navigate_left():
            self._start_transition(left_scene, TransitionDirection.RIGHT)
            return True
        return False
    
    def navigate_right(self) -> bool:
        """Navigate to the next scene (swipe left gesture)."""
        _, right_scene = self.get_adjacent_scenes()
        
        if right_scene and self.scenes[self.current_scene].can_navigate_right():
            self._start_transition(right_scene, TransitionDirection.LEFT)
            return True
        return False
    
    def _start_transition(self, to_scene: str, direction: TransitionDirection):
        """Start a transition animation."""
        if self.transition.active:
            return
        
        self.transition.active = True
        self.transition.direction = direction
        self.transition.progress = 0.0
        self.transition.from_scene = self.current_scene
        self.transition.to_scene = to_scene
        
        # Notify scenes
        self.scenes[self.current_scene].on_exit(to_scene, direction)
        self.scenes[to_scene].on_enter(self.current_scene, direction)
    
    def update(self, gesture_state: Any, delta_time: float) -> bool:
        """
        Update the scene manager.
        
        Args:
            gesture_state: The current gesture state
            delta_time: Time since last update
        
        Returns:
            True if still running, False to exit
        """
        # Handle transition animation
        if self.transition.active:
            self.transition.progress += self.transition.speed * delta_time
            
            if self.transition.progress >= 1.0:
                self.current_scene = self.transition.to_scene
                self.transition.active = False
                self.transition.progress = 0.0
            
            return True
        
        # Handle swipe navigation
        if hasattr(gesture_state, 'swipe_left') and gesture_state.swipe_left:
            self.navigate_right()  # Swipe left = go right
        elif hasattr(gesture_state, 'swipe_right') and gesture_state.swipe_right:
            self.navigate_left()  # Swipe right = go left
        
        # Update current scene
        if self.current_scene:
            result = self.scenes[self.current_scene].update(gesture_state, delta_time)
            
            # Handle scene requesting a transition
            if result and result in self.scenes:
                self.set_scene(result)
        
        return True
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render the current scene(s) with transitions."""
        if self.transition.active:
            self._render_transition(frame, gesture_state)
        elif self.current_scene:
            self.scenes[self.current_scene].render(frame, gesture_state)
        
        # Draw navigation indicators
        if self.show_indicators:
            self._render_indicators(frame)
    
    def _render_transition(self, frame: np.ndarray, gesture_state: Any):
        """Render transition animation between scenes."""
        from_frame = np.zeros_like(frame)
        to_frame = np.zeros_like(frame)
        
        # Render both scenes
        if self.transition.from_scene:
            self.scenes[self.transition.from_scene].render(from_frame, gesture_state)
        if self.transition.to_scene:
            self.scenes[self.transition.to_scene].render(to_frame, gesture_state)
        
        # Calculate slide offset
        progress = self._ease_out_cubic(self.transition.progress)
        offset = int(self.width * progress)
        
        if self.transition.direction == TransitionDirection.LEFT:
            # Sliding left (to right scene)
            # From scene slides out left, to scene slides in from right
            from_offset = -offset
            to_offset = self.width - offset
        else:
            # Sliding right (to left scene)
            from_offset = offset
            to_offset = -self.width + offset
        
        # Composite frames with offset
        self._blit_with_offset(frame, from_frame, from_offset)
        self._blit_with_offset(frame, to_frame, to_offset)
    
    def _ease_out_cubic(self, t: float) -> float:
        """Cubic ease-out for smooth deceleration."""
        return 1 - pow(1 - t, 3)
    
    def _blit_with_offset(self, dest: np.ndarray, src: np.ndarray, x_offset: int):
        """Blit source onto destination with horizontal offset."""
        h, w = dest.shape[:2]
        
        if x_offset >= w or x_offset <= -w:
            return
        
        if x_offset >= 0:
            src_start = 0
            src_end = w - x_offset
            dest_start = x_offset
            dest_end = w
        else:
            src_start = -x_offset
            src_end = w
            dest_start = 0
            dest_end = w + x_offset
        
        dest[:, dest_start:dest_end] = src[:, src_start:src_end]
    
    def _render_indicators(self, frame: np.ndarray):
        """Render scene position indicators."""
        if not self.scene_order or not self.current_scene:
            return
        
        current_idx = self.scene_order.index(self.current_scene)
        num_scenes = len(self.scene_order)
        
        # Draw at bottom of screen
        y = self.height - 25
        
        # Calculate total width of indicators
        dot_radius = 6
        dot_spacing = 30
        total_width = (num_scenes - 1) * dot_spacing
        start_x = (self.width - total_width) // 2
        
        # Get current scene label
        current_scene_obj = self.scenes[self.current_scene]
        label = current_scene_obj.get_indicator_label()
        
        # Draw label above dots
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        cv2.putText(frame, label, ((self.width - tw) // 2, y - 20), 
                   font, 0.5, (150, 150, 160), 1)
        
        # Draw dots
        for i in range(num_scenes):
            x = start_x + i * dot_spacing
            
            if i == current_idx:
                # Current scene - filled bright
                cv2.circle(frame, (x, y), dot_radius, (200, 200, 220), -1)
            else:
                # Other scenes - outlined
                cv2.circle(frame, (x, y), dot_radius, (80, 80, 100), 2)
        
        # Navigation hints
        left_scene, right_scene = self.get_adjacent_scenes()
        
        if left_scene and self.scenes[self.current_scene].can_navigate_left():
            # Left arrow
            self._draw_arrow(frame, 30, self.height // 2, "left", (80, 80, 100))
        
        if right_scene and self.scenes[self.current_scene].can_navigate_right():
            # Right arrow
            self._draw_arrow(frame, self.width - 30, self.height // 2, "right", (80, 80, 100))
    
    def _draw_arrow(self, frame: np.ndarray, x: int, y: int, direction: str, color: Tuple[int, int, int]):
        """Draw a navigation arrow."""
        size = 15
        
        if direction == "left":
            pts = np.array([
                [x + size, y - size],
                [x, y],
                [x + size, y + size]
            ], np.int32)
        else:
            pts = np.array([
                [x - size, y - size],
                [x, y],
                [x - size, y + size]
            ], np.int32)
        
        cv2.polylines(frame, [pts], False, color, 2)
