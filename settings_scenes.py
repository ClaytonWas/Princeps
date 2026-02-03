"""
Settings Scenes
================
Settings menu scenes for the game.
"""

import cv2
import numpy as np
from typing import Optional, Any, Dict
from abc import ABC, abstractmethod

from settings_menu import (
    GameSettings, ControlSettings, GraphicsSettings,
    SettingsButton, SettingsSlider, SettingsToggle,
    ResolutionSelector, InputModeSelector,
    GestureVisualizer, KeyBindingDisplay
)


# =====================================================
# SETTINGS SCENE BASE
# =====================================================

class SettingsSceneBase(ABC):
    """Base class for settings scenes."""
    
    def __init__(self, game: Any):
        self.game = game
        self.settings: GameSettings = game.settings
    
    @abstractmethod
    def update(self, gesture_state: Any, key: int) -> Optional[str]:
        """Update scene. Returns next scene name or None."""
        pass
    
    @abstractmethod
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render the scene."""
        pass


# =====================================================
# MAIN SETTINGS MENU
# =====================================================

class SettingsMainMenu(SettingsSceneBase):
    """Main settings menu with 4 options."""
    
    def __init__(self, game: Any):
        super().__init__(game)
        
        self.buttons = []
        self._create_buttons()
    
    def _create_buttons(self):
        """Create menu buttons."""
        w, h = self.game.width, self.game.height
        btn_w, btn_h = 250, 55
        start_y = h // 3
        center_x = (w - btn_w) // 2
        
        self.buttons = [
            (SettingsButton(center_x, start_y, btn_w, btn_h, "Return to Game"), "resume"),
            (SettingsButton(center_x, start_y + 70, btn_w, btn_h, "Controls"), "controls"),
            (SettingsButton(center_x, start_y + 140, btn_w, btn_h, "Graphics"), "graphics"),
            (SettingsButton(center_x, start_y + 210, btn_w, btn_h, "Quit to Desktop", (80, 50, 50)), "quit"),
        ]
    
    def update(self, gesture_state: Any, key: int) -> Optional[str]:
        """Check for button activations."""
        for btn, action in self.buttons:
            if btn.update(
                gesture_state.cursor_x, gesture_state.cursor_y,
                gesture_state.is_pinching,
                self.game.width, self.game.height,
                gesture_state.timestamp
            ):
                return action
        
        return None
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render the main settings menu."""
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.game.width, self.game.height), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        title = "SETTINGS"
        (tw, _), _ = cv2.getTextSize(title, font, 1.2, 2)
        cv2.putText(frame, title, ((self.game.width - tw) // 2, 100), 
                   font, 1.2, (200, 200, 220), 2)
        
        # Subtitle
        subtitle = "Press ESC to return"
        (tw, _), _ = cv2.getTextSize(subtitle, font, 0.5, 1)
        cv2.putText(frame, subtitle, ((self.game.width - tw) // 2, 130), 
                   font, 0.5, (120, 120, 140), 1)
        
        # Buttons
        for btn, _ in self.buttons:
            btn.draw(frame, gesture_state.timestamp)
        
        # Draw cursor
        from gesture_engine import draw_cursor
        draw_cursor(frame, gesture_state)


# =====================================================
# CONTROLS SETTINGS
# =====================================================

class ControlsSettingsMenu(SettingsSceneBase):
    """Controls settings with gesture visualization."""
    
    def __init__(self, game: Any):
        super().__init__(game)
        
        self.back_button = None
        self.input_mode_selector = None
        self.sliders = []
        self.gesture_viz = None
        self.key_bindings = None
        
        self._create_ui()
    
    def _create_ui(self):
        """Create UI elements."""
        w, h = self.game.width, self.game.height
        ctrl = self.settings.controls
        
        # Back button
        self.back_button = SettingsButton(30, h - 70, 120, 40, "< Back")
        
        # Left column - Gesture settings
        col1_x = 50
        
        # Input mode selector
        self.input_mode_selector = InputModeSelector(col1_x, 130, ctrl.input_mode)
        
        # Gesture sensitivity sliders
        self.sliders = [
            (SettingsSlider(col1_x, 200, 200, "Swipe Distance", 0.02, 0.15, ctrl.swipe_min_distance, 0.01), "swipe_min_distance"),
            (SettingsSlider(col1_x, 260, 200, "Swipe Velocity", 0.1, 0.6, ctrl.swipe_min_velocity, 0.05), "swipe_min_velocity"),
            (SettingsSlider(col1_x, 320, 200, "Pinch Threshold", 0.03, 0.08, ctrl.pinch_threshold, 0.005), "pinch_threshold"),
            (SettingsSlider(col1_x, 380, 200, "Selection Hold Time", 0.2, 1.5, ctrl.pinch_hold_time, 0.1), "pinch_hold_time"),
            (SettingsSlider(col1_x, 440, 200, "Mouse Smoothing", 0.0, 0.8, ctrl.mouse_smoothing, 0.1), "mouse_smoothing"),
        ]
        
        # Key bindings
        self.key_bindings = KeyBindingDisplay(col1_x, 510, ctrl)
        
        # Right column - Gesture visualizer
        viz_x = w - 320
        viz_w = 290
        viz_h = 280
        self.gesture_viz = GestureVisualizer(viz_x, 120, viz_w, viz_h)
    
    def update(self, gesture_state: Any, key: int) -> Optional[str]:
        """Update controls settings."""
        w, h = self.game.width, self.game.height
        ctrl = self.settings.controls
        
        # Handle key binding input
        if key != -1 and key != 255:
            if self.key_bindings.handle_key(key):
                self.settings.save()
        
        # Back button
        if self.back_button.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.is_pinching, w, h, gesture_state.timestamp
        ):
            self.settings.save()
            return "main"
        
        # Input mode selector
        if self.input_mode_selector.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.pinch_started, w, h, gesture_state.timestamp
        ):
            ctrl.input_mode = self.input_mode_selector.current
            self._apply_input_mode()
        
        # Sliders
        for slider, attr_name in self.sliders:
            if slider.update(
                gesture_state.cursor_x, gesture_state.cursor_y,
                gesture_state.is_pinching, w, h, gesture_state.timestamp
            ):
                setattr(ctrl, attr_name, slider.value)
                self._apply_gesture_settings()
        
        # Key binding clicks
        self.key_bindings.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.pinch_started, w, h
        )
        
        # Update gesture visualizer
        self.gesture_viz.update(gesture_state)
        
        return None
    
    def _apply_input_mode(self):
        """Apply the current input mode setting."""
        if hasattr(self.game, 'set_input_mode'):
            self.game.set_input_mode(self.settings.controls.input_mode)
    
    def _apply_gesture_settings(self):
        """Apply gesture sensitivity settings to the engine."""
        if hasattr(self.game, 'engine') and hasattr(self.game.engine, 'config'):
            ctrl = self.settings.controls
            config = self.game.engine.config
            
            # Update gesture engine config
            config.SWIPE_MIN_DISTANCE = ctrl.swipe_min_distance
            config.SWIPE_MIN_VELOCITY = ctrl.swipe_min_velocity
            config.PINCH_THRESHOLD = ctrl.pinch_threshold
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render controls settings."""
        # Background
        frame[:] = (25, 25, 30)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = self.game.width
        
        # Title
        title = "CONTROLS"
        (tw, _), _ = cv2.getTextSize(title, font, 1.0, 2)
        cv2.putText(frame, title, ((w - tw) // 2, 60), font, 1.0, (200, 200, 220), 2)
        
        # Section: Input Mode
        cv2.putText(frame, "INPUT SETTINGS", (50, 110), font, 0.5, (150, 180, 200), 1)
        self.input_mode_selector.draw(frame)
        
        # Section: Sensitivity
        cv2.putText(frame, "GESTURE SENSITIVITY", (50, 185), font, 0.5, (150, 180, 200), 1)
        for slider, _ in self.sliders:
            slider.draw(frame)
        
        # Section: Key bindings
        self.key_bindings.draw(frame)
        
        # Gesture visualizer
        cv2.putText(frame, "LIVE GESTURE PREVIEW", (w - 320, 105), font, 0.5, (150, 180, 200), 1)
        self.gesture_viz.draw(frame, gesture_state, gesture_state.timestamp)
        
        # Help text
        help_text = "Adjust sliders by pinching and dragging"
        cv2.putText(frame, help_text, (w - 320, 420), font, 0.35, (120, 120, 140), 1)
        
        # Back button
        self.back_button.draw(frame, gesture_state.timestamp)
        
        # Draw cursor
        from gesture_engine import draw_cursor
        draw_cursor(frame, gesture_state)


# =====================================================
# GRAPHICS SETTINGS
# =====================================================

class GraphicsSettingsMenu(SettingsSceneBase):
    """Graphics settings - resolution and display options."""
    
    def __init__(self, game: Any):
        super().__init__(game)
        
        self.back_button = None
        self.resolution_selector = None
        self.fullscreen_toggle = None
        self.pip_toggle = None
        self.apply_button = None
        
        self.pending_changes = False
        
        self._create_ui()
    
    def _create_ui(self):
        """Create UI elements."""
        w, h = self.game.width, self.game.height
        gfx = self.settings.graphics
        
        center_x = w // 2 - 100
        
        # Resolution selector
        self.resolution_selector = ResolutionSelector(center_x, 180, gfx.resolution)
        
        # Toggles
        self.fullscreen_toggle = SettingsToggle(center_x, 260, "Fullscreen", gfx.fullscreen)
        self.pip_toggle = SettingsToggle(center_x, 320, "Show Camera PIP", gfx.show_pip)
        
        # Buttons
        self.apply_button = SettingsButton(center_x, 400, 150, 45, "Apply Changes", (60, 80, 60))
        self.back_button = SettingsButton(30, h - 70, 120, 40, "< Back")
    
    def update(self, gesture_state: Any, key: int) -> Optional[str]:
        """Update graphics settings."""
        w, h = self.game.width, self.game.height
        gfx = self.settings.graphics
        
        # Back button
        if self.back_button.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.is_pinching, w, h, gesture_state.timestamp
        ):
            self.settings.save()
            return "main"
        
        # Resolution
        if self.resolution_selector.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.pinch_started, w, h, gesture_state.timestamp
        ):
            gfx.resolution = self.resolution_selector.current
            self.pending_changes = True
        
        # Fullscreen toggle
        if self.fullscreen_toggle.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.is_pinching, gesture_state.pinch_started,
            w, h, gesture_state.timestamp
        ):
            gfx.fullscreen = self.fullscreen_toggle.value
            self.pending_changes = True
        
        # PIP toggle
        if self.pip_toggle.update(
            gesture_state.cursor_x, gesture_state.cursor_y,
            gesture_state.is_pinching, gesture_state.pinch_started,
            w, h, gesture_state.timestamp
        ):
            gfx.show_pip = self.pip_toggle.value
            self.game.show_pip = gfx.show_pip
        
        # Apply button
        if self.pending_changes:
            if self.apply_button.update(
                gesture_state.cursor_x, gesture_state.cursor_y,
                gesture_state.is_pinching, w, h, gesture_state.timestamp
            ):
                self._apply_graphics_changes()
                self.pending_changes = False
        
        return None
    
    def _apply_graphics_changes(self):
        """Apply graphics changes to the game."""
        gfx = self.settings.graphics
        
        # Update game dimensions
        self.game.width, self.game.height = gfx.resolution
        
        # Apply fullscreen
        if gfx.fullscreen:
            cv2.setWindowProperty(self.game.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(self.game.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.game.title, gfx.resolution[0], gfx.resolution[1])
        
        # Save settings
        self.settings.save()
        
        # Recreate UI with new dimensions
        self._create_ui()
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render graphics settings."""
        # Background
        frame[:] = (25, 25, 30)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        w = self.game.width
        
        # Title
        title = "GRAPHICS"
        (tw, _), _ = cv2.getTextSize(title, font, 1.0, 2)
        cv2.putText(frame, title, ((w - tw) // 2, 60), font, 1.0, (200, 200, 220), 2)
        
        # Section header
        cv2.putText(frame, "DISPLAY SETTINGS", (w // 2 - 100, 155), font, 0.5, (150, 180, 200), 1)
        
        # Resolution
        self.resolution_selector.draw(frame)
        
        # Toggles
        self.fullscreen_toggle.draw(frame)
        self.pip_toggle.draw(frame)
        
        # Apply button (only if changes pending)
        if self.pending_changes:
            self.apply_button.draw(frame, gesture_state.timestamp)
            
            cv2.putText(frame, "* Changes pending", (w // 2 - 60, 470), 
                       font, 0.4, (200, 200, 100), 1)
        
        # Current display info
        info_y = 520
        cv2.putText(frame, f"Current: {self.game.width}x{self.game.height}", 
                   (w // 2 - 80, info_y), font, 0.4, (120, 120, 140), 1)
        
        # Back button
        self.back_button.draw(frame, gesture_state.timestamp)
        
        # Draw cursor
        from gesture_engine import draw_cursor
        draw_cursor(frame, gesture_state)


# =====================================================
# SETTINGS MANAGER
# =====================================================

class SettingsManager:
    """
    Manages settings menu state and scene transitions.
    Integrates with the main game loop.
    """
    
    def __init__(self, game: Any):
        self.game = game
        self.is_open = False
        
        # Load or create settings
        if not hasattr(game, 'settings'):
            game.settings = GameSettings.load()
        
        self.settings = game.settings
        
        # Settings scenes
        self.scenes = {
            'main': SettingsMainMenu(game),
            'controls': ControlsSettingsMenu(game),
            'graphics': GraphicsSettingsMenu(game),
        }
        self.current_scene = 'main'
    
    def open(self):
        """Open the settings menu."""
        self.is_open = True
        self.current_scene = 'main'
        
        # Recreate scenes to ensure correct dimensions
        self.scenes = {
            'main': SettingsMainMenu(self.game),
            'controls': ControlsSettingsMenu(self.game),
            'graphics': GraphicsSettingsMenu(self.game),
        }
    
    def close(self):
        """Close the settings menu."""
        self.is_open = False
        self.settings.save()
    
    def toggle(self):
        """Toggle settings menu open/closed."""
        if self.is_open:
            self.close()
        else:
            self.open()
    
    def update(self, gesture_state: Any, key: int) -> Optional[str]:
        """
        Update the settings menu.
        
        Returns:
            'quit' if user wants to quit, None otherwise
        """
        if not self.is_open:
            return None
        
        # ESC closes settings or goes back
        if key == 27:  # ESC
            if self.current_scene == 'main':
                self.close()
            else:
                self.current_scene = 'main'
            return None
        
        # Update current scene
        scene = self.scenes[self.current_scene]
        result = scene.update(gesture_state, key)
        
        if result == 'resume':
            self.close()
        elif result == 'quit':
            return 'quit'
        elif result in self.scenes:
            self.current_scene = result
        
        return None
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        """Render the settings menu over the game frame."""
        if not self.is_open:
            return
        
        scene = self.scenes[self.current_scene]
        scene.render(frame, gesture_state)
