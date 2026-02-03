"""
Princeps MVP
============
Minimal game framework with gesture controls.

Core features:
- Scene-based game loop
- Pinch to select
- Swipe left/right for yes/no
- Nameplate capture
- Debug UI (press D)
"""

import cv2
import numpy as np
import time
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from datetime import datetime

from gesture_engine import GestureEngine, GestureState, GestureConfig, draw_cursor


# =====================
# SCENE BASE
# =====================
class Scene(ABC):
    """Base class for game scenes."""
    
    def __init__(self):
        self.game: Optional['Game'] = None
        self.shared_data: Dict[str, Any] = {}
    
    def on_enter(self, previous_scene: Optional[str] = None):
        pass
    
    def on_exit(self, next_scene: Optional[str] = None):
        pass
    
    @abstractmethod
    def update(self, state: GestureState) -> Optional[str]:
        """Return scene name to transition, or None to stay."""
        pass
    
    @abstractmethod
    def render(self, frame: np.ndarray, state: GestureState):
        """Render scene to frame."""
        pass


# =====================
# BUTTON
# =====================
class Button:
    """Hold-to-activate button."""
    
    def __init__(self, x: int, y: int, w: int, h: int, text: str, 
                 color=(60, 60, 70), active_color=(60, 180, 60)):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.color = color
        self.active_color = active_color
        self.hover_time = 0
        self.activation_time = 0.5
    
    def update(self, state: GestureState, frame_w: int, frame_h: int) -> bool:
        if state.cursor_x is None:
            self.hover_time = 0
            return False
        
        px = int(state.cursor_x * frame_w)
        py = int(state.cursor_y * frame_h)
        
        in_btn = self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h
        
        if in_btn and state.is_pinching:
            if self.hover_time == 0:
                self.hover_time = state.timestamp
            elif state.timestamp - self.hover_time >= self.activation_time:
                self.hover_time = state.timestamp + 0.5
                return True
        else:
            if not state.is_pinching:
                self.hover_time = 0
        return False
    
    def draw(self, frame: np.ndarray, t: float):
        progress = 0
        if self.hover_time > 0:
            progress = min((t - self.hover_time) / self.activation_time, 1.0)
        
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
        
        if progress > 0:
            pw = int(self.w * progress)
            cv2.rectangle(frame, (self.x, self.y), (self.x + pw, self.y + self.h), self.active_color, -1)
        
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (150, 150, 150), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(self.text, font, 0.7, 2)
        cv2.putText(frame, self.text, (self.x + (self.w - tw) // 2, self.y + (self.h + th) // 2),
                   font, 0.7, (255, 255, 255), 2)


# =====================
# DRAWING CANVAS
# =====================
class DrawingCanvas:
    """Pinch-to-draw canvas."""
    
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.canvas = np.ones((h, w, 3), dtype=np.uint8) * 235
        self.last_pos = None
        self.has_content = False
    
    def clear(self):
        self.canvas[:] = 235
        self.last_pos = None
        self.has_content = False
    
    def draw(self, state: GestureState, frame_w: int, frame_h: int):
        if not state.is_pinching or state.cursor_x is None:
            self.last_pos = None
            return
        
        px = int(state.cursor_x * frame_w)
        py = int(state.cursor_y * frame_h)
        
        if not (self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h):
            self.last_pos = None
            return
        
        lx, ly = px - self.x, py - self.y
        
        if self.last_pos:
            cv2.line(self.canvas, self.last_pos, (lx, ly), (30, 30, 30), 5)
        else:
            cv2.circle(self.canvas, (lx, ly), 2, (30, 30, 30), -1)
        
        self.last_pos = (lx, ly)
        self.has_content = True
    
    def render(self, frame: np.ndarray):
        frame[self.y:self.y + self.h, self.x:self.x + self.w] = self.canvas
    
    def save(self, path: str):
        cv2.imwrite(path, self.canvas)
        return path
    
    def get_image(self):
        return self.canvas.copy()


# =====================
# DEBUG UI
# =====================
class DebugUI:
    """Debug overlay - toggle with D key."""
    
    def __init__(self):
        self.enabled = False
        self.events = []
    
    def toggle(self):
        self.enabled = not self.enabled
    
    def log(self, msg: str, t: float):
        self.events.append((t, msg))
        if len(self.events) > 8:
            self.events.pop(0)
    
    def update(self, state: GestureState):
        if state.pinch_started:
            self.log(f"PINCH ({state.pinch_finger})", state.timestamp)
        if state.swipe_left:
            self.log("SWIPE LEFT", state.timestamp)
        if state.swipe_right:
            self.log("SWIPE RIGHT", state.timestamp)
    
    def render(self, frame: np.ndarray, state: GestureState):
        if not self.enabled:
            return
        
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 300), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        y = 30
        cv2.putText(frame, "DEBUG [D]", (20, y), font, 0.5, (0, 255, 255), 1)
        y += 25
        
        # Status
        colors = {'detected': (0, 255, 0), 'predicted': (0, 255, 255), 
                  'persisted': (0, 165, 255), 'lost': (0, 0, 255)}
        cv2.putText(frame, f"Hand: {state.hand_status}", (20, y), font, 0.45, 
                   colors.get(state.hand_status, (128, 128, 128)), 1)
        y += 22
        
        if state.cursor_x:
            cv2.putText(frame, f"Pos: ({state.cursor_x:.2f}, {state.cursor_y:.2f})", (20, y), font, 0.4, (180, 180, 180), 1)
        y += 22
        
        cv2.putText(frame, f"Vel X: {state.velocity_x:+.2f}", (20, y), font, 0.4, (180, 180, 180), 1)
        y += 20
        cv2.putText(frame, f"Vel Y: {state.velocity_y:+.2f}", (20, y), font, 0.4, (180, 180, 180), 1)
        y += 22
        
        pinch_col = (0, 255, 0) if state.is_pinching else (100, 100, 100)
        cv2.putText(frame, f"Pinch: {state.is_pinching}", (20, y), font, 0.45, pinch_col, 1)
        y += 30
        
        cv2.putText(frame, "Events:", (20, y), font, 0.45, (200, 200, 200), 1)
        y += 20
        
        for evt_t, evt_msg in reversed(self.events[-5:]):
            age = state.timestamp - evt_t
            alpha = max(0.3, 1.0 - age / 3.0)
            col = tuple(int(c * alpha) for c in (150, 255, 150))
            cv2.putText(frame, f"  {evt_msg}", (20, y), font, 0.35, col, 1)
            y += 16


# =====================
# GAME
# =====================
class Game:
    """Main game class."""
    
    def __init__(self, width=1280, height=720, title="Princeps"):
        self.width = width
        self.height = height
        self.title = title
        
        self.engine = GestureEngine()
        self.scenes: Dict[str, Scene] = {}
        self.current_scene: Optional[str] = None
        self.shared_data: Dict[str, Any] = {}
        
        self.debug = DebugUI()
        self.show_pip = True
    
    def add_scene(self, name: str, scene: Scene):
        scene.game = self
        scene.shared_data = self.shared_data
        self.scenes[name] = scene
    
    def transition(self, name: str):
        if self.current_scene:
            self.scenes[self.current_scene].on_exit(name)
        prev = self.current_scene
        self.current_scene = name
        self.scenes[name].on_enter(prev)
    
    def run(self, start_scene: str):
        self.transition(start_scene)
        
        try:
            while self.engine.running:
                state = self.engine.update()
                self.debug.update(state)
                
                scene = self.scenes[self.current_scene]
                next_scene = scene.update(state)
                
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                scene.render(frame, state)
                
                if self.show_pip:
                    self.engine.render_pip(frame, 180, 135, 'bottom-right')
                
                self.debug.render(frame, state)
                
                if not self.engine.show(frame, self.title):
                    break
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('d'):
                    self.debug.toggle()
                
                if next_scene:
                    self.transition(next_scene)
        finally:
            self.engine.close()


# =====================
# PREFAB SCENES
# =====================
class MenuScene(Scene):
    """Simple menu with buttons."""
    
    def __init__(self, title: str, options: Dict[str, str]):
        super().__init__()
        self.title = title
        self.options = options
        self.buttons = []
    
    def on_enter(self, prev=None):
        self.buttons = []
        w, h = self.game.width, self.game.height
        btn_w, btn_h = 220, 55
        start_y = h // 3
        
        for i, (text, target) in enumerate(self.options.items()):
            btn = Button((w - btn_w) // 2, start_y + i * 75, btn_w, btn_h, text)
            self.buttons.append((btn, target))
    
    def update(self, state: GestureState) -> Optional[str]:
        for btn, target in self.buttons:
            if btn.update(state, self.game.width, self.game.height):
                return target
        return None
    
    def render(self, frame: np.ndarray, state: GestureState):
        frame[:] = (30, 30, 40)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, _), _ = cv2.getTextSize(self.title, font, 1.4, 3)
        cv2.putText(frame, self.title, ((self.game.width - tw) // 2, 100), font, 1.4, (220, 220, 220), 3)
        
        for btn, _ in self.buttons:
            btn.draw(frame, state.timestamp)
        
        draw_cursor(frame, state)


class NameplateScene(Scene):
    """Capture handwritten nameplate."""
    
    def __init__(self, next_scene: str = 'game'):
        super().__init__()
        self.next_scene = next_scene
        self.canvas = None
        self.done_btn = None
        self.clear_btn = None
    
    def on_enter(self, prev=None):
        w, h = self.game.width, self.game.height
        cw, ch = int(w * 0.6), int(h * 0.3)
        cx, cy = (w - cw) // 2, int(h * 0.28)
        
        self.canvas = DrawingCanvas(cx, cy, cw, ch)
        
        btn_y = cy + ch + 30
        self.done_btn = Button(w // 2 - 110, btn_y, 100, 45, "DONE", active_color=(50, 150, 50))
        self.clear_btn = Button(w // 2 + 10, btn_y, 100, 45, "CLEAR", color=(80, 50, 50))
    
    def update(self, state: GestureState) -> Optional[str]:
        self.canvas.draw(state, self.game.width, self.game.height)
        
        if self.done_btn.update(state, self.game.width, self.game.height):
            if self.canvas.has_content:
                os.makedirs('nameplates', exist_ok=True)
                path = f"nameplates/nameplate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                self.canvas.save(path)
                self.shared_data['nameplate_path'] = path
                self.shared_data['nameplate_image'] = self.canvas.get_image()
                return self.next_scene
        
        if self.clear_btn.update(state, self.game.width, self.game.height):
            self.canvas.clear()
        
        return None
    
    def render(self, frame: np.ndarray, state: GestureState):
        frame[:] = (35, 30, 25)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        title = "WRITE YOUR NAME"
        (tw, _), _ = cv2.getTextSize(title, font, 1.0, 2)
        cv2.putText(frame, title, ((self.game.width - tw) // 2, 70), font, 1.0, (200, 180, 140), 2)
        
        # Canvas border
        cv2.rectangle(frame, (self.canvas.x - 5, self.canvas.y - 5),
                     (self.canvas.x + self.canvas.w + 5, self.canvas.y + self.canvas.h + 5), (80, 70, 60), 3)
        
        self.canvas.render(frame)
        self.done_btn.draw(frame, state.timestamp)
        self.clear_btn.draw(frame, state.timestamp)
        
        draw_cursor(frame, state, size=10)


# =====================
# RUN
# =====================
if __name__ == "__main__":
    print("PRINCEPS MVP")
    print("Controls: Pinch=select, Swipe=yes/no, D=debug")
    
    # Example gameplay scene
    class GameScene(Scene):
        def __init__(self):
            super().__init__()
            self.score = 0
        
        def update(self, state: GestureState) -> Optional[str]:
            if state.swipe_right:
                self.score += 10
            if state.swipe_left:
                self.score -= 10
            return None
        
        def render(self, frame: np.ndarray, state: GestureState):
            frame[:] = (40, 40, 50)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Show nameplate
            img = self.shared_data.get('nameplate_image')
            if img is not None:
                h, w = img.shape[:2]
                scale = 0.4
                resized = cv2.resize(img, (int(w * scale), int(h * scale)))
                rh, rw = resized.shape[:2]
                frame[20:20 + rh, 20:20 + rw] = resized
            
            # Score
            cv2.putText(frame, f"Score: {self.score}", (self.game.width // 2 - 80, 120),
                       font, 1.3, (255, 255, 255), 2)
            
            cv2.putText(frame, "Swipe RIGHT = +10  |  Swipe LEFT = -10", 
                       (self.game.width // 2 - 220, self.game.height - 50),
                       font, 0.6, (140, 140, 140), 1)
            
            draw_cursor(frame, state)
    
    game = Game(title="Princeps")
    game.add_scene('menu', MenuScene("PRINCEPS", {"NEW GAME": "nameplate", "CONTINUE": "game"}))
    game.add_scene('nameplate', NameplateScene(next_scene='game'))
    game.add_scene('game', GameScene())
    game.run('menu')
