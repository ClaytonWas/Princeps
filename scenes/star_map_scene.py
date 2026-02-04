"""
Star Map Scene
==============
Game scene for the star map navigation.

Features:
- Pinch and hold to select destination
- Visual path preview
- Travel animation
- Selection progress indicator
"""

import cv2
import numpy as np
from typing import Optional, Any

from core.scene_manager import NavigableScene, TransitionDirection
from systems.star_map import (
    GalaxyGenerator, StarMapRenderer, StarMapState, StarMapController, PathFinder
)
from core.gesture_engine import GestureState, draw_cursor


class StarMapScene(NavigableScene):
    """
    Interactive star map scene.
    
    Controls:
    - Move cursor over stars to highlight them
    - Pinch and hold on a star to select it as destination
    - Release to start traveling (if valid path exists)
    - Swipe left/right to switch to other scenes
    """
    
    def __init__(self, seed: int = 12345, num_stars: int = 12):
        super().__init__()
        
        self.seed = seed
        self.num_stars = num_stars
        
        # Will be initialized on_enter
        self.state: Optional[StarMapState] = None
        self.renderer: Optional[StarMapRenderer] = None
        self.controller: Optional[StarMapController] = None
        
        # UI state
        self.message = ""
        self.message_time = 0.0
        self.message_duration = 2.0
    
    def on_enter(self, from_scene: Optional[str] = None, direction: TransitionDirection = TransitionDirection.NONE):
        """Initialize or restore star map."""
        # Check if we already have a galaxy in shared data
        if 'galaxy_seed' in self.shared_data and self.shared_data['galaxy_seed'] == self.seed:
            # Restore existing state
            self.state = self.shared_data.get('star_map_state')
        
        if self.state is None:
            # Generate new galaxy
            self._generate_galaxy()
            self.shared_data['galaxy_seed'] = self.seed
            self.shared_data['star_map_state'] = self.state
        
        # Always recreate renderer and controller with current dimensions
        width = self.scene_manager.width if self.scene_manager else 1280
        height = self.scene_manager.height if self.scene_manager else 720
        
        self.renderer = StarMapRenderer(width, height)
        self.controller = StarMapController(self.state, width, height)
        
        self._show_message("STAR MAP - Pinch to select destination")
    
    def _generate_galaxy(self):
        """Generate a new galaxy."""
        generator = GalaxyGenerator(self.seed)
        stars = generator.generate(
            num_stars=self.num_stars,
            min_connections=2,
            max_connections=4,
            min_distance=0.15
        )
        
        self.state = StarMapState(stars=stars, current_star_id=0)
    
    def regenerate(self, new_seed: int):
        """Regenerate the galaxy with a new seed."""
        self.seed = new_seed
        self._generate_galaxy()
        self.shared_data['galaxy_seed'] = self.seed
        self.shared_data['star_map_state'] = self.state
        
        # Recreate controller
        if self.scene_manager:
            self.controller = StarMapController(
                self.state, 
                self.scene_manager.width, 
                self.scene_manager.height
            )
        
        self._show_message(f"New galaxy generated (seed: {new_seed})")
    
    def _show_message(self, text: str):
        """Show a temporary message."""
        self.message = text
        self.message_time = 0.0
    
    def can_navigate_left(self) -> bool:
        """Allow navigation only when not traveling."""
        return not self.state.is_traveling if self.state else True
    
    def can_navigate_right(self) -> bool:
        """Allow navigation only when not traveling."""
        return not self.state.is_traveling if self.state else True
    
    def get_indicator_label(self) -> str:
        return "STAR MAP"
    
    def update(self, gesture_state: GestureState, delta_time: float) -> Optional[str]:
        """Update star map logic."""
        if not self.state or not self.controller:
            return None
        
        # Update message timer
        if self.message:
            self.message_time += delta_time
            if self.message_time >= self.message_duration:
                self.message = ""
        
        # Update controller
        event = self.controller.update(
            cursor_x=gesture_state.cursor_x,
            cursor_y=gesture_state.cursor_y,
            is_pinching=gesture_state.is_pinching,
            pinch_started=gesture_state.pinch_started,
            pinch_ended=gesture_state.pinch_ended,
            timestamp=gesture_state.timestamp,
            delta_time=delta_time
        )
        
        # Handle events
        if event == "travel_started":
            if self.state.travel_path and len(self.state.travel_path) > 1:
                dest_star = next((s for s in self.state.stars if s.id == self.state.travel_path[-1]), None)
                if dest_star:
                    self._show_message(f"Traveling to {dest_star.name}...")
        
        elif event == "arrived":
            current_star = next((s for s in self.state.stars if s.id == self.state.current_star_id), None)
            if current_star:
                self._show_message(f"Arrived at {current_star.name}")
        
        return None
    
    def render(self, frame: np.ndarray, gesture_state: GestureState):
        """Render the star map."""
        if not self.state or not self.renderer:
            frame[:] = (20, 20, 30)
            return
        
        # Render star map
        self.renderer.render(
            frame=frame,
            stars=self.state.stars,
            current_star_id=self.state.current_star_id,
            selected_star_id=self.state.selected_star_id,
            hover_star_id=self.state.hover_star_id,
            travel_path=self.state.travel_path,
            travel_progress=self.state.travel_progress
        )
        
        # Draw selection progress ring if holding on a star
        if self.controller and gesture_state.is_pinching and self.state.hover_star_id is not None:
            progress = self.controller.get_selection_progress(gesture_state.timestamp)
            if progress > 0:
                self._draw_selection_ring(frame, gesture_state, progress)
        
        # Draw UI overlay
        self._render_ui(frame, gesture_state)
        
        # Draw cursor
        draw_cursor(frame, gesture_state, size=8)
    
    def _draw_selection_ring(self, frame: np.ndarray, gesture_state: GestureState, progress: float):
        """Draw a progress ring around the hovered star."""
        if self.state.hover_star_id is None:
            return
        
        hover_star = next((s for s in self.state.stars if s.id == self.state.hover_star_id), None)
        if not hover_star:
            return
        
        width = self.scene_manager.width if self.scene_manager else 1280
        height = self.scene_manager.height if self.scene_manager else 720
        
        px, py = hover_star.pixel_pos(width, height)
        radius = hover_star.size + 20
        
        # Draw progress arc
        start_angle = -90
        end_angle = start_angle + int(360 * progress)
        
        # Background ring
        cv2.ellipse(frame, (px, py), (radius, radius), 0, 0, 360, (50, 50, 70), 2)
        
        # Progress ring
        color = (100, 200, 255) if progress < 1.0 else (100, 255, 150)
        cv2.ellipse(frame, (px, py), (radius, radius), 0, start_angle, end_angle, color, 3)
        
        # "HOLD" text
        if progress < 1.0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "HOLD"
            (tw, th), _ = cv2.getTextSize(text, font, 0.4, 1)
            cv2.putText(frame, text, (px - tw // 2, py + radius + 20), font, 0.4, (150, 200, 255), 1)
    
    def _render_ui(self, frame: np.ndarray, gesture_state: GestureState):
        """Render UI overlay elements."""
        width = self.scene_manager.width if self.scene_manager else 1280
        height = self.scene_manager.height if self.scene_manager else 720
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Current location panel (top left)
        current_star = next((s for s in self.state.stars if s.id == self.state.current_star_id), None)
        if current_star:
            panel_text = f"Location: {current_star.name}"
            cv2.rectangle(frame, (10, 10), (300, 45), (20, 20, 35), -1)
            cv2.rectangle(frame, (10, 10), (300, 45), (60, 60, 80), 1)
            cv2.putText(frame, panel_text, (20, 33), font, 0.55, (200, 220, 200), 1)
        
        # Travel status (top right)
        if self.state.is_traveling and self.state.travel_path:
            dest_star = next((s for s in self.state.stars if s.id == self.state.travel_path[-1]), None)
            if dest_star:
                jumps_remaining = len(self.state.travel_path) - 1
                progress_pct = int(self.state.travel_progress * 100)
                
                status_text = f"Jumping to {dest_star.name} ({jumps_remaining} jumps)"
                
                # Panel
                cv2.rectangle(frame, (width - 350, 10), (width - 10, 70), (20, 30, 40), -1)
                cv2.rectangle(frame, (width - 350, 10), (width - 10, 70), (80, 120, 150), 1)
                
                cv2.putText(frame, status_text, (width - 340, 33), font, 0.45, (150, 200, 255), 1)
                
                # Progress bar
                bar_x = width - 340
                bar_y = 45
                bar_w = 320
                bar_h = 15
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 60), -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * self.state.travel_progress), bar_y + bar_h), 
                             (100, 200, 255), -1)
        
        # Message display (center)
        if self.message:
            alpha = 1.0 - (self.message_time / self.message_duration) * 0.5
            
            (tw, th), _ = cv2.getTextSize(self.message, font, 0.6, 2)
            tx = (width - tw) // 2
            ty = height - 80
            
            # Background
            cv2.rectangle(frame, (tx - 10, ty - th - 5), (tx + tw + 10, ty + 10), (30, 30, 45), -1)
            
            color = tuple(int(c * alpha) for c in (220, 220, 180))
            cv2.putText(frame, self.message, (tx, ty), font, 0.6, color, 1)
        
        # Hovered star info
        if self.state.hover_star_id is not None and not self.state.is_traveling:
            hover_star = next((s for s in self.state.stars if s.id == self.state.hover_star_id), None)
            if hover_star and hover_star.id != self.state.current_star_id:
                # Calculate path info
                path = PathFinder.find_path(self.state.stars, self.state.current_star_id, hover_star.id)
                
                if path:
                    jumps = len(path) - 1
                    info_text = f"{hover_star.name} - {hover_star.star_type.display_name} ({jumps} jump{'s' if jumps > 1 else ''})"
                    
                    (tw, th), _ = cv2.getTextSize(info_text, font, 0.5, 1)
                    tx = (width - tw) // 2
                    ty = 80
                    
                    cv2.rectangle(frame, (tx - 8, ty - th - 5), (tx + tw + 8, ty + 8), (30, 25, 40), -1)
                    cv2.putText(frame, info_text, (tx, ty), font, 0.5, (180, 180, 200), 1)
        
        # Controls hint (bottom left)
        hints = "Pinch + Hold: Travel  |  Swipe: Switch View"
        cv2.putText(frame, hints, (15, height - 55), font, 0.4, (100, 100, 120), 1)


# =====================
# PLACEHOLDER SCENES (for testing navigation)
# =====================

class ShipScene(NavigableScene):
    """
    Ship interior hub scene.
    
    This is the main hub where players manage their ship and crew.
    Demonstrates how to integrate the ship_systems module.
    """
    
    def __init__(self):
        super().__init__()
        self.ship_state = None
        
        # Interaction state
        self.hovered_station = None
        self.selected_station = None
        self.selected_crew = None
        
        # UI state
        self.show_crew_panel = False
        self.hover_radius = 50
    
    def on_enter(self, from_scene=None, direction=None):
        """Initialize or restore ship state."""
        from systems.ship_systems import ShipStateManager
        
        # Get or create ship state from shared data
        self.ship_state = ShipStateManager.get_or_create(self.shared_data, seed=42)
    
    def get_indicator_label(self) -> str:
        return "SHIP"
    
    def update(self, gesture_state: Any, delta_time: float) -> Optional[str]:
        if not self.ship_state:
            return None
        
        width = self.scene_manager.width if self.scene_manager else 1280
        height = self.scene_manager.height if self.scene_manager else 720
        
        # Update hovered station
        self.hovered_station = None
        if gesture_state.cursor_x is not None:
            px = int(gesture_state.cursor_x * width)
            py = int(gesture_state.cursor_y * height)
            
            for station in self.ship_state.stations:
                sx = int(station.x * width)
                sy = int(station.y * height)
                dist = ((px - sx) ** 2 + (py - sy) ** 2) ** 0.5
                
                if dist < self.hover_radius:
                    self.hovered_station = station.id
                    break
        
        # Handle station selection on pinch
        if gesture_state.pinch_started and self.hovered_station:
            if self.selected_station == self.hovered_station:
                self.selected_station = None  # Deselect
            else:
                self.selected_station = self.hovered_station
        
        return None
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        from systems.ship_systems import ResourceBar, CrewPortrait, StationMarker
        
        frame[:] = (25, 22, 18)
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if not self.ship_state:
            cv2.putText(frame, "Loading ship...", (w//2 - 80, h//2), font, 0.6, (150, 150, 150), 1)
            return
        
        # Draw title
        title = f"SHIP - Day {self.ship_state.day}"
        (tw, _), _ = cv2.getTextSize(title, font, 0.8, 2)
        cv2.putText(frame, title, ((w - tw) // 2, 35), font, 0.8, (180, 160, 140), 2)
        
        # Draw resource bars (top right)
        res = self.ship_state.resources
        bar_x = w - 180
        ResourceBar.draw(frame, bar_x, 60, 160, res.fuel, 100, "Fuel", (100, 180, 200))
        ResourceBar.draw(frame, bar_x, 85, 160, res.food, 100, "Food", (100, 200, 100))
        ResourceBar.draw(frame, bar_x, 110, 160, res.hull, 100, "Hull", (200, 150, 100))
        
        # Credits
        cv2.putText(frame, f"Credits: {res.credits}", (bar_x, 145), font, 0.45, (220, 200, 100), 1)
        
        # Draw stations
        for station in self.ship_state.stations:
            is_hovered = station.id == self.hovered_station
            is_selected = station.id == self.selected_station
            StationMarker.draw(frame, station, w, h, is_hovered, is_selected)
        
        # Draw crew panel (left side)
        cv2.rectangle(frame, (10, 55), (145, 55 + len(self.ship_state.crew) * 90 + 15), (35, 32, 28), -1)
        cv2.putText(frame, "CREW", (20, 75), font, 0.45, (150, 140, 130), 1)
        
        for i, crew in enumerate(self.ship_state.crew):
            is_selected = self.selected_crew == crew.id
            CrewPortrait.draw(frame, 15, 85 + i * 90, crew, is_selected)
        
        # Selected station info
        if self.selected_station:
            station = next((s for s in self.ship_state.stations if s.id == self.selected_station), None)
            if station:
                panel_x, panel_y = w // 2 - 150, h - 120
                cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 300, panel_y + 80), (40, 38, 35), -1)
                cv2.rectangle(frame, (panel_x, panel_y), (panel_x + 300, panel_y + 80), (100, 90, 80), 1)
                
                cv2.putText(frame, station.name, (panel_x + 10, panel_y + 25), font, 0.6, (220, 200, 180), 1)
                cv2.putText(frame, station.description, (panel_x + 10, panel_y + 50), font, 0.4, (150, 140, 130), 1)
                
                status = "Operational" if station.is_functional else "DAMAGED"
                status_color = (100, 180, 100) if station.is_functional else (100, 100, 200)
                cv2.putText(frame, status, (panel_x + 10, panel_y + 70), font, 0.35, status_color, 1)
        
        draw_cursor(frame, gesture_state)


class CommsScene(NavigableScene):
    """Placeholder scene for communications."""
    
    def __init__(self):
        super().__init__()
    
    def get_indicator_label(self) -> str:
        return "COMMS"
    
    def update(self, gesture_state: Any, delta_time: float) -> Optional[str]:
        return None
    
    def render(self, frame: np.ndarray, gesture_state: Any):
        frame[:] = (20, 25, 35)
        
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Title
        title = "COMMUNICATIONS"
        (tw, _), _ = cv2.getTextSize(title, font, 1.2, 2)
        cv2.putText(frame, title, ((w - tw) // 2, 80), font, 1.2, (140, 160, 200), 2)
        
        # Placeholder message
        msg = "[ Communications scene coming soon ]"
        (tw, _), _ = cv2.getTextSize(msg, font, 0.6, 1)
        cv2.putText(frame, msg, ((w - tw) // 2, h // 2), font, 0.6, (100, 120, 150), 1)
        
        draw_cursor(frame, gesture_state)


# =====================
# DEMO / TEST
# =====================
if __name__ == "__main__":
    """Test the star map with scene navigation."""
    
    import time
    from gesture_engine import GestureEngine
    from scene_manager import SceneCarouselManager
    
    print("STAR MAP DEMO")
    print("=" * 40)
    print("Controls:")
    print("  - Move hand to hover over stars")
    print("  - Pinch and HOLD to select destination")
    print("  - Swipe left/right to switch scenes")
    print("  - Press 'R' to regenerate galaxy")
    print("  - Press 'D' for debug overlay")
    print("  - Press 'Q' to quit")
    print("=" * 40)
    
    # Initialize
    WIDTH, HEIGHT = 1280, 720
    engine = GestureEngine()
    
    # Create scene manager
    manager = SceneCarouselManager(WIDTH, HEIGHT)
    
    # Create and add scenes
    ship_scene = ShipScene()
    star_map_scene = StarMapScene(seed=42, num_stars=15)
    comms_scene = CommsScene()
    
    # Order: Ship - Star Map - Comms
    manager.add_scene("ship", ship_scene)
    manager.add_scene("starmap", star_map_scene)
    manager.add_scene("comms", comms_scene)
    
    # Start on star map
    manager.set_scene("starmap")
    
    # Debug state
    show_debug = False
    last_time = time.time()
    
    try:
        while engine.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - last_time
            last_time = current_time
            
            # Get gesture state
            state = engine.update()
            
            # Update scene manager
            manager.update(state, delta_time)
            
            # Render
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            manager.render(frame, state)
            
            # PIP camera view
            engine.render_pip(frame, 180, 135, 'bottom-right')
            
            # Debug overlay
            if show_debug:
                cv2.rectangle(frame, (10, 10), (220, 120), (20, 20, 30), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Hand: {state.hand_status}", (20, 35), font, 0.45, (150, 200, 150), 1)
                cv2.putText(frame, f"Pinch: {state.is_pinching}", (20, 55), font, 0.45, (150, 200, 150), 1)
                cv2.putText(frame, f"Scene: {manager.current_scene}", (20, 75), font, 0.45, (150, 200, 150), 1)
                cv2.putText(frame, f"FPS: {1/max(delta_time, 0.001):.0f}", (20, 95), font, 0.45, (150, 200, 150), 1)
            
            # Show frame
            if not engine.show(frame, "Princeps - Star Map"):
                break
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                show_debug = not show_debug
            elif key == ord('r'):
                # Regenerate with random seed
                import random
                new_seed = random.randint(1, 999999)
                star_map_scene.regenerate(new_seed)
                print(f"Regenerated galaxy with seed: {new_seed}")
    
    finally:
        engine.close()
        cv2.destroyAllWindows()
