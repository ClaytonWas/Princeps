"""
Display Module
==============
Pygame-based display wrapper for proper windowing, fullscreen, and scaling.
Replaces cv2.imshow() with proper game window handling.
"""

import pygame
import numpy as np
from typing import Tuple, Optional


class GameDisplay:
    """
    Pygame-based display for the game.
    
    Features:
    - True fullscreen support
    - Resolution scaling (render at lower res, display at screen res)
    - Proper window events (resize, minimize, close)
    - VSync support
    """
    
    def __init__(self, 
                 render_width: int = 1280, 
                 render_height: int = 720,
                 title: str = "Princeps",
                 fullscreen: bool = False,
                 vsync: bool = True):
        """
        Initialize the display.
        
        Args:
            render_width: Internal rendering resolution width
            render_height: Internal rendering resolution height
            title: Window title
            fullscreen: Start in fullscreen mode
            vsync: Enable vertical sync
        """
        pygame.init()
        pygame.display.set_caption(title)
        
        self.render_width = render_width
        self.render_height = render_height
        self.title = title
        self.vsync = vsync
        self._fullscreen = fullscreen
        self._running = True
        
        # Get screen info
        info = pygame.display.Info()
        self.screen_width = info.current_w
        self.screen_height = info.current_h
        
        # Create window
        self._create_window(fullscreen)
        
        # Clock for frame timing
        self.clock = pygame.time.Clock()
        self.target_fps = 60  # 0 = uncapped
        
        # For tracking actual FPS
        self.actual_fps = 0.0
    
    def set_target_fps(self, fps: int):
        """Set target FPS. Use 0 for uncapped."""
        self.target_fps = fps
        
    def _create_window(self, fullscreen: bool):
        """Create or recreate the window."""
        if fullscreen:
            # True fullscreen at native resolution
            flags = pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF
            if self.vsync:
                flags |= pygame.SCALED
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height), 
                flags
            )
            self.display_width = self.screen_width
            self.display_height = self.screen_height
        else:
            # Windowed mode at render resolution
            flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            self.screen = pygame.display.set_mode(
                (self.render_width, self.render_height), 
                flags
            )
            self.display_width = self.render_width
            self.display_height = self.render_height
        
        self._fullscreen = fullscreen
        
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        self._create_window(not self._fullscreen)
        return self._fullscreen
    
    def set_fullscreen(self, fullscreen: bool):
        """Set fullscreen mode explicitly."""
        if fullscreen != self._fullscreen:
            self._create_window(fullscreen)
    
    @property
    def is_fullscreen(self) -> bool:
        return self._fullscreen
    
    def get_render_size(self) -> Tuple[int, int]:
        """Get the size to render at (internal resolution)."""
        return (self.render_width, self.render_height)
    
    def get_display_size(self) -> Tuple[int, int]:
        """Get the actual display size."""
        return (self.display_width, self.display_height)
    
    def resize(self, width: int, height: int):
        """Resize the window and render buffer."""
        self.render_width = width
        self.render_height = height
        self.display_width = width
        self.display_height = height
        if not self._fullscreen:
            self.screen = pygame.display.set_mode(
                (width, height),
                pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE
            )
    
    def process_events(self) -> dict:
        """
        Process pygame events and return relevant game events.
        
        Returns:
            Dictionary with:
            - 'quit': True if window should close
            - 'keys': Set of pressed key codes
            - 'key_down': List of keys just pressed this frame
            - 'mouse_pos': (x, y) normalized 0-1
            - 'mouse_buttons': (left, middle, right) booleans
            - 'mouse_clicked': True if left button just clicked
            - 'resized': New size if window was resized, None otherwise
        """
        events = {
            'quit': False,
            'keys': set(),
            'key_down': [],
            'mouse_pos': (0, 0),
            'mouse_buttons': (False, False, False),
            'mouse_clicked': False,
            'resized': None,
        }
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events['quit'] = True
                self._running = False
                
            elif event.type == pygame.KEYDOWN:
                events['key_down'].append(event.key)
                
                # F11 for fullscreen toggle (common convention)
                if event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                
                # Prevent ESC from doing anything except what game handles
                # (Some window managers try to close fullscreen on ESC)
                if event.key == pygame.K_ESCAPE:
                    continue  # Don't let it propagate
                    
            elif event.type == pygame.VIDEORESIZE:
                # Window was resized by user dragging - just track the new size
                # Don't recreate surface here as pygame handles it automatically
                self.display_width = event.w
                self.display_height = event.h
                self.render_width = event.w
                self.render_height = event.h
                events['resized'] = (event.w, event.h)
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    events['mouse_clicked'] = True
        
        # Get current key state
        keys = pygame.key.get_pressed()
        for i, pressed in enumerate(keys):
            if pressed:
                events['keys'].add(i)
        
        # Get mouse state
        mouse_pos = pygame.mouse.get_pos()
        # Normalize to 0-1 range based on display size
        events['mouse_pos'] = (
            mouse_pos[0] / self.display_width if self.display_width > 0 else 0,
            mouse_pos[1] / self.display_height if self.display_height > 0 else 0
        )
        events['mouse_buttons'] = pygame.mouse.get_pressed()
        
        return events
    
    def show_frame(self, frame: np.ndarray):
        """
        Display an OpenCV frame (BGR numpy array).
        
        The frame is automatically scaled to fit the display.
        
        Args:
            frame: BGR numpy array from OpenCV
        """
        # OpenCV is BGR, Pygame expects RGB
        # Also need to swap axes (OpenCV is HxWxC, pygame wants WxHxC)
        frame_rgb = frame[:, :, ::-1]  # BGR to RGB
        
        # Create pygame surface from the frame
        # pygame.surfarray expects (width, height, channels) but numpy is (height, width, channels)
        # So we need to transpose/swapaxes
        surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Scale to display size if needed
        if (surface.get_width() != self.display_width or 
            surface.get_height() != self.display_height):
            surface = pygame.transform.scale(surface, (self.display_width, self.display_height))
        
        # Draw to screen
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()
        
        # Cap framerate and track actual FPS
        # 0 = uncapped, any positive value = cap at that FPS
        if self.target_fps > 0:
            self.clock.tick(self.target_fps)
        else:
            self.clock.tick()  # Uncapped - just track time
        self.actual_fps = self.clock.get_fps()
    
    def get_fps(self) -> float:
        """Get the actual frames per second."""
        return self.actual_fps
    
    @property
    def running(self) -> bool:
        """Check if the display is still running (not closed)."""
        return self._running
    
    def close(self):
        """Close the display and clean up pygame."""
        self._running = False
        pygame.quit()


# Key constants for easy access
class Keys:
    """Pygame key constants for convenience."""
    ESCAPE = pygame.K_ESCAPE
    RETURN = pygame.K_RETURN
    SPACE = pygame.K_SPACE
    F11 = pygame.K_F11
    D = pygame.K_d
    Q = pygame.K_q
    
    UP = pygame.K_UP
    DOWN = pygame.K_DOWN
    LEFT = pygame.K_LEFT
    RIGHT = pygame.K_RIGHT
    
    W = pygame.K_w
    A = pygame.K_a
    S = pygame.K_s
