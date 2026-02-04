"""
Ship Systems Module
===================
Core game systems for the ship hub.

ARCHITECTURE GUIDE
==================

This module demonstrates how to build game systems that:
1. Have their own state (data)
2. Have their own logic (update)
3. Have their own visuals (render)
4. Integrate with the gesture system

DEVELOPMENT PATTERN
-------------------
For each new system, follow this cycle:

1. DATA FIRST: Define what state you need (dataclass)
2. LOGIC SECOND: Write update() - how state changes
3. VISUALS THIRD: Write render() - how to display it
4. INTEGRATE: Connect to scene via shared_data

This keeps systems modular and testable!
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum, auto
import random


# =====================================================
# STEP 1: DATA STRUCTURES
# =====================================================
# Define your game state as dataclasses.
# These are just data - no logic, no rendering.
# This makes them easy to save/load and reason about.
# =====================================================

class CrewRole(Enum):
    """Roles crew members can have."""
    PILOT = ("Pilot", "Navigation and evasion")
    ENGINEER = ("Engineer", "Ship repairs and upgrades")
    MEDIC = ("Medic", "Crew health and morale")
    GUNNER = ("Gunner", "Weapons and defense")
    SCIENTIST = ("Scientist", "Research and scanning")
    COOK = ("Cook", "Food and morale boost")
    
    def __init__(self, display_name: str, description: str):
        self.display_name = display_name
        self.description = description


class CrewMood(Enum):
    """Crew member emotional states."""
    HAPPY = ("Happy", (100, 200, 100))
    CONTENT = ("Content", (180, 180, 100))
    STRESSED = ("Stressed", (100, 150, 200))
    UNHAPPY = ("Unhappy", (100, 100, 200))
    CRITICAL = ("Critical", (80, 80, 220))
    
    def __init__(self, display_name: str, color: tuple):
        self.display_name = display_name
        self.color = color


@dataclass
class CrewMember:
    """A crew member on your ship."""
    id: int
    name: str
    role: CrewRole
    
    # Stats (0-100)
    health: int = 100
    morale: int = 75
    skill: int = 50
    
    # State
    is_assigned: bool = False
    assigned_to: Optional[str] = None  # Station name
    
    @property
    def mood(self) -> CrewMood:
        """Derive mood from morale."""
        if self.morale >= 80:
            return CrewMood.HAPPY
        elif self.morale >= 60:
            return CrewMood.CONTENT
        elif self.morale >= 40:
            return CrewMood.STRESSED
        elif self.morale >= 20:
            return CrewMood.UNHAPPY
        return CrewMood.CRITICAL
    
    @property
    def effectiveness(self) -> float:
        """How effective is this crew member? (0-1)"""
        health_factor = self.health / 100
        morale_factor = (self.morale + 50) / 150  # Morale has less impact
        skill_factor = self.skill / 100
        return health_factor * morale_factor * skill_factor


@dataclass
class ShipResources:
    """Ship's consumable resources."""
    fuel: int = 100       # 0-100, consumed when traveling
    food: int = 100       # 0-100, consumed over time
    credits: int = 500    # Currency
    hull: int = 100       # Ship health
    
    def can_travel(self, fuel_cost: int) -> bool:
        return self.fuel >= fuel_cost


@dataclass 
class ShipStation:
    """An interactive station on the ship."""
    id: str
    name: str
    description: str
    
    # Screen position (normalized 0-1)
    x: float
    y: float
    
    # Visual
    icon: str  # Simple text icon for now
    size: int = 60
    
    # State
    assigned_crew: Optional[int] = None  # Crew member ID
    is_functional: bool = True
    efficiency: float = 1.0  # 0-1, affected by damage/crew


@dataclass
class ShipState:
    """Complete state of the ship - this is your save game data!"""
    crew: List[CrewMember] = field(default_factory=list)
    resources: ShipResources = field(default_factory=ShipResources)
    stations: List[ShipStation] = field(default_factory=list)
    
    # Events/decisions pending
    pending_events: List[Dict] = field(default_factory=list)
    
    # Time tracking
    day: int = 1
    hour: int = 0


# =====================================================
# STEP 2: GAME LOGIC (Systems)
# =====================================================
# Systems operate on data. They have update() methods
# that take state + inputs and modify state.
# Keep rendering OUT of here - pure logic only.
# =====================================================

class CrewNameGenerator:
    """Generate random crew names."""
    
    FIRST_NAMES = [
        "Alex", "Jordan", "Morgan", "Casey", "Riley", "Quinn",
        "Zara", "Kai", "Nova", "Orion", "Luna", "Sage",
        "Felix", "Maya", "Leo", "Iris", "Ash", "River"
    ]
    
    LAST_NAMES = [
        "Chen", "Okonkwo", "Petrov", "Nakamura", "Santos", "Weber",
        "Kim", "Osei", "Volkov", "Tanaka", "Silva", "Mueller",
        "Park", "Mensah", "Sokolov", "Yamamoto", "Costa", "Fischer"
    ]
    
    @classmethod
    def generate(cls, rng: random.Random) -> str:
        return f"{rng.choice(cls.FIRST_NAMES)} {rng.choice(cls.LAST_NAMES)}"


class ShipInitializer:
    """Creates initial ship state."""
    
    @staticmethod
    def create_default_ship(seed: int = None) -> ShipState:
        """Create a new ship with default configuration."""
        rng = random.Random(seed)
        
        # Create stations
        stations = [
            ShipStation("bridge", "Bridge", "Command center", 0.5, 0.2, "[B]"),
            ShipStation("engine", "Engine Room", "Power and propulsion", 0.2, 0.5, "[E]"),
            ShipStation("medbay", "Medical Bay", "Healing and research", 0.8, 0.5, "[M]"),
            ShipStation("cargo", "Cargo Hold", "Storage and supplies", 0.5, 0.7, "[C]"),
        ]
        
        # Create initial crew
        crew = []
        roles = [CrewRole.PILOT, CrewRole.ENGINEER, CrewRole.MEDIC]
        
        for i, role in enumerate(roles):
            member = CrewMember(
                id=i,
                name=CrewNameGenerator.generate(rng),
                role=role,
                health=rng.randint(80, 100),
                morale=rng.randint(60, 90),
                skill=rng.randint(40, 70)
            )
            crew.append(member)
        
        return ShipState(
            crew=crew,
            stations=stations,
            resources=ShipResources(fuel=80, food=90, credits=500, hull=100)
        )


class TimeSystem:
    """Handles time passing and its effects."""
    
    @staticmethod
    def advance_time(state: ShipState, hours: int = 1):
        """Advance time and apply effects."""
        for _ in range(hours):
            state.hour += 1
            
            if state.hour >= 24:
                state.hour = 0
                state.day += 1
                TimeSystem._daily_update(state)
    
    @staticmethod
    def _daily_update(state: ShipState):
        """Things that happen each day."""
        # Consume food based on crew size
        food_consumption = len(state.crew) * 2
        state.resources.food = max(0, state.resources.food - food_consumption)
        
        # Morale effects
        for crew in state.crew:
            # Food affects morale
            if state.resources.food < 20:
                crew.morale = max(0, crew.morale - 5)
            elif state.resources.food > 50:
                crew.morale = min(100, crew.morale + 1)


class TravelSystem:
    """Handles star travel mechanics."""
    
    @staticmethod
    def calculate_fuel_cost(path_length: int) -> int:
        """Calculate fuel cost for a journey."""
        return path_length * 10  # 10 fuel per jump
    
    @staticmethod
    def can_travel(state: ShipState, path_length: int) -> tuple[bool, str]:
        """Check if travel is possible."""
        fuel_cost = TravelSystem.calculate_fuel_cost(path_length)
        
        if state.resources.fuel < fuel_cost:
            return False, f"Need {fuel_cost} fuel (have {state.resources.fuel})"
        
        if state.resources.hull < 20:
            return False, "Hull too damaged for jump"
        
        # Check for pilot
        has_pilot = any(c.role == CrewRole.PILOT and c.health > 20 for c in state.crew)
        if not has_pilot:
            return False, "No healthy pilot available"
        
        return True, "Ready to jump"
    
    @staticmethod
    def execute_travel(state: ShipState, path_length: int) -> List[str]:
        """Execute travel and return events that occurred."""
        events = []
        
        fuel_cost = TravelSystem.calculate_fuel_cost(path_length)
        state.resources.fuel -= fuel_cost
        events.append(f"Used {fuel_cost} fuel")
        
        # Time passes during travel
        TimeSystem.advance_time(state, hours=path_length * 2)
        events.append(f"{path_length * 2} hours passed")
        
        # Random travel events could happen here
        # ...
        
        return events


# =====================================================
# STEP 3: UI COMPONENTS
# =====================================================
# Reusable visual components that know how to render
# themselves. They take data and draw it.
# =====================================================

class ResourceBar:
    """Renders a resource bar with label."""
    
    @staticmethod
    def draw(frame: np.ndarray, x: int, y: int, width: int, 
             value: int, max_value: int, label: str,
             color: tuple = (100, 180, 100)):
        """Draw a labeled resource bar."""
        height = 20
        
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 40, 50), -1)
        
        # Fill
        fill_width = int((value / max_value) * width)
        
        # Color shifts to red when low
        if value / max_value < 0.25:
            color = (80, 80, 200)
        elif value / max_value < 0.5:
            color = (80, 150, 200)
        
        cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)
        
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 120), 1)
        
        # Label
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{label}: {value}/{max_value}"
        cv2.putText(frame, text, (x + 5, y + 15), font, 0.4, (220, 220, 220), 1)


class CrewPortrait:
    """Renders a crew member portrait/card."""
    
    @staticmethod
    def draw(frame: np.ndarray, x: int, y: int, 
             crew: CrewMember, is_selected: bool = False):
        """Draw a crew member card."""
        width, height = 120, 80
        
        # Background
        bg_color = (50, 50, 60) if not is_selected else (60, 80, 60)
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
        
        # Mood indicator stripe
        cv2.rectangle(frame, (x, y), (x + 5, y + height), crew.mood.color, -1)
        
        # Border
        border_color = (150, 180, 150) if is_selected else (80, 80, 100)
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, 2 if is_selected else 1)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, crew.name, (x + 10, y + 20), font, 0.4, (220, 220, 220), 1)
        cv2.putText(frame, crew.role.display_name, (x + 10, y + 40), font, 0.35, (150, 180, 200), 1)
        
        # Health/Morale mini bars
        bar_y = y + 55
        bar_width = 50
        
        # Health
        cv2.rectangle(frame, (x + 10, bar_y), (x + 10 + bar_width, bar_y + 8), (40, 40, 50), -1)
        cv2.rectangle(frame, (x + 10, bar_y), (x + 10 + int(bar_width * crew.health / 100), bar_y + 8), (80, 180, 80), -1)
        
        # Morale
        cv2.rectangle(frame, (x + 65, bar_y), (x + 65 + bar_width, bar_y + 8), (40, 40, 50), -1)
        cv2.rectangle(frame, (x + 65, bar_y), (x + 65 + int(bar_width * crew.morale / 100), bar_y + 8), (180, 180, 80), -1)


class StationMarker:
    """Renders an interactive ship station."""
    
    @staticmethod
    def draw(frame: np.ndarray, station: ShipStation, 
             screen_width: int, screen_height: int,
             is_hovered: bool = False, is_selected: bool = False):
        """Draw a station marker."""
        px = int(station.x * screen_width)
        py = int(station.y * screen_height)
        size = station.size
        
        # Glow if hovered/selected
        if is_selected:
            cv2.circle(frame, (px, py), size + 15, (80, 150, 80), 2)
        elif is_hovered:
            cv2.circle(frame, (px, py), size + 10, (100, 100, 130), 2)
        
        # Station circle
        color = (60, 80, 60) if station.is_functional else (80, 50, 50)
        cv2.circle(frame, (px, py), size, color, -1)
        cv2.circle(frame, (px, py), size, (120, 120, 140), 2)
        
        # Icon
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(station.icon, font, 0.7, 2)
        cv2.putText(frame, station.icon, (px - tw // 2, py + th // 4), 
                   font, 0.7, (200, 200, 220), 2)
        
        # Name below
        (tw, _), _ = cv2.getTextSize(station.name, font, 0.4, 1)
        cv2.putText(frame, station.name, (px - tw // 2, py + size + 20),
                   font, 0.4, (150, 150, 170), 1)


# =====================================================
# STEP 4: INTEGRATION HELPERS  
# =====================================================
# These help connect systems to scenes and shared_data.
# =====================================================

class ShipStateManager:
    """
    Manages ship state in shared_data.
    
    Use this pattern to:
    1. Initialize state if it doesn't exist
    2. Get state from shared_data
    3. Save state back (for persistence later)
    """
    
    STATE_KEY = "ship_state"
    
    @classmethod
    def get_or_create(cls, shared_data: Dict, seed: int = None) -> ShipState:
        """Get existing ship state or create new one."""
        if cls.STATE_KEY not in shared_data:
            shared_data[cls.STATE_KEY] = ShipInitializer.create_default_ship(seed)
        return shared_data[cls.STATE_KEY]
    
    @classmethod
    def get(cls, shared_data: Dict) -> Optional[ShipState]:
        """Get ship state if it exists."""
        return shared_data.get(cls.STATE_KEY)
    
    @classmethod
    def save(cls, shared_data: Dict, state: ShipState):
        """Save state back to shared_data."""
        shared_data[cls.STATE_KEY] = state


# =====================================================
# EXAMPLE: Putting it all together
# =====================================================

if __name__ == "__main__":
    """Test the ship systems."""
    
    print("=== Ship Systems Test ===\n")
    
    # Create a ship
    ship = ShipInitializer.create_default_ship(seed=42)
    
    print(f"Day {ship.day}, Hour {ship.hour:02d}:00")
    print(f"\nResources:")
    print(f"  Fuel: {ship.resources.fuel}")
    print(f"  Food: {ship.resources.food}")
    print(f"  Credits: {ship.resources.credits}")
    print(f"  Hull: {ship.resources.hull}")
    
    print(f"\nCrew ({len(ship.crew)} members):")
    for crew in ship.crew:
        print(f"  {crew.name} - {crew.role.display_name}")
        print(f"    Health: {crew.health}, Morale: {crew.morale} ({crew.mood.display_name})")
        print(f"    Effectiveness: {crew.effectiveness:.1%}")
    
    print(f"\nStations ({len(ship.stations)}):")
    for station in ship.stations:
        print(f"  {station.name}: {station.description}")
    
    # Test travel
    print("\n--- Testing Travel ---")
    path_length = 3
    can_go, reason = TravelSystem.can_travel(ship, path_length)
    print(f"Can travel {path_length} jumps? {can_go} - {reason}")
    
    if can_go:
        events = TravelSystem.execute_travel(ship, path_length)
        print(f"Travel events: {events}")
        print(f"Fuel remaining: {ship.resources.fuel}")
    
    # Test time
    print("\n--- Advancing Time ---")
    print(f"Before: Day {ship.day}, Crew morale: {[c.morale for c in ship.crew]}")
    TimeSystem.advance_time(ship, hours=48)
    print(f"After: Day {ship.day}, Crew morale: {[c.morale for c in ship.crew]}")
