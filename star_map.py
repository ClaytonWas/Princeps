"""
Star Map Module
===============
Procedurally generated star map with navigation system.

Features:
- Seed-based star generation
- Connected star pathways for travel
- Pinch to select destination
- Visual link paths between stars
"""

import cv2
import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from enum import Enum


# =====================
# STAR TYPES
# =====================
class StarType(Enum):
    """Different star classifications with visual properties."""
    RED_DWARF = ("Red Dwarf", (80, 80, 255), 0.6, "Common, stable, long-lived")
    YELLOW_STAR = ("Yellow Star", (100, 220, 255), 1.0, "Sun-like, habitable zones")
    BLUE_GIANT = ("Blue Giant", (255, 200, 150), 1.4, "Hot, short-lived, luminous")
    WHITE_DWARF = ("White Dwarf", (255, 255, 255), 0.5, "Stellar remnant, dense")
    ORANGE_GIANT = ("Orange Giant", (80, 165, 255), 1.2, "Evolved star, resource-rich")
    NEUTRON = ("Neutron Star", (255, 150, 255), 0.4, "Extreme gravity, dangerous")
    BINARY = ("Binary System", (200, 255, 200), 1.1, "Twin stars, complex orbits")
    PULSAR = ("Pulsar", (255, 100, 255), 0.5, "Rotating neutron star, beacons")
    
    def __init__(self, display_name: str, color: Tuple[int, int, int], size_mult: float, description: str):
        self.display_name = display_name
        self.color = color
        self.size_mult = size_mult
        self.description = description


# =====================
# REGION TYPES (for flavor)
# =====================
class RegionType(Enum):
    """Different space regions with properties."""
    NORMAL = ("Open Space", (30, 25, 40), 1.0)
    NEBULA = ("Nebula", (60, 40, 50), 0.8)          # Slower travel, hide ships
    ASTEROID = ("Asteroid Field", (40, 40, 35), 0.7) # Dangerous, resources
    VOID = ("Deep Void", (15, 12, 20), 1.2)          # Fast travel, nothing there
    ANOMALY = ("Anomaly Zone", (50, 30, 60), 0.5)    # Strange effects
    
    def __init__(self, display_name: str, color: Tuple[int, int, int], travel_mult: float):
        self.display_name = display_name
        self.color = color
        self.travel_mult = travel_mult


# =====================
# POINTS OF INTEREST
# =====================
class POIType(Enum):
    """Special locations at stars."""
    NONE = ("", (0, 0, 0), "")
    STATION = ("Station", (100, 200, 100), "Trading post, repairs")
    OUTPOST = ("Outpost", (150, 150, 100), "Small settlement")
    RUINS = ("Ancient Ruins", (150, 100, 200), "Mysterious artifacts")
    DERELICT = ("Derelict", (100, 100, 120), "Abandoned ship")
    WORMHOLE = ("Wormhole", (200, 100, 255), "Shortcut to distant star")
    BEACON = ("Beacon", (255, 200, 100), "Navigation signal")
    HAZARD = ("Hazard", (80, 80, 200), "Dangerous area")
    
    def __init__(self, display_name: str, color: Tuple[int, int, int], description: str):
        self.display_name = display_name
        self.color = color
        self.description = description


# =====================
# STAR DATA
# =====================
@dataclass
class Star:
    """A star in the galaxy."""
    id: int
    name: str
    x: float  # Normalized 0-1
    y: float  # Normalized 0-1
    star_type: StarType
    connections: Set[int] = field(default_factory=set)
    
    # Enhanced properties
    region: RegionType = RegionType.NORMAL
    poi: POIType = POIType.NONE
    danger_level: float = 0.0     # 0-1, affects encounters
    resource_level: float = 0.0   # 0-1, mining/trading potential
    population: int = 0           # Settlement size
    visited: bool = False         # Has player been here?
    discovered: bool = True       # Is it visible on map?
    wormhole_to: Optional[int] = None  # ID of connected wormhole star
    
    # Visual state
    base_size: int = 8
    pulse_phase: float = 0.0
    
    @property
    def size(self) -> int:
        return int(self.base_size * self.star_type.size_mult)
    
    @property
    def color(self) -> Tuple[int, int, int]:
        return self.star_type.color
    
    def distance_to(self, other: 'Star') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def pixel_pos(self, width: int, height: int, margin: int = 80) -> Tuple[int, int]:
        """Get screen position with margins."""
        usable_w = width - 2 * margin
        usable_h = height - 2 * margin
        return (
            int(margin + self.x * usable_w),
            int(margin + self.y * usable_h)
        )


# =====================
# STAR NAME GENERATOR
# =====================
class StarNameGenerator:
    """Generates procedural star names."""
    
    PREFIXES = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
        "Proxima", "Nova", "Kepler", "Gliese", "HD", "Wolf", "Barnard's",
        "Tau", "Sigma", "Omega", "Vega", "Rigel", "Antares", "Betel"
    ]
    
    SUFFIXES = [
        "Prime", "Major", "Minor", "Secundus", "Tertius", "Quartus",
        "A", "B", "C", "I", "II", "III", "IV", "V"
    ]
    
    ROOTS = [
        "Cygni", "Centauri", "Eridani", "Draconis", "Pegasi", "Aquarii",
        "Orionis", "Lyrae", "Cassiopeiae", "Andromedae", "Persei", "Tauri",
        "Scorpii", "Leonis", "Virginis", "Carinae", "Velorum", "Hydrae"
    ]
    
    @classmethod
    def generate(cls, rng: random.Random) -> str:
        style = rng.randint(0, 3)
        
        if style == 0:
            # Greek letter + constellation
            return f"{rng.choice(cls.PREFIXES)} {rng.choice(cls.ROOTS)}"
        elif style == 1:
            # Catalog style (HD 12345)
            prefix = rng.choice(["HD", "Gliese", "Kepler", "Wolf", "Ross"])
            return f"{prefix}-{rng.randint(100, 9999)}"
        elif style == 2:
            # Named star + suffix
            return f"{rng.choice(cls.PREFIXES)} {rng.choice(cls.SUFFIXES)}"
        else:
            # Simple designation
            return f"{rng.choice(cls.ROOTS)} {rng.choice(cls.SUFFIXES)}"


# =====================
# GALAXY GENERATOR
# =====================
class GalaxyShape(Enum):
    """Different galaxy generation patterns."""
    RANDOM = "Random scatter"
    SPIRAL = "Spiral arms"
    CLUSTER = "Star clusters"
    RING = "Ring galaxy"
    CORRIDOR = "Travel corridor"


class GalaxyGenerator:
    """Generates a connected star map from a seed."""
    
    def __init__(self, seed: int):
        self.seed = seed
        self.rng = random.Random(seed)
    
    def generate(self, 
                 num_stars: int = 15,
                 min_connections: int = 2,
                 max_connections: int = 4,
                 min_distance: float = 0.12,
                 shape: GalaxyShape = None) -> List[Star]:
        """
        Generate a galaxy with connected stars.
        
        Args:
            num_stars: Number of stars to generate
            min_connections: Minimum connections per star
            max_connections: Maximum connections per star
            min_distance: Minimum distance between stars (normalized)
            shape: Galaxy shape pattern (random if None)
        
        Returns:
            List of connected Star objects
        """
        # Pick random shape if not specified
        if shape is None:
            shape = self.rng.choice(list(GalaxyShape))
        
        stars = self._place_stars(num_stars, min_distance, shape)
        self._create_connections(stars, min_connections, max_connections)
        self._assign_properties(stars)
        self._create_wormholes(stars)
        
        return stars
    
    def _place_stars(self, num_stars: int, min_distance: float, shape: GalaxyShape) -> List[Star]:
        """Place stars based on galaxy shape."""
        if shape == GalaxyShape.SPIRAL:
            return self._place_spiral(num_stars, min_distance)
        elif shape == GalaxyShape.CLUSTER:
            return self._place_clusters(num_stars, min_distance)
        elif shape == GalaxyShape.RING:
            return self._place_ring(num_stars, min_distance)
        elif shape == GalaxyShape.CORRIDOR:
            return self._place_corridor(num_stars, min_distance)
        else:
            return self._place_random(num_stars, min_distance)
    
    def _place_random(self, num_stars: int, min_distance: float) -> List[Star]:
        """Place stars randomly with minimum spacing."""
        stars = []
        max_attempts = 1000
        
        for i in range(num_stars):
            for _ in range(max_attempts):
                x = self.rng.uniform(0.08, 0.92)
                y = self.rng.uniform(0.10, 0.82)
                
                if self._is_valid_position(x, y, stars, min_distance):
                    stars.append(self._create_star(i, x, y))
                    break
        
        return stars
    
    def _place_spiral(self, num_stars: int, min_distance: float) -> List[Star]:
        """Place stars in spiral arm pattern."""
        stars = []
        center_x, center_y = 0.5, 0.45
        num_arms = self.rng.randint(2, 4)
        
        for i in range(num_stars):
            for _ in range(100):
                # Pick an arm
                arm = i % num_arms
                arm_offset = (2 * math.pi * arm) / num_arms
                
                # Distance from center with some randomness
                t = self.rng.uniform(0.1, 1.0)
                r = t * 0.4
                
                # Spiral angle
                angle = arm_offset + t * 2.5 + self.rng.gauss(0, 0.3)
                
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle) * 0.9  # Slightly flattened
                
                # Add some scatter
                x += self.rng.gauss(0, 0.03)
                y += self.rng.gauss(0, 0.03)
                
                x = max(0.08, min(0.92, x))
                y = max(0.10, min(0.82, y))
                
                if self._is_valid_position(x, y, stars, min_distance):
                    stars.append(self._create_star(i, x, y))
                    break
        
        return stars
    
    def _place_clusters(self, num_stars: int, min_distance: float) -> List[Star]:
        """Place stars in cluster formations."""
        stars = []
        num_clusters = self.rng.randint(3, 5)
        
        # Generate cluster centers
        cluster_centers = []
        for _ in range(num_clusters):
            cx = self.rng.uniform(0.2, 0.8)
            cy = self.rng.uniform(0.2, 0.7)
            cluster_centers.append((cx, cy))
        
        for i in range(num_stars):
            for _ in range(100):
                # Pick a cluster
                cx, cy = self.rng.choice(cluster_centers)
                
                # Place near cluster center
                angle = self.rng.uniform(0, 2 * math.pi)
                r = abs(self.rng.gauss(0, 0.12))
                
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                
                x = max(0.08, min(0.92, x))
                y = max(0.10, min(0.82, y))
                
                if self._is_valid_position(x, y, stars, min_distance * 0.8):
                    stars.append(self._create_star(i, x, y))
                    break
        
        return stars
    
    def _place_ring(self, num_stars: int, min_distance: float) -> List[Star]:
        """Place stars in a ring pattern."""
        stars = []
        center_x, center_y = 0.5, 0.45
        
        for i in range(num_stars):
            for _ in range(100):
                angle = self.rng.uniform(0, 2 * math.pi)
                
                # Ring with some variation
                base_r = 0.3
                r = base_r + self.rng.gauss(0, 0.08)
                
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle) * 0.85
                
                x = max(0.08, min(0.92, x))
                y = max(0.10, min(0.82, y))
                
                if self._is_valid_position(x, y, stars, min_distance):
                    stars.append(self._create_star(i, x, y))
                    break
        
        # Add a few central stars
        for i in range(min(3, num_stars // 5)):
            for _ in range(50):
                x = center_x + self.rng.gauss(0, 0.05)
                y = center_y + self.rng.gauss(0, 0.05)
                
                if self._is_valid_position(x, y, stars, min_distance):
                    stars.append(self._create_star(len(stars), x, y))
                    break
        
        return stars
    
    def _place_corridor(self, num_stars: int, min_distance: float) -> List[Star]:
        """Place stars along a travel corridor with branches."""
        stars = []
        
        # Main path from left to right with curves
        control_points = [
            (0.1, self.rng.uniform(0.3, 0.6)),
            (0.35, self.rng.uniform(0.2, 0.7)),
            (0.65, self.rng.uniform(0.2, 0.7)),
            (0.9, self.rng.uniform(0.3, 0.6))
        ]
        
        for i in range(num_stars):
            for _ in range(100):
                # Pick position along the path
                t = self.rng.uniform(0, 1)
                
                # Cubic bezier interpolation
                x, y = self._bezier_point(t, control_points)
                
                # Add perpendicular scatter
                perp_x = self.rng.gauss(0, 0.08)
                perp_y = self.rng.gauss(0, 0.12)
                
                x = max(0.08, min(0.92, x + perp_x))
                y = max(0.10, min(0.82, y + perp_y))
                
                if self._is_valid_position(x, y, stars, min_distance):
                    stars.append(self._create_star(i, x, y))
                    break
        
        return stars
    
    def _bezier_point(self, t: float, points: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate point on cubic bezier curve."""
        p0, p1, p2, p3 = points
        mt = 1 - t
        x = mt**3 * p0[0] + 3*mt**2*t * p1[0] + 3*mt*t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3*mt**2*t * p1[1] + 3*mt*t**2 * p2[1] + t**3 * p3[1]
        return x, y
    
    def _is_valid_position(self, x: float, y: float, stars: List[Star], min_dist: float) -> bool:
        """Check if position is valid (far enough from existing stars)."""
        for star in stars:
            dist = math.sqrt((x - star.x) ** 2 + (y - star.y) ** 2)
            if dist < min_dist:
                return False
        return True
    
    def _create_star(self, star_id: int, x: float, y: float) -> Star:
        """Create a star with randomized properties."""
        # Weight star types by rarity
        type_weights = [
            (StarType.RED_DWARF, 30),
            (StarType.YELLOW_STAR, 25),
            (StarType.ORANGE_GIANT, 15),
            (StarType.WHITE_DWARF, 10),
            (StarType.BLUE_GIANT, 10),
            (StarType.BINARY, 5),
            (StarType.NEUTRON, 3),
            (StarType.PULSAR, 2),
        ]
        
        total = sum(w for _, w in type_weights)
        roll = self.rng.randint(1, total)
        cumulative = 0
        star_type = StarType.YELLOW_STAR
        
        for stype, weight in type_weights:
            cumulative += weight
            if roll <= cumulative:
                star_type = stype
                break
        
        return Star(
            id=star_id,
            name=StarNameGenerator.generate(self.rng),
            x=x,
            y=y,
            star_type=star_type,
            pulse_phase=self.rng.uniform(0, math.pi * 2)
        )
    
    def _assign_properties(self, stars: List[Star]):
        """Assign gameplay properties to stars."""
        for star in stars:
            # Region based on position and randomness
            region_roll = self.rng.random()
            if region_roll < 0.1:
                star.region = RegionType.NEBULA
            elif region_roll < 0.18:
                star.region = RegionType.ASTEROID
            elif region_roll < 0.25:
                star.region = RegionType.VOID
            elif region_roll < 0.30:
                star.region = RegionType.ANOMALY
            
            # Danger level (higher for certain star types)
            base_danger = self.rng.random() * 0.6
            if star.star_type in [StarType.NEUTRON, StarType.PULSAR]:
                base_danger += 0.3
            if star.region == RegionType.ANOMALY:
                base_danger += 0.2
            star.danger_level = min(1.0, base_danger)
            
            # Resources (based on star type)
            if star.star_type in [StarType.ORANGE_GIANT, StarType.BLUE_GIANT]:
                star.resource_level = 0.4 + self.rng.random() * 0.6
            elif star.region == RegionType.ASTEROID:
                star.resource_level = 0.5 + self.rng.random() * 0.5
            else:
                star.resource_level = self.rng.random() * 0.6
            
            # Population (more likely around yellow stars)
            if star.star_type == StarType.YELLOW_STAR:
                star.population = self.rng.randint(1, 5)
            elif star.star_type in [StarType.RED_DWARF, StarType.ORANGE_GIANT]:
                star.population = self.rng.randint(0, 3)
            else:
                star.population = self.rng.randint(0, 1)
            
            # Points of interest (rarer)
            poi_roll = self.rng.random()
            if poi_roll < 0.02:
                star.poi = POIType.WORMHOLE
            elif poi_roll < 0.08:
                star.poi = POIType.STATION
            elif poi_roll < 0.15:
                star.poi = POIType.OUTPOST
            elif poi_roll < 0.20:
                star.poi = POIType.RUINS
            elif poi_roll < 0.25:
                star.poi = POIType.DERELICT
            elif poi_roll < 0.28:
                star.poi = POIType.BEACON
            elif poi_roll < 0.32 and star.danger_level >= 3:
                star.poi = POIType.HAZARD
    
    def _create_wormholes(self, stars: List[Star]):
        """Create wormhole connections between distant stars."""
        wormhole_stars = [s for s in stars if s.poi == POIType.WORMHOLE]
        
        if len(wormhole_stars) >= 2:
            # Connect wormholes to each other (shortcuts across the map!)
            # Pair them up - each wormhole connects to one other
            shuffled = wormhole_stars.copy()
            self.rng.shuffle(shuffled)
            
            for i in range(0, len(shuffled) - 1, 2):
                star_a = shuffled[i]
                star_b = shuffled[i + 1]
                
                # Set up bidirectional wormhole connection
                star_a.wormhole_to = star_b.id
                star_b.wormhole_to = star_a.id
                
                # Also add to regular connections for pathfinding
                star_a.connections.add(star_b.id)
                star_b.connections.add(star_a.id)
    
    def _create_connections(self, stars: List[Star], min_connections: int, max_connections: int):
        """Create connections between stars."""
        # First ensure all stars are connected via MST
        self._create_mst(stars)
        
        # Then add extra connections for variety
        for star in stars:
            current_connections = len(star.connections)
            if current_connections >= max_connections:
                continue
            
            desired = self.rng.randint(min_connections, max_connections)
            if current_connections >= desired:
                continue
            
            # Find nearby stars sorted by distance
            distances = []
            for other in stars:
                if other.id != star.id and other.id not in star.connections:
                    dist = star.distance_to(other)
                    distances.append((dist, other))
            
            distances.sort(key=lambda x: x[0])
            
            # Add connections to nearest unconnected stars
            for dist, other in distances[:desired - current_connections]:
                if dist < 0.35:  # Max connection distance
                    star.connections.add(other.id)
                    other.connections.add(star.id)
    
    def _create_mst(self, stars: List[Star]):
        """Create minimum spanning tree to ensure connectivity."""
        if len(stars) < 2:
            return
        
        # Prim's algorithm
        in_tree = {stars[0].id}
        
        while len(in_tree) < len(stars):
            best_edge = None
            best_dist = float('inf')
            
            for star in stars:
                if star.id not in in_tree:
                    continue
                
                for other in stars:
                    if other.id in in_tree:
                        continue
                    
                    dist = star.distance_to(other)
                    if dist < best_dist:
                        best_dist = dist
                        best_edge = (star, other)
            
            if best_edge:
                star_a, star_b = best_edge
                star_a.connections.add(star_b.id)
                star_b.connections.add(star_a.id)
                in_tree.add(star_b.id)


# =====================
# PATHFINDING
# =====================
class PathFinder:
    """Find paths between stars using BFS."""
    
    @staticmethod
    def find_path(stars: List[Star], start_id: int, end_id: int) -> Optional[List[int]]:
        """Find shortest path between two stars."""
        if start_id == end_id:
            return [start_id]
        
        star_map = {s.id: s for s in stars}
        
        # BFS
        queue = [(start_id, [start_id])]
        visited = {start_id}
        
        while queue:
            current_id, path = queue.pop(0)
            current = star_map[current_id]
            
            for neighbor_id in current.connections:
                if neighbor_id == end_id:
                    return path + [neighbor_id]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))
        
        return None  # No path found


# =====================
# STAR MAP RENDERER
# =====================
class StarMapRenderer:
    """Renders the star map with visual effects."""
    
    # Colors
    BG_COLOR = (15, 12, 20)
    GRID_COLOR = (30, 25, 40)
    CONNECTION_COLOR = (60, 50, 80)
    WORMHOLE_COLOR = (255, 100, 255)  # Purple for wormholes
    PATH_COLOR = (100, 200, 255)
    CURRENT_STAR_GLOW = (0, 255, 200)
    SELECTED_STAR_GLOW = (0, 150, 255)
    HOVER_STAR_GLOW = (150, 150, 255)
    
    # Region colors (tinted backgrounds)
    REGION_COLORS = {
        RegionType.NORMAL: None,
        RegionType.NEBULA: (40, 20, 60),      # Purple haze
        RegionType.ASTEROID: (50, 45, 35),    # Brown/rocky
        RegionType.VOID: (5, 5, 10),          # Extra dark
        RegionType.ANOMALY: (20, 50, 50),     # Teal mystery
    }
    
    # POI icons (simple shapes for now)
    POI_COLORS = {
        POIType.STATION: (200, 200, 100),     # Yellow
        POIType.OUTPOST: (150, 150, 80),      # Dim yellow
        POIType.RUINS: (100, 200, 150),       # Teal
        POIType.DERELICT: (100, 100, 100),    # Gray
        POIType.WORMHOLE: (255, 100, 255),    # Magenta
        POIType.BEACON: (100, 150, 255),      # Light blue
        POIType.HAZARD: (200, 80, 80),        # Red
    }
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.time = 0.0
    
    def render(self, 
               frame: np.ndarray, 
               stars: List[Star],
               current_star_id: int,
               selected_star_id: Optional[int] = None,
               hover_star_id: Optional[int] = None,
               travel_path: Optional[List[int]] = None,
               travel_progress: float = 0.0,
               show_details: bool = True):
        """Render the complete star map."""
        
        self.time += 0.016  # Assume ~60fps
        
        # Background
        frame[:] = self.BG_COLOR
        
        # Draw region backgrounds (nebulae, etc.)
        self._draw_region_backgrounds(frame, stars)
        
        # Subtle grid
        self._draw_grid(frame)
        
        # Background stars (decoration)
        self._draw_background_stars(frame)
        
        # Connections (wormholes drawn differently)
        self._draw_connections(frame, stars, current_star_id, travel_path)
        
        # Region hazards (asteroids, etc.)
        self._draw_region_hazards(frame, stars)
        
        # Stars
        for star in stars:
            is_current = star.id == current_star_id
            is_selected = star.id == selected_star_id
            is_hover = star.id == hover_star_id
            is_on_path = travel_path and star.id in travel_path
            
            self._draw_star(frame, star, is_current, is_selected, is_hover, is_on_path)
            
            # Draw POI indicator
            if star.poi:
                self._draw_poi_indicator(frame, star)
        
        # Draw danger/resource indicators for hovered star
        if hover_star_id is not None and show_details:
            hover_star = next((s for s in stars if s.id == hover_star_id), None)
            if hover_star:
                self._draw_star_details(frame, hover_star)
        
        # Travel animation
        if travel_path and len(travel_path) >= 2 and travel_progress > 0:
            self._draw_travel_ship(frame, stars, travel_path, travel_progress)
    
    def _draw_region_backgrounds(self, frame: np.ndarray, stars: List[Star]):
        """Draw atmospheric backgrounds for special regions."""
        overlay = np.zeros_like(frame)
        
        for star in stars:
            if star.region == RegionType.NORMAL:
                continue
                
            px, py = star.pixel_pos(self.width, self.height)
            color = self.REGION_COLORS.get(star.region)
            if color is None:
                continue
            
            # Draw a soft circular region around the star
            region_radius = 80 + int(star.size * 5)
            
            if star.region == RegionType.NEBULA:
                # Nebula: colorful, cloudy effect
                for r in range(region_radius, 0, -10):
                    alpha = 0.3 * (r / region_radius)
                    # Animated swirl
                    offset_x = int(math.sin(self.time * 0.5 + star.pulse_phase) * 5)
                    offset_y = int(math.cos(self.time * 0.3 + star.pulse_phase) * 5)
                    cv2.circle(overlay, (px + offset_x, py + offset_y), r, color, -1)
            
            elif star.region == RegionType.VOID:
                # Void: extra dark, ominous
                cv2.circle(overlay, (px, py), region_radius, color, -1)
            
            elif star.region == RegionType.ANOMALY:
                # Anomaly: pulsing, mysterious
                pulse = math.sin(self.time * 2 + star.pulse_phase) * 0.3 + 0.7
                r = int(region_radius * pulse)
                cv2.circle(overlay, (px, py), r, color, -1)
        
        # Blend overlay with frame
        cv2.addWeighted(overlay, 0.3, frame, 1.0, 0, frame)
    
    def _draw_region_hazards(self, frame: np.ndarray, stars: List[Star]):
        """Draw hazards like asteroids in regions."""
        rng = random.Random(12345)  # Consistent positions
        
        for star in stars:
            if star.region != RegionType.ASTEROID:
                continue
            
            px, py = star.pixel_pos(self.width, self.height)
            
            # Draw scattered asteroids around the star
            for _ in range(15):
                angle = rng.uniform(0, 2 * math.pi)
                dist = rng.uniform(30, 70)
                ax = int(px + math.cos(angle) * dist)
                ay = int(py + math.sin(angle) * dist)
                
                # Small asteroid rocks
                size = rng.randint(2, 5)
                gray = rng.randint(40, 70)
                cv2.circle(frame, (ax, ay), size, (gray, gray - 5, gray - 10), -1)
    
    def _draw_poi_indicator(self, frame: np.ndarray, star: Star):
        """Draw a POI indicator near the star."""
        if not star.poi or star.poi == POIType.NONE:
            return
            
        px, py = star.pixel_pos(self.width, self.height)
        color = self.POI_COLORS.get(star.poi, (200, 200, 200))
        
        # Offset position (top-right of star)
        ix = px + star.size + 8
        iy = py - star.size - 5
        
        # Different shapes for different POI types
        if star.poi == POIType.STATION:
            # Diamond shape
            pts = np.array([[ix, iy - 6], [ix + 6, iy], [ix, iy + 6], [ix - 6, iy]], np.int32)
            cv2.fillPoly(frame, [pts], color)
        elif star.poi == POIType.OUTPOST:
            # Small square
            cv2.rectangle(frame, (ix - 4, iy - 4), (ix + 4, iy + 4), color, -1)
        elif star.poi == POIType.WORMHOLE:
            # Ring/portal
            cv2.circle(frame, (ix, iy), 6, color, 2)
            cv2.circle(frame, (ix, iy), 3, (100, 50, 100), -1)
        elif star.poi == POIType.HAZARD:
            # Warning triangle with !
            pts = np.array([[ix, iy - 6], [ix + 5, iy + 4], [ix - 5, iy + 4]], np.int32)
            cv2.fillPoly(frame, [pts], color)
            cv2.putText(frame, "!", (ix - 2, iy + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        elif star.poi == POIType.RUINS or star.poi == POIType.DERELICT:
            # Triangle (mystery/ancient)
            pts = np.array([[ix, iy - 6], [ix + 5, iy + 4], [ix - 5, iy + 4]], np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
        elif star.poi == POIType.BEACON:
            # Antenna/broadcast symbol
            cv2.circle(frame, (ix, iy), 4, color, -1)
            cv2.line(frame, (ix, iy - 4), (ix, iy - 8), color, 2)
        else:
            # Default: small circle
            cv2.circle(frame, (ix, iy), 4, color, -1)
    
    def _draw_star_details(self, frame: np.ndarray, star: Star):
        """Draw detailed info panel for a hovered star."""
        px, py = star.pixel_pos(self.width, self.height)
        
        # Panel position (offset from star)
        panel_x = px + 30
        panel_y = py - 40
        panel_w = 150
        panel_h = 80
        
        # Ensure panel stays on screen
        if panel_x + panel_w > self.width - 10:
            panel_x = px - panel_w - 30
        if panel_y < 10:
            panel_y = py + 20
        if panel_y + panel_h > self.height - 10:
            panel_y = self.height - panel_h - 10
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (30, 25, 45), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                     (80, 70, 100), 1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Text content
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = panel_y + 16
        
        # Star type
        type_name = star.star_type.display_name if star.star_type else "Unknown"
        cv2.putText(frame, type_name, (panel_x + 5, y_offset), font, 0.35, star.color, 1)
        y_offset += 14
        
        # Region
        if star.region != RegionType.NORMAL:
            region_text = f"Region: {star.region.display_name}"
            cv2.putText(frame, region_text, (panel_x + 5, y_offset), font, 0.3, (180, 180, 200), 1)
            y_offset += 12
        
        # Danger level (red bar)
        cv2.putText(frame, "Danger:", (panel_x + 5, y_offset), font, 0.3, (180, 180, 200), 1)
        bar_x = panel_x + 50
        bar_w = 90
        cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + bar_w, y_offset), (40, 30, 30), -1)
        danger_w = int(bar_w * star.danger_level)
        danger_color = (0, 0, int(100 + 155 * star.danger_level))  # Red intensity
        cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + danger_w, y_offset), danger_color, -1)
        y_offset += 14
        
        # Resources (green bar)
        cv2.putText(frame, "Resources:", (panel_x + 5, y_offset), font, 0.3, (180, 180, 200), 1)
        cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + bar_w, y_offset), (30, 40, 30), -1)
        res_w = int(bar_w * star.resource_level)
        cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + res_w, y_offset), (0, 150, 50), -1)
        y_offset += 14
        
        # Population
        pop_text = f"Pop: {star.population:,}"
        cv2.putText(frame, pop_text, (panel_x + 5, y_offset), font, 0.3, (180, 180, 200), 1)
        
        # POI if present
        if star.poi and star.poi != POIType.NONE:
            poi_text = f"POI: {star.poi.display_name}"
            cv2.putText(frame, poi_text, (panel_x + 70, y_offset), font, 0.3, 
                       self.POI_COLORS.get(star.poi, (200, 200, 200)), 1)
    
    def _draw_grid(self, frame: np.ndarray):
        """Draw subtle background grid."""
        spacing = 60
        for x in range(0, self.width, spacing):
            cv2.line(frame, (x, 0), (x, self.height), self.GRID_COLOR, 1)
        for y in range(0, self.height, spacing):
            cv2.line(frame, (0, y), (self.width, y), self.GRID_COLOR, 1)
    
    def _draw_background_stars(self, frame: np.ndarray):
        """Draw decorative background stars."""
        rng = random.Random(42)  # Fixed seed for consistency
        for _ in range(100):
            x = rng.randint(0, self.width)
            y = rng.randint(0, self.height)
            brightness = rng.randint(30, 80)
            
            # Twinkle effect
            twinkle = math.sin(self.time * rng.uniform(1, 3) + rng.uniform(0, 6.28))
            brightness = int(brightness * (0.7 + 0.3 * twinkle))
            
            cv2.circle(frame, (x, y), 1, (brightness, brightness, brightness), -1)
    
    def _draw_connections(self, frame: np.ndarray, stars: List[Star], 
                          current_star_id: int, travel_path: Optional[List[int]]):
        """Draw connection lines between stars."""
        star_map = {s.id: s for s in stars}
        drawn = set()
        
        # First pass: regular connections
        for star in stars:
            for conn_id in star.connections:
                edge_key = tuple(sorted([star.id, conn_id]))
                if edge_key in drawn:
                    continue
                drawn.add(edge_key)
                
                other = star_map.get(conn_id)
                if not other:
                    continue
                    
                p1 = star.pixel_pos(self.width, self.height)
                p2 = other.pixel_pos(self.width, self.height)
                
                # Check if this connection is on the travel path
                is_path = False
                if travel_path:
                    for i in range(len(travel_path) - 1):
                        if {travel_path[i], travel_path[i + 1]} == {star.id, conn_id}:
                            is_path = True
                            break
                
                if is_path:
                    # Highlighted path
                    cv2.line(frame, p1, p2, self.PATH_COLOR, 3)
                    cv2.line(frame, p1, p2, (150, 230, 255), 1)
                else:
                    # Normal connection
                    cv2.line(frame, p1, p2, self.CONNECTION_COLOR, 1)
        
        # Second pass: wormhole connections (drawn on top with special style)
        for star in stars:
            if not hasattr(star, 'wormhole_to') or star.wormhole_to is None:
                continue
            
            other = star_map.get(star.wormhole_to)
            if not other:
                continue
                
            p1 = star.pixel_pos(self.width, self.height)
            p2 = other.pixel_pos(self.width, self.height)
            
            # Animated dashed line for wormhole
            self._draw_wormhole_connection(frame, p1, p2)
    
    def _draw_wormhole_connection(self, frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int]):
        """Draw an animated wormhole connection between two points."""
        # Calculate line length and direction
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Draw animated dashed line with pulsing glow
        dash_length = 10
        gap_length = 8
        segment_length = dash_length + gap_length
        
        # Animate the dash offset
        offset = int((self.time * 30) % segment_length)
        
        current_pos = offset
        while current_pos < length:
            # Start and end of this dash
            start_dist = current_pos
            end_dist = min(current_pos + dash_length, length)
            
            if end_dist > 0:
                sx = int(p1[0] + dx * max(0, start_dist))
                sy = int(p1[1] + dy * max(0, start_dist))
                ex = int(p1[0] + dx * end_dist)
                ey = int(p1[1] + dy * end_dist)
                
                # Pulsing color
                pulse = math.sin(self.time * 4) * 0.3 + 0.7
                color = (int(180 * pulse), int(80 * pulse), int(200 * pulse))
                
                cv2.line(frame, (sx, sy), (ex, ey), color, 2)
            
            current_pos += segment_length
        
        # Draw portal circles at endpoints
        pulse = math.sin(self.time * 3) * 0.2 + 0.8
        portal_size = int(6 * pulse)
        cv2.circle(frame, p1, portal_size + 3, (100, 50, 120), 1)
        cv2.circle(frame, p2, portal_size + 3, (100, 50, 120), 1)
    
    def _draw_star(self, frame: np.ndarray, star: Star, 
                   is_current: bool, is_selected: bool, is_hover: bool, is_on_path: bool):
        """Draw a single star with effects."""
        px, py = star.pixel_pos(self.width, self.height)
        
        # Pulse animation
        pulse = math.sin(self.time * 2 + star.pulse_phase) * 0.2 + 1.0
        size = int(star.size * pulse)
        
        # Glow effect for special stars
        if is_current:
            glow_size = size + 15 + int(math.sin(self.time * 3) * 3)
            self._draw_glow(frame, px, py, glow_size, self.CURRENT_STAR_GLOW)
        elif is_selected:
            glow_size = size + 12 + int(math.sin(self.time * 4) * 2)
            self._draw_glow(frame, px, py, glow_size, self.SELECTED_STAR_GLOW)
        elif is_hover:
            glow_size = size + 8
            self._draw_glow(frame, px, py, glow_size, self.HOVER_STAR_GLOW)
        elif is_on_path:
            glow_size = size + 6
            self._draw_glow(frame, px, py, glow_size, self.PATH_COLOR)
        
        # Danger indicator ring (subtle red for dangerous systems)
        if star.danger_level > 0.6:
            danger_pulse = math.sin(self.time * 5 + star.pulse_phase) * 0.3 + 0.7
            danger_alpha = int(star.danger_level * 100 * danger_pulse)
            cv2.circle(frame, (px, py), size + 4, (0, 0, danger_alpha), 1)
        
        # Star core
        cv2.circle(frame, (px, py), size, star.color, -1)
        
        # Bright center
        cv2.circle(frame, (px, py), max(2, size // 2), (255, 255, 255), -1)
        
        # Visited indicator (small tick mark)
        if star.visited and not is_current:
            cv2.circle(frame, (px + size + 3, py - size - 3), 3, (100, 200, 100), -1)
        
        # Star name (only for special or hovered stars)
        if is_current or is_selected or is_hover:
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = star.name
            if is_current:
                label = f"[YOU] {star.name}"
            
            (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
            tx = px - tw // 2
            ty = py - size - 10
            
            # Background for readability
            cv2.rectangle(frame, (tx - 4, ty - th - 2), (tx + tw + 4, ty + 4), (20, 15, 30), -1)
            cv2.putText(frame, label, (tx, ty), font, 0.4, (200, 200, 220), 1)
    
    def _draw_glow(self, frame: np.ndarray, x: int, y: int, radius: int, color: Tuple[int, int, int]):
        """Draw a glowing effect."""
        overlay = frame.copy()
        for r in range(radius, 0, -3):
            alpha = 0.1 * (r / radius)
            cv2.circle(overlay, (x, y), r, color, -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def _draw_travel_ship(self, frame: np.ndarray, stars: List[Star], 
                          path: List[int], progress: float):
        """Draw the ship traveling along the path."""
        if len(path) < 2:
            return
        
        star_map = {s.id: s for s in stars}
        
        # Calculate which segment we're on
        num_segments = len(path) - 1
        segment_progress = progress * num_segments
        segment_idx = min(int(segment_progress), num_segments - 1)
        local_progress = segment_progress - segment_idx
        
        # Get positions
        start_star = star_map[path[segment_idx]]
        end_star = star_map[path[segment_idx + 1]]
        
        p1 = start_star.pixel_pos(self.width, self.height)
        p2 = end_star.pixel_pos(self.width, self.height)
        
        # Interpolate position
        ship_x = int(p1[0] + (p2[0] - p1[0]) * local_progress)
        ship_y = int(p1[1] + (p2[1] - p1[1]) * local_progress)
        
        # Draw ship (simple triangle)
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        ship_size = 8
        
        # Ship points (triangle pointing in direction of travel)
        pts = np.array([
            [ship_x + int(math.cos(angle) * ship_size), 
             ship_y + int(math.sin(angle) * ship_size)],
            [ship_x + int(math.cos(angle + 2.5) * ship_size * 0.7), 
             ship_y + int(math.sin(angle + 2.5) * ship_size * 0.7)],
            [ship_x + int(math.cos(angle - 2.5) * ship_size * 0.7), 
             ship_y + int(math.sin(angle - 2.5) * ship_size * 0.7)],
        ], np.int32)
        
        # Engine glow
        cv2.circle(frame, (ship_x, ship_y), 12, (50, 100, 150), -1)
        cv2.fillPoly(frame, [pts], (200, 220, 255))
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


# =====================
# STAR MAP STATE
# =====================
@dataclass
class StarMapState:
    """Holds the current state of the star map."""
    stars: List[Star] = field(default_factory=list)
    current_star_id: int = 0
    selected_star_id: Optional[int] = None
    hover_star_id: Optional[int] = None
    
    # Travel state
    is_traveling: bool = False
    travel_path: Optional[List[int]] = None
    travel_progress: float = 0.0
    travel_speed: float = 0.3  # Progress per second
    
    # Selection state
    pinch_start_time: float = 0.0
    pinch_held: bool = False
    selection_threshold: float = 0.5  # Seconds to hold for selection


class StarMapController:
    """Handles star map interaction logic."""
    
    def __init__(self, state: StarMapState, width: int, height: int):
        self.state = state
        self.width = width
        self.height = height
        self.hover_radius = 35  # Pixels
    
    def update(self, cursor_x: Optional[float], cursor_y: Optional[float],
               is_pinching: bool, pinch_started: bool, pinch_ended: bool,
               timestamp: float, delta_time: float) -> Optional[str]:
        """
        Update star map state based on input.
        
        Returns:
            Event string if something happened, None otherwise
        """
        # Handle traveling
        if self.state.is_traveling:
            self.state.travel_progress += self.state.travel_speed * delta_time
            
            if self.state.travel_progress >= 1.0:
                # Arrived at destination
                if self.state.travel_path:
                    self.state.current_star_id = self.state.travel_path[-1]
                self.state.is_traveling = False
                self.state.travel_path = None
                self.state.travel_progress = 0.0
                self.state.selected_star_id = None
                return "arrived"
            return None
        
        # Update hover
        self.state.hover_star_id = None
        if cursor_x is not None and cursor_y is not None:
            px = int(cursor_x * self.width)
            py = int(cursor_y * self.height)
            
            for star in self.state.stars:
                sx, sy = star.pixel_pos(self.width, self.height)
                dist = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
                
                if dist < self.hover_radius:
                    self.state.hover_star_id = star.id
                    break
        
        # Handle pinch selection
        if pinch_started and self.state.hover_star_id is not None:
            self.state.pinch_start_time = timestamp
            self.state.pinch_held = True
        
        if pinch_ended:
            if self.state.pinch_held and self.state.hover_star_id is not None:
                held_time = timestamp - self.state.pinch_start_time
                
                if held_time >= self.state.selection_threshold:
                    # Select and start travel
                    target_id = self.state.hover_star_id
                    
                    if target_id != self.state.current_star_id:
                        path = PathFinder.find_path(
                            self.state.stars, 
                            self.state.current_star_id, 
                            target_id
                        )
                        
                        if path:
                            self.state.selected_star_id = target_id
                            self.state.travel_path = path
                            self.state.travel_progress = 0.0
                            self.state.is_traveling = True
                            return "travel_started"
            
            self.state.pinch_held = False
        
        # Visual feedback for holding
        if is_pinching and self.state.pinch_held and self.state.hover_star_id is not None:
            held_time = timestamp - self.state.pinch_start_time
            if held_time >= self.state.selection_threshold:
                self.state.selected_star_id = self.state.hover_star_id
        
        return None
    
    def get_selection_progress(self, timestamp: float) -> float:
        """Get progress of current selection (0-1)."""
        if not self.state.pinch_held:
            return 0.0
        
        held_time = timestamp - self.state.pinch_start_time
        return min(1.0, held_time / self.state.selection_threshold)
