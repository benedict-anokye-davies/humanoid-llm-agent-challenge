"""Partially observable 2D grid world with fog-of-war."""

from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Tuple
import math


@dataclass(frozen=True)
class Pos:
    x: int
    y: int

    def __add__(self, other: "Pos") -> "Pos":
        return Pos(self.x + other.x, self.y + other.y)

    def dist(self, other: "Pos") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)


DIRECTIONS = {
    "N": Pos(0, -1),
    "E": Pos(1, 0),
    "S": Pos(0, 1),
    "W": Pos(-1, 0),
}
DIR_ORDER = ["N", "E", "S", "W"]


class GridWorld:
    """
    8×8 maze. Fog-of-war: agent sees Manhattan distance <= VISION_RADIUS,
    but NOT through walls.
    """

    VISION_RADIUS = 2
    WIDTH = 8
    HEIGHT = 8

    def __init__(self):
        self.agent_pos = Pos(0, 0)
        self.agent_dir_idx = 1  # facing East
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.max_steps = 40

        self.key_pos = Pos(6, 1)
        self.door_pos = Pos(6, 6)

        # Maze walls
        self.walls: Set[Pos] = {
            Pos(3, 0),
            Pos(2, 1), Pos(3, 1), Pos(4, 1), Pos(5, 1),
            Pos(5, 2),
            Pos(1, 3), Pos(2, 3), Pos(3, 3), Pos(5, 3),
            Pos(1, 5), Pos(2, 5), Pos(3, 5), Pos(4, 5), Pos(5, 5),
        }
        # Ensure special cells are free
        for p in (self.key_pos, self.door_pos, self.agent_pos):
            self.walls.discard(p)

    @property
    def agent_dir(self) -> str:
        return DIR_ORDER[self.agent_dir_idx]

    def _in_bounds(self, p: Pos) -> bool:
        return 0 <= p.x < self.WIDTH and 0 <= p.y < self.HEIGHT

    def _is_wall(self, p: Pos) -> bool:
        return p in self.walls

    def _line_of_sight(self, p: Pos) -> bool:
        """Bresenham-style line of sight. Simple: check cells along path."""
        # For small grids, just check if any wall blocks the straight path
        # Actually for Manhattan fog, we just check distance and no wall at p
        return self._in_bounds(p) and not self._is_wall(p)

    def visible_cells(self) -> Set[Pos]:
        """All cells the agent can currently see."""
        visible = set()
        for dy in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
            for dx in range(-self.VISION_RADIUS, self.VISION_RADIUS + 1):
                p = Pos(self.agent_pos.x + dx, self.agent_pos.y + dy)
                if self._in_bounds(p) and self.agent_pos.dist(p) <= self.VISION_RADIUS:
                    # Simple occlusion: if there's a wall directly between agent and p, hide it
                    if not self._wall_between(self.agent_pos, p):
                        visible.add(p)
        return visible

    def _wall_between(self, a: Pos, b: Pos) -> bool:
        """Check if any wall blocks direct view (simplified for small grids)."""
        # If Manhattan distance is 1, never blocked
        if a.dist(b) <= 1:
            return False
        # Check intermediate cells along x then y
        x_step = 1 if b.x > a.x else -1 if b.x < a.x else 0
        y_step = 1 if b.y > a.y else -1 if b.y < a.y else 0
        cx, cy = a.x, a.y
        while (cx, cy) != (b.x, b.y):
            if cx != b.x:
                cx += x_step
            elif cy != b.y:
                cy += y_step
            if (cx, cy) != (b.x, b.y) and self._is_wall(Pos(cx, cy)):
                return True
        return False

    def get_local_observation(self) -> Dict:
        """What the agent perceives RIGHT NOW."""
        visible = self.visible_cells()
        nearby = {}
        for p in visible:
            desc = "floor"
            if p == self.agent_pos:
                desc = "self"
            elif self._is_wall(p):
                desc = "wall"
            elif p == self.door_pos:
                desc = "door" + ("_open" if self.door_open else "_locked")
            elif p == self.key_pos and not self.has_key:
                desc = "key"
            nearby[f"({p.x},{p.y})"] = desc

        ahead = self.agent_pos + DIRECTIONS[self.agent_dir]
        ahead_desc = "boundary"
        if self._in_bounds(ahead):
            if self._is_wall(ahead):
                ahead_desc = "wall"
            elif ahead == self.door_pos:
                ahead_desc = "door" + ("_open" if self.door_open else "_locked")
            elif ahead == self.key_pos and not self.has_key:
                ahead_desc = "key"
            else:
                ahead_desc = "floor"

        return {
            "position": {"x": self.agent_pos.x, "y": self.agent_pos.y},
            "facing": self.agent_dir,
            "ahead": ahead_desc,
            "vision_radius": self.VISION_RADIUS,
            "visible_cells": nearby,
            "inventory": {"key": self.has_key},
            "door_open": self.door_open,
            "steps_remaining": self.max_steps - self.steps,
            "goal_reached": self.door_open and self.agent_pos == self.door_pos,
        }

    def step(self, action: str, direction: Optional[str] = None) -> Dict:
        self.steps += 1
        msg = ""

        if action == "move":
            target = self.agent_pos + DIRECTIONS.get(direction, Pos(0, 0))
            if not self._in_bounds(target):
                msg = "Bump! Out of bounds."
            elif self._is_wall(target):
                msg = "Bump! Wall."
            elif target == self.door_pos and not self.door_open:
                msg = "Bump! Door is locked."
            else:
                self.agent_pos = target
                msg = f"Moved {direction} to ({target.x},{target.y})."

        elif action == "turn":
            if direction == "left":
                self.agent_dir_idx = (self.agent_dir_idx - 1) % 4
                msg = f"Turned left. Now facing {self.agent_dir}."
            elif direction == "right":
                self.agent_dir_idx = (self.agent_dir_idx + 1) % 4
                msg = f"Turned right. Now facing {self.agent_dir}."
            else:
                msg = "Invalid turn direction."

        elif action == "look":
            msg = f"You look around. Visible cells: {len(self.visible_cells())}"

        elif action == "pick_up":
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                msg = "You picked up the key!"
            else:
                msg = "Nothing to pick up here."

        elif action == "open_door":
            # Must be adjacent or on the door
            if self.agent_pos.dist(self.door_pos) <= 1:
                if self.has_key and not self.door_open:
                    self.door_open = True
                    msg = "You unlocked and opened the door!"
                elif self.door_open:
                    msg = "Door already open."
                else:
                    msg = "Door is locked. You need a key."
            else:
                msg = "You are not near the door."

        else:
            msg = f"Unknown action: {action}"

        obs = self.get_local_observation()
        obs["message"] = msg
        return obs

    def is_done(self) -> bool:
        return (self.door_open and self.agent_pos == self.door_pos) or self.steps >= self.max_steps

    def render(self, agent_memory: Optional[Dict] = None) -> str:
        """ASCII render. If agent_memory provided, only show explored cells."""
        lines = []
        for y in range(self.HEIGHT):
            row = ""
            for x in range(self.WIDTH):
                p = Pos(x, y)
                if p == self.agent_pos:
                    sym = {"N": "^", "E": ">", "S": "v", "W": "<"}
                    row += sym[self.agent_dir]
                elif p == self.door_pos:
                    row += "=" if self.door_open else "D"
                elif p == self.key_pos and not self.has_key:
                    row += "K"
                elif self._is_wall(p):
                    row += "#"
                else:
                    row += "."
            lines.append(row)
        return "\n".join(lines)
