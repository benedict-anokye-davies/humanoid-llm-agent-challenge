"""Simple 2D grid world for the LLM agent challenge."""

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.x + other.x, self.y + other.y)


DIRECTIONS = {
    "N": Position(0, -1),
    "E": Position(1, 0),
    "S": Position(0, 1),
    "W": Position(-1, 0),
}

# Agent orientation: index into ordered list [N, E, S, W]
DIR_ORDER = ["N", "E", "S", "W"]


class GridWorld:
    """
    A 2D grid world with:
    - Walls (#)
    - Empty floor (.)
    - Key (K)
    - Locked door (D) — requires key to open
    - Agent (A)

    Goal: pick up the key, then open the door.
    """

    def __init__(self, width: int = 5, height: int = 5):
        self.width = width
        self.height = height
        self.agent_pos = Position(0, 0)
        self.agent_dir_idx = 1  # facing East
        self.has_key = False
        self.door_open = False
        self.steps = 0
        self.max_steps = 30

        # Static object positions
        self.key_pos = Position(2, 0)
        self.door_pos = Position(2, 2)

        # Walls (immutable)
        self.walls: set = {
            Position(1, 1),
            Position(3, 1),
            Position(1, 3),
        }

        # Ensure special cells are not walls
        for p in (self.key_pos, self.door_pos, self.agent_pos):
            self.walls.discard(p)

    @property
    def agent_dir(self) -> str:
        return DIR_ORDER[self.agent_dir_idx]

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def _is_wall(self, pos: Position) -> bool:
        return pos in self.walls

    def _look_ahead(self) -> Position:
        return self.agent_pos + DIRECTIONS[self.agent_dir]

    def step(self, action: str, direction: Optional[str] = None) -> dict:
        """
        Execute one action. Returns observation dict.
        """
        self.steps += 1
        msg = ""

        if action == "move":
            target = self.agent_pos + DIRECTIONS.get(direction, Position(0, 0))
            if not self._in_bounds(target):
                msg = "Bump! Wall or boundary ahead."
            elif self._is_wall(target):
                msg = "Bump! You hit a wall."
            elif target == self.door_pos and not self.door_open:
                msg = "Bump! The door is locked."
            else:
                self.agent_pos = target
                msg = f"Moved {direction} to ({target.x}, {target.y})."

        elif action == "turn":
            if direction == "left":
                self.agent_dir_idx = (self.agent_dir_idx - 1) % 4
                msg = f"Turned left. Now facing {self.agent_dir}."
            elif direction == "right":
                self.agent_dir_idx = (self.agent_dir_idx + 1) % 4
                msg = f"Turned right. Now facing {self.agent_dir}."
            else:
                msg = "Invalid turn direction. Use left or right."

        elif action == "look":
            # Look around in all 4 directions and report what is visible
            view = {}
            for d_name, delta in DIRECTIONS.items():
                p = self.agent_pos + delta
                if not self._in_bounds(p):
                    view[d_name] = "boundary"
                elif self._is_wall(p):
                    view[d_name] = "wall"
                elif p == self.door_pos:
                    view[d_name] = "door" + (" (open)" if self.door_open else " (locked)")
                elif p == self.key_pos and not self.has_key:
                    view[d_name] = "key"
                else:
                    view[d_name] = "floor"
            msg = f"You look around: {view}"

        elif action == "pick_up":
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                msg = "You picked up the key!"
            else:
                msg = "Nothing to pick up here."

        elif action == "open_door":
            if self.agent_pos == self.door_pos or self._look_ahead() == self.door_pos:
                if self.has_key and not self.door_open:
                    self.door_open = True
                    msg = "You unlocked and opened the door!"
                elif self.door_open:
                    msg = "The door is already open."
                else:
                    msg = "The door is locked. You need a key."
            else:
                msg = "You are not near the door."

        else:
            msg = f"Unknown action: {action}"

        return self.get_observation(message=msg)

    def get_observation(self, message: str = "") -> dict:
        """Return structured observation for the LLM agent."""
        ahead = self._look_ahead()
        if not self._in_bounds(ahead):
            ahead_desc = "boundary"
        elif self._is_wall(ahead):
            ahead_desc = "wall"
        elif ahead == self.door_pos:
            ahead_desc = "door" + (" (open)" if self.door_open else " (locked)")
        elif ahead == self.key_pos and not self.has_key:
            ahead_desc = "key"
        else:
            ahead_desc = "floor"

        return {
            "position": {"x": self.agent_pos.x, "y": self.agent_pos.y},
            "facing": self.agent_dir,
            "ahead": ahead_desc,
            "inventory": {"key": self.has_key},
            "door_open": self.door_open,
            "steps_remaining": self.max_steps - self.steps,
            "message": message,
            "goal_reached": self.door_open and self.agent_pos == self.door_pos,
        }

    def render(self) -> str:
        """ASCII render of the world."""
        lines = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                p = Position(x, y)
                if p == self.agent_pos:
                    symbols = {"N": "^", "E": ">", "S": "v", "W": "<"}
                    row += symbols[self.agent_dir]
                elif p == self.door_pos:
                    row += "D" if not self.door_open else "="
                elif p == self.key_pos and not self.has_key:
                    row += "K"
                elif p in self.walls:
                    row += "#"
                else:
                    row += "."
            lines.append(row)
        return "\n".join(lines)

    def is_done(self) -> bool:
        return self.door_open and self.agent_pos == self.door_pos or self.steps >= self.max_steps
