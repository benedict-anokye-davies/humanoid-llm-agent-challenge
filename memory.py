"""Agent spatial memory for partial observability."""

from typing import Dict, Set, Tuple, Optional
from world import Pos


class AgentSpatialMemory:
    """
    The agent's mental map. It only knows what it has seen.
    This mirrors real robotics: sensors give partial data; the robot builds a map.
    """

    def __init__(self):
        self.explored: Set[Tuple[int, int]] = set()
        self.walls: Set[Tuple[int, int]] = set()
        self.known_objects: Dict[Tuple[int, int], str] = {}  # pos -> description
        self.visited_path: list = []  # history of positions
        self.frontier: Set[Tuple[int, int]] = set()  # explored neighbours of unexplored cells

    def update(self, observation: Dict):
        """Incorporate a new observation into memory."""
        pos = observation["position"]
        self.visited_path.append((pos["x"], pos["y"]))
        self.explored.add((pos["x"], pos["y"]))

        for coord_str, desc in observation.get("visible_cells", {}).items():
            # coord_str is like "(3,4)"
            x, y = map(int, coord_str.strip("()").split(","))
            if desc == "wall":
                self.walls.add((x, y))
                self.explored.add((x, y))
            elif desc == "self":
                self.explored.add((x, y))
            else:
                self.explored.add((x, y))
                if desc not in ("floor", "self"):
                    self.known_objects[(x, y)] = desc

        self._recompute_frontier(observation)

    def _recompute_frontier(self, observation: Dict):
        """Cells adjacent to explored but not yet explored."""
        self.frontier = set()
        for (x, y) in self.explored:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb = (x + dx, y + dy)
                if nb not in self.explored and nb not in self.walls:
                    # Check bounds
                    if 0 <= nb[0] < 8 and 0 <= nb[1] < 8:
                        self.frontier.add(nb)

    def get_memory_summary(self) -> str:
        """Text summary for the LLM."""
        lines = [
            f"Explored {len(self.explored)} cells.",
            f"Known walls: {len(self.walls)}.",
            f"Known objects: {self.known_objects}.",
            f"Frontier (unexplored neighbours): {len(self.frontier)}.",
        ]
        if self.frontier:
            lines.append(f"Nearest frontier: {min(self.frontier, key=lambda p: abs(p[0]) + abs(p[1]))}")
        return "\n".join(lines)

    def get_grid_map(self, world_width: int = 8, world_height: int = 8) -> str:
        """ASCII map of what the agent knows."""
        lines = []
        for y in range(world_height):
            row = ""
            for x in range(world_width):
                p = (x, y)
                if p in self.walls:
                    row += "#"
                elif p == self.visited_path[-1] if self.visited_path else None:
                    row += "A"
                elif p in self.known_objects:
                    obj = self.known_objects[p]
                    if "door" in obj:
                        row += "D"
                    elif "key" in obj:
                        row += "K"
                    else:
                        row += "?"
                elif p in self.explored:
                    row += "."
                else:
                    row += " "
            lines.append(row)
        return "\n".join(lines)
