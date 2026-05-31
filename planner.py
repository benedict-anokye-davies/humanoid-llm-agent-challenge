"""A* pathfinding as a tool the agent can invoke."""

import heapq
from typing import Dict, List, Optional, Set, Tuple
from world import Pos


def astar_path(start: Pos, goal: Pos, walls: Set[Tuple[int, int]], width: int = 8, height: int = 8) -> Optional[List[str]]:
    """
    A* on grid. Returns list of directions (N,E,S,W) or None.
    The agent calls this as a tool: 'plan_route(target)'.
    """
    if start == goal:
        return []

    open_set = [(0, 0, start, [])]
    visited = set()

    while open_set:
        _, _, current, path = heapq.heappop(open_set)
        if (current.x, current.y) in visited:
            continue
        visited.add((current.x, current.y))

        if current == goal:
            return path

        for name, delta in [("N", Pos(0, -1)), ("E", Pos(1, 0)), ("S", Pos(0, 1)), ("W", Pos(-1, 0))]:
            nb = current + delta
            if not (0 <= nb.x < width and 0 <= nb.y < height):
                continue
            if (nb.x, nb.y) in walls:
                continue
            if (nb.x, nb.y) in visited:
                continue

            g = len(path) + 1
            h = abs(nb.x - goal.x) + abs(nb.y - goal.y)
            heapq.heappush(open_set, (g + h, g, nb, path + [name]))

    return None


def plan_route(start_pos: Dict, target_pos: Dict, known_walls: List[List[int]], world_width: int = 8, world_height: int = 8) -> Dict:
    """
    Tool callable by the LLM.
    Input: {"start": {"x": 0, "y": 0}, "target": {"x": 6, "y": 1}, "known_walls": [[1,1], [2,2]]}
    Output: {"path": ["E","E","S"], "distance": 3, "reachable": true}
    """
    start = Pos(start_pos["x"], start_pos["y"])
    goal = Pos(target_pos["x"], target_pos["y"])
    walls = {tuple(w) for w in known_walls}

    path = astar_path(start, goal, walls, world_width, world_height)
    if path is None:
        return {"path": [], "distance": -1, "reachable": False, "message": "No route found with current known walls."}

    return {"path": path, "distance": len(path), "reachable": True, "message": f"Route found: {len(path)} steps."}
