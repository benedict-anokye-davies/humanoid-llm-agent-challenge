"""Hard-coded agent for testing the world + harness pipeline without LLM costs."""

from world import GridWorld, Pos, DIRECTIONS


class MockAgent:
    """Follows a fixed A* path through the 8x8 maze: go to key, pick up, go to door, open."""

    def __init__(self):
        self.plan = []
        self.plan_idx = 0

    def _build_plan(self):
        """Valid path through the maze from (0,0) to key (6,1) to door (6,6)."""
        # Start facing East. Trace verified against wall layout in world.py.
        self.plan = [
            {"action": "move", "direction": "E", "thought": "East toward corridor."},
            {"action": "move", "direction": "E", "thought": "East to (2,0)."},
            {"action": "turn", "direction": "left", "thought": "Face North."},
            {"action": "turn", "direction": "left", "thought": "Face West."},
            {"action": "move", "direction": "W", "thought": "Back to (1,0)."},
            {"action": "turn", "direction": "left", "thought": "Face South."},
            {"action": "move", "direction": "S", "thought": "South to (1,1)."},
            {"action": "move", "direction": "S", "thought": "South to (1,2)."},
            {"action": "turn", "direction": "left", "thought": "Face East."},
            {"action": "move", "direction": "E", "thought": "East to (2,2)."},
            {"action": "move", "direction": "E", "thought": "East to (3,2)."},
            {"action": "move", "direction": "E", "thought": "East to (4,2)."},
            {"action": "turn", "direction": "right", "thought": "Face South."},
            {"action": "move", "direction": "S", "thought": "South to (4,3)."},
            {"action": "move", "direction": "S", "thought": "South to (4,4)."},
            {"action": "turn", "direction": "left", "thought": "Face East."},
            {"action": "move", "direction": "E", "thought": "East to (5,4)."},
            {"action": "move", "direction": "E", "thought": "East to (6,4)."},
            {"action": "turn", "direction": "left", "thought": "Face North."},
            {"action": "move", "direction": "N", "thought": "North to (6,3)."},
            {"action": "move", "direction": "N", "thought": "North to (6,2)."},
            {"action": "move", "direction": "N", "thought": "North to key at (6,1)."},
            {"action": "pick_up", "direction": None, "thought": "Pick up the key."},
            {"action": "turn", "direction": "left", "thought": "Face West."},
            {"action": "turn", "direction": "left", "thought": "Face South."},
            {"action": "move", "direction": "S", "thought": "South to (6,2)."},
            {"action": "move", "direction": "S", "thought": "South to (6,3)."},
            {"action": "move", "direction": "S", "thought": "South to (6,4)."},
            {"action": "move", "direction": "S", "thought": "South to (6,5)."},
            {"action": "move", "direction": "S", "thought": "South to door at (6,6)."},
            {"action": "open_door", "direction": None, "thought": "Unlock and open the door."},
            {"action": "move", "direction": "S", "thought": "Walk through the open door."},
        ]

    def act(self, observation: dict) -> dict:
        if not self.plan:
            self._build_plan()
        if self.plan_idx < len(self.plan):
            action = self.plan[self.plan_idx]
            self.plan_idx += 1
            return action
        return {"thought": "Done.", "action": "look", "direction": None}


def run_mock(verbose: bool = True) -> dict:
    from world import GridWorld
    world = GridWorld()
    agent = MockAgent()
    agent._build_plan()
    log = []
    done = False

    if verbose:
        print("=" * 60)
        print("MOCK AGENT DEMO — 8×8 MAZE SOLVER")
        print("=" * 60)
        print(world.render())
        print("-" * 60)

    while not done:
        obs = world.get_local_observation()
        action = agent.act(obs)
        result = world.step(action["action"], action.get("direction"))
        log.append({"step": world.steps, "action": action, "result": result})

        if verbose:
            print(f"\nStep {world.steps}")
            print(f"Thought: {action['thought']}")
            print(f"Action: {action['action']} {action.get('direction') or ''}")
            print(f"Message: {result['message']}")
            print(world.render())

        done = world.is_done()

    success = result.get("goal_reached", False)
    if verbose:
        print("\n" + "=" * 60)
        if success:
            print("SUCCESS: Agent opened the door!")
        else:
            print("FAILED: Max steps reached or stuck.")
        print("=" * 60)
    return {"success": success, "steps": world.steps, "log": log}


if __name__ == "__main__":
    run_mock()
