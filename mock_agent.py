"""Hard-coded agent for testing the world + harness pipeline without LLM costs."""

from world import GridWorld, Position, DIRECTIONS


class MockAgent:
    """Follows a fixed policy: go to key, pick up, move south, face door, open it, step through."""

    def __init__(self):
        self.plan = []
        self.plan_idx = 0

    def _build_plan(self, world: GridWorld):
        """Shortest valid path for the default 5x5 world."""
        # Start (0,0) facing E -> key at (2,0) -> door at (2,2)
        # Walls at (1,1), (3,1), (1,3)
        # Path: E, E, pick_up, move S, turn right (face S), open_door, move S
        self.plan = [
            {"action": "move", "direction": "E", "thought": "Heading east toward the key."},
            {"action": "move", "direction": "E", "thought": "Almost at the key."},
            {"action": "pick_up", "direction": None, "thought": "Pick up the key."},
            {"action": "move", "direction": "S", "thought": "Move south toward the door."},
            {"action": "turn", "direction": "right", "thought": "Face south toward the door."},
            {"action": "open_door", "direction": None, "thought": "Unlock and open the door with the key."},
            {"action": "move", "direction": "S", "thought": "Walk through the open door."},
        ]

    def act(self, observation: dict) -> dict:
        if not self.plan:
            self._build_plan(None)
        if self.plan_idx < len(self.plan):
            action = self.plan[self.plan_idx]
            self.plan_idx += 1
            return action
        return {"thought": "Done.", "action": "look", "direction": None}


def run_mock(verbose: bool = True) -> dict:
    from world import GridWorld
    world = GridWorld()
    agent = MockAgent()
    agent._build_plan(world)
    log = []
    done = False

    if verbose:
        print("=" * 40)
        print("MOCK AGENT DEMO (no LLM required)")
        print("=" * 40)
        print(world.render())
        print("-" * 40)

    while not done:
        obs = world.get_observation()
        action = agent.act(obs)
        result = world.step(action["action"], action.get("direction"))
        log.append({"step": world.steps, "action": action, "result": result})

        if verbose:
            print(f"\nStep {world.steps}")
            print(f"Thought: {action['thought']}")
            print(f"Action: {action['action']} {action.get('direction') or ''}")
            print(result["message"])
            print(world.render())

        done = world.is_done()

    success = result.get("goal_reached", False)
    if verbose:
        print("\n" + "=" * 40)
        print("SUCCESS!" if success else "FAILED")
        print("=" * 40)
    return {"success": success, "steps": world.steps, "log": log}


if __name__ == "__main__":
    run_mock()
