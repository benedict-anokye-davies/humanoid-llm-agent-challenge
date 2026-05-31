"""Run the LLM agent in the virtual world."""

import json
import os
from world import GridWorld
from agent import LLMAgent


def run_episode(verbose: bool = True, max_steps: int = 30) -> dict:
    world = GridWorld(width=5, height=5)
    api_key = os.getenv("OPENAI_API_KEY")
    agent = LLMAgent(api_key=api_key)

    log = []
    done = False

    if verbose:
        print("=" * 40)
        print("LLM Agent in a Virtual World")
        print("Goal: pick up the key (K), then open the door (D)")
        print("=" * 40)
        print(world.render())
        print("-" * 40)

    while not done:
        obs = world.get_observation()
        log.append({"step": world.steps, "observation": obs})

        if verbose:
            print(f"\nStep {world.steps + 1}/{max_steps}")
            print(f"Agent at {obs['position']}, facing {obs['facing']}, ahead: {obs['ahead']}")
            print(f"Inventory: {obs['inventory']}, Door open: {obs['door_open']}")

        action = agent.act(obs)
        action_name = action.get("action", "look")
        direction = action.get("direction")

        if verbose:
            print(f"Thought: {action.get('thought', '')}")
            print(f"Action: {action_name} {direction or ''}")

        result = world.step(action_name, direction)
        log.append({"step": world.steps, "result": result, "action": action})

        if verbose:
            print(result["message"])
            print(world.render())

        done = world.is_done()

    success = result.get("goal_reached", False)
    if verbose:
        print("\n" + "=" * 40)
        if success:
            print("SUCCESS: Agent opened the door!")
        else:
            print("FAILED: Max steps reached or agent got stuck.")
        print("=" * 40)

    return {
        "success": success,
        "steps": world.steps,
        "log": log,
        "world_final": world.render(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--save", type=str, default="run_log.json", help="Save log to file")
    args = parser.parse_args()

    result = run_episode(verbose=not args.quiet)

    with open(args.save, "w") as f:
        json.dump(result, f, indent=2)

    if not args.quiet:
        print(f"\nLog saved to {args.save}")
