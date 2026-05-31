"""Run an LLM agent episode with live terminal render and JSONL logging."""

import argparse
import json
import os
import sys
from datetime import datetime

from world import GridWorld
from agent import LLMAgent


def color(text: str, code: str) -> str:
    """ANSI color wrapper."""
    codes = {
        "green": "\033[92m", "red": "\033[91m", "yellow": "\033[93m",
        "blue": "\033[94m", "cyan": "\033[96m", "bold": "\033[1m",
        "reset": "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"


def run_episode(agent: LLMAgent, world: GridWorld, verbose: bool = True, log_path: Optional[str] = None):
    log = []
    done = False

    if verbose:
        print(color("=" * 60, "bold"))
        print(color("LLM AGENT IN A VIRTUAL WORLD", "bold"))
        print(color("Partial observability | Spatial memory | A* tool use", "cyan"))
        print(color("=" * 60, "bold"))
        print(world.render())
        print(color("-" * 60, "bold"))

    while not done:
        obs = world.get_local_observation()
        step_record = {"step": world.steps + 1, "observation": obs}

        action = agent.act(obs)
        step_record["agent_output"] = action

        result = world.step(action["action"], action.get("direction"))
        step_record["result"] = result
        log.append(step_record)

        if verbose:
            print(f"\n{color(f'Step {world.steps}/{world.max_steps}', 'yellow')}")
            print(f"Thought: {color(action.get('thought', ''), 'cyan')}")
            print(f"Action: {color(action['action'], 'bold')} {action.get('direction', '')}")
            if action.get("tool_result"):
                print(f"Tool: {json.dumps(action['tool_result'], indent=2)}")
            print(f"Message: {result['message']}")
            print(world.render())
            # Show agent's known map
            print(color("Agent's mental map:", "blue"))
            print(agent.memory.get_grid_map())

        done = world.is_done()

    success = result.get("goal_reached", False)
    if verbose:
        print(color("\n" + "=" * 60, "bold"))
        if success:
            print(color("SUCCESS: Agent opened the door!", "green"))
        else:
            print(color("FAILED: Max steps reached.", "red"))
        print(color("=" * 60, "bold"))

    summary = {
        "success": success,
        "steps": world.steps,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": agent.model,
        "log": log,
    }

    if log_path:
        with open(log_path, "w") as f:
            for entry in log:
                f.write(json.dumps(entry) + "\n")
        if verbose:
            print(f"Log written to {log_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    parser.add_argument("--log", default="demo_log.jsonl", help="JSONL log path")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY environment variable.")
        sys.exit(1)

    world = GridWorld()
    agent = LLMAgent(model=args.model, api_key=api_key)
    summary = run_episode(agent, world, verbose=not args.quiet, log_path=args.log)

    if args.quiet:
        print(json.dumps({"success": summary["success"], "steps": summary["steps"]}))


if __name__ == "__main__":
    main()
