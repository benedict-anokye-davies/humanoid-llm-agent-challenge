"""Run the agent through the REAL LLMAgent harness using a deterministic mock LLM.

This proves the architecture works end-to-end:
- Observation formatting → LLM → JSON parse → action validation → world step
- Memory updates, frontier tracking, planner tool calls all get exercised
"""

from world import GridWorld
from agent import LLMAgent
from mock_llm import MockLLMClient


def run_mock(verbose: bool = True) -> dict:
    world = GridWorld()
    mock_client = MockLLMClient()
    agent = LLMAgent(client=mock_client)
    log = []
    done = False

    if verbose:
        print("=" * 60)
        print("MOCK LLM → REAL HARNESS — End-to-End Integration Test")
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
            print(f"Thought: {action.get('thought', '')}")
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

    return {
        "success": success,
        "steps": world.steps,
        "log": log,
        "world_final": world.render(),
    }


if __name__ == "__main__":
    run_mock()
