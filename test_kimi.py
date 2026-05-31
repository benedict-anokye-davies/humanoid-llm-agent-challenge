import os
from world import GridWorld
from agent_kimi import KimiAgent

api_key = "fw_3NBTX5CFJRpT7BCPhJqSC6"
base_url = "https://api.fireworks.ai/inference/v1"
model = "accounts/fireworks/models/kimi-k2p6"

world = GridWorld()
agent = KimiAgent(model=model, api_key=api_key, base_url=base_url)

print("Testing KimiAgent with compact observations...")
for i in range(8):
    obs = world.get_local_observation()
    print(f"\n--- Step {i+1} ---")
    print(f"Pos: {obs['position']}, Ahead: {obs['ahead']}, Key: {obs['inventory']['key']}")
    action = agent.act(obs)
    print(f"Thought: {action.get('thought', '')}")
    print(f"Action: {action['action']} {action.get('direction', '')}")
    result = world.step(action["action"], action.get("direction"))
    print(f"Result: {result['message']}")
    print(world.render())
    if world.is_done():
        print("DONE!")
        break

print("\nTest complete.")
