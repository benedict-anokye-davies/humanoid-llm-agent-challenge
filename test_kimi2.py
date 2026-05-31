import os
from world import GridWorld
from agent_kimi import KimiAgent

api_key = "fw_3NBTX5CFJRpT7BCPhJqSC6"
base_url = "https://api.fireworks.ai/inference/v1"

world = GridWorld()
agent = KimiAgent(api_key=api_key, base_url=base_url)

print("Testing KimiAgent v2 with ASCII scene...")
for i in range(15):
    obs = world.get_local_observation()
    action = agent.act(obs)
    print(f"\nStep {i+1}: {action['action']} {action.get('direction','')} | {action['thought'][:60]}")
    result = world.step(action["action"], action.get("direction"))
    print(f"  -> {result['message'][:60]}")
    if world.is_done():
        print("SUCCESS!")
        break

if not world.is_done():
    print(f"\nNot done after 15 steps. Pos: {world.agent_pos.x},{world.agent_pos.y}, Key: {world.has_key}, Door: {world.door_open}")
