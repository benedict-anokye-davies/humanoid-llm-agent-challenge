"""Agent harness optimised for DeepSeek v4 Pro on Fireworks.

DeepSeek ignores response_format={"type": "json_object"} on Fireworks.
Solution: explicit JSON-only system prompt + regex extraction.
"""

import json
import os
import re
from typing import Dict, List, Optional

from memory import AgentSpatialMemory
from planner import plan_route

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT_DEEPSEEK = """You are an agent in an 8x8 grid maze.

GOAL: find the KEY (K), pick it up, then go to the DOOR (D) and open it.
Walls (#) block movement. You only see a 3x3 area around you.

VALID ACTIONS: move, turn, look, pick_up, open_door

You MUST respond with ONLY a JSON object. No text before or after.

Format: {"thought":"1 sentence","action":"...","direction":"..."}

Examples:
{"thought":"Path east clear","action":"move","direction":"E"}
{"thought":"Wall ahead, turn south","action":"turn","direction":"right"}
{"thought":"At key, pick up","action":"pick_up","direction":null}
{"thought":"Next to door, open it","action":"open_door","direction":null}

Rules:
1. Act every turn. Never just "think" or "consider".
2. If key is visible, move toward it.
3. If blocked, go around.
4. Pick up key BEFORE opening door.
"""


def _obs_to_ascii(obs: Dict, mem: AgentSpatialMemory) -> str:
    pos = obs["position"]
    facing = obs["facing"]
    nearby = obs.get("visible_cells", {})
    lines = [f"You are at ({pos['x']},{pos['y']}) facing {facing}."]
    lines.append("You see:")
    for dy in (-1, 0, 1):
        row = ""
        for dx in (-1, 0, 1):
            cx, cy = pos["x"] + dx, pos["y"] + dy
            key = f"({cx},{cy})"
            if dx == 0 and dy == 0:
                row += "A "
            elif key in nearby:
                desc = nearby[key]
                if "wall" in desc:
                    row += "# "
                elif "key" in desc:
                    row += "K "
                elif "door" in desc:
                    row += "D "
                else:
                    row += ". "
            else:
                row += "? "
        lines.append(row.rstrip())
    if mem.known_objects:
        lines.append("Known: " + str({str(k): v for k, v in list(mem.known_objects.items())[:3]}))
    lines.append(f"Inventory: key={'yes' if obs['inventory']['key'] else 'no'}, door_open={'yes' if obs['door_open'] else 'no'}")
    return "\n".join(lines)


class DeepSeekAgent:
    def __init__(self, model: str = "accounts/fireworks/models/deepseek-v4-pro",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.memory = AgentSpatialMemory()
        self.client = None

        if OpenAI is None:
            raise RuntimeError("openai package not installed")

        key = api_key or os.getenv("FIREWORKS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set FIREWORKS_API_KEY or OPENAI_API_KEY")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def act(self, observation: Dict) -> Dict:
        self.memory.update(observation)
        ascii_scene = _obs_to_ascii(observation, self.memory)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DEEPSEEK},
            {"role": "user", "content": ascii_scene},
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=80,
                )
                raw = response.choices[0].message.content or ""
                parsed = self._parse(raw)
                if parsed["action"] != "look" or attempt == 2:
                    return parsed
            except Exception as e:
                if attempt == 2:
                    return {"thought": f"LLM error: {e}", "action": "look", "direction": None}

        return {"thought": "Fallback", "action": "look", "direction": None}

    def _parse(self, raw: str) -> Dict:
        raw = raw.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            return {"thought": "No JSON found", "action": "look", "direction": None}

        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {"thought": "JSON parse fail", "action": "look", "direction": None}

        action = str(parsed.get("action", "look")).lower()
        direction = parsed.get("direction")
        if direction:
            direction = str(direction).upper() if str(direction).upper() in ("N", "E", "S", "W") else str(direction).lower()

        return {"thought": str(parsed.get("thought", "")), "action": action, "direction": direction}
