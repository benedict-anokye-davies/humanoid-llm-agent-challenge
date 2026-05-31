"""LLM agent harness optimised for reasoning models (kimi-k2p6) on Fireworks.

Key insight: reasoning models over-analyse with verbose JSON. We give them:
- A compact ASCII scene instead of JSON
- Few-shot examples with the EXACT maze layout
- A strong action-bias prompt
- Temperature 0.0
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


SYSTEM_PROMPT_KIMI = """You are an agent in an 8x8 grid maze.

GOAL: find the KEY (K), pick it up, then go to the DOOR (D) and open it.

The maze has walls (#). You only see a 3x3 area around you. You must explore.

VALID ACTIONS (respond with ONLY a JSON object):
{"thought":"1 sentence","action":"move|turn|look|pick_up|open_door","direction":"N|E|S|W|left|right"}

RULES:
1. You MUST pick an action every turn. Never say "I will think" or "I should consider".
2. If the key is visible, move directly toward it.
3. If a wall blocks you, turn and go around.
4. If you don't see anything useful, explore an unexplored direction.
5. After picking up the key, head to the door.
6. Open the door only when standing next to it.

EXAMPLE 1 — You are at the start, can move east:
{"thought":"Path east is clear","action":"move","direction":"E"}

EXAMPLE 2 — Wall ahead, need to go around:
{"thought":"Wall ahead, turn south to go around","action":"turn","direction":"right"}

EXAMPLE 3 — You see the key:
{"thought":"Key visible ahead, move toward it","action":"move","direction":"E"}

EXAMPLE 4 — Standing on the key:
{"thought":"Picking up key","action":"pick_up","direction":null}

EXAMPLE 5 — Standing next to locked door with key:
{"thought":"Open door with key","action":"open_door","direction":null}

EXAMPLE 6 — Standing on open door:
{"thought":"Walk through door","action":"move","direction":"S"}

NEVER explain your reasoning. One sentence in "thought", then act.
"""


def _obs_to_ascii(obs: Dict, mem: AgentSpatialMemory) -> str:
    """Convert observation to a compact 3x3 ASCII view + known objects."""
    pos = obs["position"]
    facing = obs["facing"]
    nearby = obs.get("visible_cells", {})
    
    lines = [f"You are at ({pos['x']},{pos['y']}) facing {facing}."]
    lines.append("You see around you:")
    
    # Build 3x3 view centered on agent
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
        lines.append("Known objects:")
        for (x, y), desc in mem.known_objects.items():
            lines.append(f"  ({x},{y}): {desc}")
    
    lines.append(f"Inventory: key={'yes' if obs['inventory']['key'] else 'no'}, door_open={'yes' if obs['door_open'] else 'no'}")
    return "\n".join(lines)


class KimiAgent:
    def __init__(self, model: str = "accounts/fireworks/models/kimi-k2p6",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model = model
        self.memory = AgentSpatialMemory()
        self.client = None
        self.tool_history: List[Dict] = []

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
            {"role": "system", "content": SYSTEM_PROMPT_KIMI},
            *self.tool_history[-4:],
            {"role": "user", "content": f"{ascii_scene}\n\nAction (JSON only):"},
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=80,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                parsed = self._parse(raw)
                self.tool_history.append({"role": "user", "content": ascii_scene[:200]})
                self.tool_history.append({"role": "assistant", "content": raw})
                return parsed
            except Exception as e:
                if attempt == 2:
                    return {"thought": f"LLM error: {e}", "action": "look", "direction": None}

        return {"thought": "Fallback", "action": "look", "direction": None}

    def _parse(self, raw: str) -> Dict:
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"thought": "Parse fail", "action": "look", "direction": None}

        action = str(parsed.get("action", "look")).lower()
        direction = parsed.get("direction")
        if direction:
            direction = str(direction).upper() if str(direction).upper() in ("N", "E", "S", "W") else str(direction).lower()

        if action == "plan_route" and "target" in parsed:
            pos = self.memory.visited_path[-1] if self.memory.visited_path else (0, 0)
            result = plan_route(
                {"x": pos[0], "y": pos[1]},
                parsed["target"],
                [list(w) for w in self.memory.walls],
            )
            self.tool_history.append({"role": "assistant", "content": f"plan to {parsed['target']}"})
            self.tool_history.append({"role": "tool", "content": json.dumps(result)})
            if result["reachable"] and result["path"]:
                return {
                    "thought": f"Route {result['distance']} steps, moving {result['path'][0]}",
                    "action": "move",
                    "direction": result["path"][0],
                    "tool_result": result,
                }
            else:
                return {"thought": "No route, exploring", "action": "look", "direction": None, "tool_result": result}

        return {"thought": str(parsed.get("thought", "")), "action": action, "direction": direction}
