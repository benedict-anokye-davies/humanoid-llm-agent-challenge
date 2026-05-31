"""LLM agent harness with tool use, memory, and resilient parsing."""

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


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "plan_route",
            "description": "Find the shortest path to a target position using A* search on the agent's known map.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        "required": ["x", "y"],
                    }
                },
                "required": ["target"],
            },
        },
    }
]


SYSTEM_PROMPT = """You are an autonomous agent navigating a partially observable 2D maze.

GOAL: find the key (K), then unlock and open the door (D).

You CANNOT see the whole map. You have a vision radius of 2 cells. You must explore to discover walls and objects.

RULES:
1. Use 'look' to gather observations, then update your memory.
2. Use 'plan_route' tool to compute shortest paths over your known map.
3. Pick up the key BEFORE opening the door.
4. You must be adjacent to the door to open it.
5. If a planned path is blocked by an unknown wall, backtrack and explore.

ACTION FORMAT:
Respond with a JSON object:
{
  "thought": "1-2 sentence reasoning",
  "action": "move | turn | look | pick_up | open_door",
  "direction": "N | E | S | W | left | right"  // for move or turn
}

Or call a tool by wrapping it in <tool> tags:
<tool name="plan_route">{"target": {"x": 6, "y": 1}}</tool>
"""


class LLMAgent:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.memory = AgentSpatialMemory()
        self.client = None
        self.tool_history: List[Dict] = []

        if OpenAI is None:
            raise RuntimeError("openai package not installed. Run: pip install openai")

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=key)

    def act(self, observation: Dict) -> Dict:
        """Main entry: observe, reason, act or call tool."""
        self.memory.update(observation)

        obs_text = json.dumps(observation, indent=2)
        mem_text = self.memory.get_memory_summary()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._build_history(),
            {
                "role": "user",
                "content": f"CURRENT OBSERVATION:\n{obs_text}\n\nMEMORY:\n{mem_text}\n\nWhat is your next action or tool call?",
            },
        ]

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=400,
                )
                msg = response.choices[0].message

                # Tool call
                if msg.tool_calls:
                    return self._handle_tool_call(msg.tool_calls[0])

                # Plain text response
                raw = msg.content or ""
                return self._parse_response(raw)

            except Exception as e:
                if attempt == 2:
                    return {"thought": f"LLM failed after 3 attempts: {e}", "action": "look", "direction": None}

        return {"thought": "Fallback.", "action": "look", "direction": None}

    def _build_history(self) -> List[Dict]:
        """Recent history for LLM context."""
        return self.tool_history[-6:]

    def _handle_tool_call(self, tool_call) -> Dict:
        """Execute tool and return a special action dict."""
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments or "{}")

        if name == "plan_route":
            pos = self.memory.visited_path[-1] if self.memory.visited_path else (0, 0)
            result = plan_route(
                {"x": pos[0], "y": pos[1]},
                args["target"],
                [list(w) for w in self.memory.walls],
            )
            self.tool_history.append({"role": "assistant", "content": f"<tool name='plan_route'>{json.dumps(args)}</tool>"})
            self.tool_history.append({"role": "tool", "content": json.dumps(result)})
            # Convert first step of planned path into a move action
            if result["reachable"] and result["path"]:
                direction = result["path"][0]
                return {
                    "thought": f"Planned route to {args['target']}: {result['distance']} steps. Moving {direction}.",
                    "action": "move",
                    "direction": direction,
                    "tool_result": result,
                }
            else:
                return {
                    "thought": f"No route to {args['target']} with current map. Need to explore more.",
                    "action": "look",
                    "direction": None,
                    "tool_result": result,
                }

        return {"thought": f"Unknown tool {name}", "action": "look", "direction": None}

    def _parse_response(self, raw: str) -> Dict:
        """Extract JSON from markdown or raw text."""
        raw = raw.strip()
        # Remove markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        # Extract JSON object
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            return {"thought": "Failed to parse LLM response. Defaulting to look.", "action": "look", "direction": None}

        action = str(parsed.get("action", "look")).lower()
        direction = parsed.get("direction")
        if direction:
            direction = str(direction).upper() if str(direction).upper() in ("N", "E", "S", "W") else str(direction).lower()

        return {
            "thought": str(parsed.get("thought", "")),
            "action": action,
            "direction": direction,
        }
