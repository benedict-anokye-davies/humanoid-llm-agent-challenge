"""LLM agent harness: observation → reasoning → action."""

import json
import os
from typing import Optional


try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


SYSTEM_PROMPT = """You are an autonomous agent navigating a 2D grid world.

Your goal: find the key (K), then unlock and open the door (D).

OBSERVATION FORMAT (JSON):
- position: your (x, y) coordinates
- facing: N/E/S/W
- ahead: what is directly in front of you (floor, wall, key, door (locked/open), boundary)
- inventory: whether you carry the key
- door_open: whether the door has been unlocked
- steps_remaining: how many moves you have left
- message: result of your last action

ACTION FORMAT (JSON):
Respond ONLY with a JSON object containing:
{
  "thought": "brief reasoning about what to do next",
  "action": "move | turn | look | pick_up | open_door",
  "direction": "N | E | S | W | left | right"  // only for move or turn
}

RULES:
- You must pick_up the key BEFORE you can open_door.
- You must be adjacent to the door to open_door.
- You cannot walk through walls or boundaries.
- If the door is locked and ahead, use turn to face it, then open_door.
- Keep your thought concise (1-2 sentences).
"""


class LLMAgent:
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.history: list = []
        self.client = None

        if OpenAI is None:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )
        self.client = OpenAI(api_key=key)

    def act(self, observation: dict) -> dict:
        """Send observation to LLM, return parsed action."""
        obs_text = json.dumps(observation, indent=2)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": f"OBSERVATION:\n{obs_text}\n\nWhat is your next action?"},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or ""
            action = self._parse(raw)
            self.history.append({"role": "user", "content": f"OBSERVATION: {obs_text}"})
            self.history.append({"role": "assistant", "content": raw})
            return action
        except Exception as e:
            return {
                "thought": f"LLM call failed: {e}",
                "action": "look",
                "direction": None,
            }

    def _parse(self, raw: str) -> dict:
        """Extract JSON from markdown or raw text."""
        raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.startswith("```"):
            raw = raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: heuristic extraction
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
            except (ValueError, json.JSONDecodeError):
                return {"thought": "Failed to parse JSON, defaulting to look.", "action": "look", "direction": None}

        return {
            "thought": str(parsed.get("thought", "")),
            "action": str(parsed.get("action", "look")),
            "direction": parsed.get("direction") if parsed.get("direction") is not None else None,
        }
