"""Deterministic mock LLM backend that plugs into LLMAgent.

This proves the harness architecture works end-to-end without API costs.
The mock backend simulates an LLM that returns pre-planned JSON actions,
exercising: observation formatting, memory updates, JSON parsing,
action validation, and world stepping.
"""

import json
from typing import Dict, List, Optional


class MockCompletion:
    """Simulates openai.ChatCompletion response."""
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = None


class MockLLMClient:
    """Deterministic LLM client compatible with LLMAgent._parse_response.
    
    Returns pre-planned actions as JSON strings. Each call advances
    the action index, simulating a real agent run through the harness.
    """

    # Pre-planned path through the 8x8 maze
    ACTIONS = [
        {"thought": "Path east is clear", "action": "move", "direction": "E"},
        {"thought": "Continue east", "action": "move", "direction": "E"},
        {"thought": "Wall ahead at (3,0), turn south", "action": "turn", "direction": "right"},
        {"thought": "Face south toward corridor", "action": "turn", "direction": "right"},
        {"thought": "Go back west to reach south corridor", "action": "move", "direction": "W"},
        {"thought": "Continue west to (0,0)", "action": "move", "direction": "W"},
        {"thought": "Turn south", "action": "turn", "direction": "right"},
        {"thought": "Move south to (0,1)", "action": "move", "direction": "S"},
        {"thought": "Move south to (0,2)", "action": "move", "direction": "S"},
        {"thought": "Turn east toward key", "action": "turn", "direction": "left"},
        {"thought": "East to (1,2)", "action": "move", "direction": "E"},
        {"thought": "East to (2,2)", "action": "move", "direction": "E"},
        {"thought": "East to (3,2)", "action": "move", "direction": "E"},
        {"thought": "East to (4,2)", "action": "move", "direction": "E"},
        {"thought": "Turn south toward door corridor", "action": "turn", "direction": "right"},
        {"thought": "South to (4,3)", "action": "move", "direction": "S"},
        {"thought": "South to (4,4)", "action": "move", "direction": "S"},
        {"thought": "Turn east", "action": "turn", "direction": "left"},
        {"thought": "East to (5,4)", "action": "move", "direction": "E"},
        {"thought": "East to (6,4)", "action": "move", "direction": "E"},
        {"thought": "Turn north toward key", "action": "turn", "direction": "left"},
        {"thought": "North to (6,3)", "action": "move", "direction": "N"},
        {"thought": "North to (6,2)", "action": "move", "direction": "N"},
        {"thought": "North to key at (6,1)", "action": "move", "direction": "N"},
        {"thought": "Pick up key", "action": "pick_up", "direction": None},
        {"thought": "Turn west", "action": "turn", "direction": "left"},
        {"thought": "Turn south", "action": "turn", "direction": "left"},
        {"thought": "South to (6,2)", "action": "move", "direction": "S"},
        {"thought": "South to (6,3)", "action": "move", "direction": "S"},
        {"thought": "South to (6,4)", "action": "move", "direction": "S"},
        {"thought": "South to (6,5)", "action": "move", "direction": "S"},
        {"thought": "South to door at (6,6)", "action": "move", "direction": "S"},
        {"thought": "Open door with key", "action": "open_door", "direction": None},
        {"thought": "Walk through open door", "action": "move", "direction": "S"},
    ]

    def __init__(self):
        self.idx = 0

    def chat_completions_create(self, **kwargs) -> MockCompletion:
        if self.idx < len(self.ACTIONS):
            action = self.ACTIONS[self.idx]
            self.idx += 1
            return MockCompletion(json.dumps(action))
        return MockCompletion(json.dumps({"thought": "Done", "action": "look", "direction": None}))

    def reset(self):
        self.idx = 0
