# LLM Agent in a Virtual World

A minimal but complete system that places an LLM agent into a 2D grid world, where it perceives its surroundings, reasons about goals, and takes actions in a loop.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Run the agent
python main.py
```

The agent will navigate the grid, pick up the key, and open the door. Output is printed to the terminal and saved to `run_log.json`.

## System overview

| Component | File | Responsibility |
|-----------|------|----------------|
| **World** | `world.py` | 2D grid environment, state transitions, observation generator |
| **Agent** | `agent.py` | LLM harness: observation → reasoning → JSON action |
| **Runner** | `main.py` | Episode loop, logging, success/fail reporting |

## The harness (what we care about)

The core challenge is the **interface between the LLM and the environment** — not the world itself.

### Observation format

The agent receives structured JSON:

```json
{
  "position": {"x": 0, "y": 0},
  "facing": "E",
  "ahead": "floor",
  "inventory": {"key": false},
  "door_open": false,
  "steps_remaining": 25,
  "message": "Moved E to (1, 0).",
  "goal_reached": false
}
```

**Why this representation?** It gives the LLM exactly what a human player would need: where am I, what is in front of me, what do I carry, and how much time is left. No raw pixel matrices, no hidden state.

### Action space

The agent responds with JSON:

```json
{
  "thought": "The key is to my east; I should move forward.",
  "action": "move",
  "direction": "E"
}
```

Supported actions: `move`, `turn`, `look`, `pick_up`, `open_door`. This discrete, typed interface prevents the LLM from hallucinating invalid commands.

### ReAct-style loop

Each turn follows:

1. **Observe** — world emits structured state  
2. **Reason** — LLM generates a brief thought  
3. **Act** — LLM emits a valid JSON action  
4. **Transition** — world updates, new observation emitted

The agent's conversation history is maintained across steps so the LLM retains context without re-deriving the map every turn.

## World design

```
# = wall   K = key   D = door   > = agent (facing East)

0  . . K . .
1  . # . # .
2  . . D . .
3  . # . . .
4  . . . . .
```

- Agent starts at (0,0) facing East.
- Key is at (2,0).
- Door is at (2,2), locked until the key is picked up.
- Walls create simple obstacles requiring navigation.

## Design choices & trade-offs

1. **2D grid over 3D or text** — Keeps observation space small and deterministic. The LLM reasons about (x,y) coordinates rather than parsing raw pixels or ambiguous prose.

2. **Discrete actions over free text** — Prevents the LLM from emitting invalid commands. The world layer validates every action before applying state changes.

3. **Structured JSON over natural language** — Removes parsing ambiguity. The harness extracts JSON from markdown code blocks if the LLM wraps its output, making the system robust to minor formatting variations.

4. **Local state over learned world model** — The agent does not build an internal map; it relies on the world to tell it what is ahead. This is simpler and sufficient for small grids. Scaling up would add a spatial memory layer (e.g. an explored-cell set).

## Optional extensions

- Replace `gpt-4o-mini` with any OpenAI-compatible model (Claude via Anthropic SDK, local LLM via llama.cpp).
- Add fog-of-war: agent only sees a 3×3 neighbourhood instead of the whole grid.
- Add multiple goals, hazards, or a second agent for multi-agent coordination.

## Author

Built for the Humanoid Software Engineering Internship challenge. The same agent-harness pattern powers my open-source project Asterion, an async multi-tool agent system.
