# LLM Agent in a Virtual World

> A partially observable maze-solving agent that builds a spatial memory, calls an A* planner as a tool, and navigates an 8×8 grid to find a key and unlock a door.

Built for the **Humanoid Software Engineering Internship** challenge.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
python main.py
```

The agent runs live in the terminal, printing its thoughts, actions, and its evolving mental map of the maze. A JSONL log is written to `demo_log.jsonl`.

Or open the **interactive web demo** (`demo.html`) in any browser — no server required. Shows side-by-side real world + agent's mental map with play/pause/step controls.

## System architecture

```
┌─────────────┐     observation     ┌─────────────┐
│   World     │ ──────────────────> │   Agent     │
│  (fog,      │                     │  (LLM +      │
│   walls)    │ <────────────────── │   memory +   │
│             │      action         │   planner)  │
└─────────────┘                     └─────────────┘
```

| Module | Responsibility |
|--------|---------------|
| `world.py` | Partially observable 8×8 grid; fog-of-war with occlusion |
| `memory.py` | AgentSpatialMemory: explored cells, known walls, objects, frontier |
| `planner.py` | A* pathfinder exposed as a `plan_route` tool callable by the LLM |
| `agent.py` | Harness: OpenAI function-calling API, retry logic, tool execution |
| `main.py` | Episode loop, coloured terminal render, JSONL logging |
| `test_all.py` | pytest: world physics, memory updates, A*, agent parsing |

## The challenge, solved

### 1. Partial observability (fog of war)

The agent only sees cells within a Manhattan-radius of 2, and only if a wall does not block the straight-line path. This mirrors real robotics: cameras and LiDAR have range limits and occlusion.

**Design choice:** We use a simple ray-cast occlusion check rather than giving the LLM raw pixel matrices. The LLM reasons about discrete symbolic observations (`wall`, `key`, `door_locked`) rather than interpreting visual features. This is closer to how indoor robots build occupancy grids.

### 2. Spatial memory

The agent maintains an `AgentSpatialMemory` object across turns:

- `explored`: every cell the agent has ever seen
- `walls`: known obstacles (persists across turns)
- `known_objects`: persistent map of key, door locations once observed
- `frontier`: unexplored cells adjacent to explored ones (classical exploration target)

This memory is serialized into a text summary and an ASCII grid map that the LLM reads every turn. The agent therefore **learns the map** rather than re-deriving it from scratch each observation.

### 3. Tool use: A* planner

Rather than emitting raw `move` commands for every step, the agent can call:

```json
<tool name="plan_route">{"target": {"x": 6, "y": 1}}</tool>
```

The harness executes A* over the agent's **known** walls, returns the shortest path, and the agent converts the first step into a `move` action. This separates **strategic planning** (LLM decides the goal) from **tactical pathfinding** (classical search on a known map).

This design is inspired by **Voyager** (Wang et al., 2023): the LLM writes high-level skill code, while low-level motor control is handled by deterministic algorithms.

### 4. Resilient harness

- **Function-calling API:** Structured tool calls reduce hallucination.
- **Retry loop:** Bad JSON or API errors trigger up to 3 retries with a fallback to `look`.
- **Markdown stripping:** The parser tolerates ` ```json ... ``` ` wrappers.
- **Validation:** Unknown actions default to `look`; out-of-bounds directions are sanitised.

### 5. ReAct-style reasoning loop

Each turn follows the ReAct pattern (Yao et al., 2022):

1. **Observe** — world emits partial observation + memory summary
2. **Reason** — LLM generates a `thought` string
3. **Act** — LLM emits a structured action or tool call
4. **Transition** — world updates, new observation emitted

The conversation history is truncated to the last 6 turns to stay within context limits while retaining short-term memory.

## The maze

```
0 > . . # . . . .
1 . . # # # # K .
2 . . . . . # . .
3 . # # # . # . .
4 . . . . . . . .
5 . # # # # # . .
6 . . . . . . D .
7 . . . . . . . .
```

- `>` = agent (facing East)
- `#` = wall
- `K` = key
- `D` = locked door
- `.` = floor

The agent must explore corridors, discover the key behind a wall cluster, plan a route, pick up the key, then navigate to the door.

## Design note: why this approach?

### Observation format: structured JSON, not pixels or prose

Robots do not feed raw camera frames into an LLM. They run SLAM, object detection, or semantic segmentation, then produce symbolic scene graphs. Our observation format (`position`, `facing`, `ahead`, `visible_cells`) is a minimal scene graph. It gives the LLM exactly the information a human player would need, without hallucination-inducing ambiguity.

### Action space: discrete, typed

A free-text action space invites the LLM to invent invalid commands (`"jump over wall"`, `"ask for hint"`). By restricting actions to `{move, turn, look, pick_up, open_door}` with typed parameters, the world layer can validate every input. Invalid actions produce informative error messages that the LLM learns from on the next turn.

### Tool use vs end-to-end control

End-to-end LLM control (emitting `move N` 20 times in a row) works on small grids but fails on larger ones because:
- The LLM loses count of steps
- It hallucinates walls that do not exist
- It cannot backtrack efficiently

By giving the LLM a `plan_route` tool, we offload path arithmetic to A* and let the LLM focus on **goal selection** and **re-planning when the map changes**. This is the same division of labour used in autonomous driving (LLM for intent, MPC for trajectory).

### Comparison to my own work

This harness pattern is the same one I use in **Asterion**, my open-source async multi-tool agent system. Both systems:
- Separate tool definitions from tool implementations
- Maintain persistent state (memory) across turns
- Validate and retry LLM outputs before executing side effects
- Log structured traces for debugging

The difference is scale: Asterion runs 10+ parallel tool calls over HTTP APIs with conversation trees; this challenge is a single-threaded grid-world proof of concept. The underlying architecture is identical.

## Optional extensions

- Replace `gpt-4o-mini` with any OpenAI-compatible endpoint (Claude via Anthropic SDK, local LLM via llama.cpp, or Fireworks kimi-k2p6).
- Add a patrolling guard that the agent must avoid, turning the maze into a pursuit-evasion game.
- Add multiple keys (colour-coded) and conditional doors, forcing the LLM to plan a fetch-quest sequence.
- Replace fog-of-war with a probabilistic occupancy grid: the agent maintains P(wall) for unseen cells rather than binary known/unknown.

## Author

Built by Benedict Anokye-Davies for the Humanoid Software Engineering Internship challenge. The agent-harness pattern is adapted from my open-source project [Asterion](https://github.com/benedict-anokye-davies/asterion-agent).
