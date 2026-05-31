# 🤖 LLM Agent in a Virtual World

> Partially observable maze-solving agent with spatial memory, A* tool use, and a ReAct reasoning loop.

Built for the **Humanoid Software Engineering Internship** challenge. The submission focuses on the **harness** — the interface between an LLM and an environment — not on graphics or world-building.

**Live demo:** open `demo.html` in any browser (no server needed) → [side-by-side world view + agent's mental map]

---

## Quick start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Run with OpenAI (validated backend)
export OPENAI_API_KEY="sk-..."
python main.py

# 3. Or run the deterministic mock agent (no API key)
# This exercises the FULL harness: memory, parse, validate, step
python mock_agent.py

# 4. Or open the interactive web demo
# Download demo.html, double-click — works offline
```

---

## System architecture

```
┌─────────────┐     observation     ┌─────────────┐
│   World     │ ──────────────────> │   Agent     │
│  (fog,      │                     │  (LLM +      │
│   walls)    │ <────────────────── │   memory +   │
│             │      action         │   planner)  │
└─────────────┘                     └─────────────┘
```

| Module | Lines | Responsibility |
|--------|------:|--------------|
| `world.py` | ~230 | 8×8 partially observable grid; fog-of-war with occlusion |
| `memory.py` | ~90 | AgentSpatialMemory: explored cells, known walls, objects, frontier |
| `planner.py` | ~60 | A* pathfinder; exposed as `plan_route` tool callable by the LLM |
| `agent.py` | ~240 | Harness: OpenAI function-calling API, retry logic, parser, validator |
| `mock_llm.py` | ~110 | Deterministic mock backend; plugs into `agent.py` to prove the harness |
| `main.py` | ~110 | Episode loop, coloured terminal render, JSONL logging |
| `test_all.py` | ~110 | 14 pytest tests covering world physics, memory, planner, parser |
| `.github/workflows/ci.yml` | ~20 | GitHub Actions: pytest on every push |

**Tech stack:** Python 3.11+, OpenAI SDK, pytest, GitHub Actions

---

## The challenge, solved

### 1. Partial observability (fog of war)

The agent only sees cells within a Manhattan-radius of 2, and only if a wall does not block the straight-line path. This mirrors real robotics: cameras and LiDAR have range limits and occlusion.

**Why this matters:** Most applicant submissions build fully observable grids. The agent sees everything. That's not how robots work. We built fog-of-war because real agents must explore.

### 2. Spatial memory

The agent maintains an `AgentSpatialMemory` object across turns:

- `explored`: every cell the agent has ever seen
- `walls`: known obstacles (persist across turns)
- `known_objects`: persistent map of key, door locations once observed
- `frontier`: unexplored cells adjacent to explored ones (classical exploration target)

The memory is serialised into an ASCII grid map that the LLM reads every turn. The agent **learns the map** rather than re-deriving it from scratch each observation.

### 3. Tool use: A* planner

Rather than emitting raw `move` commands for every step, the agent can call:

```json
{"action": "plan_route", "target": {"x": 6, "y": 1}}
```

The harness executes A* over the agent's **known** walls, returns the shortest path, and the agent converts the first step into a `move` action. This separates **strategic planning** (LLM decides the goal) from **tactical pathfinding** (classical search on a known map).

This is the same division of labour used in autonomous driving: LLM for intent, MPC for trajectory.

### 4. Resilient harness

- **Native function calling** (OpenAI): structured tool calls reduce hallucination
- **Retry loop:** bad JSON or API errors trigger up to 3 retries with fallback to `look`
- **Markdown stripping:** parser tolerates ` ```json ... ``` ` wrappers
- **Validation:** unknown actions default to `look`; out-of-bounds directions are sanitised
- **Injectable backend:** `LLMAgent` accepts any client implementing the same interface, enabling deterministic testing without API costs

### 5. ReAct-style reasoning loop

Each turn follows the ReAct pattern (Yao et al., 2022):

1. **Observe** — world emits partial observation + memory summary
2. **Reason** — LLM generates a brief thought
3. **Act** — LLM emits a structured action or tool call
4. **Transition** — world updates, new observation emitted

Conversation history is truncated to the last 6 turns to stay within context limits.

---

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

---

## Design note: why this approach?

### Observation format: structured JSON, not pixels or prose

Robots do not feed raw camera frames into an LLM. They run SLAM, object detection, or semantic segmentation, then produce symbolic scene graphs. Our observation format (`position`, `facing`, `ahead`, `visible_cells`) is a minimal scene graph. It gives the LLM exactly the information a human player would need, without hallucination-inducing ambiguity.

### Action space: discrete, typed

A free-text action space invites the LLM to invent invalid commands (`"jump over wall"`, `"ask for hint"`). By restricting actions to `{move, turn, look, pick_up, open_door}` with typed parameters, the world layer validates every input. Invalid actions produce informative error messages that the LLM learns from on the next turn.

### Tool use vs end-to-end control

End-to-end LLM control (emitting `move N` 20 times in a row) works on small grids but fails on larger ones because:
- The LLM loses count of steps
- It hallucinates walls that do not exist
- It cannot backtrack efficiently

By giving the LLM a `plan_route` tool, we offload path arithmetic to A* and let the LLM focus on **goal selection** and **re-planning when the map changes**.

---

## Backend validation

We tested multiple LLM backends to find the one that actually works for agentic tasks:

| Backend | Result | Why |
|---------|--------|-----|
| **gpt-4o-mini** | ✅ Works | Native function calling, decisive action, tuned for tool use |
| **kimi-k2p6 (Fireworks)** | ❌ Fails | Deep reasoning model over-analyses; 12 consecutive `move E` into a wall |
| **MockLLMClient** | ✅ Works | Deterministic backend proving the harness without API calls |

This is not a prompt-engineering failure — kimi-k2p6 is a reasoning model designed for math and long-context synthesis, not for discrete action selection. Picking the right model for the task is part of the engineering.

---

## How we compare to other submissions

We reviewed the other public repos for this same challenge. Here's what differentiates this submission:

| Feature | Us | [danushpravin](https://github.com/danushpravin/llm-agent-virtual-world) | [sukiraex](https://github.com/sukiraex/humanoid-challenge) | [CharSB](https://github.com/CharSB/humanoid-challenge) |
|---------|----|---------|---------|---------|
| **Partial observability** | ✅ Fog-of-war | ❌ Full visibility | ❌ Full visibility | Unknown |
| **Spatial memory** | ✅ Frontier tracking | ❌ No memory | Unknown | Unknown |
| **Tool use (A*)** | ✅ `plan_route` tool | ❌ Raw text actions | Unknown | Unknown |
| **Mock backend** | ✅ Plugs into real harness | ❌ Separate code path | ✅ Mentioned | Unknown |
| **Tests** | ✅ 14 pytest + CI | ❌ No tests | ✅ pytest | Unknown |
| **Web demo** | ✅ Static HTML, no server | ✅ Flask (needs server) | Unknown | Unknown |
| **Backend comparison** | ✅ gpt-4o-mini vs kimi-k2p6 | ❌ Claude only | Unknown | Unknown |

---

## Optional extensions

- Replace fog-of-war with a probabilistic occupancy grid: the agent maintains P(wall) for unseen cells rather than binary known/unknown
- Add a patrolling guard that the agent must avoid, turning the maze into a pursuit-evasion game
- Add multiple keys (colour-coded) and conditional doors, forcing the LLM to plan a fetch-quest sequence
- Wire up a local LLM via llama.cpp for fully offline operation

---

## Author

Built by **Benedict Anokye-Davies** for the Humanoid Software Engineering Internship challenge.

The agent-harness pattern is adapted from [Asterion](https://github.com/benedict-anokye-davies/asterion-agent), my open-source async multi-tool agent runtime.
