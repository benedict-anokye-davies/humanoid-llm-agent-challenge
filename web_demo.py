"""Generate a self-contained HTML demo of the agent trace.

Reads trace.json and produces an interactive visualisation.
"""

import json
import os


def generate_html(trace_path: str = "trace.json", output: str = "demo.html"):
    with open(trace_path) as f:
        trace = json.load(f)

    # Re-run and capture state per step using the real harness + mock LLM
    from world import GridWorld
    from agent import LLMAgent
    from mock_llm import MockLLMClient

    world = GridWorld()
    mock_client = MockLLMClient()
    agent = LLMAgent(client=mock_client)

    states = []

    for i in range(trace["steps"] + 1):
        obs = world.get_local_observation()
        action = agent.act(obs)

        state = {
            "step": i + 1,
            "agent_pos": {"x": world.agent_pos.x, "y": world.agent_pos.y},
            "agent_dir": world.agent_dir,
            "has_key": world.has_key,
            "door_open": world.door_open,
            "thought": action.get("thought", ""),
            "action": action.get("action", ""),
            "direction": action.get("direction", ""),
            "visible": [(p.x, p.y) for p in world.visible_cells()],
            "message": "",
        }

        result = world.step(action["action"], action.get("direction"))
        state["message"] = result["message"]

        # Build agent memory grid
        mem = agent.memory
        mem_grid = []
        for y in range(world.HEIGHT):
            row = []
            for x in range(world.WIDTH):
                p = (x, y)
                if p in mem.walls:
                    row.append("wall")
                elif p in mem.known_objects:
                    obj = mem.known_objects[p]
                    row.append(obj)
                elif p in mem.explored:
                    row.append("explored")
                else:
                    row.append("unknown")
            mem_grid.append(row)
        state["memory"] = mem_grid

        states.append(state)

    # Build static world layout
    world_layout = []
    for y in range(world.HEIGHT):
        row = []
        for x in range(world.WIDTH):
            p = (x, y)
            if p == (world.key_pos.x, world.key_pos.y):
                row.append("key")
            elif p == (world.door_pos.x, world.door_pos.y):
                row.append("door")
            elif p in [(w.x, w.y) for w in world.walls]:
                row.append("wall")
            else:
                row.append("floor")
        world_layout.append(row)

    states_json = json.dumps(states)
    layout_json = json.dumps(world_layout)

    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LLM Agent in a Virtual World — Demo</title>
<style>
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
.container { max-width: 1200px; margin: 0 auto; }
h1 { color: #38bdf8; margin-bottom: 5px; }
.subtitle { color: #94a3b8; margin-bottom: 20px; font-size: 14px; }
.panels { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.panel { background: #1e293b; border-radius: 12px; padding: 20px; }
.panel h2 { color: #f472b6; margin-top: 0; font-size: 16px; }
.grid { display: grid; grid-template-columns: repeat(8, 40px); gap: 2px; justify-content: center; }
.cell { width: 40px; height: 40px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 18px; font-weight: bold; transition: all 0.3s; }
.wall { background: #475569; }
.floor { background: #334155; }
.key { background: #fbbf24; color: #000; }
.door { background: #ef4444; color: #fff; }
.door-open { background: #22c55e; color: #fff; }
.agent-N::after { content: "↑"; }
.agent-E::after { content: "→"; }
.agent-S::after { content: "↓"; }
.agent-W::after { content: "←"; }
.agent { background: #38bdf8; color: #0f172a; font-size: 22px; }
.fog { background: #0f172a; border: 1px dashed #334155; }
.explored { background: #1e3a5f; }
.unknown { background: #0f172a; border: 1px dashed #1e293b; }
.controls { display: flex; gap: 10px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; }
button { background: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; }
button:hover { background: #7dd3fc; }
button:disabled { opacity: 0.5; cursor: not-allowed; }
.slider { flex: 1; min-width: 200px; }
.info { background: #1e293b; border-radius: 12px; padding: 20px; }
.info-row { display: flex; gap: 20px; margin-bottom: 10px; flex-wrap: wrap; }
.info-item { background: #0f172a; padding: 10px 15px; border-radius: 8px; }
.info-label { color: #94a3b8; font-size: 12px; text-transform: uppercase; }
.info-value { color: #f472b6; font-weight: bold; font-size: 16px; }
.log { background: #0f172a; border-radius: 8px; padding: 15px; max-height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 13px; }
.log-entry { margin-bottom: 8px; padding: 8px; border-radius: 6px; }
.log-entry.action { background: #1e3a5f; }
.log-entry.thought { background: #1e293b; color: #94a3b8; }
.log-entry.message { background: #0f172a; border-left: 3px solid #38bdf8; padding-left: 12px; }
.step-count { color: #f472b6; font-weight: bold; min-width: 50px; display: inline-block; }
</style>
</head>
<body>
<div class="container">
<h1>🤖 LLM Agent in a Virtual World</h1>
<div class="subtitle">Humanoid SWE Internship Challenge — Partial Observability, Spatial Memory, A* Tool Use</div>

<div class="controls">
<button id="playBtn" onclick="togglePlay()">▶ Play</button>
<button onclick="stepPrev()">← Prev</button>
<button onclick="stepNext()">Next →</button>
<input type="range" id="slider" class="slider" min="0" max=""" + str(len(states)-1) + """" value="0" oninput="setFrame(this.value)">
<span id="frameCounter">Step 0 / """ + str(len(states)-1) + """</span>
</div>

<div class="panels">
<div class="panel">
<h2>🌍 Real World</h2>
<div id="worldGrid" class="grid"></div>
</div>
<div class="panel">
<h2>🧠 Agent's Mental Map</h2>
<div id="memoryGrid" class="grid"></div>
</div>
</div>

<div class="info">
<div class="info-row">
<div class="info-item"><div class="info-label">Step</div><div class="info-value" id="stepNum">0</div></div>
<div class="info-item"><div class="info-label">Position</div><div class="info-value" id="pos">0,0</div></div>
<div class="info-item"><div class="info-label">Facing</div><div class="info-value" id="facing">E</div></div>
<div class="info-item"><div class="info-label">Has Key</div><div class="info-value" id="hasKey">❌</div></div>
<div class="info-item"><div class="info-label">Door</div><div class="info-value" id="doorStatus">🔒</div></div>
<div class="info-item"><div class="info-label">Visible</div><div class="info-value" id="visibleCount">0</div></div>
</div>
<div class="info-row">
<div class="info-item" style="flex: 1;"><div class="info-label">Thought</div><div class="info-value" id="thought" style="color: #94a3b8; font-weight: normal;">—</div></div>
</div>
<div class="info-row">
<div class="info-item" style="flex: 1;"><div class="info-label">Action</div><div class="info-value" id="action" style="color: #38bdf8;">—</div></div>
</div>
<div class="info-row">
<div class="info-item" style="flex: 1;"><div class="info-label">Result</div><div class="info-value" id="result" style="color: #fbbf24; font-weight: normal;">—</div></div>
</div>
</div>

<h3 style="color: #f472b6; margin-top: 20px;">📋 Event Log</h3>
<div id="log" class="log"></div>
</div>

<script>
const worldLayout = """ + layout_json + """;
const states = """ + states_json + """;
let currentFrame = 0;
let playing = false;
let timer = null;

function render() {
    const s = states[currentFrame];
    const worldEl = document.getElementById('worldGrid');
    const memEl = document.getElementById('memoryGrid');
    worldEl.innerHTML = '';
    memEl.innerHTML = '';
    const visibleSet = new Set(s.visible.map(p => p[0] + ',' + p[1]));
    for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
            const wCell = document.createElement('div');
            wCell.className = 'cell';
            const isVisible = visibleSet.has(x + ',' + y);
            if (s.agent_pos.x === x && s.agent_pos.y === y) {
                wCell.className += ' agent agent-' + s.agent_dir;
            } else if (worldLayout[y][x] === 'wall') {
                wCell.className += isVisible ? ' wall' : ' fog';
                if (isVisible) wCell.textContent = '#';
            } else if (worldLayout[y][x] === 'key' && !s.has_key) {
                wCell.className += isVisible ? ' key' : ' fog';
                if (isVisible) wCell.textContent = 'K';
            } else if (worldLayout[y][x] === 'door') {
                wCell.className += isVisible ? (s.door_open ? ' door-open' : ' door') : ' fog';
                if (isVisible) wCell.textContent = s.door_open ? '=' : 'D';
            } else {
                wCell.className += isVisible ? ' floor' : ' fog';
            }
            worldEl.appendChild(wCell);
            const mCell = document.createElement('div');
            mCell.className = 'cell';
            const mem = s.memory[y][x];
            if (s.agent_pos.x === x && s.agent_pos.y === y) {
                mCell.className += ' agent agent-' + s.agent_dir;
            } else if (mem === 'wall') {
                mCell.className += ' wall'; mCell.textContent = '#';
            } else if (mem === 'key') {
                mCell.className += ' key'; mCell.textContent = 'K';
            } else if (mem === 'door') {
                mCell.className += (s.door_open ? ' door-open' : ' door');
                mCell.textContent = s.door_open ? '=' : 'D';
            } else if (mem === 'explored') {
                mCell.className += ' explored';
            } else {
                mCell.className += ' unknown';
            }
            memEl.appendChild(mCell);
        }
    }
    document.getElementById('stepNum').textContent = currentFrame + 1;
    document.getElementById('pos').textContent = s.agent_pos.x + ',' + s.agent_pos.y;
    document.getElementById('facing').textContent = s.agent_dir;
    document.getElementById('hasKey').textContent = s.has_key ? '✅' : '❌';
    document.getElementById('doorStatus').textContent = s.door_open ? '🔓' : '🔒';
    document.getElementById('visibleCount').textContent = s.visible.length;
    document.getElementById('thought').textContent = s.thought || '—';
    document.getElementById('action').textContent = s.action + (s.direction ? ' ' + s.direction : '');
    document.getElementById('result').textContent = s.message || '—';
    document.getElementById('slider').value = currentFrame;
    document.getElementById('frameCounter').textContent = 'Step ' + (currentFrame + 1) + ' / ' + states.length;
    updateLog();
}

function updateLog() {
    const logEl = document.getElementById('log');
    logEl.innerHTML = '';
    for (let i = 0; i <= currentFrame; i++) {
        const s = states[i];
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        if (s.action) {
            entry.className += ' action';
            entry.innerHTML = '<span class="step-count">#' + (i+1) + '</span> <b>' + s.action.toUpperCase() + '</b> ' + (s.direction || '') + '<br><span style="color:#94a3b8">💭 ' + s.thought + '</span>';
            if (s.message) {
                const msg = document.createElement('div');
                msg.className = 'log-entry message';
                msg.textContent = '→ ' + s.message;
                entry.appendChild(msg);
            }
        }
        logEl.appendChild(entry);
    }
    logEl.scrollTop = logEl.scrollHeight;
}

function setFrame(n) {
    currentFrame = Math.max(0, Math.min(states.length - 1, parseInt(n)));
    render();
}
function stepNext() {
    if (currentFrame < states.length - 1) setFrame(currentFrame + 1);
}
function stepPrev() {
    if (currentFrame > 0) setFrame(currentFrame - 1);
}
function togglePlay() {
    playing = !playing;
    document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
    if (playing) {
        timer = setInterval(() => {
            if (currentFrame >= states.length - 1) { togglePlay(); return; }
            stepNext();
        }, 600);
    } else {
        clearInterval(timer);
    }
}
render();
</script>
</body>
</html>""")

    with open(output, "w") as f:
        f.write("".join(html_parts))

    abs_path = os.path.abspath(output)
    print(f"Demo generated: {abs_path}")
    return abs_path


if __name__ == "__main__":
    generate_html()
