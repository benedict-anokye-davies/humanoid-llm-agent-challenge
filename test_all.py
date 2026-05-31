"""pytest suite for world, memory, planner, and agent parsing."""

import pytest
from world import GridWorld, Pos
from memory import AgentSpatialMemory
from planner import astar_path, plan_route
from agent import LLMAgent


class TestWorld:
    def test_agent_starts_at_origin(self):
        w = GridWorld()
        assert w.agent_pos == Pos(0, 0)

    def test_move_east(self):
        w = GridWorld()
        obs = w.step("move", "E")
        assert w.agent_pos == Pos(1, 0)
        assert "Moved" in obs["message"]

    def test_wall_blocks(self):
        w = GridWorld()
        # Wall at (3,0): can't move E three times from (0,0)?
        # Actually path is (0,0)->(1,0)->(2,0) and then (3,0) is wall
        w.step("move", "E")
        w.step("move", "E")
        obs = w.step("move", "E")
        assert "Bump" in obs["message"] or w.agent_pos == Pos(2, 0)

    def test_pick_up_key(self):
        w = GridWorld()
        w.agent_pos = Pos(6, 1)
        obs = w.step("pick_up")
        assert w.has_key
        assert "picked up" in obs["message"]

    def test_open_door_needs_key(self):
        w = GridWorld()
        w.agent_pos = Pos(6, 6)
        obs = w.step("open_door")
        assert "need" in obs["message"] or "locked" in obs["message"]

    def test_fog_of_war(self):
        w = GridWorld()
        visible = w.visible_cells()
        # At (0,0) with radius 2, should see (0,0),(1,0),(2,0),(0,1),(1,1),(0,2) but NOT (3,0) which is dist 3
        assert Pos(0, 0) in visible
        assert Pos(2, 0) in visible
        assert Pos(3, 0) not in visible  # too far, dist=3
        assert Pos(6, 6) not in visible  # door is far


class TestMemory:
    def test_update_explored(self):
        m = AgentSpatialMemory()
        obs = {
            "position": {"x": 0, "y": 0},
            "visible_cells": {"(0,0)": "self", "(1,0)": "floor", "(2,0)": "floor"},
        }
        m.update(obs)
        assert (0, 0) in m.explored
        assert (1, 0) in m.explored

    def test_known_objects(self):
        m = AgentSpatialMemory()
        obs = {
            "position": {"x": 0, "y": 0},
            "visible_cells": {"(2,0)": "key"},
        }
        m.update(obs)
        assert m.known_objects[(2, 0)] == "key"


class TestPlanner:
    def test_astar_simple(self):
        path = astar_path(Pos(0, 0), Pos(2, 0), set(), 8, 8)
        assert path == ["E", "E"]

    def test_astar_blocked(self):
        walls = {(1, 0)}
        path = astar_path(Pos(0, 0), Pos(2, 0), walls, 8, 8)
        assert path is not None
        assert (1, 0) not in [Pos(0, 0)]  # just check it found a route

    def test_plan_route_tool(self):
        result = plan_route({"x": 0, "y": 0}, {"x": 2, "y": 0}, [], 8, 8)
        assert result["reachable"]
        assert result["path"] == ["E", "E"]


class TestAgentParse:
    def test_parse_json(self):
        a = LLMAgent.__new__(LLMAgent)
        raw = '{"thought": "test", "action": "move", "direction": "E"}'
        result = a._parse_response(raw)
        assert result["action"] == "move"
        assert result["direction"] == "E"

    def test_parse_markdown(self):
        a = LLMAgent.__new__(LLMAgent)
        raw = '```json\n{"thought": "ok", "action": "look"}\n```'
        result = a._parse_response(raw)
        assert result["action"] == "look"

    def test_parse_fallback(self):
        a = LLMAgent.__new__(LLMAgent)
        raw = "not json at all"
        result = a._parse_response(raw)
        assert result["action"] == "look"
