"""Offline protocol/filter tests; never invoke Codex or a model."""
import io
import json
import os
from pathlib import Path
import queue
import time
import unittest
from unittest.mock import patch
import uuid
import tempfile

import review_runner_appserver as runner

_TEMP = tempfile.TemporaryDirectory(prefix="review-runner-tests-")
ROOT = Path(_TEMP.name)
SCHEMA = {"type": "object", "properties": {"score": {"type": "number", "minimum": 1, "maximum": 10}},
          "required": ["score"], "additionalProperties": False}


class FakeProc:
    pid = os.getpid()
    returncode = 0
    stdin = io.StringIO()


class FakeServer(runner.AppServer):
    variant = "ok"

    def __init__(self, work, executable):
        from collections import Counter
        self.started = time.perf_counter()
        self.queue = queue.Queue()
        self.next_id = 0
        self.events = []
        self.event_counts = Counter()
        self.tool_items = {}
        self.errors = []
        self.transport_warnings = []
        self.usage = self.final = self.turn_status = self.user_echo = None
        self.stderr_bytes = 0
        self.proc = FakeProc()

    def send(self, *args, **kwargs):
        return 1

    def request(self, method, params, timeout=60):
        if method == "initialize":
            return {"userAgent": "offline test"}
        if method == "account/read":
            return {"account": {"type": "chatgpt"}}
        if method == "config/read":
            return {"config": {}}
        if method == "thread/start":
            return {"thread": {"id": str(uuid.uuid4()), "ephemeral": True, "turns": [], "forkedFromId": None},
                    "model": runner.MODEL, "reasoningEffort": "low" if self.variant == "wrong_effort" else "medium",
                    "instructionSources": []}
        if method == "turn/start":
            items = list(params["input"])
            if self.variant == "wrong_order":
                items.reverse()
            def add(method, p):
                self.queue.put({"method": method, "params": p})
            add("item/completed", {"item": {"type": "userMessage", "id": "u", "content": items}})
            add("item/started", {"item": {"type": "reasoning", "id": "r", "content": ["SYNTHETIC_PRIVATE_SENTINEL"]}})
            add("item/reasoning/textDelta", {"delta": "SYNTHETIC_PRIVATE_SENTINEL"})
            add("item/completed", {"item": {"type": "reasoning", "id": "r", "content": ["SYNTHETIC_PRIVATE_SENTINEL"]}})
            if self.variant == "tool":
                for method in ("item/started", "item/completed"):
                    add(method, {"item": {"type": "dynamicToolCall", "id": "x", "tool": "functions.exec",
                                           "arguments": {"code": "SYNTHETIC_TOOL_ARGUMENT_SENTINEL"}}})
            if self.variant == "nested_tool_event":
                add("item/commandExecution/outputDelta", {"itemId": "nested", "delta": "SYNTHETIC_TOOL_ARGUMENT_SENTINEL"})
            score = 11 if self.variant == "invalid_schema" else 7
            if self.variant in ("recovered_connection", "fatal_connection", "unknown_connection"):
                params = {"error": {"message": "Synthetic connection event"}}
                if self.variant != "unknown_connection":
                    params["willRetry"] = self.variant == "recovered_connection"
                add("error", params)
            add("item/completed", {"item": {"type": "agentMessage", "id": "a", "phase": "final_answer",
                                            "text": json.dumps({"score": score})}})
            add("thread/tokenUsage/updated", {"tokenUsage": {"total": {"inputTokens": 3, "outputTokens": 4,
                                                                        "reasoningOutputTokens": 2, "totalTokens": 7}}})
            add("turn/completed", {"turn": {"status": "completed"}})
            return {"turn": {"id": "synthetic-turn"}}
        raise AssertionError(method)

    def close(self):
        pass


class Tests(unittest.TestCase):
    def run_variant(self, variant):
        FakeServer.variant = variant
        target = ROOT / variant
        with patch.object(runner, "AppServer", FakeServer):
            metrics = runner.run_review([
                {"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
                SCHEMA, target, run_id="test", paper_id="synthetic")
        serialized = "\n".join(p.read_text(encoding="utf-8") for p in target.glob("*.json"))
        self.assertNotIn("SYNTHETIC_PRIVATE_SENTINEL", serialized)
        self.assertNotIn("SYNTHETIC_TOOL_ARGUMENT_SENTINEL", serialized)
        return metrics

    def test_success_and_order(self):
        result = self.run_variant("ok")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["ordered_input_echo_verified"])
        self.assertEqual(result["tool_call_count"], 0)
        self.assertEqual(result["usage"]["total"]["reasoningOutputTokens"], 2)

    def test_order_rejected(self):
        result = self.run_variant("wrong_order")
        self.assertEqual(result["status"], "failed")
        self.assertIn("preserve ordered", result["error"])

    def test_tool_quarantined(self):
        result = self.run_variant("tool")
        self.assertEqual(result["status"], "quarantined")
        self.assertEqual(result["tool_call_count"], 1)

    def test_nested_tool_event_quarantined(self):
        result = self.run_variant("nested_tool_event")
        self.assertEqual(result["status"], "quarantined")
        self.assertEqual(result["tool_call_count"], 1)

    def test_wrong_effort_rejected(self):
        result = self.run_variant("wrong_effort")
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["turn_id"])

    def test_invalid_schema_rejected(self):
        result = self.run_variant("invalid_schema")
        self.assertEqual(result["status"], "failed")
        self.assertFalse((ROOT / "invalid_schema" / "final.json").exists())

    def test_recovered_connection_can_complete(self):
        result = self.run_variant("recovered_connection")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["transport_warnings"]), 1)
        self.assertEqual(result["terminal_turn_status"], "completed")

    def test_fatal_connection_stays_failed(self):
        result = self.run_variant("fatal_connection")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["errors"])

    def test_unknown_connection_stays_failed(self):
        result = self.run_variant("unknown_connection")
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["errors"])


if __name__ == "__main__":
    ROOT.mkdir(parents=True,exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    runner.save(ROOT / "summary.json", {"tests_run": result.testsRun,
                "failures": len(result.failures), "errors": len(result.errors), "model_calls": 0})
    raise SystemExit(not result.wasSuccessful())
