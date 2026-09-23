"""Independent, ordered multimodal Codex app-server worker.

No credential files are read by this client. Codex uses its existing ChatGPT login.
Only final answer, public event metadata, effective config and usage are exported.
No hidden reasoning or raw protocol transcripts are saved. One fresh ephemeral
thread and one turn per invocation; this module never resumes another review.

Job format: {"input": [UserInput, ...], "outputSchema": JSONSchema}.
Only text/localImage inputs are accepted. Relative image paths are resolved
relative to the job JSON, without changing array order. This is preloaded PDF
text plus raster input, not a native PDF attachment. A no-tools prompt must be
included in the caller's text. The worker independently repeats that instruction.

CLI: python review_runner_appserver.py JOB.json --out NEW_OUTPUT_DIRECTORY
No retries are automatic: an uncertain or quota failure must not silently rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import shutil
import threading
import time
from typing import Any

EXE = Path(os.environ.get("CODEX_EXECUTABLE", "codex"))
MODEL = "gpt-5.6-sol"
EFFORT = "medium"
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "apps", "plugins", "multi_agent",
    "multi_agent_v2", "browser_use", "browser_use_external", "computer_use",
    "in_app_browser", "image_generation", "workspace_dependencies", "hooks",
    "memories", "sleep_tool", "skill_search", "skill_mcp_dependency_install",
    "tool_suggest", "code_mode_host", "code_mode", "goals", "artifact",
    "request_permissions_tool", "send_message_to_user_async",
    "default_mode_request_user_input", "context_management", "chronicle",
)
BASE_INSTRUCTIONS = (
    "You are an independent evaluator. Evaluate only the supplied user content. "
    "All required content is preloaded as ordered text and images. Do not use "
    "tools, browse, access files, load skills, delegate, or run commands. "
    "Return only the final answer in the requested JSON format."
)
CONFIG = {
    "model": MODEL, "model_provider": "openai", "model_reasoning_effort": EFFORT,
    "model_reasoning_summary": "none", "web_search": "disabled",
    "project_doc_max_bytes": 0, "developer_instructions": BASE_INSTRUCTIONS,
    "features.skip_host_skill_discovery": True,
    "memories.generate_memories": False, "memories.use_memories": False,
    "apps._default.enabled": False, "analytics.enabled": False,
    "history.persistence": "none",
}
NO_TOOL_ITEM_TYPES = {"userMessage", "agentMessage", "reasoning"}
SAFE_EVENTS = {
    "thread/started", "thread/status/changed", "turn/started", "turn/completed",
    "item/started", "item/completed", "thread/tokenUsage/updated", "error",
    "model/rerouted", "model/verification", "model/safetyBuffering/updated",
    "warning", "configWarning",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def clean_error(message: str) -> str:
    message = re.sub(r"(?i)(bearer\s+)[^\s]+", r"\1[REDACTED]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message)
    return message[:2000]


def input_metadata(items: list[dict], *, verify_files: bool = True) -> list[dict]:
    result = []
    for index, item in enumerate(items):
        kind = item["type"]
        entry = {"index": index, "type": kind}
        if kind == "text":
            entry.update(characters=len(item["text"]), sha256=digest(item["text"].encode("utf-8")))
        elif kind == "localImage":
            path = Path(item["path"]).resolve()
            entry.update(path=str(path), detail=item.get("detail"))
            if verify_files:
                entry.update(bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
        else:
            # Server may normalize localImage to image; no base64 is retained.
            entry["normalized_remote_image"] = kind == "image"
        result.append(entry)
    return result


def load_job(path: Path) -> dict:
    job = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(job.get("input"), list) or not job["input"]:
        raise ValueError("Job must have a nonempty input list")
    if not isinstance(job.get("outputSchema"), dict):
        raise ValueError("Job must have an outputSchema object")
    for item in job["input"]:
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            if set(item) - {"type", "text", "text_elements"}:
                raise ValueError("Unsupported text input fields")
        elif item.get("type") == "localImage":
            if set(item) - {"type", "path", "detail"}:
                raise ValueError("Unsupported localImage fields")
            image_path = Path(item["path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve(strict=True)
            if not image_path.is_file():
                raise ValueError("Image path is not a file")
            item["path"] = str(image_path)
        else:
            raise ValueError("Only text and localImage inputs are supported")
    return job


def validate_schema(value: Any, schema: dict, location: str = "$", root: dict | None = None) -> None:
    """Validate the strict JSON-schema subset used by these evaluation jobs.

    Fail on unsupported assertions instead of silently skipping validation.
    Supports local $refs, object/array/scalar types, enum/const, anyOf/oneOf,
    numeric and length bounds. No third-party dependency is required.
    """
    root = schema if root is None else root
    allowed = {"$schema", "$id", "$defs", "definitions", "$ref", "title", "description",
               "type", "properties", "required", "additionalProperties", "items", "enum", "const",
               "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
               "minLength", "maxLength", "anyOf", "oneOf", "allOf"}
    if set(schema) - allowed:
        raise ValueError(f"Unsupported schema assertions at {location}: {sorted(set(schema) - allowed)}")
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise ValueError("Only local schema references are allowed")
        target = root
        for key in ref[2:].split("/"):
            target = target[key.replace("~1", "/").replace("~0", "~")]
        validate_schema(value, target, location, root)
    for combinator in ("anyOf", "oneOf", "allOf"):
        if combinator in schema:
            valid = 0
            for branch in schema[combinator]:
                try:
                    validate_schema(value, branch, location, root)
                    valid += 1
                except ValueError:
                    pass
            expected = len(schema[combinator]) if combinator == "allOf" else 1
            if (combinator == "anyOf" and not valid) or (combinator != "anyOf" and valid != expected):
                raise ValueError(f"{location}: {combinator} failed")
    types = schema.get("type")
    if isinstance(types, str):
        types = [types]
    checks = {"object": isinstance(value, dict), "array": isinstance(value, list),
              "string": isinstance(value, str), "integer": isinstance(value, int) and not isinstance(value, bool),
              "number": isinstance(value, (int, float)) and not isinstance(value, bool),
              "boolean": isinstance(value, bool), "null": value is None}
    if types and not any(checks.get(t, False) for t in types):
        raise ValueError(f"{location}: expected {types}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location}: value outside enum")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{location}: constant mismatch")
    if isinstance(value, dict):
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{location}: missing required properties")
        properties = schema.get("properties", {})
        extras = set(value) - set(properties)
        if schema.get("additionalProperties") is False and extras:
            raise ValueError(f"{location}: additional properties {sorted(extras)}")
        for key in value:
            sub = properties.get(key, schema.get("additionalProperties", {}))
            if isinstance(sub, dict):
                validate_schema(value[key], sub, location + "." + key, root)
    if isinstance(value, list):
        for index, item in enumerate(value):
            validate_schema(item, schema.get("items", {}), f"{location}[{index}]", root)
    for bound, compare in (("minimum", lambda a, b: a >= b), ("maximum", lambda a, b: a <= b),
                           ("exclusiveMinimum", lambda a, b: a > b), ("exclusiveMaximum", lambda a, b: a < b)):
        if bound in schema and not compare(value, schema[bound]):
            raise ValueError(f"{location}: {bound} failed")
    for bound, compare in (("minItems", lambda a, b: a >= b), ("maxItems", lambda a, b: a <= b),
                           ("minLength", lambda a, b: a >= b), ("maxLength", lambda a, b: a <= b)):
        if bound in schema and not compare(len(value), schema[bound]):
            raise ValueError(f"{location}: {bound} failed")


class AppServer:
    def __init__(self, work: Path, executable: Path = EXE):
        self.started = time.perf_counter()
        self.queue: queue.Queue = queue.Queue()
        self.next_id = 0
        self.events: list[dict] = []
        self.event_counts: Counter = Counter()
        self.tool_items: dict[str, dict] = {}
        self.errors: list[dict] = []
        self.transport_warnings: list[dict] = []
        self.usage = None
        self.final = None
        self.turn_status = None
        self.user_echo = None
        self.stderr_bytes = 0
        command = shutil.which(str(executable))
        if command is None:
            candidate = Path(executable).expanduser()
            if not candidate.is_file():
                raise FileNotFoundError("Codex CLI not found; use --executable or CODEX_EXECUTABLE")
            command = str(candidate.resolve())
        args = [command, "app-server", "--stdio"]
        for name, value in CONFIG.items():
            args += ["-c", name + "=" + json.dumps(value, ensure_ascii=False)]
        for feature in DISABLED_FEATURES:
            args += ["--disable", feature]
        env = os.environ.copy()
        # No token, auth-file access, or API-key forwarding. Let Codex authenticate.
        for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"):
            env.pop(key, None)
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                     errors="replace", env=env, cwd=work,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def stdout_reader():
            for line in self.proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    self.queue.put({"method": "nonJsonStdout", "params": {"characters": len(line)}})
                    continue
                self.queue.put(event)
            self.queue.put(None)
        def stderr_reader():
            # Do not retain raw diagnostics: they can contain context or credentials.
            for line in self.proc.stderr:
                self.stderr_bytes += len(line.encode("utf-8"))
        threading.Thread(target=stdout_reader, daemon=True).start()
        threading.Thread(target=stderr_reader, daemon=True).start()

    def send(self, method: str, params: dict, *, request: bool = True) -> int | None:
        event = {"method": method, "params": params}
        if request:
            self.next_id += 1
            event["id"] = self.next_id
        self.proc.stdin.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return event.get("id")

    def observe(self, event: dict) -> None:
        method = event.get("method", "")
        self.event_counts[method] += 1
        elapsed = time.perf_counter() - self.started
        params = event.get("params") or {}
        if "id" in event and method:
            self.errors.append({"kind": "unexpected_server_request", "method": method})
            request_item_id = params.get("itemId") or "server-request-" + str(event["id"])
            self.tool_items[request_item_id] = {
                "item_type": "unexpectedServerRequest", "method": method,
                "elapsed_seconds": elapsed}
            self.proc.stdin.write(json.dumps({"id": event["id"], "error": {
                "code": -32601, "message": "No tools or external requests are permitted by this client"}}) + "\n")
            self.proc.stdin.flush()
        # Native or nested function execution can also emit tool-specific events.
        # Keep only identifiers, never tool arguments or output.
        lower_method = method.lower()
        if (method.startswith("item/") and method not in ("item/started", "item/completed")
                and any(word in lower_method for word in (
                    "toolcall", "commandexecution", "filechange", "websearch", "imageview",
                    "imagegeneration", "requestapproval", "requestuserinput"))):
            key = params.get("itemId") or "tool-event-" + method
            self.tool_items.setdefault(key, {"item_type": "toolActivity", "item_id": key,
                                           "method": method, "elapsed_seconds": elapsed})
        if method not in SAFE_EVENTS:
            return  # In particular, discard all reasoning deltas and legacy raw events.
        metadata = {"elapsed_seconds": round(elapsed, 6), "method": method}
        if method in ("item/started", "item/completed"):
            item = params.get("item", {})
            kind = item.get("type")
            metadata.update(item_type=kind, item_id=item.get("id"))
            if kind not in NO_TOOL_ITEM_TYPES:
                key = item.get("id") or f"unknown-{len(self.tool_items)}"
                self.tool_items[key] = {"item_type": kind, "item_id": key, "elapsed_seconds": elapsed}
            if kind == "userMessage":
                self.user_echo = input_metadata(item.get("content", []), verify_files=False)
                metadata["content_types"] = [x["type"] for x in self.user_echo]
            if kind == "agentMessage":
                metadata["phase"] = item.get("phase")
                if method == "item/completed" and item.get("phase") in (None, "final_answer"):
                    self.final = item.get("text", "")
        elif method == "thread/tokenUsage/updated":
            self.usage = params.get("tokenUsage")
        elif method == "turn/completed":
            self.turn_status = params.get("turn", {}).get("status")
            metadata["status"] = self.turn_status
            error = params.get("turn", {}).get("error")
            if error:
                self.errors.append({"kind": "turn_error", "message": clean_error(str(error.get("message", "")))})
        elif method in ("error", "model/rerouted", "model/verification"):
            record = {"kind": method,
                      "message": clean_error(str(params.get("error", {}).get("message", "")))}
            if method == "error" and params.get("willRetry") is True:
                # The server explicitly says this is a recoverable transport
                # event. Completion, final JSON, input order and tool checks
                # are still mandatory; the coordinator starts no new attempt.
                record["will_retry"] = True
                self.transport_warnings.append(record)
                metadata["will_retry"] = True
            else:
                self.errors.append(record)
        elif method in ("warning", "configWarning"):
            metadata["message"] = clean_error(str(params.get("message", "")))
        self.events.append(metadata)

    def receive(self, timeout: float) -> dict:
        try:
            event = self.queue.get(timeout=max(0.01, timeout))
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for app-server") from exc
        if event is None:
            raise RuntimeError(f"App-server closed stdout (exit={self.proc.poll()})")
        if "method" in event:
            self.observe(event)
        return event

    def request(self, method: str, params: dict, timeout: float | None = 60) -> dict:
        request_id = self.send(method, params)
        deadline = time.perf_counter() + timeout if timeout is not None else None
        while deadline is None or time.perf_counter() < deadline:
            try:
                event = self.receive(deadline - time.perf_counter() if deadline else 60)
            except TimeoutError:
                if deadline is None:
                    continue
                raise
            if event.get("id") == request_id and "method" not in event:
                if "error" in event:
                    raise RuntimeError(method + ": " + clean_error(str(event["error"].get("message", "RPC error"))))
                return event.get("result", {})
        raise TimeoutError(method)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(timeout=5)


def run_job(job: dict, out: Path, executable: Path = EXE, timeout: float | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    work = out / "empty_worker_directory"
    work.mkdir()
    inputs = job["input"]
    config = {
        "prepared_utc": utc_now(), "model": MODEL, "effort": EFFORT,
        "transport": "app-server stdio JSON-RPC; ordered UserInput array in one turn",
        "auth_mode": "existing ChatGPT login managed by Codex",
        "fresh_ephemeral_thread": True, "disabled_features": list(DISABLED_FEATURES),
        "config_overrides": CONFIG, "input_items": input_metadata(inputs),
        "output_schema": job["outputSchema"], "cli": str(executable),
        "source": "https://developers.openai.com/codex/app-server",
        "hidden_reasoning_saved": False,
        "run_id": job.get("run_id"), "paper_id": job.get("paper_id"),
    }
    save(out / "config.json", config)
    server = None
    metrics = {"status": "failed", "started_utc": utc_now(), "thread_id": None, "turn_id": None,
               "run_id": job.get("run_id"), "paper_id": job.get("paper_id"), "worker_pid": os.getpid()}
    def checkpoint(stage: str) -> None:
        save(out / "checkpoint.json", {
            "stage": stage, "updated_utc": utc_now(), "worker_pid": os.getpid(),
            "app_server_pid": server.proc.pid if server else None,
            "run_id": job.get("run_id"), "paper_id": job.get("paper_id"),
            "thread_id": metrics["thread_id"], "turn_id": metrics["turn_id"],
            "usage": server.usage if server else None,
            "tool_call_count": len(server.tool_items) if server else 0,
            "elapsed_seconds": time.perf_counter() - server.started if server else 0})
    checkpoint("starting")
    try:
        server = AppServer(work, executable)
        checkpoint("preflight")
        initialize = server.request("initialize", {
            "clientInfo": {"name": "offline_review_runner", "version": "1.0.0"},
            "capabilities": {"experimentalApi": True, "optOutNotificationMethods": [
                "item/reasoning/textDelta", "item/reasoning/summaryTextDelta",
                "item/reasoning/summaryPartAdded", "item/agentMessage/delta"]}})
        server.send("initialized", {}, request=False)
        config["server_user_agent"] = initialize.get("userAgent")
        account = server.request("account/read", {"refreshToken": False})
        account_type = (account.get("account") or {}).get("type")
        config["verified_auth_type"] = account_type
        del account
        if account_type != "chatgpt":
            raise RuntimeError("Existing ChatGPT authentication is required")
        # Inspect resolved MCP server names only; no raw config is exported.
        resolved = server.request("config/read", {"cwd": str(work), "includeLayers": False})
        server_names = sorted((resolved.get("config", {}).get("mcp_servers") or {}).keys())
        del resolved
        overrides = dict(CONFIG)
        overrides.update({f"features.{feature}": False for feature in DISABLED_FEATURES})
        overrides.update({f"mcp_servers.{name}.enabled": False for name in server_names})
        config["disabled_mcp_servers"] = server_names
        thread = server.request("thread/start", {
            "model": MODEL, "modelProvider": "openai", "cwd": str(work),
            "ephemeral": True, "approvalPolicy": "never", "sandbox": "read-only",
            "baseInstructions": BASE_INSTRUCTIONS, "developerInstructions": BASE_INSTRUCTIONS,
            "config": overrides, "dynamicTools": [], "environments": [],
            "allowProviderModelFallback": False, "experimentalRawEvents": False})
        metrics["thread_id"] = thread["thread"]["id"]
        if (not thread["thread"].get("ephemeral") or thread["thread"].get("forkedFromId")
                or thread["thread"].get("turns")):
            raise RuntimeError("Thread is not a fresh, empty, independent ephemeral thread")
        config["independence_checks"] = {
            "ephemeral": True, "no_fork_parent": True, "initial_turn_count": 0,
            "created_with_thread_start": True,
            "cross_run_thread_id_uniqueness": "Must also be checked by the batch coordinator"}
        config["effective_thread"] = {key: thread.get(key) for key in (
            "model", "modelProvider", "reasoningEffort", "approvalPolicy", "sandbox",
            "instructionSources", "serviceTier")}
        if thread.get("model") != MODEL or thread.get("reasoningEffort") != EFFORT:
            raise RuntimeError("Model/effort was not honored by thread/start")
        if thread.get("instructionSources"):
            raise RuntimeError("Unexpected external instruction sources were loaded")
        save(out / "config.json", config)
        turn_start = time.perf_counter()
        checkpoint("submitting")
        response = server.request("turn/start", {
            "threadId": metrics["thread_id"], "input": inputs,
            "model": MODEL, "effort": EFFORT, "summary": "none",
            "outputSchema": job["outputSchema"], "approvalPolicy": "never",
            "environments": []}, timeout=None)
        metrics["turn_id"] = response["turn"]["id"]
        checkpoint("running")
        deadline = turn_start + timeout if timeout is not None else None
        while not server.turn_status:
            try:
                server.receive(deadline - time.perf_counter() if deadline else 60)
            except TimeoutError:
                if deadline is None:
                    checkpoint("running")
                    continue
                raise
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Turn time limit exceeded")
            if server.tool_items:
                server.send("turn/interrupt", {"threadId": metrics["thread_id"], "turnId": metrics["turn_id"]})
                raise RuntimeError("Tool or non-answer activity observed; review invalidated")
            if server.user_echo is not None:
                expected_echo = input_metadata(inputs, verify_files=False)
                if server.user_echo != expected_echo:
                    server.send("turn/interrupt", {"threadId": metrics["thread_id"], "turnId": metrics["turn_id"]})
                    raise RuntimeError("Server input echo did not preserve ordered content")
            checkpoint("running")
        metrics["turn_elapsed_seconds"] = time.perf_counter() - turn_start
        if server.turn_status != "completed" or server.errors:
            raise RuntimeError("Turn did not complete cleanly")
        if not server.final:
            raise RuntimeError("No final answer received")
        if server.user_echo != input_metadata(inputs, verify_files=False):
            raise RuntimeError("Exact ordered server input echo was not verified")
        metrics["ordered_input_echo_verified"] = True
        (out / "final.txt").write_text(server.final + "\n", encoding="utf-8")
        result = json.loads(server.final)
        validate_schema(result, job["outputSchema"])
        save(out / "final.json", result)
        metrics["status"] = "completed"
        metrics["output_schema_valid"] = True
    except Exception as exc:
        metrics["error"] = clean_error(str(exc))
        if server and server.tool_items:
            metrics["status"] = "quarantined"
    finally:
        if server is not None:
            server.close()
            metrics.update(elapsed_seconds=time.perf_counter() - server.started,
                           usage=server.usage, tool_call_count=len(server.tool_items),
                           tool_items=list(server.tool_items.values()), events=server.events,
                           event_counts=dict(server.event_counts), errors=server.errors,
                           transport_warnings=server.transport_warnings,
                           terminal_turn_status=server.turn_status,
                           input_echo_metadata=server.user_echo,
                           stderr_bytes_discarded=server.stderr_bytes,
                           process_exit_code=server.proc.returncode)
        metrics.update(completed_utc=utc_now(), hidden_reasoning_exported=False,
                       timing_scope="Worker startup through shutdown; input preprocessing excluded")
        save(out / "config.json", config)
        save(out / "metrics.json", metrics)
        checkpoint(metrics["status"])
    return metrics


def run_review(items: list[dict], output_schema: dict, out: Path, *,
               run_id: str, paper_id: str, executable: Path = EXE) -> dict:
    """Programmatic entry point; all localImage paths must be absolute.

    Does not load paper metadata or any existing review. Caller supplies frozen
    prompt text and preloaded anonymous content in exact order. No timeout or
    retry is imposed. Every invocation launches its own server and fresh thread.
    """
    for item in items:
        if item.get("type") not in ("text", "localImage"):
            raise ValueError("Only text/localImage inputs are allowed")
        if item["type"] == "localImage" and not Path(item["path"]).is_absolute():
            raise ValueError("run_review requires absolute image paths")
    return run_job({"input": items, "outputSchema": output_schema,
                    "run_id": run_id, "paper_id": paper_id}, out, executable)


def read_rate_limits(work: Path, executable: Path = EXE) -> dict:
    """Read account quota through existing login without starting a model turn.

    Returns native rateLimits/rateLimitsByLimitId. usedPercent is consumed quota;
    remaining = max(0, 100-usedPercent). Null values are unavailable, not zero.
    """
    server = AppServer(work, executable)
    try:
        server.request("initialize", {"clientInfo": {"name": "offline_review_quota", "version": "1.0.0"}})
        server.send("initialized", {}, request=False)
        result = server.request("account/rateLimits/read", {"excludeResetCreditDetails": True})
        return {"read_utc": utc_now(), **{key: result.get(key) for key in (
            "ordinaryUsageAllowed", "rateLimits", "rateLimitsByLimitId")}}
    finally:
        server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=EXE)
    parser.add_argument("--timeout", type=float, default=None,
                        help="Optional turn timeout; omitted means wait indefinitely without retrying")
    args = parser.parse_args()
    metrics = run_job(load_job(args.job.resolve()), args.out.resolve(), args.executable, args.timeout)
    print(json.dumps({key: metrics.get(key) for key in (
        "status", "error", "thread_id", "turn_id", "elapsed_seconds", "usage", "tool_call_count")}, ensure_ascii=True))
    raise SystemExit(0 if metrics["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
