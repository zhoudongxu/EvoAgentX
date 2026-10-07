"""Qwen/vLLM chat client with bounded retries, streamed usage and per-task budgets."""
from __future__ import annotations

import asyncio
import inspect
from contextlib import aclosing
import json
import os
import re
import time
import threading
import weakref
from urllib.parse import urlparse

from .replay import digest


_MODEL_GATES = weakref.WeakKeyDictionary()


def model_gate(base_url, concurrency):
    loop = asyncio.get_running_loop()
    gates = _MODEL_GATES.setdefault(loop, {})
    # Every client for the same service shares one process-wide call bound.
    if base_url in gates and gates[base_url][0] != concurrency:
        raise ValueError("conflicting concurrency for one model service")
    if base_url not in gates:
        gates[base_url] = (concurrency, asyncio.Semaphore(concurrency))
    return gates[base_url][1]


class TokenBudgetExceeded(RuntimeError):
    pass


def format_options(client, method, response_format):
    """Explicit wire format; legacy test doubles may omit this optional API."""
    params = inspect.signature(getattr(client, method)).parameters
    return {"response_format": response_format} if response_format is not None and "response_format" in params else {}


def decoding_schema(schema):
    """Keep unsupported uniqueness checks in validate_spec, not the decoder."""
    if isinstance(schema, dict):
        return {k: decoding_schema(v) for k,v in schema.items() if k != 'uniqueItems'}
    if isinstance(schema, list):
        return [decoding_schema(v) for v in schema]
    return schema


def ndjson_output_regex(fields):
    """Constrain the wire format, never the value; retain per-field streaming."""
    string = r'"(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"'
    return r'\n'.join(re.escape('{"field":' + json.dumps(f) + ',"value":') + string + r'\}'
                       for f in fields) + r'\n?'


class ModelClient:
    def __init__(self, config: dict, *, token_budget: int = 65536):
        self.config = config
        self.base_url = os.environ.get("COMPACTFLOW_API_BASE", config["base_url"]).rstrip("/")
        if urlparse(self.base_url).scheme not in {"http", "https"}:
            raise ValueError("model base_url must be HTTP(S)")
        self.records = []
        self.budget = token_budget
        self.used = 0
        self.reserved = 0
        self.semaphore = asyncio.Semaphore(config.get("concurrency", 4))
        self.lock = asyncio.Lock()

    def _payload(self, system: str, prompt: str, component: str, seed: int, *, stream: bool, response_format=None):
        settings = self.config["components"][component]
        payload = {"model": self.config["name"], "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": settings["temperature"], "top_p": self.config["top_p"],
                "top_k": self.config["top_k"], "repetition_penalty": self.config["repetition_penalty"],
                "max_tokens": settings["max_tokens"], "seed": seed, "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {})}
        if self.config.get("typed_output_constraints", False) and response_format:
            kind = response_format["type"]
            if kind == "json_schema":
                payload["structured_outputs"] = {"json": decoding_schema(response_format["schema"])}
            elif kind == "ndjson_fields":
                fields = response_format["fields"]
                if not fields or len(set(fields)) != len(fields) or any(not isinstance(f,str) for f in fields):
                    raise ValueError("invalid typed output fields")
                payload["structured_outputs"] = {"regex": ndjson_output_regex(fields)}
            elif kind == "json_object":
                payload["response_format"] = {"type": "json_object"}
            elif kind != "text":
                raise ValueError("unknown model response format")
        return payload

    async def _reserve(self, payload):
        # Conservative byte-based bound for Qwen's byte-level tokenizer plus
        # chat-template overhead. Actual counts always come from provider usage.
        prompt_bound = sum(len(m["content"].encode()) for m in payload["messages"]) + 256
        amount = prompt_bound + payload["max_tokens"]
        if amount > self.config["context_length"]:
            raise TokenBudgetExceeded("context bound exceeded; inputs are never silently truncated")
        async with self.lock:
            if self.used + self.reserved + amount > self.budget:
                raise TokenBudgetExceeded("per-task input/output token budget exhausted")
            self.reserved += amount
        return amount

    async def stream(self, system: str, prompt: str, *, component: str, seed: int, response_format=None):
        import requests
        payload = self._payload(system, prompt, component, seed, stream=True, response_format=response_format)
        request_id = digest(payload)
        queue = asyncio.Queue()
        stop_requested = threading.Event()
        transport_started = threading.Event()
        loop = asyncio.get_running_loop()
        key = os.environ.get(self.config.get("api_key_env", "COMPACTFLOW_API_KEY"), "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        try:
            async with model_gate(self.base_url, self.config.get("concurrency", 4)):
                try:
                    reservation = await self._reserve(payload)
                except TokenBudgetExceeded as exc:
                    self.records.append({"request_id": request_id, "component": component, "seed": seed,
                                         "request_status": "not_sent", "error": str(exc),
                                         "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
                    raise
                def worker():
                    started = time.perf_counter()
                    # Once HTTP transport starts, an absent response cannot prove that
                    # inference was unbilled. Never silently repeat that request.
                    for attempt in range(1):
                        if stop_requested.is_set():break
                        published = False
                        try:
                            transport_started.set()
                            with requests.post(self.base_url + "/chat/completions", headers=headers, json=payload,
                                               timeout=(self.config["connect_timeout_seconds"], self.config["timeout_seconds"]), stream=True) as response:
                                response.raise_for_status()
                                usage = None
                                returned_model = None
                                for line in response.iter_lines():
                                    if not line or not line.startswith(b"data:"):
                                        continue
                                    raw = line[5:].strip()
                                    if raw == b"[DONE]":
                                        break
                                    value = json.loads(raw)
                                    if value.get("error"):
                                        raise ValueError("model server stream error: " + str(value["error"]))
                                    returned_model = value.get("model", returned_model)
                                    if value.get("usage"):
                                        usage = value["usage"]
                                    for choice in value.get("choices", []):
                                        text = choice.get("delta", {}).get("content") or ""
                                        if text:
                                            published = True
                                            loop.call_soon_threadsafe(queue.put_nowait, ("text", text))
                                if usage is None:
                                    raise ValueError("model server omitted token usage; cannot report token cost")
                                loop.call_soon_threadsafe(queue.put_nowait, ("usage", {
                                    "request_id": request_id, "component": component, "seed": seed,
                                    "attempt": attempt, "model": returned_model, "usage": usage,
                                    "elapsed_seconds": time.perf_counter() - started}))
                                break
                        except Exception as error:  # noqa: BLE001 - transport/JSON boundary
                            loop.call_soon_threadsafe(queue.put_nowait, ("error", error))
                            break
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))
                job = asyncio.create_task(asyncio.to_thread(worker))
                usage_recorded = False
                try:
                    while True:
                        kind, value = await queue.get()
                        if kind == "text":
                            yield value
                        elif kind == "usage":
                            self.records.append(value)
                            self.used += int(value["usage"]["prompt_tokens"]) + int(value["usage"]["completion_tokens"])
                            usage_recorded = True
                        elif kind == "error":
                            raise value
                        else:
                            break
                    await job
                finally:
                    stop_requested.set()
                    # A logical timeout stops publication, but the HTTP worker may
                    # already have billable inference in flight. Drain only that
                    # existing request to obtain provider usage; never start a retry
                    # or publish its late output. The existing request timeout bounds
                    # this accounting cleanup separately from logical execution.
                    if not usage_recorded:
                        try:
                            await asyncio.wait_for(asyncio.shield(job), self.config["timeout_seconds"])
                        except (asyncio.TimeoutError, asyncio.CancelledError):
                            pass
                        while not queue.empty():
                            kind, value = queue.get_nowait()
                            if kind == "usage" and not usage_recorded:
                                self.records.append({**value, "late_usage_after_cancellation": True})
                                self.used += int(value["usage"]["prompt_tokens"]) + int(value["usage"]["completion_tokens"])
                                usage_recorded = True
                    self.reserved -= reservation
                    if not usage_recorded and not transport_started.is_set():
                        self.records.append({"request_id": request_id, "component": component,
                                             "request_status": "not_sent",
                                             "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
                        usage_recorded = True
                    if not usage_recorded:
                        self.used += reservation
                        self.records.append({"request_id": request_id, "component": component,
                                             "usage_missing": True, "reserved_token_upper_bound": reservation})
                    if not job.done():
                        job.cancel()
        finally:
            # Cancellation may occur while waiting for a shared model slot,
            # before a worker or billable request exists.
            if not transport_started.is_set() and not any(r.get("request_id") == request_id for r in self.records):
                self.records.append({"request_id": request_id, "component": component,
                                     "request_status": "not_sent",
                                     "usage": {"prompt_tokens": 0, "completion_tokens": 0}})


    async def text(self, system: str, prompt: str, *, component: str, seed: int, response_format=None) -> str:
        chunks = []
        async with aclosing(self.stream(system, prompt, component=component, seed=seed, response_format=response_format)) as stream:
            async for chunk in stream:
                chunks.append(chunk)
        return "".join(chunks)

    async def json(self, system: str, prompt: str, *, component: str, seed: int, response_format=None) -> dict:
        raw = await self.text(system + " Return one JSON object only, without Markdown fences.", prompt,
                              component=component, seed=seed, response_format=response_format or {"type": "json_object"})
        value = raw.strip()
        if value.startswith("```"):
            value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("model must return a JSON object")
        return parsed

    def accounting(self) -> dict:
        by_component = {}
        for record in self.records:
            component = by_component.setdefault(record["component"], {"input": 0, "output": 0, "requests": 0, "unknown_usage": 0})
            component["requests"] += 1
            if record.get("usage_missing"):
                component["unknown_usage"] += 1
            else:
                component["input"] += record["usage"]["prompt_tokens"]
                component["output"] += record["usage"]["completion_tokens"]
        return {"components": by_component, "total_tokens": sum(v["input"] + v["output"] for v in by_component.values()),
                "usage_complete": not any(v["unknown_usage"] for v in by_component.values())}


class HashEmbedder:
    """Explicit deterministic offline encoder; never presented as learned retrieval."""
    def __init__(self, dimensions=256):
        from .policy import DeterministicTextEmbedder
        self.encoder = DeterministicTextEmbedder(dimension=dimensions)

    def embed(self, text):
        return self.encoder.embed(text)
