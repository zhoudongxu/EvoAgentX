"""Qwen/vLLM chat client with bounded retries, streamed usage and per-task budgets."""
from __future__ import annotations

import asyncio
import json
import os
import time
from urllib.parse import urlparse

from .replay import digest


class TokenBudgetExceeded(RuntimeError):
    pass


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

    def _payload(self, system: str, prompt: str, component: str, seed: int, *, stream: bool):
        settings = self.config["components"][component]
        return {"model": self.config["name"], "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": settings["temperature"], "top_p": self.config["top_p"],
                "top_k": self.config["top_k"], "repetition_penalty": self.config["repetition_penalty"],
                "max_tokens": settings["max_tokens"], "seed": seed, "stream": stream,
                **({"stream_options": {"include_usage": True}} if stream else {})}

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

    async def stream(self, system: str, prompt: str, *, component: str, seed: int):
        import requests
        payload = self._payload(system, prompt, component, seed, stream=True)
        request_id = digest(payload)
        queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        key = os.environ.get(self.config.get("api_key_env", "COMPACTFLOW_API_KEY"), "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        async with self.semaphore:
            reservation = await self._reserve(payload)
            def worker():
                started = time.perf_counter()
                for attempt in range(self.config["max_retries"] + 1):
                    published = False
                    try:
                        with requests.post(self.base_url + "/chat/completions", headers=headers, json=payload,
                                           timeout=(self.config["connect_timeout_seconds"], self.config["timeout_seconds"]), stream=True) as response:
                            if response.status_code in self.config["retryable_status_codes"] and attempt < self.config["max_retries"]:
                                time.sleep(min(self.config["backoff_seconds"] * 2 ** attempt, self.config["max_backoff_seconds"]))
                                continue
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
                        # Never repeat a stream after consumers could observe it.
                        retryable = isinstance(error, (requests.Timeout, requests.ConnectionError))
                        if retryable and not published and attempt < self.config["max_retries"]:
                            time.sleep(min(self.config["backoff_seconds"] * 2 ** attempt, self.config["max_backoff_seconds"]))
                            continue
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
                self.reserved -= reservation
                if not usage_recorded:
                    # Account conservatively when failed requests omit usage.
                    self.used += reservation
                    self.records.append({"request_id": request_id, "component": component,
                                         "usage_missing": True, "reserved_token_upper_bound": reservation})
                if not job.done():
                    job.cancel()

    async def text(self, system: str, prompt: str, *, component: str, seed: int) -> str:
        chunks = []
        async for chunk in self.stream(system, prompt, component=component, seed=seed):
            chunks.append(chunk)
        return "".join(chunks)

    async def json(self, system: str, prompt: str, *, component: str, seed: int) -> dict:
        raw = await self.text(system + " Return one JSON object only, without Markdown fences.", prompt,
                              component=component, seed=seed)
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
