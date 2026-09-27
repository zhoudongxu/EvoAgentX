"""Strict, argument-keyed call recording and timed replay (paper Appendix G.4)."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from .compiler import GFRGCompiler
from .runtime import _deep_merge
from .schema import (
    CallSpec,
    Complete,
    DataDependency,
    EffectDependency,
    Failure,
    Partial,
    StreamContract,
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def call_key(workflow_id: str, call_id: str, arguments: dict) -> str:
    return digest({"workflow_id": workflow_id, "call_id": call_id, "arguments": arguments})


class ReplayMiss(ValueError):
    """Arguments differ from the recorded execution; never call a live service."""


class ReplayBundle:
    def __init__(self, workflow_id: str, *, records: dict | None = None, metadata: dict | None = None):
        self.workflow_id = workflow_id
        self.records = copy.deepcopy(records or {})
        self.metadata = copy.deepcopy(metadata or {})

    def wrap(self, call: CallSpec) -> CallSpec:
        async def recording(**arguments):
            started = time.perf_counter()
            events = []
            key = call_key(self.workflow_id, call.id, arguments)
            if key in self.records:
                raise ValueError(f"duplicate recorded call {call.id}; use a distinct workflow/attempt ID")
            # Publish the record before yielding. Consumers may stop immediately
            # at the terminal event without resuming this generator.
            self.records[key] = {"workflow_id": self.workflow_id, "call_id": call.id,
                                 "arguments": copy.deepcopy(arguments), "events": events}
            async def get_events():
                if inspect.isasyncgenfunction(call.target):
                    value = call.target(**arguments)
                elif inspect.iscoroutinefunction(call.target):
                    value = await call.target(**arguments)
                else:
                    value = await asyncio.to_thread(call.target, **arguments)
                if hasattr(value, "__aiter__"):
                    async for event in value:
                        yield event
                else:
                    yield value if isinstance(value, (Partial, Complete, Failure)) else Complete(value)
            terminal = False
            accumulated = {}
            try:
                async for event in get_events():
                    if isinstance(event, (Partial, Complete)):
                        accumulated = _deep_merge(accumulated, event.data)
                    item = {"sequence": len(events), "at": time.perf_counter() - started,
                            "kind": type(event).__name__.lower()}
                    if isinstance(event, Failure):
                        item["error"] = str(event.error)
                    else:
                        item.update(data=copy.deepcopy(dict(event.data)), effects=list(event.effects))
                        if isinstance(event, Partial):
                            item["stable_fields"] = list(event.stable_fields)
                    events.append(item)
                    if isinstance(event, (Complete, Failure)):
                        terminal = True
                    yield event
                    if terminal:
                        break
                if not terminal:
                    event = Complete(accumulated)
                    events.append({"sequence": len(events), "at": time.perf_counter() - started,
                                   "kind": "complete", "data": accumulated, "effects": []})
                    yield event
            except Exception as error:  # noqa: BLE001 - arbitrary tool boundary
                events.append({"sequence": len(events), "at": time.perf_counter() - started,
                               "kind": "failure", "error": f"{type(error).__name__}: {error}"})
                yield Failure(str(error))
            finally:
                self.records[key] = {"workflow_id": self.workflow_id, "call_id": call.id,
                                     "arguments": copy.deepcopy(arguments), "events": events}
        return replace(call, target=recording)

    def bind(self, call_id: str, *, time_scale: float = 1.0, batch_size: int = 1):
        if not math.isfinite(time_scale) or time_scale < 0 or batch_size < 1:
            raise ValueError("invalid replay timing/batching")
        async def replay(**arguments):
            key = call_key(self.workflow_id, call_id, arguments)
            if key not in self.records:
                raise ReplayMiss(f"replay miss: workflow={self.workflow_id}, call={call_id}, key={key}")
            record = self.records[key]
            if record["arguments"] != arguments:
                raise ReplayMiss("replay arguments do not match")
            events = self._batch(record["events"], batch_size)
            started = asyncio.get_running_loop().time()
            for item in events:
                delay = item["at"] * time_scale - (asyncio.get_running_loop().time() - started)
                if delay > 0:
                    await asyncio.sleep(delay)
                if item["kind"] == "failure":
                    yield Failure(item["error"])
                elif item["kind"] == "partial":
                    yield Partial(copy.deepcopy(item["data"]), tuple(item["stable_fields"]), tuple(item["effects"]))
                else:
                    yield Complete(copy.deepcopy(item["data"]), tuple(item["effects"]))
        return replay

    @staticmethod
    def _batch(events: list[dict], size: int) -> list[dict]:
        result, pending = [], []
        def flush():
            if not pending:
                return
            data, stable, effects = {}, [], []
            for event in pending:
                data = _deep_merge(data, event["data"])
                stable.extend(event["stable_fields"])
                effects.extend(event["effects"])
            result.append({"kind": "partial", "at": pending[-1]["at"], "data": data,
                           "stable_fields": list(dict.fromkeys(stable)), "effects": list(dict.fromkeys(effects))})
            pending.clear()
        for event in events:
            if event["kind"] == "partial":
                pending.append(event)
                if len(pending) == size:
                    flush()
            else:
                flush()
                result.append(event)
        flush()
        return result

    def save(self, path: str | Path) -> None:
        value = {"schema_version": 1, "workflow_id": self.workflow_id,
                 "metadata": self.metadata, "records": self.records}
        Path(path).write_text(canonical(value) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> ReplayBundle:
        value = json.loads(Path(path).read_text())
        if value.get("schema_version") != 1:
            raise ValueError("unsupported replay schema")
        bundle = cls(value["workflow_id"], records=value["records"], metadata=value["metadata"])
        for key, record in bundle.records.items():
            if key != call_key(bundle.workflow_id, record["call_id"], record["arguments"]):
                raise ValueError("invalid replay record key")
            previous = -1.0
            for index, event in enumerate(record["events"]):
                if event["sequence"] != index or not math.isfinite(event["at"]) or event["at"] < max(0, previous):
                    raise ValueError("replay events must have consecutive sequence IDs and monotone times")
                if event["kind"] not in {"partial", "complete", "failure"}:
                    raise ValueError("unknown replay event")
                if event["kind"] != "partial" and index != len(record["events"]) - 1:
                    raise ValueError("events after terminal replay event")
                previous = event["at"]
            if not record["events"] or record["events"][-1]["kind"] == "partial":
                raise ValueError("incomplete recording cannot be replayed")
        return bundle


def graph_to_dict(graph, sink_ids: list[str] | None = None) -> dict:
    calls = []
    for call in graph.calls:
        contract = call.stream_contract
        calls.append({"id": call.id, "input_schema": call.input_schema, "output_schema": call.output_schema,
                      "early_safe": call.early_safe, "resources": call.resources.as_dict(),
                      "effects": list(call.effects), "metadata": call.metadata,
                      "stream_contract": None if contract is None else {
                          "stable_fields": list(contract.stable_fields), "mutable_fields": list(contract.mutable_fields),
                          "monotone": contract.monotone}})
    from dataclasses import asdict
    return {"calls": calls, "data_dependencies": [asdict(d) for d in graph.data_dependencies],
            "effect_dependencies": [asdict(d) for d in graph.effect_dependencies],
            "resource_capacity": graph.resource_capacity.as_dict(), "sink_ids": sink_ids}


def graph_from_replay(value: dict, bundle: ReplayBundle, *, capacity: dict | None = None,
                      time_scale: float = 1.0, batch_size: int = 1):
    calls = []
    for item in value["calls"]:
        item = copy.deepcopy(item)
        contract = item.pop("stream_contract")
        calls.append(CallSpec(**item, target=bundle.bind(item["id"], time_scale=time_scale, batch_size=batch_size),
                              stream_contract=StreamContract(**contract) if contract else None))
    return GFRGCompiler(capacity if capacity is not None else value["resource_capacity"]).compile(
        calls, [DataDependency(**item) for item in value["data_dependencies"]],
        [EffectDependency(**item) for item in value["effect_dependencies"]])
