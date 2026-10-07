"""Paper-wide target gate shared by model and tool execution entry points."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_target_allowed(config, split, benchmark=None):
    context = config.get("paper_context")
    if split != "target" or not context:
        return
    gate = Path(context["root"]) / "target_gate.lock.json"
    if not gate.is_file():
        raise ValueError("paper target gate is closed: all construction variants must freeze first")
    value = json.loads(gate.read_text())
    if value.get("run_identity") != context["run_identity"] or not value.get("artifacts"):
        raise ValueError("paper target gate identity mismatch")
    if benchmark is not None and "ready_benchmarks" in value and benchmark not in value["ready_benchmarks"]:
        raise ValueError("target benchmark blocked by discovery health: "+benchmark)
    root = Path(context["root"]).resolve()
    for relative, expected in value["artifacts"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != expected:
            raise ValueError("paper frozen artifact changed: " + relative)
