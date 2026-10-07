"""Frozen-manifest construction experiments; native search is separate from evolution."""

from __future__ import annotations
import copy, csv, fcntl, hashlib, json, math, platform, importlib.metadata
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
from .benchmarks import evaluate, validate_partitions
from .evolution import atomic_json
from .replay import digest
from .baseline_workflows import graph_topology, topology_statistics, UnsupportedWorkflow

REQUIRED_METHODS = (
    "aflow",
    "evoagentx",
    "base_planner",
    "expert_policies",
    "static_library",
    "compactflow",
)
SUPPORTED_METHODS = REQUIRED_METHODS + ("a2flow",)
REQUIRED_BENCHMARKS = ("MBPP", "HotpotQA", "MATH", "GAIA")


class BaselineUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowCandidate:
    workflow_id: str
    method: str
    benchmark: str
    artifact: dict
    metadata: dict = field(default_factory=dict)

    @classmethod
    def create(cls, method, benchmark, artifact, **metadata):
        return cls(
            digest([method, benchmark, artifact]), method, benchmark, artifact, metadata
        )


@dataclass
class BaselineRecord:
    method: str
    benchmark: str
    task_id: str
    split: str
    seed: int
    workflow_id: str
    generation_seed: int
    status: str
    quality: float | None
    token_cost: int | None
    latency: float | None
    node_count: int | None
    edge_count: int | None
    topology: dict | None
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class BaselineAdapter(Protocol):
    method: str

    def preflight(self, config): ...
    async def search(self, source_tasks, *, seed, config, workspace, score): ...
    async def execute(
        self, candidate, public_task, *, seed, workspace, budget=None
    ): ...


class FrozenBenchmarkView:
    """Only public source records are exposed to native optimizer train/dev APIs."""

    def __init__(self, benchmark, source):
        self.name, self._source = benchmark, copy.deepcopy(tuple(source))

    def get_dev_data(self, indices=None, sample_k=None, **kwargs):
        if indices is not None or sample_k is not None:
            raise ValueError("resampling frozen manifest forbidden")
        return copy.deepcopy(list(self._source))

    get_train_data = get_dev_data

    def get_test_data(self, *args, **kwargs):
        raise BaselineUnavailable("target/validation not exposed to optimizer")


def verify_manifest(tasks, manifest):
    rows = [json.loads(l) for l in Path(manifest).read_text().splitlines() if l.strip()]
    indexed = {(t.benchmark, t.task_id): t for t in tasks}
    seen = set()
    if len(indexed) != len(tasks):
        raise ValueError("duplicate raw task")
    for r in rows:
        key = r["benchmark"], str(r["task_id"])
        if key in seen or key not in indexed:
            raise ValueError("duplicate/unknown manifest task")
        seen.add(key)
        t = indexed[key]
        if (r["split"], r["family_id"], r.get("validation_fold")) != (
            t.split,
            t.family_id,
            t.metadata.get("validation_fold"),
        ):
            raise ValueError("manifest split/family/fold mismatch")
        if r.get("task_digest") != digest(asdict(t)):
            raise ValueError("manifest task content mismatch")
    if seen != set(indexed):
        raise ValueError("manifest/task coverage mismatch")
    validate_partitions(list(tasks))
    return rows


class ConstructionBaselineRunner:
    def __init__(
        self,
        tasks,
        *,
        output,
        adapters,
        config,
        manifest,
        evaluate_fn=None,
        evolution_dir=None,
        dataset_coverage=None,
        code_identity=None,
    ):
        self.tasks = tuple(tasks)
        self.output = Path(output)
        self.manifest = Path(manifest)
        self.adapters = tuple(adapters)
        self.config = copy.deepcopy(config)
        self.manifest_rows = verify_manifest(self.tasks, manifest)
        self.evolution_dir = Path(evolution_dir) if evolution_dir else None
        self.dataset_coverage = dict(dataset_coverage or {})
        self.evaluate_fn = evaluate_fn or (
            lambda t, a: evaluate(t, a, sandbox=config["tools"].get("mbpp_sandbox"))
        )
        self.code_identity = code_identity or self._code_identity()
        self.records, self.coverage, self._frozen = {}, {}, {}
        self.generation_seeds = tuple(config["evaluation"]["generation_seeds"])
        self.validation_seeds = tuple(config["construction"]["heldout_seeds"])
        if (
            not self.generation_seeds
            or not self.validation_seeds
            or len(set(self.generation_seeds)) != len(self.generation_seeds)
            or len(set(self.validation_seeds)) != len(self.validation_seeds)
            or any(
                type(s) is not int
                for s in self.generation_seeds + self.validation_seeds
            )
        ):
            raise ValueError("nonempty distinct seeds required")
        if len({a.method for a in adapters}) != len(adapters):
            raise ValueError("duplicate adapters")

    def _code_identity(self):
        root = Path(__file__).resolve().parents[2]
        files = {}
        for d in (
            "evoagentx",
            "examples/compactflow",
            "examples/aflow",
        ):
            for p in sorted((root / d).rglob("*")):
                if (
                    p.suffix in {".py", ".json", ".txt"}
                    and "__pycache__" not in p.parts
                ):
                    files[str(p.relative_to(root))] = hashlib.sha256(
                        p.read_bytes()
                    ).hexdigest()
        try:
            revision = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = "unversioned"
        return digest({"revision": revision, "files": files})

    def _tasks(self, split, b):
        return sorted(
            (t for t in self.tasks if t.split == split and t.benchmark == b),
            key=lambda t: t.task_id,
        )

    def _prepare(self, resume):
        if self.evolution_dir:
            self._frozen = {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(
                    (self.evolution_dir / "policy_snapshots").glob("*.json")
                )
            }
            if not self._frozen:
                raise ValueError("missing frozen policy snapshots")
        lock = {
            "config": self.config,
            "manifest": digest(self.manifest_rows),
            "code": self.code_identity,
            "policies": self._frozen,
            "adapters": [a.method for a in self.adapters],
            "datasets": self.dataset_coverage,
        }
        p = self.output / "baseline.lock.json"
        if p.exists():
            if not resume:
                raise FileExistsError("immutable output exists; use --resume")
            if json.loads(p.read_text()) != lock:
                raise ValueError("baseline lock mismatch")
        else:
            if set(p.name for p in self.output.iterdir()) - {
                ".process.lock",
                "preflight.json",
            }:
                raise ValueError("nonempty unlocked output")
            atomic_json(p, lock)
            (self.output / "sample_manifest.jsonl").write_bytes(
                self.manifest.read_bytes()
            )
            atomic_json(self.output / "config.lock.json", self.config)
            versions = {}
            for name in ("numpy", "pydantic", "networkx", "openai", "litellm"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "unavailable"
            atomic_json(
                self.output / "environment.lock.json",
                {
                    "python": platform.python_version(),
                    "packages": versions,
                    "code_sha256": self.code_identity,
                },
            )
        if (
            self.output / "sample_manifest.jsonl"
        ).read_bytes() != self.manifest.read_bytes():
            raise ValueError("manifest changed")
        self.records = {
            p.stem: json.loads(p.read_text())
            for p in (self.output / "records").glob("*.json")
        }
        if self.config.get("tools",{}).get("gaia",{}).get("enabled"):
            from .gaia import initialize_gaia_run
            initialize_gaia_run(self.output,self.config,self.tasks)
        for adapter in self.adapters:
            if hasattr(adapter,"bind_run"):
                adapter.bind_run(self.output)

    def _check_frozen(self):
        for p, sha in self._frozen.items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha:
                raise ValueError("policy snapshot changed")

    async def _evaluate(
        self, adapter, candidate, task, seed, generation_seed, workspace, budget=None
    ):
        from .paper_phase import assert_target_allowed
        assert_target_allowed(self.config, task.split, task.benchmark)
        if (candidate.method, candidate.benchmark) != (adapter.method, task.benchmark):
            raise ValueError("candidate pair key mismatch")
        if candidate.workflow_id != digest(
            [candidate.method, candidate.benchmark, candidate.artifact]
        ):
            raise ValueError("candidate digest mismatch")
        key = digest(
            [
                adapter.method,
                task.benchmark,
                task.task_id,
                task.split,
                seed,
                generation_seed,
                candidate.workflow_id,
            ]
        )
        if key in self.records:
            return self.records[key]
        result = await adapter.execute(
            copy.deepcopy(candidate),
            copy.deepcopy(task.public_input()),
            seed=seed,
            workspace=workspace / key,
            budget=budget,
        )
        if result.get("runtime") != "complete_dependency":
            raise BaselineUnavailable("common runtime required")
        usage = result.get("tokens", {})
        valid = bool(result.get("valid"))
        quality = None
        failure = result.get("error")
        status = (
            "complete"
            if usage.get("usage_complete") and not result.get("infrastructure_error")
            else "incomplete"
        )
        if status == "complete":
            try:
                quality = (
                    float(self.evaluate_fn(task, result.get("answer", ""))["quality"])
                    if valid
                    else 0.0
                )
                if not math.isfinite(quality):
                    raise ValueError("nonfinite quality")
            except Exception as e:
                status, failure = "incomplete", f"evaluator: {type(e).__name__}: {e}"
        topology = result.get("topology")
        if topology is None and candidate.artifact.get("format") in {
            "aflow_static_v1",
            "sew_native_v1",
        }:
            topology = graph_topology(candidate.artifact)
        r = BaselineRecord(
            adapter.method,
            task.benchmark,
            task.task_id,
            task.split,
            seed,
            candidate.workflow_id,
            generation_seed,
            status,
            quality,
            usage.get("total_tokens") if usage.get("usage_complete") else None,
            result.get("latency"),
            len(topology["nodes"]) if topology else None,
            len(topology["edges"]) if topology else None,
            topology,
            {
                "valid": valid,
                "error": failure,
                "tokens": usage,
                "model_requests": result.get("model_requests", []),
                "tool_accounting": result.get("tool_accounting", {}),
                "level": task.metadata.get("level"),
                "family_id": task.family_id,
                "runtime": "complete_dependency",
                "answer": result.get("answer", ""),
            },
        ).to_dict()
        if task.split == "target":
            self._check_frozen()
        atomic_json(self.output / "records" / (key + ".json"), r)
        self.records[key] = r
        print(json.dumps({"phase":"construction_evaluation","method":adapter.method,
                          "benchmark":task.benchmark,"task_id":task.task_id,"split":task.split,
                          "status":status,"completed_records":len(self.records),
                          "time":datetime.now(timezone.utc).isoformat(),
                          "execution_status":"complete" if valid else "failed",
                          "usage_complete":usage.get("usage_complete"),
                          "infrastructure_error":result.get("infrastructure_error",False),
                          "error":failure,"elapsed_seconds":result.get("latency")}),flush=True)
        return r

    async def run(self, *, resume=False, stage="all"):
        if stage not in {"all", "freeze", "target"}:
            raise ValueError("invalid baseline stage")
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / ".process.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._prepare(resume)
            pending = []
            for adapter in self.adapters:
                for b in REQUIRED_BENCHMARKS:
                    cell = {"status": "incomplete", "seeds": {}}
                    self.coverage.setdefault(adapter.method, {})[b] = cell
                    if b not in self.config.get("baseline_runner", {}).get(
                        "benchmarks", REQUIRED_BENCHMARKS
                    ):
                        cell["reason"] = "outside explicitly selected pilot benchmarks"
                        continue
                    blocked=self.config.get("baseline_runner",{}).get("blocked_benchmarks",{})
                    if b in blocked:
                        cell["reason"]="discovery infrastructure or accounting incomplete"
                        continue
                    source, validation, target = (
                        self._tasks(s, b) for s in ("source", "validation", "target")
                    )
                    if b == "GAIA" and self.config.get("tools",{}).get("gaia",{}).get("enabled"):
                        from .gaia import capability_report
                        report = capability_report(self.config)
                        if report["status"] != "complete":
                            cell["reason"] = "; ".join(report["reasons"])
                            continue
                    if not source or not validation or not target:
                        cell["reason"] = "missing frozen three-way dataset"
                        continue
                    try:
                        adapter.preflight(self.config)
                    except (BaselineUnavailable, ImportError) as e:
                        cell["reason"] = str(e)
                        continue
                    for gen in self.generation_seeds:
                        ws = self.output / "baselines" / adapter.method / b / str(gen)
                        ws.mkdir(parents=True, exist_ok=True)
                        selected_path = ws / "selected.json"
                        try:
                            if selected_path.exists():
                                selected = json.loads(selected_path.read_text())
                                if selected["digest"] != digest(selected["candidate"]):
                                    raise ValueError("selected digest mismatch")
                                candidate = WorkflowCandidate(**selected["candidate"])
                            else:
                                if stage == "target":
                                    raise BaselineUnavailable("baseline must freeze before target")
                                if (self.output / "target.lock.json").exists():
                                    raise BaselineUnavailable(
                                        "search frozen when target evaluation began; create a new run"
                                    )

                                async def score(
                                    c,
                                    eval_seed,
                                    *,
                                    budget=None,
                                    _a=adapter,
                                    _tasks=source,
                                    _gen=gen,
                                    _ws=ws,
                                ):
                                    rows = [
                                        await self._evaluate(
                                            _a,
                                            c,
                                            t,
                                            eval_seed,
                                            _gen,
                                            _ws / "calls",
                                            budget,
                                        )
                                        for t in _tasks
                                    ]
                                    if any(r["status"] != "complete" for r in rows):
                                        raise BaselineUnavailable(
                                            "source infrastructure/usage incomplete"
                                        )
                                    return sum(r["quality"] for r in rows) / len(rows)

                                found = await adapter.search(
                                    [copy.deepcopy(t.public_input()) for t in source],
                                    seed=gen,
                                    config=self.config,
                                    workspace=ws,
                                    score=score,
                                )
                                if not found:
                                    raise BaselineUnavailable("no canonical candidates")
                                choices = []
                                for c in found:
                                    rows = [
                                        await self._evaluate(
                                            adapter,
                                            c,
                                            t,
                                            s,
                                            gen,
                                            ws / "calls",
                                            getattr(adapter, "search_budget", None),
                                        )
                                        for t in validation
                                        for s in self.validation_seeds
                                    ]
                                    if all(r["status"] == "complete" for r in rows):
                                        choices.append(
                                            (
                                                sum(r["quality"] for r in rows)
                                                / len(rows),
                                                c.workflow_id,
                                                c,
                                            )
                                        )
                                if not choices:
                                    raise BaselineUnavailable(
                                        "no complete validation candidate"
                                    )
                                candidate = sorted(
                                    choices, key=lambda v: (-v[0], v[1])
                                )[0][2]
                                payload = asdict(candidate)
                                atomic_json(
                                    selected_path,
                                    {
                                        "candidate": payload,
                                        "digest": digest(payload),
                                        "selection": "validation mean, digest tie-break",
                                    },
                                )
                            pending.append((adapter, candidate, target, gen, ws))
                            cell["seeds"][str(gen)] = {
                                "status": "selected",
                                "workflow_id": candidate.workflow_id,
                            }
                        except (
                            BaselineUnavailable,
                            UnsupportedWorkflow,
                            ImportError,
                        ) as e:
                            cell["seeds"][str(gen)] = {
                                "status": "incomplete",
                                "reason": str(e),
                            }
            # All methods complete search/selection before any target is accessed.
            self._check_frozen()
            target_lock = self.output / "target.lock.json"
            selection = {
                str(p.relative_to(self.output)): digest(json.loads(p.read_text()))
                for p in sorted((self.output / "baselines").glob("*/*/*/selected.json"))
            }
            if target_lock.exists():
                if json.loads(target_lock.read_text()) != selection:
                    raise ValueError("frozen target selection changed")
            elif pending:
                atomic_json(target_lock, selection)
            if stage == "freeze":
                value = {"stage": "frozen", "methods": self.coverage, "selected": len(pending)}
                atomic_json(self.output / "freeze_summary.json", value)
                return value
            from .paper_phase import assert_target_allowed
            assert_target_allowed(self.config, "target")
            for adapter, c, target, seed, ws in pending:
                before = (ws / "selected.json").read_bytes()
                cell = self.coverage[adapter.method][c.benchmark]["seeds"][str(seed)]
                try:
                    rows = [
                        await self._evaluate(adapter, c, t, seed, seed, ws / "target")
                        for t in target
                    ]
                    cell.update(
                        status="complete"
                        if all(r["status"] == "complete" for r in rows)
                        else "incomplete",
                        target_records=len(rows),
                        valid_records=sum(r["metadata"]["valid"] for r in rows),
                    )
                except (BaselineUnavailable, UnsupportedWorkflow, ImportError) as e:
                    cell.update(status="incomplete", reason=str(e))
                if before != (ws / "selected.json").read_bytes():
                    raise ValueError("target modified selected snapshot")
            self._check_frozen()
            for datasets in self.coverage.values():
                for cell in datasets.values():
                    if len(cell["seeds"]) == len(self.generation_seeds) and all(
                        s["status"] == "complete" for s in cell["seeds"].values()
                    ):
                        cell["status"] = "complete"
            return self._write_outputs()

    def _write_outputs(self):
        rows = sorted(
            self.records.values(),
            key=lambda r: (
                r["method"],
                r["benchmark"],
                r["split"],
                r["generation_seed"],
                r["workflow_id"],
                r["task_id"],
                r["seed"],
            ),
        )
        (self.output / "baseline_records.jsonl").write_text(
            "".join(
                json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows
            )
        )
        imported = []
        if self.evolution_dir and self.config.get("baseline_runner", {}).get(
            "import_evolution_records", False
        ):
            p = self.evolution_dir / "target_records.jsonl"
            for line in p.read_text().splitlines() if p.exists() else []:
                r = json.loads(line)
                f = r["evidence"]["feedback"]
                if r["variant"] in {a.method for a in self.adapters}:
                    continue
                imported.append(
                    {
                        "method": r["variant"],
                        "benchmark": r["benchmark"],
                        "task_id": r["task_id"],
                        "seed": r["seed"],
                        "split": "target",
                        "status": "complete"
                        if r["record"].get("tokens", {}).get("usage_complete", False)
                        else "incomplete",
                        "quality": f["quality"],
                        "token_cost": f["token_cost"],
                        "node_count": f["node_count"],
                        "edge_count": f["edge_count"],
                        "latency": f["latency"],
                        "topology": None,
                    }
                )
        table = []
        requested = tuple(
            dict.fromkeys(
                (
                    *REQUIRED_METHODS,
                    *self.config.get("baseline_runner", {}).get("required_methods", ()),
                )
            )
        )
        for method in requested:
            for b in REQUIRED_BENCHMARKS:
                group = [
                    r
                    for r in rows + imported
                    if r["method"] == method
                    and r["benchmark"] == b
                    and r["split"] == "target"
                ]
                expected = {
                    (t.task_id, s)
                    for t in self._tasks("target", b)
                    for s in self.generation_seeds
                }
                keys = [(r["task_id"], r["seed"]) for r in group]
                complete = (
                    bool(expected)
                    and set(keys) == expected
                    and len(keys) == len(expected)
                    and all(r["status"] == "complete" for r in group)
                )
                if method in self.coverage:
                    complete = (
                        complete and self.coverage[method][b]["status"] == "complete"
                    )

                def mean(k):
                    vals = [r.get(k) for r in group]
                    return (
                        sum(vals) / len(vals)
                        if vals and all(isinstance(v, (int, float)) for v in vals)
                        else None
                    )

                instability = []
                timeouts = 0
                pair_count = 0
                for tid in sorted({r["task_id"] for r in group}):
                    stat = topology_statistics(
                        [
                            r["topology"]
                            for r in group
                            if r["task_id"] == tid and r.get("topology")
                        ],
                        timeout=self.config.get("baseline_runner", {}).get(
                            "ged_timeout_seconds", 1.0
                        ),
                    )
                    timeouts += stat["timed_out"]
                    pair_count += stat["pairs"]
                    if stat["value"] is not None:
                        instability.append(stat["value"])
                ledgers = [
                    json.loads(p.read_text())
                    for p in (self.output / "baselines" / method / b).glob(
                        "*/search_budget.json"
                    )
                ]
                offline_tokens = sum(
                    e["tokens"] for ledger in ledgers for e in ledger.values()
                )
                offline_complete = all(
                    e["usage_complete"] for ledger in ledgers for e in ledger.values()
                )
                target_tokens = mean("token_cost")
                table.append(
                    {
                        "method": method,
                        "benchmark": b,
                        "status": "complete" if complete else "incomplete",
                        "target_records": len(group),
                        "expected_records": len(expected),
                        "quality_mean": mean("quality"),
                        "tokens_mean": target_tokens,
                        "offline_tokens": offline_tokens
                        if ledgers and offline_complete
                        else None,
                        "amortized_tokens_mean": target_tokens
                        + offline_tokens / len(group)
                        if ledgers
                        and offline_complete
                        and group
                        and target_tokens is not None
                        else None,
                        "nodes_mean": mean("node_count"),
                        "edges_mean": mean("edge_count"),
                        "latency_mean": mean("latency"),
                        "topology_instability": sum(instability) / len(instability)
                        if instability and not timeouts
                        else None,
                        "topology_pairs": pair_count,
                        "topology_timeouts": timeouts,
                    }
                )
        complete = (
            all(r["status"] == "complete" for r in table)
            and self.config.get("profile") == "reference"
            and all(
                self.dataset_coverage.get("dataset:" + b, {}).get("status")
                == "complete"
                for b in REQUIRED_BENCHMARKS
            )
        )
        offline = {}
        for ledger in (self.output / "baselines").glob("*/*/*/search_budget.json"):
            entries = json.loads(ledger.read_text()).values()
            offline[str(ledger.parent.relative_to(self.output / "baselines"))] = {
                "tokens": sum(e["tokens"] for e in entries),
                "usage_complete": all(e["usage_complete"] for e in entries),
                "budget_charged": sum(e["charged"] for e in entries),
            }
        incomplete = {
            r["method"]
            + ":"
            + r["benchmark"]: "missing or incomplete paired target records"
            for r in table
            if r["status"] != "complete"
        }
        for method in ("a2flow",):
            status = (
                self.config.get("baselines", {})
                .get(method, {})
                .get("integration_status")
            )
            if status and status != "runtime_probe":
                incomplete[method] = status
        summary = {
            "publication_status": "complete" if complete else "incomplete",
            "methods": self.coverage,
            "records": len(rows),
            "construction_table": table,
            "topology_metric": "normalized labeled directed graph edit distance; unit costs / total nodes and edges of both graphs; timeouts are null",
            "required_methods": list(requested),
            "required_benchmarks": list(REQUIRED_BENCHMARKS),
        }
        summary["offline_search_and_validation"] = offline
        from .gaia import gaia_summary
        summary["gaia"] = gaia_summary(rows)
        if incomplete:
            summary["publication_status"] = "incomplete"
        summary["coverage_incomplete"] = incomplete
        if self.evolution_dir and (self.evolution_dir / "summary.json").exists():
            summary["evolution_summary"] = json.loads(
                (self.evolution_dir / "summary.json").read_text()
            )
        for name in ("baseline_summary.json", "summary.json"):
            atomic_json(self.output / name, summary)
        atomic_json(self.output / "baseline_coverage.json", self.coverage)
        atomic_json(
            self.output / "coverage.json",
            {
                "methods": self.coverage,
                "datasets": self.dataset_coverage,
                "incomplete": incomplete,
            },
        )
        with (self.output / "tables.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        return summary


# Imported at module bottom; native heavyweight imports happen only in preflight/search.
from .baseline_native import (
    AFlowAdapter as AFlowAdapter,
    EvoAgentXAdapter as EvoAgentXAdapter,
    A2FlowAdapter as A2FlowAdapter,
)
