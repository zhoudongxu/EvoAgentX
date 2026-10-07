"""Frozen policy controls using the same planner, executor and complete runtime."""

from __future__ import annotations
import copy, json, time
from contextlib import aclosing
from pathlib import Path
from types import SimpleNamespace
from .baseline_native import ModelSession
from .baselines import BaselineUnavailable, WorkflowCandidate
from .models import CompactnessPolicy
from .paper_workflow import plan_workflow, compile_spec
from .policy import (
    PolicyLibrary,
    PolicyMatch,
    CompatibilitySelector,
    SelectionConfig,
    default_applicability,
)
from .runtime import CompactFlowRuntime
from .replay import digest


class SessionClient:
    """Journal all model requests, including repair/fallback and execution."""

    def __init__(self, session):
        self.session = session

    @property
    def records(self):
        return self.session.records

    def accounting(self):
        return self.session.accounting()

    async def text(self, system, prompt, *, component, seed, response_format=None):
        return await self.session.text(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            component,
            component,
            seed=seed, response_format=response_format,
        )

    async def json(self, system, prompt, *, component, seed, response_format=None):
        raw = await self.text(
            system + " Return one JSON object only, without Markdown fences.",
            prompt,
            component=component,
            seed=seed, response_format=response_format or {"type":"json_object"},
        )
        value = raw.strip()
        fence = chr(96) * 3
        if value.startswith(fence):
            value = value.split("\n", 1)[1].rsplit(fence, 1)[0].strip()
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise TypeError("model must return a JSON object")
        return parsed

    async def stream(self, system, prompt, *, component, seed, response_format=None):
        async with aclosing(self.session.stream(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            component, component, seed=seed, response_format=response_format,
        )) as stream:
            async for chunk in stream:
                yield chunk


class PolicyBaselineAdapter:
    def bind_run(self, output):
        self.run_output = Path(output)

    def __init__(self, method, *, evolution_dir, client_factory=None, embedder=None):
        if method not in {
            "base_planner",
            "expert_policies",
            "static_library",
            "compactflow",
        }:
            raise ValueError(method)
        self.method, self.evolution_dir = method, Path(evolution_dir)
        self.client_factory, self.embedder = client_factory, embedder

    def preflight(self, config):
        self.config = config
        checkpoint = self.evolution_dir / "checkpoint.json"
        if not checkpoint.exists() or json.loads(checkpoint.read_text()).get(
            "stage"
        ) not in {"frozen", "target", "complete"}:
            raise BaselineUnavailable("policy evolution must reach frozen target stage")
        name = {
            "base_planner": "final",
            "expert_policies": "initial",
            "static_library": "bootstrap",
            "compactflow": "final",
        }[self.method]
        self.snapshot = self.evolution_dir / "policy_snapshots" / (name + ".json")
        if not self.snapshot.exists():
            raise BaselineUnavailable(f"missing {name} policy snapshot")
        self.frozen = json.loads(self.snapshot.read_text())
        policies = self.frozen.get("policies", [])
        if self.method == "base_planner":
            policies = []
        if self.method == "compactflow":
            policies = [p for p in policies if p["status"] == "verified"]
        if self.method == "static_library":
            policies = [p for p in policies if p["status"] != "rejected"]
        self.policies = policies

    async def search(self, source_tasks, *, seed, config, workspace, score):
        self.preflight(config)
        return [
            WorkflowCandidate.create(
                self.method,
                source_tasks[0]["benchmark"],
                {
                    "format": "frozen_policies_v1",
                    "policies": copy.deepcopy(self.policies),
                    "snapshot_sha256": digest(self.frozen),
                },
            )
        ]

    async def execute(self, candidate, public_task, *, seed, workspace, budget=None):
        from examples.compactflow.run_evolution import LiveAdapter, PinnedEmbedder
        from .gaia import GaiaUnavailable

        kwargs = {"client_factory": self.client_factory} if self.client_factory else {}
        session = ModelSession(
            self.config, seed, workspace / "model_calls", budget=budget, **kwargs
        )
        client = SessionClient(session)
        public = copy.deepcopy(public_task)
        gaia = None
        if public["benchmark"] == "GAIA":
            from .gaia import GaiaToolSession
            gaia = GaiaToolSession(self.config,public,output=self.run_output,workspace=workspace,seed=seed,model_session=session)
        task = SimpleNamespace(
            benchmark=public["benchmark"],
            task_id=public["task_id"],
            public_input=lambda: copy.deepcopy(public),
        )
        policies = [
            CompactnessPolicy.from_dict(p) for p in candidate.artifact["policies"]
        ]
        before = digest([p.to_dict() for p in policies])
        topology = None
        valid = False
        answer = ""
        error = None
        infra = False
        start = time.perf_counter()
        try:
            if public.get("attachments") and gaia is None:
                raise BaselineUnavailable("missing attachment capability")
            if self.method == "expert_policies":
                c = self.config["construction"]
                context = {
                    "benchmark": task.benchmark,
                    "capabilities": {"typed_workflow": True},
                    "tools": self.config["tools"]["registry"],
                }
                matches = [
                    PolicyMatch(
                        p,
                        0.0,
                        default_applicability(p, context),
                        default_applicability(p, context),
                    )
                    for p in sorted(policies, key=lambda p: p.id)
                ]
                selected = (
                    CompatibilitySelector(
                        SelectionConfig(
                            max_policies=c["max_policies"],
                            minimum_score=0,
                            minimum_applicability=c["minimum_applicability"],
                        )
                    )
                    .select(matches)
                    .policies
                )
            else:
                library = PolicyLibrary(
                    policies,
                    embedder=self.embedder or PinnedEmbedder(self.config["encoder"]),
                )
                selected, _ = await LiveAdapter(self.config, library)._select(
                    client, task, policies, seed
                )
            spec, _ = await plan_workflow(
                client,
                task,
                list(selected),
                self.config["construction"]["planner"],
                seed=seed,
            )
            graph = compile_spec(
                spec,
                public,
                client,
                seed=seed,
                capacity=self.config["execution"]["external_call_capacity"],
                tool_session=gaia,
            )
            ids = {n["id"]: i for i, n in enumerate(spec["nodes"])}
            topology = {
                "nodes": [n["tool"] for n in spec["nodes"]],
                "edges": sorted(
                    {
                        (ids[d.producer], ids[d.consumer])
                        for d in graph.data_dependencies
                        if d.producer is not None
                    }
                ),
            }
            result = await CompactFlowRuntime(
                graph,
                mode="complete",
                sink_ids=spec["sinks"],
                call_timeout=self.config["execution"]["call_timeout_seconds"],
                workflow_timeout=self.config["execution"]["workflow_timeout_seconds"],
            ).execute(public)
            if result.errors:
                raise ValueError(str(result.errors))
            answer = result.outputs.get(spec["sinks"][0], {}).get("answer", "")
            valid = True
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            infra = isinstance(e, (BaselineUnavailable, GaiaUnavailable, ImportError, OSError))
        if digest([p.to_dict() for p in policies]) != before:
            raise ValueError("control changed frozen policy state")
        usage = session.accounting()
        if gaia and gaia.incomplete:
            infra = True
        return {
            "runtime": "complete_dependency",
            "tool_accounting": gaia.accounting() if gaia else {},
            "answer": answer,
            "valid": valid,
            "error": error,
            "infrastructure_error": infra or not usage["usage_complete"],
            "tokens": usage,
            "model_requests": session.records,
            "latency": time.perf_counter() - start,
            "topology": topology,
        }
