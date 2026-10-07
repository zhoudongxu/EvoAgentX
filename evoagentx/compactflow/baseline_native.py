"""Native optimizer bridges, actual ModelClient accounting, common runtime execution."""

from __future__ import annotations
import asyncio, copy, importlib, json, os, random, re, threading, time
from contextlib import aclosing
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from .baseline_workflows import (
    UnsupportedWorkflow,
    export_aflow,
    expression_value,
    lower_aflow,
    graph_topology,
)
from .evolution import atomic_json
from .llm import ModelClient, TokenBudgetExceeded, format_options
from .replay import digest
from .runtime import CompactFlowRuntime


class SearchBudget:
    """Persistent method/benchmark/generation-seed search + validation cap."""

    def __init__(self, path, limit):
        self.path, self.limit = Path(path), int(limit)
        self.lock = threading.Lock()
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else {}

    @property
    def used(self):
        return sum(r["charged"] for r in self.entries.values())

    def reserve(self, key, amount):
        from .baselines import BaselineUnavailable

        with self.lock:
            if key in self.entries:
                raise BaselineUnavailable(
                    "journaled request has no durable response; refusing unaccounted retry"
                )
            if self.used + amount > self.limit:
                raise TokenBudgetExceeded("offline search token budget exhausted")
            self.entries[key] = {
                "charged": amount,
                "tokens": 0,
                "usage_complete": False,
                "state": "pending",
            }
            atomic_json(self.path, self.entries)
            return True

    def finish(self, key, amount, tokens, complete):
        with self.lock:
            self.entries[key] = {
                "charged": tokens if complete else amount,
                "tokens": tokens,
                "usage_complete": complete,
                "state": "finished",
            }
            atomic_json(self.path, self.entries)


class ModelSession:
    def __init__(self, config, seed, path, *, budget=None, client_factory=ModelClient):
        self.config, self.seed, self.path = config, int(seed), Path(path)
        self.budget, self.client_factory = budget, client_factory
        self.records = []
        self.counts = {}
        self.used = 0
        self.reserved = 0
        self.lock = asyncio.Lock()
        self.incomplete = False
        self.external = {}

    async def reserve_external(self, key, amount, *, replay=False):
        async with self.lock:
            if self.used + self.reserved + amount > self.config["evaluation"]["task_token_budget"]:
                raise TokenBudgetExceeded("shared task token budget exhausted by auxiliary model")
            self.reserved += amount
        try:
            if self.budget and not (replay and self.budget.entries.get(key, {}).get("state") == "finished"):
                self.budget.reserve(key, amount)
            self.external[key] = amount
        except BaseException:
            self.reserved -= amount
            raise

    def finish_external(self, key, bound, usage, complete, *, cached=False):
        amount = self.external.pop(key, bound)
        self.reserved -= amount
        tokens = int(usage.get("total_tokens", 0))
        self.used += tokens if complete else bound
        self.incomplete |= not complete
        self.records.append({"component":"vision", "usage":{"prompt_tokens":usage.get("prompt_tokens",0),
                             "completion_tokens":usage.get("completion_tokens",0)},"usage_missing":not complete,
                             "cache_hit":cached,"physical_tokens":0 if cached else tokens})
        if self.budget:
            self.budget.finish(key,bound,tokens,complete)

    def _not_sent(self, cache, component, error):
        record = {"component": component, "request_status": "not_sent",
                  "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                  "usage_missing": False, "physical_tokens": 0}
        self.records.append(record)
        atomic_json(cache, {"state": "finished", "text": "", "chunks": [],
                           "records": [record], "charged": 0, "usage_complete": True,
                           "error": error, "failure_kind": "execution",
                           "terminal_status": "budget_exhausted"})

    async def text(self, messages, component, scope, *, seed=None, response_format=None):
        from .baselines import BaselineUnavailable

        if self.incomplete:
            raise BaselineUnavailable("session contains unknown usage; refusing further inference")
        seed = self.seed if seed is None else int(seed)
        ordinal = self.counts.get(scope, 0)
        self.counts[scope] = ordinal + 1
        key = digest([scope, ordinal, messages, component, seed, response_format])
        cache = self.path / (key + ".json")
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("state") == "inflight":
                self.incomplete = True
                raise BaselineUnavailable("interrupted model request; refusing unaccounted retry")
            self.records.extend([{**r, "cache_hit": True, "physical_tokens": 0} for r in saved["records"]])
            self.used += saved["charged"]
            self.incomplete |= not saved.get("usage_complete", False)
            if self.budget:
                key_in_ledger = str(cache.resolve())
                if (
                    self.budget.entries.get(key_in_ledger, {}).get("state")
                    != "finished"
                ):
                    self.budget.finish(
                        key_in_ledger,
                        saved["charged"],
                        saved["charged"],
                        saved.get("usage_complete", False),
                    )
            if saved.get("error"):
                if saved.get("usage_complete") and saved.get("failure_kind") == "execution":
                    raise ValueError(saved["error"])
                raise BaselineUnavailable(saved["error"])
            return saved["text"]
        systems = [m["content"] for m in messages if m["role"] == "system"]
        system = "\n".join(systems)
        prompt = "\n".join(
            m["content"] if m["role"] == "user" else m["role"] + ": " + m["content"]
            for m in messages
            if m["role"] != "system"
        )
        settings = self.config["model"]["components"][component]
        bound = (
            len(system.encode()) + len(prompt.encode()) + 256 + settings["max_tokens"]
        )
        async with self.lock:
            if (
                self.used + self.reserved + bound
                > self.config["evaluation"]["task_token_budget"]
            ):
                self._not_sent(cache, component, "shared task token budget exhausted")
                raise TokenBudgetExceeded("shared task token budget exhausted")
            self.reserved += bound
        ledger_key = str(cache.resolve())
        reservation = False
        try:
            if self.budget:
                reservation = self.budget.reserve(ledger_key, bound)
            client = self.client_factory(
                self.config["model"],
                token_budget=self.config["evaluation"]["task_token_budget"],
            )
            error = None
            text = ""
            atomic_json(cache, {"state": "inflight", "usage_complete": False})
            try:
                text = await client.text(system, prompt, component=component, seed=seed, **format_options(client, "text", response_format))
            except asyncio.CancelledError:
                usage = client.accounting()
                known = bool(client.records and usage["usage_complete"])
                self.incomplete |= not known
                charged = usage["total_tokens"] if known else bound
                atomic_json(cache,{"text":"","records":client.records,"charged":charged,
                                   "error":"cancelled model request", "terminal_status":"cancelled",
                                   "state":"finished" if known else "incomplete",
                                   "failure_kind":"execution" if known else "infrastructure",
                                   "usage_complete":known})
                self.records.extend(client.records)
                self.used += charged
                if self.budget and reservation:
                    self.budget.finish(ledger_key,bound,usage["total_tokens"],known)
                    reservation = False
                raise
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            usage = client.accounting()
            complete = bool(usage["usage_complete"] and client.records)
            self.incomplete |= not complete
            saved = {
                "text": text,
                "records": client.records,
                "charged": usage["total_tokens"] if complete else bound,
                "error": error,
                "failure_kind": "execution" if complete else "infrastructure",
                "usage_complete": complete,
            }
            atomic_json(cache, saved)
            self.records.extend(client.records)
            self.used += saved["charged"]
            if self.budget and reservation:
                self.budget.finish(ledger_key, bound, usage["total_tokens"], complete)
                reservation = False
            if error:
                if complete: raise ValueError(error)
                raise BaselineUnavailable(error)
            return text
        except TokenBudgetExceeded as exc:
            # SearchBudget can refuse before a ModelClient is created.
            self._not_sent(cache, component, str(exc))
            raise
        except BaselineUnavailable:
            self.incomplete = True
            raise
        finally:
            self.reserved -= bound
            if self.budget and reservation:
                self.budget.finish(ledger_key, bound, 0, False)

    async def stream(self, messages, component, scope, *, seed=None, response_format=None):
        """Forward live chunks immediately; persist exact timing and measured usage."""
        import time
        from .baselines import BaselineUnavailable
        if self.incomplete:
            raise BaselineUnavailable("session contains unknown usage; refusing further inference")
        seed = self.seed if seed is None else int(seed)
        ordinal = self.counts.get(scope, 0)
        self.counts[scope] = ordinal + 1
        key = digest(["stream-v2", scope, ordinal, messages, component, seed, response_format])
        cache = self.path / (key + ".json")
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get("state") != "finished" or not saved.get("usage_complete"):
                self.incomplete = True
                raise BaselineUnavailable("interrupted/incomplete stream; refusing unaccounted retry")
            self.used += saved["charged"]
            self.records.extend([{**r, "cache_hit": True, "physical_tokens": 0} for r in saved["records"]])
            if self.budget and str(cache.resolve()) not in self.budget.entries:
                self.budget.finish(str(cache.resolve()), saved["charged"], saved["charged"], True)
            started = time.perf_counter()
            for chunk in saved["chunks"]:
                await asyncio.sleep(max(0, chunk["at"] - (time.perf_counter() - started)))
                yield chunk["text"]
            if saved.get("error"): raise ValueError(saved["error"])
            return
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        prompt = "\n".join(m["content"] for m in messages if m["role"] != "system")
        bound = len(system.encode()) + len(prompt.encode()) + 256 + self.config["model"]["components"][component]["max_tokens"]
        async with self.lock:
            if self.used + self.reserved + bound > self.config["evaluation"]["task_token_budget"]:
                self._not_sent(cache, component, "shared task token budget exhausted")
                raise TokenBudgetExceeded("shared task token budget exhausted")
            self.reserved += bound
        client = self.client_factory(self.config["model"], token_budget=self.config["evaluation"]["task_token_budget"])
        ledger_key, reserved = str(cache.resolve()), False
        chunks, failure, complete = [], None, False
        started = time.perf_counter()
        try:
            if self.budget:
                reserved = self.budget.reserve(ledger_key, bound)
            atomic_json(cache, {"state": "inflight", "usage_complete": False})
            async with aclosing(client.stream(system, prompt, component=component, seed=seed,
                                              **format_options(client, "stream", response_format))) as stream:
                async for chunk in stream:
                    chunks.append({"at": time.perf_counter() - started, "text": chunk})
                    yield chunk
            complete = bool(client.accounting()["usage_complete"])
            if not complete:
                raise BaselineUnavailable("stream ended without measured usage")
        except BaseException as exc:
            failure = type(exc).__name__ + ": " + str(exc)
            if isinstance(exc, TokenBudgetExceeded) and not client.records:
                client.records.append({"component": component, "request_status": "not_sent",
                                       "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
            raise
        finally:
            usage = client.accounting()
            complete = bool(usage["usage_complete"] and client.records)
            charged = usage["total_tokens"] if complete else bound
            atomic_json(cache, {"state": "finished" if complete else "incomplete",
                "chunks": chunks, "records": client.records, "charged": charged,
                "usage_complete": complete, "error": failure,
                "terminal_status": "failed" if failure else "complete"})
            self.records.extend(client.records)
            self.used += charged
            self.reserved -= bound
            self.incomplete |= not complete
            if self.budget and reserved:
                self.budget.finish(ledger_key, bound, usage["total_tokens"], complete)

    def accounting(self):
        missing = any(r.get("usage_missing") for r in self.records)
        total = sum(
            r.get("usage", {}).get("prompt_tokens", 0)
            + r.get("usage", {}).get("completion_tokens", 0)
            for r in self.records
        )
        return {
            "total_tokens": total,
            "physical_tokens": sum(r.get("physical_tokens", r.get("usage", {}).get("prompt_tokens", 0) + r.get("usage", {}).get("completion_tokens", 0)) for r in self.records),
            "usage_complete": not missing and not self.incomplete,
            "requests": len(self.records),
        }


def native_llm(session, component, scope):
    from evoagentx.models.base_model import BaseLLM
    from evoagentx.models.model_configs import LLMConfig

    class Bridge(BaseLLM):
        def init_model(self):
            pass

        def formulate_messages(self, prompts, system_messages=None):
            system_messages = system_messages or ["You are a helpful assistant."] * len(
                prompts
            )
            return [
                [{"role": "system", "content": s or ""}, {"role": "user", "content": p}]
                for p, s in zip(prompts, system_messages)
            ]

        def single_generate(self, messages, **kwargs):
            return asyncio.run(self.single_generate_async(messages, **kwargs))

        def batch_generate(self, batch_messages, **kwargs):
            return [self.single_generate(m, **kwargs) for m in batch_messages]

        async def single_generate_async(self, messages, **kwargs):
            return await session.text(messages, component, scope)

    return Bridge(
        LLMConfig(
            llm_type="CompactFlowModelClient", model=session.config["model"]["name"]
        )
    )


def public_inputs(task):
    if task.get("attachments") and task.get("benchmark") != "GAIA":
        raise UnsupportedWorkflow("attachments require a released tool adapter")
    question = task["question"]
    if task.get("context"):
        question = (
            "Context: "
            + json.dumps(task["context"], ensure_ascii=False)
            + "\n\nQuestion: "
            + question
        )
    match = re.search(r"\bdef\s+(\w+)\s*\(", task["question"])
    # Never inspect hidden tests or reference code for an entry point.
    entry = match.group(1) if match else None
    return {"problem": question, "entry_point": entry}


def export_sew(graph, *, max_nodes):
    from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

    if not isinstance(graph, SequentialWorkFlowGraph):
        raise UnsupportedWorkflow("SEW requires SequentialWorkFlowGraph")
    info = graph.get_graph_info()
    if not 1 <= len(info["tasks"]) <= max_nodes:
        raise UnsupportedWorkflow("shared max_nodes exceeded")
    for t in info["tasks"]:
        if set(t.get("tool_names") or []) - {"gaia_agent"} or t.get("parse_func") or t.get("prompt_template"):
            raise UnsupportedWorkflow("unexportable SEW tool/parser/template")
        if not t.get("prompt") or not t.get("outputs"):
            raise UnsupportedWorkflow("SEW prompt/output required")
    graph._validate_workflow_structure()
    graph._check_workflow_inputs_outputs()
    return {"format": "sew_native_v1", "graph": json.loads(json.dumps(info))}


class NativeAdapter:
    method = ""

    def bind_run(self, output):
        self.run_output = Path(output)

    def __init__(self, *, client_factory=ModelClient):
        self.client_factory = client_factory
        self.search_budget = None
        self.config = None

    def preflight(self, config):
        from .baselines import BaselineUnavailable

        self.config = config
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            importlib.import_module(
                "evoagentx.optimizers."
                + ("aflow_optimizer" if self.method in {"aflow", "a2flow"} else "sew_optimizer")
            )
            importlib.import_module("evoagentx.compactflow.adapter")
        except ImportError as e:
            raise BaselineUnavailable(
                f"native {self.method} dependency missing: {e}"
            ) from e

    async def execute(self, candidate, public_task, *, seed, workspace, budget=None):
        from .baselines import BaselineUnavailable
        from .gaia import GaiaToolSession, GaiaUnavailable
        from .baseline_controls import SessionClient

        config = self.config
        session = ModelSession(
            config,
            seed,
            workspace / "model_calls",
            budget=budget,
            client_factory=self.client_factory,
        )
        start = time.perf_counter()
        answer = ""
        valid = False
        error = None
        infra = False
        gaia = None
        try:
            if public_task["benchmark"] == "GAIA":
                gaia = GaiaToolSession(config,public_task,output=self.run_output,workspace=workspace,seed=seed,model_session=session)
            artifact = candidate.artifact
            inputs = public_inputs(public_task)
            cap = config["execution"]["external_call_capacity"]
            if artifact["format"] == "aflow_static_v1":
                from evoagentx.workflow import operators

                def factory(op, scope):
                    if op == "GAIAToolAgent":
                        if gaia is None:
                            raise UnsupportedWorkflow("GAIA operator outside GAIA benchmark")
                        async def call(input, instruction="Solve the task."):
                            value = await gaia.run_agent(SessionClient(session),instruction,{"problem":input},["response"],scope=scope)
                            return value
                        return call
                    native_op = {"A2Solve": "Custom", "A2Verify": "Custom", "A2Aggregate": "Custom"}.get(op, op)
                    return getattr(operators, native_op)(
                        llm=native_llm(session, "executor", scope)
                    )

                graph = lower_aflow(artifact, inputs, factory, cap)
                result = await CompactFlowRuntime(
                    graph,
                    mode="complete",
                    call_timeout=config["execution"]["call_timeout_seconds"],
                    workflow_timeout=config["execution"]["workflow_timeout_seconds"],
                ).execute({})
                if result.errors:
                    raise ValueError(str(result.errors))
                answer = expression_value(
                    artifact["result"],
                    inputs,
                    {k: v["result"] for k, v in result.outputs.items()},
                )
                if isinstance(answer, dict):
                    raise UnsupportedWorkflow(
                        "workflow return must be final answer string"
                    )
            elif artifact["format"] == "sew_native_v1":
                from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph
                from evoagentx.agents.agent_manager import AgentManager
                from .adapter import CompactFlowWorkFlow, NodeExecutionContract

                native = SequentialWorkFlowGraph.from_dict(
                    copy.deepcopy(artifact["graph"])
                )
                bridge = native_llm(session, "executor", "sew_executor")
                contracts = {
                    n.name: NodeExecutionContract(resources={"external_call": 1})
                    for n in native.nodes
                }
                manager = AgentManager()
                operations = {}
                if gaia:
                    task_specs = {t["name"]:t for t in artifact["graph"]["tasks"]}
                    for node in native.nodes:
                        async def operation(_node=node, _prompt=task_specs[node.name]["prompt"], **arguments):
                            return await gaia.run_agent(SessionClient(session),_prompt,arguments,[p.name for p in _node.outputs],scope=_node.name)
                        operations[node.name] = operation
                else:
                    manager.add_agents_from_workflow(native,llm_config=bridge.config,llm=bridge)
                runtime = CompactFlowWorkFlow(
                    native,
                    mode="complete",
                    llm=bridge,
                    agent_manager=manager,
                    contracts=contracts,
                    operations=operations,
                    resource_capacity={"external_call": cap},
                )
                if gaia:
                    from .gaia_contracts import annotate_graph
                    runtime.gfrg = annotate_graph(runtime.gfrg, {n.name:"gaia_agent" for n in native.nodes}, footprint_source="native_workflow_declared_bindings")
                execution = await CompactFlowRuntime(
                    runtime.gfrg,
                    mode="complete",
                    call_timeout=config["execution"]["call_timeout_seconds"],
                    workflow_timeout=config["execution"]["workflow_timeout_seconds"],
                ).execute({"problem": inputs["problem"]})
                values, missing = runtime._project_outputs(execution)
                if execution.errors or missing:
                    raise ValueError(str(execution.errors or missing))
                answer = values.get(
                    "answer", next(reversed(values.values())) if values else ""
                )
            else:
                raise UnsupportedWorkflow("unknown canonical workflow format")
            if not isinstance(answer, str):
                raise ValueError("non-string answer")
            valid = True
        except UnsupportedWorkflow:
            raise
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            infra = isinstance(e, (BaselineUnavailable, GaiaUnavailable, ImportError, OSError))
        usage = session.accounting()
        if gaia and gaia.incomplete:
            infra = True
        if not usage["usage_complete"]:
            infra = True
        return {
            "runtime": "complete_dependency",
            "tool_accounting": gaia.accounting() if gaia else {},
            "answer": answer,
            "valid": valid,
            "error": error,
            "infrastructure_error": infra,
            "tokens": usage,
            "model_requests": session.records,
            "latency": time.perf_counter() - start,
            "topology": graph_topology(candidate.artifact),
        }

    async def search(self, source_tasks, *, seed, config, workspace, score):
        from .baselines import WorkflowCandidate, BaselineUnavailable

        self.preflight(config)
        saved = workspace / "candidates.json"
        limit = config["evaluation"]["benchmark_seed_search_token_budget"]
        self.search_budget = SearchBudget(workspace / "search_budget.json", limit)
        if saved.exists():
            value = json.loads(saved.read_text())
            if value.get("incomplete"):
                raise BaselineUnavailable(value["incomplete"])
            return [WorkflowCandidate(**c) for c in value["candidates"]]
        self._source, self._seed, self._workspace, self._score = (
            source_tasks,
            seed,
            workspace,
            score,
        )
        self._found = {}
        self._failures = []
        # Native optimize() owns event loops; keep it off the runner loop.
        try:
            await asyncio.to_thread(self._search_sync)
        except TokenBudgetExceeded as exc:
            # The bounded search ends with the workflows already produced;
            # validation still uses the same ledger and unchanged budget.
            self._failures.append({"status": "budget_exhausted", "reason": str(exc)})
        except BaselineUnavailable:
            raise
        except Exception as e:
            raise BaselineUnavailable(
                f"{self.method} optimizer failed: {type(e).__name__}: {e}"
            ) from e
        candidates = list(self._found.values())
        if any(not row.get("usage_complete") for row in self.search_budget.entries.values()):
            reason = "native search has unknown usage; candidates cannot be selected"
            atomic_json(saved, {"incomplete": reason, "candidates": [], "rejected_exports": self._failures})
            raise BaselineUnavailable(reason)
        atomic_json(
            saved,
            {
                "candidates": [asdict(c) for c in candidates],
                "rejected_exports": self._failures,
            },
        )
        return candidates

    def _candidate(self, artifact):
        from .baselines import WorkflowCandidate

        c = WorkflowCandidate.create(
            self.method,
            self._source[0]["benchmark"],
            artifact,
            generation_seed=self._seed,
        )
        self._found[c.workflow_id] = c
        return c

    def _session(self, scope):
        return ModelSession(
            self.config,
            self._seed,
            self._workspace / scope,
            budget=self.search_budget,
            client_factory=self.client_factory,
        )


@lru_cache(maxsize=1)
def _optimizer_types():
    # Define registered framework subclasses once, not once per seed/run.
    from evoagentx.optimizers.aflow_optimizer import AFlowOptimizer
    from evoagentx.optimizers.sew_optimizer import SEWOptimizer

    class CompactFlowBoundAFlow(AFlowOptimizer):
        def init_module(self, **kwargs):
            self.bridge_init(self)

        async def _execute_with_retry(self, func, max_retries=1):
            return await func()

    class CompactFlowBoundSEW(SEWOptimizer):
        def step(self, **kwargs):
            self._pending_graph = super().step(**kwargs)
            return self._pending_graph

        def evaluate(self, dataset, eval_mode="dev", graph=None, **kwargs):
            if eval_mode != "dev":
                raise ValueError("source evaluation only")
            return self.bridge_evaluate(
                graph or getattr(self, "_pending_graph", self.graph)
            )

        def convergence_check(self, **kwargs):
            return self.bridge_convergence()

    return CompactFlowBoundAFlow, CompactFlowBoundSEW


class AFlowAdapter(NativeAdapter):
    method = "aflow"

    def _prepare_search(self):
        pass

    def _export_graph(self, graph_source, prompt_source):
        return export_aflow(graph_source, prompt_source,
                            max_nodes=self.config["baselines"]["shared"]["max_nodes"])

    def _initial_graph(self, graph_source, instruction):
        return graph_source, "SOLVE = " + repr(instruction)

    def _operators(self):
        return self.config["baselines"][self.method]["operators_by_benchmark"][self._source[0]["benchmark"]]

    def _operator_description(self, operator):
        return None

    def _search_constraints(self):
        return ""

    def _search_sync(self):
        from .baselines import FrozenBenchmarkView, BaselineUnavailable
        from evoagentx.utils.aflow_utils.graph_utils import GraphUtils, OPERATOR_MAP

        parent = self
        cfg = self.config["baselines"][self.method]
        self._prepare_search()
        root = self._workspace / ("native_" + self.method)
        # Replay native rounds from durable model/evidence journals after a crash.
        # Remove only this runner-owned scratch directory to avoid stale experience.
        import shutil

        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        graph_source = """class Workflow:
    def __init__(self, name, llm_config, benchmark):
        self.name = name
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
    async def __call__(self, problem):
        result = await self.custom(input=problem, instruction=prompt_custom.SOLVE)
        return result['response']
"""
        instruction = "Solve the task. Return only the final answer. "
        if self._source[0]["benchmark"] == "GAIA":
            graph_source = graph_source.replace("operator.Custom", "operator.GAIAToolAgent")
            instruction += "Use the shared GAIA tools to inspect attachments and retrieve evidence. "
        if self._source[0]["benchmark"] == "MBPP":
            instruction = "Write Python code solving the specification, with the requested function names. Return code only. "
        graph_source, prompt_source = self._initial_graph(graph_source, instruction)
        for name, value in {
            "graph.py": graph_source,
            "prompt.py": prompt_source,
        }.items():
            (root / name).write_text(value)

        class StaticGraphUtils(GraphUtils):
            def _load_operator_description(self, index, operator, llm):
                custom = parent._operator_description(operator)
                if custom is not None:
                    return custom
                if operator == "GAIAToolAgent":
                    return "GAIAToolAgent(llm): async __call__(input: str, instruction: str) -> {'response': str}. Shared bounded GAIA attachment/search/browser/vision/audio/Python executor. Use operator.GAIAToolAgent(self.llm)."
                return super()._load_operator_description(index,operator,llm)

            def create_graph_optimize_prompt(self, *args, **kwargs):
                return super().create_graph_optimize_prompt(*args, **kwargs) + parent._search_constraints()

            def update_prompt_import(self, *args):
                pass

            def load_graph(self, round_number, workflows_path):
                p = Path(workflows_path) / f"round_{round_number}"
                try:
                    return parent._export_graph(
                        (p / "graph.py").read_text(),
                        (p / "prompt.py").read_text(),
                    )
                except (UnsupportedWorkflow, SyntaxError) as e:
                    parent._failures.append({"round": round_number, "reason": str(e)})
                    return None

        class SharedEvaluation:
            async def evaluate_graph_async(
                self, optimizer, validation_n, data, initial=False
            ):
                round_id = optimizer.round if initial else optimizer.round + 1
                if optimizer.graph is None:
                    scores = [0.0]
                else:
                    candidate = parent._candidate(optimizer.graph)
                    scores = [
                        await parent._score(candidate, s, budget=parent.search_budget)
                        for s in parent.config["construction"]["heldout_seeds"][
                            :validation_n
                        ]
                    ]
                score = sum(scores) / len(scores)
                data.append(
                    optimizer.data_utils.create_result_data(round_id, score, 0, 0)
                )
                optimizer.data_utils.save_results(
                    optimizer.data_utils.get_results_file_path(str(root)), data
                )
                return score

        def initialize(opt):
            from evoagentx.utils.aflow_utils.data_utils import DataUtils
            from evoagentx.utils.aflow_utils.experience_utils import ExperienceUtils
            from evoagentx.utils.aflow_utils.convergence_utils import ConvergenceUtils

            opt.root_path = opt.optimized_path or opt.graph_path
            opt.graph_utils = StaticGraphUtils(opt.root_path)
            opt.data_utils = DataUtils(opt.root_path)
            opt.experience_utils = ExperienceUtils(opt.root_path)
            opt.convergence_utils = ConvergenceUtils(opt.root_path)
            opt.evaluation_utils = SharedEvaluation()
            opt.round = 0
            opt.graph = None
            p = Path(opt.root_path) / "round_0"
            p.mkdir(parents=True, exist_ok=True)
            for name in ("graph.py", "prompt.py"):
                (p / name).write_text((Path(opt.graph_path) / name).read_text())

        BoundAFlow, _ = _optimizer_types()
        operators = self._operators()
        if any(op not in OPERATOR_MAP and op != "GAIAToolAgent" and self._operator_description(op) is None for op in operators):
            raise BaselineUnavailable("configured AFlow operator not implemented")
        random.seed(self._seed)
        import numpy as np

        np.random.seed(self._seed)
        session = self._session("optimizer_requests")
        llm = native_llm(session, "planner", self.method + "_optimizer")
        opt = BoundAFlow(
            bridge_init=initialize,
            graph_path=str(root),
            optimized_path=str(root),
            optimizer_llm=llm,
            executor_llm=llm,
            operators=operators,
            question_type="code"
            if self._source[0]["benchmark"] == "MBPP"
            else "math"
            if self._source[0]["benchmark"] == "MATH"
            else "qa",
            sample=cfg["population_sample"],
            max_rounds=cfg["max_rounds"],
            validation_rounds=cfg["validation_rounds"],
            eval_rounds=cfg["test_rounds"],
            check_convergence=False,
        )
        opt.optimize(FrozenBenchmarkView(self._source[0]["benchmark"], self._source))
        atomic_json(
            self._workspace / "search_accounting.json",
            {
                "tokens": session.accounting(),
                "ledger": self.search_budget.entries,
                "implementation": "evoagentx.optimizers.aflow_optimizer.AFlowOptimizer",
                "adapter": "static export + shared runtime scoring",
            },
        )


class A2FlowAdapter(AFlowAdapter):
    """Paper-faithful A2Flow construction adapter on top of EvoAgentX AFlow.

    The paper's adaptive operator extraction is performed only from public source
    task traces.  Generated Python is represented as bounded operator metadata;
    it is never imported or executed on the host.
    """
    method = "a2flow"

    def _operator_names(self):
        return ["A2Solve", "A2Verify", "A2Aggregate"]

    def _prepare_search(self):
        root = self._workspace / "a2flow_extraction"
        root.mkdir(parents=True, exist_ok=True)
        marker = root / "operators.json"
        if marker.exists():
            self._a2ops = json.loads(marker.read_text())
            return
        # Six paths x three refinements, matching Sec. 3.3.  Each request is
        # source-only and charged through the common search budget.
        session = self._session("a2flow_extraction")
        records = []
        async def extract():
            for path in range(6):
                memory = []
                for step in range(3):
                    prompt = json.dumps({"task": self._source[0]["question"],
                        "path": path, "step": step, "memory": memory,
                        "instruction": "Return one bounded operator description."})
                    text = await session.text([{"role":"user","content":prompt}], "distiller", f"path_{path}")
                    memory.append(text[:2000]); records.append({"path":path,"step":step,"text":text})
            return records
        try:
            records = asyncio.run(extract())
        except RuntimeError:
            loop = asyncio.new_event_loop(); records = loop.run_until_complete(extract()); loop.close()
        self._a2ops = {"operators": self._operator_names(), "paths": records,
                       "method":"a2flow_paper_reimplementation_v1"}
        atomic_json(marker, self._a2ops)

    def _operators(self):
        return self._operator_names()

    def _operator_description(self, operator):
        labels = {"A2Solve":"Source-derived abstract solve operator with task-local memory.",
                  "A2Verify":"Source-derived verification operator.",
                  "A2Aggregate":"Source-derived answer aggregation operator."}
        return f"{operator}: {labels[operator]}"

    def _search_constraints(self):
        return "\nUse only the extracted A2Flow operators A2Solve, A2Verify, A2Aggregate; preserve task-local operator memory and bounded calls."

    def _export_graph(self, graph_source, prompt_source):
        specs = {n:{"input","instruction"} for n in self._operator_names()}
        return export_aflow(graph_source, prompt_source,
                            max_nodes=self.config["baselines"]["shared"]["max_nodes"],
                            operator_specs=specs)

    def _initial_graph(self, graph_source, instruction):
        graph_source = graph_source.replace("operator.Custom", "operator.A2Solve")
        return graph_source, "SOLVE = " + repr(instruction)



class EvoAgentXAdapter(NativeAdapter):
    method = "evoagentx"

    def _search_sync(self):
        from .baselines import FrozenBenchmarkView
        from evoagentx.evaluators.evaluator import Evaluator
        from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

        parent = self
        cfg = self.config["baselines"]["evoagentx"]
        random.seed(self._seed)
        parameter = lambda n: {
            "name": n,
            "type": "string",
            "required": True,
            "description": n,
        }
        instruction = "Solve the task and return only the final answer.\n{problem}"
        if self._source[0]["benchmark"] == "MBPP":
            instruction = (
                "Write Python code solving this task. Return code only.\n{problem}"
            )
        graph = SequentialWorkFlowGraph(
            goal="Solve " + self._source[0]["benchmark"] + " tasks",
            tasks=[
                {
                    "name": "Solve",
                    "description": "Solve task",
                    "inputs": [parameter("problem")],
                    "outputs": [parameter("answer")],
                    "prompt": instruction,
                    "parse_mode": "str",
                }
            ],
        )
        if self._source[0]["benchmark"] == "GAIA":
            graph.nodes[0].tool_names = ["gaia_agent"]
        session = self._session("optimizer_requests")
        llm = native_llm(session, "planner", "sew_optimizer")

        def score_graph(graph):
            try:
                artifact = export_sew(
                    graph, max_nodes=parent.config["baselines"]["shared"]["max_nodes"]
                )
            except UnsupportedWorkflow as e:
                parent._failures.append({"reason": str(e)})
                return {"quality": 0.0}
            c = parent._candidate(artifact)
            scores = [
                asyncio.run(parent._score(c, s, budget=parent.search_budget))
                for s in parent.config["construction"]["heldout_seeds"][
                    : cfg["validation_rounds"]
                ]
            ]
            return {"quality": sum(scores) / len(scores)}

        _, BoundSEW = _optimizer_types()
        # YAML uses the native safe parser instead of its Python eval path.
        opt = BoundSEW(
            bridge_evaluate=score_graph,
            bridge_convergence=lambda: (
                parent.search_budget.used >= parent.search_budget.limit
            ),
            graph=graph,
            evaluator=Evaluator(llm),
            llm=llm,
            max_steps=cfg["max_iterations"],
            eval_every_n_steps=1,
            eval_rounds=cfg["validation_rounds"],
            repr_scheme="yaml",
            optimize_mode="all",
            order="zero-order",
        )
        opt.optimize(FrozenBenchmarkView(self._source[0]["benchmark"], self._source))
        atomic_json(
            self._workspace / "search_accounting.json",
            {
                "tokens": session.accounting(),
                "ledger": self.search_budget.entries,
                "implementation": "evoagentx.optimizers.sew_optimizer.SEWOptimizer",
                "adapter": "native lower_workflow_graph + shared runtime scoring",
            },
        )
