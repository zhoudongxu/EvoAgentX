# CompactFlow

**Self-Evolving Compact Workflow Construction and Guarded Opportunistic
Execution**

[English](./README.md) | [简体中文](./README-zh.md)

> **Implementation scope for review.** This repository uses
> [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) as its base agent
> framework. The contribution in this repository is the CompactFlow
> construction plane, execution plane, policy-evolution logic, experiment
> infrastructure, and tests described below. The core implementation is under
> `evoagentx/compactflow`; it is not part of the upstream EvoAgentX project.

CompactFlow targets two sources of inefficiency in agentic workflows:

1. **Construction-time redundancy:** workflows may contain unnecessary
   agents, edges, routes, or stopping steps.
2. **Execution-time blocking:** a downstream call often waits for an entire
   upstream result even when its exact required fields are already stable.

The implementation addresses both problems while retaining explicit quality,
readiness, resource, and side-effect checks. It is an optional sidecar:
existing EvoAgentX workflow classes and their default runtime remain
available.

## Review map

The following paths contain the implementation introduced for CompactFlow:

| Path | Review target |
| --- | --- |
| `evoagentx/compactflow/` | Construction plane, policy library/evolution, graph compiler, guarded runtime, EvoAgentX adapter, metrics, and experiment utilities |
| `examples/compactflow/run_experiments.py` | Executable offline smoke experiment and configuration validation CLI |
| `examples/compactflow/configs/` | Runnable smoke configuration and non-runnable formal-evaluation protocol template |
| `examples/compactflow/policies/` | Seed compactness-policy templates |
| `tests/src/compactflow/` | Unit, integration, safety, metrics, and artifact-privacy tests |
| `examples/compactflow/README.md` | Detailed API, invariants, configuration, and experiment documentation |

Files elsewhere in `evoagentx/` are primarily the upstream framework on which
the integration is built. This separation lets reviewers inspect the
CompactFlow contribution directly without conflating it with EvoAgentX.

## What is implemented

| Component | Delivered behavior | Status |
| --- | --- | --- |
| Construction plane | Retrieves and reranks compactness policies, selects a compatible subset, conditions the native planner, validates the constructed `WorkFlowGraph`, and records planner/fallback traces. | Implemented |
| Policy evolution | Maintains a quality-cost Pareto archive, strictly pairs baseline and candidate evidence, applies deterministic `ADMIT`/`MERGE`/`REJECT` decisions, and atomically persists the library. | Implemented; the candidate-distillation model is pluggable |
| Execution compiler | Lowers an EvoAgentX `WorkFlowGraph` into a guarded function-call relation graph (GFRG), validates JSON Schemas, resolves exact field paths, and synthesizes readiness guards. | Implemented |
| Guarded runtime | Processes `Partial`, `Complete`, and `Failure` events, releases consumers only when compiler-validated explicit guards hold, enforces effect/resource constraints, propagates failures, and dispatches each logical call at most once. | Implemented |
| Native integration | Executes native action-graph and agent-backed nodes, with explicit async operation overrides for streaming and early dispatch. | Implemented |
| Evaluation support | Provides deterministic splits, paired quality/cost/latency/structure metrics, readiness and safety diagnostics, traces, anonymous manifests, and a network-free smoke experiment. | Implemented |
| Formal four-benchmark suite | Specifies the intended MBPP, HotPotQA, MATH, and GAIA protocol in a validated configuration template. | Protocol scaffold only; formal runner and results are not included |

## Architecture

```text
                              CONSTRUCTION PLANE

 Task + schemas
      |
      v
 Policy retrieval --> applicability/utility reranking --> compatibility selection
      ^                                                        |
      |                                                        v
 Policy library <---- ADMIT / MERGE / REJECT <---- policy-conditioned planner
      ^                                                        |
      |                                                        v
 Paired evidence <---- quality + cost metrics <---- validated WorkFlowGraph
                                                               |
                            EXECUTION PLANE                     |
                                                               v
                         native graph lowering
                                  |
                                  v
              exact field dependencies + effect dependencies
                                  |
                                  v
            schema validation + stable-field guard synthesis
                                  |
                                  v
         Partial / Complete / Failure event-driven scheduling
                                  |
                                  v
        workflow result + latency/token/structure/safety traces
```

## Construction plane

### Policy retrieval and compatible selection

`PolicyLibrary` stores typed condition-operation policies and their execution
evidence in deterministic JSON. The retrieval path:

1. performs semantic top-\(K_0\) retrieval;
2. reranks candidates using semantic similarity, structured applicability,
   historical utility, and confidence; and
3. selects at most \(k\) mutually compatible policies while recording why
   other candidates were skipped.

Normal retrieval uses only `verified` policies. The included seed templates
start as `candidate` policies and enter retrieval only when exploration is
explicitly enabled. They are not presented as benchmark-verified policies.

Main code:

- `evoagentx/compactflow/models.py`
- `evoagentx/compactflow/policy.py`
- `evoagentx/compactflow/construction.py`

### Policy-conditioned construction

`PolicyGuidedWorkflowGenerator` wraps EvoAgentX's existing
`WorkFlowGenerator`. It renders the selected policies into the planner input,
constructs the compact workflow directly, and validates:

- DAG structure and reachability;
- node executors;
- typed node interfaces; and
- graph consistency.

The wrapper stores a `PlannerTrace`. If conditioned planning or validation
fails, it can fall back to the original planner; strict experiments can
disable this fallback.

Main code: `evoagentx/compactflow/planner.py`.

### Evidence-based policy evolution

Execution evidence is matched by
`(benchmark, split, task_id, seed)`, rather than insertion order. Candidate
policies are assessed using paired evidence from the split that the experiment
runner designates as held out:

```text
quality:   mean(candidate - baseline) >= -epsilon_Q
cost:      mean(normalized cost reduction) >= delta_C
validity:  candidate valid rate >= configured minimum
novelty:   nearest-policy similarity < merge threshold
```

The evolution result is deterministic:

- `ADMIT` when quality, cost, validity, and novelty pass;
- `MERGE` when quality, cost, and validity pass but the candidate is a near
  duplicate; or
- `REJECT` otherwise.

The package implements evidence pairing, Pareto filtering, decision logic,
merge semantics, and persistence. The model that distills a new
`CompactnessPolicy` candidate is intentionally injectable rather than tied to
one LLM provider.

## Execution plane

### Native workflow lowering

`lower_workflow_graph` converts a native EvoAgentX `WorkFlowGraph` into a
guarded function-call relation graph:

- input/output `Parameter` objects become JSON Schemas;
- matching producer and consumer fields become field-level data dependencies;
- control-only edges become effect-order dependencies;
- action-graph nodes execute directly;
- agent-backed nodes execute as isolated one-node workflows; and
- an explicit operation can override either form to expose an async stream.

Main code:

- `evoagentx/compactflow/schema.py`
- `evoagentx/compactflow/adapter.py`

### Exact readiness guards

A consumer is released on partial producer output only when all required
safety conditions hold:

1. the dependency identifies an exact source and target field path;
2. the dependency permits early use;
3. the producer declares a monotone `StreamContract`;
4. the exact source path is listed in the contract's `stable_fields`;
5. the emitted `Partial` event also marks that path stable;
6. the consumer declares `early_safe=True`;
7. the consumer is a cancellable native coroutine or async generator; and
8. effect-order and resource predicates are satisfied.

Otherwise the compiler retains a complete-result barrier. Prefix matching is
not used: declaring `document` stable does not implicitly stabilize
`document.title`.

The compiler validates each declared JSON Schema against its metaschema. The
runtime validates call arguments, accumulated partial outputs, and complete
outputs. Partial validation relaxes only `required`; every field already
present must still satisfy its type and nested constraints.

Main code: `evoagentx/compactflow/compiler.py`.

### Event-driven runtime

`CompactFlowRuntime` supports three directly comparable execution modes:

| Mode | Scheduling behavior |
| --- | --- |
| `sequential` | Runs at most one semantically ready call at a time |
| `complete` | Runs independent calls concurrently but keeps completion barriers for dependent data |
| `guarded` | Also releases dependent calls when compiler-validated explicit exact-field guard conditions become true |

All modes use one event loop over `Partial`, `Complete`, and `Failure` events.
The runtime records call states, arguments, outputs, relative timestamps,
latency, time to first runtime output, normalized traces, and violations. It
also:

- dispatches each logical call at most once;
- propagates failures to unreachable descendants;
- enforces vector resource capacities; and
- keeps effects behind completion barriers unless explicitly declared safe
  for partial release.

Main code: `evoagentx/compactflow/runtime.py`.

## Quick start

Install the project and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Native planner and agent-backed adapter integrations may require the optional
EvoAgentX dependencies:

```bash
python -m pip install -e ".[all,dev]"
```

Run the network-free implementation smoke experiment:

```bash
python examples/compactflow/run_experiments.py smoke \
  --output outputs/compactflow/smoke
```

The command produces:

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Sanitized configuration, configuration digest, dataset fingerprint/counts, and runtime version |
| `records.jsonl` | Strict benchmark-task-seed-method records |
| `records.csv` | Flat export of the same records |
| `summary.json` | Paired quality, cost, timing, and diagnostic aggregates |
| `traces.json` | Relative-time scheduling traces |

Local smoke timing is an implementation check, not a paper result.

## Minimal integration

Construction plane:

```python
from evoagentx.compactflow.construction import (
    ConstructionConfig,
    ConstructionPlane,
)
from evoagentx.compactflow.planner import PolicyGuidedWorkflowGenerator
from evoagentx.compactflow.policy import PolicyLibrary, RetrievalConfig

library = PolicyLibrary.load(
    "examples/compactflow/policies/seed_policies.json"
)
# The bundled seeds are unverified candidates. Enable them only for
# exploration; a frozen evaluation should use verified policies.
construction = ConstructionPlane(
    library,
    config=ConstructionConfig(
        retrieval=RetrievalConfig(include_candidates=True),
    ),
)
generator = PolicyGuidedWorkflowGenerator(
    generator=workflow_generator,
    construction_plane=construction,
)

compact_graph = generator.generate_workflow(goal)
planner_trace = generator.last_trace
```

Guarded execution:

```python
from evoagentx.compactflow.adapter import (
    CompactFlowWorkFlow,
    NodeExecutionContract,
)
from evoagentx.compactflow.schema import ExecutionMode

workflow = CompactFlowWorkFlow(
    compact_graph,
    mode=ExecutionMode.GUARDED,
    operations={"retrieve": streaming_retrieve},
    contracts={
        "retrieve": NodeExecutionContract(
            stream_contract=retrieve_stream_contract,
        ),
        "summarize": NodeExecutionContract(early_safe=True),
    },
    resource_capacity={"external_call": 4},
)

result = await workflow.async_execute(inputs)
```

See [the detailed implementation guide](./examples/compactflow/README.md) for
complete contracts, nested field paths, evolution APIs, experiment records,
and failure semantics.

## Experiments and reproduction boundary

### Executable in this repository

- network-free construction/execution smoke experiment;
- strict configuration validation;
- deterministic source/validation/target splitting;
- paired record aggregation;
- quality, latency, graph, readiness, and safety diagnostics;
- token metric schema and aggregation, without claiming LLM token collection;
  and
- a sanitized manifest plus synthetic smoke JSON, JSONL, CSV, and trace
  artifacts.

### Not claimed as completed

`examples/compactflow/configs/paper_template.json` is deliberately marked
`"runner": "protocol_template"` and `"runnable": false`. It documents the
formal comparison protocol, but the current CLI does not implement:

- MBPP, HotPotQA, MATH, and GAIA end-to-end orchestration;
- AFlow and EvoAgentX baseline orchestration;
- online source/validation/target policy-evolution loops;
- LLM token collection and controlled replay;
- independent-only and percentage-threshold scheduling baselines; or
- a GAIA benchmark adapter.

The supplied paper material also leaves the backbone model, exact splits,
sample counts, rollout budget, seeds, selection weights, admission thresholds,
resource capacities, and replay settings unresolved. For these reasons, this
repository does **not** copy illustrative table values or claim formal
reproduction of the paper's numerical results.

## Current implementation boundaries

- Execution graphs are finite static DAGs.
- Every logical call currently has one attempt; retry-aware input-version
  invalidation and compensating side effects are not implemented.
- Provider/tool timeouts must be applied inside operations; there is no
  scheduler-level per-call timeout.
- Dynamic per-item call instantiation is not implemented.
- Native agent-backed nodes return complete results. Early overlap requires an
  explicitly instrumented async streaming operation and `StreamContract`.
- Synchronous operations run in worker threads and cannot be force-cancelled
  safely, so they never receive partial-data or partial-effect early dispatch.
- `ttfo` records the first output anywhere in the runtime, not necessarily the
  sink's first useful output.

## Validation

Run the CompactFlow test suite and style checks from the repository root:

```bash
python -m pytest -q tests/src/compactflow
ruff check evoagentx/compactflow \
  examples/compactflow/run_experiments.py \
  tests/src/compactflow
```

With only `.[dev]`, tests that require optional EvoAgentX integrations are
skipped when those dependencies are unavailable. Install `.[all,dev]` to
exercise the native planner and adapter integration tests.

The test suite covers policy persistence/retrieval, compatibility conflicts,
paired admission, schema/path validation, safe and unsafe streaming,
at-most-once dispatch, failure propagation, effect/resource constraints,
native lowering, metrics, configuration validation, stable splits, and
manifest sanitization.

## Base framework and license

CompactFlow is implemented on top of the open-source
[EvoAgentX framework](https://github.com/EvoAgentX/EvoAgentX). Upstream
framework documentation is available from that project. This repository keeps
the upstream license in [LICENSE](./LICENSE).
