# CompactFlow on EvoAgentX

## Implementation status

CompactFlow is implemented as an optional sidecar package under
`evoagentx/compactflow`. It adds construction-time workflow compaction,
guarded opportunistic execution, evidence-based policy evolution, and
experiment infrastructure without modifying the existing EvoAgentX workflow
classes.

| Component | Delivered behavior | Status |
| --- | --- | --- |
| Construction plane | Retrieves and reranks reusable compactness policies, selects a compatible subset, conditions the native EvoAgentX planner, validates the resulting `WorkFlowGraph`, and records fallback and planner traces. | Implemented |
| Policy evolution | Builds a quality-cost Pareto archive, strictly pairs baseline and candidate evidence by benchmark/split/task/seed, applies deterministic `ADMIT`/`MERGE`/`REJECT` rules, and persists the updated library atomically. | Implemented; candidate distillation model is pluggable |
| Execution compiler | Lowers a native `WorkFlowGraph` into a guarded function-call relation graph (GFRG), validates JSON Schemas, resolves exact field paths, and synthesizes stable-field readiness guards. | Implemented |
| Guarded runtime | Executes `Partial`, `Complete`, and `Failure` events, releases consumers only when compiler-validated explicit guard conditions hold, enforces effect/resource constraints, propagates failures, and dispatches each logical call at most once. | Implemented |
| EvoAgentX integration | Adapts native action-graph and agent-backed nodes while allowing explicit async streaming operations and contracts where early dispatch is required. | Implemented |
| Experiment infrastructure | Provides deterministic splits, strict paired records, quality/token/latency/structure/safety metrics, sanitized anonymous manifests, trace artifacts, and a network-free smoke experiment. | Implemented |
| Four-benchmark paper suite | Documents the required MBPP, HotPotQA, MATH, and GAIA comparisons in a validated configuration template. | Protocol scaffold only; formal runner and results are not included |

The supplied code therefore supports component and integration validation and
provides building blocks for custom benchmark integration. It does **not**
claim reproduction of the
paper's numerical tables: the paper configuration leaves key model, split,
budget, seed, and threshold values unresolved, and the repository does not
contain the missing formal runner or GAIA adapter.

### End-to-end flow

```text
task + tool schemas
        |
        v
policy retrieval -> applicability/utility reranking -> compatibility selection
        |                                                    |
        +---------------- selected policies -----------------+
                                                             v
                                              EvoAgentX workflow planner
                                                             |
                                                             v
                                                  validated WorkFlowGraph
                                                             |
                                                             v
                   exact field dependencies -> guard synthesis -> GFRG
                                                             |
                                                             v
                      partial / complete / failure event-driven execution
                                                             |
                                                             v
                    quality + tokens + latency + structure + violations
                                                             |
                                                             v
                  Pareto evidence -> paired verification -> policy update
```

The implementation is a sidecar package. Existing EvoAgentX workflow classes
are not modified, and normal workflows can continue to use the original
runtime.

### Source map

| Area | Main implementation |
| --- | --- |
| Policy and evidence data contracts | `evoagentx/compactflow/models.py` |
| Policy storage, retrieval, and compatible selection | `evoagentx/compactflow/policy.py` |
| Pareto archive and Admit/Merge/Reject evolution | `evoagentx/compactflow/construction.py` |
| Policy-conditioned EvoAgentX planner | `evoagentx/compactflow/planner.py` |
| Field-readiness graph and stream contracts | `evoagentx/compactflow/schema.py` |
| Static dependency analysis and guard synthesis | `evoagentx/compactflow/compiler.py` |
| Event-driven guarded runtime | `evoagentx/compactflow/runtime.py` |
| Native `WorkFlowGraph` lowering | `evoagentx/compactflow/adapter.py` |
| Paired metrics and execution diagnostics | `evoagentx/compactflow/metrics.py` |
| Config validation, stable splits, and artifacts | `evoagentx/compactflow/experiments.py` |

## Installation

The dependency-free CompactFlow logic and offline experiment need the normal
project dependencies plus the development test tools:

```bash
python -m pip install -e ".[dev]"
```

Using the native EvoAgentX planner and agent-backed workflow adapter may also
require the optional dependencies imported by the corresponding EvoAgentX
components:

```bash
python -m pip install -e ".[all,dev]"
```

The guarded runtime itself does not call an LLM and does not require network
access.

## Construction plane

### 1. Policy library and selection

`PolicyLibrary` stores typed condition-operation policies and execution
evidence in a deterministic JSON format. Writes are atomic. The default
retriever first applies semantic top-\(K_0\), then reranks with semantic
similarity, structured applicability, historical utility, and confidence.
`CompatibilitySelector` greedily selects at most \(k\) mutually compatible
policies and records why the remaining candidates were skipped.

The default text embedder is deterministic and dependency-free. It is intended
for offline tests and as a fallback; a production run should inject the same
embedding model used by the experiment protocol.

```python
from evoagentx.compactflow.construction import (
    ConstructionConfig,
    ConstructionPlane,
)
from evoagentx.compactflow.policy import (
    PolicyLibrary,
    RetrievalConfig,
)

library = PolicyLibrary.load(
    "examples/compactflow/policies/seed_policies.json"
)
plane = ConstructionPlane(
    library,
    config=ConstructionConfig(
        retrieval=RetrievalConfig(
            # Explicit exploration mode for the unverified seed templates.
            include_candidates=True,
        )
    ),
)

guidance = plane.prepare(
    query="compact a retrieval and aggregation workflow",
    context={
        "capabilities": {
            "typed_workflow": True,
            "explicit_streaming": True,
        }
    },
)
selected = guidance.selected_policies
```

The included seed library contains six policy templates: linear-chain fusion,
overlapping-role fusion, unused-output pruning, redundant-edge simplification,
stable-consensus stopping, and routing simplification. They start as
`candidate`; the file does not claim that they have passed a benchmark
verification run. Normal retrieval uses only `verified` policies. Candidate or
rejected policies enter retrieval only through explicit exploration flags; do
not enable those flags for a frozen target-split evaluation.

### 2. Policy-conditioned EvoAgentX planning

`PolicyGuidedWorkflowGenerator` wraps the existing `WorkFlowGenerator`. It
renders selected policies into the planner's existing `suggestion` input,
constructs the compact graph directly, validates DAG structure, reachability,
typed interfaces, and executors, and records a `PlannerTrace`.

```python
from evoagentx.compactflow.planner import PolicyGuidedWorkflowGenerator

compact_generator = PolicyGuidedWorkflowGenerator(
    generator=workflow_generator,
    construction_plane=plane,
)
graph = compact_generator.generate_workflow(goal)
trace = compact_generator.last_trace
```

If policy-conditioned planning or validation fails, the wrapper uses the base
generator without policy guidance. Set `fallback_on_error=False` when a run
must fail instead of falling back.

### 3. Evidence-grounded policy evolution

Execution evidence is grouped by benchmark, split, task, and seed, not by
insertion order. `ParetoArchive` maximizes quality while minimizing token,
latency, and graph cost. A distilled candidate is evaluated on paired
executions from the split that the experiment runner designates as held out:

```text
quality condition:  mean(candidate quality - baseline quality) >= -epsilon_Q
cost condition:     mean(normalized cost reduction) >= delta_C
validity condition: candidate valid rate >= configured minimum
novel candidate:    similarity to nearest policy < merge threshold
```

The deterministic verdict is:

- `ADMIT` when quality, cost, validity, and novelty conditions pass;
- `MERGE` when the first three pass but the candidate is a near duplicate; or
- `REJECT` otherwise.

```python
from evoagentx.compactflow.construction import pair_evidence

pairs = pair_evidence(
    baseline_evidence,
    candidate_evidence,
    strict=True,
)
decision = plane.evolve(
    distilled_policy,
    pairs,
    persist=True,
)
```

Policy distillation is intentionally model-pluggable: a caller creates a
`CompactnessPolicy` from its model output, while this package owns the
frontier filter, paired evidence matching, deterministic admission, merge, and
persistence semantics.

For source-target transfer, use `stable_hash_split` from
`evoagentx.compactflow.experiments`: discover policies only on `source`, admit
them only on `validation`, freeze the library, and evaluate on `target`.

## Execution plane

### 1. Lowering an EvoAgentX workflow

`lower_workflow_graph` converts a native `WorkFlowGraph` into a guarded
function-call relation graph (GFRG):

- node input/output `Parameter` objects become JSON schemas;
- matching producer output and consumer input names become field-level data
  dependencies;
- control-only graph edges become effect-order dependencies;
- action-graph nodes execute directly;
- agent-backed nodes execute as isolated one-node workflows; and
- explicit operations can override either form, which is the recommended way
  to expose streaming tools.

The compiler validates every declared JSON Schema against its metaschema. The
runtime then validates call arguments, every accumulated partial output, and
the complete output. Partial validation relaxes only `required`; fields that
are present must still satisfy their declared types, nested constraints, and
`additionalProperties`.

```python
from evoagentx.compactflow.adapter import (
    CompactFlowWorkFlow,
    NodeExecutionContract,
)
from evoagentx.compactflow.schema import ExecutionMode

workflow = CompactFlowWorkFlow(
    graph,
    mode=ExecutionMode.GUARDED,
    operations={"retrieve": streaming_retrieve},
    contracts={
        "retrieve": NodeExecutionContract(
            stream_contract=retrieve_contract,
        ),
        "summarize": NodeExecutionContract(early_safe=True),
    },
    resource_capacity={"external_call": 4},
)
result = await workflow.async_execute(inputs)
```

Top-level fields are inferred from native parameters. Use
`extra_data_dependencies` for exact nested paths such as
`documents[0].content`.

### 2. Exact guard synthesis

A partial producer output can release a consumer only when all of the
following are true:

1. the dependency identifies an exact source and target field path;
2. the dependency allows early use;
3. the producer declares a monotone `StreamContract`;
4. the exact source path is listed in `stable_fields`;
5. the emitted `Partial` event also marks that path stable;
6. the consumer explicitly declares `early_safe=True`;
7. the consumer is a native coroutine or async-generator function that can
   actually be cancelled; and
8. effect-order and resource predicates are satisfied.

Otherwise the compiler retains a complete-result barrier. Prefix matching is
not used: declaring `document` stable does not implicitly make
`document.title` stable. A partial effect dependency likewise requires a
native async consumer.

```python
from evoagentx.compactflow import (
    CallSpec,
    Complete,
    DataDependency,
    Partial,
    StreamContract,
)

async def produce():
    yield Partial(
        {"first_item": "ready"},
        stable_fields=("first_item",),
    )
    yield Complete(
        {"first_item": "ready", "remaining_items": ["later"]},
    )

producer = CallSpec(
    id="producer",
    target=produce,
    output_schema=producer_schema,
    stream_contract=StreamContract(
        stable_fields=("first_item",),
        mutable_fields=("remaining_items",),
    ),
)
consumer = CallSpec(
    id="consumer",
    target=consume,
    input_schema=consumer_schema,
    early_safe=True,
)
edge = DataDependency(
    producer="producer",
    consumer="consumer",
    source_path="first_item",
    target_path="item",
)
```

### 3. Runtime modes and invariants

`CompactFlowRuntime` supports:

| Mode | Behavior |
| --- | --- |
| `sequential` | Runs at most one semantically ready call at a time. |
| `complete` | Runs independent calls concurrently but waits for producer completion on dependent data. |
| `guarded` | Also releases dependent calls when compiler-validated explicit exact-field guard conditions become true. |

The runtime processes `Partial`, `Complete`, and `Failure` events through one
queue, dispatches each logical call at most once, propagates failures to
unreachable descendants, and records:

- call state, arguments, outputs, and timestamps;
- end-to-end latency and time to first runtime output;
- normalized event traces;
- argument mismatch, duplicate dispatch, effect-order, and resource-capacity
  violations.

Resources are vectors. A call runs only when every declared demand fits the
remaining capacity. Effects remain completion barriers unless a dependency
explicitly permits a partial effect.

## Experiments

### Offline implementation smoke

The smoke run is network-free and uses scripted async functions. It checks:

- policy retrieval and compatibility selection;
- guarded fan-out from two independently stable producer fields;
- fallback to completion for a mutable field;
- quality parity between complete-dependency and guarded modes;
- strict paired aggregation and readiness diagnostics; and
- anonymous JSON, JSONL, CSV, and trace artifacts.

Run it with:

```bash
python examples/compactflow/run_experiments.py smoke
```

Use an isolated output directory when comparing repeated runs:

```bash
python examples/compactflow/run_experiments.py smoke \
  --output outputs/compactflow/smoke-run
```

Generated files:

| File | Contents |
| --- | --- |
| `manifest.json` | Sanitized configuration, configuration digest, dataset fingerprint/counts, and runtime version |
| `records.jsonl` | One strict benchmark-task-seed-method record per line |
| `records.csv` | Flat export of the same records |
| `summary.json` | Paired quality/cost/timing comparison and diagnostics |
| `traces.json` | Relative-time event traces for schedule inspection |

Local smoke timings are expected to vary with the operating system and load.
They are implementation checks, not benchmark results.

### Concrete Qwen3-Coder/A100 reference settings

[CONFIGURATION.md](CONFIGURATION.md) documents the complete reference and pilot
profiles, six model roles, pinned artifacts, Appendix G table mapping and Eq. (38)
safety accounting. These profiles specify new reproducibility choices; the full
four-benchmark evolution/baseline orchestration is still incomplete.

### Original paper-protocol scaffold

`configs/paper_template.json` lists the construction and execution comparisons
for MBPP, HotPotQA, MATH, and GAIA. It is a protocol scaffold, not a formal
benchmark runner. It is deliberately marked `"runner": "protocol_template"`
and `"runnable": false`; unresolved values are `null` and listed in
`required`.

Validate the template structure:

```bash
python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/paper_template.json
```

This command checks the template's structure while permitting its documented
placeholders. `--for-run` deliberately rejects `protocol_template`. A formal
runner must first implement and register the benchmark loaders, baseline
methods, evolution loop, token accounting, and replay semantics; only its own
resolved configuration should be marked runnable.

The formal paired protocol to implement is:

1. use one backbone model, tool schema set, task split, and rollout budget for
   every construction method;
2. compare AFlow, EvoAgentX, the base planner, expert-only policies, a frozen
   static library, and the evolving CompactFlow library;
3. execute construction outputs with the same runtime;
4. replay identical call outputs where possible when comparing sequential,
   independent-branch, complete-dependency, percentage-threshold, and exact
   guarded scheduling;
5. pair records by `(benchmark, task_id, seed)`;
6. report native quality, total input/output tokens, nodes, edges, critical
   path, latency, TTFO, opportunity coverage, readiness-gap percentiles,
   early-dispatch rate, and all four violation categories; and
7. freeze the admitted policy library before target-split evaluation.

Local task adapters and evaluators now exist for MBPP, HotpotQA, MATH and GAIA.
The GAIA adapter does not provide its missing attachment/browser tool environment.

The current CLI implements only `offline_smoke`. It does not implement AFlow
or EvoAgentX baseline orchestration, online source/validation/target policy
evolution or the complete baseline comparison. Model token collection, strict
replay and internal independent/percentage scheduling controls are implemented
as components; external baseline adapters remain separate work.

### Reproduction boundary

Some supplied figure captions say illustrative placeholders, and the supplied
Appendix G does not resolve the backbone model, exact dataset splits, sample counts,
rollout budget, seeds, \(K_0\), \(K\), \(k\), score weights, admission
thresholds, resource capacities, replay policy, or materialization settings.
Consequently:

- this implementation does not copy the numerical values from its tables;
- the paper template refuses to run while those fields are unresolved; and
- no generated artifact should be described as a paper reproduction until
  those values and the missing benchmark integration are supplied.

## Current boundaries

- The execution graph is a finite static DAG.
- Each logical call has one attempt. Retry-aware input-version invalidation and
  compensating side effects are not yet implemented.
- There is no scheduler-level per-call timeout. Apply provider/tool timeouts
  inside native async operations.
- Dynamic per-item call instantiation is not yet implemented; finite item
  fields can still be exposed explicitly and guarded independently.
- Native agent-backed nodes produce a complete result. To overlap their
  outputs, provide an explicitly instrumented streaming operation and
  `StreamContract`.
- Synchronous operations run in worker threads and cannot be force-cancelled
  safely. They never receive partial-data or partial-effect early dispatch;
  use a native async operation when cancellation is part of the contract.
- A stream field must be declared stable by contract and by event. Raw token
  progress or a percentage threshold is never treated as semantic readiness.
- `ttfo` is the first output observed anywhere in the runtime. Use the trace to
  compute sink-specific first-useful-output metrics when required.
- The supplied four-benchmark file is a non-runnable protocol scaffold; the
  only executable experiment in this directory is the offline smoke.

## Validation

Run the core tests and style checks from the repository root:

```bash
python -m pytest -q tests/src/compactflow
ruff check evoagentx/compactflow examples/compactflow/run_experiments.py \
  tests/src/compactflow
```

With only `.[dev]`, the native planner and adapter test modules are skipped
when EvoAgentX optional integrations are unavailable. Install `.[all,dev]` and
rerun the same command to execute those integration tests as well.

The tests cover policy persistence and retrieval, compatibility conflicts,
Pareto selection, paired admission, schema/path validation, safe and unsafe
streaming cases, runtime value validation, failure propagation,
effect/resource constraints, at-most-once dispatch, native workflow lowering,
paired metrics, strict config validation, stable splits, and manifest
sanitization.

## Artifact privacy

`build_anonymous_manifest` never reads the current account name, home
directory, working directory, host name, process environment, or credentials.
It removes identity- and secret-like keys, redacts absolute paths and common
credential formats, and rejects a manifest that still contains forbidden
values. Keep task contents and model outputs out of public artifacts unless
they have been reviewed separately; the manifest sanitizer does not claim to
anonymize arbitrary natural-language benchmark data.
