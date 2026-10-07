# Qwen3-Coder / A100 reference configuration


## Model and deployment

| Setting | Reference value |
|---|---|
| Backbone | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| Revision | b2cff646eb4bb1d68355c01b18ae02e7cf42d120 |
| Precision | BF16; no quantization |
| Hardware | Active profiles: main model on GPU 7, one A100 80 GB, TP/PP 1/1; GAIA vision/audio on CPU, FP32, eight threads, shared auxiliary concurrency one |
| Context limit | 32,768 tokens |
| Sampling | top-p 0.8, top-k 20, repetition penalty 1.05 |
| Query role | temperature 0.2; max output 512 |
| Selector role | temperature 0.2; max output 1,024 |
| Planner role | temperature 0.7; max output 4,096 |
| Distiller role | temperature 0.2; max output 2,048 |
| Checker role | temperature 0; max output 1,024 |
| Executor role | temperature 0.7; max output 4,096 |
| Serving limits | max sequences 4; batched tokens 8,192; GPU memory fraction 0.82 |
| Serving behavior | eager; prefix caching disabled; chunked prefill enabled; seed 42 |
| Client limits | concurrency 4; connect timeout 10 s; response timeout 180 s |
| Retries | Current ModelClient: one HTTP attempt; no automatic retry after transport starts, including before first output. Legacy max_retries/backoff fields are inactive |
| Runtime failures | call timeout 180 s; workflow timeout 600 s; no logical-call retry |
| Token budget | 65,536 per task; 2,000,000 offline search tokens per benchmark/seed |

The model's upstream recommendation supplies top-p, top-k, repetition penalty
and executor temperature. Role-specific temperatures and output limits are
reference choices. Qwen3-Coder is used in its non-thinking mode. All actual token
costs require provider usage; missing usage marks affected coverage incomplete. The client
uses a conservative UTF-8 byte bound for request admission and rejects oversized
requests rather than truncating them. It does not claim this bound is an exact
tokenizer count. Seeds are recorded; bit-identical GPU generation is not promised.

The observed serving environment is pinned in `serving.observed_runtime`, with
vLLM `0.26.1rc1.dev416+g2dfb8ba59`. This is an existing development build, not a
claim that a corresponding PyPI wheel exists. The launcher prepends the serving
Python's bin directory to PATH, which is needed to find Ninja during FlashInfer
kernel compilation. Four sequences are an upper scheduler limit; four simultaneous
full-length contexts may require KV-cache preemption on one A100 at this memory
fraction. Record server logs for latency measurements.

From the repository root:

```bash
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.pilot.json
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.latency.reference.json
python examples/compactflow/serve_model.py --preflight
```

Local model config and tokenizer-config hashes are checked at launch. This check
does not claim all weight shards have been rehashed. Hub loading uses the exact
model revision. The service binds loopback. Use an SSH tunnel for remote clients;
credentials come from environment variables and must not enter experiment logs.

## Task-family, transfer and library protocol (W3)

### Data allocation and leakage checks

The pinned pools are MBPP sanitized/test, HotpotQA distractor/validation, MATH
test across all seven subjects, and GAIA 2023_all/validation with Level 1–3.
Source/validation/target are internal experimental partitions, not provider
train/validation/test splits. Raw input uses `sources.json` and checksum-pinned
local files; the runner does not silently download or substitute data.

| Per benchmark | Reference | Source-only pilot |
|---|---|---|
| Selected pool | 150 | 15 drawn only from frozen formal source |
| Source / validation / target | 90 / 30 / 30 | 9 / 3 / 3 |
| CompactFlow validation folds | Five disjoint folds of six | One fold of three |
| Bootstrap / source per round / rounds | 20 / 10 / 5 | 2 / 2 / 1 |
| Maximum candidates per source task / benchmark-round | 1 / 4 | 1 / 1 |
| Source generation seed | 42 | 42 |
| Held-out verification seeds | 101, 202, 303 | 101 |
| Target generation seeds | 42, 43, 44 | 42 |

Sampling seed is 43; partition seed is 42. Whole union groups must fit exact
partition and fold sizes. Infeasibility marks a benchmark incomplete; families
are never split to reach a count. Reference evolution consumes 70 of the 90
source tasks per benchmark; its remaining 20 are reserved/unused by evolution.
Other requested source studies/searches have their separately locked allocation.
Pilot task identities and whole families come exclusively from formal source,
so pilot work does not expose formal validation or target.

`benchmarks.py` defines the deterministic family/group procedure:

1. `normalize_qa` lowercases the question, deletes ASCII punctuation, removes the
   standalone articles `a`, `an`, `the`, and collapses whitespace.
2. Numeric regex matches in that normalized text are replaced by `NUM`.
   A nonempty dataset `family_id` takes precedence; otherwise SHA-256 of this
   numeric template is the fallback family (`normalized_numeric_template_v1`).
3. Union-find joins records sharing any family ID, normalized-question hash,
   numeric-template hash or explicitly supplied `leakage_keys`. The transitive
   union group, not just the original family label, is indivisible.
4. This union is scoped by benchmark. It prevents recorded-key overlap across
   splits/folds within that benchmark; it does not claim cross-benchmark
   deduplication or detect arbitrary paraphrases/shared entities without keys.

`sample_manifest.jsonl` records benchmark, task ID, family ID/method, partition
group ID/keys, split, validation fold, leakage keys, revision and task digest.
GAIA also records level and task assets; `gaia_assets.lock.json` freezes attachment
hashes. These are generated/frozen before inference and checked on resume.
Exact IDs belong in the run manifest, not a second hand-maintained Markdown list.
Planner/tool public inputs exclude gold answers, hidden tests and answer metadata.

### Initial library and shared final library

`policies/seed_policies.json` contains six developer-written templates:
linear-chain-fusion, overlapping-role-fusion, unused-output-pruning,
redundant-edge-simplification, stable-consensus-stopping and routing-simplification.
Its SHA-256 is `c3e7c770a7d423ef96a318f79d6b6e2d5ab552a405c069e7c39c661cad560abc`.
All start as `candidate`; bootstrap evidence does not automatically make them
verified. They are explicit initial priors, not policies inferred from target
answers or already validated by their presence in the file.

There is **one shared library per evolution run/variant**, across the included
benchmarks. Bootstrap iterates benchmarks in sorted name order; each subsequent
round likewise processes benchmark source batches in sorted order. Candidate
metadata retains its source benchmark/task/round. Retrieval has no benchmark-only
library filter. Only verified policies remain in the ordinary final snapshot;
an empty verified library is a valid outcome. The without-heldout-verification
ablation is an explicitly separate experimental library.

Source runs use seed 42. Verification uses 101/202/303. Target workflow generation
uses 42/43/44 **against the same frozen final library**; these seeds do not train
three independent CompactFlow libraries. Baseline construction searches have
their own method/benchmark/generation-seed selections. Do not describe CompactFlow
snapshots as per-benchmark or independently evolved per-seed final libraries.

### Validation and target isolation

Source eligibility compares the current source library with a paired no-policy
planner. Candidate admission compares the current **verified** library against
that same library plus the candidate, on matching benchmark/split/task/seed keys.
Every candidate, including a near duplicate, passes schema/contract/safety checks
and held-out verification before Admit/Merge; proximity never bypasses verification.

Each benchmark-round uses its designated disjoint validation fold. Candidates in
the same round reuse the six tasks and three seeds; disjoint rounds reduce reuse
but do not prove absence of adaptive validation overfitting. AFlow/SEW construction
selection instead reuses the full 30-task validation pool and held-out seeds; it
does not use CompactFlow evolution folds. Selection-overfitting risk is retained
in the interpretation of both protocols.

The paper runner completes discovery/selection for all requested variants, freezes
their libraries and baseline workflows, then opens the target gate. Blocked
benchmarks/methods remain incomplete. Target cannot update retrieval statistics,
utility/confidence, distillation, verification, merge or eviction. Frozen artifacts
are hash-checked; changes require a new run directory.

### Artifacts required to disclose W3

The following paths are relative to a `run_paper.py` output directory. A standalone
evolution run stores its snapshots and records directly under its own root.

| W3 question | Configuration/source | Required run artifact |
|---|---|---|
| Family definition and leakage checks | Appendix G Table 12; benchmarks.py grouping | sample_manifest.jsonl, including partition keys/group IDs |
| Exact allocation | partition; runner validation/source settings | sample_manifest.jsonl; config.lock.json; experiment_manifest.json |
| Initial library | Table 17; policies/seed_policies.json | evolution/base/policy_snapshots/initial.json |
| Library evolution and evidence | runner; verify-before-merge admission | bootstrap.json, round-01.json … round-05.json under evolution/base/policy_snapshots/; source_records.jsonl, validation_records.jsonl and candidate_decisions.jsonl under evolution/base/ |
| Library used by each target seed | One shared library; target seeds 42/43/44 | evolution/base/policy_snapshots/final.json; target_gate.lock.json (artifact hashes); construction/freeze_summary.json; construction/baseline_records.jsonl |
| Expert / Static controls | Initial / bootstrap snapshot respectively | initial.json / bootstrap.json; there are no separate expert.json or static.json files |

Other evolution variants have their own `evolution/<variant>/` directories (or
explicit snapshot aliases). `final.json` is created only when that discovery run
finishes, not merely because the configured path exists. Release actual hashes
with the records; a partial source run cannot provide final-library evidence yet.

This protocol evaluates held-out cross-task reuse within the included benchmark
suite. All included benchmarks supply source/validation data to the shared
library, and union groups are benchmark-scoped. Therefore it does not establish
transfer to an unseen benchmark, domain or tool ecosystem. Such a claim requires
learning elsewhere and freezing before access to that held-out domain.

## Experiment selection and effective scale

| Selected experiment | Formal allocation | Four-benchmark pilot |
|---|---|---|
| Construction main | 4 x 30 target x 3 seeds x 6 methods = 2,160 records | 4 x 3 target x seed 42 x 6 methods = 72 records |
| Construction diagnostic | 4 x 50 source x 6 seeds = 1,200 base workflows; at most 12 interventions each | Three source tasks per benchmark, seed 42, at most four interventions each |
| Independent execution main | 4 x 25 target x 10 seeds = 1,000 shared workflows; 3,000 warmups and 15,000 timed replays | Eight shared workflows; 24 warmups and 48 timed replays |

The six default construction methods are AFlow, EvoAgentX, Base Planner, Expert
Policies, Static Library and CompactFlow, all using the same complete-dependency
runtime for this comparison. `baseline_runner.required_methods` selects them.
A2Flow is an optional local reimplementation; adding it explicitly in a new run
would make seven methods. The inherited `study_allocation.construction_main`
requested seven-method/2,520-workflow description is not the effective six-method
allocation used by `paper_studies.allocation`.

The independent execution main flow fixes LLMOrch, LLMCompiler and CompactFlow;
it uses one Base Planner graph without policies for each task/seed. The inherited
five-scheduler/25,000-replay diagnostic description does not override this flow.
Mixed internal execution diagnostics retain their actual frozen list and should
not be relabeled as the independent main comparison.

Construction diagnostic seeds are 11/22/33/44/55/66; execution seeds are
1011–1020. Formal replay uses one warmup and five timed repetitions with ordering
seeds 31415–31419. Pilot uses one warmup and two timed repetitions. Confidence
intervals use 2,000 paired task-family bootstrap resamples, seed 2718, 95% coverage.
Insufficient samples or a single-seed topology comparison produce null estimates.
Replay and intervention executions do not count as new base workflow generations.

## Construction and execution choices

The reference specifies sentence-transformers/all-MiniLM-L6-v2 for semantic
retrieval, with normalized 384-dimensional embeddings, CPU execution, exact
cosine search and a pinned revision. K0/K/k = 20/5/3. Retrieval weights
are semantic 0.55, structural 0.25, utility 0.10 and confidence 0.10. Minimum
semantic/applicability/selection scores are 0.15/1.0/0.25. This includes the
confidence term in Algorithm 3 explicitly.

Admission requires zero allowed quality loss, at least 5% weighted cost reduction
and 100% contract-valid candidate evaluations. Cost weights are tokens 0.50,
latency 0.25 and graph 0.25; graph cost is nodes + edges. Verification uses
6 validation tasks × seeds 101/202/303 (18 pairs). Each round uses its own disjoint fold within each benchmark. Verify before merging; cosine merge
threshold is 0.90. Normalization floor is 1e-9; other-cost tolerance is 0.05.

Per benchmark, bootstrap uses 20 source tasks, followed by five rounds of 10
source tasks, at most four candidate policies per round and one per task. Library capacity
is 256, with deterministic eviction and retained negative evidence. The planner
has at most 12 nodes, two repair attempts and a bounded fallback to the base
planner. The JSON specifies update formulas, compatibility, snapshots and tie
breaks. The runner persists these snapshots and paired evidence; see the W3
section for the shared library, validation and seed boundaries.

Internal execution controls remain separately available: sequential, independent
topological levels, complete dependency, percentage threshold and guarded field
readiness. Main capacity/batch/percentage values are 4/1/0.5. Capacity and batch
sweeps use 1,2,4,8,16; percentage thresholds use 0.25,0.5,0.75. These controls
are distinct from the LLMCompiler native scheduler and LLMOrch core reproduction.

Footprints are deterministically compiled from schema-checked planner bindings;
output names are planner-declared. Developers register stream/effect contracts
and early-safe rules. LLM fields stabilize on full single-assignment emission;
GAIA primitive fields stabilize on complete independent observations, while the
adaptive agent's final answer remains complete-only. GAIA tool effects include
network, asset and auxiliary operations; they are not all pure. No per-task manual
or answer-oracle labels are used. Semantic annotation accuracy is unmeasured;
schema/runtime checks do not establish it. Scheduler retry, compensation and
dynamic graph mutation are disabled in the reference profile.

Fixed-graph execution comparisons require full argument-keyed call replay with
recorded event timing, identical workflow/model outputs and no live fallback
on a replay miss. E2E ends when all declared sinks complete; TTFO is the first
declared sink output. Internal-event timing and runtime drain are separate.
Speedup is the ratio of paired mean latencies: the independent latency report
uses LLMOrch as denominator; the mixed internal report uses complete_dependency.
Construction/search costs remain separate from replay timing. Target online
tokens do not include offline library/search cost automatically: use complete
request journals and an explicit target denominator for amortized accounting.
Rejected candidates and failed attempts with known usage remain in those costs.

## Safety reporting and paper inconsistency

The supplied PDF's introduction says 0.0% observed contract violations;
Table 5 says 0.63% and ΔQ = −0.1. The abstract has no 0.0% statement.
Table 6 reports three zero components but omits duplicate dispatch. This
configuration does not assume which number is correct or hard-code either value.

`metrics.contract_safety` follows Appendix B.2 Eq. (38): count argument mismatch,
duplicate dispatch, effect-order violation and capacity overload over the same
early-call cohort, assigning each violating logical call one primary category.
The explicit reference tie order is argument, duplicate, effect, capacity.
Report the four raw counts, their sum, early-call denominator and unassessed
count. No early calls or incomplete audits produce a null rate, not a zero claim.
Pool counts and denominators before computing the aggregate. The earlier
all-dispatch incident rate remains a separately named diagnostic.

`evoagentx.compactflow.safety.audit_execution` supplies this audit for recorded
execution traces. Failures without comparable reference/final arguments remain
unassessed. A missing audit cannot establish contract preservation. A nonzero
quality difference requires a separate check of replay coverage, failure cases
and evaluator determinism; it is not another contract-failure category.

## Implementation and access boundaries

The configuration validator, typed planner/executor adapter, local benchmark
adapters, strict replay, sink timing, failure invalidation and safety audit have
component tests. A model connectivity/generation check is separate from a
formal benchmark experiment. `validate-config --for-run` deliberately does not
certify the entire study as executable merely because settings are present.

The source/validation/target evolution runner is available in run_evolution.py.
The construction baseline runner now invokes the local AFlowOptimizer and SEWOptimizer
through canonical exports and the common complete-dependency runtime. It also runs
Base Planner, Expert Policies, Static Library, and frozen CompactFlow controls.
Full paper results still require measured, complete benchmark coverage.
`baseline_native.A2FlowAdapter` implements bounded source-derived operator
extraction on EvoAgentX AFlow and canonical export. It is a local reimplementation,
not author code recovered from the recorded upstream repository, and is not
selected in the default six-method experiment. LLMOrch likewise has an explicitly
labeled core scheduler reproduction in `llmorch.py`; the latency profiles enable
it. Native LLMCompiler TaskFetchingUnit is vendored with its upstream lock/license. GAIA requires authorized dataset access and its full tool/
attachment environment. Missing assets or methods must be exposed as incomplete
coverage. Do not present pure text subsets as full GAIA or synthetic smoke runs
as paper results. The tools section specifies service rate limits for the future
full runner; the current model client enforces concurrency and token admission,
not the cross-client requests/tokens-per-minute quotas.

All prompts, schemas and seed policies have checked SHA-256 hashes in each full
profile. The Appendix coverage map connects each of Tables 9–25 to its concrete
configuration sections. Runtime manifests must additionally freeze actual
dataset membership, environment, policy snapshots, generated graphs, requests,
replay bundles, traces and measured results.


## Construction baseline execution contract

Run run_baselines.py with --evolution-dir pointing to a completed evolution run.
Its exact manifest is authoritative: raw rows are reconstructed by frozen IDs and
content digests, without sampling or assigning a new split. The original model,
encoder, dataset, partition, evaluation, construction, execution and tools settings
must match the evolution config.lock.json. Use a new baseline output directory;
the earlier evolution directory remains immutable.

Native search sees public source questions/context and source quality scores only.
The central evaluator owns gold answers and hidden tests. Every canonical candidate
is selected using the entire frozen validation partition and heldout seeds, after
which selection is locked before any method reaches target. This validation pool
is for final construction-workflow selection, not CompactFlow policy evolution;
its reuse across candidates can cause selection overfitting. It does not imply
round-disjoint baseline validation or eliminate adaptive source overfitting.

Reference: AFlow 20 rounds, population 4, one candidate per round, 3 validation
seeds; SEW 20 iterations, one candidate per iteration, 3 validation seeds; both
have max 12 nodes and target generation seeds 42/43/44. Pilot: one search step,
one validation seed (101), one generation seed (42). Source/validation share the
2,000,000-token method/benchmark/generation-seed cap; target has the shared 65,536
per-task cap. Token admission uses conservative request reservations; reports use
only actual provider usage. Missing usage or infrastructure failures prevent
complete coverage. Search, validation and amortized target costs are separate.

AFlow uses the repository's EvoAgentX implementation of AFlow, not a fresh checkout
of the cited MetaGPT revision. Operator names are native: MBPP uses Custom,
CustomCodeGenerate and ScEnsemble; MATH uses Custom, AnswerGenerate and ScEnsemble;
HotpotQA uses Custom, AnswerGenerate and QAScEnsemble. GAIA additionally exposes the statically exportable GAIAToolAgent operator. Static export supports
explicit awaited calls and fixed asyncio.gather; generated Python is parsed but
never executed. Dynamic branches/loops, Test and Programmer are outside this
controlled-runtime adapter and yield rejected-export records, with no invented
graph metrics. Report this restricted operator/search space with the results.
SEW keeps native search and prompt mutation, uses its safe YAML representation,
and lowers SequentialWorkFlowGraph through lower_workflow_graph. The adapter
explicitly evaluates the returned candidate rather than stale optimizer state.

Expert Policies uses the initial snapshot with deterministic applicability and
compatibility selection (no semantic retrieval). Static Library uses bootstrap
with retrieval and no updates. CompactFlow uses verified policies in the final
snapshot. All four controls use the same existing typed planner and compiler.
Policy statistics, merges, and eviction are disabled during construction target
runs. A target lock also prevents restarting unfinished search after target began.

Topology instability is labeled directed graph edit distance with unit node/edge
edits, normalized by total nodes plus edges of the two graphs. A one-second bound
per pair is a declared reproducibility choice. Timed-out pairs and missing graphs
produce null metrics; timeouts are reported explicitly. It is not an ordinal
node-ID comparison or an estimated graph count from generated Python.

Resume checks config, manifest, code and snapshots, then reuses task records and
provider-response journals. Completed native rounds can be replayed locally from
those journals without repeating inference/evaluation. Interrupted requests with
unknown usage remain incomplete rather than being silently retried. Dependency
availability is probed at runtime. Coverage enumerates the selected construction methods and required benchmarks;
implementation availability does not imply their selection or completed results.
Missing data, usage, selected methods or study artifacts prevent complete coverage.
Pilot, selected-study and original-paper publication status remain distinct.

## GAIA extension

[GAIA.md](GAIA.md) specifies the shared six-method tool session, main GPU 7 plus either GPU 0 auxiliaries or the separate CPU/FP32 auxiliary profiles, pinned auxiliary revisions, 12-step / 20-call limits, phase isolation, asset and observation locks, cost accounting, and real pilot acceptance commands. These settings augment the tables above; pilot or tool smoke success is not publication completeness.

## Execution-only latency profiles

The `qwen3_coder_a100.latency.{pilot,reference}.json` profiles use the dedicated
`run_latency.py` entrypoint; an exact `run_paper.py --studies execution_main`
selection dispatches there before construction preflight. Three methods only:
LLMOrch core reproduction, LLMCompiler TaskFetchingUnit, CompactFlow guarded.
Common workflows come from Base Planner once per task/seed, with no policy library.
Search/evolution/retrieval settings inherited in these JSON files are inactive.

| Setting | Pilot | Reference |
|---|---|---|
| Sampling | Whole-family 9/3/3 from formal source only | Frozen formal target |
| Evaluated tasks per benchmark | 2 pilot target | 25 target |
| Seeds | 42 | 1011 through 1020 |
| Warmups / timed repetitions per method | 1 / 2 | 1 / 5 |
| Total workflows, four benchmarks | 8 | 1,000 |
| Timed / warmup records | 48 / 24 | 15,000 / 3,000 |

GPU 7 serves the pinned Qwen3-Coder-30B BF16 model; GAIA vision/speech retain the
CPU FP32 services (8 threads, concurrency 1). Call concurrency remains 4 within
each workflow. Different timing repetitions are not run concurrently. Original
call timing is preserved, so full real-time replay can still take days. Planner,
collection, model/tool costs are reported separately from replay latency. See
[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md#table-5-only-three-execution-schedulers)
for start, resume, capture and report commands.

Typed wire constraints are enabled in the latency profiles:
`model.typed_output_constraints=true`. vLLM receives the existing workflow JSON
schema for planning and an NDJSON grammar derived from each node's declared
output fields for execution. This constrains syntax, field identity, and string
types without supplying an answer. Complete fields still arrive as real streaming
events. Sampling settings, budgets, timeouts and the guarded contract are unchanged;
unsupported structured-output backends fail rather than silently relaxing checks.
Decoder schemas omit unsupported `uniqueItems`; the full original schema checks
uniqueness before execution. Binding guidance supplies actual public field names
and producer.field examples. A single-output sink can be canonically named
`answer`, with that alias recorded; ambiguous multi-output sinks still fail.
Raw model responses and validation/repair traces remain available.

## Execution outcome, infrastructure and cost accounting

The CPU construction profiles enable `execution_accounting_infrastructure_v2`,
typed output constraints and binding guidance. Output formats are explicit:
planner workflow JSON, executor NDJSON, GAIA agent JSON and native optimizer text
are distinct protocols. The same client is shared across methods.

Known-cost format errors, wrong answers and normal budget exhaustion remain failed
samples in the denominator. A request rejected before send records not_sent and
zero additional cost, preserving earlier usage. After a request starts, missing
usage is unknown cost, never zero. Stream cancellation/early closure drains the
existing request for bounded accounting cleanup; it does not start a retry.
Search returning no results is an observable tool failure and can be revised
within the existing agent budget; it is not missing model usage.

Unknown usage or unavailable infrastructure marks affected coverage incomplete;
independent tasks can continue. Such evidence cannot verify a policy. Discovery
health and the freeze manifest identify blocked benchmarks/methods before target.
All costs require actual provider/tool records; token reservations are admission
bounds rather than measured usage. Auxiliary vision tokens, audio duration/time,
and cached logical versus additional physical cost are recorded separately.

Resume validates the existing locks and reuses completed records; changing code,
data, budgets, model identity or frozen observations requires a new run directory.
Report reads recorded artifacts without inference. Configuration Markdown is a
protocol explanation; it must not promote an ongoing source phase, a pilot or a
missing final snapshot to a completed formal result.
