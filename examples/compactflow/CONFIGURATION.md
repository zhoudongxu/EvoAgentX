# Qwen3-Coder / A100 reference configuration

The concrete configuration is `configs/qwen3_coder_a100.reference.json`.
It resolves the supplied paper's Appendix G Tables 9–25 into explicit reference
choices. The supplied PDF leaves the original values as TBD; these settings
do not recover the authors' original experimental setup or reproduce its numbers.
The PDF identity and SHA-256 are embedded in each profile.

`qwen3_coder_a100.pilot.json` reduces dataset and search sizes for integration.
`qwen3_coder_a100.model.json` exports the identical model/server settings.
The validator checks that this export matches the reference profile. All six
model roles share the same pinned Qwen backbone, tokenizer and endpoint.

## Model and deployment

| Setting | Reference value |
|---|---|
| Backbone | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| Revision | b2cff646eb4bb1d68355c01b18ae02e7cf42d120 |
| Precision | BF16; no quantization |
| Hardware | One A100 80 GB; tensor/pipeline parallelism 1/1 |
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
| Retries | at most 2 before any output; backoff 1, 2 s, ceiling 8 s |
| Runtime failures | call timeout 180 s; workflow timeout 600 s; no logical-call retry |
| Token budget | 65,536 per task; 2,000,000 offline search tokens per benchmark/seed |

The model's upstream recommendation supplies top-p, top-k, repetition penalty
and executor temperature. Role-specific temperatures and output limits are
reference choices. Qwen3-Coder is used in its non-thinking mode. All actual token
costs require provider usage; missing usage makes a run non-reportable. The client
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
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.reference.json
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.pilot.json
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config --config examples/compactflow/configs/qwen3_coder_a100.model.json
# Set COMPACTFLOW_MODEL_PATH to the local model directory and select an available GPU.
CUDA_VISIBLE_DEVICES=7 python examples/compactflow/serve_model.py --launch
python examples/compactflow/serve_model.py --preflight
```

Local model config and tokenizer-config hashes are checked at launch. This check
does not claim all weight shards have been rehashed. Hub loading uses the exact
model revision. The service binds loopback. Use an SSH tunnel for remote clients;
credentials come from environment variables and must not enter experiment logs.

## Dataset, split and repetition protocol

The JSON locks immutable dataset revisions for MBPP sanitized/test, HotpotQA
distractor/validation, MATH test across all seven subjects, and GAIA 2023_all/
validation. The reference samples 150 tasks per benchmark, then assigns disjoint
family groups to nominal 90 source / 30 validation / 30 target tasks. Sampling
seed is 43; partition seed is 42. Pilot uses 15 tasks with nominal 9/3/3 splits.
Actual counts must be frozen in a manifest before inference. Never split a family
to reach a nominal count; insufficient cohorts must stop the requested study.

Families, duplicate normalized prompts and supplied entity/template leakage keys
are unioned before splitting. A numeric-template fallback is specified, with the
explicit limitation that it cannot prove semantic/entity disjointness. Target
labels are evaluator-only; policy discovery, verification and statistic updates
must finish before the target library is frozen.

Main generation seeds are 42, 43 and 44. Runtime replay uses one warmup and five
timed repetitions, with ordering seeds 31415–31419. Confidence intervals use
2,000 paired family bootstrap resamples, seed 2718 and 95% coverage. Pilot has
one generation seed, no warmup and two timed repetitions.

Separate allocations make the paper's workflow counts checkable: construction
diagnosis has 4 × 50 × 6 = 1,200 base workflows; execution diagnosis has
4 × 25 × 10 = 1,000 fixed workflows, yielding 25,000 timed replays over five
schedulers. Replays and intervention executions are not new generated workflows.
The requested main construction table has 4 × 30 × 3 × 7 = 2,520 workflows,
subject to baseline availability; it is not an assertion that those runs exist.

## Construction and execution choices

Semantic retrieval uses normalized all-MiniLM-L6-v2 embeddings, 384 dimensions,
CPU, exact cosine search and a pinned revision. K0/K/k = 20/5/3. Retrieval weights
are semantic 0.55, structural 0.25, utility 0.10 and confidence 0.10. Minimum
semantic/applicability/selection scores are 0.15/1.0/0.25. This includes the
confidence term in Algorithm 3 explicitly.

Admission requires zero allowed quality loss, at least 5% weighted cost reduction
and 100% contract-valid candidate evaluations. Cost weights are tokens 0.50,
latency 0.25 and graph 0.25; graph cost is nodes + edges. Verification uses
10 validation tasks × seeds 101/202/303. Verify before merging; cosine merge
threshold is 0.90. Normalization floor is 1e-9; other-cost tolerance is 0.05.

Bootstrap uses 20 source tasks, followed by five rounds of 10 source tasks,
at most four candidate policies per round and one per task. Library capacity
is 256, with deterministic eviction and retained negative evidence. The planner
has at most 12 nodes, two repair attempts and a bounded fallback to the base
planner. The JSON specifies update formulas, compatibility, snapshots and tie
breaks. These library/statistic settings define the intended orchestration;
the complete evolution runner is still an implementation task.

Execution compares five explicitly named internal controls: sequential,
independent topological levels, complete dependency, percentage threshold and
guarded field readiness. These internal controls are not labeled as external
LLMCompiler/LLMOrch implementations. Main capacity/batch/percentage values are
4/1/0.5. Capacity and batch sweeps use 1,2,4,8,16; percentage thresholds use
0.25,0.5,0.75. Reference calls are pure, with static bounded graphs, explicit
input footprints and immutable complete JSON string fields. Raw token prefixes
do not count as stable fields. Retry, compensation and dynamic graph mutation
are disabled in this profile.

Fixed-graph execution comparisons require full argument-keyed call replay with
recorded event timing, identical workflow/model outputs and no live fallback
on a replay miss. E2E ends when all declared sinks complete; TTFO is the first
declared sink output. Internal-event timing and runtime drain are separate.
Speedup is the ratio of paired mean latencies. Construction/search costs remain
separate from replay timing and are also reported with explicit amortization.

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

The full source/validation/target evolution runner, study tables, and upstream
baseline adapters remain to be completed. The pinned A2FLOW source currently
contains a README without implementation; proposed matching budgets are labeled
reference choices. GAIA requires authorized dataset access and its full tool/
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
