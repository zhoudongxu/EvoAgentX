# Unified CompactFlow paper experiments

This entrypoint implements the selected studies from Tables 1–6 and Figures 2–7
of the supplied CompactFlow manuscript. Figures read actual records only. Pilot
success is separate from formal evaluation and original-paper completeness.

## Scope and phase order

Construction uses AFlow, EvoAgentX (native SEW), Base Planner, Expert Policies,
Static Library, and CompactFlow with one frozen manifest, model, seeds, evaluator,
token budget and execution boundary. AFlow and SEW remain construction optimizers.

All source diagnostics, separate evolution variants, and baseline search and
validation finish before the global target gate opens. The gate hashes the
manifest, variant configurations, policy snapshots and selected workflows.
Every target model/tool call checks it. Frozen changes require a new directory.
The evolution and baseline entrypoints also support --stage freeze and target.

Reference retains 90/30/30 tasks per benchmark and five validation folds of six.
The new pilot chooses 15 whole-family tasks exclusively from formal source, then
splits them 9/3/3. No pilot task touches formal validation/target. Infeasible
whole-family counts fail. Earlier pilot outputs are preserved.

| Study | Reference | GAIA pilot |
|---|---|---|
| Construction main | 4 × 30 target × 3 seeds × 6 methods = 2,160 records | 3 target × seed 42 × 6 methods = 18 records |
| Source diagnostic | 4 × 50 source × 6 seeds; up to 12 interventions | First 3 source tasks; up to 4 interventions |
| Execution base allocation | 4 × 25 target × 10 seeds = 1,000 workflows | First 2 target tasks; seed 42 |
| Five internal controls | 1 warmup + 5 timed each; 25,000 timed records | 1 warmup + 2 timed each; 20 timed records |
| Native LLMCompiler | Additional 5,000 timed records | Additional 4 timed records |
| Construction variants | Every target task | Fixed first target task per point |

Interventions include typed node bypass, edge removal, fusion and early stopping.
Inapplicable operations retain reasons; tasks are never substituted to obtain
more favorable structures. Ablations and one-factor sensitivity variants have
separate discovery and frozen libraries. Exact baseline parameter values may
alias the base run. Unverified-ablation candidates never enter verified storage.
Unsafe execution ablations run only against exact offline replay, never live
models, tools or network. Missing argument-matched recordings fail closed.

## Native execution baseline provenance

LLMCompiler uses SqueezeAILab/LLMCompiler revision
 a00c9d35507507da70e8c637eee64efc8c1857ae.
The vendored TaskFetchingUnit.schedule and Task are invoked directly. Only its
optional global print logger is replaced with a no-op. UPSTREAM.json records
original and adapted SHA-256 checksums, with the original MIT license retained.

This is a fixed-workflow execution comparison: planner and joiner are excluded
because all schedulers execute the same captured workflow and calls. The adapter
binds canonical typed arguments, supplies explicit effect-order edges as
conservative predecessor completion, applies shared resource capacity at the
call boundary, and records common audit events. The native 10 ms scheduling
interval is retained. Report this adaptation explicitly, rather than describing
it as unmodified end-to-end LLMCompiler.

LLMOrch core scheduling is independently reproduced from arXiv:2504.14872v2
Sections III-B/C and Algorithms 1–3. The paper links an author archive at
https://www.hostize.com/v/c3oLTBMUwn (HTTP 404 observed 2026-10-06); this is not
an upstream-source implementation. Records identify it as
`llmorch_paper_reimplementation_v1`, with a source checksum.

The adapter implements def-use ranks, a separate ready-batch queue/coordinator,
I/O-first dispatch, exclusive compute worker slots, and shared capacity bounds.
All input predecessors must complete, including unequal-rank joins, as stated
in Section III-C's prose. Explicit effect dependencies add conservative completion
edges for every external scheduler. The replay models worker slots, not the
paper's C/MPI processes. Remote model requests and bounded GAIA tool agents are
classified as nonblocking I/O; alternative compute classifications must be fixed
in configuration before a run. No CPU-scaling or MPI-overhead claim is made.
Query translation, automatic repair, retries and recovery are outside this fixed
workflow comparison, by design. Missing observations fail; no live fallback.

### Table 5 only: three execution schedulers

`run_latency.py` and `run_paper.py --studies execution_main` now run only
LLMOrch (core reimplementation), native LLMCompiler TaskFetchingUnit, and
CompactFlow guarded scheduling. They do not run evolution, AFlow/SEW search,
retrieval, or construction-method target evaluation. Mixed paper-study selections
retain the full construction-first pipeline.

Every task/seed uses one Base Planner workflow without policies. Its validated
specification is saved before executing it under complete dependencies to collect
real streaming events and exact call inputs/outputs. The execution protocol is
frozen before target access; all captures are frozen before comparison. GAIA
agents retain full-result dependencies. Planning and collection costs are reported
separately; scheduler latency excludes both. A lack of partial-readiness windows
is retained in the sample, not repaired by inventing parallel opportunities.

Start the four-benchmark source-only pilot (8 captures, 48 timed + 24 warmup rows):

```bash
cd /data/Optx/CompactFlow
PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_latency.py run \
  --config examples/compactflow/configs/qwen3_coder_a100.latency.pilot.json \
  --reference-config examples/compactflow/configs/qwen3_coder_a100.latency.reference.json \
  --formal-manifest /data/Optx/outputs/compactflow/gaia-data-preflight-v1/reference.sample_manifest.jsonl \
  --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia \
  --benchmarks MBPP,HotpotQA,MATH,GAIA \
  --output /data/Optx/outputs/compactflow/execution-only-pilot-v5 --resume
```

Formal capture + replay (1,000 captures, 15,000 timed + 3,000 warmup rows):

```bash
cd /data/Optx/CompactFlow
PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_latency.py run \
  --config examples/compactflow/configs/qwen3_coder_a100.latency.reference.json \
  --reference-config examples/compactflow/configs/qwen3_coder_a100.latency.reference.json \
  --formal-manifest /data/Optx/outputs/compactflow/gaia-data-preflight-v1/reference.sample_manifest.jsonl \
  --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia \
  --benchmarks MBPP,HotpotQA,MATH,GAIA \
  --output /data/Optx/outputs/compactflow/execution-only-reference-v1 --resume
```

Use the identical command to resume. `plan` reads configuration and pinned data
without inference; `preflight` additionally checks model/tool availability;
`capture` stops after freezing traces. `report --output <directory>` only reads
artifacts, does not require services, and never runs inference. Pure execution
completion does not make the entire paper publication-complete.

To reuse an existing frozen paper capture, use `run_latency.py run
--captures-from <frozen-directory> --output <new-directory> --resume`. New runs
compare three methods; resuming/reporting historical five-method latency runs
honors their original method lock. Missing exact captures fail closed. Outputs
are `tables/table5_latency.csv`, `tables/execution_records.csv`,
`figures/fig5_execution_latency.{png,pdf}`, and `summary.json`. Raw-data runs keep
the immutable graphs, responses, observations and traces in `capture_run/`.
`progress.json` reports replay progress; `capture_run/progress.json` reports
collection progress. Estimates use observed wall time, not model throughput.

Replay preserves original event delays and measures one scheduler run at a time;
it is offline but not instantaneous. Record/report speedup is T_llmorch/T_method
on exactly paired samples. Unknown usage, missing observations or missing rows
prevent completion. Wrong answers remain in the denominator, and a zero early-call
denominator gives an undefined violation rate. This intentionally selected
three-method study omits Sequential and Percentage threshold from the original
five-row paper Table 5; those historical controls are not silently declared run.

## Models, tools and accounting

Main inference remains GPU 7 Qwen3-Coder-30B. CPU profiles retain pinned vision
and speech revisions, FP32, eight threads, and auxiliary concurrency one. Main
clients share the configured four-call endpoint limit. The 12-step, 20-tool-call,
total token and timeout budgets remain fixed before target access.

Identical tool requests share frozen observations only within the same task,
split, normalized parameters, assets and backend identity. Sessions stay separate.
Model streams preserve live chunk arrival times and provider token usage. Logical
cost and new physical inference cost are recorded separately; speech duration is
separate. Unknown usage or infrastructure failure is incomplete. Wrong answers
and normal bounded exhaustion stay in the denominator. Interrupted calls with
unknown usage are never silently repeated and called free. GAIA agent nodes keep
complete-result dependencies, without invented streaming opportunities.

Control-overhead records use `control_profile_version=exclusive_v1`. Guard,
materialization, and queue/dispatch times exclude nested profiled calls; queue
scanning and sorting are included, while lock waits and model/tool execution
are excluded. Static analysis times the GFRG compiler separately from replay
binding/deserialization. Records retain all four components in seconds;
`tables/execution.csv` exports their means in seconds and milliseconds plus
their sum. The existing `compilation_seconds` retains total graph reconstruction
time; `static_analysis_seconds` and `graph_setup_seconds` split that interval.
Legacy inclusive timers and unprofiled external schedulers are never treated
as zero-cost measurements or mixed into the corrected means. Figure 6 overhead
coverage requires the corrected guarded-run records. Existing frozen results
remain unchanged; new measurements require a new run directory.

## A100 commands

Run from the CompactFlow checkout. Install plotting dependencies explicitly:

    /data/Optx/venvs/compactflow/bin/python -m pip install -r examples/compactflow/requirements-paper.txt

Use the same arguments for plan, preflight and run. Plan is read-only; preflight
checks data, tools, models, encoder and optimizer imports. Start the source-only
pilot as follows:

    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_paper.py run \
      --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.pilot.json \
      --reference-config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json \
      --formal-manifest /data/Optx/outputs/compactflow/gaia-data-preflight-v1/reference.sample_manifest.jsonl \
      --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia \
      --benchmarks GAIA \
      --output /data/Optx/outputs/compactflow/paper-gaia-cpu-pilot-v1

Append --resume to the identical command after an interruption. Finished calls,
rounds, searches and study records are restored. Code, data, configuration,
assets and backend locks must match.

For formal experiments use qwen3_coder_a100.gaia_cpu.reference.json as --config,
--benchmarks MBPP,HotpotQA,MATH,GAIA, and a fresh output such as
/data/Optx/outputs/compactflow/paper-reference-v1. Append --resume to resume.
This documentation does not start the formal run. --studies selects a comma-
separated subset before the experiment directory is locked.

Reports run without model inference:

    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_paper.py report \
      --output /data/Optx/outputs/compactflow/paper-gaia-cpu-pilot-v1

Import human labels by adding --applicability-labels labels.jsonl to report.
Rows require benchmark, task_id, policy_id, Boolean applicable, and annotator.
Imports are immutable. Missing or incomplete annotations remain incomplete.

## Artifacts and status

The output includes paper.lock.json, experiment_manifest.json,
sample_manifest.jsonl, variant_configs, checkpoint.json, target_gate.lock.json,
evolution/<variant>, construction, captures, studies/*/records, tables, figures,
summary.json, coverage.json, and GAIA data/asset/backend/observation locks.
Reports include paired task-family bootstrap, distributions, library evidence,
GAIA levels/tool costs and planner overhead. Single-seed topology instability
and confidence intervals with fewer than two families are not estimable.

implementation_status, pilot_status, selected_run_status, formal_results_status
and publication_status are separate. A pilot never completes formal results or
original-paper coverage while external baselines or manual labels are missing.

Typed wire constraints are enabled in the latency profiles:
`model.typed_output_constraints=true`. vLLM receives the existing workflow JSON
schema for planning and an NDJSON grammar derived from each node's declared
output fields for execution. This constrains syntax, field identity, and string
types without supplying an answer. Complete fields still arrive as real streaming
events. Sampling settings, budgets, timeouts and the guarded contract are unchanged;
unsupported structured-output backends fail rather than silently relaxing checks.
The first real pilot exposed malformed references/field output and is retained as
incomplete in `execution-only-pilot-v1`; the corrected protocol uses a new run.

Decoder schemas omit unsupported `uniqueItems`; the full original schema still
checks uniqueness before execution. The v2 attempt is also retained as incomplete
due to the server schema-compatibility error; v3 uses the compatible decoder schema.

Latency profiles also ground planner input references in the actual public fields
and provide explicit producer.field examples and valid-reference repair feedback.
This changes planner protocol guidance, not scheduler semantics. The v3 attempt
with invalid generated DAGs is preserved; no task was replaced or hand-edited.

The typed planner canonicalizes a sink with exactly one declared output to the
required `answer` port, recording `single_sink_output_alias` in its planning trace.
Raw model responses are retained; values, nodes, and edges are not synthesized.
Ambiguous multi-output sinks still fail validation. The v4 attempt is retained
as incomplete because one MBPP workflow used a single `code` sink port.

## Construction failure protocol v2

The CPU construction profiles enable explicit planner JSON and executor NDJSON
constraints. GAIA agents use JSON tool-call/final objects; native optimizer
text/code requests remain text. Formats are explicit API arguments and form part
of each request cache identity.

Execution failure, token accounting, and infrastructure availability are separate.
Known-cost malformed output, wrong answers, and bounded exhaustion remain failed
observations in the denominator. Streams close through all wrappers before
journaling; late provider usage is retained. Requests rejected before transport
record no new tokens. Budget reservations are not reported as actual usage.
Unknown usage is never converted into successful zero-cost output or silently
retried. Empty search results are feedback for the tool agent, not a missing
model usage flag.

Discovery continues across independent tasks and reports each completed task and
round. Detailed eligibility checks retain the original quality, cost and Pareto
thresholds. A source/candidate/validation infrastructure or usage gap blocks its
benchmark's target phase, with reasons frozen in discovery_health.json. Other
ready benchmarks and selected methods may proceed; missing cells keep the
publication status incomplete. Zero admitted policies is reported as observed.

Old run directories remain immutable. After regression tests, run a fresh
four-benchmark source-only pilot with --studies construction_main,
--config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.pilot.json,
--reference-config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json,
and --output /data/Optx/outputs/compactflow/construction-main-pilot-v4.
Keep the same pinned data and formal-manifest arguments documented above.
Acceptance requires 72 paired target rows across six methods, complete measured
usage and no infrastructure gaps; zero quality or zero admissions do not imply
missing records. The accepted formal run uses the reference config and
/data/Optx/outputs/compactflow/construction-main-reference-v2 with --resume.
It must not import v1 observations or rewrite its lock. Repeat the identical
command for resume; use report for inference-free artifact generation.

Formal launch and resume after pilot acceptance:

```bash
cd /data/Optx/CompactFlow
PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_paper.py run \
  --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json \
  --reference-config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json \
  --formal-manifest /data/Optx/outputs/compactflow/gaia-data-preflight-v1/reference.sample_manifest.jsonl \
  --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia \
  --benchmarks MBPP,HotpotQA,MATH,GAIA \
  --studies construction_main \
  --output /data/Optx/outputs/compactflow/construction-main-reference-v2 \
  --resume
```

For the source-only pilot, change only `--config` to the CPU pilot profile and
`--output` to `construction-main-pilot-v4`; retain the reference configuration
and formal manifest arguments. The source-only split is generated from formal
source tasks. Do not run a second writer against a directory already in use.

Artifact-only reporting:

```bash
PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_paper.py report \
  --output /data/Optx/outputs/compactflow/construction-main-reference-v2
```

Pilots v2 and v3 were stopped during transport and search-budget audits and
remain preserved. Pilot v4 validates the final protocol: no automatic HTTP retry after transport
starts without measured usage, including timeouts before the first output.
Planner/schema repair counts and experiment budgets are unchanged.

Exhausting the shared search budget records a known-cost refusal before sending.
Already produced candidates still undergo the original validation rules. A native
optimizer with unknown request usage cannot produce an eligible frozen candidate,
even if the native optimizer catches its own generation errors.
