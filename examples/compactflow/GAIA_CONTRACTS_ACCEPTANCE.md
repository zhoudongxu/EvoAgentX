# GAIA observation-unit contract acceptance

Contract: `gaia_observation_units_v1`. This change was made while experiments and
the automatic continuation heartbeat were paused. No formal experiment was resumed.

## Evidence on A100

Audit root: `/data/Optx/outputs/compactflow/gaia-contract-implementation`.

- Full CompactFlow regression: 263 passed (`tests-final3.log`). Subsequent focused
  contract/config tests: 43 passed (`tests-final-focused.log`), including actual
  LLMOrch/LLMCompiler replay adapters. Deployment verification is recorded separately.
- Successful real-service check: `smoke-v5/summary.json`, status complete.
- Main model: existing GPU-7 Qwen3-Coder-30B endpoint; auxiliary: existing CPU service.
- Synthetic source-scoped CSV, red image and silent audio; no benchmark
  validation/target task was accessed. The planner emitted a valid primitive
  read-file workflow without repairs. A separate fixed typed workflow exercised
  the two independent row observations and returned `red,blue`.
- Actual guarded trace: read-file started before its argument-builder completed;
  the first-row consumer started before the reader's second unit completed.
  No artificial delay or fabricated stable field was introduced. These are
  functional observations, not benchmark speedup estimates.
- LLMOrch core reimplementation, upstream LLMCompiler scheduler and CompactFlow
  replayed identical recorded calls at their original time scale, with matching
  final outputs. No live target survived conversion to the replay graph.
- Aggregate ModelSession usage: 4,983 tokens, including 292 measured vision tokens.
  Audio: 0.5 seconds processed. Logical/physical costs are separately journaled;
  all measured usage was complete.

Files include `planner_workflow.json`, `workflow.json`, `graph.json`,
`live_trace.json`, `replay.json`, `contract_records.jsonl`, `tool_records.jsonl`,
model request journals, tool snapshots and `summary.json`. Earlier development
attempts smoke-v1 through smoke-v4 are retained with their failures and incurred
request costs; they are not counted as successful experiments. They exposed
planner binding/unit-format errors and smoke-driver scheduler invocation errors,
which were corrected before smoke-v5.

## Scope

Early eligibility is developer-specified per tool; footprints come from explicit
validated input bindings. Semantic annotation accuracy has not been measured.
Independent observation units are complete bounded requests. The adaptive agent's
final answer, uncompleted visual generation and evolving ASR prefixes remain
completion-only. Conservative task-asset effect barriers may prevent overlap.

These checks establish implementation behavior and service connectivity, not
full GAIA accuracy or completion of the paper's experiments. Old frozen results
are unchanged. A future experiment must use a new output directory and the new
contract/code identity; restarting the old paused run under changed code is invalid.
