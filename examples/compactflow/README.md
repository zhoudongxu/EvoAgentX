# CompactFlow commands

Run these commands from the repository root.

## Install

```bash
python -m pip install -e ".[dev]"
```

For native integrations:

```bash
python -m pip install -e ".[all,dev]"
```

## Validate configuration

```bash
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json

PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/qwen3_coder_a100.pilot.json

PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/paper_template.json
```

## Offline smoke

```bash
PYTHONPATH=. python examples/compactflow/run_experiments.py smoke \
  --output outputs/compactflow/smoke
```

## Qwen3-Coder service

```bash
export COMPACTFLOW_MODEL_PATH=/path/to/Qwen3-Coder
CUDA_VISIBLE_DEVICES=7 python examples/compactflow/serve_model.py \
  --config examples/compactflow/configs/qwen3_coder_a100.model.json --launch
```

In another shell:

```bash
PYTHONPATH=. python examples/compactflow/serve_model.py \
  --config examples/compactflow/configs/qwen3_coder_a100.model.json --preflight
```

## Benchmark trial

```bash
PYTHONPATH=. python examples/compactflow/prepare_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --source-dir <download-directory> --output <cohort-directory> \
  --count 10 --seed 43

PYTHONPATH=. python examples/compactflow/run_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --cohort <cohort-directory> --output <results-directory>
```

## Cross-task policy evolution

Supply a local raw bundle with sources.json and pinned files in the same format
as the benchmark trial. The runner never implicitly downloads data or encoders.
PyArrow, sentence-transformers, a cached pinned encoder, and the pinned MBPP
Docker sandbox are required for the corresponding live workloads.

```bash
PYTHONPATH=. python examples/compactflow/run_evolution.py validate \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --data-dir <raw-data-bundle>

PYTHONPATH=. python examples/compactflow/run_evolution.py run \
  --config examples/compactflow/configs/qwen3_coder_a100.pilot.json \
  --data-dir <raw-data-bundle> --output outputs/compactflow/evolution-pilot \
  --resume
```

Use the reference profile and a new output directory for 90/30/30 tasks per
benchmark. Shared family IDs, normalized prompts, numeric templates and supplied
leakage keys form indivisible connected groups. Five validation folds of six
tasks are packed jointly with the splits. Infeasible cohorts are incomplete.

Budgets are per benchmark: bootstrap on 20 source tasks, then five rounds of ten
new source tasks, at most one proposal per task and four validated candidates per
round. The remaining 20 source tasks are reserved. All batch proposals are
distilled before viewing the round's validation results. Each candidate uses
seeds 101/202/303 and is paired with the current verified library without that
candidate. Multiple candidates share the round's fold; disjoint rounds do not
prove absence of selection overfitting or arbitrary semantic/entity leakage.

The target library is frozen before paired no-policy/CompactFlow evaluation.
Artifacts include locks, the task manifest, initial/bootstrap/round/final
snapshots, execution records, model usage, candidate decisions, coverage,
summary and CSV tables. Resume requires the same config, raw bundle, task
content, code (including uncommitted edits), and initial library. Completed
calls and decisions are reused. An interrupted remote request without a durable
response may run again.

Missing data, unsupported GAIA tools/attachments and unavailable baselines are
explicit in coverage. No surrogate results are generated. Exit code 2 means
incomplete publication coverage. These are newly specified implementation
settings; Appendix G Table 12 in the supplied paper leaves the exact rules TBD.

## Tests and lint

```bash
python -m pytest -q tests/src/compactflow
ruff check evoagentx/compactflow \
  examples/compactflow/run_experiments.py \
  tests/src/compactflow
```

## Documents

- [Configuration](CONFIGURATION.md)
- [Appendix G table](APPENDIX_G_CONFIGURATION.md)
- [Benchmark trial](BENCHMARK_TRIAL.md)


## Unified construction baselines

The live entrypoint is examples/compactflow/run_baselines.py. It accepts run or
preflight and requires --config, --data-dir, --evolution-dir and --output.
Default methods are aflow,evoagentx,base_planner,expert_policies,static_library,compactflow.
Use --benchmarks MATH for a single-benchmark pilot and --resume to reuse durable
records. Preflight performs dependency/data/model checks without running search.
The frozen evolution manifest and protocol must match exactly; prepare a separate
reference evolution run before using the 90/30/30 reference profile.

AFlow and SEW call the repository's real optimizers. Search uses source only,
selection uses validation, and target runs through the same complete-dependency
runtime after selection is frozen. The adapter exports bounded static AFlow DAGs
and native SEW workflows. Unsupported exports or dependencies remain explicit in
coverage; no surrogate results enter the table. Native integration tests use a
fake ModelClient and the real optimizers; they are not Qwen benchmark results.
See CONFIGURATION.md for the exact operator boundary, budgets, seeds and metrics.

Outputs include baseline_records.jsonl, baseline_summary.json, coverage.json,
summary.json and tables.csv. The summary includes the earlier evolution summary
without modifying its artifacts. 
