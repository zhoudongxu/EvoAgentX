# Small live benchmark trial

`prepare_benchmark_trial.py` freezes a small development cohort from the immutable
dataset revisions in the reference configuration. Supply downloaded Parquet files
under `<source-dir>/<benchmark>/<repository-relative-path>` and a `sources.json`
list recording `benchmark`, `repository_relative_path`, `url`, `sha256`, and `bytes`.
Preparation requires PyArrow. The runner requires the reference model service,
client dependencies, and the pinned Docker image for MBPP evaluation.

```bash
PYTHONPATH=. python examples/compactflow/prepare_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --source-dir <download-directory> --output <cohort-directory> --count 10 --seed 43
PYTHONPATH=. python examples/compactflow/run_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --cohort <cohort-directory> --output <new-results-directory>
```

This trial uses the base planner with no policy discovery, generation seed 42,
one live complete execution, and one paired complete/guarded replay per task.
Each replay uses the exact recorded argument-keyed calls and original timing.
Tasks run serially; calls within a task retain the configured capacity of four.
Runtime order is shuffled using seed 31415. Latencies are development diagnostics
with no warmup or repeated measurement. This does not establish full-method paper
quality or speedup.

MBPP and HotpotQA IDs are hash-ranked with sampling seed 43. MATH additionally
uses subject round-robin to cover all seven subjects. Sampling never uses model
results or answer correctness. MBPP questions include only required function
headers derived from the reference implementation and test call names. Solution
bodies, hidden assertions, expected outputs, and dataset labels are excluded
from planner and executor inputs. The private task file is evaluator input and
must not be published as a model prompt. GAIA is deferred pending authorized data.

Every attempted task contributes to the quality denominator, including planning
and execution failures. Infrastructure failures remain explicitly labeled.
HotpotQA reports answer F1 and exact match; MATH uses normalized exact match;
MBPP uses sandboxed pass@1. Raw traces, outputs, model usage, task IDs, configuration,
revision, and checksums are retained. Safety reports all four Eq. (38) categories
and the early-logical-call denominator; zero early calls yields an undefined rate.
Separate dispatch diagnostics count actual repeated starts and refused attempts.
