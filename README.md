# CompactFlow

Command entry point for the CompactFlow repository.

## Install

```bash
python -m pip install -e ".[dev]"
```

For native EvoAgentX integrations:

```bash
python -m pip install -e ".[all,dev]"
```

Run the following commands from the repository root.

## Validate configuration

```bash
PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json

PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/qwen3_coder_a100.pilot.json

PYTHONPATH=. python examples/compactflow/run_experiments.py validate-config \
  --config examples/compactflow/configs/paper_template.json
```

## Run offline smoke

```bash
PYTHONPATH=. python examples/compactflow/run_experiments.py smoke \
  --output outputs/compactflow/smoke
```

## Launch the Qwen3-Coder service

Set `COMPACTFLOW_MODEL_PATH` to a local model directory when using local
weights. The default configuration targets GPU 7 on the A100 setup.

```bash
export COMPACTFLOW_MODEL_PATH=/path/to/Qwen3-Coder
CUDA_VISIBLE_DEVICES=7 python examples/compactflow/serve_model.py \
  --config examples/compactflow/configs/qwen3_coder_a100.model.json --launch
```

In another shell, run the service preflight:

```bash
PYTHONPATH=. python examples/compactflow/serve_model.py \
  --config examples/compactflow/configs/qwen3_coder_a100.model.json --preflight
```

## Run the benchmark trial

Prepare authorized benchmark files first:

```bash
PYTHONPATH=. python examples/compactflow/prepare_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --source-dir <download-directory> --output <cohort-directory> \
  --count 10 --seed 43
```

Run the prepared cohort:

```bash
PYTHONPATH=. python examples/compactflow/run_benchmark_trial.py \
  --config examples/compactflow/configs/qwen3_coder_a100.reference.json \
  --cohort <cohort-directory> --output <results-directory>
```

## Run checks

```bash
python -m pytest -q tests/src/compactflow
ruff check evoagentx/compactflow \
  examples/compactflow/run_experiments.py \
  tests/src/compactflow
```

## Configuration files

- [Configuration guide](examples/compactflow/CONFIGURATION.md)
- [Appendix G configuration table](examples/compactflow/APPENDIX_G_CONFIGURATION.md)
- [Benchmark trial protocol](examples/compactflow/BENCHMARK_TRIAL.md)
- [Reference JSON](examples/compactflow/configs/qwen3_coder_a100.reference.json)
