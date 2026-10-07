# A100 GAIA integration acceptance

Status (2026-09-29): CPU auxiliary deployment and independent live-tool acceptance
passed; full benchmark acceptance remains incomplete.
These checks do not constitute GAIA benchmark accuracy or completion of the paper.

## Verified

- All 169 CompactFlow regression tests passed after the file-adapter and session
  changes. The final planner change passed its 77 relevant regression tests.
  Coverage includes the six native/control adapters with fake models/tools,
  disjoint folds, private-label exclusion, target freeze, strict replay,
  cancellation accounting, resume, ZIP restrictions and global tool concurrency.
- Reference data preflight imports 600 tasks across all four benchmarks. GAIA
  has exactly 90 source / 30 validation / 30 target tasks, with five six-task
  validation folds. GAIA level counts are 48 / 81 / 21 for levels 1 / 2 / 3.
- Pilot data preparation has exactly 15 GAIA tasks, 9 / 3 / 3. Its fixed sample
  contains 7 level-1 and 8 level-2 tasks; it was not changed to add modalities.
- Both GAIA ZIP inventories were inspected during data preparation. Legacy XLS
  and XML support was added for nested assets, alongside the declared formats.
- Real Qwen3-Coder-30B on GPU 7 completed a synthetic CSV task through the planner,
  canonical runtime and shared GAIA tool agent: answer 5, valid execution,
  3 model requests, 3,276 measured tokens, 2 agent steps and 1 read_file call.
  This is an independent infrastructure fixture, not a selected benchmark task.
- Live PDF rendering, CSV reading, isolated Docker Python, Firefox browsing and
  fixed Bing HTML search passed. DuckDuckGo was unreachable from A100; Bing was
  fixed in the profiles before experiment manifests were created. No automatic
  provider fallback is allowed.
- A separate vision invocation returned the correct image color and measured
  300 input / 9 output tokens. Audio produced a transcription of an independent
  11-second speech fixture. These isolated calls do not certify service stability.

## CPU auxiliary deployment

The user-authorized CPU profile runs as `compactflow-gaia-cpu.service` at
`127.0.0.1:8020`, using the independent auxiliary environment. Both original
pinned models now execute in FP32 with eight compute threads and serialized
inference. `/health` confirms CPU placement; neither service process appears in
the GPU compute-process inventory. GPU 7 continues serving the main model.

The new independent `gaia-tool-smoke-cpu-v1` acceptance returned `complete`:

- Vision correctly answered `Red`, 5.808 seconds, 307 measured input/output tokens.
- Speech transcribed the 11-second independent fixture in 13.237 seconds.
- PDF rendering, CSV reading, sandboxed Python, Firefox browsing and fixed Bing
  search all returned successfully. The sandbox returned exit code 0 and sum 5.
- Seven live tool calls incurred 37.456 seconds, with complete usage accounting
  and zero cache hits. All observations belong to an independent source fixture.
- Preflight confirms matching pinned model identities, CPU device and thread count.
- The 40 focused GAIA/CPU regression tests passed before deployment; the full
  CompactFlow regression suite then passed all 178 tests in 38.19 seconds.

These are short acceptance measurements, not a long-duration load test. CPU and
GPU timing/precision conditions must remain separate. Use the new `gaia_cpu`
pilot/reference profiles and new experiment output directories.

## Original GPU-profile blocker

The shared GPU-0 auxiliary service repeatedly exits after receiving SIGKILL,
including when managed by a user service and when vision/audio are separated
into processes. The signal sender/cause is not established. The service manager
reports signal termination (status 9), not successful shutdown. The experiment
preflight for that GPU profile therefore reported the auxiliary endpoint unavailable.
The CPU deployment above passes preflight under its own configuration; it does
not establish the cause of the earlier GPU-profile SIGKILL.

Model revisions and transferred files were checksum-verified. Preparation used
an independent auxiliary environment; GPU 7 and other users' GPU processes were
not stopped or replaced. GPU 0 was rechecked before launches. The service is not
configured for an automatic restart loop.

The 15-task evolution plus six-method GAIA pilot has NOT completed, and the
four-benchmark reference experiment has NOT started. No GAIA target inference
or target evidence collection was performed. Publication status remains
incomplete; the blocked preflight run contains zero result records.

## Evidence on A100

- /data/Optx/outputs/compactflow/gaia-tool-smoke-cpu-v1/cpu_acceptance.json
- /data/Optx/outputs/compactflow/gaia-tool-smoke-cpu-v1/smoke.json
- /data/Optx/outputs/compactflow/gaia-tool-smoke-cpu-v1/tool_records.jsonl

- /data/Optx/outputs/compactflow/gaia-data-preflight-v1/data_preflight.json
- /data/Optx/outputs/compactflow/gaia-data-preflight-v1/reference.sample_manifest.jsonl
- /data/Optx/outputs/compactflow/gaia-data-preflight-v1/pilot.sample_manifest.jsonl
- /data/Optx/outputs/compactflow/gaia-regression-final-v3.log
- /data/Optx/outputs/compactflow/gaia-planner-regression.log
- /data/Optx/outputs/compactflow/gaia-agent-smoke-v3/source_fixture_record.json
- /data/Optx/outputs/compactflow/gaia-tool-smoke-v2/smoke.json
- /data/Optx/outputs/compactflow/evolution-gaia-pilot-preflight-v1/coverage.json
- /data/Optx/outputs/compactflow/evolution-gaia-pilot-preflight-v1/summary.json
- /data/Optx/outputs/compactflow/gaia-traced-service.log
- /data/Optx/outputs/compactflow/gaia-service-signal.*
- /data/Optx/outputs/compactflow/gaia-aux-environment.lock.txt
- /data/Optx/outputs/compactflow/gaia-client-environment.lock.txt

The data-preflight manifests are prepared-data artifacts. Formal runs create
their own immutable manifests and locks from the same pinned bundle and sampling
rules. Use fresh output directories after any code or configuration change.
With CPU tool acceptance passed, follow GAIA.md's pilot evolution → six methods
→ four-benchmark reference acceptance sequence using the CPU profiles.
