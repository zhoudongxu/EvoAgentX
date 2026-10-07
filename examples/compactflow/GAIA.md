# GAIA integration and acceptance protocol

The six construction methods (AFlow, EvoAgentX, Base Planner, Expert Policies,
Static Library, CompactFlow) share GaiaToolSession, the typed guarded
runtime, task evaluator, model endpoint and generation seeds. AFlow and SEW keep
their native search logic and select workflows using the frozen validation pool.
They do not participate in policy evolution rounds. AFlow's GAIAToolAgent is a
statically exported operator; generated dynamic Python is still rejected.

## Data and partitions

Use the authorized local bundle at
/data/Optx/outputs/compactflow/evolution-raw-data-with-gaia. The upstream revision
is 682dd723ee1e1697e00360edccf2366dc8418dd9. The labeled 2023 validation pool
contains 165 tasks; the reference deterministically selects 150. The internal
90/30/30 source/validation/target partition is not the provider's original split.
The reference has five disjoint six-task CompactFlow validation folds. The pilot
selects 15 tasks with 9/3/3 partitions, one evolution round and generation seed 42.
The pilot retains the unchanged deterministic sample, even if a level or modality
is absent. Synthetic modality checks are independent infrastructure checks.

Level is read from the original capitalized field and preserved as an integer.
Family IDs, normalized templates and supplied leakage keys are unioned before
exact partitioning. This does not claim arbitrary semantic/entity disjointness.
Missing files cause incomplete coverage; no attachment-based task filtering occurs.

Each task asset has a relative path, type, size and SHA-256 in the manifest and
gaia_assets.lock.json. Only the current question and authorized task assets enter
planning/execution. Final answer and Annotator Metadata remain evaluator/preparation
data. Dataset metadata files never enter the Python mount. A generated asset becomes
visible to another method only through the same exact recorded tool observation;
method-local asset authorization and conversation state remain independent.

## Models and budgets

| Setting | Locked value |
|---|---|
| Main model | GPU 7, Qwen3-Coder-30B-A3B-Instruct, existing endpoint 127.0.0.1:8019/v1 |
| Vision | GPU 0, Qwen/Qwen2.5-VL-7B-Instruct, cc594898137f460bfe9f0759e9844b3ce807cfb5, BF16, greedy decode |
| Audio | GPU 0, Systran/faster-whisper-large-v3, edaa852ec7e145841d8ffdb056a99866b5f0a478, FP16, beam 5, temperature 0, no previous-text conditioning |
| Auxiliary endpoint | 127.0.0.1:8020; vision and ASR serialized by one shared inference lock |
| Task limits | At most 12 tool-agent inference steps and 20 tool calls across all nodes |
| Main task token cap | Existing 65,536, including measured vision input/output tokens |
| Vision output/context | At most 1,024 output tokens; at most 8,192 input plus output; processor image bounds 256–1,280 visual tokens |
| Execution | Existing external call capacity 4, call timeout 180 s, workflow timeout 600 s |
| Python | Existing digest-pinned image; no network; current session assets only; 256 MiB, 1 CPU, 64 PIDs; 20 s |
| Search/browser | Fixed Bing HTML search through EvoAgentX SearchBase; isolated Firefox session using an explicitly prepared binary and driver |
| Capture | Phase-scoped HTTP(S), source/validation in their own stages; target only after frozen library/workflow |
| Replay | Exact task/split/arguments/assets/backend identity; missing observation fails; no live fallback |

The original GPU auxiliary configuration uses two GPUs in total. The separate
CPU auxiliary profiles described below keep only the main model on GPU 7. Report main and auxiliary model
costs separately: actual main/vision input/output tokens, audio seconds, and tool
elapsed time. Cached observations incur logical cost and zero additional inference
cost; these counters are separate. Unknown usage, cancellation and infrastructure
errors are incomplete, never zero-token successes. Wrong answers and ordinary
budget exhaustion remain in the attempted-task denominator.

Canonical graph nodes and edges count the exported workflow. The bounded agent's
internal model and tool calls are separately recorded; one agent node is not a
zero-cost computation. GAIA now uses versioned, developer-owned tool contracts.
Tools can start when their exact bound input fields are stable and effect/resource
guards permit it. Explicit primitive tools may publish independently completed
observation units. The adaptive gaia_agent publishes only its complete final answer.
Other benchmarks retain their existing tool access. See the contract protocol below.

## Explicit preparation

### CPU auxiliary profile

`configs/qwen3_coder_a100.gaia_cpu.reference.json` and
`configs/qwen3_coder_a100.gaia_cpu.pilot.json` retain the corresponding experiment
partitions, seeds, budgets, model revisions and main GPU-7 endpoint. They explicitly
set the auxiliary device to `cpu`, omit GPU selection, use eight CPU compute
threads, and use FP32 for both Qwen2.5-VL-7B and faster-whisper-large-v3. No model
quantization is introduced. The launcher hides CUDA before importing inference
libraries, and does not query or reserve any GPU in CPU mode.

CPU execution is a separate hardware/precision condition from the original
BF16/FP16 GPU auxiliary profile. Use these same CPU settings for all six methods,
start a new experiment directory, and report measured latency and usage. Do not
merge its timing/results into an existing GPU run or resume a GPU run with it.
Model identity, device, thread count and precision are included in preflight and
the frozen configuration/model locks. Existing call and workflow timeouts are
preserved; a slower request still obeys those budgets.

Start from the repository root after explicitly preparing model files:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  /data/Optx/venvs/compactflow-gaia-runtime/bin/python \
  examples/compactflow/serve_gaia_tools.py \
  --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json \
  --asset-root /data/Optx/outputs/compactflow --port 8020
```

Use the matching `gaia_cpu` configuration with preparation, evolution and baseline
commands. The service serializes vision/audio inference and reports actual vision
placement and precision at `/health`.

GAIA specifies tasks and evaluation, not these particular auxiliary models or a
GPU requirement. Text/search/table-only tasks need no vision or transcription
call. Image/scanned-document/video evidence needs visual understanding; spoken
audio needs transcription. The full capture protocol checks both capabilities up
front so unavailable modalities cannot silently remove difficult tasks. Exact
offline tool replay does not need the live auxiliary service, but requires every
requested observation to exist under the matching task/assets/backend identity.
CPU and GPU observations have different identities.

### GPU auxiliary profile

Install document/search/browser packages into the experiment environment and
vision/audio packages into the separate auxiliary environment. Models, browser,
driver and Docker image must already exist when experiments start. The runner
does not install packages or download model weights.

Install requirements-gaia-tools.txt in the experiment environment and requirements-gaia-aux.txt in the new auxiliary environment. The prepared A100 auxiliary environment uses torch 2.6.0 and transformers 4.57.6,
faster-whisper 1.2.1 and CTranslate2 4.6.0. CUDA runtime libraries are discovered
from explicitly installed NVIDIA wheel directories by the service launcher.
Vision and ASR run in separate processes to isolate their CUDA libraries, with a shared parent inference gate. The exact auxiliary package versions are recorded by /health and locked per run.

From /data/Optx/CompactFlow:

    PYTHONPATH=. /data/Optx/venvs/compactflow-gaia-runtime/bin/python examples/compactflow/prepare_gaia.py models --config examples/compactflow/configs/qwen3_coder_a100.reference.json
    CUDA_VISIBLE_DEVICES=0 /data/Optx/venvs/compactflow-gaia-runtime/bin/python examples/compactflow/serve_gaia_tools.py --config examples/compactflow/configs/qwen3_coder_a100.reference.json --asset-root /data/Optx/outputs/compactflow --port 8020

The launcher rechecks GPU 0 and refuses to start if another compute process is
present. It verifies model preparation.lock.json checksums before loading. For
networks where the A100 cannot validate the model host certificate, transfer the
same pinned, hash-verified files from the local machine; do not disable TLS checks.
The browser binary and driver paths are explicit tools.gaia configuration entries. The initial DuckDuckGo probe was unreachable from A100; the final profile fixes Bing HTML search before creating any experiment manifest. There is no automatic search-provider fallback.

## Acceptance sequence

    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/prepare_gaia.py preflight --config examples/compactflow/configs/qwen3_coder_a100.reference.json
    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/prepare_gaia.py smoke --config examples/compactflow/configs/qwen3_coder_a100.reference.json --audio-fixture /data/Optx/compactflow-tools/jfk.flac --output /data/Optx/outputs/compactflow/gaia-tool-smoke-v1
    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_evolution.py run --config examples/compactflow/configs/qwen3_coder_a100.pilot.json --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia --benchmarks GAIA --output /data/Optx/outputs/compactflow/evolution-gaia-pilot-v1
    PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python examples/compactflow/run_baselines.py run --config examples/compactflow/configs/qwen3_coder_a100.pilot.json --data-dir /data/Optx/outputs/compactflow/evolution-raw-data-with-gaia --evolution-dir /data/Optx/outputs/compactflow/evolution-gaia-pilot-v1 --benchmarks GAIA --output /data/Optx/outputs/compactflow/construction-gaia-pilot-v1

Use a new output directory for each changed code/configuration revision. Resume
with the identical command plus --resume; never relabel or append to a previous
experiment's frozen manifest. For the four-benchmark reference use the reference
profile, omit --benchmarks, and use new evolution and construction output paths.
Each GAIA method then requires 30 target tasks × 3 seeds = 90 target records.

New artifacts are gaia_assets.lock.json, gaia_capabilities.json,
auxiliary_models.lock.json, tool_records.jsonl and tool_snapshots/. Resume checks
code, config, data, attachments, backend packages and observation digests. Durable
model/tool results are reused without new inference. An interrupted request without
a durable response fails closed; it is not silently retried with unknown cost.
Target locks and before/after policy hashes enforce no target-time learning.

Execution scheduler comparisons use the existing recorded fixed-graph replay
bundles, which include the elapsed tool-agent execution. They never call live
tools or models. Tool-level replay additionally requires exact frozen observations.

GAIA summaries include level counts, per-method/level target accuracy, logical and
physical auxiliary usage. Pilot completion cannot make publication_status complete:
all four benchmarks, six main methods and other required study artifacts must pass
their completeness checks. Upstream external execution baseline gaps remain
explicit coverage gaps; these adapters do not invent those results.

The pinned GAIA ZIP inventory also contains legacy XLS and XML files. The shared file adapter reads XLS using xlrd and XML as plain text; ZIP archives retain the same path and size restrictions.

Current A100 verification and outstanding deployment blockers are recorded in [GAIA_ACCEPTANCE.md](GAIA_ACCEPTANCE.md). Legacy XLS test-fixture generation additionally uses xlwt==1.3.0; runtime XLS reading only requires xlrd.


## GAIA stream contract protocol: gaia_observation_units_v1

`evoagentx/compactflow/gaia_contracts.py` is the authoritative registry. This version
replaces the former blanket early_safe=False and misleading pure effect label for
GAIA tools. Planner schemas now admit registered primitive tools as well as the
adaptive gaia_agent. The policy controls use that shared planner; native AFlow and
SEW agent nodes use the same contract/effect registry while retaining their native
complete-output granularity. No native intermediate outputs are invented.

| Annotation | Source and guarantee |
|---|---|
| Input footprints | Compiled from explicit, schema-checked workflow input bindings. Binding proposals may be model-generated. No arbitrary-code dependency inference is claimed. |
| Output names | Declared by the workflow; the adapter validates exact unit-to-field matching before invoking any unit. |
| Stream contract and early-safe | Developer-authored per registered tool, versioned with code/configuration and serialized in graph metadata. Exact bound values, task-local context and independent effect/resource checks are required. |
| Stable event | A complete independent observation is cached, hashed and durably journaled before publication. It is immutable after publication, even if a later independent unit fails. The runtime still invalidates descendants of a failed producer. |
| Effect classes | Network reads, task asset reads/writes, auxiliary inference, or isolated container execution. No blanket pure classification. Asset conflicts receive conservative task-wide completion barriers ordered by the original DAG topology. |
| Manual/oracle labels | No per-task manual labels or gold-answer annotations. Developer-written adapter contracts are manual engineering specifications, not inferred-label accuracy measurements. |
| Accuracy | Semantic annotation accuracy remains unmeasured. Unit and runtime checks test implementation properties, not semantic precision/recall. |

A primitive node with one output accepts the ordinary tool argument object, normally
as an `arguments` JSON string emitted by an LLM builder with explicit question/assets
bindings. For multiple outputs the string contains:

```json
{"units":[
  {"field":"page1","arguments":{"asset_id":"sha256:AUTHORIZED_ID","unit":{"kind":"pdf_page","index":1}}},
  {"field":"page2","arguments":{"asset_id":"sha256:AUTHORIZED_ID","unit":{"kind":"pdf_page","index":2}}}
]}
```

The gaia_read_file node declares outputs `["page1","page2"]`; downstream inputs
reference exactly `reader.page1` or `reader.page2`. Each unit runs in declared-field
order through GaiaToolSession.invoke and consumes one of the existing 20 task-wide
tool calls. All costs and existing time/token limits apply. Nested units, duplicate
fields and missing/extra output mappings fail before calls. Whole observations are
serialized as JSON strings. A completed ordinary tool-error observation may also be
stable; infrastructure/cost failures cannot commit a successful output.

Supported bounded selectors: PDF page, DOCX paragraph, PPTX slide, XLSX/XLS/CSV row
range, text offset/limit, ZIP member list, audio start_seconds + seconds (at most
120 seconds), and a single video frame_seconds. Image questions and searches may
also be separate explicit units. Units are separate complete requests, not native
vision-token or ASR-prefix streaming. The agent's evolving thoughts, provisional
answers and incomplete external responses are never stable final-answer fields.
Tools that conflict on task assets remain completion-ordered; this conservative
scope may limit speedup and does not imply all GAIA graphs offer early execution.

`contract_records.jsonl` records task/split/seed/node/field, observation key, value
hash, commit timestamp and contract version. `tool_records.jsonl` and exact tool
snapshots retain usage and physical/logical costs. The workflow ReplayBundle records
these actual Partial/Complete events and timing for all three execution schedulers.
Offline scheduler replay uses no live tool/model target. An unresolved request
remains incomplete and cannot silently run again; completed exact observations are
reused. Missing ASR duration or vision token usage is incomplete. Atomic merged
asset indexes preserve separate session authorization under concurrent execution.

All new config profiles lock workflow_contract_version. Changing code/contracts
requires a NEW experiment directory; do not resume an old frozen run under this
protocol. Existing experiments and automatic continuation were paused for this
change. This check does not restart them or inspect benchmark validation/target:

```bash
cd /data/Optx/CompactFlow
PYTHONPATH=. /data/Optx/venvs/compactflow/bin/python \
  examples/compactflow/check_gaia_contracts.py \
  --config examples/compactflow/configs/qwen3_coder_a100.gaia_cpu.reference.json \
  --output /data/Optx/outputs/compactflow/gaia-contract-smoke-NEW \
  --auxiliary
```

The smoke uses generated source-scoped CSV/image/audio fixtures, the real planner
and streaming executor, a fixed typed primitive workflow, exact offline replay,
and optionally the existing CPU vision/ASR service. Its separate fixed workflow
checks contracts rather than measuring benchmark accuracy or latency gains. The
smoke-only vision output bound is 16 tokens and is saved in its config lock; formal
budgets are unchanged. It requires a fresh output path and deliberately has no
automatic retry/resume of unknown requests.
