# Appendix G: reference configuration and implementation mapping


### Table 9: LLM configuration

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Backbone model | Qwen/Qwen3-Coder-30B-A3B-Instruct; revision b2cff646eb4bb1d68355c01b18ae02e7cf42d120 | `model.repository`<br>`model.revision` |
| Provider and endpoint | Local vLLM 0.26.1rc1.dev416+g2dfb8ba59; OpenAI-compatible Chat Completions; http://127.0.0.1:8019/v1. No separately versioned hosted API. | `model.api_product`<br>`model.base_url`<br>`serving.version` |
| Query generation | Qwen3-Coder-30B-A3B-Instruct | `model.components.query` |
| Policy selection | Qwen3-Coder-30B-A3B-Instruct | `model.components.selector` |
| Workflow planner | Qwen3-Coder-30B-A3B-Instruct | `model.components.planner` |
| Policy distillation | Qwen3-Coder-30B-A3B-Instruct | `model.components.distiller` |
| Semantic/contract checker | Qwen3-Coder-30B-A3B-Instruct; deterministic JSON Schema and graph checks are mandatory | `model.components.checker` |
| Workflow executors | Qwen3-Coder-30B-A3B-Instruct | `model.components.executor` |
| Temperature | query 0.2; selector 0.2; planner 0.7; distiller 0.2; checker 0; executor 0.7 | `model.components` |
| Top-p | All roles: top-p 0.8; top-k 20; repetition penalty 1.05. | `model.top_p`<br>`model.top_k`<br>`model.repetition_penalty` |
| Maximum output length | query 512; selector 1,024; planner 4,096; distiller 2,048; checker 1,024; executor 4,096 | `model.components` |
| Context length | 32,768 tokens; reject oversized requests, no silent truncation. Client admission uses a conservative UTF-8 byte bound. | `model.context_length`<br>`model.truncation` |
| Random seed | Main generation: 42, 43, 44; server seed 42. Seeds are forwarded; bitwise determinism is not guaranteed. Diagnostic seeds: Table 13. | `evaluation.generation_seeds`<br>`serving.seed`<br>`model.seed_support` |
| Reasoning mode | Non-thinking; provider prompt_tokens + completion_tokens. Include any provider-reported reasoning tokens in completion usage. Missing usage makes token results non-reportable. | `model.reasoning_mode`<br>`model.tokenizer` |
| Retry policy | The current shared ModelClient sends one HTTP request per logical attempt and does not automatically retry after transport starts, even before output. Unknown usage remains incomplete. Distinct planner repair calls retain their configured budget. Legacy max_retries/backoff fields do not enable transport retries. | `model.max_retries` (legacy)<br>`model.timeout_seconds`<br>`model.connect_timeout_seconds`<br>`runner.failure_protocol` |
| Prompt version | Six prompts under examples/compactflow/prompts/. SHA-256 for each is in artifacts.source_hashes in the reference JSON. The per-file hashes in the locked profile are authoritative. | `artifacts.source_hashes` |
| Tokenizer | Qwen/Qwen3-Coder-30B-A3B-Instruct at b2cff646eb4bb1d68355c01b18ae02e7cf42d120; tokenizers 0.22.2; accounting uses provider usage. | `model.tokenizer`<br>`serving.observed_runtime.tokenizers` |

### Table 10: Retrieval encoder and index

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2; revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; 384 dimensions; CPU; sentence-transformers backend. | `encoder` |
| Input template | Policy text: description + operation.type + metadata.tags. Query: query.txt. Exact text and encoder revision form the cache key. | `encoder.policy_template`<br>`encoder.query_template`<br>`encoder.cache` |
| Vector normalization | L2 normalization of every query/policy vector. | `encoder.normalize_embeddings` |
| Similarity function | Cosine similarity for retrieval and near-duplicate detection. | `encoder.similarity`<br>`encoder.shared_deduplication_index` |
| Index implementation | Exact brute-force search; no approximate index. | `encoder.index` |
| Index parameters | 384 dimensions; normalized vectors; exact top-K0=20; no HNSW/FAISS tuning parameters. Backend package lock must accompany the formal run. | `encoder.dimension`<br>`construction.semantic_top_k0` |
| Update schedule | Immediately after verified policy admission. | `encoder.index_update` |
| Deduplication index | Shared encoder, normalization, similarity and index; merge threshold 0.90. | `encoder.shared_deduplication_index`<br>`construction.merge_similarity_threshold` |

### Table 11: Benchmarks and quality metrics

| Setting | Reference value | JSON key(s) |
|---|---|---|
| MBPP | google-research-datasets/mbpp @ 4bb6404fdc6cacfda99d4ac4205087b89d32030c. Sanitized test; required reference pool 150 tasks; pass@1 using released dataset tests and a pinned Docker sandbox. Exact selected IDs and split assignments are in sample_manifest.jsonl frozen before inference. | `datasets.0` |
| HotpotQA | hotpotqa/hotpot_qa @ 1908d6afbbead072334abe2965f91bd2709910ab. Distractor validation; required reference pool 150 tasks; token F1 and exact match; supplied distractor context in original record order. No FullWiki index. | `datasets.1` |
| MATH | EleutherAI/hendrycks_math @ 21a5633873b6a120296cce3e2df9d5550074f4a3. Test, all seven subject configurations and levels 1–5; required reference pool 150 stratified tasks; released boxed-answer extraction and normalized exact match. | `datasets.2` |
| GAIA | gaia-benchmark/GAIA @ 682dd723ee1e1697e00360edccf2366dc8418dd9. 2023_all validation, Level 1–3; required reference pool 150 tasks; released task-accuracy evaluator. Assets are task-scoped and SHA-256 locked. Missing assets/capabilities mark coverage incomplete; no silent subset substitution. | `datasets.3` |

### Table 12: Source / validation / target partition

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Task-family definition | Provided nonempty family_id; otherwise SHA-256 of normalized_numeric_template_v1: lowercase, remove ASCII punctuation and articles a/an/the, collapse whitespace, then replace numeric matches with NUM. The manifest records family_id and family_method; see CONFIGURATION.md for union groups and scope. | `partition.family_definition` |
| Source proportion | 60%; exactly 90/150 tasks per benchmark, without splitting a union group. Actual IDs/counts are frozen in sample_manifest.jsonl before inference. Evolution uses 20 bootstrap + 5 x 10 source tasks; the remaining 20 are unused by evolution. | `partition.source_fraction`<br>`partition.nominal_counts_per_benchmark.source` |
| Validation proportion | 20%; exactly 30/150 tasks per benchmark. Five mutually disjoint, whole-group folds of six; evolution round r uses fold r. Candidates within the same round reuse that fold. IDs and validation_fold are frozen in sample_manifest.jsonl. | `partition.validation_fraction`<br>`partition.nominal_counts_per_benchmark.validation`<br>`runner.validation_folds` |
| Target proportion | 20%; exactly 30/150 tasks per benchmark. Frozen task IDs; access after discovery/selection and library/workflow freeze. Infeasible exact whole-group allocation marks the benchmark incomplete. | `partition.target_fraction`<br>`partition.nominal_counts_per_benchmark.target` |
| Leakage prevention | Union shared family IDs, normalized-question hashes, numeric-template hashes and supplied leakage_keys within each benchmark. Sampling seed 43; split seed 42. Never split a group across partitions/folds. No cross-benchmark deduplication or arbitrary paraphrase/entity-disjointness guarantee. | `partition.leakage_prevention`<br>`partition.seed`<br>`partition.sample_seed`<br>`partition.group_assignment` |
| Library state at target evaluation | One shared final verified library per evolution run/variant, used by all included benchmarks and target generation seeds 42/43/44. No target-time distillation, verification, admission, merge, eviction or statistic updates. This is not independently evolved per-seed libraries. | `partition.target_library`<br>`runner.source_seed`<br>`runner.target_seeds` |

### Table 13: Scale and randomization

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Construction workflows | 4 benchmarks x 50 source tasks x 6 seeds x 1 base-planner method = 1,200. Seeds: 11,22,33,44,55,66. Up to 12 interventions/workflow; intervention runs counted separately. | `study_allocation.construction_diagnostic` |
| Execution workflows | Independent execution_main: 4 x 25 target tasks x 10 seeds (1011–1020) = 1,000 Base Planner workflows without policies. LLMOrch/LLMCompiler/CompactFlow each use one warmup and five timed replays: 3,000 warmups and 15,000 timed records. Mixed diagnostic studies keep their separately frozen method list. | `study_allocation.execution_diagnostic` (task/seed allocation)<br>`paper_experiments.execution_methods` (latency profiles) |
| Generation seeds | Main: 42,43,44; construction diagnosis: 11,22,33,44,55,66; execution diagnosis: 1011-1020; held-out verification: 101,202,303. | `evaluation.generation_seeds`<br>`study_allocation`<br>`construction.heldout_seeds` |
| Runtime seeds | Replay ordering: 31415, 31416, 31417, 31418, 31419; bootstrap seed 2718. | `evaluation.runtime_seeds`<br>`evaluation.bootstrap_seed` |
| Rollouts per task | One workflow per task-method-generation-seed pair; three generation seeds in the main study. Replay repetitions are not new rollouts. | `evaluation.rollout_budget`<br>`evaluation.generation_seeds` |
| Repeated executions | 1 untimed warmup + 5 timed executions for each fixed workflow and scheduler. | `evaluation.warmups`<br>`evaluation.repetitions` |
| Evolution rounds | Five rounds after 20 bootstrap source tasks per benchmark. One shared library; source generation seed 42. Within each round benchmarks are processed in sorted name order. | `runner.rounds`<br>`runner.bootstrap_tasks`<br>`runner.source_seed` |
| Batch size per round | Per benchmark: 10 unused source tasks per round; at most one candidate per source task and four retained candidates per round. The candidate cap is not a global four-benchmark cap. | `construction.distillation` |
| Method order | Seeded permutation for each fixed workflow and timed repetition. Pair task, workflow, model outputs and runtime seed across methods. | `execution.replay.method_order` |

### Table 14: Baseline configuration

| Setting | Reference value | JSON key(s) |
|---|---|---|
| AFlow | https://github.com/geekan/MetaGPT @ 11cdf466d042aece04fc6cfd13b28e1a70341b1f; local EvoAgentX AFlowOptimizer adapter; 20 rounds, population sample 4, 1 candidate/round, 3 validation and 3 test rounds; native Custom/CustomCodeGenerate/AnswerGenerate/ScEnsemble/QAScEnsemble (benchmark-specific; static export only); stop at 20 candidates or search token budget. | `baselines.aflow` |
| EvoAgentX | https://github.com/EvoAgentX/EvoAgentX @ d77fd6b9a3e76c8dd83bebe3374c53a3f5d16f54; local SEWOptimizer with safe YAML and lower_workflow_graph; 20 iterations, 1 candidate/iteration, 3 validation rounds, max 12 nodes. Shared model and budget; stop at 20 candidates or token cap. | `baselines.evoagentx` |
| LLMCompiler | https://github.com/SqueezeAILab/LLMCompiler @ a00c9d35507507da70e8c637eee64efc8c1857ae; shared backbone; max concurrency 4. Official TaskFetchingUnit is vendored with MIT license and source checksums. The fixed-workflow adapter invokes the native scheduler with shared typed call/resource boundaries; no upstream planner/joiner is invoked in the execution-only comparison. Internal level barriers remain separately labeled. | `baselines.external_execution.llmcompiler` |
| LLMOrch | llmorch_paper_reimplementation_v1 in evoagentx/compactflow/llmorch.py; enabled by the latency profiles. Core ready-batch/I/O-priority/logical compute-slot coordination with complete predecessor dependencies and shared capacity. No author-code commit, MPI overhead or recovery reproduction is claimed. This is distinct from internal complete_dependency. | `baselines.external_execution.llmorch` |
| Percentage threshold | Internal implementation, versioned by the run code hash; threshold 0.5. Present top-level output fields / declared output properties. Required values must exist; stability check omitted; effect/capacity checks retained; batch 1. An internal diagnostic control, not part of the three-method execution-only main table. | `execution.percentage_threshold`<br>`execution.percentage_denominator`<br>`execution.method_definitions.percentage_threshold` |
| Sequential | Internal implementation, versioned by the run code hash; one semantically ready call at a time, including independent calls; complete-result dependencies; first-ready FIFO with topological tie-breaking. An internal diagnostic control. | `execution.method_definitions.sequential`<br>`execution.queue_policy` |

### Table 15: Tools and runtime environment

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Agent/tool schemas | llm and retrieve_context; GAIA additionally uses gaia_agent and registered primitive tools under tools.gaia.registry. workflow.schema.json, candidate.schema.json and per-graph schemas are code/config locked; gaia_observation_units_v1 identifies the GAIA contract registry. | `tools.registry`<br>`tools.schemas`<br>`artifacts.source_hashes` |
| Retrieval corpora | HotpotQA uses the pinned dataset-provided distractor context, in record order. No live corpus index. GAIA web/tool evidence requires immutable per-task snapshots. | `datasets.1`<br>`datasets.3` |
| External APIs | Loopback vLLM and auxiliary model services; fixed bing_html search and isolated Firefox via the shared GAIA tools. Explicit preparation uses pinned Hugging Face artifacts. Per-observation provenance, backend identity and capture time are recorded; live web content is not claimed immutable without its frozen observation. | `model.base_url`<br>`tools.gaia.auxiliary_url`<br>`tools.gaia.search_backend` |
| Sandbox | python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254; 10 s; 256 MiB RAM, 1 CPU, 64 PIDs, 16 MiB tmpfs; no network, read-only root, drop all capabilities, no-new-privileges, UID/GID 65534:65534. | `tools.mbpp_sandbox` |
| Correctness evaluators | Code-locked benchmarks.py: evaluate_mbpp (pass@1), hotpot_score (F1/EM), boxed_answer + normalize_math (exact match), gaia_score (task accuracy). Gold answers, hidden tests and GAIA answer metadata remain evaluator-only. | `datasets`<br>`tools.mbpp_sandbox` |
| CPU/GPU | Active CPU profiles: main Qwen3-Coder on GPU 7, NVIDIA A100 80 GB, BF16, TP/PP 1/1; GAIA vision and ASR on CPU FP32, eight compute threads, shared auxiliary inference concurrency one. Recorded shared host: Xeon Platinum 8378A, 128 logical CPUs, 503.5 GiB RAM, ext4. No exclusive host allocation claimed; environment locks/service records identify each run. | `serving.device_requirement`<br>`serving.dtype`<br>`tools.gaia.device`<br>`tools.gaia.cpu_threads` |
| Software | Ubuntu 24.04.1 LTS; kernel 6.17.0-40-generic; serving Python 3.11.15, torch 2.13.0+cu132, CUDA build 13.2, transformers 5.14.1, vLLM 0.26.1rc1.dev416+g2dfb8ba59. Full per-run dependency lock still required. | `serving.observed_runtime`<br>`serving.version` |
| Model-service region | Same A100 host as execution client; loopback endpoint. Physical geographic region is not recorded; no remote hosted-model region applies. | `model.base_url`<br>`serving.host` |
| Network | Loopback for live model calls. External network is permitted for explicit preparation and phase-isolated GAIA evidence capture; target capture starts only after freezing. Timed fixed-graph replay is offline and reproduces recorded call delays. | `tools.network`<br>`execution.replay.network_state` |
| Concurrency limit | Client/server max inflight sequences 4; external_call capacity 4; batched tokens 8,192; GPU memory fraction 0.82; eager, chunked prefill on, prefix cache off. Four full-length contexts may require KV-cache preemption. | `model.concurrency`<br>`serving`<br>`execution.external_call_capacity` |
| API rate limits | Protocol limits: 60 requests/min, 100,000 tokens/min, burst 4. Current client enforces concurrency/token admission; cross-client RPM/TPM throttling is not yet integrated. | `tools.request_limits` |
| Clock source | time.perf_counter / asyncio loop.time; CLOCK_MONOTONIC. Reported clock resolution 1e-9 s; this is not a claim of nanosecond measurement accuracy. | `tools.clock` |

### Table 16: Retrieval and policy selection

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Semantic pre-retrieval size K0 | 20 | `construction.semantic_top_k0` |
| Comprehensive re-ranking size K | 5 | `construction.top_k` |
| Maximum selected policies k | 3 | `construction.max_policies` |
| Semantic weight lambda_sem | 0.55 | `construction.retrieval_weights.semantic` |
| Structural weight lambda_str | 0.25 | `construction.retrieval_weights.structural` |
| Historical-utility weight lambda_hist | 0.1 | `construction.retrieval_weights.historical` |
| Confidence weight lambda_conf | 0.1 | `construction.retrieval_weights.confidence` |
| Minimum semantic similarity theta_sem | 0.15 | `construction.minimum_semantic_score` |
| Minimum applicability theta_app | 1 | `construction.minimum_applicability` |
| Minimum selection score theta_sel | 0.25 | `construction.minimum_selection_score` |
| Compatibility rule | No conflicts_with pair or shared exclusive_group; every structured precondition holds. Model ranking cannot bypass deterministic checks. Active retrieval weights sum to 1. | `construction.compatibility`<br>`construction.retrieval_weights` |
| Tie breaking | Descending total score, then ascending policy ID. | `construction.tie_breaking` |

### Table 17: Policy-library management

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Initial library P0 | Six developer-written candidate templates in examples/compactflow/policies/seed_policies.json; SHA-256 c3e7c770a7d423ef96a318f79d6b6e2d5ab552a405c069e7c39c661cad560abc. Initial candidate status does not constitute held-out verification. Each run saves policy_snapshots/initial.json. | `construction.library.initial_path`<br>`construction.library.initial_status`<br>`artifacts.source_hashes` |
| Maximum capacity | 256 | `construction.library.capacity` |
| Expiration rule | No automatic expiry; retain evidence indefinitely. | `construction.library.expiry` |
| Eviction rule | Lowest confidence, then lowest utility, then oldest update round, then policy ID. | `construction.library.eviction` |
| Negative evidence | Persist every rejected paired outcome; exclude rejected evidence/policies from default target retrieval. Deduplicate by candidate version, benchmark, task ID and seed. | `construction.library.negative_evidence`<br>`construction.statistics.duplicate_pair_handling` |
| Versioning | Stable IDs; increment version on merge; preserve parent links and original operation/precondition; union evidence IDs and recompute statistics. | `construction.library.versioning`<br>`construction.library.merge` |

### Table 18: Planner validation and repair

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Output grammar | workflow.schema.json; SHA-256 ff4a867308e40ba7a0785ba0dc430ba9c096bd23872a8af809328f6208ebb22d; bounded typed DAG, max 12 nodes, one answer sink. | `construction.planner.output_schema`<br>`construction.planner.max_nodes`<br>`artifacts.source_hashes` |
| Validation checks | JSON schema, unique IDs, acyclic graph, every node reaches sink, typed field bindings, registered tools, selected policy IDs. Structured policy preconditions remain required at selection. | `construction.planner.checks` |
| Maximum repair attempts | 2 after the initial attempt; bounded fallback has the same repair limit. | `construction.planner.max_repairs` |
| Repair prompt | plan.txt; current source hash 4637283957684a815f2e8f45c3a1928a75a941677c793be98b265c6710b9a1fd. Full workflow schema, public-field binding guidance, previous workflow and validation error; seed incremented per attempt. Locked artifacts.source_hashes is authoritative. | `construction.planner.prompt`<br>`artifacts.source_hashes` |
| Policy removal order | After conditioned attempts are exhausted, remove all selected policies and enter one bounded base-planner phase. No incremental score-based removal is implemented. | `construction.planner.policy_removal` |
| Final fallback | If policy-conditioned planning fails, run the base planner with up to 1 initial + 2 repair calls. If that phase fails, fail the task. Base-only planning has no further fallback. | `construction.planner.fallback`<br>`construction.planner.max_repairs` |

### Table 19: Statistics and distillation

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Utility update | u0=0.5; mean of clip(weighted paired cost reduction, 0, 1) over unique held-out pairs; duplicate evidence counted once. | `construction.statistics.utility_initial`<br>`construction.statistics.utility_update` |
| Confidence update | c0=0.25; Wilson lower bound for the quality-and-cost success fraction; z=1.96. Count unique held-out pairs; no target updates. | `construction.statistics.confidence_initial`<br>`construction.statistics.confidence_update`<br>`construction.statistics.wilson_z`<br>`construction.statistics.target_updates` |
| Utility normalization | Per-pair relative cost reductions; denominator floor 1e-9; token/latency/graph weights 0.50/0.25/0.25. Utility clipped to [0,1]. Graph cost = nodes + edges. | `construction.aggregate_cost`<br>`construction.normalization_floor`<br>`construction.cost_weights` |
| Eligibility predicate | Valid execution; zero observed contract violations with complete audit; quality no lower than paired base; weighted cost reduction >=5%; source Pareto frontier. Near duplicates still require held-out verification. | `construction.distillation.eligibility`<br>`construction.admission_order` |
| Reference baseline | Source eligibility: paired no-policy planner versus current source library. Admission: current verified library versus that same library plus candidate, using identical benchmark/split/task/seed. Both use the complete-dependency runtime; source and validation comparisons are not interchangeable. | `construction.distillation.reference` (source)<br>`runner.heldout_seeds` (admission) |
| Candidate budget | Per benchmark: at most one candidate per source task and four retained candidates per evolution round. | `construction.distillation.max_candidates_per_task`<br>`construction.distillation.max_candidates_per_round` |
| Distillation prompt | distill.txt SHA-256 3cb6c0a726f0e0e3ea62e1560eaf9445ddb39d443d36ded46b2c5dd50e797b64; candidate.schema.json SHA-256 d49d3e6a4ab51aa2fea14a856e6b7d0dc62991cc71400bbe5dffab9cbd51bfa0. | `construction.distillation.prompt`<br>`construction.distillation.schema`<br>`artifacts.source_hashes` |
| Minimum candidate validity | 1 (100% of candidate verification runs). | `construction.minimum_candidate_valid_rate` |

### Table 20: Admission and evolution

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Allowed quality degradation epsilon_Q | 0 in native fractional score units. | `construction.quality_tolerance` |
| Minimum cost reduction delta_C | 0.05 (5% weighted paired reduction). | `construction.minimum_cost_reduction` |
| Merge similarity threshold delta_merge | 0.9 cosine; verify before merge. | `construction.merge_similarity_threshold` |
| Token-cost weight alpha_tok | 0.5 | `construction.cost_weights.tokens` |
| Latency-cost weight alpha_lat | 0.25 | `construction.cost_weights.latency` |
| Graph-cost weight alpha_G | 0.25; nodes + edges. | `construction.cost_weights.graph` |
| Candidate minimum validity eta_valid | 1 (100%). | `construction.minimum_candidate_valid_rate` |
| Held-out tasks per candidate N_heldout | 6 in the current round fold | `construction.heldout_tasks_per_candidate` |
| Seeds per held-out task R_heldout | 3 seeds: 101,202,303; 6 x 3 = 18 paired evaluations per candidate. Five disjoint folds of six tasks per benchmark. | `construction.heldout_seeds` |
| Maximum candidates per round B_cand | 4 | `construction.distillation.max_candidates_per_round` |
| Other-cost tolerance epsilon_other | 0.05 (at most 5% increase in non-target costs). | `construction.other_cost_tolerance` |
| Motif-clustering threshold theta_motif | 0.9 cosine with the same normalized encoder. | `construction.motif_similarity_threshold` |

### Table 21: Schemas and dependency artifacts

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Input schema | Per-call JSON Schema derived from explicit input bindings; additionalProperties=false. Empty declared inputs stay empty. Stored with each compiled replay graph. | `tools.schemas`<br>`construction.planner.output_schema` |
| Output schema | Every declared output is a required string field; additionalProperties=false; all declared fields required at Complete. Stored per workflow. | `tools.schemas` |
| Consumer footprint | Compiled from schema-checked, LLM-proposed explicit source-field/input-argument bindings; no implicit inherited inputs for explicitly bound calls. This checks declared dependencies, not semantic completeness of arbitrary model-generated code. | `execution.stream`<br>`tools.schemas` |
| Stable fields | LLM: one fully parsed, single-assignment NDJSON field. GAIA primitives: completed independent observation units, journaled before publication. gaia_agent final answer remains complete-only. Published values are immutable; raw token/provisional answer prefixes are not stable. | `execution.stream.llm_stability`<br>`execution.stream.gaia_stability` |
| Mutable fields | Mutable fields never enable early guards. Reference LLM fields are immutable once emitted; other adapters must declare mutable fields explicitly. | `execution.stream.mutable_fields` |
| Early-safe consumers | Developer-owned registered early_safe rules, with exact declared-field readiness, monotone producer, cancellation, effect and resource checks. GAIA tool contracts include network/asset/auxiliary effects. No per-task manual or answer-oracle labels; semantic annotation accuracy is unmeasured. | `execution.method_definitions.guarded`<br>`execution.effects.early_dispatch_default` |
| Footprint extraction | explicit_typed_input_bindings_v1: deterministic compilation of planner-proposed bindings. Output names are planner-declared; adapter contracts are developer-authored. Schema/runtime checks do not measure semantic annotation accuracy. | `tools.schemas` |

### Table 22: Effects and resource constraints

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Effect class | llm/retrieve_context calls are pure at the workflow boundary; GAIA primitive/agent effects use the developer registry with network, asset, auxiliary and isolated-execution annotations. Vocabulary: pure, idempotent, compensable, non_commuting; unclassified effects default conservatively. No blanket pure label for GAIA. | `execution.effects` |
| Effect ordering | Explicit effect-dependency edges; require completion unless a partial-safe effect is explicitly declared. Save the effect-edge manifest with each graph. | `execution.effects.ordering` |
| Compensation | Disabled in the reference profile. Compensable calls retain complete-result barriers; no compensation handler mapping is claimed. | `execution.effects.compensation` |
| Capacity vector C | C[external_call]=4 concurrent slots; capacity sweep {1,2,4,8,16}. | `execution.resource_capacity`<br>`execution.sweeps.external_call_capacity` |
| Call demand rho(v) | llm/retrieve_context each reserve external_call=1. GAIA call demands/effect metadata are compiled with the registered operation; GaiaToolSession additionally enforces shared main-model limits, task tool budgets and auxiliary inference concurrency one. | `execution.resource_demands`<br>`tools.gaia` |
| Admission check | For every resource dimension: current usage + demand <= capacity. Check, reserve, record dispatch and READY-to-RUNNING transition under one scheduler lock; release on termination/cancellation. | `execution.resource_capacity` |
| Main operating point | Capacity 4; materialization batch 1. The configured client concurrency remains 4; capacities above 4 are informative for offline replay only unless live serving limits are changed and re-locked. | `execution.external_call_capacity`<br>`execution.materialization_batch_size`<br>`model.concurrency` |

### Table 23: Streaming and scheduling

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Percentage threshold | Main 0.5; sweep {0.25,0.5,0.75}; number of present top-level fields divided by number of declared output properties. All required consumer values must exist. | `execution.percentage_threshold`<br>`execution.sweeps.percentage_threshold`<br>`execution.percentage_denominator` |
| Materialization batch size | Main 1; sweep {1,2,4,8,16}. Replay groups this many partial events; merged event time is the last constituent event time. | `execution.materialization_batch_size`<br>`execution.sweeps.materialization_batch_size` |
| Partial event format | Typed payload: data, stable_fields, effects, call_id. Recorder adds consecutive per-call sequence (from 0) and relative monotonic time. Model wire format: NDJSON field/value strings. | `execution.stream` |
| Complete event | Terminal success with full output data and effects; all required output fields validated; stable fields must match prior publication. | `execution.stream.format` |
| Failure event | Terminal error string/exception and call ID, plus recorded sequence/time. Current interface has no standardized error-code taxonomy. Scheduler retry is disabled; consumed partial descendants are invalidated. | `execution.stream.format`<br>`execution.failure_recovery` |
| Out-of-order chunks | Replay loader rejects nonconsecutive sequence IDs or nonmonotone timestamps. Live HTTP stream order is trusted; arbitrary unordered transport needs a separate ordering adapter. | `execution.stream.out_of_order` |
| Duplicate chunks | Per-call replay sequence/timestamps must be valid. NDJSON repeated stable-field emissions are rejected; the materializer rejects value changes. Generic transport event-ID deduplication is not implemented; the inherited execution.stream.duplicates description is not an exactly-once transport guarantee. | `execution.stream.duplicates` |
| Dynamic per-item calls | Disabled. Use bounded, statically declared instances with unique call IDs; no runtime graph mutation or per-item spawning. | `execution.stream.dynamic_per_item` |
| Scheduler queue | FIFO by first-ready timestamp; topological order breaks ties; scan for calls whose resource demand fits available capacity. | `execution.queue_policy` |
| Fairness | Finite DAG and FIFO fit scan; no unbounded spawning or priority aging. No stronger starvation bound is claimed. | `execution.fairness` |
| Timeout/retry/backoff | Per-call 180 s; workflow 600 s; exactly one logical runtime attempt; scheduler retry backoff 0. The current ModelClient also does not retry a started HTTP request. Cancellation may require bounded usage-drain time, reported separately from logical execution. | `execution.call_timeout_seconds`<br>`execution.workflow_timeout_seconds`<br>`execution.max_runtime_attempts`<br>`execution.scheduler_retry_backoff_seconds` |
| Failure recovery | Fail the call; cancel/invalidate dependent descendants, including completed speculative results; clear invalid outputs and mark task failed. No scheduler reissue or compensation. | `execution.failure_recovery` |

### Table 24: Replay and timed execution

| Setting | Reference value | JSON key(s) |
|---|---|---|
| Replay coverage | 100% of model/tool calls inside the fixed execution graph; planner/search excluded from replay timing and costed separately. Record once, then replay identical call outputs for every scheduler. | `execution.replay.coverage` |
| Chunk timing | Preserve recorded relative chunk times; time scale 1.0 for formal timing. | `execution.replay.preserve_inter_chunk_delays`<br>`execution.replay.time_scale` |
| Replay key | SHA-256 of canonical JSON {workflow_id, call_id, arguments}; sorted keys, UTF-8, compact separators, finite values only. | `execution.replay.key` |
| Replay miss | Fail explicitly. No live service fallback or nearest-argument match. | `execution.replay.miss` |
| Warmup | 1 untimed execution per workflow/method. | `execution.replay.warmup_repetitions` |
| Timed repetitions | 5 | `execution.replay.timed_repetitions` |
| Method order | Seeded per-workflow, per-repetition permutation; runtime-order seeds 31415-31419. | `execution.replay.method_order`<br>`evaluation.runtime_seeds` |
| Cache state | Preload replay bundle before timing; disable model prefix caching; HTTP/model caches do not participate in offline replay. | `execution.replay.cache_state`<br>`execution.replay.prefix_cache` |
| Network state | No live network calls during timed replay; recorded call delays are replayed. | `execution.replay.network_state` |

### Table 25: Metric accounting

| Setting | Reference value | JSON key(s) |
|---|---|---|
| E2E start | Monotonic timestamp when the fixed graph is submitted to the runtime. Report construction time separately. | `metrics.e2e_start` |
| E2E end | Last declared sink completes; report runtime-drain time separately. Invalidated sink output does not count as a successful final answer. | `metrics.e2e_end`<br>`metrics.runtime_drain` |
| TTFO | First externally visible declared sink output; internal producer/tool partial events do not start workflow TTFO. | `metrics.ttfo` |
| Speedup aggregation | Ratio of paired means. Independent latency main table: mean(T_llmorch) / mean(T_method), labeled LLMOrch core reproduction. Mixed internal-study reports use complete_dependency. Read the frozen study/method list; legacy metrics.speedup_baseline does not change latency.py aggregation. | `metrics.speedup`<br>`metrics.speedup_baseline` (mixed study)<br>`paper_experiments.execution_methods` (latency profiles) |
| Latency statistic | Mean, median, P90; paired task-family bootstrap 95% CI, 2,000 resamples, seed 2718. Keep method/seed pairs together. | `metrics.latency_statistics`<br>`evaluation.bootstrap_resamples`<br>`evaluation.bootstrap_seed`<br>`evaluation.confidence_level`<br>`evaluation.bootstrap_unit` |
| Token scope | Shared request journals include query, selector, planner, repairs/fallback, executors, distiller, checker and verification, including rejected/failed attempts with known usage. Target online cost and offline library/search cost are separate; amortization requires the actual offline total and declared target denominator. GAIA logical/physical auxiliary costs are separate. Unknown usage is never zero. | `metrics.token_scope`<br>`metrics.token_reporting` |
| Quality delta | Per-benchmark paired candidate-minus-sequential score x100 percentage points; macro-average benchmark deltas, with native metrics separately reported. Model/contract failures stay in the denominator with score 0; infrastructure failures block publication. | `metrics.quality_delta_baseline`<br>`metrics.quality_units`<br>`metrics.cross_benchmark_aggregation`<br>`evaluation.failures` |
| Unsafe dispatch | Unsafe-dispatch rate uses all dispatched calls as denominator. Eq.(38) separately reports (N_arg + N_dup + N_effect + N_cap)/N_early: one primary category per violating early logical call; priority arg, dup, effect, cap. Publish all four raw counts and N_early. Zero denominator or incomplete audit gives null, not 0%. No assumed observed violation rate. | `metrics.unsafe_dispatch`<br>`metrics.contract_safety`<br>`metrics.early_unsafe_rate` |

## Supplementary controls and implementation scope

| Setting | Reference value or scope |
|---|---|
| A2Flow (Table 2 supplement) | A2FlowAdapter is implemented in baseline_native.py on EvoAgentX AFlow, with bounded source-only operator extraction and canonical export. This is a paper-inspired reimplementation, not the absent author implementation at the recorded upstream commit. Configured reference: 20 rounds, population 4, one candidate/round, three validation seeds, shared budget. It is optional and not selected by the default six-method baseline_runner.required_methods. |
| Shared search budget | Same backbone/decoding, max 12 nodes, 65,536 tokens/task and 2,000,000 search tokens/benchmark/generation-seed. AFlow/SEW: 20 search candidates, selected on all 30 frozen validation tasks x seeds 101/202/303. CompactFlow instead uses six round-fold tasks x those three seeds; these are different validation protocols. |
| Library snapshots | Each evolution run/variant writes initial.json, bootstrap.json, round-01.json through round-05.json, then final.json under policy_snapshots/. Expert Policies reads initial; Static Library reads bootstrap. The final library is shared across included benchmarks and target seeds; no independently trained per-benchmark/seed snapshots are produced. Release actual hashes/evidence; final.json exists only after discovery completes. |
| Main construction comparison | Selected default: 4 benchmarks x 30 target tasks x 3 generation seeds x 6 methods = 2,160 target records. Methods: AFlow, EvoAgentX, Base Planner, Expert Policies, Static Library, CompactFlow. Optional A2Flow makes seven only when explicitly selected in a new locked run. Inherited requested-allocation metadata still lists seven; effective allocation uses baseline_runner.required_methods. |
| Pilot profile | Whole-family 15-task pool per benchmark, drawn only from formal source and split 9/3/3. Generation seed 42; bootstrap 2; one round of 2 source tasks; one candidate/round; all 3 pilot validation tasks with seed 101. Four-benchmark/six-method construction pilot requires 72 target records. Execution-only pilot selects 2 target tasks/benchmark, one warmup plus two timed repetitions: 24 warmup and 48 timed records. |
| Duplicate-dispatch protection | Within one in-memory workflow execution: READY-state check, dispatch_ledger by logical call ID, atomic check/reserve/state transition under one lock. Repeated-readiness regression passed. Legacy duplicate_dispatch counts blocked attempts; Eq.(38) audit counts actual repeated START events. No cross-restart or external-service exactly-once guarantee. |
| Verification scope | Configuration presence and component tests do not certify formal results. Pilot, selected-study and publication status are separate. Completion requires paired records, complete actual usage, task/assets/backend/code identities, frozen policies/workflows and study-specific coverage. Neither pilot nor Lean abstract safety proofs recover the PDF violation counts or supply measured Figures 6/7. |

## Evidence needed for manuscript claims

The manuscript should name the actual LLM and encoder in Section 7.1 and
provide the principal settings in Appendix G. A repository link supplements
that disclosure. This reference file can only describe a measured table or
figure after its values have been checked against the corresponding run records.

A complete reproducibility record for Figure 6 needs measured scheduler
overhead and capacity/materialization sweeps with repetitions, timing units,
hardware, and raw traces. Figure 7 needs round-by-round policy counts,
admit/merge/reject evidence, and the evaluated quality/cost results at each
admission threshold. Illustrative curves cannot supply these measurements.
If the measured artifacts are unavailable, label the figures as illustrations
and remove claims that treat them as experimental evidence, or omit them.

## GAIA extension

[GAIA.md](GAIA.md) specifies the shared six-method tool session, main GPU 7 plus either GPU 0 auxiliaries or the separate CPU/FP32 auxiliary profiles, pinned auxiliary revisions, 12-step / 20-call limits, phase isolation, asset and observation locks, cost accounting, and real pilot acceptance commands. These settings augment the tables above; pilot or tool smoke success is not publication completeness.

## Unified paper and independent execution entrypoints

See [PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) for commands. Mixed studies use
the global discovery/freeze/target gate. An exact execution_main selection routes
to the independent Base Planner capture plus three-scheduler replay flow without
policy evolution or construction search. LLMCompiler uses its vendored native
TaskFetchingUnit; LLMOrch uses the labeled core reproduction; A2Flow is an optional
construction reimplementation. Internal complete_dependency is a distinct
scheduler. Implementation availability does not establish author-code fidelity,
selection in a run, or completed formal coverage.


### GAIA observation-unit contract update

`tools.gaia.workflow_contract_version=gaia_observation_units_v1` enables the registered
primitive tool contracts described in [GAIA.md](GAIA.md#gaia-stream-contract-protocol-gaia_observation_units_v1).
This adds no budget or seed changes: each independent complete observation consumes
one existing tool call; generated field bindings do not bypass asset effect barriers.
GAIA agent answers remain complete-only, while primitive observations support stable
field publication. Contract metadata records developer provenance and unmeasured
semantic annotation accuracy. CPU auxiliary profiles remain FP32, eight threads,
concurrency one; the main model remains on GPU 7. Old frozen runs are not resumed
under the new contract version.
