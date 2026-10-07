"""Validate the fully specified Appendix-G reference protocol separately from v1 smoke configs."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .construction import AdmissionConfig, CostWeights
from .policy import RetrievalConfig, SelectionConfig


class ReproductionConfigError(ValueError):
    pass


def validate_model_export(export: dict, reference: dict) -> None:
    if export.get("schema_version") != 2 or any(export.get(key) != reference[key] for key in ("model", "serving")):
        raise ReproductionConfigError("standalone model configuration differs from reference protocol")


def validate_reproduction_config(config: dict, *, root: Path | None = None) -> dict:
    def require(condition, message):
        if not condition:
            raise ReproductionConfigError(message)
    def walk(value, path="config"):
        require(value is not None, f"{path} is unresolved")
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, path + "." + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif isinstance(value, float):
            require(math.isfinite(value), f"{path} must be finite")
    walk(config)
    required = {"paper", "model", "serving", "encoder", "datasets", "partition", "evaluation", "study_allocation",
                "construction", "execution", "baselines", "ablations", "metrics", "tools", "artifacts", "provenance", "appendix_coverage"}
    require(config.get("schema_version") == 2, "schema_version must be 2")
    require(required <= config.keys(), "missing protocol sections: " + str(sorted(required - config.keys())))
    require(config.get("paper_replication_claim") is False, "reference defaults cannot claim recovery of original settings")
    require(config.get("profile") in {"reference", "pilot"}, "unknown profile")
    require(set(config["appendix_coverage"]) == {str(i) for i in range(9, 26)}, "must cover Appendix G Tables 9-25")
    for paths in config["appendix_coverage"].values():
        for path in paths:
            value = config
            for field in path.split("."):
                require(field in value, "coverage points to missing config: " + path)
                value = value[field]
    model, serving = config["model"], config["serving"]
    require(re.fullmatch(r"[a-f0-9]{40}", model["revision"]) is not None, "model revision must be immutable")
    require(model["repository"] == serving["model_repository"] and model["revision"] == serving["model_revision"], "model/server revision mismatch")
    require(model["name"] == serving["served_model_name"], "served alias mismatch")
    require(model["context_length"] == serving["max_model_len"], "client/server context mismatch")
    require(model["concurrency"] <= serving["max_num_seqs"], "client concurrency exceeds configured serving sequence capacity")
    require(model["base_url"] == f'http://{serving["host"]}:{serving["port"]}/v1', "endpoint and server binding disagree")
    require(serving["host"] == "127.0.0.1", "reference model endpoint must bind loopback")
    require(serving["dtype"] == "bfloat16" and serving["quantization"] == "none", "reference numerical precision changed")
    require(0 < serving["gpu_memory_utilization"] < 1, "invalid GPU memory fraction")
    require(0 < model["top_p"] <= 1 and model["top_k"] >= 1 and model["repetition_penalty"] > 0, "invalid decoding configuration")
    require(set(model["components"]) == {"query", "selector", "planner", "distiller", "checker", "executor"}, "all six model roles must be specified")
    for name, settings in model["components"].items():
        require(settings["model"] == model["name"], f"{name}: shared backbone differs")
        require(0 <= settings["temperature"] <= 2, f"{name}: invalid temperature")
        require(0 < settings["max_tokens"] < model["context_length"], f"{name}: invalid output length")
    encoder = config["encoder"]
    require(re.fullmatch(r"[a-f0-9]{40}", encoder["revision"]) is not None, "encoder must be pinned")
    require(encoder["backend"] == "sentence_transformers" and encoder["normalize_embeddings"] is True, "reference requires normalized semantic encoder")
    require(encoder["dimension"] == 384 and encoder["similarity"] == "cosine", "encoder index mismatch")
    require(len(config["datasets"]) == 4 and {x["name"] for x in config["datasets"]} == {"MBPP", "HotpotQA", "MATH", "GAIA"}, "four benchmark definitions required")
    for data in config["datasets"]:
        require(re.fullmatch(r"[a-f0-9]{40}", data["revision"]) is not None, f'{data["name"]}: unpinned dataset')
        require(data["sample_count"] > 0 and data["configurations"] and data["original_splits"], "empty dataset protocol")
        if data["name"] == "GAIA":
            require(data["original_splits"] == ["validation"], "GAIA reference uses labeled validation only")
    partition = config["partition"]
    require(all(partition[k] > 0 for k in ("source_fraction", "validation_fraction", "target_fraction")), "partition fractions must be positive")
    require(abs(sum(partition[k] for k in ("source_fraction", "validation_fraction", "target_fraction")) - 1) < 1e-10, "partition fractions must sum to one")
    require(sum(partition["nominal_counts_per_benchmark"].values()) == config["datasets"][0]["sample_count"], "nominal partition counts inconsistent")
    c = config["construction"]
    weights = c["retrieval_weights"]
    RetrievalConfig(semantic_top_k0=c["semantic_top_k0"], top_k=c["top_k"], semantic_weight=weights["semantic"],
                    applicability_weight=weights["structural"], utility_weight=weights["historical"], confidence_weight=weights["confidence"],
                    minimum_semantic_score=c["minimum_semantic_score"], minimum_applicability=c["minimum_applicability"])
    SelectionConfig(c["max_policies"], c["minimum_selection_score"], c["minimum_applicability"])
    require(c["max_policies"] <= c["top_k"] <= c["semantic_top_k0"], "require k <= K <= K0")
    require(abs(sum(weights.values()) - 1) < 1e-9, "retrieval weights must sum to one")
    cost = c["cost_weights"]
    AdmissionConfig(quality_tolerance=c["quality_tolerance"], minimum_cost_reduction=c["minimum_cost_reduction"],
                    merge_similarity_threshold=c["merge_similarity_threshold"], minimum_candidate_valid_rate=c["minimum_candidate_valid_rate"],
                    cost_weights=CostWeights(cost["tokens"], cost["latency"], cost["graph"]))
    require(c["admission_order"] == "verify_then_merge" and c["library"]["target_frozen"], "admission and target-freeze invariants required")
    require(c["heldout_tasks_per_candidate"] <= partition["nominal_counts_per_benchmark"]["validation"], "heldout sample exceeds validation partition")
    require(c["library"]["bootstrap_tasks"] + c["distillation"]["rounds"] * c["distillation"]["source_tasks_per_round"] <= partition["nominal_counts_per_benchmark"]["source"], "source task budget exceeded")
    if "runner" in config:
        from .evolution import EvolutionConfig
        runner = EvolutionConfig.from_mapping(config)
        require(runner.validation_folds * runner.validation_tasks_per_candidate == partition["nominal_counts_per_benchmark"]["validation"],
                "validation folds must cover the complete per-benchmark validation pool")
        require(runner.bootstrap_tasks + runner.rounds * runner.source_tasks_per_round <= partition["nominal_counts_per_benchmark"]["source"],
                "runner source budget exceeded")
        require(config["runner"].get("target_policy_updates") is False, "target policy updates must be disabled")
        require(runner.rounds == c["distillation"]["rounds"] and runner.validation_folds == c["validation_folds"]
                and runner.validation_tasks_per_candidate == c["heldout_tasks_per_candidate"], "runner/construction fold settings disagree")
    evaluation, execution = config["evaluation"], config["execution"]
    for name in ("generation_seeds", "runtime_seeds"):
        seeds = evaluation[name]
        require(seeds and len(seeds) == len(set(seeds)) and all(type(s) is int for s in seeds), f"invalid {name}")
    require(len(evaluation["runtime_seeds"]) == evaluation["repetitions"], "one runtime-order seed is required per timed repetition")
    require(execution["replay"]["timed_repetitions"] == evaluation["repetitions"], "replay repetitions disagree")
    require(execution["replay"]["warmup_repetitions"] == evaluation["warmups"], "warmup counts disagree")
    require(execution["resource_capacity"]["external_call"] == execution["external_call_capacity"], "resource capacity aliases disagree")
    require(0 < execution["percentage_threshold"] <= 1 and execution["materialization_batch_size"] >= 1, "invalid materialization protocol")
    require(execution["max_runtime_attempts"] == 1, "reference profile does not claim retry-version/compensation support")
    require(execution["replay"]["time_scale"] == 1 and execution["replay"]["preserve_inter_chunk_delays"], "main replay must preserve recorded timings")
    require(not serving["enable_prefix_caching"], "reference disables prefix caching")
    require(config["metrics"]["speedup"] == "ratio_of_paired_means", "main speedup estimator must be fixed")
    safety = config["metrics"]["contract_safety"]
    from .safety import CATEGORIES
    require(tuple(safety["categories"]) == CATEGORIES, "Eq.38 requires all four contract-failure categories")
    require(tuple(safety["primary_category_priority"]) == CATEGORIES, "primary category precedence differs from the audit implementation")
    require(safety["zero_denominator"] == "undefined; serialize null, not 0.0", "an unobserved early-call cohort cannot be reported as zero violations")
    require("@sha256:" in config["tools"]["mbpp_sandbox"]["image"], "sandbox image must be pinned by digest")
    if config["profile"] == "reference":
        studies = config["study_allocation"]
        d = studies["construction_diagnostic"]
        require(d["benchmarks"] * d["source_tasks_per_benchmark"] * len(d["generation_seeds"]) * len(d["methods"]) == d["unique_generated_workflows"] == 1200, "construction diagnostic count mismatch")
        d = studies["execution_diagnostic"]
        require(d["benchmarks"] * d["target_tasks_per_benchmark"] * len(d["generation_seeds"]) == d["unique_generated_workflows"] == 1000, "execution diagnostic count mismatch")
        require(d["unique_generated_workflows"] * d["scheduler_methods"] * d["timed_repetitions"] == d["timed_replays"], "replay count mismatch")
        d = studies["construction_main"]
        require(d["benchmarks"] * d["target_tasks_per_benchmark"] * len(d["generation_seeds"]) * len(d["requested_methods"]) == d["requested_unique_generated_workflows"], "main construction count mismatch")
    if "baseline_runner" in config:
        b=config["baselines"]; shared=b["shared"]
        require(shared["runtime"]=="complete_dependency", "construction baselines require common complete runtime")
        gaia = config.get("tools",{}).get("gaia",{})
        if gaia:
            from urllib.parse import urlparse
            require(gaia.get("enabled") is True, "GAIA profile requires explicit tool enablement")
            require(gaia.get("mode") in {"capture","replay"}, "invalid GAIA observation mode")
            require(gaia.get("max_agent_steps")==12 and gaia.get("max_tool_calls")==20, "GAIA task tool budgets differ from protocol")
            auxiliary_device = gaia.get("device", "cuda")
            require(auxiliary_device in {"cpu", "cuda"}, "invalid GAIA auxiliary device")
            if auxiliary_device == "cpu":
                require(gaia.get("gpu") is None, "CPU auxiliary profile must not select a GPU")
                require(gaia.get("models",{}).get("vision",{}).get("dtype")=="float32" and gaia.get("models",{}).get("audio",{}).get("compute_type")=="float32", "CPU auxiliary profile requires float32 models")
            else:
                require(gaia.get("gpu")==0, "GAIA auxiliary models require authorized GPU 0")
            require(type(gaia.get("cpu_threads",4)) is int and 1 <= gaia.get("cpu_threads",4) <= 32, "invalid auxiliary CPU thread bound")
            require(urlparse(gaia.get("auxiliary_url","")).hostname=="127.0.0.1", "GAIA service must bind loopback")
            for model in gaia.get("models",{}).values():
                require(re.fullmatch(r"[a-f0-9]{40}",model.get("revision","")) is not None, "auxiliary model revision must be immutable")
            require(set(gaia.get("models",{}))=={"vision","audio"}, "GAIA vision/audio configuration missing")
        require(shared["task_total_token_budget"]==evaluation["task_token_budget"], "baseline task budget differs")
        require(shared["max_nodes"]==c["planner"]["max_nodes"]==12, "baseline node bound differs")
        require(shared["heldout_seeds"]==c["heldout_seeds"], "baseline validation seed aliases differ")
        require(shared["heldout_tasks_per_candidate"]==partition["nominal_counts_per_benchmark"]["validation"], "construction selection requires the frozen validation pool")
        for method in ("aflow","evoagentx"):
            require(b[method]["integration_status"]=="runtime_probe" and b[method].get("adapter_path") and b[method].get("canonical_exporter"), "native baseline adapter unresolved")
            if method != "a2flow":
                require(b[method]["validation_rounds"]==len(c["heldout_seeds"]), "baseline validation rounds differ from seed count")
        if "a2flow" in b and b["a2flow"].get("integration_status") == "runtime_probe":
            require(b["a2flow"].get("adapter_path") and b["a2flow"].get("canonical_exporter"), "A2Flow adapter metadata missing")
        require(b["aflow"]["test_rounds"]==len(evaluation["generation_seeds"]), "AFlow target rounds differ from paired generation seeds")
        require(b["aflow"]["candidates_per_round"]==b["evoagentx"]["candidates_per_iteration"]==1, "one candidate per search step required")
        if config["profile"]=="reference":
            require(b["aflow"]["max_rounds"]==b["evoagentx"]["max_iterations"]==20 and b["aflow"]["population_sample"]==4, "reference construction search budget changed")
    if root is not None:
        required_artifacts = {c["distillation"][k] for k in ("prompt", "schema")}
        required_artifacts.update(c["planner"][k] for k in ("prompt", "output_schema"))
        required_artifacts.add(c["library"]["initial_path"])
        require(required_artifacts <= config["artifacts"]["source_hashes"].keys(), "missing source-artifact hashes")
        for relative, expected in config["artifacts"]["source_hashes"].items():
            path = (root / relative).resolve()
            require(path.is_relative_to(root.resolve()), "source artifact escapes repository")
            require(path.is_file(), "missing source artifact: " + relative)
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "source artifact hash mismatch: " + relative)
        for relative in [c["planner"]["output_schema"], c["distillation"]["schema"]]:
            require((root / relative).is_file(), "missing schema: " + relative)
    return config


def load_reproduction_config(path: str | Path, *, root: Path | None = None) -> dict:
    return validate_reproduction_config(json.loads(Path(path).read_text()), root=root)
