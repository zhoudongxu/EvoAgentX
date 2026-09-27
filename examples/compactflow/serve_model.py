"""Launch or probe the Qwen3-Coder server from the versioned model configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import urllib.request
from pathlib import Path


def server_command(config: dict, model_path: str, python: str) -> list[str]:
    server = config["serving"]
    result = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model_path]
    mapping = {"served_model_name": "served-model-name", "host": "host", "port": "port", "dtype": "dtype",
               "tensor_parallel_size": "tensor-parallel-size", "pipeline_parallel_size": "pipeline-parallel-size",
               "gpu_memory_utilization": "gpu-memory-utilization", "max_model_len": "max-model-len",
               "max_num_seqs": "max-num-seqs", "max_num_batched_tokens": "max-num-batched-tokens", "seed": "seed"}
    for key, flag in mapping.items():
        result.extend(["--" + flag, str(server[key])])
    if not Path(model_path).is_dir():
        result.extend(["--revision", server["model_revision"]])
    for key in ("enforce_eager", "enable_prefix_caching", "enable_chunked_prefill"):
        result.append("--" + ("" if server[key] else "no-") + key.replace("_", "-"))
    if server["quantization"] != "none":
        result.extend(["--quantization", server["quantization"]])
    return result


def check_local_identity(config: dict, model_path: str) -> dict:
    path = Path(model_path)
    if not path.is_dir():
        return {"source": "hub_pinned_revision", "revision": config["model"]["revision"]}
    hashes = {}
    for filename, key in (("config.json", "config_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256")):
        actual = hashlib.sha256((path / filename).read_bytes()).hexdigest()
        if actual != config["model"][key]:
            raise ValueError(f"local model identity mismatch: {filename}")
        hashes[filename] = actual
    return {"source": "local_model_directory", "configuration_hashes": hashes,
            "weight_hashes_verified": False}


def probe(config: dict) -> dict:
    model = config["model"]
    base = os.environ.get("COMPACTFLOW_API_BASE", model["base_url"]).rstrip("/")
    headers = {"Content-Type": "application/json"}
    if os.environ.get(model["api_key_env"]):
        headers["Authorization"] = "Bearer " + os.environ[model["api_key_env"]]
    request = urllib.request.Request(base + "/models", headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        models = json.load(response)
    if model["name"] not in {item["id"] for item in models["data"]}:
        raise ValueError("configured model alias is unavailable")
    payload = {"model": model["name"], "messages": [{"role": "user", "content": "Reply with only the number 2."}],
               "temperature": 0, "max_tokens": 16, "top_p": model["top_p"], "top_k": model["top_k"],
               "repetition_penalty": model["repetition_penalty"], "seed": config["serving"]["seed"]}
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=model["timeout_seconds"]) as response:
        result = json.load(response)
    if not result.get("usage") or result["usage"].get("completion_tokens", 0) <= 0:
        raise ValueError("server did not return usable token accounting")
    return {"status": "ok", "model": result["model"], "output": result["choices"][0]["message"]["content"],
            "usage": result["usage"], "scope": "model connectivity and generation only; not benchmark evaluation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/qwen3_coder_a100.model.json")
    parser.add_argument("--model-path", default=os.environ.get("COMPACTFLOW_MODEL_PATH"))
    parser.add_argument("--python", default=sys.executable)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--launch", action="store_true")
    action.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.preflight:
        print(json.dumps(probe(config), indent=2))
        return
    model_path = args.model_path or config["serving"]["model_path_default"]
    identity = check_local_identity(config, model_path)
    command = server_command(config, model_path, args.python)
    if args.launch:
        env = os.environ.copy()
        env["PATH"] = str(Path(args.python).resolve().parent) + os.pathsep + env.get("PATH", "")
        os.execvpe(args.python, command, env)
    print(json.dumps({"command": shlex.join(command), "model_identity": identity}, indent=2))


if __name__ == "__main__":
    main()
