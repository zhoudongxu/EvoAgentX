"""Serve explicitly prepared vision/ASR models on CPU or the authorized GPU."""

from __future__ import annotations
import argparse, asyncio, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_prepared(model):
    root = Path(model["path"])
    lock = json.loads((root / "preparation.lock.json").read_text())
    if (lock["repository"], lock["revision"]) != (
        model["repository"],
        model["revision"],
    ):
        raise ValueError("prepared model identity differs")
    for entry in lock["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != entry["sha256"]:
            raise ValueError("prepared model checksum mismatch: " + str(path))


def configure_device(cfg):
    device = cfg.get("device", "cuda")
    threads = cfg.get("cpu_threads", 4)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("cpu_threads must be an integer in [1, 32]")
    if device == "cpu":
        if cfg.get("gpu") is not None:
            raise ValueError("CPU service must not select a GPU")
        if cfg["models"]["vision"]["dtype"] != "float32" or cfg["models"]["audio"]["compute_type"] != "float32":
            raise ValueError("CPU profile requires explicit float32 vision and audio")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device == "cuda":
        if cfg.get("gpu") != 0 or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("set CUDA_VISIBLE_DEVICES to the authorized GPU 0")
        active = subprocess.check_output(
            ["nvidia-smi", "-i", "0", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            raise RuntimeError("auxiliary GPU already has compute processes; refusing to displace them")
    else:
        raise ValueError("auxiliary device must be cpu or cuda")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    return device, threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8020)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())["tools"]["gaia"]
    device, threads = configure_device(cfg)
    # CTranslate2 loads CUDA libraries at runtime. Include explicitly installed
    # wheel libraries; never install anything or change the primary serving env.
    libraries = []
    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if root.is_dir():
            libraries.extend(str(p) for p in sorted(root.glob("*/lib")) if p.is_dir())
    desired = ":".join(
        dict.fromkeys(
            libraries
            + [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
        )
    )
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "false")
    for model in cfg["models"].values():
        verify_prepared(model)
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    from fastapi import FastAPI, HTTPException
    import uvicorn

    vision = cfg["models"]["vision"]
    processor = AutoProcessor.from_pretrained(
        vision["path"],
        local_files_only=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vision["path"],
        local_files_only=True,
        dtype=getattr(torch, vision["dtype"]),
        device_map={"": "cpu" if device == "cpu" else "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    # The parent serializes all inference. A separate process prevents the two
    # CUDA runtimes from loading incompatible global libraries into one process.
    asr = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(Path(__file__).with_name("gaia_audio_worker.py")),
            "--config",
            str(args.config.resolve()),
            "--asset-root",
            str(args.asset_root.resolve()),
        ],
        env=dict(os.environ, **({"LD_LIBRARY_PATH": desired} if device == "cuda" else {})),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    import atexit

    atexit.register(lambda: asr.terminate() if asr.poll() is None else None)
    ready = asr.stdout.readline()
    if not ready or json.loads(ready).get("status") != "ready":
        raise RuntimeError("isolated audio worker failed to load")
    app = FastAPI()
    gate = asyncio.Lock()
    root = args.asset_root.resolve()
    packages = {
        x: importlib.metadata.version(x)
        for x in [
            "torch",
            "transformers",
            "faster-whisper",
            "ctranslate2",
            "qwen-vl-utils",
        ]
    }

    def asset(payload, extensions):
        p = Path(payload["path"]).resolve()
        if (
            not p.is_relative_to(root)
            or p.suffix.lower() not in extensions
            or not p.is_file()
        ):
            raise ValueError("unapproved auxiliary asset")
        if p.stat().st_size > cfg["max_asset_bytes"] or sha(p) != payload["sha256"]:
            raise ValueError("auxiliary asset checksum/size mismatch")
        return p

    def inputs(payload):
        path = asset(payload, {".png", ".jpg", ".jpeg"})
        image = Image.open(path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(path)},
                    {"type": "text", "text": payload["question"]},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(text=[text], images=[image], padding=True, return_tensors="pt")

    @app.get("/health")
    async def health():
        return {
            "status": "ready" if asr.poll() is None else "incomplete",
            "models": cfg["models"],
            "device": device,
            "gpu": cfg.get("gpu"),
            "cpu_threads": threads,
            "vision_device": str(vlm.device),
            "vision_dtype": str(vlm.dtype),
            "packages": packages,
        }

    @app.post("/estimate")
    async def estimate(payload: dict):
        async with gate:
            try:
                value = await asyncio.to_thread(inputs, payload)
                return {"prompt_tokens": int(value["input_ids"].shape[-1])}
            except Exception as exc:
                raise HTTPException(400, str(exc)) from exc

    def infer_vision(payload):
        value = inputs(payload).to(vlm.device)
        n = int(value["input_ids"].shape[-1])
        limit = int(payload.get("max_tokens", 1024))
        if not 1 <= limit <= cfg["vision_max_tokens"] or n + limit > 8192:
            raise ValueError("vision context/output bound exceeded")
        with torch.inference_mode():
            result = vlm.generate(**value, max_new_tokens=limit, do_sample=False)
        generated = result[0, n:]
        count = int(generated.shape[-1])
        text = processor.decode(generated, skip_special_tokens=True)
        return {
            "text": text,
            "_usage": {
                "prompt_tokens": n,
                "completion_tokens": count,
                "total_tokens": n + count,
                "usage_complete": True,
            },
        }

    @app.post("/vision")
    async def vision_request(payload: dict):
        async with gate:
            try:
                return await asyncio.to_thread(infer_vision, payload)
            except Exception as exc:
                raise HTTPException(503, str(exc)) from exc

    def infer_audio(payload):
        asset(payload, {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".webm"})
        if asr.poll() is not None:
            raise RuntimeError("audio worker exited; usage unavailable")
        asr.stdin.write(json.dumps(payload) + "\n")
        asr.stdin.flush()
        line = asr.stdout.readline()
        if not line:
            raise RuntimeError("audio worker exited without a measured response")
        value = json.loads(line)
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    @app.post("/transcribe")
    async def audio_request(payload: dict):
        async with gate:
            try:
                return await asyncio.to_thread(infer_audio, payload)
            except Exception as exc:
                raise HTTPException(503, str(exc)) from exc

    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
