"""Explicit GAIA preparation, preflight and independent live-tool acceptance.

No experiment runner calls this script or downloads models implicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path


def prepare_models(config):
    from huggingface_hub import snapshot_download
    from evoagentx.compactflow.gaia import file_sha
    from evoagentx.compactflow.evolution import atomic_json

    for model in config["tools"]["gaia"]["models"].values():
        root = Path(model["path"])
        if (root / "preparation.lock.json").exists():
            from examples.compactflow.serve_gaia_tools import verify_prepared

            verify_prepared(model)
            continue
        snapshot_download(
            repo_id=model["repository"],
            revision=model["revision"],
            local_dir=str(root),
            allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
        )
        files = [p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts]
        atomic_json(
            root / "preparation.lock.json",
            {
                "repository": model["repository"],
                "revision": model["revision"],
                "files": [
                    {
                        "path": str(p.relative_to(root)),
                        "sha256": file_sha(p),
                        "bytes": p.stat().st_size,
                    }
                    for p in sorted(files)
                ],
            },
        )


async def smoke(config, output, audio_fixture=None):
    from evoagentx.compactflow.gaia import (
        GaiaToolSession,
        capability_report,
        attach_assets,
    )
    from evoagentx.compactflow.baseline_native import ModelSession
    from evoagentx.compactflow.baseline_controls import SessionClient
    from evoagentx.compactflow.benchmarks import BenchmarkTask
    from evoagentx.compactflow.evolution import atomic_json
    from PIL import Image, ImageDraw
    import copy
    import shutil

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    raw = output / "synthetic-fixtures"
    raw.mkdir(exist_ok=True)
    cfg["tools"]["gaia"]["asset_root"] = str(raw)
    cfg["tools"]["gaia"]["replay_time_scale"] = 0
    img = Image.new("RGB", (500, 300), "white")
    ImageDraw.Draw(img).rectangle((100, 50, 400, 250), fill="red")
    img.save(raw / "red-rectangle.png")
    img.save(raw / "page.pdf", "PDF")
    (raw / "numbers.csv").write_text("name,value\na,2\nb,3\n")
    names = ["red-rectangle.png", "page.pdf", "numbers.csv"]
    if audio_fixture:
        dest = raw / ("speech" + audio_fixture.suffix)
        shutil.copyfile(audio_fixture, dest)
        names.append(dest.name)
    task = BenchmarkTask(
        "GAIA",
        "independent-tool-smoke",
        "Independent infrastructure fixture.",
        "",
        "independent-tool-smoke",
        attachments=names,
        split="source",
    )
    attach_assets([task], raw)
    owner = ModelSession(cfg, 42, output / "main-model")
    session = GaiaToolSession(
        cfg,
        task.public_input(),
        output=output,
        workspace=output / "call",
        seed=42,
        model_session=owner,
    )
    assets = {a["name"]: a["asset_id"] for a in task.public_input()["assets"]}
    results = {"capabilities": capability_report(config), "tools": {}}

    async def check(name, tool, args):
        started = time.monotonic()
        try:
            value = await session.invoke(tool, args)
            if value.get("error"):
                raise RuntimeError(str(value["error"]))
            results["tools"][name] = {
                "status": "complete",
                "result": value,
                "seconds": time.monotonic() - started,
            }
        except Exception as exc:
            results["tools"][name] = {
                "status": "incomplete",
                "error": type(exc).__name__ + ": " + str(exc),
            }
        atomic_json(output / "smoke.json", results)

    try:
        text = await SessionClient(owner).text(
            "Reply with OK.", "Service acceptance check.", component="checker", seed=42
        )
        results["main_model"] = {
            "status": "complete",
            "response": text,
            "usage": owner.accounting(),
        }
    except Exception as exc:
        results["main_model"] = {"status": "incomplete", "error": str(exc)}
    await check("spreadsheet", "read_file", {"asset_id": assets["numbers.csv"]})
    await check("pdf", "read_file", {"asset_id": assets["page.pdf"], "page": 1})
    await check(
        "vision",
        "vision",
        {
            "asset_id": assets["red-rectangle.png"],
            "question": "What color is the rectangle? Answer with one color.",
        },
    )
    if audio_fixture:
        await check("audio", "transcribe", {"asset_id": assets[dest.name]})
    else:
        results["tools"]["audio"] = {
            "status": "incomplete",
            "error": "supply an independent speech fixture with --audio-fixture",
        }
    await check(
        "python",
        "python",
        {"code": 'import os\nprint(sum([2,3]))\nprint(sorted(os.listdir("/assets")))'},
    )
    await check("browser", "browse", {"url": "https://example.com"})
    await check(
        "search",
        "search",
        {"query": "Python programming language official documentation"},
    )
    results["usage"] = owner.accounting()
    results["tool_usage"] = session.accounting()
    results["status"] = (
        "complete"
        if all(
            r["status"] == "complete"
            for r in [
                results["capabilities"],
                results["main_model"],
                *results["tools"].values(),
            ]
        )
        else "incomplete"
    )
    atomic_json(output / "smoke.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["models", "preflight", "smoke"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audio-fixture", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.command == "models":
        prepare_models(cfg)
        return 0
    if args.command == "preflight":
        from evoagentx.compactflow.gaia import capability_report

        result = capability_report(cfg)
    else:
        if args.output is None:
            parser.error("smoke requires a new --output directory")
        result = asyncio.run(smoke(cfg, args.output, args.audio_fixture))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
