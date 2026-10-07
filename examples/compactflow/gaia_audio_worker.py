"""Private serial ASR worker; isolates CTranslate2 from PyTorch CUDA libraries."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())["tools"]["gaia"]
    root = args.asset_root.resolve()
    device = cfg.get("device", "cuda")
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from faster_whisper import WhisperModel

    asr = WhisperModel(
        cfg["models"]["audio"]["path"],
        device=device,
        device_index=0,
        compute_type=cfg["models"]["audio"]["compute_type"],
        local_files_only=True,
        cpu_threads=cfg.get("cpu_threads", 4),
        num_workers=1,
    )
    print(json.dumps({"status": "ready"}), flush=True)
    for line in sys.stdin:
        try:
            value = json.loads(line)
            path = Path(value["path"]).resolve()
            if not path.is_relative_to(root) or path.suffix.lower() not in {
                ".mp3",
                ".wav",
                ".m4a",
                ".ogg",
                ".flac",
                ".mp4",
                ".webm",
            }:
                raise ValueError("unapproved audio asset")
            if (
                path.stat().st_size > cfg["max_asset_bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]
            ):
                raise ValueError("audio checksum/size mismatch")
            segments, info = asr.transcribe(
                str(path),
                beam_size=5,
                temperature=0,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            rows = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            result = {
                "text": " ".join(s["text"] for s in rows),
                "segments": rows,
                "language": info.language,
                "_usage": {
                    "total_tokens": 0,
                    "audio_seconds": info.duration,
                    "usage_complete": True,
                },
            }
        except Exception as exc:
            result = {"error": type(exc).__name__ + ": " + str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
