#!/usr/bin/env python3
"""Download WikiText from Hugging Face and export a plain UTF-8 text corpus.

TorchLean's GPT-style examples consume ordinary text files rather
than depending on Python dataset libraries at runtime.  This script is a small
data-prep helper: it uses the Hugging Face Dataset Viewer metadata to locate the
parquet shards for `Salesforce/wikitext`, reads the `text` column, and writes a
single newline-separated file under `data/real/text/`.

Recommended quick run:

    python3 scripts/datasets/download_wikitext.py \
      --config wikitext-2-raw-v1 \
      --split train \
      --output data/real/text/wikitext2_train.txt

Recommended larger run:

    python3 scripts/datasets/download_wikitext.py \
      --config wikitext-103-raw-v1 \
      --split train \
      --max-bytes 120000000 \
      --output data/real/text/wikitext103_train_120mb.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq


DATASET = "Salesforce/wikitext"
DATASET_SERVER = "https://datasets-server.huggingface.co"
LICENSE_NOTE = "WikiText license: CC BY-SA 3.0 / GFDL (see Hugging Face dataset card)."
DEFAULT_TIMEOUT_SECONDS = 60.0


def require_https(url: str) -> None:
    """Reject non-HTTPS URLs before querying metadata or downloading shards."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise SystemExit(f"refusing non-https URL: {url}")


def fetch_json(url: str) -> dict:
    """Fetch and decode one JSON response from the Hugging Face Dataset Viewer."""
    require_https(url)
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, path: Path) -> None:
    """Download a parquet shard unless a non-empty cached copy already exists."""
    require_https(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"[download] {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=DEFAULT_TIMEOUT_SECONDS) as response, path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def parquet_files(config: str, split: str) -> list[dict]:
    """Return Dataset Viewer parquet metadata for one WikiText config and split."""
    url = f"{DATASET_SERVER}/parquet?dataset={DATASET}"
    files = fetch_json(url)["parquet_files"]
    selected = [
        file
        for file in files
        if file["config"] == config and file["split"] == split
    ]
    if not selected:
        raise SystemExit(f"no parquet files found for config={config!r}, split={split!r}")
    return selected


def export_text(files: list[dict], cache_dir: Path, output: Path, max_bytes: int | None) -> int:
    """Concatenate WikiText parquet rows into one UTF-8 corpus file."""
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as out:
        out.write("# Source: Hugging Face dataset Salesforce/wikitext\n")
        out.write(f"# {LICENSE_NOTE}\n\n")
        for file in files:
            shard = cache_dir / file["config"] / file["split"] / file["filename"]
            download(file["url"], shard)
            table = pq.read_table(shard, columns=["text"])
            for maybe_text in table.column("text").to_pylist():
                if not maybe_text:
                    continue
                text = str(maybe_text)
                if max_bytes is not None:
                    remaining = max_bytes - written
                    if remaining <= 0:
                        return written
                    encoded = (text + "\n").encode("utf-8")
                    if len(encoded) > remaining:
                        # Byte caps can split a UTF-8 codepoint. Decode with
                        # `ignore` and return immediately so the output stays
                        # valid UTF-8 and respects the requested cap.
                        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
                        out.write(chunk)
                        written += len(chunk.encode("utf-8"))
                        return written
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
                written += len((text if text.endswith("\n") else text + "\n").encode("utf-8"))
    return written


def main() -> int:
    """CLI entry point for WikiText export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/real/text/wikitext2_train.txt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/real/hf_cache/wikitext"))
    parser.add_argument("--max-bytes", type=int, default=None)
    args = parser.parse_args()

    files = parquet_files(args.config, args.split)
    print(f"[dataset] {DATASET} config={args.config} split={args.split}", file=sys.stderr)
    print(f"[dataset] shards={len(files)} {LICENSE_NOTE}", file=sys.stderr)
    written = export_text(files, args.cache_dir, args.output, args.max_bytes)
    print(f"[output] wrote {written} bytes to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
