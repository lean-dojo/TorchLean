#!/usr/bin/env python3
"""Save recent CI timings for the static website, without browser-side API requests."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


REPOSITORY = "lean-dojo/TorchLean"
API = f"https://api.github.com/repos/{REPOSITORY}"
OUTPUT = Path(__file__).resolve().parents[2] / "home_page/assets/ci-timings.json"


def request_json(url: str) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "TorchLean-website",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urlopen(Request(url, headers=headers), timeout=15) as response:
        return json.load(response)


def collect() -> dict:
    data = request_json(
        f"{API}/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=8"
    )
    runs = []
    for run in data["workflow_runs"]:
        jobs = request_json(f"{API}/actions/runs/{run['id']}/jobs?filter=latest&per_page=100")
        job = next((job for job in jobs["jobs"]
                    if job["name"] in {"CPU build and checks", "build_and_test"}
                    and job["conclusion"] == "success"), None)
        if job is None:
            continue
        runs.append({
            "id": run["id"],
            "head_sha": run["head_sha"],
            "display_title": run["display_title"],
            "html_url": run["html_url"],
            "run_started_at": run["run_started_at"],
            "job": {
                "started_at": job["started_at"],
                "completed_at": job["completed_at"],
                "steps": [
                    {key: step[key] for key in ("name", "started_at", "completed_at")}
                    for step in job["steps"] if step["conclusion"] == "success"
                ],
            },
        })
    if not runs:
        raise ValueError("no successful CPU CI jobs were found")
    return {
        "repository": REPOSITORY,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output
    try:
        snapshot = collect()
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
        # A network failure must not replace the last usable snapshot with empty data.
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
            if (previous["repository"] == REPOSITORY
                    and isinstance(previous["runs"], list) and previous["runs"]
                    and datetime.fromisoformat(previous["generatedAt"])):
                print(f"CI timings: keeping the saved snapshot ({error}).", file=sys.stderr)
                return 0
        except (OSError, ValueError, KeyError, TypeError):
            pass
        print(f"CI timings: no snapshot available ({error}).", file=sys.stderr)
        return 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    print(f"CI timings: saved {len(snapshot['runs'])} runs to {output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
