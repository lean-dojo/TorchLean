#!/usr/bin/env python3
"""Download and prepare real datasets used by TorchLean examples.

The Lean examples stay offline and deterministic by default.  This script is the opt-in bridge for
people who want the same examples to run on real public datasets.

Outputs:
  data/real/cifar10/cifar10_train_X.npy
  data/real/cifar10/cifar10_train_y.npy
  data/real/cifar10/cifar10_test_X.npy
  data/real/cifar10/cifar10_test_y.npy
  data/real/household_power/household_power_X.npy
  data/real/household_power/household_power_Y.npy
  data/real/auto_mpg/auto_mpg.csv
  data/real/text/tiny_shakespeare.txt
  data/real/text/tinystories_valid.txt

Only stdlib + NumPy are required.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import tarfile
import urllib.request
import warnings
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
TINYSTORIES_VALID_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
)
HOUSEHOLD_POWER_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
AUTO_MPG_URL = "https://archive.ics.uci.edu/static/public/9/auto+mpg.zip"
DEFAULT_TIMEOUT_SECONDS = 60.0


def require_https(url: str) -> None:
    """Reject non-HTTPS dataset URLs before any network request is made."""
    scheme = urlparse(url).scheme
    if scheme != "https":
        raise SystemExit(f"refusing non-https URL: {url}")


def download(
    url: str,
    out: Path,
    *,
    md5: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Path:
    """Download `url` to `out`, reusing a cached copy when the checksum matches."""
    require_https(url)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (md5 is None or file_md5(out) == md5):
        print(f"[skip] {out}")
        return out
    print(f"[download] {url}")
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    out.write_bytes(data)
    if md5 is not None:
        got = hashlib.md5(data).hexdigest()
        if got != md5:
            out.unlink(missing_ok=True)
            raise SystemExit(f"md5 mismatch for {out}: expected {md5}, got {got}")
    print(f"[ok] {out}")
    return out


def file_md5(path: Path) -> str:
    """Compute an MD5 digest by streaming the file in chunks."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-10 batch as normalized NCHW images and float labels."""
    with path.open("rb") as f:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            obj = pickle.load(f, encoding="bytes")
    x = obj[b"data"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    labels = obj.get(b"labels", obj.get(b"fine_labels"))
    y = np.asarray(labels, dtype="float32")
    return x, y


def prepare_cifar10(root: Path, *, limit_train: int | None, limit_test: int | None) -> None:
    """Download CIFAR-10, extract it, and export train/test `.npy` tensors."""
    raw_dir = root / "raw"
    out_dir = root / "cifar10"
    archive = download(CIFAR10_URL, raw_dir / "cifar-10-python.tar.gz", md5=CIFAR10_MD5)
    extract_root = raw_dir / "cifar-10-batches-py"
    if not extract_root.exists():
        print(f"[extract] {archive}")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(raw_dir, filter="data")

    train_xs: list[np.ndarray] = []
    train_ys: list[np.ndarray] = []
    for i in range(1, 6):
        x, y = load_cifar_batch(extract_root / f"data_batch_{i}")
        train_xs.append(x)
        train_ys.append(y)
    train_x = np.concatenate(train_xs, axis=0)
    train_y = np.concatenate(train_ys, axis=0)
    test_x, test_y = load_cifar_batch(extract_root / "test_batch")

    if limit_train is not None:
        train_x = train_x[:limit_train]
        train_y = train_y[:limit_train]
    if limit_test is not None:
        test_x = test_x[:limit_test]
        test_y = test_y[:limit_test]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cifar10_train_X.npy", train_x)
    np.save(out_dir / "cifar10_train_y.npy", train_y)
    np.save(out_dir / "cifar10_test_X.npy", test_x)
    np.save(out_dir / "cifar10_test_y.npy", test_y)
    print(f"[write] {out_dir}/cifar10_train_X.npy {train_x.shape}")
    print(f"[write] {out_dir}/cifar10_train_y.npy {train_y.shape}")
    print(f"[write] {out_dir}/cifar10_test_X.npy {test_x.shape}")
    print(f"[write] {out_dir}/cifar10_test_y.npy {test_y.shape}")


def prepare_tiny_shakespeare(root: Path) -> None:
    """Download the Tiny Shakespeare text corpus used by sequence examples."""
    download(TINY_SHAKESPEARE_URL, root / "text" / "tiny_shakespeare.txt")


def prepare_tinystories_valid(root: Path) -> None:
    """Download the TinyStories validation corpus used by text examples."""
    download(TINYSTORIES_VALID_URL, root / "text" / "tinystories_valid.txt")


def prepare_household_power(root: Path, *, windows: int, stride: int) -> None:
    """Prepare UCI household power as next-hour forecasting windows.

    Source:
      Hebrail and Berard, Individual Household Electric Power Consumption,
      UCI Machine Learning Repository, https://doi.org/10.24432/C58K54
      License: CC BY 4.0.

    The raw file is minute-level and contains missing values. The LSTM tutorial uses averaged valid
    `Global_active_power` values into hourly readings, normalize them to `[0, 1]`, and export
    `(24, 1) -> (24, 1)` one-step-shifted windows.
    """

    if windows <= 0:
        raise SystemExit("--household-power-windows must be > 0")
    if stride <= 0:
        raise SystemExit("--household-power-stride must be > 0")

    raw_dir = root / "raw"
    out_dir = root / "household_power"
    archive = download(HOUSEHOLD_POWER_URL, raw_dir / "household_power_consumption.zip")

    hourly: list[float] = []
    bucket_sum = 0.0
    bucket_count = 0
    minute_in_bucket = 0
    last_value = 0.0
    needed_hours = (windows - 1) * stride + 25

    with zipfile.ZipFile(archive) as zf:
        txt_name = next(
            (name for name in zf.namelist() if name.endswith("household_power_consumption.txt")),
            None,
        )
        if txt_name is None:
            raise SystemExit(f"could not find household_power_consumption.txt in {archive}")

        with zf.open(txt_name) as raw:
            header = raw.readline().decode("utf-8", errors="replace").strip().split(";")
            try:
                power_idx = header.index("Global_active_power")
            except ValueError as exc:
                raise SystemExit("raw household power file is missing Global_active_power") from exc

            for line_b in raw:
                parts = line_b.decode("utf-8", errors="replace").strip().split(";")
                if len(parts) <= power_idx:
                    continue

                value = parts[power_idx]
                if value and value != "?":
                    try:
                        last_value = float(value)
                        bucket_sum += last_value
                        bucket_count += 1
                    except ValueError:
                        pass

                minute_in_bucket += 1
                if minute_in_bucket == 60:
                    hourly.append(bucket_sum / bucket_count if bucket_count else last_value)
                    bucket_sum = 0.0
                    bucket_count = 0
                    minute_in_bucket = 0
                    if len(hourly) >= needed_hours:
                        break

    if len(hourly) < needed_hours:
        raise SystemExit(f"household power file only produced {len(hourly)} hourly values")

    series = np.asarray(hourly, dtype=np.float32)
    lo = float(series.min())
    hi = float(series.max())
    denom = hi - lo if hi > lo else 1.0
    series = (series - lo) / denom

    X = np.zeros((windows, 24, 1), dtype=np.float32)
    Y = np.zeros((windows, 24, 1), dtype=np.float32)
    for i in range(windows):
        start = i * stride
        X[i, :, 0] = series[start : start + 24]
        Y[i, :, 0] = series[start + 1 : start + 25]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "household_power_X.npy", X)
    np.save(out_dir / "household_power_Y.npy", Y)
    print(f"[write] {out_dir}/household_power_X.npy {X.shape}")
    print(f"[write] {out_dir}/household_power_Y.npy {Y.shape}")
    print("[source] UCI Individual Household Electric Power Consumption, CC BY 4.0")


def prepare_auto_mpg(root: Path) -> None:
    """Prepare UCI Auto MPG as a normalized tabular regression CSV.

    Source:
      Quinlan, Auto MPG, UCI Machine Learning Repository, https://doi.org/10.24432/C5859H
      License: CC BY 4.0.

    The raw file has rows:
      mpg cylinders displacement horsepower weight acceleration model_year origin car_name

    Rows with missing horsepower are dropped, then the seven numeric predictors and target are normalized to
    `[0, 1]`, and write columns `x1..x7,y` for the Lean MLP tutorial.
    """

    raw_dir = root / "raw"
    out_dir = root / "auto_mpg"
    archive = download(AUTO_MPG_URL, raw_dir / "auto_mpg.zip")

    rows: list[list[float]] = []
    with zipfile.ZipFile(archive) as zf:
        data_name = next((name for name in zf.namelist() if name.endswith("auto-mpg.data")), None)
        if data_name is None:
            raise SystemExit(f"could not find auto-mpg.data in {archive}")
        with zf.open(data_name) as raw:
            for line_b in raw:
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) < 8 or fields[3] == "?":
                    continue
                mpg = float(fields[0])
                features = [
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                    float(fields[4]),
                    float(fields[5]),
                    float(fields[6]),
                    float(fields[7]),
                ]
                rows.append(features + [mpg])

    arr = np.asarray(rows, dtype=np.float32)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    denom = np.where(hi > lo, hi - lo, 1.0)
    arr = (arr - lo) / denom

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "auto_mpg.csv"
    header = "x1,x2,x3,x4,x5,x6,x7,y"
    np.savetxt(out, arr, delimiter=",", header=header, comments="", fmt="%.8f")
    print(f"[write] {out} rows={arr.shape[0]} cols={arr.shape[1]}")
    print("[source] UCI Auto MPG, CC BY 4.0")


def main() -> None:
    """CLI entry point selecting which public example datasets to prepare."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/real"))
    ap.add_argument("--cifar10", action="store_true", help="download and prepare CIFAR-10")
    ap.add_argument(
        "--household-power",
        action="store_true",
        help="download UCI household power and prepare LSTM regression windows",
    )
    ap.add_argument("--auto-mpg", action="store_true", help="download UCI Auto MPG tabular regression")
    ap.add_argument("--tiny-shakespeare", action="store_true", help="download tiny-shakespeare")
    ap.add_argument("--tinystories-valid", action="store_true", help="download TinyStories valid split")
    ap.add_argument(
        "--all",
        action="store_true",
        help="prepare CIFAR-10, household power, Auto MPG, tiny-shakespeare, and TinyStories valid",
    )
    ap.add_argument(
        "--cifar10-limit-train",
        type=int,
        default=200,
        help="number of CIFAR-10 train images to export; use -1 for all 50000",
    )
    ap.add_argument(
        "--cifar10-limit-test",
        type=int,
        default=100,
        help="number of CIFAR-10 test images to export; use -1 for all 10000",
    )
    ap.add_argument(
        "--household-power-windows",
        type=int,
        default=512,
        help="number of 24-hour forecasting windows to export",
    )
    ap.add_argument(
        "--household-power-stride",
        type=int,
        default=3,
        help="hour stride between exported household-power windows",
    )
    args = ap.parse_args()

    root: Path = args.root
    if not any(
        [
            args.all,
            args.cifar10,
            args.household_power,
            args.auto_mpg,
            args.tiny_shakespeare,
            args.tinystories_valid,
        ]
    ):
        ap.error("choose at least one dataset flag, e.g. --cifar10 or --all")

    if args.all or args.cifar10:
        prepare_cifar10(
            root,
            limit_train=None if args.cifar10_limit_train < 0 else args.cifar10_limit_train,
            limit_test=None if args.cifar10_limit_test < 0 else args.cifar10_limit_test,
        )
    if args.all or args.household_power:
        prepare_household_power(
            root,
            windows=args.household_power_windows,
            stride=args.household_power_stride,
        )
    if args.all or args.auto_mpg:
        prepare_auto_mpg(root)
    if args.all or args.tiny_shakespeare:
        prepare_tiny_shakespeare(root)
    if args.all or args.tinystories_valid:
        prepare_tinystories_valid(root)


if __name__ == "__main__":
    main()
