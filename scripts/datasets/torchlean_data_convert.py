#!/usr/bin/env python3
"""Convert common dataset artifacts into TorchLean's canonical tensor files.

TorchLean's Lean-side loaders stay small and deterministic:
they read numeric CSV tables and NumPy `.npy` tensors.  This script is the
interop bridge for the other common artifacts people keep on disk:

* NumPy `.npy` / `.npz`
* MATLAB `.mat` files, when SciPy is installed
* PyTorch `.pt` / `.pth` tensors or dictionaries, when PyTorch is installed
* numeric CSV tables
* image folders, when Pillow is installed

The output is always `.npy`, optionally accompanied by a small JSON manifest.
That keeps TorchLean examples and training code simple:

    python3 scripts/datasets/torchlean_data_convert.py tensor --input data.pt --key x --output X.npy
    lake exe torchlean cnn --cuda --x X.npy --y y.npy --n-total 1000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def die(msg: str) -> None:
    """Exit with a consistent converter error prefix."""
    raise SystemExit(f"error: {msg}")


def select_key(obj: Any, key: str | None, *, source: Path) -> Any:
    """Select an array-like payload from a keyed object, requiring `--key` when ambiguous."""
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.files)
        if key is None:
            if len(keys) != 1:
                die(f"{source}: choose --key; available keys: {keys}")
            key = keys[0]
        if key not in obj.files:
            die(f"{source}: key {key!r} not found; available keys: {keys}")
        return obj[key]

    if isinstance(obj, dict):
        keys = [str(k) for k in obj.keys()]
        if key is None:
            public = [k for k in keys if not k.startswith("__")]
            if len(public) != 1:
                die(f"{source}: choose --key; available keys: {public}")
            key = public[0]
        if key not in obj:
            die(f"{source}: key {key!r} not found; available keys: {keys}")
        return obj[key]

    if key is not None:
        die(f"{source}: --key was provided, but the loaded object is not keyed")
    return obj


def to_numpy(x: Any, *, source: Path) -> np.ndarray:
    """Convert common tensor-like objects into a NumPy array."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    try:
        return np.asarray(x)
    except Exception as exc:
        die(f"{source}: could not convert object of type {type(x).__name__} to numpy: {exc}")


def cast_array(arr: np.ndarray, dtype: str) -> np.ndarray:
    """Return a contiguous array, casting when a dtype is requested."""
    if dtype == "preserve":
        return np.ascontiguousarray(arr)
    try:
        return np.ascontiguousarray(arr.astype(dtype, copy=False))
    except TypeError as exc:
        die(f"unsupported dtype {dtype!r}: {exc}")


def write_manifest(out: Path, arr: np.ndarray, *, source: Path, key: str | None, kind: str) -> None:
    """Write a small JSON sidecar describing an exported `.npy` tensor."""
    manifest = {
        "format": "torchlean-npy",
        "kind": kind,
        "file": out.name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "source": str(source),
    }
    if key is not None:
        manifest["key"] = key
    path = out.with_suffix(out.suffix + ".json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_csv_array(path: Path, *, skip_header: int = 0) -> np.ndarray:
    """Load a numeric CSV, retrying once for a likely header row."""
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=skip_header)
    if skip_header == 0 and np.size(arr) > 0 and np.isnan(arr).any():
        # Common spreadsheet exports include one textual header row. NumPy turns
        # that row into NaNs, so retry once with a skipped header before failing.
        retry = np.genfromtxt(path, delimiter=",", dtype=np.float32, skip_header=1)
        if np.size(retry) > 0 and not np.isnan(retry).all():
            return retry
    return arr


def load_torch_artifact(path: Path, *, trusted_legacy_pickle: bool) -> Any:
    """Load a PyTorch artifact with pickle disabled unless explicitly requested."""
    try:
        import torch  # type: ignore
    except ImportError:
        die("PyTorch checkpoint conversion requires torch: python3 -m pip install torch")
    if trusted_legacy_pickle:
        return torch.load(path, map_location="cpu", weights_only=False)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        die(
            "installed PyTorch does not support weights_only=True; upgrade PyTorch or pass "
            "--trusted-legacy-pickle only for files you trust."
        )


def load_tensor(
    path: Path,
    key: str | None,
    *,
    csv_skip_header: int = 0,
    trusted_legacy_pickle: bool = False,
) -> Any:
    """Load one tensor-like artifact from `.npy`, `.npz`, `.mat`, `.pt`, `.pth`, or CSV."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        return select_key(np.load(path, allow_pickle=False), key, source=path)
    if suffix == ".mat":
        try:
            import scipy.io  # type: ignore
        except ImportError:
            die("MATLAB .mat conversion requires scipy: python3 -m pip install scipy")
        return select_key(scipy.io.loadmat(path), key, source=path)
    if suffix in {".pt", ".pth"}:
        obj = load_torch_artifact(path, trusted_legacy_pickle=trusted_legacy_pickle)
        return select_key(obj, key, source=path)
    if suffix == ".csv":
        return load_csv_array(path, skip_header=csv_skip_header)
    die(f"unsupported tensor input suffix {suffix!r}")


def cmd_tensor(args: argparse.Namespace) -> None:
    """Implement the `tensor` subcommand."""
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    obj = load_tensor(
        inp,
        args.key,
        csv_skip_header=args.skip_header,
        trusted_legacy_pickle=args.trusted_legacy_pickle,
    )
    arr = cast_array(to_numpy(obj, source=inp), args.dtype)
    np.save(out, arr)
    if args.manifest:
        write_manifest(out, arr, source=inp, key=args.key, kind="tensor")
    print(f"[write] {out} shape={tuple(arr.shape)} dtype={arr.dtype}")


def cmd_pair(args: argparse.Namespace) -> None:
    """Implement the `pair` subcommand for feature/label tensor exports."""
    x_args = argparse.Namespace(
        input=args.x_input,
        output=args.x_output,
        key=args.x_key,
        dtype=args.dtype,
        manifest=args.manifest,
        skip_header=0,
        trusted_legacy_pickle=args.trusted_legacy_pickle,
    )
    y_args = argparse.Namespace(
        input=args.y_input,
        output=args.y_output,
        key=args.y_key,
        dtype=args.label_dtype,
        manifest=args.manifest,
        skip_header=0,
        trusted_legacy_pickle=args.trusted_legacy_pickle,
    )
    cmd_tensor(x_args)
    cmd_tensor(y_args)


def read_labels_csv(path: Path, label_col: str | None) -> list[int]:
    """Read integer labels from a header or no-header CSV file."""
    with path.open(newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample)
        if has_header:
            reader = csv.DictReader(f)
            if label_col is None:
                die("--label-col is required for a header CSV label file")
            return [int(float(row[label_col])) for row in reader]
        reader = csv.reader(f)
        return [int(float(row[0])) for row in reader if row]


def cmd_labels(args: argparse.Namespace) -> None:
    """Implement the `labels` subcommand, including optional class-range checks."""
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if inp.suffix.lower() == ".csv":
        labels = np.asarray(read_labels_csv(inp, args.label_col), dtype=args.dtype)
    else:
        obj = load_tensor(
            inp,
            args.key,
            csv_skip_header=args.skip_header,
            trusted_legacy_pickle=args.trusted_legacy_pickle,
        )
        labels = cast_array(to_numpy(obj, source=inp), args.dtype).reshape(-1)
    if args.classes is not None:
        bad = labels[(labels < 0) | (labels >= args.classes)]
        if bad.size:
            die(f"labels outside [0,{args.classes}): first bad label {bad[0]}")
    np.save(out, labels)
    if args.manifest:
        write_manifest(out, labels, source=inp, key=args.key, kind="labels")
    print(f"[write] {out} shape={tuple(labels.shape)} dtype={labels.dtype}")


def cmd_image_folder(args: argparse.Namespace) -> None:
    """Convert an image tree to an NCHW tensor and optional label vector."""
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        die("image-folder conversion requires Pillow: python3 -m pip install pillow")

    root = Path(args.input)
    x_out = Path(args.x_output)
    y_out = Path(args.y_output) if args.y_output else None
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext}

    if args.labels_from_dirs:
        class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not class_dirs:
            die(f"{root}: no class subdirectories found")
        # Class IDs come from sorted directory names so exports are reproducible
        # across filesystems.
        class_to_id = {p.name: i for i, p in enumerate(class_dirs)}
        files: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            for p in sorted(cls_dir.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts:
                    files.append((p, class_to_id[cls_dir.name]))
    else:
        files = [(p, -1) for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in exts]

    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        die(f"{root}: no images found")

    h, w = args.height, args.width
    images: list[np.ndarray] = []
    labels: list[int] = []
    for p, label in files:
        img = Image.open(p).convert("RGB").resize((w, h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(np.transpose(arr, (2, 0, 1)))
        if label >= 0:
            labels.append(label)

    X = np.stack(images, axis=0).astype(args.dtype, copy=False)
    x_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(x_out, X)
    if args.manifest:
        write_manifest(x_out, X, source=root, key=None, kind="images")
    print(f"[write] {x_out} shape={tuple(X.shape)} dtype={X.dtype}")

    if args.labels_from_dirs:
        if y_out is None:
            die("--y-output is required with --labels-from-dirs")
        y = np.asarray(labels, dtype=np.float32)
        y_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(y_out, y)
        if args.manifest:
            write_manifest(y_out, y, source=root, key=None, kind="labels")
        print(f"[write] {y_out} shape={tuple(y.shape)} dtype={y.dtype}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the multi-subcommand converter parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tensor", help="convert one tensor-like artifact to .npy")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key", help="key for .npz/.mat/.pt dictionaries")
    p.add_argument("--skip-header", type=int, default=0, help="CSV rows to skip before reading")
    p.add_argument("--dtype", default="float32", help="float32, float64, int64, or preserve")
    p.add_argument("--manifest", action="store_true", help="write OUTPUT.npy.json metadata")
    p.add_argument(
        "--trusted-legacy-pickle",
        action="store_true",
        help="allow pickle-based torch.load for trusted legacy .pt/.pth files",
    )
    p.set_defaults(func=cmd_tensor)

    p = sub.add_parser("pair", help="convert X and y artifacts for supervised/labeled training")
    p.add_argument("--x-input", required=True)
    p.add_argument("--y-input", required=True)
    p.add_argument("--x-output", required=True)
    p.add_argument("--y-output", required=True)
    p.add_argument("--x-key")
    p.add_argument("--y-key")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--label-dtype", default="float32")
    p.add_argument("--manifest", action="store_true")
    p.add_argument(
        "--trusted-legacy-pickle",
        action="store_true",
        help="allow pickle-based torch.load for trusted legacy .pt/.pth files",
    )
    p.set_defaults(func=cmd_pair)

    p = sub.add_parser("labels", help="convert label files to a float32/int npy vector")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--key")
    p.add_argument("--skip-header", type=int, default=0, help="CSV rows to skip before reading")
    p.add_argument("--label-col")
    p.add_argument("--classes", type=int)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--manifest", action="store_true")
    p.add_argument(
        "--trusted-legacy-pickle",
        action="store_true",
        help="allow pickle-based torch.load for trusted legacy .pt/.pth files",
    )
    p.set_defaults(func=cmd_labels)

    p = sub.add_parser("image-folder", help="convert an image folder to NCHW .npy tensors")
    p.add_argument("--input", required=True)
    p.add_argument("--x-output", required=True)
    p.add_argument("--y-output")
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp", ".webp"])
    p.add_argument("--limit", type=int)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--labels-from-dirs", action="store_true")
    p.add_argument("--manifest", action="store_true")
    p.set_defaults(func=cmd_image_folder)

    return ap


def main() -> None:
    """Parse arguments and dispatch to the selected converter subcommand."""
    ap = build_parser()
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
