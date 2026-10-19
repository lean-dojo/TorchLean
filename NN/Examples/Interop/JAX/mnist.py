# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cpu]==0.11.2", "numpy==2.5.4"]
# ///
"""Train MNIST in JAX, then replay the trained forward graph in TorchLean.

Run from the repository root with ``uv run NN/Examples/Interop/JAX/mnist.py OUTPUT``.
Build ``verify`` first. Training and test accuracy are experiments; the Lean checker
validates and executes the captured forward computation, not the training algorithm.
"""

from __future__ import annotations

import argparse
from functools import partial
from fractions import Fraction
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
# Import the shared downloader without leaving generated files under scripts/.
sys.dont_write_bytecode = True

import torchlean
from NN.Runtime.JAX.jaxpr import encode, export
from scripts.datasets.download_example_data import download_atomic


# MNIST's published archive checksums, also used by torchvision.datasets.MNIST.
SOURCE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
FORMATS = {"float32": jnp.float32, "float16": jnp.float16, "bfloat16": jnp.bfloat16}


def load(directory: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the checked IDX archives; normalize pixels to float32 in [0, 1]."""
    arrays = []
    count = 60_000 if split == "train" else 10_000
    for kind, magic, shape in (("images-idx3", 2051, (count, 28, 28)),
                               ("labels-idx1", 2049, (count,))):
        name = f"{split}-{kind}-ubyte.gz"
        path = directory / name
        download_atomic(SOURCE + name, path, timeout=60, md5=FILES[name])
        with gzip.open(path, "rb") as source:
            header = source.read(4 * (len(shape) + 1))
            if header != struct.pack(">" + "I" * (len(shape) + 1), magic, *shape):
                raise ValueError(f"invalid MNIST header: {path}")
            values = np.frombuffer(source.read(), dtype=np.uint8)
        if values.size != np.prod(shape):
            raise ValueError(f"invalid MNIST payload length: {path}")
        arrays.append(values.reshape(shape))
    images, labels = arrays
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be digits 0 through 9")
    return images.reshape(count, 784).astype(np.float32) / np.float32(255), labels


def forward(x, w1, b1, w2, b2, *, precision=jnp.bfloat16):
    """Round stored forward tensors; perform both matrix products in float32.

    The casts are part of the captured graph. This is mixed precision, not a claim
    that XLA or TorchLean performs a pure bfloat16/float16 matrix multiplication.
    """
    def store(value):
        return value.astype(precision).astype(jnp.float32)

    hidden = store(jax.nn.relu(store(x) @ store(w1) + store(b1)))
    return store(hidden @ store(w2) + store(b2))


def train(images, labels, *, width: int, batch: int, epochs: int, seed: int,
          precision, rate: float):
    """Use ordinary JAX autodiff and SGD with float32 master parameters.

    JAX supplies the derivative rules through casts. This example does not prove
    those rules are derivatives of the discontinuous mathematical rounding map.
    """
    rng = np.random.default_rng(seed)
    parameters = (
        jnp.asarray(rng.normal(size=(784, width)).astype(np.float32) * np.float32(np.sqrt(2 / 784))),
        jnp.zeros(width, dtype=jnp.float32),
        jnp.asarray(rng.normal(size=(width, 10)).astype(np.float32) * np.float32(np.sqrt(2 / width))),
        jnp.zeros(10, dtype=jnp.float32),
    )
    model = partial(forward, precision=precision)

    def loss(parameters, x, y):
        logits = model(x, *parameters)
        return -jnp.mean(jnp.sum(jax.nn.one_hot(y, 10) * jax.nn.log_softmax(logits), axis=1))

    @jax.jit
    def step(parameters, x, y):
        value, gradient = jax.value_and_grad(loss)(parameters, x, y)
        return jax.tree.map(lambda p, g: p - jnp.float32(rate) * g,
                            parameters, gradient), value

    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(images))
        total = 0.0
        for start in range(0, len(images), batch):
            selected = order[start:start + batch]
            parameters, value = step(parameters, images[selected], labels[selected])
            total += float(value) * len(selected)
        average = total / len(images)
        if not np.isfinite(average):
            raise ValueError("training produced a nonfinite loss")
        history.append(average)
        print(f"epoch {epoch + 1}: training loss {average:.6f}", flush=True)
    return parameters, history


def write(path: Path, value) -> None:
    """Never replace artifacts from an earlier run."""
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def replay(model, arguments, directory: Path, name: str, checker: Path) -> dict:
    """Keep execution and exact-word comparison separate, with both reports saved."""
    artifact = export(model, *arguments)
    artifact["arguments"] = [encode(value) for value in arguments]
    expected = model(*arguments)
    artifact["reference_outputs"] = [encode(expected)]
    write(directory / f"{name}.json", artifact)
    checked = torchlean.verify(artifact, checks=("structure", "types", "execute"), checker=checker)
    write(directory / f"{name}-checks.json", checked)
    if not checked["ok"]:
        raise RuntimeError(f"{name}: capture or reference execution failed; inspect its report")
    compared = torchlean.verify(artifact, checks=("compare",), checker=checker)
    write(directory / f"{name}-compare.json", compared)
    words = np.asarray(checked["outputs"][0]["bits"], dtype=np.uint32)
    actual = words.view(np.float32).reshape(expected.shape)
    expected = np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("cannot compare decisions with nonfinite logits")
    # Exact rational differences avoid rounding a near-threshold margin comparison.
    a = [Fraction(float(x)) for x in actual.reshape(-1)]
    e = [Fraction(float(x)) for x in expected.reshape(-1)]
    winner = int(expected.argmax())
    drift = max(abs(x - y) for x, y in zip(a, e))
    margin = e[winner] - max(x for i, x in enumerate(e) if i != winner)
    result = dict(nodes=len(artifact["nodes"]), reference_execution=True,
                  words_equal=compared["ok"], jax_prediction=winner,
                  reference_prediction=int(actual.argmax()),
                  max_logit_difference=str(drift), jax_margin=str(margin),
                  twice_difference_below_margin=2 * drift < margin,
                  scope="One held-out image; rational sample calculation, not a Lean theorem.")
    print(f"{name}: replay passed; words equal={compared['ok']}; "
          f"prediction JAX/reference={winner}/{actual.argmax()}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/real/mnist")
    parser.add_argument("--precision", choices=FORMATS, default="bfloat16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--checker", type=Path, default=Path(os.environ.get(
        "TORCHLEAN_VERIFY", ROOT / ".lake/build/bin/verify")))
    args = parser.parse_args()
    if min(args.epochs, args.width, args.batch) < 1 or not np.isfinite(args.rate) or args.rate <= 0:
        parser.error("epochs, width, batch and learning rate must be positive and finite")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not args.checker.is_file():
        parser.error("build verify first: scripts/lake.sh build verify")
    args.checker = args.checker.resolve()
    if args.directory.exists():
        parser.error("choose a new output directory; existing runs are never overwritten")

    # Keep this onboarding run independent of the machine's accelerator configuration.
    jax.config.update("jax_platforms", "cpu")
    images, labels = load(args.data, "train")
    test_images, test_labels = load(args.data, "t10k")
    args.directory.mkdir(parents=True)
    parameters, history = train(images, labels, width=args.width, batch=args.batch,
                                epochs=args.epochs, seed=args.seed,
                                precision=FORMATS[args.precision], rate=args.rate)
    np.savez(args.directory / "weights.npz", **dict(zip(("w1", "b1", "w2", "b2"),
                                                       map(np.asarray, parameters))))
    results = {}
    for name, dtype in FORMATS.items():
        model = partial(forward, precision=dtype)
        predict = jax.jit(model)
        logits = np.concatenate([np.asarray(predict(test_images[i:i + args.batch], *parameters))
                                 for i in range(0, len(test_images), args.batch)])
        if not np.isfinite(logits).all():
            raise ValueError(f"{name}: nonfinite test logits")
        correct = int(np.sum(logits.argmax(axis=1) == test_labels))
        print(f"{name}: test accuracy {correct}/{len(test_labels)}", flush=True)
        results[name] = dict(correct=correct, total=len(test_labels),
                             replay=replay(model, (jnp.asarray(test_images[:1]), *parameters),
                                           args.directory, name, args.checker))
    summary = dict(
        format="torchlean.mnist.v1", seed=args.seed, epochs=args.epochs, width=args.width,
        batch=args.batch, rate=args.rate, training_precision=args.precision,
        master_parameters="float32", accumulation="float32", loss="float32",
        train_count=len(images), test_count=len(test_images), training_losses=history,
        heldout_image_index=0, heldout_label=int(test_labels[0]),
        jax=jax.__version__, numpy=np.__version__, devices=list(map(str, jax.devices())),
        source=SOURCE, archive_md5=FILES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(args.checker.read_bytes()).hexdigest(),
        results=results, training_correctness="not_proved", frontend_equivalence="not_proved",
    )
    write(args.directory / "summary.json", summary)
    print(f"Saved weights, captures and reports to {args.directory}")


if __name__ == "__main__":
    main()
