# /// script
# requires-python = ">=3.12"
# dependencies = ["jax[cpu]==0.11.2", "numpy==2.5.4"]
# ///
"""Train MNIST in JAX, then replay the trained forward graph in TorchLean.

Run from the repository root with ``uv run TorchLeanExamples/Interop/JAX/mnist.py OUTPUT``
after ``scripts/lake.sh build verify TorchLean.Verify.Robustness.LipschitzCertified``: the
generated Lean proofs import that module. For one held-out test image (``--image-index``) the
script compares logit vectors in pairs: JAX against the TorchLean IR at each precision, and the
float32 IR against the float16 and bfloat16 IR. When the two vectors of a pair differ by less than
half the reference's top-two gap in every logit, Lean proves that they pick the same class.
Training losses and test accuracies are measurements.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))
# Import the shared downloader without leaving generated files under scripts/.
sys.dont_write_bytecode = True

import torchlean
from torchlean.jaxpr import encode, export, references
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
    """Read one MNIST split as float32 pixels in [0, 1] and labels.

    An archive that is missing or fails its MD5 check is downloaded first.
    """
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
    """Logits of a two-layer ReLU network whose stored tensors are rounded to ``precision``.

    The input, the parameters, the hidden layer and the logits are each rounded to ``precision``
    and back to float32. Both matrix products and the bias additions run in float32 on the rounded
    values. The casts are part of the captured graph.
    """
    def store(value):
        return value.astype(precision).astype(jnp.float32)

    hidden = store(jax.nn.relu(store(x) @ store(w1) + store(b1)))
    return store(hidden @ store(w2) + store(b2))


def train(images, labels, *, width: int, batch: int, epochs: int, seed: int,
          precision, rate: float):
    """Train with JAX autodiff and SGD on float32 master parameters.

    Returns the parameters and the mean training loss of each epoch. JAX's own rules carry the
    gradient through the casts, and they are not proved to be derivatives of the discontinuous
    rounding map.
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
    """Write ``value`` as JSON to ``path``, refusing to replace an earlier run's file."""
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write("\n")


def prove_decision(path: Path, expected: list[Fraction], actual: list[Fraction],
                   winner: int, margin: Fraction, drift: Fraction, *,
                   reference: str = "JAX", candidate: str = "IR reference") -> dict:
    """Write and check a Lean proof that ``actual`` picks the top class ``winner`` of ``expected``.

    The proof states both vectors as exact rationals, proves the margin and the drift bound, and
    applies ``argmaxClassifier_eq_of_distance_le``. It is about these two recorded vectors only.
    Unless twice the drift is below the margin, the result is ``inconclusive`` and no file is
    written.
    """
    if not 2 * drift < margin:
        return dict(status="inconclusive", reason="the observed drift reaches half the margin")

    def literal(value: Fraction) -> str:
        return f"({value.numerator} / {value.denominator})"

    def tensor(values: list[Fraction]) -> str:
        return "Tensor.ofFn ![\n  " + ",\n  ".join(map(literal, values)) + "]"

    source = f'''module

public import TorchLean.Verify.Robustness.LipschitzCertified
public import TorchLean.Tensor.OfFn
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-!
# MNIST recorded-logit decision certificate

These are the exact rational values of the {reference} and {candidate} output words
saved beside this file. The proofs establish their margins and decision stability.
They do not prove the frontend translation, training, or an error bound for other inputs.
-/

@[expose] public section

open TorchLean
open TorchLean.Robustness
open FloatLib.Numerics (MathFunctions)

namespace {path.stem}

/-- Recorded {reference} logits, without decimal rounding. -/
noncomputable def referenceLogits : Tensor ℝ [{len(expected)}] := {tensor(expected)}

/-- Recorded {candidate} logits, in the same class order. -/
noncomputable def candidateLogits : Tensor ℝ [{len(actual)}] := {tensor(actual)}

/-- The class with the strictly largest recorded reference logit. -/
def winner : Fin {len(expected)} := ⟨{winner}, by decide⟩

/-- The gap between the largest and second-largest reference logits. -/
noncomputable def margin : ℝ := {literal(margin)}

/-- The largest observed absolute difference between corresponding logits. -/
noncomputable def drift : ℝ := {literal(drift)}

/-- Every competing reference logit is below the winner by this margin. -/
theorem reference_margin : HasLogitMargin referenceLogits winner margin := by
  intro k hk
  fin_cases k <;> norm_num [referenceLogits, winner, margin] at *

/-- Check the observed difference in every coordinate using exact arithmetic. -/
theorem drift_bound :
    distance (α := ℝ) (linfNorm (α := ℝ)) referenceLogits candidateLogits ≤ drift := by
  rw [distance_eq_norm_sub]
  apply List.foldl_max_le_of_le (List.finRange {len(expected)})
    (fun i => linfNorm (α := ℝ) ((referenceLogits - candidateLogits).unstack i))
    (by norm_num [drift])
  intro i _
  fin_cases i <;> norm_num [linfNorm, MathFunctions.abs, Tensor.item, Tensor.unstack,
    Tensor.sub_def, referenceLogits, candidateLogits, drift]

/-- Any logits within less than half this margin retain the same winning class. -/
theorem prediction_stable (logits : Tensor ℝ [{len(expected)}]) (δ : ℝ)
    (hδ : 2 * δ < margin)
    (hdist : distance (α := ℝ) (linfNorm (α := ℝ)) referenceLogits logits ≤ δ) :
    argmaxClassifier logits = winner.val :=
  argmaxClassifier_eq_of_distance_le reference_margin hδ hdist

/-- The measured difference cannot change this recorded prediction. -/
theorem candidate_prediction : argmaxClassifier candidateLogits = winner.val :=
  prediction_stable candidateLogits drift (by norm_num [drift, margin]) drift_bound

end {path.stem}
'''
    with path.open("x", encoding="utf-8") as target:
        target.write(source)
    subprocess.run([str(ROOT / "scripts/lake.sh"), "--no-build", "env", "lean",
                    "-DwarningAsError=true", str(path.resolve())], cwd=ROOT, check=True)
    return dict(status="proved", source=path.name,
                scope="Recorded logit vectors and conditional output-error stability only.")


def replay(model, arguments, directory: Path, name: str, checker: Path) -> dict:
    """Capture ``model`` on ``arguments`` as ``<name>.json`` and check it in two checker runs.

    Structure, types and execution go to ``<name>-checks.json`` and must pass. The comparison of
    output words with JAX goes to ``<name>-compare.json``. A mismatch there is recorded, and the
    decision proof is attempted either way.
    """
    document = export(model, *arguments, checker=checker)
    document["arguments"] = [encode(value) for value in arguments]
    document.update(references(model, *arguments))
    recorded = document["reference_outputs"][0]
    expected = np.asarray(recorded["bits"], dtype=np.uint32).view(np.float32).reshape(
        recorded["shape"])
    write(directory / f"{name}.json", document)
    checked = torchlean.verify(document, checks=("structure", "types", "execute"), checker=checker)
    write(directory / f"{name}-checks.json", checked)
    if not checked["ok"]:
        raise RuntimeError(f"{name}: capture or reference execution failed; inspect its report")
    compared = torchlean.verify(document, checks=("compare",), checker=checker)
    write(directory / f"{name}-compare.json", compared)
    words = np.asarray(checked["outputs"][0]["bits"], dtype=np.uint32)
    actual = words.view(np.float32).reshape(expected.shape)
    expected = np.asarray(expected)
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("cannot compare decisions with nonfinite logits")
    # Preserve every output bit when calculating the proof's margin and error budget.
    a = [Fraction(float(x)) for x in actual.reshape(-1)]
    e = [Fraction(float(x)) for x in expected.reshape(-1)]
    winner = int(expected.argmax())
    drift = max(abs(x - y) for x, y in zip(a, e))
    margin = e[winner] - max(x for i, x in enumerate(e) if i != winner)
    result = dict(nodes=len(document["artifact"]["nodes"]), reference_execution=True,
                  words_equal=compared["ok"], jax_prediction=winner,
                  reference_prediction=int(actual.argmax()),
                  max_logit_difference=str(drift), jax_margin=str(margin),
                  twice_difference_below_margin=2 * drift < margin,
                  decision=prove_decision(directory / f"{name.capitalize()}Decision.lean",
                                          e, a, winner, margin, drift),
                  scope="One held-out image; no uniform numerical error bound.")
    print(f"{name}: replay passed; words equal={compared['ok']}; "
          f"prediction JAX/reference={winner}/{actual.argmax()}; "
          f"decision certificate={result['decision']['status']}", flush=True)
    return result


def compare_precision(directory: Path, name: str) -> dict:
    """Compare the same image and checkpoint under two IR precision choices.

    The logits come from the float32 and ``name`` execution reports (``-checks.json``), so the
    graphs do not run again. The Lean proof is about these two recorded vectors of one input.
    """
    def read_logits(precision: str) -> list[Fraction]:
        report = json.loads((directory / f"{precision}-checks.json").read_text())
        words = np.asarray(report["outputs"][0]["bits"], dtype=np.uint32)
        values = words.view(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("cannot compare precision with nonfinite logits")
        return [Fraction(float(value)) for value in values]

    baseline, candidate = read_logits("float32"), read_logits(name)
    winner = max(range(len(baseline)), key=baseline.__getitem__)
    margin = baseline[winner] - max(value for i, value in enumerate(baseline) if i != winner)
    drift = max(abs(a - b) for a, b in zip(baseline, candidate, strict=True))
    decision = prove_decision(directory / f"{name.capitalize()}Precision.lean", baseline,
                              candidate, winner, margin, drift,
                              reference="float32 IR", candidate=f"{name} IR")
    print(f"float32 → {name}: max logit difference {float(drift):.8g}; "
          f"half-margin {float(margin / 2):.8g}; decision certificate={decision['status']}",
          flush=True)
    return dict(reference="float32", candidate=name, max_logit_difference=str(drift),
                reference_margin=str(margin), decision=decision,
                scope="Recorded binary IR outputs for the same checkpoint and held-out image.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/real/mnist")
    parser.add_argument("--precision", choices=FORMATS, default="bfloat16")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-index", type=int, default=0,
                        help="held-out test image to capture and compare (0–9999)")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--checker", type=Path, default=Path(os.environ.get(
        "TORCHLEAN_VERIFY", ROOT / ".lake/build/bin/verify")))
    args = parser.parse_args()
    if min(args.epochs, args.width, args.batch) < 1 or not np.isfinite(args.rate) or args.rate <= 0:
        parser.error("epochs, width, batch and learning rate must be positive and finite")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    if not 0 <= args.image_index < 10_000:
        parser.error("image-index must be between 0 and 9999")
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
                             replay=replay(model, (jnp.asarray(test_images[
                                 args.image_index:args.image_index + 1]), *parameters),
                                           args.directory, name, args.checker))
    precision_comparisons = {name: compare_precision(args.directory, name)
                             for name in ("float16", "bfloat16")}
    summary = dict(
        format="torchlean.mnist.v1", seed=args.seed, epochs=args.epochs, width=args.width,
        batch=args.batch, rate=args.rate, training_precision=args.precision,
        master_parameters="float32", accumulation="float32", loss="float32",
        train_count=len(images), test_count=len(test_images), training_losses=history,
        heldout_image_index=args.image_index, heldout_label=int(test_labels[args.image_index]),
        jax=jax.__version__, numpy=np.__version__, devices=list(map(str, jax.devices())),
        source=SOURCE, archive_md5=FILES,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(args.checker.read_bytes()).hexdigest(),
        results=results, precision_comparisons=precision_comparisons,
        training_correctness="not_proved", frontend_equivalence="not_proved",
    )
    write(args.directory / "summary.json", summary)
    print(f"Saved weights, captures and reports to {args.directory}")


if __name__ == "__main__":
    main()
