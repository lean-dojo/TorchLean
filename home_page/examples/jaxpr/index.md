---
title: JAX Computations and Lean Proofs
---

I wanted to take a model someone had already written in JAX and use Lean to ask a question
about it. Maybe we want to know if an attention implementation computes the formula in a
paper, or what happens to the predictions when we lower the precision. I'd like to do that
without rewriting the model or asking anyone to switch backends.

TorchLean reads the supported operations in a traced Jaxpr into its tensor IR, keeping the
axes, constants, and precision choices. We can inspect the graph, run it with FloatLib
arithmetic, and use Lean to prove properties of its semantics. The JAX program still runs on
its usual backend. Connecting a proof about the imported graph to that run requires a faithful
translation and a model of the backend arithmetic.

```python
import jax.numpy as jnp

def sum_squares(x):
    return jnp.sum(x * x)
```

Reordering a real vector leaves its sum of squares unchanged. We'll prove that for every
permutation of the imported graph's inputs. Then we'll bring in a two-layer network and a
trained MNIST classifier to see what changing precision does to the predictions.

## Use the JAX function directly

From the TorchLean checkout, install the Python package with
`python -m pip install -e python` and build the checker once with
`scripts/lake.sh build verify`. Then pass the function and its tensor arguments
to `torchlean.verify`. Capture and translation happen inside that call; we don't
need to export a file first.

```python
import jax
import jax.numpy as jnp
import torchlean

jax.config.update("jax_enable_x64", True)
x = jnp.arange(1, 5, dtype=jnp.float64)
report = torchlean.verify(sum_squares, x, frontend="jax")
print(report["ok"])
```

[`jax.make_jaxpr`](https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html)
records a Jaxpr at this input shape. TorchLean reads its equations and constructs
three TorchLean IR nodes: the input, elementwise multiplication, and a scalar sum.
It sends that graph to Lean for shape and type checks and returns their report.
Constants in other computations travel with the graph. Each argument is taken as
JAX reads it (`jnp.asarray`), and the trace, the serialized arguments and the
reference outputs all use those values. With 64-bit mode off, a NumPy float64,
int64 or uint64 array becomes its 32-bit counterpart, as in any traced call.
An operation traced inside a local `jax.enable_x64()` context keeps that recorded dtype
during reference evaluation, even after the context ends.

```text
input [4] → multiply input by itself [4] → sum []
```

On the Lean side, `TorchLean.JAX.load` checks dependencies, operator arguments,
shapes, and the element type of each node, and decodes each constant from its
interchange words. `TorchLean.JAX.load_ok` proves that every program of the flow it
returns passes these checks. The load also refuses a program that gives a value another element
type or shape than JAX's trace records for it (the document's `declared_types`,
which `TorchLean.IR.loadFlow` checks, each in the block its entry names). It reads
the same artifact format,
`torchlean.ir`, as the PyTorch capture, so JAX does not get a separate tensor
representation.

## What comes out of the pipeline?

For JAX, we capture a Jaxpr and translate its supported equations into TorchLean
IR. For PyTorch, we read a captured graph and translate its supported operations.
We are importing these frontend graphs, not XLA's or LibTorch's compiled kernels.

Here's what we get back, depending on which steps we run:

| Output | What we get |
| --- | --- |
| Optional export | The operations, dependencies, constants, tensor shapes, element types, explicit casts and designated outputs, saved as JSON or embedded in a Lean module. |
| Check report | Results for the requested checks, including failures and unsupported cases. A successful report applies to those checks. |
| Reference results | Computed output tensors when evaluation is requested, index results with their bounds, and the run's complete conversion status on the configured binary path: `inexact`, `overflow`, `underflow`, `saturated`, `wrapped`, `mapped_special` and `invalid`. Sample comparisons relate these results to recorded frontend outputs. |
| Lean theorem | A separately written, kernel-checked proof about the graph's Lean meaning. Passing the checks does not generate this proof. |

For our input `[1, 2, 3, 4]`, the sum-of-squares result is `30`. Requesting `compare`
also evaluates the IR and compares its output with JAX for these arguments.
Later, we'll prove that reordering any real input
vector leaves this graph's result unchanged. That lets us say something about
inputs beyond the one we ran.

## Run the examples

The companion script captures a small MLP, sums, broadcasts, reshapes,
axis permutations, a batched contraction, a comparison, and functions returning several tensors.
It reports unsupported captures by name and continues with the supported ones:

```bash
python3 TorchLeanExamples/Interop/JAX/export.py /tmp/torchlean-jax-examples
scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/sum_squares.json
scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/mlp.json
```

Try `--width 7 --batch 3` when capturing the examples. Those are example dimensions,
not restrictions of the importer or the permutation theorem.

The script also captures `jax.grad(sum_squares)`. Its result is another tensor
program that can enter the same IR; comparing its outputs is not, on its own,
a proof that the captured program is the mathematical derivative.

The Lean runner loads each artifact, checks its graph and constants, and compares
all designated outputs with the recorded JAX results. Every example uses FloatLib
at each node's declared precision and compares interchange words. These are checks
of the recorded inputs, not proofs of agreement for all inputs or a replay of XLA's
chosen kernels.

## Bring in a neural network

Here is a two-layer network with a ReLU between its affine layers:

$$
F(X;W_1,b_1,W_2,b_2)
=\operatorname{ReLU}(XW_1+b_1)W_2+b_2.
$$

We'll pass the parameters as tensor arguments, so they remain visible in the
captured interface. The same function works at different compatible widths;
the dimensions below choose one example to capture.

Using the same checker, we can pass the network and all five tensor arguments:

```python
import jax
import jax.numpy as jnp
import torchlean

jax.config.update("jax_enable_x64", True)

def mlp(x, w1, b1, w2, b2):
    hidden = jax.nn.relu(x @ w1 + b1)
    return hidden @ w2 + b2

batch, input_width, hidden_width, output_width = 3, 7, 5, 2
dtype = jnp.float64
x = jnp.arange(batch * input_width, dtype=dtype)
x = x.reshape(batch, input_width)
x = (x - batch * input_width // 2) / 8
w1 = jnp.arange(input_width * hidden_width, dtype=dtype)
w1 = w1.reshape(input_width, hidden_width) / 16
b1 = jnp.full((hidden_width,), -0.25, dtype=dtype)
w2 = jnp.arange(hidden_width * output_width, dtype=dtype)
w2 = w2.reshape(hidden_width, output_width) / 32
b2 = jnp.zeros((output_width,), dtype=dtype)

report = torchlean.verify(
    mlp, x, w1, b1, w2, b2, frontend="jax",
    checks=("structure", "types", "compare"),
)
print(report["ok"])
print(report["checks"])
```

The divisions above prepare the example tensors before capture. The network
function itself uses matrix multiplication, addition and ReLU. We checked this
five-input example: the graph passed its structure and type checks, and the
reference output matched JAX's complete binary64 words for these arguments.
The result has shape `[batch, output_width]`, here `[3, 2]`.

Inside the graph, bias addition includes explicit broadcasting. That matters when
we reason about the model: adding one bias per hidden feature is a different
operation from reducing across the batch. The node shapes and broadcast axes
tell us which calculation we imported.

`import torchlean` currently uses the Python directory in this checkout; it is
not a separately published package. The
[captured-computation page]({{ '/examples/verify/' | relative_url }}) explains the
individual report fields. If we want to keep this capture for a proof, add
`export="Proofs/MLP.lean"` to the same call. No separate capture is needed.

## Try a trained MNIST classifier

The hand-written weights above let us follow a small graph. To try the same
interface with learned weights, run the MNIST companion from the checkout after
`scripts/lake.sh build verify TorchLean.Verify.Robustness.LipschitzCertified`, which
builds `verify` and the module its Lean proofs import. `uv run` installs the script's
declared JAX and NumPy versions.
Use a fresh output directory for each run:

```bash
uv run TorchLeanExamples/Interop/JAX/mnist.py /tmp/torchlean-mnist \
  --precision bfloat16 --epochs 3 --width 32 --batch 128 --seed 0
```

The script downloads MNIST if needed, trains on its 60,000 training images and
evaluates on its separate 10,000-image test set. This two-layer classifier runs
on CPU JAX. We keep the master parameters, SGD updates and loss in float32, and
round the forward pass's inputs, weights, biases, hidden activations and logits
to bfloat16. Fewer bits give us fewer representable numbers, so each cast rounds
to one of them. Before each matrix product, we convert the rounded operands back
to float32. This lets us study these rounding choices while keeping dot-product
arithmetic in float32.

JAX computes the training gradients through those casts; the Lean checks below
do not verify its gradient rules. To train with float16 rounding instead, use
`--precision float16`; `--precision float32` gives a baseline with no low-precision
casts in the forward pass. After training, the script evaluates the same
checkpoint in all three formats and measures accuracy on the test set.

For replay, the script captures the first test image and the learned
parameters for each format, then calls `torchlean.verify` on each artifact with
`checks=("structure", "types", "execute")`. Those checks inspect the captured
graph and run Lean's reference computation for that input. They do not prove the
training procedure correct or bound the accuracy loss from changing precision.

The output directory contains the checkpoint in `weights.npz` and the accuracies
and run details in `summary.json`. For bfloat16, `bfloat16.json` keeps the capture
and its input words, `bfloat16-checks.json` holds the structure, type and execution
report, and `bfloat16-compare.json` holds the bitwise comparison. The other formats
use the same naming pattern. The capture also records the JAX execution platform
and jaxlib version.
Use `--image-index 42` to capture a different test image with the same training
settings. `--width`, `--batch`, `--epochs`, `--seed` and `--rate` control the
training run.

The comparison checks every output bit. Matrix products can use different
accumulation orders in JAX and Lean's reference run, so their final bits may
differ. A mismatch needs investigation; it does not by itself identify a JAX bug.

What if the bits differ but the prediction stays the same? Let $m$ be the gap
between the largest and second-largest JAX logits, and let $\delta$ be the
largest absolute difference between corresponding JAX and IR logits. If
$2\delta < m$, no competitor can overtake the winner.

The script writes a Lean proof of this calculation and checks it before reporting
`decision certificate=proved`. For example, `Float32Decision.lean` contains the
two recorded logit vectors as exact rational values, proves their margin and
distance bound, and applies TorchLean's existing classifier-margin theorem.
`candidate_prediction` proves that the IR logits select the same winning class.
`prediction_stable` also covers any logit vector whose distance from the recorded
JAX vector is less than half that margin. We can open the generated file in Lean
or check it again with:

```bash
scripts/lake.sh --no-build env lean /tmp/torchlean-mnist/Float32Decision.lean
```

This proves a property of the recorded output vectors. It does not prove that the
frontend or the typed run produced those recordings, or give a rounding-error bound
for every input. An end-to-end numerical theorem would need those connections too.
If the measured drift reaches half the margin, the script reports an inconclusive
decision check instead of generating a stability proof.

We also compare the float32 IR output with the float16 and bfloat16 IR outputs
for the same image and checkpoint. This asks a different question from replay:
does changing forward precision leave this prediction unchanged? The script reads
both outputs from the checker's execution reports and writes `Float16Precision.lean`
and `Bfloat16Precision.lean`. They have the same form as `Float32Decision.lean`:
Lean proves the float32 margin, the distance between the two recorded output
vectors, and that the lower-precision output keeps the winning class. As before,
the measured distance must be below half the margin, or the decision is reported
as inconclusive.

These proofs concern the recorded output vectors; they do not re-execute the graphs.
`TorchLean.Capture.checkError_sound` shows that a check which replays both
programs bounds their actual outputs. A closed proof of that check for this network
would need the kernel to evaluate the typed run over all the embedded parameter
words, which is impractical at this size. This is still a concrete-input result, not
an operation-level error bound over an input box. It does not verify JAX's translation
or XLA's kernels. Measurements and proof status appear under `precision_comparisons`
in `summary.json`.

We ran three training jobs with the settings above, changing only `--precision`.
Each cell gives the number of correct test predictions out of 10,000, using the
same trained checkpoint across its row:

| Training forward precision | Float32 evaluation | Float16 evaluation | Bfloat16 evaluation |
| --- | --- | --- | --- |
| Float32 | 9,350 | 9,350 | 9,352 |
| Float16 | 9,354 | 9,353 | 9,352 |
| Bfloat16 | 9,351 | 9,351 | 9,351 |

All nine captures passed the structure, type and execution checks on the first
test image, whose label is 7. Both JAX and Lean predicted 7 in every case. All
float16 and bfloat16 sample comparisons matched exactly; all three float32
comparisons detected different output bits. For the bfloat16-trained checkpoint,
the largest float32 logit difference was `1/2097152`, while JAX's gap between its
top two logits was `11263229/2097152`. That small difference left this prediction
unchanged. These are measurements from these three runs and one captured test
image per format; they do not establish an accuracy ranking between formats or
a bound for other inputs.

For that same checkpoint and image, changing the IR forward precision from
float32 to float16 changed each logit by at most about `0.0024252`; changing it
to bfloat16 gave about `0.017474`. Both were below half the float32 margin,
about `2.6853631`.

We also replayed all 10,000 test images from the bfloat16-trained checkpoint in
the IR, using batches of 128. The IR and JAX predictions agreed within each
precision setting. Bfloat16 outputs agreed bit for bit; float32 and float16 had
some differing logits. Switching from float32 to bfloat16 changed three predictions,
despite both runs getting 9,351 images right. One image changed from a correct 8
to an incorrect 1; another changed from an incorrect 5 to the correct 8.
This full-test-set replay was a separate run; `mnist.py` captures one image.
Its results are execution comparisons, not 10,000 Lean proofs.

## Keep a computation for a proof

Most calls only need the report. When we're ready to write a theorem, we can
save the same capture as a Lean module:

```python
vector = jnp.arange(7, dtype=jnp.float64)
report = torchlean.verify(
    sum_squares, vector, frontend="jax", export="Proofs/SumSquares.lean",
)
```

`SumSquares.artifact` contains the captured operations, shapes, types and constants.
`SumSquares.program` is `SumSquares.artifact.program`. The file proves artifact decoding and
`Program.check .reference`, which checks the program's dependencies, shapes, types and interface.
Lean checks those proofs when we open the module or compile it:

```bash
scripts/lake.sh env lean Proofs/SumSquares.lean
```

The report also validates the constant payload and `declared_types` through `IR.loadFlow`, which
is `IR.load` for an artifact without a flow. The export includes no proof of those load checks.

We can import `Proofs.SumSquares` in our proof project and state further properties
about this program. The export never replaces an existing file, so rerunning a
capture cannot overwrite a proof we've started writing.

The module embeds JSON constructors rather than calling a file parser at runtime.
The decoding and `Program.check` proofs use kernel reduction and no `native_decide`.
The permutation argument below supplies the mathematical property, and more involved
computations need their own arguments.

For inspecting a capture in another tool, use `export="sum_squares.json"` instead.
Both exports are optional and use the artifact already sent to the checker.

## What we have checked

We ran the companion examples at width 4 with batch size 2, then at width 7 with
batch size 3. Every supported companion artifact replayed successfully at both
sizes. The checks covered several parts of a network rather than only scalar
arithmetic:

| Computation | What we checked |
| --- | --- |
| Two-layer MLP | Matrix products, bias broadcasting and ReLU; the binary64 output matched JAX's complete words. The five-input network above also passed an exact-word comparison. |
| Batched contraction | Batch and contraction axes were retained in the IR contraction operator; the supplied batched outputs were compared. |
| `jax.grad(sum_squares)` | JAX's generated gradient program entered the IR and its sample output agreed. This did not prove the derivative rule. |
| Reshape, transpose, broadcast and concatenation | The imported shapes and sample tensor entries agreed after the transformations. |
| Several returned tensors | Every designated output was checked in the captured order. |
| Float16, bfloat16 and mixed precision | Explicit casts and accumulation precisions were retained; the selected examples matched complete output words. |

These runs compared CPU JAX results with Lean's reference execution. They did not
measure GPU performance or verify XLA's compiled kernels.

We also checked captured PyTorch LayerNorm and a matrix-product/softmax attention
calculation on supplied inputs. Those passed sample comparisons through the
[same verification interface]({{ '/examples/verify/' | relative_url }}).
JAX expands those operations differently. A LayerNorm written with `jnp.mean` and
`lax.rsqrt` imports and executes through division and the reciprocal square root,
and `jax.nn.softmax` imports as the primitives JAX expands it into, with `reduce_max`
as one general reduction. A passing PyTorch example therefore does not imply that the
corresponding JAX function imports; each Jaxpr is checked on its own. The
batched-contraction example covers attention's dot-product linear algebra, not a
complete Transformer.

## Numerical differences we found

Inspecting the captured operations helped us catch assumptions that shapes alone
would miss. The adapter gives each primitive its JAX meaning. Division and negation
import as the IR's own `div` and `neg`: replacing `x / y` with `x * (1 / y)` adds a
rounding step and can change exceptional values, and `0 - x` changes the sign of
zero. An integer power imports as the multiplications XLA performs, square-and-multiply
in JAX's order, because a correctly rounded power can differ from that product in
the last bit.

XLA's compiled CPU code does not always keep that meaning. It can multiply by the
rounded reciprocal of a broadcast divisor, giving different words from the IEEE
quotient. It also flushes subnormal values to zero,
approximates the reciprocal square root, and under `jax.jit` rewrites some divisions
and constant powers. The adapter does not copy these rewrites; it records whether the
function is one `jax.jit` call. A comparison that finds different words fails. The
checker then runs the program again as XLA computes it, and only if that run reproduces
the reference does the failure list the documented behaviour it is consistent with. A
wrong operand order next to the division by `3.0` lists nothing. The
[IR guide](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/IR/README.md#frontend-lowering)
lists what we measured.

Precision labels also need executable meaning. We changed a multiplication's
declared output to float64 while leaving its operands float32; the type checker
rejected the artifact and requested an explicit cast. In a separate valid
float64 → float32 → float64 round trip, the input `16777217` became `16777216`.
The outputs agreed with JAX, and the typed run reported `inexact = true`.
So both implementations agreed, even though the conversion lost the final digit.

Square root needs an explicit policy too. JAX's root imports as `Op.sqrt` with the `.raw` policy,
including its behavior on negative inputs and signed zero. A guarded root is a different
operation; agreement on a positive capture sample would not justify substituting it.

These are findings about our import and reference-execution paths. We have not
identified them as bugs in JAX or XLA. For a network, such details matter before
we attach a numerical theorem: we need its casts, domains and reduction order to
match the operations that theorem describes.

## Prove a property of the graph

The source is
[`JAX.lean`](https://github.com/lean-dojo/TorchLean/blob/main/TorchLeanExamples/Interop/JAX.lean).
Its `sumSquaresGraph` builds the same three nodes as the captured example from
registered operators, with a parameterized input shape. `sumSquaresGraph_eval` proves
that evaluating the complete graph over the reals returns `Tensor.sum (x * x)`.

`sumSquaresGraph_permutation` then proves that this evaluation is unchanged for
any vector length `n`, any permutation of its indices, and any two tensors whose
entries agree under that permutation. The proof reuses TorchLean's sum and
pointwise-product lemmas, then mathlib's theorem that a finite sum is unchanged
by reindexing with an equivalence.

The runner recognizes this graph pattern and points to that theorem. A different
graph can have a different property: importing it does not automatically produce
a permutation-invariance proof.

Exact real addition is important here. A floating-point sum can change when we
reorder its terms. To study that difference, we need a numerical error statement
and an explicit reduction schedule, rather than the real-valued equality above.

For a larger network, we start in the same way: identify the exact imported graph
and state a property of its evaluation. For example, we might ask whether a
row-wise MLP commutes with a permutation of the batch, or whether every output
stays inside a chosen interval when the input ranges over a box. The proof must
account for the graph's actual axes, constants and operations. We can reuse
TorchLean's tensor and network lemmas and mathlib where their hypotheses match.

The MLP comparisons above do not include either of those universal theorems.
The completed graph theorem on this page is the real-valued sum-of-squares
permutation result. Numerical enclosures and classifier guarantees have their own
[worked examples]({{ '/examples/numerical-runtime/' | relative_url }}) and
[verification tools]({{ '/examples/verification/' | relative_url }}); applying them
to an imported network requires the corresponding operation coverage and proofs.
A manual proof or an agent-assisted proof still needs Lean to check that connection.

We can also work directly with a typed program rather than first connect it to a
named model specification. `TorchLean.IR.Program.run_eq_foldlM` states that a run of
the typed program of the import is the fold of its steps, so an invariant on
intermediate tensors and conversion flags carries through a successful run one
step at a time, and `TorchLean.IR.Program.check_step` gives the form of each step.

For a norm bound, the local rules would explain how each operation changes the
bound. These rules need mathematical proofs; the common traversal argument
is reusable. It does not require an MLP or transformer tag, and it does not
automatically produce such a bound for the MLP comparison above.

## Which computations can we import?

The adapter (`torchlean.jaxpr`) imports 167 of the 219 `lax` primitives of JAX 0.11.1 and rejects
the others by name. It handles static tensor computations built from supported
elementwise arithmetic and elementary functions, explicit broadcasting, reshape,
transpose, concatenation, sums, and `dot_general` contractions. Floating-point contractions
record their batch and contracted axis pairs in the IR contraction operator.
Supported `jit` and custom-JVP wrappers are inlined using their
forward bodies; this does not verify their custom derivative rules.
Nondefault dot precision and algorithm choices are rejected: with them FP32 inputs and outputs
can use BF16 or TF32 multiplication internally, which the IR's FP32 matrix product does not.

Close over parameter tensors or pass them as runtime arguments to `torchlean.verify`.
Each argument keeps its own shape and dtype; the returned tensors may also have different shapes and
dtypes. The artifact records tensor argument order, including tensor leaves in built-in lists,
tuples and dictionaries. The low-level `torchlean.jaxpr.export` interface takes flat tensor
arguments. `cond`, `scan`, and `while` can preserve structured control flow in the IR.
The [branches and loops example](../verify/#branches-and-loops) runs a loop whose trip
count depends on a tensor input. `gather`, `scatter`, and their relatives become registered
operators at runtime indices. Effects and primitives without a lowering are refused by name. `div`,
`pow`, `neg`, `rsqrt` and `erf` import as IR pointwise operators, so JAX's mean and LayerNorm
expansions import; integer division rounds toward zero, as `lax.div` does. In binary16 and bfloat16,
`pow`, `rsqrt` and `erf` are computed in binary32 and rounded once, as XLA's CPU backend computes
them. A conversion imports as one cast, which rounds the exact value once; XLA's CPU backend rounds
an integer or binary64 value converted to binary16 or bfloat16 twice, through binary32, so the two
can differ near a rounding boundary. `integer_pow` imports as XLA's multiplication schedule. The
reference target has no binary error function, so a graph using `erf` passes its structure and type
checks but does not execute. JAX's exact GELU uses `erfc`, which imports as `Op.special`, the real
function; no binary format executes it, so such a graph does not execute either. Square root
imports as IEEE's root at every operand, `Op.sqrt` at `.raw`. JAX's softmax expansion imports with
its `reduce_max` and its non-finite constants, which keep only their exact element words.

### Linear algebra

For example, we can capture a Cholesky factorization and compare the reference run
with JAX on a matrix whose factor has exactly representable entries.

```python
from jax import lax
import jax.numpy as jnp
import torchlean

a = jnp.array([[4., 2.], [2., 2.]], dtype=jnp.float32)
report = torchlean.verify(lax.linalg.cholesky, a, checks=("structure", "types", "compare"))
print(report["ok"])  # True
```

Here the lower-triangular factor has rows $[2,0]$ and $[1,1]$. The importer also has
programs for triangular solves, Cholesky updates, Householder products and applications,
LU pivot permutations, and tridiagonal solves. Each has its own dtype and input
restrictions in the [IR linear-algebra reference](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/IR/README.md#jax-reductions-and-linear-algebra).

On the proof side, we can specify what it means for an output to be a QR, LU, SVD,
eigenvalue, Schur, Hessenberg, or tridiagonal decomposition. For example, the real
Schur existence theorem gives an orthogonal $Q$ and a quasi-triangular $T$ with
$A = QTQ^\top$. These relations give us statements to prove about a factorization.
Connecting a JAX import or the output of a native routine to its relation requires
a separate correctness proof. See [the factorization specifications](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Tensor/Factorization/Decomposition.lean)
and [the real Schur theorem](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/Tensor/Factorization/RealSchur.lean).

## Mixed precision

Let's make the precision changes visible:

```python
def mixed_square(x):
    y = x.astype(jnp.float32)
    return (y * y).astype(jnp.float16)

report = torchlean.verify(
    mixed_square, jnp.array([1.25, 2.5], dtype=jnp.float64), frontend="jax",
    checks=("structure", "types", "compare"),
)
```

The graph contains two casts and a multiplication. Its input is binary64, its
multiplication is binary32, and its result is binary16. A cast is an operation,
not a new label on an unchanged tensor. The exporter also records complete binary
words for constants, so loading a value does not first round it to a Python or
Lean binary64 number.

A typed program (`TorchLean.IR.Program`) records these types and the input/output
interface with its steps. Ordinary arithmetic requires matching operand and result
types. A missing cast is rejected. `TorchLean.JAX.load` imports the program under the
reference runtime as a flow of typed programs of registered operators, one program where
the capture decides every `cond` and `while`, whose run executes the numerical
nodes with FloatLib's configured arithmetic and casts with its explicit rounding semantics:

```lean
import TorchLean.NN.JAX

open TorchLean

/-- The outputs of the flow of an exported document, run on the document's arguments. -/
def runExport (document : Lean.Json) : IO (List (IR.Program.Value .reference)) := do
  let (flow, fuel) ← IO.ofExcept (JAX.load document)
  -- The arguments are typed tensor values, loaded from their exact interchange words.
  let arguments ← IO.ofExcept (IR.loadArguments .reference document)
  -- `run {}` also returns the conversion flags of the run.
  let (outputs, _flags) ← IO.ofExcept
    ((flow.eval (IR.Flow.programs .reference) fuel arguments.toList).run {})
  pure outputs
```

Try the companion's `mixed_precision.json`, `float16_sum.json`, `bfloat16_sum.json`, and
`threshold.json` with the same `torchlean jaxpr` command. The second and third show JAX's
explicit conversion to float32 before accumulation and conversion back afterward, and the
last selects entries through a Boolean comparison.

The reference runtime accepts any validated FloatLib binary descriptor, including
binary128 and custom exponent and fraction widths. JAX capture supplies
float16, bfloat16, float32, float64, and the float8 formats `float8_e5m2`, `float8_e4m3fn`,
`float8_e4m3fnuz` and `float8_e5m2fnuz`. Those are different limits: a precision can
be supported by the Lean runtime without being a JAX array dtype. Each float8 sum, difference and
product is rounded once in the format, including intermediate sums and products in reductions.
Contractions instead accumulate in binary32 and round the result to the format. Several other
operations, including division and the elementary functions, also use binary32, which holds every
float8 value, before rounding to the format. For the 20 elementary functions, such as
`exp`, `log` and `tanh`, a test checks at every word of each format that this gives the exact
value rounded once; no theorem proves it. Operations such as `atan2` and `pow` use binary64 before rounding.

`TorchLean.IR.Binary.cast_value` proves that the binary cast returns FloatLib's specified
directed conversion, including exceptional encodings. It does not verify XLA's
implementation. Reductions follow TorchLean's reference order. The exponential and
logarithm are FloatLib's correctly rounded ones in these IEEE formats, and the
trigonometric and hyperbolic functions are FloatLib's approximations. XLA's need not
agree with either. A contraction's `preferred_element_type` is accepted when that type can
represent every operand value exactly: the operands are converted to it before contraction.
Narrowing choices are rejected. Jaxpr capture also preserves Boolean, integer and complex
component types and exact constant words. Complex words alternate real and imaginary
components. The reference runtime executes Boolean and fixed-width
integer layout operations, using the same shared tensor transformations. Integers also support
negation, truncating and flooring division, wrapping powers, `remainder`, `add`, `sub` and `mul`
modulo `2 ^ width`, `max` and `min` in the type's order, comparisons, and `clamp_min`. Integer and
Boolean values cast to binary formats and between integer types, and binary values to integers in
the cast's rounding direction. The bitwise operators, shifts and bit counts act on the integer
words. A complex value is held in paired-real form, its real and imaginary parts on an appended
last axis. FFTs, complex products, and elementary functions such as `log` and `sqrt` preserve
that representation. For the complex elementary functions, FloatLib supplies the configured
binary implementation. Under the rounding theorems' premises, each component is the corresponding
principal value rounded once. The derivative certificates concern the smooth domains of the exact
functions. See [complex import and rounding](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/IR/README.md#complex-values).

The importer also accepts the special functions of `lax`, `rng_uniform` with TorchLean's generator,
and `top_k`.
The special functions (`Op.special`) are specified
over the reals only: no binary format executes them, and their derivative certificate holds for
`igamma` and `igammac` where both arguments are positive, for `zeta` where `s > 1` and `q > 0`, and
for `betainc` where `a` and `b` are positive and `x ∈ (0, 1)`.

The proofs on this page concern the accepted graph's Lean semantics. The Python
translation and the link to XLA execution remain outside those proofs.

## What the IR covers

The same IR serves general tensor programs, whether we're working on a neural network,
a numerical algorithm such as a Cholesky factorization or a triangular solve in binary32
and binary64, or a smaller calculation like the one above. A node of a graph of
registered operators (`TorchLean.IR.Graph`) is a graph input, a constant, or an operator of
`TorchLean.Op`, and the graph runs by the rules of its operators (`Graph.eval`). The
frontends reuse these meanings: JAX's `mul` and PyTorch's `aten.mul.Tensor` both import
as `Op.mul`, so a theorem about the rule of `Op.mul` applies to the products of both
graphs. `@[op_cert]` registers such a theorem as a certificate of its operator and checks
that its statement mentions that rule, and the
[Registered Operators]({{ '/blueprint/Semantics-and-Graphs/The-Operator-Table/' | relative_url }})
chapter describes the operators and their certificates.

The Jaxpr adapter and the `torch.export` capture write one format, `torchlean.ir`: a
typed program whose operator steps are registered operators (`TorchLean.IR.Artifact`),
and `Artifact.ofJson_toJson` proves that decoding the JSON of any artifact returns it.
`TorchLean.JAX.load` checks the frontend marker and loads the flow of programs with their
payloads, and the flow format (`TorchLean.IR.FlowArtifact`) holds the blocks between the branches
and loops that the capture leaves to run time.

Try changing the network's widths or the casts in the mixed-precision example and
inspect what enters the graph. I'd love to hear which computations you'd want to bring
into Lean next : )
