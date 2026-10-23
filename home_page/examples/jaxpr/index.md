---
title: JAX Computations and Lean Proofs
---

I've been thinking about how we can bring Lean closer to the code people already
write for machine learning. If we have a model in JAX or PyTorch, can we keep that
code and prove something about its computations without writing the model again
in Lean? That's what we're working toward with TorchLean's intermediate
representation (IR).

The idea is to capture the tensor operations from the original code and translate
the supported ones into our IR. Then we can work with their Lean definitions,
reuse mathlib and our tensor lemmas, and bring in FloatLib when rounding matters.
We need to keep the shapes, axes, constants and precision choices along the way.
Otherwise, we could end up proving something about a different computation.

Thinking about the future, I'd love to be able to take an attention implementation
and ask whether it computes the formula we intended, or take a normalization layer
and prove which transformations leave its output unchanged. We could also study
what changes when we train or run a model at a lower precision. The model would
still run with its usual backend; we'd use Lean to work through these questions
about the imported computation.

We're still building the IR and its importers. Today we accept supported Jaxpr
and captured PyTorch graphs, check them, and run reference computations. Some
graphs also have Lean proofs, including the permutation theorem below. Importing
a graph does not itself prove the translation faithful or verify the kernels that
JAX or PyTorch executes.

The numerical issues we found were in our translation and reference paths. For
example, replacing division with multiplication by a rounded reciprocal can
change the answer. We corrected those paths or rejected captures that would
change the calculation. These examples have not identified a bug in JAX or XLA.

For an ML engineer considering lower precision, this gives us a way to
inspect where a captured model rounds its values and run that calculation in
Lean. We can measure accuracy on test data, then state a separate mathematical
property of the imported computation. The MNIST example below starts with that
familiar train-and-evaluate workflow.

Let's start with a JAX function we might already have in a project:

```python
import jax.numpy as jnp

def sum_squares(x):
    return jnp.sum(x * x)
```

Reordering the entries should not change the sum of their squares. We can test a few
permutations in Python, but here we'll state that property for every real vector and
every permutation, and prove it about the graph's Lean semantics.

We'll start small so we can follow the whole computation. After that, we'll bring
in a two-layer JAX network, including its weights and biases, through the same IR.

## Use the JAX function directly

From the TorchLean checkout, build the checker once with
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
Constants in other computations travel with the graph.

```text
input [4] → multiply input by itself [4] → sum []
```

On the Lean side, `Import.JAX.parseProgram` checks dependencies, operator arguments,
shapes, and the element type of each node. `Import.JAX.parseGraph` remains available
when we want a homogeneous graph with a scalar type chosen in Lean. Both use the
existing graph syntax; JAX does not get a separate tensor representation.

## What comes out of the pipeline?

For JAX, we capture a Jaxpr and translate its supported equations into TorchLean
IR. For PyTorch, we read a captured graph and translate its supported operations.
We are importing these frontend graphs, not XLA's or LibTorch's compiled kernels.

Here's what we get back, depending on which steps we run:

| Output | What we get |
| --- | --- |
| Optional export | The operations, dependencies, constants, tensor shapes, element types, explicit casts and designated outputs, saved as JSON or embedded in a Lean module. |
| Check report | Results for the requested checks, including failures and unsupported cases. A successful report applies to those checks. |
| Reference results | Computed output tensors when evaluation is requested, with conversion flags on the configured binary path. Sample comparisons relate these results to recorded frontend outputs. |
| Lean theorem | A separately written, kernel-checked proof about the graph's Lean meaning. Passing the checks does not generate this proof. |

For our input `[1, 2, 3, 4]`, the sum-of-squares result is `30`. Requesting `compare`
also evaluates the IR and compares its output with JAX for these arguments.
Later, we'll prove that reordering any real input
vector leaves this graph's result unchanged. That lets us say something about
inputs beyond the one we ran.

## Run the examples

The companion script captures a small MLP, sums, broadcasts, reshapes,
axis permutations, a batched contraction, and functions returning several tensors.
It reports unsupported captures by name and continues with the supported ones:

```bash
python3 NN/Examples/Interop/JAX/export.py /tmp/torchlean-jax-examples
scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/sum_squares.json
scripts/lake.sh exe torchlean jaxpr /tmp/torchlean-jax-examples/mlp.json
```

Try `--width 7 --batch 3` when capturing the examples. Those are example dimensions,
not restrictions of the importer or the permutation theorem.

The script also captures `jax.grad(sum_squares)`. Its result is another tensor
program that can enter the same IR; comparing its outputs is not, on its own,
a proof that the captured program is the mathematical derivative.

The Lean runner loads each artifact, checks its graph and constants, and compares
all designated outputs with the recorded JAX results. Homogeneous examples use
native Lean `Float`, which is binary64, at absolute tolerance `2e-5`. Mixed-precision
examples use FloatLib at each node's declared precision and compare interchange
words. These are checks of the recorded inputs, not proofs of agreement for all
inputs or a replay of XLA's chosen kernels.

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
interface with learned weights, run the MNIST companion from the checkout.
`uv run` installs the script's declared JAX and NumPy versions. Use a fresh output
directory for each run:

```bash
uv run NN/Examples/Interop/JAX/mnist.py /tmp/torchlean-mnist \
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
and run details in `summary.json`. For bfloat16, `bfloat16.json` keeps the capture,
`bfloat16-checks.json` holds the structure, type and execution report, and
`bfloat16-compare.json` holds the bitwise comparison. The other formats use the
same naming pattern.

The comparison checks every output bit. Matrix products can use different
accumulation orders in JAX and the reference interpreter, so their final bits may
differ. A mismatch needs investigation; it does not by itself identify a JAX bug.

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
`SumSquares.program` is the shared importer's result for that artifact. The file
includes proofs that the import succeeds and the program passes the shared checker.
Lean checks those proofs when we open the module or compile it:

```bash
scripts/lake.sh env lean Proofs/SumSquares.lean
```

We can import `Proofs.SumSquares` in our proof project and state further properties
about this program. The export never replaces an existing file, so rerunning a
capture cannot overwrite a proof we've started writing.

The module embeds JSON constructors rather than calling a file parser at runtime.
The import and validation proofs use kernel reduction; no `native_decide` is used.
These are import and validation proofs. The permutation argument below supplies
the mathematical property, and more involved computations need their own arguments.

For inspecting a capture in another tool, use `export="sum_squares.json"` instead.
Both exports are optional and use the artifact already sent to the checker.

## What we have checked

We ran the companion examples at width 4 with batch size 2, then at width 7 with
batch size 3. Every supported companion artifact replayed successfully at both
sizes. The checks covered several parts of a network rather than only scalar
arithmetic:

| Computation | What we checked |
| --- | --- |
| Two-layer MLP | Matrix products, bias broadcasting and ReLU; the binary64 diagnostic output passed its `2e-5` tolerance. The five-input network above also passed an exact-word comparison. |
| Batched contraction | The contraction axes lowered through shared tensor transformations and matrix multiplication; the supplied batched outputs agreed. |
| `jax.grad(sum_squares)` | JAX's generated gradient program entered the IR and its sample output agreed. This did not prove the derivative rule. |
| Reshape, transpose, broadcast and concatenation | The imported shapes and sample tensor entries agreed after the transformations. |
| Several returned tensors | Every designated output was checked in the captured order. |
| Float16, bfloat16 and mixed precision | Explicit casts and accumulation precisions were retained; the selected examples matched complete output words. |

These runs compared CPU JAX results with Lean's reference execution. They did not
measure GPU performance or verify XLA's compiled kernels. The companion's
homogeneous diagnostic evaluator uses binary64 even for a float32 capture; the
exact-word checks use the typed reference interpreter instead. A printed error
of `0.000000` in the diagnostic output is not a claim of bitwise equality.

We also checked captured PyTorch LayerNorm and a matrix-product/softmax attention
calculation on supplied inputs. Those passed sample comparisons through the
[same verification interface]({{ '/examples/verify/' | relative_url }}).
JAX expands those operations differently. A LayerNorm written with `jnp.mean` and
`lax.rsqrt` imports and executes through division and the reciprocal square root,
but softmax needs `reduce_max`, which this Jaxpr adapter does not lower. A passing
PyTorch example therefore does not imply that the corresponding JAX function can
be imported. The batched-contraction example covers attention's dot-product
linear algebra, not a complete Transformer.

## Numerical differences we found

Inspecting the captured operations helped us catch assumptions that shapes alone
would miss. The adapter gives each primitive its JAX meaning. Division and negation
import as the IR's own `div` and `neg`: replacing `x / y` with `x * (1 / y)` adds a
rounding step and can change exceptional values, and `0 - x` changes the sign of
zero. An integer power imports as the multiplications XLA performs, square-and-multiply
in JAX's order, because a correctly rounded power can differ from that product in
the last bit.

XLA's compiled CPU code does not always keep that meaning. It multiplies by the
rounded reciprocal of a broadcast divisor, so its `x / 3.0` differs from the IEEE
quotient on 1382 of 4096 binary32 samples. It also flushes subnormal values to zero,
approximates the reciprocal square root, and under `jax.jit` rewrites some divisions
and constant powers. The adapter does not copy these rewrites; it records whether the
function is one `jax.jit` call. A comparison that finds different words fails. The
checker then runs the program again as XLA computes it, and only if that run reproduces
the reference does the failure list the documented behaviour it is consistent with. A
wrong operand order next to the division by `3.0` lists nothing. The
[IR guide](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/README.md#frontend-lowering)
lists what we measured.

Precision labels also need executable meaning. We changed a multiplication's
declared output to float64 while leaving its operands float32; the type checker
rejected the artifact and requested an explicit cast. In a separate valid
float64 → float32 → float64 round trip, the input `16777217` became `16777216`.
The outputs agreed with JAX, and the interpreter reported `inexact = true`.
So both implementations agreed, even though the conversion lost the final digit.

Square root needs similar care. JAX's raw square root and the IR's guarded square
root behave differently for negative inputs and signed zero. The recognizer checks
expression patterns that exclude negative values and negative zero; a positive
capture sample cannot justify accepting the function for other inputs.

These are findings about our import and reference-execution paths. We have not
identified them as bugs in JAX or XLA. For a network, such details matter before
we attach a numerical theorem: we need its casts, domains and reduction order to
match the operations that theorem describes.

## Prove a property of the graph

The source is
[`JAX.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/JAX.lean).
Its `sumSquaresGraph` has the same three nodes as the captured example, with a
parameterized input shape. `sumSquaresGraph_denote` proves that evaluating the
complete graph over the reals returns `Tensor.sum (x * x)`.

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
named model specification. `NN.IR.Properties` supplies three composition rules.
`Interpreter.run_induction` carries an invariant on intermediate tensors and
conversion flags through a successful run. `Interpreter.run_rel` compares two
successful runs of the same program, allowing different inputs, parameter
payloads, or scalar interpreters. `Interpreter.eval_rel` carries a pointwise
relation through to the returned tensors, in their declared order.

For a norm bound, the local rules would explain how each operation changes the
bound. For a permutation property, they would explain how each operation acts on
the permuted inputs. These rules still need mathematical proofs; the common
traversal argument is now reusable. It does not require an MLP or transformer tag,
and it does not automatically produce either of those properties for the MLP
comparison above.

## Which computations can we import?

The adapter handles static tensor computations built from supported
elementwise arithmetic and elementary functions, explicit broadcasting, reshape,
transpose, concatenation, sums, and `dot_general` contractions. Contractions expose
their batch and contracted axes using permutations and reshapes before the existing
IR matrix product. Supported `jit` and custom-JVP wrappers are inlined using their
forward bodies; this does not verify their custom derivative rules.
Nondefault dot precision and algorithm choices are rejected for now. FP32 inputs and outputs
can still use BF16 or TF32 multiplication internally; we must represent that choice before
accepting it, rather than silently turning it into an ordinary FP32 matrix product.

Close over parameter tensors or pass them as runtime arguments with
`export(function, *inputs)`. Each argument keeps its own shape and dtype; the
returned tensors may also have different shapes and dtypes. The artifact records the
argument order in `input_ids`. A nested input pytree needs a container interface and is rejected.
Dynamic control flow, effects, gather/scatter, and primitives without a lowering are rejected
by name. `div`, `pow`, `neg`, `rsqrt` and `erf` import as IR pointwise operators, so JAX's mean
and LayerNorm expansions import; integer division rounds toward zero, as `lax.div` does.
In binary16 and bfloat16, `pow`, `rsqrt` and `erf` are computed in binary32 and rounded once,
as XLA's CPU backend computes them. A conversion of an integer, Boolean or binary64 value to
binary16 or bfloat16 imports as two casts, through binary32, as XLA converts it.
`integer_pow` imports as XLA's multiplication schedule.
The reference interpreter has no binary error function, so a graph using `erf` passes its
structure and type checks but does not execute, and JAX's exact GELU uses `erfc`, which has no
IR operator. Square root requires a structurally nonnegative operand, because the existing IR
square root clamps negative inputs. The capture input alone cannot establish that condition for
every input.
The adapter does not import JAX's full softmax expansion, which includes `reduce_max` and
non-finite constants, so a conventional attention function using that expansion is unsupported.

## Mixed precision

Let's make the precision changes visible:

```python
def mixed_square(x):
    y = x.astype(jnp.float32)
    return (y * y).astype(jnp.float16)

artifact = export(mixed_square, jnp.array([1.25, 2.5], dtype=jnp.float64))
```

The graph contains two casts and a multiplication. Its input is binary64, its
multiplication is binary32, and its result is binary16. A cast is an operation,
not a new label on an unchanged tensor. The exporter also records complete binary
words for constants, so loading a value does not first round it to a Python or
Lean binary64 number.

`NN.IR.Program` attaches these types and the input/output interface to the existing
graph. Ordinary arithmetic requires matching operand and result types. A missing
cast is rejected. `NN.IR.Interpreter.binary` executes the numerical nodes with
FloatLib's configured arithmetic and casts with its explicit rounding semantics:

```lean
import NN.Runtime.JAX

-- After reading the exported JSON into `artifact`:
let program ← IO.ofExcept (Import.JAX.parseProgram artifact)
let payload ← IO.ofExcept (Import.JAX.parseBinaryPayload artifact)
-- `input` is a typed tensor value, loaded from its exact interchange words.
let (outputs, conversionFlags) ← IO.ofExcept
  (NN.IR.Interpreter.binary.eval program payload #[input])
```

Try the companion's `mixed_precision.json`, `float16_sum.json`, and
`bfloat16_sum.json` with the same `torchlean jaxpr` command. The last two show JAX's
explicit conversion to float32 before accumulation and conversion back afterward.

The binary interpreter accepts any validated FloatLib binary descriptor, including
binary128 and custom exponent and fraction widths. JAX capture currently supplies
float16, bfloat16, float32, and float64. Those are different limits: a precision can
be supported by the Lean interpreter without being a JAX array dtype.

`NN.IR.Binary.cast_value` proves that a successful cast returns FloatLib's specified
directed conversion, including exceptional encodings. It does not verify XLA's
implementation. Reductions follow TorchLean's reference order, and FloatLib's
elementary functions are approximations. Heterogeneous dot products with a separate
accumulator precision are rejected until their input and accumulation rules are
represented explicitly. Jaxpr capture also preserves Boolean, integer and complex
component types and exact constant words. Complex words alternate real and imaginary
components. The default `torchlean.verify` reference can execute Boolean and fixed-width
integer layout operations, using the same shared tensor transformations. Their arithmetic
and cross-type integer casts remain unsupported, as does complex execution. The explicit
`Interpreter.binary` used above remains binary-only.

The proofs on this page concern the accepted graph's Lean semantics. The Python
translation and the link to XLA execution remain outside those proofs.

## Where we're taking the IR

I'd like us to be able to use this for general tensor programs too, whether we're
working on a neural network, a numerical algorithm, or a smaller calculation like
the one above. As we add frontends, we should be able to reuse the proofs for
operations that already have the same meaning in our IR.

We're still building more of the IR. We need more operators, ways to represent
control flow and accumulation choices, and proofs connecting the captured code to
the imported graph and its backend. I'd also like the proof to stay connected to
the implementation as we edit it, so we know when a code change means we need to
revisit the theorem.

For now, try changing the network's widths or the casts in the mixed-precision
example and inspect what enters the graph. I'd love to hear which computations
you'd want to bring into Lean next : )
