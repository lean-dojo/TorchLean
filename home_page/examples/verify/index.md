---
title: Checking a Captured Computation
---

Suppose we already have a tensor calculation in Python. We'd like to bring its
operations into Lean without rewriting the model, inspect its precision changes,
and ask which checks we can run on that graph. Put `@torchlean.compile` on the model class or
function, then call it as usual. TorchLean detects
PyTorch or JAX from the model and its tensor arguments. The first call captures the computation for
the whole tensor or batch, so you don't need a loop that checks samples one at a time. TorchLean
translates supported operations into its IR and checks the graph in Lean, while the framework
executes the captured computation on its device, with its automatic differentiation intact.

From the TorchLean repository root, install the Python package and build the checker first:

```bash
python -m pip install -e python
scripts/lake.sh build verify
```

The optional Python binding in `bindings/python` calls the same checker inside Python. Once it's
installed, `compile` and `verify` select it automatically, and passing a `checker=` argument
selects an executable instead.

Then try this Python example from the repository root:

```python
import torch
import torchlean

@torchlean.compile
def square(x):
    y = x.float()
    return (y * y).half()

x = torch.tensor([1.25, 2.5], dtype=torch.float64)
y = square(x)
print(y)
print(square.torchlean.report["ok"])
print(square.torchlean.report["checks"])
```

The captured graph retains the binary64 input, the cast to binary32, the multiplication, and the
cast to binary16. PyTorch supplies its guarded FX graph and JAX its traced Jaxpr, with the frontend
selected automatically from the tensor arguments.

Put the same tag on a model class:

```python
@torchlean.compile
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)
        )

    def forward(self, inputs):
        return self.layers(inputs)

model = MLP()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
batch = torch.ones(32, 2)
prediction = model(batch)
prediction.sum().backward()
optimizer.step()
print(model.torchlean.report["ok"])
```

For an existing model, use `model = torchlean.compile(model)` and keep calling the model on a batch
(its `parameters`, `train`, `eval`, and `state_dict` methods remain available). If its class
overrides `__call__`, decorate the class instead (so that the override is captured too). Compatible
calls reuse the captured graph, and changes that require a new specialization are captured and
checked automatically. The linear layers above keep their weights and biases as live graph inputs,
so optimizer updates reach subsequent calls, but parameterized convolution and normalization raise
during compilation because their live-state lowering isn't implemented.

A compiled callable's `.torchlean.artifact` and `.torchlean.report` describe its most recently
compiled graph. A later call can reuse a different specialization without changing these fields. To
see all successfully checked specializations (in capture order), inspect `.torchlean.graphs`.

The default checks validate structure and types. To compare the IR's CPU reference execution with
PyTorch on specific arguments, call `verify` with those arguments:

```python
report = torchlean.verify(
    square, x, checks=("structure", "types", "compare")
)
print(report["ok"])
print(report["checks"])
```

For this comparison, FloatLib executes the casts and arithmetic at their declared precisions, and
the checker compares the complete binary words against PyTorch's output for `x` (not just numbers
printed to a few decimal places).

`verify` accepts an ordinary function or an `nn.Module`. It copies the module and input before
capture so that the caller's tensors, parameters, and training mode are preserved, but it can't
isolate arbitrary external effects in Python code. Its capture uses `torch.export` and doesn't
inspect generated machine code. Returned containers contribute an ordered list of tensor leaves
(their Python container structure isn't part of the IR interface), and a non-tensor return leaf is
rejected during capture.

## Read the result

Each requested check has its own result:

| Request | A successful result means |
| --- | --- |
| `structure` | The typed artifact passes the shared dependency, arity, and shape checks. |
| `types` | Operand/result types, explicit casts, and the program interface are consistent, and the program gives every value the capture lowered the element type and shape that PyTorch or JAX recorded for it (`declared_types`). |
| `execute` | CPU reference execution completed for the supplied arguments, and every result was written to `report["outputs"]`. |
| `compare` | Its output types, shapes, and complete binary words match the supplied reference for those arguments. Two NaN results match whatever their sign and payload, which the IR leaves unspecified. |
| `compare_bitwise` | As `compare`, with NaN words compared too. |

After an execution, `report["outputs"]` holds each result with its `element_type`, `dtype`, `shape`
and complete words in `bits`. A result of index tuples, such as the one `argmax` returns, has
`element_type` `"index"`, a null `dtype`, its `port` with the index bounds, and the components of
its tuples in `bits`. `report["conversion_flags"]` gives the complete conversion status of the run:
`inexact`, `overflow`, `underflow`, `saturated`, `wrapped`, `mapped_special` and `invalid`, each
`true` when a conversion of the run raised it. For example, int16 `256` converted to uint8 gives
`0`, with `wrapped` true.

`compile` runs `structure` and `types`, which are also `verify`'s default requests. Those two
checks don't run a sample comparison, prove frontend equivalence, or establish a mathematical
property for every input. Unknown `verify` requests are returned as `unsupported` and make
`report["ok"]` false, because asking for a property must never silently become a shape check.

`compile` raises if capture or lowering is unsupported, or if a required check fails, and it
doesn't run an unchecked fallback. When you call `verify` yourself, unsupported capture instead
marks every request `unsupported` and sets `checker` to `not_run` (no Lean check ran in that case).
A document with no UTF-8 JSON text (a NaN, a value JSON lacks, a lone surrogate) also gets
`not_run`, with every request `failed`, and no export.
Missing dependencies and invalid API arguments remain ordinary Python errors.

Try changing a cast in an exported artifact to an identity while leaving its output type unchanged:
the type checker rejects it. Changing one reference output word makes the comparison fail. Those
are concrete errors this interface catches, but passing the checks doesn't establish that the
program has no other mistakes.

PyTorch and the CPU reference run can disagree on reduction order or elementary-function
approximations (the IR's binary `exp` and `log` are correctly rounded, for example, while PyTorch's
vectorized kernels can differ from them in the last bit). The capture also translates Python's
`s / x` to IR true division, while eager PyTorch can compute `reciprocal(x) * s`, with an extra
rounding. A failed comparison identifies a difference on that input, but it doesn't tell us the
cause without further analysis.

The JAX importer records supported operations with their shapes, dtypes and casts, and the CPU
reference evaluates them with TorchLean and FloatLib arithmetic. XLA's compiled CPU code can
differ: it multiplies by the rounded reciprocal of a broadcast divisor, under `jax.jit` rewrites
some divisions and constant powers, approximates the reciprocal square root, and flushes subnormal
values to zero. Every mismatch fails. When XLA's CPU backend computed the reference, the checker
then runs the program again the way XLA's code computes it, and if that run reproduces every
differing element, the failed entry lists those behaviours in `consistent_with`. That list is
advice, not an excuse. An importer error with the same effect can't be ruled out, and some give
exactly XLA's words, such as a wrong operand order that a `jax.jit` rewrite undoes. For a function
that uses `jax.jit`, `torchlean.verify` also records the outputs of evaluating it one primitive at
a time (without the rewrites), and nothing is listed unless the IR agrees with those too or differs
from them in the same documented ways. An importer bug next to a division by `3.0` lists nothing.
Our [IR
guide](https://github.com/lean-dojo/TorchLean/blob/main/TorchLean/NN/IR/README.md#frontend-lowering)
gives the measured differences and the rule. In the report, frontend equivalence, external kernels,
and captured device/layout constraints are marked unchecked.

## Use JAX or another producer

A captured graph is a tensor program, not a neural-network-only format. We can capture an ordinary
tensor function without defining layers or a model class, and its inputs, constants, array
operations, and outputs go through the same checks as a network. What matters is whether the
frontend can lower the operations involved.

Use the same decorator for a JAX function:

```python
import jax.numpy as jnp
import torchlean

@torchlean.compile
def sum_squares(x):
    return jnp.sum(x * x)

value = sum_squares(jnp.array([1., 2., 3.], dtype=jnp.float32))
print(sum_squares.torchlean.report["ok"])
```

JAX uses `jit` semantics for pure functions, so pass changing weights as tensor or pytree
arguments. Mutating a closed-over value alone doesn't refresh a cached specialization.

Or pass a typed artifact dictionary to `torchlean.verify(artifact)`. A producer doesn't need to
know anything about PyTorch, because the boundary is a graph that spells out its shapes, element
types, casts, and constant words. Every operation in it must import as a registered operator that
the selected target can run.

Both capture adapters accept tensor arguments with independent shapes and dtypes, including
arguments inside built-in lists, tuples, and dictionaries. Call a compiled function with its usual
positional and keyword arguments. To run a check yourself, use
`torchlean.verify(function, *inputs, example_kwargs={...})`, which selects the frontend
automatically too. `verify` records tensor argument order in the artifact (unused arguments
included). The shared parser checks the whole interface, and execution rejects missing arguments,
extra arguments, and tensors with the wrong shape or dtype. Scalar arguments specialize the capture
to their supplied values.

JAX supports the fragment described in [JAX Computations and Lean Proofs](../jaxpr/), and the
PyTorch adapter supports explicit binary casts, tensor promotions and broadcasts, lifted binary
constants, and its existing operator subset. For constants and for the parameters of linear,
convolution, layer and batch normalization nodes, the typed import reads their exact words. Our
default CPU reference also loads Boolean and fixed-width integer tensors, so their reshape,
broadcast, permutation, transpose, flatten and concatenation can be executed and compared. Integers
retain their complete words at any positive declared width (signedness stays in the type). On
integers, the reference supports addition, subtraction and multiplication modulo `2 ^ width`,
negation, truncating and floor division, wrapping powers, remainders, `clamp_min` and comparisons,
and conversions between integer types modulo `2 ^ width`. Complex descriptors can pass
structural/type checks, but complex execution needs an implementation.

Other producers can describe configured binary and decimal formats, integer widths, posit widths,
complex component types, and named element representations. Our shared `DType` decoder validates
these descriptions without reducing them to native floats. A runtime still needs an implementation
for each type it runs. Typed custom operations similarly declare both their shape and element-type
signatures, but executing one requires an implementation supplied for it. A name or signature alone
is not executable code or a correctness proof.

Shape operations don't need floating-point arithmetic. When a runtime supplies storage for an
element type, reshape, flatten, broadcast, permutation, transpose and concatenation use the same
tensor operations at a nonnumeric element type as at a numeric one, which also lets us move
nonnumeric elements without inventing arithmetic for them. Booleans and fixed-width integers are
stored natively in a typed program (as `Bool` and `BitVec`), and additional types need their own
representation. `detach` is different: for dual-valued scalars it must clear differentiation
metadata too, so that operation retains its scalar-context semantics.

## Where the proofs enter

When you want to write a theorem about a graph, use `verify` to capture and export it:

```python
report = torchlean.verify(square, x, export="Proofs/Square.lean")
```

The generated module embeds the capture document and defines `Square.program`, the typed program of
its artifact. `Square.artifact_decoded` proves that `Square.artifact` is the decoding of that
document's artifact, and `Square.program_checked` proves `Program.check .reference`, the program's
dependency, shape, type and interface checks. Open the file in Lean, or run
`scripts/lake.sh env lean Proofs/Square.lean`, to check those proofs (export itself writes the file
without launching a compiler). Import `Proofs.Square` in a proof project to state mathematical
properties of the program, reusing TorchLean's tensor lemmas and mathlib. A person or an agent can
write those proofs, and Lean checks them.

Export uses the artifact from this call, not another run of the Python function, and its constants
retain their complete element words. Existing files are never replaced. When you only want the
interchange artifact, use a `.json` filename instead. Unsupported capture creates no export. A
failed check can still leave an export for inspection. The report also validates the constant
payload and `declared_types` through `IR.load`; the generated module includes no proof of those
load checks.

We can also state theorems about the graph itself. Our JAX example proves permutation invariance of
the real-valued sum-of-squares graph for every vector length and every permutation, and
`TorchLean.IR.Binary.cast_value` proves that the binary conversion of a typed run returns FloatLib's
specified directed rounding. Those are mathematical results, distinct from a sample comparison.

`TorchLean.IR.Reference.read_write` proves that decoding a supported element word and
encoding it again returns the original word. It covers all positive integer widths
and all valid configured binary formats, not just a list of hardware dtypes.

More properties need corresponding statements and proofs. A numerical error bound, for example,
needs a rounding model and range assumptions, and a robustness claim needs an input region and a
classification specification. This Python entry point doesn't construct those proofs automatically.
With the shared IR, we can work on the computation that was imported, without maintaining a
separate hand-written model.

For proofs about a complete typed program, `TorchLean.IR.Program.run_eq_foldlM` states
that a run of the typed program of its import is the fold of its steps, so a
predicate on intermediate values and conversion flags follows one step at a time.
We supply mathematical rules for the relevant operations, and the fold carries
them through the program and its returned tensors.
The [JAX walkthrough](../jaxpr/#prove-a-property-of-the-graph) explains how this
fits with proofs about an imported computation.

## Select a checker

The Python API uses the locally built `verify` executable and never builds it for you. Set
`TORCHLEAN_VERIFY` to an executable path, or pass `checker="/path/to/verify"`. You can also supply
a complete command as an argument list:

```python
artifact = square.torchlean.artifact
report = torchlean.verify(
    artifact,
    checker=[".lake/build/bin/verify", "ir"]
)
```

For an artifact already on disk:

```bash
scripts/lake.sh exe verify -- ir model.json structure types
scripts/lake.sh exe verify -- ir model.json compare
```

The CLI emits the same JSON report and exits unsuccessfully when a requested check
fails or is unsupported. A comparison artifact supplies exact binary tensors in
`arguments` and `reference_outputs`, each with `dtype`, `shape`, and `bits` fields.
