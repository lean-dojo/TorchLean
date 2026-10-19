---
title: Checking a Captured Computation
---

Suppose we already have a tensor calculation in Python. We'd like to bring its
operations into Lean without rewriting the model, inspect its precision changes,
and ask which checks we can run on that graph.

Pass the function and its tensor arguments directly. TorchLean captures the
frontend graph, translates supported operations into its IR, and sends it to
Lean automatically. Saving a graph file is optional.

From the TorchLean repository, build the checker first:

```bash
scripts/lake.sh build verify
```

Then try this Python example from the repository root:

```python
import torch
import torchlean

def square(x):
    y = x.float()
    return (y * y).half()

x = torch.tensor([1.25, 2.5], dtype=torch.float64)
report = torchlean.verify(
    square, x, checks=("structure", "types", "compare")
)
print(report["ok"])
print(report["checks"])
```

We get four IR nodes: a binary64 input, a cast to binary32, a multiplication,
and a cast to binary16. FloatLib executes the casts and arithmetic at those
declared precisions. The comparison checks the complete binary words against
PyTorch's output for `x`, not just numbers printed to a few decimal places.

`verify` accepts an ordinary function or an `nn.Module`. It copies the module and
input before capture so that the caller's tensors, parameters, and training mode
are preserved. It cannot isolate arbitrary external effects in Python code.
Capture uses `torch.export`, not the compiled backend's generated machine code.
Returned containers contribute an ordered list of tensor leaves. Their Python
container structure is not part of the IR interface. A non-tensor return leaf is
rejected during capture.

## Read the result

Each requested check has its own result:

| Request | A successful result means |
| --- | --- |
| `structure` | The typed artifact passes the shared dependency, arity, and shape checks. |
| `types` | Operand/result types, explicit casts, and the program interface are consistent. |
| `execute` | CPU reference execution completed for the supplied arguments. |
| `compare` | Its output types, shapes, and complete binary words match the supplied reference for those arguments. |

The default requests are `structure` and `types`. They don't run a sample comparison
or establish a mathematical property. Unknown requests are returned as `unsupported`
and make `report["ok"]` false; asking for a property must never silently become a
shape check.

If capture encounters an unsupported operation or element type, the report marks
every request `unsupported` and sets `checker` to `not_run`. No Lean check ran in
that case. Missing dependencies and invalid API arguments remain ordinary Python
errors.

Try changing a cast in an exported artifact to an identity while leaving its
output type unchanged. The type checker rejects it. Changing one reference output
word makes the comparison fail. Those are concrete errors this interface catches;
passing the checks does not establish that the program has no other mistakes.

PyTorch and the CPU reference interpreter can disagree on reduction order,
elementary-function approximations, or domain guards. For example, the IR's `log`
rejects nonpositive inputs. A failed comparison identifies a
difference on that input; it does not tell us the cause without further analysis.
The report explicitly leaves frontend equivalence, external kernels, and captured
device/layout constraints unchecked.

## Use JAX or another producer

The graph is a tensor program, not a neural-network-only format. We can capture
an ordinary tensor function without defining layers or a model class. Inputs,
constants, array operations, and outputs go through the same checks as a network.
What matters is whether the frontend can lower the operations involved.

The same checker also accepts a JAX function:

```python
import jax.numpy as jnp
import torchlean

def sum_squares(x):
    return jnp.sum(x * x)

report = torchlean.verify(
    sum_squares, jnp.array([1., 2., 3.], dtype=jnp.float32),
    frontend="jax", checks=("structure", "types", "compare")
)
```

Or pass a typed artifact dictionary directly to `torchlean.verify(artifact)`. A
producer does not need to know anything about PyTorch: the boundary is a graph
with explicit shapes, element types, casts, and constant words. Its operations
must have an IR lowering and an implementation in the selected interpreter.

Both capture adapters accept positional tensor arguments with independent shapes and dtypes:
`torchlean.verify(function, *inputs)` for PyTorch, or add `frontend="jax"` for JAX.
Argument order is recorded
in the artifact, including unused arguments. The shared parser checks the whole
interface; execution rejects missing arguments, extra arguments, and tensors with
the wrong shape or dtype. Nested input containers need explicit container rules.

JAX supports
the fragment described in [JAX Computations and Lean Proofs](../jaxpr/); the PyTorch
adapter supports explicit binary casts, tensor promotions and broadcasts, lifted
binary constants, and its existing operator subset. The typed payload loader
supports constants and linear parameters. Other parameterized layers and unsupported
types need their own exact payload loaders before execution can run them.
The default CPU reference also loads Boolean and fixed-width integer tensors, so
their reshape, broadcast, permutation, transpose, flatten and concatenation can
be executed and compared. Integers retain their complete words at any positive
declared width; signedness stays in the type. Integer arithmetic and casts between
different integer types are not implemented by this reference. Complex descriptors
can pass structural/type checks, but complex execution still needs an implementation.

Other producers can describe configured binary and decimal formats, integer
widths, posit widths, complex component types, and named element representations.
The shared `DType` decoder validates these descriptions without reducing them to
native floats. An interpreter still needs an implementation for each type it runs.
Typed custom operations similarly declare both their shape and element-type
signatures; execution requires an explicitly supplied implementation. A name or
signature alone is not executable code or a correctness proof.

Shape operations do not need floating-point arithmetic. When an interpreter supplies
storage for an element type, reshape, flatten, broadcast, permutation, transpose and
concatenation use the same tensor operations as the homogeneous evaluator. This also
lets us move nonnumeric elements without inventing arithmetic for them. Boolean and
fixed-width integer storage are supplied by `Interpreter.reference`; additional types
need their own representation. `detach` is
different: for dual-valued scalars it must clear differentiation metadata too, so
that operation retains its scalar-context semantics.

## Where the proofs enter

Keep the same capture when we want to write a theorem about it:

```python
report = torchlean.verify(square, x, export="Proofs/Square.lean")
```

The generated module embeds the artifact and defines `Square.program` through the
shared importer. `Square.program_imported` proves its connection to that artifact;
`Square.program_checked` proves the shape, type and interface checks succeed.
Open the file in Lean, or run `scripts/lake.sh env lean Proofs/Square.lean`, to
check those proofs. Export itself writes the file without launching a compiler.
Import `Proofs.Square` in a proof project to state mathematical properties of
the program, reusing TorchLean's tensor lemmas and mathlib. A person or an agent can
write those proofs; Lean checks them.

The export uses the artifact from this call, not another run of the Python function.
Its constants retain their complete element words. Existing files are never
replaced. Use a `.json` filename instead when we only want the interchange artifact.
Unsupported capture creates no export. A failed check can still leave an artifact
for inspection, but an invalid graph will not pass the generated Lean proof.

The graph is also an object we can state theorems about. The JAX example proves
permutation invariance of the real-valued sum-of-squares graph for every vector
length and every permutation. `NN.IR.Binary.cast_value` proves that the interpreter's
binary conversion returns FloatLib's specified directed rounding. Those are
mathematical results, distinct from a sample comparison.

`NN.IR.Reference.read_write` proves that decoding a supported element word and
encoding it again returns the original word. It covers all positive integer widths
and all valid configured binary formats, not just a list of hardware dtypes.

More properties need corresponding statements and proofs: for example, a numerical
error bound needs a rounding model and range assumptions; a robustness claim needs
an input region and a classification specification. This Python entry point does
not automatically construct those proofs. The shared IR lets us work on the
computation that was imported rather than maintain a separate hand-written model.

For proofs about a complete typed program, `NN.IR.Properties` provides induction
over intermediate values and conversion flags, and rules for relating two
successful executions. The interpreters and inputs can differ; the graph stays
the same. We supply mathematical rules for the relevant operations, and the
composition theorem carries them through the program and its returned tensors.
The [JAX walkthrough](../jaxpr/#prove-a-property-of-the-graph) explains how this
fits with proofs about an imported computation.

## Select a checker

The Python API uses the local built `verify` executable. Set `TORCHLEAN_VERIFY` to
an executable path, or pass `checker="/path/to/verify"`. It never launches a build
implicitly. A complete command can also be supplied as an argument list:

```python
report = torchlean.verify(
    artifact,
    checker=["scripts/lake.sh", "exe", "verify", "--", "ir"]
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
