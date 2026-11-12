# PyTorch interop

`TorchLean/NN/PyTorch` moves weights and graphs between PyTorch and Lean. The capture of
PyTorch graphs lives in the Python package `python/torchlean`. PyTorch adds no semantics of its
own: whatever it produces (code, weights, graphs or JSON), TorchLean parses it, checks it against
TorchLean's shapes and supported operators, and connects it to TorchLean's IR and specifications.
PyTorch's autograd is outside the trusted proof boundary.

We use PyTorch for practical reasons. Many workflows train in PyTorch and keep checkpoints or
`state_dict`s, and the bridge gives those parameters a path into Lean for verification and proofs.
PyTorch is also a practical runtime for trying models, training on real data and checking that
shapes and numerics behave as expected.

The export and import halves below are separate modules, and a file imports the ones it uses. From
Python, `torchlean.verify(model, example)` captures a module and runs the requested Lean checks.
`export="model.json"` saves the artifact, and `export="Model.lean"` embeds it in a Lean module.

The optional LibTorch execution provider is separate from these files, and the
[backend chapter](https://lean-dojo.github.io/TorchLean/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/)
describes it.

## Capture

From the repository root,

```bash
PYTHONPATH=python python3 -m torchlean.export model.py Net graph.json --example-shape 1,4
```

builds the model with the zero-argument constructor `Net` of `model.py` in evaluation mode,
captures it at a `torch.randn` example of shape `[1, 4]` and writes its capture document to
`graph.json`. With `--require-torch-export` the command fails where the capture would fall back to
FX. From Python, `torchlean.export.export_torchlean_graph_json(model, example, path)` writes the
same document and returns it.

The lowering reads graphs in `torch.export`'s default form and in core ATen after
`run_decompositions()`. `torchlean.export` takes the model's `torch.export` graph when every node of
it lowers and FX's symbolic trace otherwise, and with `prefer_torch_export=False` it skips
`torch.export`. `torchlean.verify` captures the `torch.export` graph and checks it, and
`@torchlean.compile` lowers the graph that `torch.compile` hands its backend. The modules of
`python/torchlean` split the work:

- `_fx` reads the dtypes, shapes, operator names and parameter payloads of FX nodes.
- `_bind.bind_arguments` gives every call's keywords PyTorch's parameter names, once per graph and
  before any lowering reads them. PyTorch's parser accepts NumPy's `axis`, `keepdims`, `x`, `a`,
  `x1` and `x2` for `dim`, `keepdim`, `input` and `other`, so `torch.sum(x, axis=1, keepdims=True)`
  is `torch.sum(x, dim=1, keepdim=True)`. A call with keywords is refused when it gives one
  parameter twice, or when its arguments bind to no overload or to overloads that name them
  differently.
- `_pointwise` plans the pointwise operators with their Python scalar operands and expands the
  plans into capture nodes.
- `_structure` plans indexing, scatters, statistics, random draws and buffer updates.
- `_emit` gives each call its node kind and expands the structural plans into capture nodes.
- `_graph` chooses the graph and resets the module state that a capture fills, one capture at a
  time.
- `_artifact.lower` writes the capture nodes as the steps of a typed artifact of format
  `torchlean.ir` (`TorchLean.IR.Artifact`): inputs, constants with their words, casts and registered
  operators, each step at its element type. The JAX capture (`jaxpr.py`) calls it too.
- `export` writes the document
  `{"artifact", "frontend": "pytorch", "capture_constraints", "declared_types"}`.
  `declared_types` gives each lowered tensor value's step output and the element type and shape
  that FX's metadata records for it, and `TorchLean.IR.load` refuses a program that gives one of
  them another type. `torchlean.verify` adds the `"arguments"` for an execution and the
  `"reference_outputs"` for a comparison.

The facts that the capture shares with Lean come from Lean's op table (`TorchLean.IR.table`,
`TorchLean/NN/IR/Table.lean`), which `torchlean._table.op_table()` reads once per process.
It reads the file `$TORCHLEAN_OP_TABLE` when that variable is set, else the table that the default
checker of `torchlean.verify` writes: `$TORCHLEAN_VERIFY ir --op-table`, the native binding's
`op_table`, or the repository's `.lake/build/bin/verify ir --op-table`. The table gives the format
name, the record version, the JSON of each element type, the codec's codes (the bitcast word and
cast source codes among them) and the steps of every pointwise operator at each operand and result
type.

## ONNX

`python3 -m torchlean.onnx MODEL.onnx OUT.json` (`python/torchlean/onnx.py`) reads a static ONNX
graph and writes it as a typed artifact (`TorchLean.IR.Artifact`) of registered operators. Each step
runs at the floating element type (binary16, bfloat16, binary32 or binary64) of the ONNX value it
computes, and each initializer the graph reads as data is a constant holding its own words.
Broadcasting becomes `Op.broadcast` steps, by each operator's ONNX version (NumPy's, or `broadcast`
and `axis` below opset 7). With `--sources SOURCES.json` it also writes the ONNX source of each
step. `Conv` is one `Op.conv` with its groups and every leading axis, and inference
`BatchNormalization` is `(x - mean) / sqrt(var + epsilon) * scale + bias` in that order. A
`ReduceSum` or `ReduceMean` is one reduction over all its axes (a row-major fold in the element
type), and `Softmax` below opset 13 keeps ONNX's coercion to two axes. Its format name, record
version, codes and element types come from Lean's op table (`op_table()`). Lean reads the result
with `Artifact.ofJson`, `Artifact.load` and `Program.eval`, a replay in the source precision.
`torchlean.onnx` rejects integer and Boolean data, nodes mixing element types, a mean over an empty
fiber and custom operator domains. Integer shape and reduction-axis initializers remain metadata.
Square roots, BatchNorm's included, are IEEE's root at every operand, `Op.sqrt` at `.raw`.

## Export

`Export.lean` and `Export/` hold the Lean modules that write PyTorch code.

- `Export.lean` holds the shared string, import and header helpers, and
  `PyTorch.checkpoints`, which emits Python functions that save and load `state_dict`
  checkpoints. `python/torchlean/state.py` writes a `state_dict` as the document that
  `TorchLean.PyTorch.state` reads.
- `Export/IR.lean` is the general model code path: it writes an operator graph with its
  payload (`TorchLean.IR.Imported`) as a standalone PyTorch module (`PyTorch.emit`).

`Export/IR.lean` writes an operator with a PyTorch function as one call of it, by its
`TorchLean.Op` constructor: the inputs in port order and the attributes as PyTorch's keywords
(`dim`, `dims`, `shape`, `rounding_mode`). Its batch normalization is the formula of
`Tensor.batchNorm`: the variance clamped at zero, plus epsilon, clamped again, and the centered
input divided by the root. It has its own code for graph inputs, constants and random draws, for the
parameters of `linear`, convolution and the normalizations, and for layouts PyTorch has no one call
for (pooling, `gather`, `scatter`, `slice`, `pad`, `broadcast`, `contract`, `indices`). The
reductions have their own code too: PyTorch names each `reduce` operation separately and reads an
empty axis list as every axis, and a product over axes is a loop over each fiber in row-major order,
which `torch.prod` would regroup. So do the IR's own formulas (sigmoid, SiLU, softplus, GELU,
`safe_log`, the clamped `sqrt`, the masked softmax, `top_k_mask`). The tuple results of `cummax`,
`cummin`, `sort` (a stable sort) and `topk` (the stable sort's first `k`) have their own code as
well, which keeps their index outputs. The GELU formula calls `torch.nn.functional.gelu` with its
`approximate`. SiLU and GELU return their limits, `+∞` at `+∞` and `-0` at `-∞`, and evaluate their
formula at zero there, so that no autograd slope is a NaN. The emitted slope at `-∞` is then `+0`,
where the operator's derivative gives `-0`. The extrema and `relu` give a zero result
IEEE 754-2019's sign, a mean over an empty fiber and the mean squared error of empty inputs are
`+0`, and a random node reads the draws `TorchLean.Random.uniform` makes for its key from a buffer
that `emit` writes. A truncating division floors the exact quotient `|a| / |b|` and takes the sign
of `a / b`, as the operator does, since `rounding_mode='trunc'` truncates the rounded quotient: `10`
at `(1, 0.1)`, where the operator gives `9`.

## Wire records

A `torchlean.ir` artifact records each registered operator by its codec record
(`TorchLean/Tensor/Op/Codec.lean`): `Op.encode` writes the record version
(`Op.Wire.formatVersion`, 3), the operator's tag and its implicit shapes and attributes, one field
each (`Op.Field`). `Op.decode` reads a record of that version back and rejects any other record,
version or tag. The round trip is proved in Lean:

```lean
theorem Op.decode_encode (op : Σ ins outs, Op ins outs) : Op.decode (Op.encode op) = some op
```

It is about the codec only. The JSON reading (`TorchLean.IR.Artifact.ofJson`, with its round trip
`TorchLean.IR.Artifact.ofJson_toJson`), the program check (`TorchLean.IR.Artifact.load`, with
`Artifact.load_ok`) and an operator's meaning are separate.

The capture writes a pointwise operator from the op table's entry for its name
(`TorchLean.PointwiseOp.name`). GELU's `approximate` and a rounded division's `rounding_mode` select
the entry (`TorchLean.PointwiseOp.keyword`), and the entry's form for the operand and result types
gives the steps, at the node's shape. `_artifact.lower` counts a negative dimension from the end of
the shape it indexes and writes axes, shapes and convolution geometry as fields of the record. Each
parameter tensor, such as the weight and bias of a `linear` node, becomes a constant step holding
its words. The format name is `TorchLean.IR.formatName`, which the capture reads from the op table.

## Captured operators

The capture matches complete ATen overload names and listed FX callables or methods. For
example, `aten.log.default` lowers to `log`, while `log1p`, `log2` and `log_softmax` have lowerings
of their own, and `aten.slice.Tensor` with a positive step lowers to one `slice`. Each operator
becomes one node with its axes as attributes: `amax`, `sum.dim_IntList` and `prod` one `reduce`,
`mean.dim` one `mean`, `_log_softmax` one `log_softmax`, `constant_pad_nd` one `pad`, `expand` one
`broadcast`, and `flip`, `cat`, `permute` and `slice.Tensor` one node each. The value components of
`native_layer_norm`, the inference batch norm and max pooling with indices, and every component of
a split, become the node itself. Another component is refused, except the index outputs described
below. Creation operators are constants PyTorch computes from the same arguments, but a binary16
or bfloat16 fill value of `full`, `full_like`, `new_full`, `fill` and `scalar_tensor` is rounded
once to that format, where PyTorch rounds through binary32. FX evaluates an input-independent
creation while tracing, so under FX the constant is the tensor PyTorch made.

The mixture-of-experts routing idiom `torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices, 1)`
lowers to one `top_k_mask` node, and the `topk` call and the zeros it scatters into must have no
other users. For partial selection (`0 < k < width`), the capture document's `capture_constraints`
records the assumption that each score row is finite and that its `k`-th largest score is strictly
above its `(k+1)`-th. Without that gap, PyTorch may select different tied experts than the IR's
lower-index rule. Selecting zero or all entries needs no gap. The check report keeps these
constraints with `capture_constraints_status="not_checked"`: a successful structure, type or
sample check does not prove them for other inputs.

The capture refuses mutating overloads, `view(dtype)` reinterpretation, `dtype` arguments of
reductions and softmax, and unsupported tuple producers, and writes no artifact for them. The
mutating overloads it accepts are an `index_put_` of a fresh destination, a `detach_` of a value
computed from constants (folded) and training batch normalization's `num_batches_tracked` update
(below).

The core ATen pointwise operators `neg`, `rsqrt`, `erf` and `gelu`, every core overload of `div` and
`pow`, `expm1`, `log1p`, `exp2`, `log2`, `log10`, `tan`, the hyperbolic functions and their
inverses, `asin`, `acos`, `atan`, `atan2`, `hypot` and `remainder` become IR pointwise nodes. So
do the exact `floor`, `ceil`, `round` (zero decimals), `trunc`, `frac`, `sign`, `copysign`, `fmod`,
`fmax` and `fmin` of floating tensors. Each means its mathematical definition or IEEE 754-2019's
operation (`rsqrt` is `1 / sqrt x` rounded twice, so binary64's `rsqrt 3` is one ulp above IEEE's
`rSqrt 3`). Where PyTorch's CPU kernels compute something else, the capture does not copy them, and
the IR guide lists the differences. Among them, `remainder` gives a zero result the divisor's sign,
as `torch.remainder` documents (Python's `%`) and eager PyTorch does not. `fmax` and `fmin` return
the values of IEEE 754-2019 `maximumNumber` and `minimumNumber`, with `+0` above `-0`, a case
PyTorch's documentation leaves open. The tanh `gelu` computes `√(2 / π)` in the format, TorchLean's
choice of schedule, one ulp below PyTorch's once-rounded constant in binary32.

TorchLean's binary kernels for the elementary functions can differ from PyTorch's in the last bits.
`asin`, `acos`, `atan` and `atan2` are rounded correctly, and on the measured samples PyTorch 2.13's
binary32 and binary64 kernels on x86-64 are within one ulp of them. Binary16 and bfloat16 inputs are
widened to binary32 where those kernels compute in binary32. Reduced-precision products with a
one-element operand of another dtype, and quotients with a one-element operand, keep that operand in
binary32 on either side, so `torch.mul(s, x)` is `x * s`, where PyTorch converts a left scalar to
the tensor's dtype. Python's `s / x` is true division, where PyTorch computes `reciprocal(x) * s`.
`pow` of a floating result with a scalar exponent or base is one FloatLib power of the scalar's own
value in the computation type. PyTorch instead dispatches on the exponent rounded to the tensor's
dtype: products for `2`, `3` and `-2` and square roots for `±0.5` in binary32, binary64 and
bfloat16, while binary16 calls its own `pow`.

A conversion is one IR cast, which rounds once, also from an integer or binary64 source to binary16
or bfloat16, which `c10::convert` rounds twice, through binary32. For one, binary64
`1 + 2^-8 + 2^-40` (bits `0x3FF0100000001000`) becomes binary32 `0x3F808000` there. That is the
bfloat16 midpoint `1 + 2^-8`, and ties to even give `0x3F80` (`1.0`). The IR gives the correctly
rounded `0x3F81` (`1.0078125`), so a comparison with PyTorch rejects such a conversion, by design.
To an integer type, a floating value truncates toward zero, a truncated value the type cannot hold
is an execution error, and integer values wrap modulo `2 ^ width`. A Python scalar operand, of these
operators or of `add`, `sub` and `mul`, and a scalar bound of `clamp_min` and `clamp_max` become
constants rounded once from the int64, uint64, binary64 or Boolean value PyTorch wraps them in. An
integer beyond `2^53` therefore rounds once to binary32, binary64, bfloat16 or binary16.
`clamp_min`, `clamp_max` and `clamp` with one bound, scalar or tensor, are IEEE maximum and minimum,
`max_elem` and `min_elem`, with the bound converted to the operation's result type. So an int32
tensor clamped at `0.5` gives binary32, `clamp_min(-0.0, 0)` is `+0.0` (PyTorch's kernel keeps
`-0.0`), and a NaN bound gives NaNs. The IR's `relu` returns `+0.0` at `-0.0`, where eager PyTorch
returns `-0.0`. The tanh GELU uses the arithmetic order of PyTorch's CPU kernels at the DEFAULT
capability level, which `ATEN_CPU_CAPABILITY=default` selects and which rounds `κ x³` and `x + κ x³`
separately. At the AVX2 and AVX-512 levels PyTorch fuses that sum on x86-64, and we have not
measured other processors or CUDA. The [IR guide](../IR/README.md#frontend-lowering) lists the rules
and the elements that PyTorch's scalar loop computes differently.

`cumsum` and `cumprod` scan in PyTorch's accumulation type (binary64 for binary32, binary32 for
binary16 and bfloat16) and round each prefix. `cummax` and `cummin` keep the later of tied entries
and the later NaN. `sort` with `stable=True`, and `max.dim` and `min.dim` (a stable `topk` of one
entry, between negations for `min.dim`), keep the first of tied entries, as ATen does. These
tuple-valued operators of a rank-0 operand are refused, values and indices alike. `topk` is the
IR's stable `topk`, its indices `Op.topk`, where ATen's result is uniquely determined. For `k > 0`,
execution fails where two adjacent entries of a fiber's first `min(k + 1, n)` in the requested
order tie (equal values, `-0` and `+0`, two NaN), which ATen orders either way. `topk` with
`sorted=False` and `k > 1`, and `sort` without `stable=True`, are refused. The indices of
`max_pool2d_with_indices` and `max_pool3d_with_indices` are each window's first maximum or last
NaN, as `cpu_max_pool` takes them on contiguous operands, through constant gathers and
`Op.argmax` (a window of padding only and an operand without planes are refused). The values are
the IR `pool`, which takes `+0` over `-0` as the IEEE 754-2019 `maximum` does, where `cpu_max_pool`
keeps the first of the two. So a window of `-0` before `+0` gives the value `+0` and the index of
`-0`.

`leaky_relu` with a positive slope is `maximum(x, x * slope)`, or `minimum(x, x * slope)` above one.
Comparisons, `isnan`, `isinf`, `isfinite` and the logical and bitwise operators of floating or
Boolean operands are Boolean nodes, and `where` and `masked_fill` select with a Boolean condition.
`clamp`, `clip` and `hardtanh` are the IR `clamp`, and `elu` and `softplus` are their kernels'
selections.

`diagonal` is a permutation, a reshape and a strided slice. `index_select`, `gather`, `scatter`
and `scatter_add` with indices from buffers or constants become one `gather` or `scatter` node
holding every coordinate. `argmax`, `argmin` and the index outputs of `max.dim`, `min.dim`,
`cummax`, `cummin` and the stable `sort` of floating operands become steps of the registered
operators `Op.argmax`, `Op.argmin`, `Op.cummax`, `Op.cummin` and `Op.sort`, with int64 results.
For `argmax` and `argmin`, `dim=None` reduces the row-major flattening and a scalar is one entry.
An int64 or int32 index computed from the inputs makes the same kind of step. `embedding` is a step
of `Op.embedding`. `index_select`, `gather`, `take_along_dim` and `index.Tensor` are steps of
`Op.gather`, with a permutation where PyTorch places the selected axes, and `scatter`,
`scatter_add` and `index_put` are steps of `Op.scatter`. A negative index of `index.Tensor` or
`index_put` counts from the end, `take_along_dim` with a dimension reads its index modulo the
extent, and an index outside its axis fails at execution, as PyTorch raises.

A repeated target of a runtime scatter or `index_put` folds in row-major order of the index tensor,
which PyTorch leaves unspecified. PyTorch's CPU `index_put` with `accumulate=True` departs from
that order on large inputs, and `index_put` without `accumulate` at a repeated target is documented
as undefined. `scatter_add` and `scatter_reduce` sums and products of binary16 and bfloat16
accumulate in binary32 and round once, as the CPU kernel does (measured), and `index_put` with
`accumulate` rounds each step in the dtype, as its kernel does. A Boolean mask index is refused.
`scatter_reduce` with `sum` or `mean` is a scatter, a count and a division. With `prod`, `amax` or
`amin` it is one scatter with the rule `prod`, `max` or `min`, folding in the index's order along
`dim` as `scatter_reduce_two` does outside its expanded-index path. The rules `max` and `min` are
IEEE maximum and minimum, where ATen's kernel keeps the target at a tie, so its maximum of `-0` and
`+0` depends on their order. An index whose producer is `aten.expand` is refused along dim 0 for a
product, and an index expanded otherwise is outside this claim. Where `include_self` is false the
base entry at a written target takes no part: a product starts from `1`, a maximum or minimum from
the target's first update. A product's pullback multiplies the others. A maximum's or minimum's
shares a tie evenly among the participants. PyTorch's backward also counts an excluded base entry
equal to the result (checked against torch autograd in `Float` and `Float32` for vectors, not in
the binary formats). A repeated target of a constant scatter is refused.

The default graph's `index_put_` is `index_put` when its destination is fresh, its indices and
values are not the destination, no later node reads it and its earlier readers allocate their
results. A fresh destination is an audited creation operator's result, an accepted `index_put_`,
or a functional core or pointwise operator's result without an alias annotation and not tagged
`maybe_aliasing_or_mutating`. A write into an input, a view or a possibly aliased tensor is
refused, and so is an in-place operator whose result is unused, a dead update of fresh scratch
storage included. The default graph also accepts `add_` of the integer 1 to an int64 scalar buffer
whose only reader is that operation, its result possibly used. This structural rule includes the
batch norm's `num_batches_tracked` increment and computes the updated value out of place at
capture.

`slice_scatter`, `select_scatter` and `col2im` (`fold`) are constant scatters. `any` and `all`
compare a binary32 sum of zeros and ones with zero, empty fibers giving false and true. Supported
deterministic integer and Boolean operations with constant operands, such as an `arange` or a
causal mask, are folded at capture only when their schemas declare no mutation. `detach_` of such a
value (a tensor literal is `lift_fresh_copy` then `detach_` in the default graph) folds to its
operand's value, since the capture keeps forward values, not autograd state. Comparisons of
computed integers read their values, integer `add`, `sub` and `mul` wrap modulo `2 ^ width` and
integer `maximum` and `minimum` follow the type's order, as ATen's integer kernels do. Bitwise
operators and shifts of integers act on their words, `>>` filling with the sign bit of a signed
type and zeros of an unsigned one, and an amount of the width or more (PyTorch: undefined) shifts
every bit out. The index outputs of an unstable `sort` are refused: ATen orders equal entries
either way there.

`scaled_dot_product_attention` of the default graph is its docstring's equivalent code (a row with
every key blocked is NaN there, where the core graph's math decomposition gives it zero weights),
binary16 and bfloat16 in binary32, and `dropout_p > 0` is refused. Adaptive average pools follow
the CPU kernel's windows and divisions, and nearest upsampling gathers at PyTorch's indices.
`view_as_real` of `rfft` and `irfft` of `view_as_complex` are `Op.rfft` and `Op.irfft` on the
paired-real form. Group norm, the training batch norm (its running statistics below), `cdist`,
`pdist`, linear and bilinear upsampling and `grid_sampler_2d` (bilinear or nearest, zeros or border
padding) are compositions. For CPU grid sampling, nonfinite coordinates with zeros padding give
NaN in bilinear mode and zero in nearest mode. Border padding clamps them to an edge; a singleton
axis reads its only pixel. The tests compare these compositions with the CPU kernels on their
fixtures within stated tolerances (no uniform error bound). `masked_scatter`, `nonzero_static` and
`_embedding_bag` with constant offsets are captured too, and `nonzero` (a data-dependent shape) is
refused.

`rand`, `randn`, `randperm` and training-mode dropout use TorchLean's generator under keys recorded
in the artifact, not PyTorch's stream. `aten.randn.default` of the four floating dtypes is
`Op.randn`, a deterministic Box-Muller approximation in the dtype's arithmetic on two 32-bit draws
per entry, each rounded to the dtype, `u1` clamped below by `1e-6` rounded to it, with no exact
normal law. The generator overloads are refused and `aten.randn_like` is not captured directly.

LayerNorm and BatchNorm nodes carry their weights, biases, eps and running statistics as
interchange words, and a mean over an empty fiber is refused. A training batch norm with running
statistics computes the batch moments and returns the supported buffer updates as outputs. The
input and both statistics must have one type, binary32 or binary64. The update rounds `momentum`
to that type before forming `1 - momentum` and divides the sum of squared deviations by `n - 1` for
`running_var`. Only the updates the graph performs are returned (`torch.nn.functional.batch_norm`
itself does not increment `num_batches_tracked`). Updated buffers sharing storage with another
captured tensor or with an input are refused. By TorchLean's convention the program's own outputs
come first, then each updated buffer in the exported signature's BUFFER input order. PyTorch's
unlifted module returns only its own outputs. Captured buffers are constants, so each execution
starts from the captured state. The tests compare the floating outputs with eager CPU execution at
`rtol = atol = 1e-5` on their fixtures.

## Import

`Import.lean` and `Import/` read JSON weights into Lean.

The bridge uses JSON because Python writes it with its standard library and Lean parses it without
depending on Python's pickle formats.

- `Import.lean` defines `parseTensor`, which turns nested JSON arrays into a `Tensor Float s`
  when the JSON shape matches `s`, and `loadWeights?`, which returns the object of such arrays in
  a document. Both return `Option`: a mismatched shape or a rejected document is a `none` the
  caller has to handle rather than a panic.
- `Import/State.lean` reads a PyTorch `state_dict` into `TensorPack α shapes` word for word
  (`TorchLean.PyTorch.state`), from the `torchlean.state` document that `python/torchlean/state.py`
  writes. `TorchLean.PyTorch.stateDict` writes that document, and `state_stateDict` proves the round
  trip.

`TorchLean.IR.load` (`TorchLean/NN/IR/Loaded.lean`) reads a capture document into a program and its
payload, or an error message. It refuses a document without an `"artifact"`, decodes the artifact
(`TorchLean.IR.Artifact.ofJson`) and loads it under the reference runtime with no registered custom
operations or calls (`TorchLean.IR.Artifact.load`). `TorchLean.IR.load_ok` proves that the program
it returns passes `Program.check .reference` and its payload passes `Payload.check`.

## Model-family examples

The MLP, CNN and Transformer examples, their runnable round trips and small reference artifacts are
in [`TorchLeanExamples/Interop/PyTorch`](../../../TorchLeanExamples/Interop/PyTorch/README.md).
The [graph-import guide](https://lean-dojo.github.io/TorchLean/examples/verify/) covers capture,
validation and reference comparisons through the shared IR.

## Supported workflows

- Write a module's `state_dict` with `python/torchlean/state.py` and read it into `TensorPack` with
  `TorchLean.PyTorch.state`, every word kept.
- Parse nested-list JSON tensors in Lean with exact shape checks (`parseTensor`).
- Capture supported PyTorch `nn.Module` graphs as `torchlean.ir` artifacts with
  `python3 -m torchlean.export`, with the exact words of their constants, and load them in Lean
  with `TorchLean.IR.load`.
- Write a conservative ONNX static graph fragment as a typed artifact with
  `python3 -m torchlean.onnx`, with its initializers as exact constants, and check and run it in
  Lean in the model's element type.
- Reuse the Lean IR semantics after import: a loaded program evaluates with the shared IR
  evaluator (`TorchLean.IR.Program.eval`).
- Load PINN checkpoints with `TorchLean.Examples.PINN.Checkpoint`
  (`TorchLeanExamples/Verification/PINN/PyTorch/Load.lean`), which parses each tensor with
  `parseTensor`.
- Emit readable PyTorch code from an operator graph with its payload (`TorchLean.IR.Imported`) with
  `PyTorch.emit`.

## PyTorch references

- PyTorch "Saving and Loading Models" / `state_dict`:
  https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
