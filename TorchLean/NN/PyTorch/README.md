# PyTorch interop

`TorchLean/NN/PyTorch` moves weights and supported computation graphs between PyTorch and Lean.
The Python package `python/torchlean` captures the graph, and Lean checks its shapes, element types
and operator records. Existing PyTorch checkpoints and `state_dict`s can supply parameters for
Lean verification and proofs.

Decorated Python models keep PyTorch's execution and autograd. The imported artifact has Lean
semantics, but its checks do not establish that the capture matches the Python source or that
PyTorch's kernels implement those semantics.

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
`@torchlean.compile` lowers the graph that `torch.compile` hands its backend. A `torch.export` graph
that sets grad mode (`no_grad` or `enable_grad`) or autocast is refused, including inside nested
regions. `torchlean.export` does not fall back to a symbolic trace there, since it would drop the
mode. A mode that PyTorch's exporter erases before capture cannot be recovered from its graph;
the checks therefore make no claim that the source's gradient choices were preserved. Capture also
checks node provenance for grad-mode regions in core ATen decompositions.

The modules of `python/torchlean` split the work:

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
- `_hop` lowers the control-flow operators that torch.export records, `torch.cond`, `while_loop`
  and `map`, to the IR's flows, unrolls a `map` whose body draws (`_hop.draws`) and inlines
  `invoke_subgraph`
  ([IR guide](../IR/README.md#frontend-lowering)). `_flow.lower` then writes the capture nodes as a
  flow artifact (`TorchLean.IR.FlowArtifact`) whose blocks are typed artifacts, as for JAX.
- `export` writes the document
  `{"artifact", "frontend": "pytorch", "capture_constraints", "declared_types"}`.
  `declared_types` gives each lowered tensor value's step output, in its block for a flow artifact,
  and the element type and shape that FX's metadata records for it, and `TorchLean.IR.loadFlow`
  refuses a program that gives one of them another type. `torchlean.verify` adds the
  `"arguments"` for an execution and the `"reference_outputs"` for a comparison.

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
`Tensor.batchNorm`: the centered input divided by the IEEE root of the running variance plus
epsilon, NaN in a channel where that sum is negative, as in PyTorch. It has its own code for graph
inputs, constants and random draws, for the parameters of `linear`, convolution and the
normalizations, and for layouts PyTorch has no one call for (pooling, `gather`, `scatter`,
`slice`, `pad`, `broadcast`, `contract`, `indices`). The reductions have their own code too:
PyTorch names each `reduce` operation separately and reads an empty axis list as every axis, and a
product over axes is a loop over each fiber in row-major order, which `torch.prod` would regroup.
So do the IR's own formulas (sigmoid, SiLU, softplus, GELU, `safe_log`, the clamped `sqrt`, the
masked softmax, `top_k_mask`). The tuple results of `cummax`, `cummin`, `sort` (a stable sort) and
`topk` (the stable sort's first `k`) have their own code as
well, which keeps their index outputs. The GELU formula calls `torch.nn.functional.gelu` with its
`approximate`. SiLU and GELU return their limits, `+∞` at `+∞` and `-0` at `-∞`, and evaluate their
formula at zero there, so that no autograd slope is a NaN. The emitted slope at `-∞` is then `+0`,
where the operator's derivative gives `-0`. The extrema and `relu` give a zero result
IEEE 754-2019's sign, a mean over an empty fiber and the mean squared error of empty inputs are
`+0`, and a random node reads the draws `TorchLean.Random.uniform` makes for its key from a buffer
that `emit` writes. A truncating division floors the exact quotient `|a| / |b|` and takes the sign
of `a / b`, as the operator does, since `rounding_mode='trunc'` truncates the rounded quotient: `10`
at `(1, 0.1)`, where the operator gives `9`. A reduction, mean, matrix product, scaled matrix
product, contraction, pool or cumulative sum, product, maximum or minimum in an order other than the
left fold (`Tensor.ReduceOrder`) is refused, as PyTorch fixes no order.

## Wire records

A `torchlean.ir` artifact records each registered operator by its codec record
(`TorchLean/Tensor/Op/Wire.lean`): `Op.encode` writes the record version
(`Op.Wire.formatVersion`, 5), the operator's tag and its implicit shapes and attributes, one field
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
the shape it indexes and writes axes, shapes and convolution geometry as fields of the record.
PyTorch, JAX and ONNX fix no order for a sum, a mean, a contraction, a matrix product, a pool or a
cumulative sum, product, maximum or minimum, so the captures write each of them in the left fold,
the field `{"order": "leftFold"}` (`Tensor.ReduceOrder.leftFold`, the operator table's default
order). Each parameter tensor, such as the weight and bias of a `linear` node, becomes a constant
step holding its words. The format name is `TorchLean.IR.formatName`, which the capture reads from
the op table.

## Captured operators

The capture matches complete ATen overload names and listed FX callables or methods. For
example, `aten.log.default` lowers to `log`, while `log1p`, `log2` and `log_softmax` have lowerings
of their own, and `aten.slice.Tensor` with a positive step lowers to one `slice`. Each operator
becomes one node with its axes as attributes: `amax`, `sum.dim_IntList` and `prod` one `reduce`,
`mean.dim` one `mean`, `_log_softmax` one `log_softmax`, `constant_pad_nd` one `pad`, `expand` one
`broadcast`, and `flip`, `cat`, `permute` and `slice.Tensor` one node each. A binary16 or bfloat16
`sum` or `mean` is that node in binary32 between two casts, since ATen's CPU kernels accumulate
these formats in binary32 and round once (measured), while `prod` rounds each step in the format,
as one `reduce` does. The value components of
`native_layer_norm`, the inference batch norm and max pooling with indices, and every component of
a split, become the node itself, and component *i* of `unbind` is the `select` of index *i* along
its axis. Another component is refused, except the index outputs and the functional batch norm's
statistics described below. Creation
operators (`arange`, `linspace` and `logspace` among them) are constants PyTorch computes from the
same arguments, and a tensor of one value
(`zeros`, `ones`, `full`, `scalar_tensor`, their `_like` and `new_` forms and `fill`) is that
value's one constant word broadcast to the shape, however large the tensor. A binary16 or
bfloat16 fill value of `full`, `full_like`, `new_full`, `fill` and `scalar_tensor` is rounded once
to that format, where PyTorch rounds through binary32. FX evaluates an input-independent creation
while tracing, so under FX the constant is the tensor PyTorch made. An uninitialized allocation
imports as the destination of a full overwrite such as `zero_` or `fill_`. The `torch.export`
path refuses a graph that reads its contents.

The default graph keeps some operators that core ATen decomposes. `Tensor.T`, `mT`, `mH`, `H`,
`adjoint`, `swapdims`, `swapaxes`, `movedim` and `moveaxis` are one `permute` each (`T` below rank 2
is its operand), `rsub` is `sub` with its operands swapped, and `silu` is `Op.silu`. PyTorch's CPU
formula, `x / (1 + exp(-x))` in the opmath type, is a NaN at `-∞` and `-0` once `exp(-x)` overflows,
where `Op.silu` returns the limit and keeps the tail. `stack` is one `cat` of its operands, each
reshaped to an axis of extent one there. `repeat_interleave` with an int count gives each entry an
axis of extent one, broadcast to the count and merged. `roll` concatenates two slices along each
rolled axis, and `tril` and `triu` select between the operand and zero by a constant mask.
`Tensor.unfold(dimension, size, step)` stacks `size` slices of stride `step` on a new last axis,
slice *j* starting at entry *j*, so that window *i* holds the entries from `i * step`.
`Tensor.cpu()` and `to` with a device, a memory format or another tensor convert to the result's
dtype. The report leaves devices unchecked (`frontend_device_layout_constraints`), and a layout
other than `torch.strided` is refused. A `linear` weight or bias that is not a parameter, buffer or
constant is an operand of the node after its input. An operator of a tensor result with no
lowering of its own that eager PyTorch runs as a composite of other ATen operators, having a
`CompositeImplicitAutograd` kernel and no CPU kernel (`einsum`, `outer`, `type_as`, `expand_as`,
`conv_transpose2d`), is replaced by that composite, traced at fake CPU tensors of its operands'
shapes and dtypes, whose operators are then captured or refused as anywhere else (`item`, whose
composite gives a number, is refused). The default graph keeps such operators, which the core
graph has decomposed already, and under FX `torch.einsum` is the one. An `einsum`
becomes diagonals, sums, permutations, reshapes and `bmm`s in the order of the contraction path
the graph records, which is opt_einsum's from three operands on when it is installed (the
`torch.einsum` docstring).
A transposed convolution (`aten.convolution` with `transposed`, which `conv_transpose1d`, `2d`
and `3d` reach) is the convolution at stride 1 of its input with `stride - 1` zeros between
entries and `dilation * (kernel - 1) - padding` zeros at each end, `output_padding` more at the
far end and a negative amount cropping, by the kernel with each group's two channel axes swapped
and its spatial axes reversed. Those zeros multiply the kernel, where eager PyTorch adds only the
products of input entries, so an infinite or NaN kernel entry gives NaN where eager PyTorch's
output can be finite. Its sums round in the IR convolution's order, which is not eager PyTorch's,
so finite results can differ in their last places; the tests compare small integers, whose sums
every order gives exactly, as for the other convolutions.

The mixture-of-experts routing idiom `torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices, 1)`
lowers to one `top_k_mask` node, and the `topk` call and the zeros it scatters into must have no
other users. For partial selection (`0 < k < width`), the capture document's `capture_constraints`
records the assumption that each score row is finite and that its `k`-th largest score is strictly
above its `(k+1)`-th. Without that gap, PyTorch may select different tied experts than the IR's
lower-index rule. Selecting zero or all entries needs no gap. The check report keeps these
constraints with `capture_constraints_status="not_checked"`: a successful structure, type or
sample check does not prove them for other inputs.

The capture refuses mutating overloads, `view(dtype)` reinterpretation and unsupported tuple
producers, and writes no artifact for them. The mutating overloads it accepts are `index_put_`,
`index_add_`, `index_copy_`, `copy_`, `scatter_`, `scatter_add_`, `zero_` and `fill_` of a fresh
destination, `detach_`, which is `Op.detach` (below), and training batch normalization's
`num_batches_tracked` update (below). A
`dtype` argument of `sum`, `prod`, `mean`, `cumsum`, `cumprod`, `softmax` or `log_softmax` casts the
input to `dtype` before the operation, as their docstrings state. Eager `mean` to binary16 or
bfloat16 from another type sums the uncast input in binary32 and rounds the quotient once instead
(measured), and the comparison with it can fail there.

The core ATen pointwise operators `neg`, `rsqrt`, `erf` and `gelu`, every core overload of `div` and
`pow`, `expm1`, `log1p`, `exp2`, `log2`, `log10`, `tan`, the hyperbolic functions and their
inverses, `asin`, `acos`, `atan`, `atan2`, `hypot` and `remainder` become IR pointwise nodes. So
do the exact `floor`, `ceil`, `round` (zero decimals), `trunc`, `frac`, `sign`, `copysign`, `fmod`,
`fmax` and `fmin` of floating tensors. Each means its mathematical definition or IEEE 754-2019's
operation (`rsqrt` is `1 / sqrt x` rounded twice, so binary64's `rsqrt 3` is one ulp above IEEE's
`rSqrt 3`). Where PyTorch's CPU kernels compute something else, the capture keeps these
definitions, and the IR guide lists the differences. Among them, `remainder` gives a zero result
the divisor's sign, as `torch.remainder` documents (Python's `%`) and eager PyTorch does not.
`fmax` and `fmin` return the values of IEEE 754-2019 `maximumNumber` and `minimumNumber`, with `+0`
above `-0`, a case PyTorch's documentation leaves open. The tanh `gelu` computes `√(2 / π)` in the
format, TorchLean's choice of schedule, one ulp below PyTorch's once-rounded constant in binary32.

The `torch.special` functions that `TorchLean.SpecialFn` names, with `polygamma`, `lgamma`,
`digamma`, `i0`, `igamma` and `igammac`, become one `Op.special` node each, of float32 and float64
operands only. Their meaning is the real function,
which no binary format executes, so a run at one refuses the step. `torch.frexp` becomes `Op.frexp`,
C's `frexp`, and `torch.linalg.householder_product`, `torch.orgqr` and `torch.ormqr` become the
products of their Householder reflectors. The IR guide states each one's meaning and refusals.

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

A value of the four float8 formats (`float8_e4m3fn`, `float8_e4m3fnuz`, `float8_e5m2`,
`float8_e5m2fnuz`) is an input, a constant, a conversion's result or a movement of a float8 value
(`view`, `reshape`, `flatten`, `permute`, `transpose`, `t`, `unsqueeze`, `squeeze`, `expand`,
`slice`, `contiguous` and `clone`), and only these and the outputs read it. Any other operator of a
float8 value is refused, and so is `float8_e8m0fnu`. Every float8 word converts exactly to binary16,
bfloat16, binary32 and binary64. A conversion to a float8 format rounds once, also from binary64,
which PyTorch rounds through binary32 first. Overflow in a conversion to `float8_e4m3fn` depends
on PyTorch's device conversion: it can saturate to `±448` or produce the format's NaN. Capture
checks that policy for the recorded source and target devices, including the `non_blocking`
setting. For saturation it clamps the operand to `[-448, 448]` before rounding, preserving NaNs;
otherwise it uses the ordinary cast. If the policy cannot be established, capture refuses the
conversion. The JAX importer's conversion gives the NaN, as XLA's does.

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

`leaky_relu` is `Op.leakyReLU` at its slope, whose pullback sends the slope at `x = 0` as
`leaky_relu_backward` does. `elu`, `selu` and `celu` are `Op.elu` at the kernel's `alpha`, `scale`
and `input_scale`: `selu` at ATen's `SELU_ALPHA` and `SELU_SCALE`, `celu(x, alpha)` at `alpha`, `1`
and `1 / alpha`. Comparisons, `isnan`, `isinf`, `isfinite` and the logical and bitwise operators of
floating or Boolean operands are Boolean nodes, and `where` and `masked_fill` select with a Boolean
condition. `clamp`, `clip` and `hardtanh` are the IR `clamp`. `softplus(x, beta, threshold)` is the
documented `log(1 + exp(beta * x)) / beta` as `Op.softplus` of `x * beta` divided by `beta`, and `x`
where `x * beta > threshold`, where the documentation says the implementation reverts to the linear
function. The CPU kernel computes `log1p(exp(x * beta))` instead, which is `+∞` once `exp`
overflows below a raised threshold: binary32 `softplus(95, threshold=100)` is `+∞` there, with a NaN
slope, and `95` here.

`diagonal` is a permutation, a reshape and a strided slice. `index_select`, `gather`, `scatter`
and `scatter_add` with indices from buffers or constants become one `gather` or `scatter` node
holding every coordinate. So does `as_strided`: entry *i* of `as_strided(x, size, stride, offset)`
is the storage word at `offset + Σ i_k stride_k`, and when every word read holds an entry of `x` in
the strides and storage offset the capture records for it, the node gathers those entries. A word
outside `x`, such as a gap of `x[:, ::2]`, is refused. Memory follows the layout of the inputs,
which `torch.export` records but does not check, so each `as_strided` adds a `strided_read`
constraint: the value is PyTorch's when every input has the strides and storage offset of the
captured one. `argmax`, `argmin` and the index outputs of `max.dim`, `min.dim`,
`cummax`, `cummin` and the stable `sort` of floating operands become steps of the registered
operators `Op.argmax`, `Op.argmin`, `Op.cummax`, `Op.cummin` and `Op.sort`, with int64 results.
The stable `sort` of integer and Boolean operands, values and indices, is `Op.sort` at their
element: integer types by value, signed or unsigned, and `false` before `true`.
For `argmax` and `argmin`, `dim=None` reduces the row-major flattening and a scalar is one entry.
An int64 or int32 index computed from the inputs makes the same kind of step. `embedding` is a step
of `Op.embedding`. `index_select`, `gather`, `take_along_dim` and `index.Tensor` are steps of
`Op.gather`, with a permutation where PyTorch places the selected axes, and `scatter`,
`scatter_add` and `index_put` are steps of `Op.scatter`. A negative index of `index.Tensor` or
`index_put` counts from the end. TorchLean expands `take_along_dim` using the installed ATen
composite. PyTorch versions that wrap indices with a dimension produce a remainder before the
gather. A gather rejects indices outside its axis.
With a runtime index, these gathers and the writes that replace entries (`index_put` without
`accumulate`, `index_copy` and `scatter` of a tensor) also move integer and Boolean values, at
their element. The writes that sum and `scatter` of a Python value are floating only.

A repeated target of a runtime scatter or `index_put` folds in row-major order of the index tensor,
which PyTorch leaves unspecified. PyTorch's CPU `index_put` with `accumulate=True` departs from
that order on large inputs, and `index_put` without `accumulate` at a repeated target is documented
as undefined. `scatter_add` and `scatter_reduce` sums and products of binary16 and bfloat16
accumulate in binary32 and round once, as the CPU kernel does (measured), and `index_put` with
`accumulate` rounds each step in the dtype, as its kernel does. A Boolean mask index is refused.
`index_add` is `index_put` with `accumulate` at one axis, folding in the order of its index as
documented and rounding each step, as PyTorch's decomposition and its CPU kernel of a vector do.
Its CPU kernel of a matrix or higher sums binary16 and bfloat16 in binary32 (measured), and the
comparison with it fails there. ATen fuses `alpha` into the sum, and `alpha` other than 1 is
refused, as for `add`. `index_copy` is `index_put` without `accumulate` at one axis, as PyTorch's
decomposition is. Its docstring leaves a repeated target nondeterministic, and the CPU kernel and
the scatter both keep the last copy (measured). `index_add` and `index_copy` reject a negative
index, which `index_put` wraps: a negative runtime index fails at execution, and a negative
constant index is refused.
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

The default graph's `index_put_`, `copy_`, `scatter_`, `scatter_add_`, `zero_` and `fill_` are
their functional operators when the destination is fresh, their other operands are not the
destination, no later node reads it and its earlier readers allocate their results. A fresh
destination is an audited creation operator's result, an accepted one of these writes, or a
functional core or pointwise operator's result without an alias annotation and not tagged
`maybe_aliasing_or_mutating`. A write into an input, a view or a possibly aliased tensor is
refused, and so is an in-place operator whose result is unused, a dead update of fresh scratch
storage included. The default graph also accepts `add_` of
the integer 1 to an int64 scalar buffer whose only reader is that operation, its result possibly
used. This structural rule includes the batch norm's `num_batches_tracked` increment and computes
the updated value out of place at capture.

`slice_scatter`, `select_scatter` and `col2im` (`fold`) are constant scatters. `any` and `all`
compare a binary32 sum of zeros and ones with zero, empty fibers giving false and true. Supported
deterministic integer and Boolean operations with constant operands, such as an `arange` or a
causal mask, are folded at capture only when their schemas declare no mutation. `detach_` of such a
value folds to its operand's value, as does that of a tensor literal (`lift_fresh_copy` then
`detach_` in the default graph). A `detach_` of another floating value is `Op.detach`, the value
with a zero tangent: torch.export writes the program's later reads of the tensor as reads of its
result, so `y = 2 * x; y.detach_(); y * x` has the derivative `2 * x`, as in PyTorch. The
out-of-place `detach` of a floating value is `Op.detach` too. PyTorch 2.13's core decomposition
writes `detach` as `alias` with `detach` as the node's `original_aten`, which is `Op.detach`, and
drops `detach_` with no trace, so a decomposed graph of the program above has the derivative of
`(2 * x) * x`. The capture itself never decomposes, and the document of a decomposed program
carries a `decomposed_program` constraint, which says that its derivatives assume no `detach_`. A
symbolic trace keeps the later reads on the operand, and a `detach_` whose operand it reads later
is refused. Comparisons of
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
`rfft` and `irfft` are `Op.rfft` and `Op.irfft` on the paired-real form, the imaginary parts that
`irfft` ignores (frequency 0 and, for an even `n`, `n / 2`) set to `+0` first. Between them a
complex64 or complex128 value is held in that form, its parts on a last axis of 2 in binary32 or
binary64: `view_as_complex` and `view_as_real` are the paired tensor, `real` and `imag` read one
part, and slices, selects, `unsqueeze`, `view` and `reshape` move the pairs. A quotient of
`a + bi` by a nonzero finite number `d` is `c10::complex`'s division by `d + 0i`,
`((a + b*r) * s, (b - a*r) * s)` with `r = 0 / d` and `s = 1 / (d + 0 * r)` in the parts' format
(`torch/headeronly/util/complex.h` of PyTorch 2.13), and gives eager's words. The product of
`a + bi` and `c + di` is `(ac - bd) + (ad + bc)i`, each operation rounded, and agrees with eager
within a tolerance only: on an AVX512 CPU the kernel fuses multiply-adds in the entries after the
last whole vector. Any other reader of a complex value, and a complex input, constant or output,
is refused. Group norm, the training batch norm (its running statistics below), `cdist`, `pdist`,
linear and bilinear upsampling and `grid_sampler_2d` (bilinear or nearest, zeros or border
padding) are compositions. Grid sampling reads a NaN coordinate as `-1`, as `grid_sample`'s
docstring says, and an infinite one as the limit of the finite ones beyond it: zero outside the
image with zeros padding and the edge with border padding, while a singleton axis that
`align_corners` or border padding keeps at its only pixel reads that pixel. With zeros padding the
CPU kernel gives NaN in bilinear mode at both and zero in nearest mode at NaN instead. The tests
compare these compositions with the CPU kernels on their fixtures within stated tolerances (no
uniform error bound), and the nonfinite coordinates with the CPU kernel at `-1` and `±1e6`.
`masked_scatter`, `nonzero_static` and `_embedding_bag` with constant offsets are captured too,
and `nonzero` (a data-dependent shape) is refused. `linalg_vector_norm` is its docstring's formula
for each `ord` (the largest or smallest magnitude for `±inf`, the count of nonzero entries for 0,
the root of the sum of squares for 2 and else the `1/ord` power of the sum of `|x|^ord`) of its
input cast to its `dtype`, and a binary16 or bfloat16 norm is computed in binary32 and rounded
once, as ATen's CPU kernel does (measured). Its tests compare it within a tolerance too.

`rand`, `randn`, `randperm` and training-mode dropout use TorchLean's generator under keys recorded
in the artifact, not PyTorch's stream. `aten.randn.default` of the four floating dtypes is
`Op.randn`, a deterministic Box-Muller approximation in the dtype's arithmetic on two 32-bit draws
per entry, each rounded to the dtype, `u1` clamped below by `1e-6` rounded to it, with no exact
normal law. The generator overloads are refused and `aten.randn_like` is not captured directly.

LayerNorm and BatchNorm nodes carry their weights, biases, eps and running statistics as
interchange words, and a mean over an empty fiber is refused. A weight, bias or running statistic
computed in the graph or passed as an input is a parent of the node instead. Batch norm, inference
or training, and group norm take the root of `var + eps` as it stands, so a negative or NaN sum
gives NaN in its channel or group, as in PyTorch. `aten._native_batch_norm_legit` is imported in
both modes: outside training its statistics
may be buffers or graph values, and in training they must be two buffers that only it reads. Its
functional form in the core dialect is imported outside training with buffer statistics, which it
returns unchanged as the updated buffers. With statistics from the graph inputs torch.export writes
them back as a user-input mutation, which is refused, and an FX capture of the raw ATen call is
refused. A binary16 or bfloat16 input of
`layer_norm` takes a binary32 weight and bias too, applied in binary32 before the one rounding, as
PyTorch's CPU kernel does (measured). A training batch norm with running statistics computes the
batch moments and returns the supported buffer updates as outputs. The
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
it returns passes `Program.check .reference` and its payload passes `Payload.check`. It refuses a
flow artifact, which `TorchLean.IR.loadFlow` loads block by block, and `TorchLean.IR.loadFlow_ok`
proves the same of every block. A graph's artifact is a flow when a `torch.cond` on a predicate not
known at capture, a `while_loop` or a `map` whose body does not draw gives a value that the graph
returns, so a graph without these operators has none.

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
  `python3 -m torchlean.export`, with the exact words of their constants, and load them in Lean with
  `TorchLean.IR.loadFlow` (`TorchLean.IR.load` for an artifact that is not a flow, such as that of a
  graph without `torch.cond`, `while_loop` or `map`).
- Write a conservative ONNX static graph fragment as a typed artifact with
  `python3 -m torchlean.onnx`, with its initializers as exact constants, and check and run it in
  Lean in the model's element type.
- Reuse the Lean IR semantics after import: a loaded program evaluates with the shared IR evaluator
  (`TorchLean.IR.Program.eval`), and a flow by `TorchLean.IR.Flow.eval` in
  `Flow.programs .reference`.
- Load PINN checkpoints with `TorchLean.Examples.PINN.Checkpoint`
  (`TorchLeanExamples/Verification/PINN/PyTorch/Load.lean`), which parses each tensor with
  `parseTensor`.
- Emit a PyTorch module from an operator graph with its payload (`TorchLean.IR.Imported`) with
  `PyTorch.emit`.

## PyTorch references

- PyTorch "Saving and Loading Models" / `state_dict`:
  https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
