# `TorchLean/Tensor`

This folder is TorchLean's tensor layer: one shape-indexed `Tensor` type that proofs and CPU
execution share, the reference definitions of the operators and layers written on it, and the
theorems about them. Tensor-only code imports the modules it uses:

```lean
import TorchLean.Tensor.LinearAlgebra
import TorchLean.Tensor.Literal
```

`LinearAlgebra` brings construction, conversion, arithmetic, reductions and factorizations, and
`Literal` the bracket literals. Spec code imports the modules it builds on, such as
`TorchLean.Tensor.Layers.Linear`. Model code imports the modules it needs from
[`TorchLean/NN`](../NN/README.md), which also holds the autograd, model, dynamics, generative and
RL specs. The layer depends on FloatLib and Mathlib and imports no other TorchLean layer.

## Directories

- `AdjacentSwap/`: the adjoint and derivative of an axis permutation.
- `Algebra/`: algebraic identities for real tensors.
- `ConfiguredBinary/`: correctly rounded elementary and inverse trigonometric functions of
  FloatLib's configured binary scalars.
- `Context/`: the real and the scoped rational `Context` instances.
- `Einops/`: the shape-pattern syntax, checks, elaborators, semantics, lowering and laws.
- `Elementwise/`: the meanings of the elementwise operators over `ℝ`.
- `Factorization/`: the Cholesky and QR predicates, exact reconstruction, the orthonormality of `Q`,
  and the exact triangular and ridge solves over `ℝ`.
- `Layers/`: forward and backward specs of common layers.
- `LinearAlgebra/`: the identity matrix, reference numerics, and norms and linear algebra over `ℝ`.
- `Movement/`: the adjoint of broadcasting, and coordinate lemmas for `cat`, `permute` and
  `gather`.
- `Native/`: the certified loops, slices, transposes, pointwise kernels, output partitions,
  parallel output assembly and einsum output tiles that compiled tensor code runs.
- `Numeric/`: inverse trigonometric functions, finite enclosures, nested quotient arithmetic with
  the `NatCast` and `RatCast` instances of `Float` and `Float32`, the bit-level widening of
  configured binary values to `Float`, and the `MathFunctions` rewrites and inequalities at `ℝ`.
- `Op/`: the registered operators with their rules, wire codec and certificates.
- `Random/`: keyed Fisher-Yates permutations.
- `Reduce/`: the largest absolute difference of two tensors, sums over one axis and the pullbacks
  of reductions, and sums and folds of real tensors.
- `Rep/`: `full`, `map`, `zipWith`, `zipWith3`, `dim` and `unstack` with their compiled
  implementations, reindexing, segments, traversal and finite fibers of `Rep`, the native tensor
  representation, and its correspondence with Mathlib matrices.
- `Shape/`: row-major coordinates (`Shape.Coord`, `Coord.linearize`, `Coord.unlinearize`), their
  index lists (`Coord.toList`), and the checked runtime coordinates `Coord.ofList?` and
  `Coord.ofArray?`.
- `ShapeChange/`: tensor coordinates, flattening and reshapes.
- `Statistics/`: the population variance `TorchLean.var` of a finite family of reals.
- [`Tactic/`](Tactic/README.md): the `verify`, `einops`, `except_cases` and `tensor_shape` tactics.
- `Window/`: pooling past an unpooled leading shape, which pools each slice.

## One Tensor Type

A value of `Tensor α shape` stores its entries in one row-major buffer, with a proof that the
buffer's length matches the static shape. Code generation erases the proof.

`Storage α` selects the physical buffer, and with it whether the scalars are stored unboxed:

- `Float` uses `FloatArray`.
- `UInt8` uses `ByteArray`.
- Other element types, `Float32` and proof-oriented types included, use `Array α`, whose entries
  may be boxed.

Proofs see every buffer through the ordinary array `Storage.toArray`. The class carries laws that
relate each buffer operation to that array (`Storage.toArray_size`, `Storage.toArray_injective`),
and lemmas such as `Storage.get_eq_toArray_getElem` follow from them. FloatLib's `roundoff` reads a
tensor entry as a stored entry only through these laws. The laws do not show that a generated
reference agrees with a separately written specification. FloatLib's `Automation/Roundoff.lean`
states the assumptions that remain about storage types and index data.

In statements and proofs a tensor also acts as a total function on its coordinates, a view that
allocates no second, recursive tensor. A tensor is more than a blob of numbers: its shape, its
scalar interpretation and the operations it allows are part of the object being specified.

## Construction And Conversion

Ordinary nested brackets construct a tensor whenever a tensor type is expected:

```lean
def threshold : Tensor Float [] := 0.5

def matrix : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]
```

`[]` is the public rank-zero shape, and an ordinary numeric literal constructs its single value, so
there is no separate scalar-tensor syntax to learn. Positive-rank tensors use brackets, and the
elaborator rejects ragged literals and shape mismatches.

## Inspection And Display

`#eval` prints a tensor through its shape-aware `Repr` instance:

```lean
def left : Tensor Nat [2, 2] := [[1, 2], [3, 4]]
def right : Tensor Nat [2, 2] := [[10, 20], [30, 40]]

#eval left
-- [[1, 2], [3, 4]]

#eval left + right
-- [[11, 22], [33, 44]]
```

Inspecting a tensor needs no conversion: use `#eval tensor` in a Lean file or
`IO.println (reprStr tensor)` inside an `IO` program. `Tensor.to` is for serialization,
interoperability and APIs that require another container.

Use the general conversion boundary for existing in-memory values:

```lean
def vector := Tensor.from (#[1.0, 2.0, 3.0, 4.0] : Array Float)

def matrix' : Tensor Float [2, 2] :=
  vector.reshape [2, 2]

def narrow : Tensor Float32 [2, 2] :=
  matrix'.cast Float32

def values : Array Float :=
  Tensor.to matrix' (Array Float)

def asList : List Float :=
  Tensor.to (matrix' + matrix') (List Float)

def asVector : Vector Float (Shape.size [2, 2]) :=
  Tensor.to matrix' (Vector Float (Shape.size [2, 2]))
```

- `Tensor.from source` is total and keeps the source element type.
- `tensor.reshape target` changes only the static shape and does not copy. For concrete shapes of
  equal size Lean proves the size equality by `decide`, and symbolic shapes may need a size proof.
- `tensor.cast TargetElement` changes the element type.
- `Tensor.to tensor TargetType` converts to a requested in-memory representation.

External files and runtime metadata are checked boundaries: their loaders validate the claimed
dimensions and payload lengths before they construct an ordinary `Tensor`.

## Mixed Element Types

Each tensor is homogeneous, but equally shaped tensors with different element types can interact:

```lean
def bytes : Tensor UInt8 [3] := [1, 2, 3]
def offsets : Tensor Float [3] := [0.5, 1.5, 2.5]
def shifted : Tensor Float [3] := bytes + offsets
```

`+`, `-`, `*`, `/`, and the corresponding named operations select a registered
common type and fuse both input conversions with output construction. Built-in promotion follows
`UInt8 < Nat < Int < Rat < Float32 < Float`. Custom scalar families extend the same API with
`ElementCast` and `ElementPromotion` instances.

## Indexing And Immutable Updates

Static indices and coordinates carry their bounds in their types:

```lean
def matrix : Tensor Nat [2, 2] := [[1, 2], [3, 4]]

def row : Fin 2 := ⟨1, by decide⟩
def column : Fin 2 := ⟨0, by decide⟩

#eval matrix[row][column]
-- 3
```

With a complete typed coordinate, use `tensor.at coordinate`, `tensor.set coordinate value` and
`tensor.modify coordinate f`. Updates are immutable copy-on-write operations. `Float` and `UInt8`
tensors stay in their packed buffers, and other element types use the ordinary array fallback. The
simplifier knows that reading the updated coordinate returns the written value (`at_set_self`,
`at_modify_self`).

For a coordinate held in a runtime array, the checked `at?`, `set?` and `modify?` return `none` on
a wrong rank or an out-of-bounds index.

## Shape Pattern Operations

`TorchLean.Tensor.Einops.Elab.Transform`, `TorchLean.Tensor.Einops.Elab.Einsum` and
`TorchLean.Tensor.Einops.Elab.Pack` provide the checked shape-pattern operations on the ordinary
`Tensor` type. Their keywords are scoped syntax, so a file turns them on with
`open TorchLean.Tensor`:

```lean
open TorchLean.Tensor

def matrix : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 4] :=
  [[1.0, 0.0, 0.0, 1.0],
   [0.0, 1.0, 0.0, 1.0],
   [0.0, 0.0, 1.0, 1.0]]

def transposed : Tensor Float [3, 2] :=
  rearrange matrix "row column -> column row"

def tiled : Tensor Float [3, 2, 3] :=
  expand matrix "row column -> batch row column" with batch := 3

def rowSums : Tensor Float [2] :=
  reduce matrix "row column -> row" by sum

def product : Tensor Float [2, 4] :=
  einsum matrix, right
    "row contracted, contracted column -> row column"

def packed :=
  pack rowSums, matrix "*"

def restored :=
  unpack packed "*"

def restoredRowSums : Tensor Float [2] :=
  restored ⟨0, by decide⟩

def restoredMatrix : Tensor Float [2, 3] :=
  restored ⟨1, by decide⟩

def dimensions : List (String × Nat) :=
  parse_shape matrix "row column"
```

`rearrange`, `expand`, `reduce`, `einsum`, `pack`, `unpack` and `parse_shape` check their literal
patterns against the static shapes while Lean elaborates the definition. `expand` is einops'
`repeat` under another keyword, since Lean has its own `repeat`. The output names of a pattern pick
and order dimensions from the inputs, so the result type carries the output shape and a type
annotation that disagrees is rejected before the program runs. Multi-input `einsum` and `pack`
promote mixed element types the same way tensor arithmetic does. The metadata from `pack` goes
straight into `unpack`. Integer metadata is written by hand only for the optional `-1`, which infers
one dimension. `parse_shape` returns ordinary named dimensions, here `[("row", 2), ("column", 3)]`.

All seven are term elaborators, and all except `parse_shape` build tensor expressions. The `einops`
tactic proves tensor identities, and `einops?` explains a transformation in the InfoView. Their
proof and report engines live in `Tactic/Einops.lean` and `Tactic/Einops/`, and importing
`TorchLean.Tensor.Tactic.Einops.Report` brings in both tactics.

## Reductions And Linear Algebra

Scalar reductions and matrix factorizations stay on the ordinary tensor type:

```lean
def matrix : Tensor Float [3, 2] :=
  [[1, 2], [3, 4], [5, 7]]

def factors := Tensor.qr matrix
def reconstructed : Tensor Float [3, 2] :=
  einsum factors.q, factors.r
    "row contracted, contracted column -> row column"

def positiveDefinite : Tensor Float [2, 2] :=
  [[4, 2], [2, 3]]

def lower := Tensor.cholesky positiveDefinite
def loss := Tensor.meanSquaredError reconstructed matrix
```

`Tensor.qr` computes both factors in one call and returns them as the fields `.q` and `.r`. For an
`m × n` input their reduced shapes are `m × min m n` and `min m n × n`. Every shape runs the
reference schedule `TorchLean.Factorization.gramSchmidtFn`, whose reconstruction and orthonormality
theorems hold for real scalars with positive diagonal pivots. A column whose residual norm is not
positive takes no basis column, so an early dependent column does not stop a later independent
column from entering the basis.
`Tensor.cholesky` returns the lower-triangular factor candidate, and `Tensor.solveRidge` solves a
regularized symmetric system through the same Cholesky path. The Cholesky reconstruction theorem is
about the logical computation over real scalars, for symmetric inputs with positive pivots. Over
`ℝ`, `solveRidge_spec` in `TorchLean.Tensor.Factorization.Solve` shows that `K.solveRidge γ b`
solves `(K + γ I) x = b` for symmetric `K` when the Cholesky pivots of `K + γ I` are positive. None
of these theorems promises exact reconstruction in floating point. Compiled Cholesky and ridge
solving run array implementations substituted through the `@[csimp]` theorems
`choleskyColsFn_eq_choleskyCols`, `cholSolveFn_eq_cholSolve` and `solveRidgeFn_eq_solveRidge`,
which prove them equal to the logical definitions for every scalar type. The executable algorithms
support generic scalar backends. Large matrices call for blocked, vectorized LAPACK or backend
kernels.

`Factorization/Reconstruction.lean` proves the exact Cholesky and QR reconstructions and
`Factorization/Orthonormal.lean` the orthonormality of `Q`. The Muon orthogonalization
certificates in `TorchLean/NN/Optimizer/Muon/QR.lean` use the latter.

## Runtime Relationship

Specifications, proofs and native CPU operations share one certified buffer representation, and
there is no second host tensor class.

- `TensorPack α shapes` is `Pack (fun s => Tensor α s) shapes`, a statically heterogeneous
  collection used for model state.
- `SomeTensor α` is internal runtime shape erasure for tapes and graph interpreters.
- Runtime value references are handles owned by an eager session or a typed graph.
- CUDA uses device buffers owned by a session. Upload and download are `IO` boundaries that check
  the shape and buffer length, which keeps dtypes, shapes and bounds safe. Whether a kernel
  computes the right numbers is part of the backend's contract and does not follow from the tensor
  type.

## Scalar Backends

The same spec can be instantiated in several scalar worlds:

- `ℝ` for clean mathematical statements.
- `FloatLib.Floats.Formats.Flocq.NF` and its `FP32` specialization for noncomputable
  rounded-real proofs.
- `FloatLib.Floats.ExecFloat.Binary` for CPU software arithmetic at a chosen precision.
- Runtime scalar backends, where a bridge states what is assumed.

On CUDA, configured precision reaches only custom computations, for `ExecFloat.Binary` formats with
at most 30 exponent bits and 4096 fraction bits. `TorchLean/CUDA/Kernel` compiles them for the
format, and their recorded tape keeps the configured words: buffer addition, subtraction,
multiplication, division and negation run generated kernels in the stored format. The native
LibTorch operations take binary32 and binary64 only.

## Specifications And Their Boundary

The goal is no gap between the network we run and the network we reason about. We define the
reference behavior once, and the runtime, graph and verifier layers each say how they connect back
to it. The terms mean:

- spec: the pure reference definitions in this folder.
- runtime: tape and graph execution, lowering, CUDA paths and training loops, in `TorchLean/NN`.
- verification: bound propagation, certificate checking and artifact replay, in `TorchLean/Verify`.

A spec definition is the reference object for everything built on it. Runtime backends, external
kernels and serialized artifacts connect to it through a lowering, a checker or a stated trust
boundary. The usual chain is:

1. define the mathematical behavior here,
2. execute or lower a runtime object elsewhere,
3. state a bridge, checker, test, or theorem connecting the runtime/artifact back to the spec.

These files define meanings, and fast kernels live elsewhere. Runtime code may execute the same
operations over `Float`, FloatLib's `ExecFloat.Binary 8 23`, CUDA buffers or external providers. A
theorem about such a path says which spec object it approximates or preserves, and a typed graph
node or ATen implementation documents which spec operation it implements or approximates.

Boolean attention masks have the hard-mask meaning: a blocked position contributes exactly zero to
the softmax numerator. An additive attention bias is a different operation and must not stand in
for that mask. In the same way, a real-valued layer spec, the same spec run in binary32 through
`FloatLib.Floats.ExecFloat.Binary 8 23` and a CUDA `Float32` kernel are three related objects, and
none of them stands in for another.

A spec keeps its reference behavior small and readable. Runtime shortcuts, foreign-library
assumptions, tolerances and file-format details belong in the runtime, verification or
trust-boundary layer that owns them. The layer and model specs here and in `TorchLean/NN/Models`
build on the primitives of this folder and add no second tensor language. The executables in
`TorchLeanExamples` exercise the specs through the public trainer and CLI.

## Registered Operators

`Op/Table.lean` defines `TorchLean.Op`, one constructor per operator of the operator table, indexed
by its input and output ports (`Op/Port.lean`), and the attribute check `Op.check`.
`Op/Table/Rule.lean` gives each operator its rule (`Op.rule?`), and one module per operator family
restates the arms as equation lemmas: `Op/Table/Elementwise.lean`, `Op/Table/Compare.lean`,
`Op/Table/Word.lean`, `Op/Table/Reduce.lean`, `Op/Table/Movement.lean`, `Op/Table/Linear.lean`,
`Op/Table/Conv.lean`, `Op/Table/Normalization.lean`, `Op/Table/Attention.lean`,
`Op/Table/Loss.lean` and `Op/Table/Random.lean`. `Op/Rule.lean` holds the executable rules
(`Op.Rule`) and `Op/Capability.lean` the optional scalar capabilities that `Op.rule?` reads.
`Op/Codec.lean` is the wire codec with its round trip. Its records are version 3, and
`Op.decode` accepts version 3 only.
`Op/Run.lean` gives checked invocation of an operator on erased values, and `Op/Cert.lean` the
`op_cert` registration of operator certificates.

`Op/Pointwise.lean` defines `PointwiseOp`, the pointwise operators of core ATen and `jax.lax` that
have no node kind of their own, which graphs represent as one operation family. Over the reals,
`Op/Pointwise/Real.lean` proves with `Context.geluErf_real` that the exact GELU is
`x (1 + erf (x / √2)) / 2`, and `Activation.geluTanh_real` gives the tanh GELU's formula.
`divFloorSchedule_real` proves that PyTorch's flooring-division schedule computes `⌊a / b⌋`, and
`remainderSchedule_real` and `Context.remainder_real` that Python's remainder schedule computes
`a - b ⌊a / b⌋`.

## Layer Definitions

`Layers/` holds TorchLean's layer definitions: common neural-network building blocks written as
pure functions on spec tensors and parameter records. Most models in `TorchLean/NN/Models`
compose them. Layers used in reverse-mode training also carry derivative or VJP rules, so their
backward meaning is written down too.

- `Activation.lean`: scalar activation formulas under `TorchLean.Activation`, tensor activation
  wrappers, and the vector and last-axis kernels of softmax and log-softmax with their VJPs and JVP.
  Stable softmax detects a stalled sum and replays it exactly, and log-softmax shifts by its first
  strict maximum.
- `Softmax.lean`: `Tensor.softmax` with an optional Boolean mask and a `Softmax.Schedule`
  (two-pass, or FlashAttention's online pass over blocks of keys), `Tensor.logSoftmax`, their VJPs
  and the log-softmax JVP along any axis `dim`, and `Tensor.logsumexp` over any axes.
- `Linear.lean`: fully connected layer spec ($y=Wx+b$) and gradients.
- `Attention.lean`: `Tensor.attention`, scaled dot-product attention over any leading batch axes
  with a mask, a multiplier on the weights and the softmax schedule (two-pass or FlashAttention's
  online schedule), with its VJP and JVP. It also defines the single-sequence
  `Attention.scaledDotProduct` and multihead attention with hard-mask semantics (a blocked entry has
  a zero numerator, and a fully blocked row has zero weights, so its output is zero at finite
  values and NaN at an infinite or NaN value), with VJPs.
- `Conv.lean`: the channels-first, unbatched convolution that the CROWN convolution matrices and
  the convolution derivative proofs are stated for: grouped, dilated
  convolution with block-diagonal dense weights, and its dense case with JVP and backward rules.
  The general operators on any layout are `Tensor.conv` and `Tensor.convTranspose` in
  `TorchLean/Tensor/Window.lean`. `Conv/Window.lean` proves that the dense
  contraction `Conv.contract` is `Tensor.conv` of the input with a batch axis of one, and that its
  input, kernel and bias gradients are `Tensor.convBackwardInput`, `Tensor.convBackwardWeight` and
  a `Tensor.reduce .sum` of the cotangent. `Window` is the kernel, stride and padding record that
  convolution, transpose convolution and pooling layers share, with one output rule for each.
- `Pooling.lean`: the output extents of spatial pooling, `Pool.outSpatial` and
  `Window.poolOutputSpatial`.
  The pools themselves are `Tensor.pool` with its JVP and backward (`Tensor.poolJVP`,
  `Tensor.poolBackward`), `Tensor.smoothMaxPool` and `Tensor.adaptivePool` in
  `TorchLean/Tensor/Window.lean`, over any axes. A max or min pool selects the first extremum in
  row-major window order, with `-0` below `0`, prefers a NaN to any number, and gives `0` for a
  window of padding only (torch's `max_pool1d` gives `-inf`). `Conv/Window.lean`
  proves that `Tensor.pool (.avg true)` over every axis of a spatial tensor, with unit dilation,
  symmetric padding and floor mode, divides by the number of kernel cells
  (`poolCount_true_finRange`). It also proves the channels-first shape equations
  (`pool_finRange_spatial`, `pool_tail_channelsFirst`). `TorchLean/Tensor/Window/PoolLeading.lean`
  proves that pooling past an unpooled leading shape pools each slice (`Tensor.pool_concat`).
- `Normalization.lean`: `Tensor.layerNorm` and `Tensor.rmsNorm` over any axes,
  `Tensor.batchNorm` along any channel axis, with batch or running statistics, and
  `Tensor.groupNorm` over contiguous channel groups along any channel axis.
  `Normalization/LayerNorm.lean` holds the JVP and VJP of LayerNorm,
  `Normalization/BatchNorm.lean` those of BatchNorm with batch statistics, and
  `Normalization/GroupNorm.lean` group normalization with channels on axis 1.
- `Embedding.lean`: `Tensor.embedding`, the gather of table rows at an index tensor of any shape,
  and `Tensor.embeddingBackward`, its scatter-add table gradient.
- `PositionalEncoding.lean`: `Tensor.sinusoidalPositionalEncoding` and `Tensor.rope` (with its
  factor tables `Tensor.ropeCos` and `Tensor.ropeSin`) over any shape, positions along one axis
  and features along another.
- `Dropout.lean`: `Dropout.masked`, the mask-driven training-mode dropout spec.
- `MoE.lean`: mixture of experts: top-`k` routing with ties to the lower index, Mixtral's gates, and
  the gated sum in increasing expert index.
- `Loss.lean`: common scalar losses and their derivatives. `Loss/Axis.lean` holds
  `Tensor.crossEntropy`, which takes its input kind (`.probabilities epsilon` or `.logits`) as an
  argument and its classes along any axis, and `Tensor.cosineLoss` over any axes.
- `SpecialForms.lean`: `Softmax.maxVec` and the channel broadcast of a `[channels, positions]`
  matrix stated as the general `reduce` and `broadcast`.
- `GCN.lean`: a compact GCN-style graph layer.
- `RNN.lean`, `LSTM.lean`, `GRU.lean`: the Elman, LSTM and GRU cells and their sequences, forward
  only. Their gradients come from the tape through the runtime cells of
  `TorchLean/NN/Layer/Recurrent.lean`, which `TorchLean/NN/Layer/Recurrent/Correctness.lean`
  proves equal to these cells under the pure interpreter.
- `StateSpace.lean`: affine scan primitives used by S4/Mamba style state space models.
- `RealFFT.lean`: packed real-DFT matrices and their inverse synthesis matrices, used by Fourier
  runtime operators and autograd. `RealFFT/Real.lean`: over the reals they are the matrices of the
  angle `2π t f / n`. `RealFFT/Complex.lean`: the complex DFT matrices `dftMatrix` and `idftMatrix`
  on the kernel of Mathlib's `ZMod.dft`, their inversion, and the entries of the packed matrices at
  `ℝ` in terms of that kernel.

## Files

### Tensors, Storage And Packs

- `Datatype.lean`: the rank-zero constructor `scalar` and its `item`, the equivalences
  `scalarEquiv`, `dimEquiv` and `vectorEquiv`, indexing, and the coordinate laws that proofs use.
- `Storage.lean`: physical-storage selection and the `toArray` laws.
- `Rep.lean`: the tensor type `Tensor α shape`, defined as `Rep α shape`, the native row-major
  buffer with its length proof. It also defines `set`, `modify`, the row-major `foldl`, which does
  not convert the buffer, and the shape-aware `Repr` and `ToString` instances.
- `Literal.lean`: bracket elaboration for tensor expected types.
- `Constructors.lean`: scalar literals, flat-index generation (`generateFlat`) and bounded-index
  constructors such as `oneHot`.
- `OfFn.lean`: total constructors that unfold definitionally: the constants `Tensor.zeros` and
  `Tensor.ones`, and `Tensor.generate`, `Tensor.ofFn` and `Tensor.matrix` from index functions.
- `Conversion.lean`: total `from` and `to`, and the zero-copy `reshape`.
- `Promotion.lean`: element conversion (`ElementCast`), the tensor `cast` built on it, and the
  promotion relation (`ElementPromotion`) behind mixed-type arithmetic.
- `Operations.lean`: indexing, stacking, concatenation, mixed-dtype arithmetic, `matvec`,
  `vecmat`, `linear` and shape helpers.
- `Pack.lean`: `TorchLean.Pack` and `TensorPack`. Runtime references, graph variables, error
  bounds and optimizer states are packs over their own families and share its lemmas.
- `Einops/Elab/Pack/Result.lean`: `Packed`, the result type shared by `pack` and `unpack`.
- `SomeTensor.lean`: `SomeTensor`, whose stored shape is checked before a typed tensor is
  recovered.
- `ShapeErasure.lean`: checked conversion between `TensorPack` and arrays of `SomeTensor`.
- `Bytes.lean`: little-endian encoding and decoding of fixed-width words and the checked
  conversion from `Nat` to `UInt32` (`toUInt32?`), with proofs that decoding inverts encoding and
  that the 8- and 4-byte readers agree with the general decoder.

### Shapes, Axes And Coordinates

- `Shape.lean`: the list-based `Shape` representation, natural-number axes, adjacent-axis swaps
  (`swapAdjacentAtDepth`), well-formedness, and broadcasting: the decidable proposition
  `CanBroadcastTo` with its typeclass wrapper `BroadcastTo`. Natural-number axes come with the
  evidence `AxisInBounds`, `NonemptyAxis` and `HasNonemptyAxis`, the extent `axisSize`, and the
  runtime checks `axisInBounds?` and `nonemptyAxis?`.
- `Axes.lean`: axes as `Fin s.rank`, so an axis is in range by construction: extents, the index of
  a coordinate along an axis, the permuted, sliced, padded and erased shapes, the placement
  `BroadcastDims`, and signed axes in PyTorch's convention (`axis?`, `axes?`, `perm?`).
- `Stride.lean`: row-major strides (`stride`), the flat index of a coordinate as the sum of its
  indices times the strides (`Coord.linearize_val_eq_sum`), and `weightedIndex`, which computes
  such a sum from a flat index and is how compiled code computes `permute`.
- `LastAxis.lean` reads last-axis rows, and `Fiber.lean` reads an operation along an axis on the
  fibers through a coordinate.
- `ConcatSlice.lean` reads concatenations, `ShapeChange.lean` holds the `reshape` lemmas and
  `sequenceFin`, and `AdjacentSwap.lean` the adjacent-axis swaps.
- `Sequence.lean`: combinators on leading shape prefixes, and sequence traversal that records its
  length in the result type.

### Scalars

- `Context.lean`: the `Context` typeclass for scalar backends, with algebraic operations,
  comparisons, casts (including `NatCast`), the elementary functions PyTorch adds (`copysign`,
  `isNaN`, `tan`, `exp2`, `log2`, `asinh`, `hypot`, ...) and the little extra structure specs
  need. It also defines the `LawfulContext` class and the capability `Rounding` (`floor`, `ceil`,
  `trunc`, `round` with ties to even, C's `fmod`) for the scalar types that have it.
  `Context/Real.lean` holds the real instances and `Context/Rational.lean` the rational instances,
  scoped in `TorchLean.RationalAlgebraic`. These compute the field operations, the comparisons,
  `abs` and integer powers exactly and give `0` for `exp`, `log`, `sqrt`, `sin`, `cos`, `sinh`,
  `cosh`, `tanh`, `pi` and non-integer powers.
- `ConfiguredBinary.lean`: the storage and `Context` instances of FloatLib's configured binary
  scalars (`FloatLib.Floats.ExecFloat.Binary`). `ConfiguredBinary/Elementary.lean` and
  `ConfiguredBinary/InverseTrigonometric.lean` give their correctly rounded elementary and inverse
  trigonometric functions, and `NF.lean` a noncomputable `Context` for FloatLib's rounded-real
  `NF`.
- `FromFloat.lean`: `FromFloat` and `ofFloat`, which convert Lean `Float`
  constants into a runtime element type in a chosen rounding mode.
- `Numeric.lean`: `Numeric.sigmoid`, the input-based slopes `sigmoidDeriv` and `tanhDeriv` with
  their pullbacks, the compensated `log1p` and `expm1`, and TorchLean's native binary32 instance of
  FloatLib's `MathFunctions`.
- `Word.lean`: integer words `BitVec w`, read as signed or unsigned integers (`toInteger`, `fits`),
  with PyTorch's integer operators (`divTrunc`, `divFloor`, `remainder`, `pow`, `lt`, `le`,
  `maximum`, `minimum`).
- `Erf.lean`: the Gauss error function `erf` on the reals, with the facts `torch.erf` and the exact
  GELU rely on.
- `Complex.lean`: small complex-number support for FFT and FNO-style specifications.

### Operators

- `Arithmetic.lean`: elementwise maps and pointwise tensor operations, most of them the packed
  kernels of `Tensor.map` and `zipWith`.
- `Elementwise.lean`: the elementwise operators of core ATen and `jax.lax` that `Arithmetic.lean`
  does not define (`sign`, the integral roundings, `fmod`, `remainder`, `maximum`, `fmax`, `clamp`
  with tensor bounds, the comparisons `eq` to `ge` and `isNaN` returning Boolean tensors, `select`,
  `lerp`, `nanToNum`, `tan`, `atan2`, ...) and their scalar functions. Each operator is built over
  `map`, `zipWith` or `zipWith3` and has a read lemma.
- The general operators take their axes as `Fin s.rank`, work at every rank and come with a read
  lemma `op_apply`:
  - `Movement.lean`: data movement (`broadcast`, `permute`, `reshape`, `slice`, `pad`, `cat`,
    `flip`, `indices`, `gather`).
  - `Reduce.lean`: `reduce` with `.sum`, `.prod`, `.max` and `.min`, `mean` over chosen axes and
    `meanDenominator` (the element count, or one for an empty shape), the scalar `sum` and
    `meanSquaredError`, folds such as `maxAbs`, `all` and `dot`, and the splits of a coordinate at
    an axis (`Coord.axisEquiv`, and `Coord.setEquiv` for a replaced extent). `Reduce/Backward.lean`
    holds sums over one axis and the pullbacks of reductions, and `Reduce/Fold.lean` holds
    `maxAbsDiff`.
  - `ArgMax.lean`: `argmax` and `argmin`. `Cumulative.lean`: `cumsum`, `cumprod`, `cummax` and
    `cummin`. `Sort.lean`: `sort` and `topk`.
  - `Contract.lean`: `contract`. `Scatter.lean`: `scatter`. `Statistics.lean`: the variance `var`.
  - `Window.lean`: `conv` and `convTranspose` with dimension numbers (`Shape.ConvDims`) and their
    gradients, `pool` with `.sum`, `.max`, `.min` and `.avg` over any axes and its gradient,
    `smoothMaxPool` and `adaptivePool`.
  - `Movement/SpecialForms.lean` reads entries of `cat`, `permute` and `gather` through their
    indices.
- `Random.lean`: the deterministic `TorchLean.Random` helpers (`splitmix64`, `counterStep`,
  `keyOf`, `nextSeed`, `sampleNat`, uniform and normal sampling, dropout keep bits) shared by the
  spec layer and the runtime. `Random/Permutation.lean` builds keyed Fisher-Yates permutations on
  `sampleNat` for data shuffles and feature sampling.

### Linear Algebra And Theorems

- `LinearAlgebra.lean`: `qr` and its result `QRFactors`. `LinearAlgebra/Identity.lean` holds
  `identity`, and `LinearAlgebra/Algorithms.lean` reference numerics such as determinants,
  inverses and power iteration (`determinant`, `inverse?`, `powerIteration`).
- `Factorization.lean`: the Cholesky and QR recurrences over any scalar backend, with
  `Tensor.cholesky`, `Tensor.solveRidge` and the triangular solves.
- `Algebra.lean`: backend-generic algebra over commutative semirings, for proofs that do not
  commit to `ℝ`, with the pairing `Rep.dot` split along the leading axis
  (`dot_eq_sum_unstack`).
- `Reduce/Real.lean`, `LinearAlgebra/Norm.lean`, `LinearAlgebra/Real.lean` and
  `ShapeChange/Flatten.lean`: the real toolkit of folds, bounds, norms, linear algebra with the
  Mathlib matrix view of rank-two tensors (`Rep.matrixEquiv`), and flattening.
- `Euclidean.lean`: the norm, inner product and Euclidean topology of real tensors.
- `AdjacentSwap/Adjoint.lean`: permuting the axes of both arguments of `Rep.dot` by one `σ` keeps
  their inner product, and the adjoint of `Tensor.permute` by `σ` is `Tensor.permute` by
  `s.unpermute σ h`. `Tensor.permuteCLM`, the permutation as a continuous linear map, is its
  derivative at every point.
- `Movement/BroadcastAdjoint.lean`: `Tensor.broadcast` reads its input along a coordinate map and
  `Tensor.reduceBroadcast` sums over the fibers of that map, so the two are adjoint for the dot
  product over any semiring and for the inner product over `ℝ`.
- `List.lean`: fold lemmas for lists.

`Euclidean.lean` transports the inner product of `EuclideanSpace ℝ s.Coord` to `Tensor ℝ s`, so
Mathlib's facts about inner-product spaces apply to real tensors directly. The autograd,
runtime-approximation and model proofs in the NN and Verify layers build on these files. Import the
module that states the fact you use, for example the Euclidean structure:

```lean
import TorchLean.Tensor.Euclidean
```
