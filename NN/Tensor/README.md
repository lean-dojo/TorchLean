# `NN.Tensor`

This folder is TorchLean's user-facing tensor API. Import it directly for tensor-only code:

```lean
import NN.Tensor
```

Ordinary model code should usually `import NN.API`.

## One Tensor Type

`Tensor α shape` is the single shape-indexed representation used for proofs and CPU execution.
Every value owns one contiguous row-major buffer whose length is certified by its static shape.
The proof is erased by code generation.

`Storage α` selects the physical buffer:

- `Float` uses `FloatArray`.
- `UInt8` uses `ByteArray`.
- Arbitrary element types, including proof-oriented types, use `Array α`.

The tensor also acts as a total coordinate function in statements and proofs. This observation view
does not allocate a second recursive tensor.

## Construction And Conversion

Ordinary nested brackets construct a tensor whenever a tensor type is expected:

```lean
def threshold : Tensor Float [] := 0.5

def matrix : Tensor Float [2, 2] :=
  [[1.0, 2.0], [3.0, 4.0]]
```

`[]` is the public rank-zero shape, and an ordinary numeric literal constructs its single value.
There is no separate scalar-tensor syntax to learn. Positive-rank tensors use brackets; the
elaborator rejects ragged literals and shape mismatches.

## Inspection And Display

Tensors define their own shape-aware `Repr` instance, which is what Lean's
`#eval` command uses:

```lean
def left : Tensor Nat [2, 2] := [[1, 2], [3, 4]]
def right : Tensor Nat [2, 2] := [[10, 20], [30, 40]]

#eval left
-- [[1, 2], [3, 4]]

#eval left + right
-- [[11, 22], [33, 44]]
```

There is no need to convert a tensor to inspect it. Use `#eval tensor` at the
command line or `IO.println (reprStr tensor)` inside an `IO` program.
`Tensor.to` is only for serialization, interoperability, and APIs that
explicitly require another container.

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

def asVector : Vector Float (Spec.Shape.size [2, 2]) :=
  Tensor.to matrix' (Vector Float (Spec.Shape.size [2, 2]))
```

- `Tensor.from source` is total and preserves the source element type.
- `tensor.reshape target` changes only the static shape and does not copy. Lean
  solves concrete equal-size shapes automatically; symbolic shapes may require
  an explicit size proof.
- `tensor.cast TargetElement` explicitly changes the element type.
- `Tensor.to tensor TargetType` converts to a requested in-memory representation.

External files and runtime metadata remain checked boundaries. Their loaders validate claimed
dimensions and payload lengths before constructing an ordinary `Tensor`.

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

Use `tensor.at coordinate`, `tensor.set coordinate value`, and
`tensor.modify coordinate f` when a complete typed coordinate is convenient.
Updates are immutable copy-on-write operations. `Float` and `UInt8` stay in
their packed buffers; generic element types use the ordinary array fallback.
The simplifier knows that reading the updated coordinate returns the new value.

Runtime coordinate arrays use the checked `at?`, `set?`, and `modify?`
boundaries and return `none` for the wrong rank or an out-of-bounds index.

## Shape Pattern Operations

Importing `NN.Tensor` or `NN` exposes every checked shape-pattern operation on
the ordinary `Tensor` type. The pattern keywords are scoped syntax, so a file
activates them with `open TorchLean.Tensor`:

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

`rearrange`, `expand`, `reduce`, `einsum`, `pack`, `unpack`, and
`parse_shape` check literal patterns and static shapes while Lean elaborates
the definition. `expand` is the einops `repeat` operation; the keyword differs
so that it does not collide with Lean's own `repeat`. The output names in a pattern select and order dimensions from
the inputs, so the result type records the output shape. A conflicting type
annotation is rejected before the program runs. Multi-input `einsum` and
`pack` use the same automatic scalar promotion as ordinary tensor arithmetic.
`pack` metadata feeds directly into `unpack`; write integer metadata manually
only when using the optional `-1` dimension inference. `parse_shape` returns
ordinary named dimension data, here `[("row", 2), ("column", 3)]`.

These are term elaborators: they construct tensor expressions. The separate `einops` proof tactic
proves tensor identities, while `einops?` explains a transformation in the InfoView. Their proof
and report engines live under `NN/Tactic/Einops`; `NN.Tensor` continues to expose both tactics.

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

`Tensor.qr` computes both factors once and returns named `.q` and `.r` fields. For an
`m × n` input, their reduced shapes are `m × min m n` and `min m n × n`. Tall and square inputs use
the theorem-backed reference factorization. The wide path retains independent basis columns, so an
early dependent source column does not prevent a later independent column from entering the basis.
`Tensor.cholesky` returns the lower-triangular factor candidate, while
`Tensor.solveRidge` solves a regularized symmetric system through the same
Cholesky path. These are executable, generic reference algorithms shared with
the proof layer. They are not replacements for blocked, vectorized LAPACK or
backend kernels on large matrices.

## Runtime Relationship

There is no second host tensor class. Specifications, proofs, and native CPU operations use the
same certified buffer representation.

- `TensorPack α shapes` is a statically heterogeneous collection used for model state.
- `SomeTensor α` is internal runtime shape erasure for tapes and graph interpreters.
- Runtime value references are handles owned by an eager session or typed graph.
- CUDA uses session-owned device buffers. Upload and download are explicit `IO`
  boundaries that validate shape and buffer length. Those checks preserve
  dtype, shape, and bounds safety; numerical kernel correctness remains an
  explicit backend contract rather than a consequence of the tensor type.

## Files

- `../Tensor.lean`: public umbrella import.
- `Storage.lean`: physical-storage selection.
- `Conversion.lean`: total `from`, `to`, and zero-copy `reshape`.
- `Constructors.lean`: constants, flat generation, and bounded-index constructors.
- `Operations.lean`: indexing, stacking, scalar `cast`, mixed-dtype arithmetic, shape helpers, and
  standard `Repr` output.
- `Reductions.lean`: scalar `sum`, `mean`, and `meanSquaredError`.
- `LinearAlgebra.lean`: identity, QR, Cholesky, and ridge solve.
- `Pack.lean`: statically heterogeneous model state.
- `ShapeErasure.lean`: checked conversion to internal runtime shape erasure.
- `Internal/Elab/TensorLiteral.lean`: ordinary bracket elaboration for tensor expected types.
