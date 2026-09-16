/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Spec.Layers.Activation -- shake: keep
public import NN.Spec.Layers.Linear -- shake: keep
public import NN.Tensor.Internal.Representation.Promotion -- shake: keep
public import NN.Spec.Core.TensorReductionShape.ConcatSlice -- shake: keep
public import NN.Tensor.Constructors -- shake: keep

/-!
# Public Tensor Operations

Shape-polymorphic lookup, axis operations, and list-shaped mapping and flattening helpers for
`TorchLean.Tensor`. The implementations delegate to the canonical specification operations.
-/

@[expose] public section

namespace TorchLean.Tensor

export Spec (get)

/-- Selecting one independently mapped batch row recovers the result for that row. -/
@[simp] theorem unstack_mapLeading {α : Type} [TorchLean.Storage α]
    {batch : Nat} {inShape outShape : Spec.Shape}
    (f : Tensor α inShape → Tensor α outShape)
    (xs : Tensor α (inShape.prependDim batch)) (i : Fin batch) :
    unstack (mapLeading [batch] f xs) i = f (unstack xs i) := by
  simp [mapLeading]

/-- Independent mapping commutes with composition across any leading shape. -/
theorem mapLeading_comp {α : Type} [TorchLean.Storage α]
    (leading : Spec.Shape) {s₁ s₂ s₃ : Spec.Shape}
    (f : Tensor α s₁ → Tensor α s₂) (g : Tensor α s₂ → Tensor α s₃)
    (xs : Tensor α (leading.concat s₁)) :
    mapLeading leading (fun x => g (f x)) xs =
      mapLeading leading g (mapLeading leading f xs) := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      simp [mapLeading, ih]

/--
Convert every tensor entry to a new element type while preserving its shape.

The target `Storage` instance selects the target physical representation.
-/
def cast {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (tensor : Tensor α shape) (target : Type)
    [TorchLean.Storage target] [TorchLean.ElementCast α target] :
    Tensor target shape :=
  TorchLean.Tensor.Internal.Rep.cast tensor target

/-- Add equally shaped tensors, automatically promoting mixed element types. -/
def add {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [TorchLean.Tensor.Internal.PointwiseAdd α β γ]
    {shape : Spec.Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  TorchLean.Tensor.Internal.Rep.add left right

/-- Multiply equally shaped tensors, automatically promoting mixed element types. -/
def mul {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [TorchLean.Tensor.Internal.PointwiseMul α β γ]
    {shape : Spec.Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  TorchLean.Tensor.Internal.Rep.mul left right

/-- Subtract equally shaped tensors, automatically promoting mixed element types. -/
def sub {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [TorchLean.Tensor.Internal.PointwiseSub α β γ]
    {shape : Spec.Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  TorchLean.Tensor.Internal.Rep.sub left right

/-- Divide equally shaped tensors, automatically promoting mixed element types. -/
def div {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [TorchLean.Tensor.Internal.PointwiseDiv α β γ]
    {shape : Spec.Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  TorchLean.Tensor.Internal.Rep.div left right

/-- Multiply every tensor entry by one scalar. Public spelling of `scaleSpec`. -/
abbrev scale {α : Type} [TorchLean.Storage α] [Mul α]
    {shape : Spec.Shape} (tensor : Tensor α shape) (scalar : α) :
    Tensor α shape :=
  TorchLean.Tensor.scaleSpec tensor scalar

/-- Clamp every tensor entry to the closed interval `[minimum, maximum]`. -/
def clamp {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (minimum maximum : α) : Tensor α shape :=
  TorchLean.Tensor.clampSpec tensor minimum maximum

/-- Square every tensor entry.

This is `squareSpec` under the weaker `[Mul α]` requirement; see `square_eq_squareSpec`. -/
def square {α : Type} [TorchLean.Storage α] [Mul α]
    {shape : Spec.Shape} (tensor : Tensor α shape) :
    Tensor α shape :=
  tensor.map (fun value => value * value)

/-- Under a full scalar `Context`, the public square is the specification square. -/
theorem square_eq_squareSpec {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Spec.Shape} (tensor : Tensor α shape) :
    square tensor = TorchLean.Tensor.squareSpec tensor :=
  rfl

/--
Multiply an `m x n` matrix by an `n x p` matrix.

Example:
```lean
-- `[2, 3] * [3, 2]` contracts the shared `3`. A mismatch is a type error, not a runtime message.
def left : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 2] := [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def product : Tensor Float [2, 2] := Tensor.matmul left right
```
-/
def matmul {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) :
    Tensor α [m, p] :=
  TorchLean.Tensor.matMulSpec left right

/--
Multiply an `m x n` matrix by an `n`-element vector.

Named after the BLAS level-2 operation the way `matmul` is named after level 3, so the three
products read as a family at a call site: `matmul`, `matvec`, `vecmat`.

Example:
```lean
-- Matrix times column vector. It stays a separate name instead of an overload of `matmul` so the
-- shapes a reader should expect are visible at the call site, the way BLAS separates gemv.
def matrix : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def rowSums : Tensor Float [2] := Tensor.matvec matrix [1.0, 1.0, 1.0]
```
-/
def matvec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n : Nat}
    (matrix : Tensor α [m, n]) (vector : Tensor α [n]) :
    Tensor α [m] :=
  TorchLean.Tensor.matVecMulSpec matrix vector

/--
Multiply an `m`-element row vector by an `m x n` matrix. See `matvec` for the naming.

Example:
```lean
-- Row vector times matrix: the same product read from the other side, with no transpose.
def matrix : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def columnSums : Tensor Float [3] := Tensor.vecmat [1.0, 1.0] matrix
```
-/
def vecmat {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n : Nat}
    (vector : Tensor α [m]) (matrix : Tensor α [m, n]) :
    Tensor α [n] :=
  TorchLean.Tensor.vecMatMulSpec vector matrix

/-- Apply rectified linear activation pointwise. -/
def relu {α : Type} [TorchLean.Storage α] [Zero α] [Max α] [BEq α]
    {shape : Spec.Shape} (tensor : Tensor α shape) :
    Tensor α shape :=
  Activation.reluSpec tensor

/-- Apply logistic sigmoid pointwise. -/
def sigmoid {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Spec.Shape} (tensor : Tensor α shape) :
    Tensor α shape :=
  Activation.sigmoidSpec tensor

/-- Apply hyperbolic tangent pointwise. -/
def tanh {α : Type} [TorchLean.Storage α] [Context α]
    {shape : Spec.Shape} (tensor : Tensor α shape) :
    Tensor α shape :=
  Activation.tanhSpec tensor

namespace Internal

/-- Apply an affine map to the innermost axis, recursing through the leading axes.

The `EndsWith.Proof` argument is what makes this total: it witnesses that `shape` really does end in
`inputWidth`, and `endsWith.replace outputWidth` computes the result shape by swapping that last
dimension. So a `[batch, seq, in]` input gives a `[batch, seq, out]` output with no reshaping. -/
def linearAlongFinalAxis
    {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {shape : Spec.Shape} {inputWidth outputWidth : Nat}
    (endsWith : Spec.Shape.EndsWith.Proof inputWidth shape)
    (input : Tensor α shape)
    (weight : Tensor α [outputWidth, inputWidth])
    (bias : Tensor α [outputWidth]) :
    Tensor α (endsWith.replace outputWidth) :=
  match endsWith with
  | .last => Spec.linearSpec { weights := weight, bias := bias } input
  | .leading inner =>
      Tensor.dim fun index =>
        linearAlongFinalAxis inner (Tensor.unstack input index) weight bias

end Internal

/--
Apply `output = input * weightᵀ + bias` along the final axis.

Every leading axis is preserved. The weight layout is `[outputWidth, inputWidth]`, matching
PyTorch's `linear` convention, and Lean infers `inputWidth` from the input tensor type.

Example:
```lean
-- The weight layout is `[outputWidth, inputWidth]`, as in PyTorch, and leading axes pass straight
-- through: a `[4, 3]` batch of rows comes back as `[4, 2]`.
def weight : Tensor Float [2, 3] := [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

def bias : Tensor Float [2] := [0.5, -0.5]

def project (rows : Tensor Float [4, 3]) : Tensor Float [4, 2] :=
  Tensor.linear rows weight bias
```
-/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {shape : Spec.Shape} {inputWidth outputWidth : Nat}
    [endsWith : Spec.Shape.EndsWith inputWidth shape]
    (input : Tensor α shape)
    (weight : Tensor α [outputWidth, inputWidth])
    (bias : Tensor α [outputWidth]) :
    Tensor α (shape.replaceLast outputWidth) :=
  endsWith.proof.replace_eq_replaceLast outputWidth ▸
    Internal.linearAlongFinalAxis endsWith.proof input weight bias

/-- Read one scalar at a statically valid coordinate. -/
@[inline] def «at» {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) : α :=
  tensor coordinate

/-- Replace one scalar at a statically valid coordinate. -/
@[inline] def set {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (value : α) : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.set tensor coordinate value

/-- Transform one scalar at a statically valid coordinate. -/
@[inline] def modify {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (f : α → α) : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.modify tensor coordinate f

/-- Reading a coordinate immediately after replacing it returns the new value. -/
@[simp] theorem at_set_self {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (value : α) :
    (tensor.set coordinate value).at coordinate = value :=
  TorchLean.Tensor.Internal.Rep.get_set_self tensor coordinate value

/-- Reading a coordinate after modifying it returns the transformed old value. -/
@[simp] theorem at_modify_self {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (f : α → α) :
    (tensor.modify coordinate f).at coordinate = f (tensor.at coordinate) :=
  TorchLean.Tensor.Internal.Rep.get_modify_self tensor coordinate f

/-- Read one scalar by runtime coordinates, returning `none` when validation fails. -/
def at? {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) : Option α :=
  (Spec.Shape.Coord.ofArray? shape coordinates).map tensor.at

/-- Replace one scalar by runtime coordinates, returning `none` when validation fails. -/
def set? {α : Type} [TorchLean.Storage α] [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) (value : α) : Option (Tensor α shape) :=
  (Spec.Shape.Coord.ofArray? shape coordinates).map fun coordinate =>
    tensor.set coordinate value

/-- Transform one scalar by runtime coordinates, returning `none` when validation fails. -/
def modify? {α : Type} [TorchLean.Storage α] [TorchLean.Storage.Update α]
    {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) (f : α → α) : Option (Tensor α shape) :=
  (Spec.Shape.Coord.ofArray? shape coordinates).map fun coordinate =>
    tensor.modify coordinate f

namespace Internal

/-- Stack `count` tensors along a chosen axis.

The recursion peels leading dimensions until the insertion point is reached; the `axis = 0` case is
plain `Tensor.dim`. The final branch is impossible, since `axis + 1 ≤ 0` cannot hold for a scalar
shape, and `grind` discharges it from the arithmetic. -/
def stackAtAxis {α : Type} [TorchLean.Storage α] {count : Nat} :
    {shape : Spec.Shape} → (axis : Nat) → (hAxis : axis ≤ shape.rank) →
      (Fin count → Tensor α shape) → Tensor α (shape.insertAxis axis count)
  | shape, 0, _, tensors => by
      rw [Spec.Shape.insertAxis_zero]
      exact TorchLean.Tensor.dim tensors
  | .dim _ rest, axis + 1, hAxis, tensors =>
      TorchLean.Tensor.dim fun i =>
        stackAtAxis axis (by simp only [Spec.Shape.rank] at hAxis; grind) fun j =>
          Spec.get (tensors j) i
  | .scalar, axis + 1, hAxis, _ => by
      simp only [Spec.Shape.rank] at hAxis
      grind

end Internal

/-- Stack equally shaped tensors along any valid insertion axis. -/
def stack {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape}
    (axis : Nat) (tensors : Fin count → Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) :
    Tensor α (shape.insertAxis axis count) :=
  Internal.stackAtAxis axis hAxis tensors

/--
Stack equally shaped tensors along a new leading axis.

Example:
```lean
-- Build a leading axis from an index function: three rows of width two become one `[3, 2]`.
def rows : Tensor Float [3, 2] :=
  Tensor.stackLeading fun (row : Fin 3) =>
    Tensor.ofFn fun (column : Fin 2) => (row.val + column.val).toFloat
```
-/
def stackLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape}
    (tensors : Fin count → Tensor α shape) :
    Tensor α (shape.prependDim count) :=
  TorchLean.Tensor.dim tensors

/-- Selecting from a leading-axis stack recovers the selected tensor. -/
@[simp] theorem unstack_stackLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape}
    (tensors : Fin count → Tensor α shape) (index : Fin count) :
    unstack (stackLeading tensors) index = tensors index := by
  simp [stackLeading]

/-- Indexing a leading-axis stack recovers the selected tensor. -/
@[simp] theorem get_stackLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape}
    (tensors : Fin count → Tensor α shape) (index : Fin count) :
    Spec.get (stackLeading tensors) index = tensors index := by
  simp [Spec.get]

/-- Repeat a tensor along any newly inserted axis. -/
def repeatAxis {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (axis count : Nat) (tensor : Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) : Tensor α (shape.insertAxis axis count) :=
  stack axis (fun _ => tensor) hAxis

/-- Repeat a tensor along a new leading axis. -/
def repeatLeading {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (count : Nat) (tensor : Tensor α shape) :
    Tensor α (shape.prependDim count) :=
  stackLeading fun _ => tensor

/-- Selecting from a leading-axis repetition recovers the repeated tensor. -/
@[simp] theorem unstack_repeatLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape} (tensor : Tensor α shape)
    (index : Fin count) :
    unstack (repeatLeading count tensor) index = tensor := by
  simp [repeatLeading]

/-- Indexing a leading-axis repetition recovers the repeated tensor. -/
@[simp] theorem get_repeatLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Spec.Shape} (tensor : Tensor α shape)
    (index : Fin count) :
    Spec.get (repeatLeading count tensor) index = tensor := by
  simp [Spec.get]

/--
Concatenate tensors along their outermost axis.

This is `concatAfter .scalar` with a simpler index shape: `.dim n shape` instead of
`Spec.Shape.scalar.concat (.dim n suffix)`. The simpler form is what the `@[simp]` lemma
`get_concat_right` below is stated against, and it is what makes a KV-cache append discharge by
`simp` instead of by a shape rewrite, so the two are kept as separate definitions rather than merged
into one with a defaulted argument.

Example:
```lean
-- Join on the leading axis: `2 + 3` rows that share the row shape `[2]`.
def top : Tensor Float [2, 2] := [[1.0, 2.0], [3.0, 4.0]]

def bottom : Tensor Float [3, 2] := [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

def stacked : Tensor Float [5, 2] := Tensor.concat top bottom
```
-/
def concat {α : Type} [TorchLean.Storage α]
    {leftCount rightCount : Nat} {shape : Spec.Shape}
    (left : Tensor α (.dim leftCount shape))
    (right : Tensor α (.dim rightCount shape)) :
    Tensor α (.dim (leftCount + rightCount) shape) :=
  TorchLean.Tensor.concatAxisSpec .scalar left right

/--
Concatenate at the axis immediately following a known leading shape.

The `After` suffix matches `flattenAfter` below: both take the leading shape that is held fixed and
act on the first axis past it. `concatAxis` was the older name, but "axis" suggested a `Nat` index
in the PyTorch `dim=` sense, which is not what the argument is.

Example:
```lean
-- The same join one axis further in, which is how a batch axis is kept out of the way:
-- `[4, 2, 3]` and `[4, 5, 3]` become `[4, 7, 3]`.
def joined (left : Tensor Float [4, 2, 3]) (right : Tensor Float [4, 5, 3]) :
    Tensor Float [4, 7, 3] :=
  Tensor.concatAfter [4] left right
```
-/
def concatAfter {α : Type} [TorchLean.Storage α]
    (leading : Spec.Shape) {leftCount rightCount : Nat} {suffix : Spec.Shape}
    (left : Tensor α (leading.concat (.dim leftCount suffix)))
    (right : Tensor α (leading.concat (.dim rightCount suffix))) :
    Tensor α (leading.concat (.dim (leftCount + rightCount) suffix)) :=
  TorchLean.Tensor.concatAxisSpec leading left right

/-- Selecting an entry from the right side of an outer-axis concatenation recovers that entry. -/
@[simp] theorem get_concat_right {α : Type} [TorchLean.Storage α]
    {leftCount rightCount : Nat} {shape : Spec.Shape}
    (left : Tensor α (.dim leftCount shape))
    (right : Tensor α (.dim rightCount shape))
    (index : Fin rightCount) :
    Spec.get (concat left right)
        ⟨leftCount + index, by grind⟩ =
      Spec.get right index := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  unfold Spec.get TorchLean.Tensor.unstack TorchLean.Tensor.Internal.Rep.unstack
  simp only [TorchLean.Tensor.Internal.Rep.get_ofFn]
  change
    (concat left right) (⟨leftCount + index, by grind⟩, coordinate) =
      right (index, coordinate)
  simp [concat, TorchLean.Tensor.concatAxisSpec,
    TorchLean.Tensor.Internal.Rep.concatenateAxis,
    TorchLean.Tensor.Internal.Rep.castShape, TorchLean.Tensor.Internal.Coord.appendEquiv]

/--
Keep the first `count` entries of any valid axis.

The bound is a proof obligation, not a runtime check, so slicing past the end of an axis cannot
compile. The default tactic unfolds `axisSize`, which closes the goal whenever the shape is a
literal; a caller with a symbolic shape passes its own proof, as the CIFAR crop in
`NN/Examples/Models/Common/RealData.lean` does.

Example:
```lean
-- Keep the first two rows. The bound `2 <= 4` is discharged at the call site, so an out-of-range
-- count fails to compile instead of failing at runtime.
def firstRows (tensor : Tensor Float [4, 3]) : Tensor Float [2, 3] :=
  Tensor.take tensor 0 2
```
-/
def take {α : Type} [TorchLean.Storage α]
    {shape : Spec.Shape} (tensor : Tensor α shape) (axis count : Nat)
    [Spec.Shape.AxisInBounds axis shape]
    (hCount : count ≤ shape.axisSize axis :=
      by simp [Spec.Shape.axisSize, Spec.Shape.getDim]) :
    Tensor α (shape.replaceAxis axis count) :=
  TorchLean.Tensor.sliceAxisRangeSpec axis tensor 0 count (by simpa using hCount)

/--
Preserve `leading` and flatten every remaining axis into one row-major vector.

Example:
```lean
-- Keep the batch axis, flatten the rest: a batch of small images becomes a batch of vectors,
-- `[8, 3, 4, 4]` to `[8, 48]`.
def vectors (images : Tensor Float [8, 3, 4, 4]) : Tensor Float [8, 48] :=
  Tensor.flattenAfter [8] images
```
-/
def flattenAfter {α : Type} [TorchLean.Storage α] [Inhabited α]
    (leading : Spec.Shape) {source : Spec.Shape}
    (tensor : Tensor α (leading.concat source)) :
    Tensor α (leading.appendDim source.size) :=
  TorchLean.Tensor.reshapeSpec tensor (by
    rw [Spec.Shape.size_concat, Spec.Shape.size_appendDim])

/-- Flatten after `leading`, then keep a checked prefix of each resulting vector. -/
def flattenThenTake {α : Type} [TorchLean.Storage α] [Inhabited α]
    (leading : Spec.Shape) (count : Nat) {source : Spec.Shape}
    (hCount : count ≤ source.size)
    (tensor : Tensor α (leading.concat source)) :
    Tensor α (leading.appendDim count) :=
  by
    let tensor' : Tensor α
        (leading.concat [source.size]) := by
      simpa only [Spec.Shape.appendDim_eq_concat] using flattenAfter leading tensor
    simpa only [Spec.Shape.appendDim_eq_concat] using
      mapLeading leading
        (fun suffix =>
          show Tensor α [count] from by
            simpa [Spec.Shape.replaceAxis] using
              take suffix 0 count (by simpa using hCount))
        tensor'

end TorchLean.Tensor
