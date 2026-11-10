/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Axes
public import TorchLean.Tensor.Elementwise
public import TorchLean.Tensor.Shape
public import TorchLean.Tensor.Movement
public import TorchLean.Tensor.Layers.Activation -- shake: keep
public import TorchLean.Tensor.Layers.Linear
public import TorchLean.Tensor.Promotion
public import TorchLean.Tensor.ConcatSlice
public import TorchLean.Tensor.Constructors -- shake: keep
public import TorchLean.Tensor.Contract

/-!
# Public Tensor Operations

Shape-polymorphic operations on `TorchLean.Tensor`: element casts, pointwise arithmetic that
promotes mixed element types, `matvec`, `vecmat` and `linear`, coordinate reads and updates, the
`mapLeading` laws, and the axis operations that stack, repeat, concatenate, take prefixes and
flatten. Each operation delegates to the canonical specification or buffer-level operation.
-/

@[expose] public section

open TorchLean.Shape (size_append)

namespace TorchLean.Tensor

/-- Selecting one independently mapped batch row recovers the result for that row. -/
@[simp] theorem unstack_mapLeading {α : Type} [TorchLean.Storage α]
    {batch : Nat} {inShape outShape : Shape}
    (f : Tensor α inShape → Tensor α outShape)
    (xs : Tensor α (Shape.dim batch inShape)) (i : Fin batch) :
    unstack (mapLeading [batch] f xs) i = f (unstack xs i) := by
  simp [mapLeading]

/-- Independent mapping commutes with composition across any leading shape. -/
theorem mapLeading_comp {α : Type} [TorchLean.Storage α]
    (leading : Shape) {s₁ s₂ s₃ : Shape}
    (f : Tensor α s₁ → Tensor α s₂) (g : Tensor α s₂ → Tensor α s₃)
    (xs : Tensor α (leading ++ s₁)) :
    mapLeading leading (fun x => g (f x)) xs =
      mapLeading leading g (mapLeading leading f xs) := by
  induction leading with
  | scalar => rfl
  | dim n rest ih =>
      simp [mapLeading, ih]

/-- Add equally shaped tensors, automatically promoting mixed element types. -/
def add {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [Tensor.Pointwise .add α β γ]
    {shape : Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  Tensor.Pointwise.apply (op := .add) left right

/-- Multiply equally shaped tensors, automatically promoting mixed element types. -/
def mul {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [Tensor.Pointwise .mul α β γ]
    {shape : Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  Tensor.Pointwise.apply (op := .mul) left right

/-- Subtract equally shaped tensors, automatically promoting mixed element types. -/
def sub {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [Tensor.Pointwise .sub α β γ]
    {shape : Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  Tensor.Pointwise.apply (op := .sub) left right

/-- Divide equally shaped tensors, automatically promoting mixed element types. -/
def div {α β γ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    [TorchLean.Storage γ]
    [Tensor.Pointwise .div α β γ]
    {shape : Shape} (left : Tensor α shape) (right : Tensor β shape) :
    Tensor γ shape :=
  Tensor.Pointwise.apply (op := .div) left right

/--
Multiply an `m x n` matrix by an `n`-element vector.

Named after the BLAS level-2 operation the way `matmul` is named after level 3, so the three
products read as a family at a call site: `matmul`, `matvec`, `vecmat`.

Example:
```lean
-- A name of its own, with no `matmul` overload, keeps the expected shapes visible at the call
-- site, the way BLAS separates gemv.
def matrix : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def rowSums : Tensor Float [2] := Tensor.matvec matrix [1.0, 1.0, 1.0]
```
-/
def matvec {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n : Nat}
    (matrix : Tensor α [m, n]) (vector : Tensor α [n]) :
    Tensor α [m] :=
  Tensor.contract matrix vector _ _ [m] .matVec

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
  Tensor.contract vector matrix _ _ [n] .vecMat

/--
Apply `output = input * weightᵀ + bias` along the final axis.

Every leading axis is preserved: the `EndsWith` instance reads the leading axes off the input
shape, and the result is `Linear.forwardLeading` over them. The weight layout is
`[outputWidth, inputWidth]`, matching PyTorch's `linear` convention, and Lean infers `inputWidth`
from the input tensor type.

Example:
```lean
-- The weight maps width 3 to width 2, so a `[4, 3]` batch of rows comes back as `[4, 2]`.
def weight : Tensor Float [2, 3] := [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

def bias : Tensor Float [2] := [0.5, -0.5]

def project (rows : Tensor Float [4, 3]) : Tensor Float [4, 2] :=
  Tensor.linear rows weight bias
```
-/
def linear {α : Type} [TorchLean.Storage α] [Add α] [Mul α] [Zero α]
    {shape : Shape} {inputWidth outputWidth : Nat}
    [endsWith : Shape.EndsWith inputWidth shape]
    (input : Tensor α shape)
    (weight : Tensor α [outputWidth, inputWidth])
    (bias : Tensor α [outputWidth]) :
    Tensor α (shape.replaceLast outputWidth) :=
  Tensor.castShape
    (Linear.forwardLeading endsWith.proof.leadingShape { weights := weight, bias := bias }
      (Tensor.castShape input endsWith.proof.leadingShape_append.symm))
    (endsWith.proof.leadingShape_append' outputWidth)

/-- Read one scalar at a statically valid coordinate. -/
@[inline] def «at» {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) : α :=
  tensor coordinate

/-- Reading a coordinate after `set` gives the value written there. -/
@[simp] theorem at_set_self {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (value : α) :
    (tensor.set coordinate value).at coordinate = value :=
  Tensor.Rep.get_set_self tensor coordinate value

/-- Reading a coordinate after `modify` with `f` gives `f` applied to the entry it replaced. -/
@[simp] theorem at_modify_self {α : Type} [TorchLean.Storage α]
    [TorchLean.Storage.Update α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinate : shape.Coord) (f : α → α) :
    (tensor.modify coordinate f).at coordinate = f (tensor.at coordinate) :=
  Tensor.Rep.get_modify_self tensor coordinate f

/-- Read one scalar by runtime coordinates, returning `none` when validation fails. -/
def at? {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) : Option α :=
  (Shape.Coord.ofArray? shape coordinates).map tensor.at

/-- Replace one scalar by runtime coordinates, returning `none` when validation fails. -/
def set? {α : Type} [TorchLean.Storage α] [TorchLean.Storage.Update α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) (value : α) : Option (Tensor α shape) :=
  (Shape.Coord.ofArray? shape coordinates).map fun coordinate =>
    tensor.set coordinate value

/-- Transform one scalar by runtime coordinates, returning `none` when validation fails. -/
def modify? {α : Type} [TorchLean.Storage α] [TorchLean.Storage.Update α]
    {shape : Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) (f : α → α) : Option (Tensor α shape) :=
  (Shape.Coord.ofArray? shape coordinates).map fun coordinate =>
    tensor.modify coordinate f

/-- Stack `count` tensors along a chosen axis.

The recursion peels leading dimensions until it reaches the insertion point, where the `axis = 0`
case is plain `Tensor.dim`. The last branch cannot occur, because `axis + 1 ≤ 0` fails for a
scalar shape, and `grind` closes it from that arithmetic. -/
def stackAtAxis {α : Type} [TorchLean.Storage α] {count : Nat} :
    {shape : Shape} → (axis : Nat) → (hAxis : axis ≤ shape.rank) →
      (Fin count → Tensor α shape) → Tensor α (shape.insertIdx axis count)
  | shape, 0, _, tensors => by
      rw [List.insertIdx_zero]
      exact TorchLean.Tensor.dim tensors
  | .dim _ rest, axis + 1, hAxis, tensors =>
      TorchLean.Tensor.dim fun i =>
        stackAtAxis axis (by simp only [Shape.rank_cons] at hAxis; grind) fun j =>
          Tensor.unstack (tensors j) i
  | .scalar, axis + 1, hAxis, _ => by
      simp only [Shape.rank_nil] at hAxis
      grind

/-- Stack equally shaped tensors along any valid insertion axis. -/
def stack {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Shape}
    (axis : Nat) (tensors : Fin count → Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) :
    Tensor α (shape.insertIdx axis count) :=
  stackAtAxis axis hAxis tensors

/-- Repeat a tensor `count` times along an axis inserted at position `axis`. -/
def repeatAxis {α : Type} [TorchLean.Storage α]
    {shape : Shape} (axis count : Nat) (tensor : Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) : Tensor α (shape.insertIdx axis count) :=
  stack axis (fun _ => tensor) hAxis

/-- Repeat a tensor `count` times along a leading axis. -/
def repeatLeading {α : Type} [TorchLean.Storage α]
    {shape : Shape} (count : Nat) (tensor : Tensor α shape) :
    Tensor α (Shape.dim count shape) :=
  Tensor.dim fun _ => tensor

/-- Selecting from a leading-axis repetition recovers the repeated tensor. -/
@[simp] theorem unstack_repeatLeading {α : Type} [TorchLean.Storage α]
    {count : Nat} {shape : Shape} (tensor : Tensor α shape)
    (index : Fin count) :
    unstack (repeatLeading count tensor) index = tensor := by
  simp [repeatLeading]

/--
Concatenate tensors along their outermost axis.

`concat` is `concatAfter .scalar` at the simpler index shape `.dim n shape`, where `concatAfter`
produces `Shape.scalar ++ .dim n suffix`. The `@[simp]` lemmas `unstack_concat_left`
and `unstack_concat_right` are stated at the simpler shape, so a KV-cache append closes by `simp`
with no shape rewrite. A single `concatAfter` with a defaulted leading shape would lose that form,
which is why `concat` is a separate definition.

Example:
```lean
-- Join on the leading axis: `2 + 3` rows that share the row shape `[2]`.
def top : Tensor Float [2, 2] := [[1.0, 2.0], [3.0, 4.0]]

def bottom : Tensor Float [3, 2] := [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

def stacked : Tensor Float [5, 2] := Tensor.concat top bottom
```
-/
def concat {α : Type} [TorchLean.Storage α]
    {leftCount rightCount : Nat} {shape : Shape}
    (left : Tensor α (.dim leftCount shape))
    (right : Tensor α (.dim rightCount shape)) :
    Tensor α (.dim (leftCount + rightCount) shape) :=
  cat (s := .dim 0 shape) ⟨0, by simp⟩ ![leftCount, rightCount]
    (Fin.cons left (Fin.cons right finZeroElim)) (.dim (leftCount + rightCount) shape)

/--
Concatenate at the axis immediately following a known leading shape.

The `After` suffix matches `flattenAfter`: both hold `leading` fixed and work on the axes after it.

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
    (leading : Shape) {leftCount rightCount : Nat} {suffix : Shape}
    (left : Tensor α (leading ++ (.dim leftCount suffix)))
    (right : Tensor α (leading ++ (.dim rightCount suffix))) :
    Tensor α (leading ++ (.dim (leftCount + rightCount) suffix)) :=
  cat (s := leading ++ (.dim 0 suffix)) ⟨leading.rank, leading.rank_lt_rank_append suffix 0⟩
    ![leftCount, rightCount]
    (Fin.cons (castShape left (leading.set_append_rank suffix 0 leftCount).symm)
      (Fin.cons (castShape right (leading.set_append_rank suffix 0 rightCount).symm)
        finZeroElim))
    (leading ++ (.dim (leftCount + rightCount) suffix))

/-- Selecting an entry from the left side of an outer-axis concatenation recovers that entry. -/
@[simp] theorem unstack_concat_left {α : Type} [TorchLean.Storage α]
    {leftCount rightCount : Nat} {shape : Shape}
    (left : Tensor α (.dim leftCount shape))
    (right : Tensor α (.dim rightCount shape))
    (index : Fin leftCount) :
    Tensor.unstack (concat left right) ⟨index, by omega⟩ = Tensor.unstack left index := by
  apply Tensor.Rep.ext
  intro coordinate
  unfold Tensor.unstack
  simp only [Tensor.Rep.get_ofFn]
  change (concat left right) (⟨index, by omega⟩, coordinate) = left (index, coordinate)
  rw [concat, cat_cons_apply, dite_eq_left index.isLt]

/-- Selecting an entry from the right side of an outer-axis concatenation recovers that entry. -/
@[simp] theorem unstack_concat_right {α : Type} [TorchLean.Storage α]
    {leftCount rightCount : Nat} {shape : Shape}
    (left : Tensor α (.dim leftCount shape))
    (right : Tensor α (.dim rightCount shape))
    (index : Fin rightCount) :
    Tensor.unstack (concat left right)
        ⟨leftCount + index, by grind⟩ =
      Tensor.unstack right index := by
  apply Tensor.Rep.ext
  intro coordinate
  unfold Tensor.unstack
  simp only [Tensor.Rep.get_ofFn]
  change
    (concat left right) (⟨leftCount + index, by grind⟩, coordinate) =
      right (index, coordinate)
  rw [concat, cat_cons_apply, dite_eq_right_of_eq_false (by simp)]
  exact congrArg right (Prod.ext (Fin.ext (by simp)) rfl)

/--
Keep the first `count` entries of any valid axis.

The bound is a proof obligation, so slicing past the end of an axis cannot compile. The default
`simp` call unfolds `axisSize` and closes the goal whenever the shape is a literal. A caller with a
symbolic shape passes its own proof, as the CIFAR crop in
`TorchLeanExamples/Models/RealData.lean` does.

Example:
```lean
-- Keep the first two rows. The default proof discharges `2 <= 4` at the call site.
def firstRows (tensor : Tensor Float [4, 3]) : Tensor Float [2, 3] :=
  Tensor.take tensor 0 2
```
-/
def take {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) (axis count : Nat)
    [Shape.AxisInBounds axis shape]
    (hCount : count ≤ shape.axisSize axis :=
      by simp [Shape.axisSize]) :
    Tensor α (shape.set axis count) :=
  Tensor.castShape
      (Tensor.slice tensor [⟨axis, Shape.AxisInBounds.proof⟩] [0] [0 + count] [1]
        (Shape.sliceValid_singleton _ ⟨axis, Shape.AxisInBounds.proof⟩ 0 count
          (by rw [← Shape.axisSize_eq_extent]; exact (by simpa using hCount))))
      (Shape.slice_singleton _ ⟨axis, Shape.AxisInBounds.proof⟩ 0 count)

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
def flattenAfter {α : Type} [TorchLean.Storage α]
    (leading : Shape) {source : Shape}
    (tensor : Tensor α (leading ++ source)) :
    Tensor α (leading ++ [source.size]) :=
  Tensor.reshape tensor _ (by
    rw [Shape.size_append, Shape.size_append_singleton])

/-- Flatten after `leading`, then keep a checked prefix of each resulting vector. -/
def flattenThenTake {α : Type} [TorchLean.Storage α]
    (leading : Shape) (count : Nat) {source : Shape}
    (hCount : count ≤ source.size)
    (tensor : Tensor α (leading ++ source)) :
    Tensor α (leading ++ [count]) :=
  by
    let tensor' : Tensor α
        (leading ++ [source.size]) := by
      exact flattenAfter leading tensor
    exact
      mapLeading leading
        (fun suffix =>
          show Tensor α [count] from by
            simpa using
              take suffix 0 count (by simpa using hCount))
        tensor'

end TorchLean.Tensor
