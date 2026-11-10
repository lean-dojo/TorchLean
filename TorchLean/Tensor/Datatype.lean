/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Shape.Coord.Conversion
public import TorchLean.Tensor.Rep.Reindex

/-!
# Scalar tensors, shape equivalences and indexing

`Tensor α shape` is TorchLean's one shape-indexed tensor representation for proofs and execution,
and this module holds its scalar and indexing API. The type itself is defined with its buffer in
`TorchLean.Tensor.Rep`, and `dim`, `unstack` and `map` in `TorchLean.Tensor.Rep.Pointwise`.

Every tensor has one row-major buffer, and the scalar type selects its physical form through
`Storage`. Generic types such as `Real` and `Float32` use `Array`, whose entries may be boxed, and
specialized instances use unboxed storage (`FloatArray` for `Float`, `ByteArray` for `UInt8`).

Proofs build tensors with `scalar` and `dim`, smart constructors over the flat buffer, and take
them apart through typed lookup, `scalarEquiv`, `dimEquiv` and extensionality.
-/

@[expose] public section

open TorchLean.Shape (rank_cons)

namespace TorchLean

open Lean PrettyPrinter Delaborator SubExpr

/-- Recognize a shape expression written as a list literal: a chain of `List.cons` ending in
`List.nil`, with any entries. -/
private meta partial def explicitTensorShape : Expr → Bool
  | shape =>
      if shape.isAppOfArity ``List.nil 1 then
        true
      else if shape.isAppOfArity ``List.cons 3 then
        explicitTensorShape shape.getAppArgs[2]!
      else
        false

/--
Show an internal tensor type under the public name `TorchLean.Tensor` in editor information and
`#check` output when the scalar type lives in `Type` and the shape is a list literal.

Universe-polymorphic proof tensors, and tensors whose shape is written any other way, keep their
internal spelling.
-/
@[app_delab Tensor.Rep]
meta def delabTensorRep : Delab := do
  withOverApp 3 do
    let expression ← getExpr
    let arguments := expression.getAppArgs
    guard <| arguments.size = 3
    guard <| (← Meta.getDecLevel arguments[0]!) == .zero
    guard <| explicitTensorShape arguments[1]!
    let scalar ← withNaryArg 0 delab
    let shape ← withNaryArg 1 delab
    pure <| Syntax.mkApp (mkIdent ``TorchLean.Tensor) #[scalar, shape]

namespace Tensor

universe u

/-- Construct a rank-zero tensor from one scalar value. -/
def scalar {α : Type} [TorchLean.Storage α] (value : α) :
    Tensor α .scalar :=
  Tensor.Rep.ofFn fun _ => value

/-- Return the value stored in a scalar tensor. -/
def item {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α .scalar) : α :=
  tensor PUnit.unit

/-- Reading back the value a scalar tensor was built from returns that value. -/
@[simp] theorem item_scalar {α : Type} [TorchLean.Storage α] (value : α) :
    (scalar value).item = value := by
  exact Tensor.Rep.get_ofFn (fun _ => value) PUnit.unit

/-- A scalar tensor evaluates to its sole scalar value. -/
@[simp] theorem scalar_apply {α : Type} [TorchLean.Storage α] (value : α) :
    (scalar value : Tensor α .scalar) PUnit.unit = value := by
  simpa [item] using item_scalar value

/-- Every shape is inhabited when the scalar type is: take the all-default tensor. -/
@[reducible, instance] def inhabited {α : Type}
    [TorchLean.Storage α] [Inhabited α] {shape : Shape} :
    Inhabited (Tensor α shape) :=
  ⟨full _ Inhabited.default⟩

/-- Rebuilding a scalar tensor from its item returns the original tensor. -/
@[simp] theorem scalar_item {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α .scalar) :
    scalar tensor.item = tensor := by
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  exact Tensor.Rep.get_ofFn (fun _ => tensor.item) PUnit.unit

/-- Two scalar tensors agreeing on their item are equal. -/
@[ext] theorem ext_scalar {α : Type} [TorchLean.Storage α]
    {left right : Tensor α .scalar}
    (h : left.item = right.item) : left = right := by
  apply Tensor.Rep.ext
  intro coordinate
  cases coordinate
  exact h

/-- Scalar tensors and scalar values are equivalent. -/
def scalarEquiv (α : Type) [TorchLean.Storage α] :
    Tensor α .scalar ≃ α where
  toFun := item
  invFun := scalar
  left_inv := scalar_item
  right_inv := item_scalar

/-- An outer axis is equivalent to a finite family of inner tensors. -/
def dimEquiv {α : Type} [TorchLean.Storage α] (n : Nat) (shape : Shape) :
    Tensor α (.dim n shape) ≃ (Fin n → Tensor α shape) where
  toFun := unstack
  invFun := dim
  left_inv := dim_unstack
  right_inv values := by
    funext index
    exact unstack_dim values index

/-- Vectors and finite scalar functions are equivalent. -/
def vectorEquiv {α : Type} [TorchLean.Storage α]
    (n : Nat) : Tensor α [n] ≃ (Fin n → α) :=
  (dimEquiv n .scalar).trans (Equiv.piCongrRight fun _ => scalarEquiv α)

/-- Cast a tensor along an equality of shapes. -/
def castShape {α : Type u} [TorchLean.Storage α] {source target : Shape}
    (tensor : Rep α source) (h : source = target) : Rep α target :=
  h ▸ tensor

/-- Casting along `rfl` is the identity, and holds by `rfl` itself. -/
@[simp] theorem castShape_rfl {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    castShape tensor rfl = tensor :=
  rfl

/-- Casting along any proof of `shape = shape` is the identity.

It holds for any witness `h`, which proof irrelevance identifies with `rfl`. -/
@[simp] theorem castShape_self {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) (h : shape = shape) :
    castShape tensor h = tensor := by
  cases h
  rfl

/-- Two casts in a row collapse into one along the composed equality. -/
@[simp] theorem castShape_trans {α : Type} [TorchLean.Storage α]
    {s₁ s₂ s₃ : Shape} (tensor : Tensor α s₁)
    (h₁₂ : s₁ = s₂) (h₂₃ : s₂ = s₃) :
    castShape (castShape tensor h₁₂) h₂₃ =
      castShape tensor (h₁₂.trans h₂₃) := by
  cases h₁₂
  cases h₂₃
  rfl

/-- The cast does not depend on which proof of the shape equality is used.

Any two proofs of a shape equality are equal (`Subsingleton.elim`), so two developments that
derived the same equality differently can share a lemma. -/
theorem castShape_proof_irrel {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    {p q : source = target} :
    castShape tensor p = castShape tensor q := by
  cases Subsingleton.elim p q
  rfl

/-- The `▸` rewrite and `castShape` are the same function.

Lean inserts `▸` on its own when a shape is rewritten in a tactic block, so without this bridge the
`castShape` simp set would silently fail to fire on goals the elaborator produced. -/
theorem eqRec_eq_castShape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (h : source = target) :
    (h ▸ tensor) = castShape tensor h := by
  cases h
  rfl

/-- Reading a cast tensor reads the original tensor at the coordinate transported back along the
shape equality. -/
theorem castShape_apply {α : Type} [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (h : source = target) (coordinate : Shape.Coord target) :
    castShape tensor h coordinate =
      tensor (_root_.cast (congrArg Shape.Coord h.symm) coordinate) := by
  cases h
  rfl

/-- `castShape_apply` for nonempty shapes, one axis at a time: the outer index is cast back along
the equality of the outer extents and the slice is cast along the equality of the inner shapes. -/
theorem castShape_dim_apply {α : Type} [TorchLean.Storage α] {m n : Nat} {source target : Shape}
    (tensor : Tensor α (.dim m source)) (h : Shape.dim m source = .dim n target)
    (coordinate : Shape.Coord (.dim n target)) :
    castShape tensor h coordinate =
      castShape (unstack tensor (Fin.cast (List.head_eq_of_cons_eq h).symm coordinate.1))
        (List.tail_eq_of_cons_eq h) coordinate.2 := by
  cases h
  exact (Tensor.unstack_apply tensor _ _).symm

/-- Reading through a tensor type equality transports the coordinate through the given shape
equality. The type-equality proof may come from elaboration or simplification. -/
theorem eq_mp_apply {α : Type} [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (typeEqual : Tensor α source = Tensor α target)
    (shapeEqual : source = target) (coordinate : Shape.Coord target) :
    (Eq.mp typeEqual tensor) coordinate =
      tensor (_root_.cast (congrArg Shape.Coord shapeEqual.symm) coordinate) := by
  cases shapeEqual
  simp only [_root_.eq_mp_eq_cast, _root_.cast_eq]

/-- `eq_mp_apply` with the transport written as `cast`. -/
theorem cast_apply {α : Type} [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (typeEqual : Tensor α source = Tensor α target)
    (shapeEqual : source = target) (coordinate : Shape.Coord target) :
    (_root_.cast typeEqual tensor) coordinate =
      tensor (_root_.cast (congrArg Shape.Coord shapeEqual.symm) coordinate) :=
  eq_mp_apply tensor typeEqual shapeEqual coordinate

/-- `castShape_proof_irrel` in `▸` form, for goals the elaborator produced. -/
theorem eqRec_proof_irrel {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    {p q : source = target} :
    (p ▸ tensor) = (q ▸ tensor) := by
  cases Subsingleton.elim p q
  rfl

attribute [grind =] castShape_rfl castShape_self castShape_trans eqRec_eq_castShape

end Tensor

namespace Tensor

/-- Recover the statically known shape from a tensor value. -/
def shape {α : Type} [Storage α]
    {shape : Shape} (_ : Tensor α shape) : Shape :=
  shape

/-! ## Indexing -/

/-- Read the scalar at a runtime coordinate list, or `none` if it is not a valid coordinate. -/
def get? {α : Type} [Storage α] {shape : Shape}
    (tensor : Tensor α shape) (indices : List Nat) : Option α :=
  (Shape.Coord.ofList? shape indices).map tensor

/-- The empty coordinate list reads the item of a rank-zero tensor. -/
@[simp] theorem get?_scalar_nil {α : Type} [Storage α]
    (tensor : Tensor α []) :
    get? tensor [] = some tensor.item := by
  simp [get?, Shape.Coord.ofList?, Tensor.item]

/-- A rank-zero tensor has no axis to index, so any nonempty coordinate list fails. -/
@[simp] theorem get?_scalar_cons {α : Type} [Storage α]
    (tensor : Tensor α []) (index : Nat) (indices : List Nat) :
    get? tensor (index :: indices) = none := by
  simp [get?, Shape.Coord.ofList?]

/-- A tensor with an axis is not a scalar, so the empty coordinate list fails. -/
@[simp] theorem get?_dim_nil {α : Type} [Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape)) :
    get? tensor [] = none := by
  simp [get?, Shape.Coord.ofList?]

/-- One step of runtime lookup: check the leading index against the axis, then recurse into the
slice. Rewriting with it turns `get?` on a literal coordinate list into a chain of `unstack`s,
which ties runtime lookup to typed slicing. -/
@[simp] theorem get?_dim_cons {α : Type} [Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape))
    (index : Nat) (indices : List Nat) :
    get? tensor (index :: indices) =
      if h : index < n then get? (Tensor.unstack tensor ⟨index, h⟩) indices else none := by
  by_cases hIndex : index < n
  · cases hCoordinate : Shape.Coord.ofList? shape indices with
    | none =>
        simp [get?, Shape.Coord.ofList?, hIndex, hCoordinate]
    | some coordinate =>
        simp [get?, Shape.Coord.ofList?, hIndex, hCoordinate]
  · simp [get?, Shape.Coord.ofList?, hIndex]

/-- A shape cast does not move any data, so runtime lookup sees straight through it. -/
@[simp] theorem get?_castShape {α : Type} [Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (h : source = target) (index : List Nat) :
    get? (Tensor.castShape tensor h) index = get? tensor index := by
  cases h
  rfl

attribute [grind =] get?_scalar_nil get?_scalar_cons get?_dim_nil get?_dim_cons

/--
Indexing a vector with a `Fin n` index returns the scalar entry.

The instance has high priority, so on vectors it wins over the slice instance. In
`matrix[row][column]` the first lookup returns the statically shaped row tensor and the second
reads its entry.
-/
instance (priority := high) instGetElemVectorFin {α : Type} [Storage α] {n : Nat} :
    GetElem (Tensor.Rep α [n]) (Fin n) α
      (fun _ _ => True) where
  getElem tensor index _ := tensor (index, PUnit.unit)

/-- Indexing with a `Fin n` index selects one slice along the outermost tensor axis. -/
instance {α : Type} [Storage α] {n : Nat} {shape : List Nat} :
    GetElem (Tensor.Rep α (n :: shape)) (Fin n)
      (Tensor α shape) (fun _ _ => True) where
  getElem tensor index _ :=
    Tensor.unstack (shape := shape) tensor index

/--
Natural-number indexing of a vector returns one scalar and asks `GetElem` to
discharge the static bound.
-/
instance (priority := high) instGetElemVectorNat {α : Type} [Storage α] {n : Nat} :
    GetElem (Tensor.Rep α [n]) Nat α
      (fun _ index => index < n) where
  getElem tensor index hIndex :=
    tensor (⟨index, hIndex⟩, PUnit.unit)

/--
Natural-number indexing selects one slice along the outermost tensor axis.

Literal indices are checked during elaboration, so ordinary code can write
`matrix[0]` without constructing a `Fin` value explicitly.
-/
instance {α : Type} [Storage α] {n : Nat} {shape : List Nat} :
    GetElem (Tensor.Rep α (n :: shape)) Nat
      (Tensor α shape) (fun _ index => index < n) where
  getElem tensor index hIndex :=
    Tensor.unstack (shape := shape) tensor ⟨index, hIndex⟩

end Tensor

namespace Tensor

/-- Runtime-list lookup is extensional. -/
@[ext] theorem ext_get? {α : Type} [TorchLean.Storage α]
    {shape : Shape} {left right : Tensor α shape}
    (h : ∀ index, get? left index = get? right index) :
    left = right := by
  apply Tensor.Rep.ext
  intro coordinate
  simpa [get?] using h (Shape.Coord.toList shape coordinate)

/-- Extract one scalar from a vector. -/
def getScalar {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) : α :=
  item (Tensor.unstack tensor index)

/-- Vector scalar lookup is ordinary coordinate evaluation. -/
theorem getScalar_eq_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) :
    getScalar tensor index = tensor (index, PUnit.unit) := by
  unfold getScalar item
  exact Tensor.unstack_apply tensor index PUnit.unit

/-- Direct storage read used when compiling vector scalar lookup. -/
@[inline] def getScalarDirect {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) : α :=
  tensor (index, PUnit.unit)

/-- Compile scalar lookup without allocating the intermediate rank-zero slice. -/
@[csimp] theorem getScalarDirect_eq :
    @getScalar = @getScalarDirect := by
  funext α storage n tensor index
  exact getScalar_eq_apply tensor index

/-- Scalar lookup on a rank-one literal returns the corresponding source entry. -/
@[simp] theorem getScalar_ofList {α : Type} [TorchLean.Storage α]
    (values : List α) (index : Fin values.length) :
    getScalar (Tensor.Rep.ofList values) index =
      values.get index := by
  rw [getScalar_eq_apply]
  exact Tensor.Rep.ofList_apply values index

/-- Scalar lookup on a vector of scalar tensors reads the indexed entry. -/
@[simp] theorem getScalar_dim_entry {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → Tensor α .scalar) (index : Fin n) :
    getScalar (dim values) index = (values index).item := by
  simp [getScalar]

/-- The `Tensor α [n] ≃ (Fin n → α)` equivalence is scalar lookup on the nose.

Stating it keeps `vectorEquiv` usable in proofs without unfolding the two equivalences it is built
from, and it is why a vector can be handed to Mathlib lemmas about functions on `Fin n`. -/
@[simp] theorem vectorEquiv_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) :
    vectorEquiv n tensor index = getScalar tensor index :=
  rfl

/-- Building a vector from a function and reading it back returns the function. -/
@[simp] theorem getScalar_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} (values : Fin n → α) (index : Fin n) :
    getScalar (dim (fun i => scalar (values i))) index = values index := by
  simp [getScalar]

/-- Vectors agreeing at every scalar index are equal. -/
@[ext] theorem ext_vector {α : Type} [TorchLean.Storage α]
    {n : Nat} {left right : Tensor α [n]}
    (h : ∀ index : Fin n, getScalar left index = getScalar right index) :
    left = right := by
  apply Tensor.Rep.ext
  intro coordinate
  rcases coordinate with ⟨index, ⟨⟩⟩
  have hIndex := h index
  unfold getScalar item at hIndex
  convert hIndex using 1
  · exact (Tensor.unstack_apply left index PUnit.unit).symm
  · exact (Tensor.unstack_apply right index PUnit.unit).symm

end Tensor

namespace Tensor

/-- Matrix element access. -/
def get2 {α : Type} [Storage α] {m n : Nat}
    (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) : α :=
  Tensor.getScalar (Tensor.unstack tensor row) column

/-- Matrix access is a row slice followed by a scalar read.

The equation holds by `rfl`, and as a named rewrite it lets a matrix proof reuse the vector lemmas
about `getScalar` at rank two. -/
theorem get2_eq_getScalar_unstack {α : Type} [Storage α]
    {m n : Nat} (tensor : Tensor α [m, n])
    (row : Fin m) (column : Fin n) :
    get2 tensor row column = Tensor.getScalar (Tensor.unstack tensor row) column :=
  rfl

/-- Matrix scalar lookup is one read at the corresponding two-axis coordinate. -/
theorem get2_eq_apply {α : Type} [Storage α]
    {m n : Nat} (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) :
    get2 tensor row column = tensor (row, column, PUnit.unit) := by
  rw [get2, Tensor.getScalar_eq_apply]
  exact Tensor.unstack_apply tensor row (column, PUnit.unit)

/-- Direct storage read used when compiling matrix scalar lookup. -/
@[inline] def get2Direct {α : Type} [Storage α]
    {m n : Nat} (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) : α :=
  tensor (row, column, PUnit.unit)

/-- Compile matrix lookup without allocating an intermediate row or scalar tensor. -/
@[csimp] theorem get2Direct_eq : @get2 = @get2Direct := by
  funext α storage m n tensor row column
  exact get2_eq_apply tensor row column

/-- Reading back a matrix built from a two-argument function returns the function. -/
@[simp] theorem get2_dim {α : Type} [Storage α]
    {m n : Nat} (values : Fin m → Fin n → α)
    (row : Fin m) (column : Fin n) :
    get2 (Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar (values i j))
      row column = values row column := by
  simp [get2]

/-- `x[(i, j)]` is the entry of a matrix at row `i` and column `j`. -/
instance instGetElemMatrix {α : Type} [Storage α] {m n : Nat} :
    GetElem (Tensor α [m, n]) (Fin m × Fin n) α (fun _ _ => True) where
  getElem tensor index _ := get2 tensor index.1 index.2

/-! ## Total indexing and shape operations -/

/-- Read the entry at a runtime coordinate list, or `0` when the list has the wrong length or an
index out of range. -/
def getAtOrZero {α : Type} [Storage α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) (indices : List Nat) : α :=
  (get? tensor indices).getD 0

/-- On a rank-zero tensor the empty coordinate list reads the item, no fallback needed. -/
@[simp] theorem getAtOrZero_scalar_nil {α : Type}
    [Storage α] [Zero α] (tensor : Tensor α []) :
    getAtOrZero tensor [] = tensor.item := by
  simp [getAtOrZero]

/-- An over-long coordinate list on a rank-zero tensor falls back to zero. -/
@[simp] theorem getAtOrZero_scalar_cons {α : Type}
    [Storage α] [Zero α] (tensor : Tensor α [])
    (index : Nat) (indices : List Nat) :
    getAtOrZero tensor (index :: indices) = 0 := by
  simp [getAtOrZero]

/-- A too-short coordinate list falls back to zero. -/
@[simp] theorem getAtOrZero_dim_nil {α : Type}
    [Storage α] [Zero α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (.dim n shape)) :
    getAtOrZero tensor [] = 0 := by
  simp [getAtOrZero]

/-- One step of total lookup: an in-range leading index recurses into its slice, and an
out-of-range one gives zero.

The zero fallback makes the function total and is the reason for the `Zero α` hypothesis, since no
shape argument can rule out a bad runtime coordinate list. -/
@[simp] theorem getAtOrZero_dim_cons {α : Type}
    [Storage α] [Zero α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (.dim n shape)) (index : Nat) (indices : List Nat) :
    getAtOrZero tensor (index :: indices) =
      if h : index < n then
        getAtOrZero (Tensor.unstack tensor ⟨index, h⟩) indices
      else
        0 := by
  by_cases hIndex : index < n
  · simp [getAtOrZero, hIndex]
  · simp [getAtOrZero, hIndex]

attribute [grind =] getAtOrZero_scalar_nil getAtOrZero_scalar_cons
  getAtOrZero_dim_nil getAtOrZero_dim_cons

/-- The index list of a coordinate reads the entry at that coordinate. The coordinate may belong
to an equal shape written differently, such as `t.data.toList` for `t.to (List Nat)`. -/
theorem getAtOrZero_coord_toList {α : Type} [Storage α] [Zero α] {shape listed : Shape}
    (tensor : Tensor α shape) (coordinate : listed.Coord) (h : listed = shape) :
    getAtOrZero tensor (Shape.Coord.toList listed coordinate) = tensor (h ▸ coordinate) := by
  subst h
  simp [getAtOrZero, get?]

/-- A leading index written as `index.val` for `index : Fin n` selects that slice. -/
theorem getAtOrZero_cons_val {α : Type} [Storage α] [Zero α] {n : Nat}
    {shape : Shape} (tensor : Tensor α (.dim n shape)) (index : Fin n) (indices : List Nat) :
    getAtOrZero tensor (index.val :: indices) =
      getAtOrZero (Tensor.unstack tensor index) indices := by
  rw [getAtOrZero_dim_cons, dite_eq_left index.isLt]

/-- A valid vector coordinate reads the same scalar through either indexing API. -/
theorem getAtOrZero_eq_getScalar {α : Type} [Storage α] [Zero α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) :
    getAtOrZero tensor [index.val] = Tensor.getScalar tensor index := by
  simp [getAtOrZero_dim_cons, index.isLt, Tensor.getScalar]

/-- Valid matrix coordinates read the same scalar through either indexing API. -/
theorem getAtOrZero_eq_get2 {α : Type} [Storage α] [Zero α]
    {m n : Nat} (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) :
    getAtOrZero tensor [row.val, column.val] = get2 tensor row column := by
  rw [getAtOrZero_dim_cons, dite_eq_left row.isLt, getAtOrZero_eq_getScalar]
  rfl

end Tensor

namespace Tensor

/-- Map an output coordinate of an axis slice back to its source coordinate. -/
def sliceAxisRangeCoordinate :
    {shape : Shape} → (dim : Fin shape.rank) →
      (start count : Nat) → (start + count ≤ Shape.axisSize shape dim.val (h := ⟨dim.isLt⟩)) →
      Shape.Coord (shape.set dim.val count) → shape.Coord
  | .dim length _, ⟨0, _⟩, start, count, hRange, coordinate =>
      (⟨start + coordinate.1.val,
          Nat.lt_of_lt_of_le
            (Nat.add_lt_add_left coordinate.1.isLt start) hRange⟩,
        coordinate.2)
  | .dim _ rest, ⟨axis + 1, h⟩, start, count, hRange, coordinate =>
      (coordinate.1,
        sliceAxisRangeCoordinate (shape := rest) ⟨axis, by simp only [Shape.rank_cons] at h; omega⟩
          start count hRange coordinate.2)

/-! ## Pointwise operations and predicates -/

/-- Mapping over a scalar tensor applies the function to its value. -/
@[simp] theorem map_scalar {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    (f : α → β) (value : α) :
    map f (scalar value) = scalar (f value) := by
  apply Tensor.Rep.ext
  intro coordinate
  simp only [scalar, Tensor.map_apply,
    Tensor.Rep.get_ofFn]

/-- The item of a mapped scalar tensor is the function applied to the item. -/
@[simp] theorem item_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    (f : α → β) (tensor : Tensor α .scalar) :
    (map f tensor).item = f tensor.item := by
  simp [item, Tensor.map_apply]

/-- Scalar lookup after a map is the function applied to the lookup. -/
@[simp] theorem getScalar_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {n : Nat} (f : α → β) (tensor : Tensor α [n]) (index : Fin n) :
    getScalar (map f tensor) index = f (getScalar tensor index) := by
  change (unstack (map f tensor) index).item = f (unstack tensor index).item
  rw [unstack_map, item_map]

/-- Every scalar entry satisfies `predicate`. -/
def Forall {α : Type} [TorchLean.Storage α] (predicate : α → Prop) :
    {shape : Shape} → Tensor α shape → Prop
  | .scalar, tensor => predicate tensor.item
  | .dim n shape, tensor =>
      ∀ index : Fin n, Forall predicate (shape := shape) (unstack tensor index)

/-- Corresponding scalar entries satisfy `relation`. -/
def Forall₂ {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    (relation : α → β → Prop) :
    {shape : Shape} → Tensor α shape → Tensor β shape → Prop
  | .scalar, left, right => relation left.item right.item
  | .dim n shape, left, right =>
      ∀ index : Fin n,
        Forall₂ relation (shape := shape) (unstack left index) (unstack right index)

/-- An entrywise predicate holds exactly when it holds at every tensor coordinate. -/
theorem forall_iff {α : Type} [TorchLean.Storage α] {shape : Shape}
    {predicate : α → Prop} {tensor : Tensor α shape} :
    Forall predicate tensor ↔ ∀ coordinate : shape.Coord, predicate (tensor coordinate) := by
  induction shape with
  | scalar =>
      constructor
      · intro h coordinate; cases coordinate; exact h
      · intro h; exact h PUnit.unit
  | dim n shape ih =>
      change (∀ i : Fin n, Forall predicate (unstack tensor i)) ↔ _
      simp only [ih, Tensor.unstack_apply]
      constructor
      · intro h coordinate; exact h coordinate.1 coordinate.2
      · intro h i j; exact h (i, j)

/-- An entrywise relation holds exactly when it holds at every shared tensor coordinate. -/
theorem forall₂_iff {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β] {shape : Shape}
    {relation : α → β → Prop} {left : Tensor α shape} {right : Tensor β shape} :
    Forall₂ relation left right ↔
      ∀ coordinate : shape.Coord, relation (left coordinate) (right coordinate) := by
  induction shape with
  | scalar =>
      constructor
      · intro h coordinate; cases coordinate; exact h
      · intro h; exact h PUnit.unit
  | dim n shape ih =>
      change (∀ i : Fin n, Forall₂ relation (unstack left i) (unstack right i)) ↔ _
      simp only [ih, Tensor.unstack_apply]
      constructor
      · intro h coordinate; exact h coordinate.1 coordinate.2
      · intro h i j; exact h (i, j)

/-- `Forall` on a scalar tensor is the predicate on its value. -/
@[simp] theorem forall_scalar {α : Type} [TorchLean.Storage α]
    {predicate : α → Prop} {value : α} :
    Forall predicate (scalar value) ↔ predicate value := by
  simp [Forall]

/-- `Forall` on a stacked tensor is `Forall` on every slice. -/
@[simp] theorem forall_dim {α : Type} [TorchLean.Storage α]
    {predicate : α → Prop} {n : Nat} {shape : Shape}
    {values : Fin n → Tensor α shape} :
    Forall predicate (dim values) ↔
      ∀ index, Forall predicate (values index) := by
  simp [Forall]

/-- `Forall₂` on scalar tensors is the relation on the two values. -/
@[simp] theorem forall₂_scalar {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {relation : α → β → Prop} {left : α} {right : β} :
    Forall₂ relation (scalar left) (scalar right) ↔ relation left right := by
  simp [Forall₂]

/-- `Forall₂` on stacked tensors is `Forall₂` slicewise. -/
@[simp] theorem forall₂_dim {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {relation : α → β → Prop} {n : Nat} {shape : Shape}
    {left : Fin n → Tensor α shape} {right : Fin n → Tensor β shape} :
    Forall₂ relation (dim left) (dim right) ↔
      ∀ index, Forall₂ relation (left index) (right index) := by
  simp [Forall₂]

/-- The trivial predicate holds of every tensor, at every shape. -/
theorem forall_true {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    Forall (fun _ => True) tensor := by
  exact forall_iff.mpr fun _ => trivial

/-- `Forall` transports along `map` when the function respects the two predicates.

A bound proved entrywise for the input carries through any elementwise function, such as an
activation, that maps the input bound into the output bound. -/
theorem forall_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {predicate : α → Prop} {resultPredicate : β → Prop}
    {f : α → β} {shape : Shape} {tensor : Tensor α shape}
    (hTensor : Forall predicate tensor)
    (hMap : ∀ value, predicate value → resultPredicate (f value)) :
    Forall resultPredicate (map f tensor) := by
  rw [forall_iff] at hTensor ⊢
  intro coordinate
  simpa only [Tensor.map_apply] using
    hMap (tensor coordinate) (hTensor coordinate)

/-- `Forall₂` transports along `map` on both sides when the functions respect the two relations. -/
theorem forall₂_map {α β γ δ : Type}
    [TorchLean.Storage α] [TorchLean.Storage β] [TorchLean.Storage γ] [TorchLean.Storage δ]
    {relation : α → β → Prop} {resultRelation : γ → δ → Prop} {f : α → γ} {g : β → δ}
    {shape : Shape} {left : Tensor α shape} {right : Tensor β shape}
    (hTensors : Forall₂ relation left right)
    (hMap : ∀ a b, relation a b → resultRelation (f a) (g b)) :
    Forall₂ resultRelation (map f left) (map g right) := by
  rw [forall₂_iff] at hTensors ⊢
  intro coordinate
  simpa only [Tensor.map_apply] using
    hMap (left coordinate) (right coordinate) (hTensors coordinate)

/-- Every mapped entry is related to its source entry when `f` relates each admissible value
to itself. -/
theorem forall₂_map_left {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {predicate : α → Prop} {relation : β → α → Prop} {f : α → β} {shape : Shape}
    {tensor : Tensor α shape} (hTensor : Forall predicate tensor)
    (hMap : ∀ value, predicate value → relation (f value) value) :
    Forall₂ relation (map f tensor) tensor := by
  rw [forall_iff] at hTensor
  rw [forall₂_iff]
  intro coordinate
  simpa only [Tensor.map_apply] using
    hMap (tensor coordinate) (hTensor coordinate)

/-- Mapping `g` and then `f` returns the tensor when `f` undoes `g` on every entry. -/
theorem map_map_eq_self_of_forall {α β : Type} [TorchLean.Storage α] [TorchLean.Storage β]
    {predicate : α → Prop} {f : β → α} {g : α → β} {shape : Shape} {tensor : Tensor α shape}
    (hTensor : Forall predicate tensor)
    (hInverse : ∀ value, predicate value → f (g value) = value) :
    map f (map g tensor) = tensor := by
  rw [forall_iff] at hTensor
  apply Tensor.Rep.ext
  intro coordinate
  simpa only [Tensor.map_apply] using
    hInverse (tensor coordinate) (hTensor coordinate)

/-- The product of the entries of a vector, a left fold of `*` from `1` in index order. -/
def prod {α : Type} [TorchLean.Storage α] [Mul α] [One α]
    {n : Nat} (tensor : Tensor α [n]) : α :=
  tensor.foldl (· * ·) 1

end Tensor

namespace Tensor

/-- Render a tensor recursively using the scalar `ToString` instance. -/
def pretty {α : Type} [Storage α] [ToString α] :
    {shape : Shape} → Tensor α shape → String
  | .scalar, tensor => toString tensor.item
  | .dim n shape, tensor =>
      "[" ++ String.intercalate ", "
        (List.ofFn fun index : Fin n =>
          pretty (shape := shape) (Tensor.unstack tensor index)) ++ "]"

end Tensor

end TorchLean
