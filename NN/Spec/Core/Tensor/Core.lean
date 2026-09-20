/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Coordinate
public import NN.Tensor.Internal.Representation.Basic.Reindex

/-!
# Core tensor datatype (`TorchLean.Tensor`)

TorchLean has one shape-indexed tensor representation for proofs and execution:
`Tensor α shape`.

Every tensor owns one contiguous row-major buffer. The scalar type selects the
physical buffer through `Storage`: arbitrary proof scalar types such as
`Real` use `Array`, while executable types with specialized instances use
packed storage (`FloatArray` for `Float`, `ByteArray` for `UInt8`).

The `scalar` and `dim` functions preserve TorchLean's proof-facing construction
API. They are smart constructors over packed storage, not physical recursive
constructors. Proofs eliminate tensors through typed lookup, `scalarEquiv`,
`dimEquiv`, and extensionality.
-/

@[expose] public section

namespace TorchLean

/--
The single tensor representation used by TorchLean proofs and execution.

The public shape is `Spec.Shape`, which is the same type as the shape carried by the buffer, so
this alias only reorders arguments. The implementation is one certified contiguous row-major buffer
selected by `Storage`.
-/
abbrev Tensor (α : Type) (shape : Spec.Shape) [Storage α] :=
  Tensor.Internal.Rep α shape

end TorchLean

open Lean PrettyPrinter Delaborator SubExpr

/-- Recognize a statically explicit dimension list in an internal tensor type. -/
private meta partial def explicitTensorShape : Expr → Bool
  | shape =>
      if shape.isAppOfArity ``List.nil 1 then
        true
      else if shape.isAppOfArity ``List.cons 3 then
        explicitTensorShape shape.getAppArgs[2]!
      else
        false

/--
Render executable internal tensors with explicit dimensions through the
canonical public type name in editor information and `#check` output.

Universe-polymorphic proof tensors and genuinely computed shape expressions
retain their exact internal spelling.
-/
@[app_delab TorchLean.Tensor.Internal.Rep]
meta def TorchLean.delabTensorRep : Delab := do
  withOverApp 3 do
    let expression ← getExpr
    let arguments := expression.getAppArgs
    guard <| arguments.size = 3
    guard <| (← Meta.getDecLevel arguments[0]!) == .zero
    guard <| explicitTensorShape arguments[1]!
    let scalar ← withNaryArg 0 delab
    let shape ← withNaryArg 1 delab
    pure <| Syntax.mkApp (mkIdent ``TorchLean.Tensor) #[scalar, shape]

open Spec TorchLean

namespace TorchLean.Tensor

/-- Construct a rank-zero tensor from one scalar value. -/
def scalar {α : Type} [TorchLean.Storage α] (value : α) :
    Tensor α .scalar :=
  TorchLean.Tensor.Internal.Rep.ofFn fun _ => value

/-- Construct an outer dimension from its shape-indexed entries. -/
def dim {α : Type} [TorchLean.Storage α] {n : Nat} {shape : Shape}
    (values : Fin n → Tensor α shape) : Tensor α (.dim n shape) :=
  TorchLean.Tensor.Internal.Rep.stack values

/-- Return the value stored in a scalar tensor. -/
def item {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α .scalar) : α :=
  tensor PUnit.unit

/-- Reading back the value a scalar tensor was built from returns that value. -/
@[simp] theorem item_scalar {α : Type} [TorchLean.Storage α] (value : α) :
    (scalar value).item = value := by
  exact TorchLean.Tensor.Internal.Rep.get_ofFn (fun _ => value) PUnit.unit

/-- A scalar tensor evaluates to its sole scalar value. -/
@[simp] theorem scalar_apply {α : Type} [TorchLean.Storage α] (value : α) :
    (scalar value : Tensor α .scalar) PUnit.unit = value := by
  simpa [item] using item_scalar value

/-- Select one leading-axis slice from a tensor. -/
def unstack {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape))
    (index : Fin n) : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.unstack tensor index

/-- Tensors with equal row-major data arrays are equal.

The packed representation is a storage buffer together with a size proof, and buffers are
determined by their array observation (`Storage.toArray_injective`). -/
theorem eq_of_data_eq {α : Type} [TorchLean.Storage α] {s : Shape}
    {a b : TorchLean.Tensor α s} (h : a.data = b.data) : a = b := by
  cases a with
  | mk aBuf _ =>
    cases b with
    | mk bBuf _ =>
      have hBuf : aBuf = bBuf := TorchLean.Storage.toArray_injective h
      subst hBuf
      rfl

/-- Slicing the outer axis of a stacked tensor returns the slice that was stacked. -/
@[simp] theorem unstack_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (values : Fin n → Tensor α shape)
    (index : Fin n) :
    unstack (dim values) index = values index := by
  exact TorchLean.Tensor.Internal.Rep.unstack_stack values index

/-- Restacking every slice of a tensor returns the tensor.

Together with `unstack_dim` this is what makes `dim` and `unstack` an isomorphism rather than merely
a pair of functions, which is why proofs can eliminate a tensor by `rw [← dim_unstack]` and then
reason slice by slice. -/
@[simp] theorem dim_unstack {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape)) :
    dim (unstack tensor) = tensor := by
  exact TorchLean.Tensor.Internal.Rep.stack_unstack tensor

/-- Default tensor value for any shape. -/
def default {α : Type} [TorchLean.Storage α] [Inhabited α]
    {shape : Shape} : Tensor α shape :=
  TorchLean.Tensor.Internal.Rep.const Inhabited.default

/-- Every shape is inhabited when the scalar type is: take the all-default tensor. -/
@[reducible, instance] def inhabited {α : Type}
    [TorchLean.Storage α] [Inhabited α] {shape : Shape} :
    Inhabited (Tensor α shape) :=
  ⟨Tensor.default⟩

/-- Rebuilding a scalar tensor from its item returns the original tensor. -/
@[simp] theorem scalar_item {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α .scalar) :
    scalar tensor.item = tensor := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  cases coordinate
  exact TorchLean.Tensor.Internal.Rep.get_ofFn (fun _ => tensor.item) PUnit.unit

/-- Two scalar tensors agreeing on their item are equal. -/
@[ext] theorem ext_scalar {α : Type} [TorchLean.Storage α]
    {left right : Tensor α .scalar}
    (h : left.item = right.item) : left = right := by
  apply TorchLean.Tensor.Internal.Rep.ext
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

/--
Proof-facing eliminator for the familiar scalar/dimension cases.

`View` does not own tensor data and is not a second tensor representation. It
only exposes the outer shape of an already packed tensor.
-/
inductive View (α : Type) [TorchLean.Storage α] : Shape → Type
  | scalar (value : α) : View α .scalar
  | dim {n : Nat} {shape : Shape}
      (values : Fin n → Tensor α shape) : View α (.dim n shape)

/-- Observe the outer constructor of a packed tensor for shape-recursive proofs. -/
def view {α : Type} [TorchLean.Storage α] :
    {shape : Shape} → Tensor α shape → View α shape
  | .scalar, tensor => .scalar tensor.item
  | .dim _ _, tensor => .dim (unstack tensor)

/-!
## Pointwise algebra

Every algebraic operation on tensors is the corresponding scalar operation applied coordinatewise.
The instances (`Zero`, `Add`, `Neg`, `Sub`, `SMul`, `AddCommMonoid`, `AddCommGroup`, `Module`) live
on the native representation in `NN.Tensor.Internal.Representation.Basic.Pointwise`, so they apply
uniformly to `Tensor α shape` whether the shape is a variable or a literal such as `[n]`. The
coordinate lemmas below are the simp normal forms; `add_apply` is restated here under the public
tensor namespace.
-/

/-- Tensor addition is coordinatewise scalar addition. -/
@[simp] theorem add_apply {α : Type} [TorchLean.Storage α] [Add α] {shape : Shape}
    (x y : Tensor α shape) (i : shape.Coord) : (x + y) i = x i + y i :=
  TorchLean.Tensor.Internal.Rep.zipWith_apply (· + ·) x y i

/-- Vectors and finite scalar functions are equivalent. -/
def vectorEquiv {α : Type} [TorchLean.Storage α]
    (n : Nat) : Tensor α [n] ≃ (Fin n → α) :=
  (dimEquiv n .scalar).trans (Equiv.piCongrRight fun _ => scalarEquiv α)

/-- Cast a tensor along an equality of shapes. -/
def castShape {α : Type} [TorchLean.Storage α] {source target : Shape}
    (tensor : Tensor α source) (h : source = target) : Tensor α target := by
  cases h
  exact tensor

/-- Casting along `rfl` is the identity, and holds by `rfl` itself. -/
@[simp] theorem cast_shape_rfl {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) :
    castShape tensor rfl = tensor :=
  rfl

/-- Casting along any proof of `shape = shape` is the identity.

`cast_shape_rfl` does not cover this: after `Shape.ofList` normalization the proof in hand is often
some derived term rather than the literal `rfl`, and simp needs to discharge those too. -/
@[simp] theorem cast_shape_self {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α shape) (h : shape = shape) :
    castShape tensor h = tensor := by
  cases h
  rfl

/-- Two casts in a row collapse into one along the composed equality. -/
@[simp] theorem cast_shape_trans {α : Type} [TorchLean.Storage α]
    {s₁ s₂ s₃ : Shape} (tensor : Tensor α s₁)
    (h₁₂ : s₁ = s₂) (h₂₃ : s₂ = s₃) :
    castShape (castShape tensor h₁₂) h₂₃ =
      castShape tensor (h₁₂.trans h₂₃) := by
  cases h₁₂
  cases h₂₃
  rfl

/-- The cast does not depend on which proof of the shape equality is used.

Shape equalities are proofs in a subsingleton, so this is provable rather than an axiom, and it is
what lets two developments that derived the same equality differently share a lemma. -/
theorem cast_shape_proof_irrel {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    {p q : source = target} :
    castShape tensor p = castShape tensor q := by
  cases Subsingleton.elim p q
  rfl

/-- The `▸` rewrite and `castShape` are the same function.

Lean inserts `▸` on its own when a shape is rewritten in a tactic block, so without this bridge the
`castShape` simp set would silently fail to fire on goals the elaborator produced. -/
theorem eqRec_eq_cast_shape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (h : source = target) :
    (h ▸ tensor) = castShape tensor h := by
  cases h
  rfl

/-- `cast_shape_proof_irrel` in `▸` form, for goals the elaborator produced. -/
theorem eqRec_proof_irrel {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    {p q : source = target} :
    (p ▸ tensor) = (q ▸ tensor) := by
  cases Subsingleton.elim p q
  rfl

attribute [grind =] cast_shape_rfl cast_shape_self cast_shape_trans eqRec_eq_cast_shape

end TorchLean.Tensor

namespace Spec

/-- Recover the statically known shape from a tensor value. -/
def shapeOf {α : Type} [TorchLean.Storage α]
    {shape : Shape} (_ : Tensor α shape) : Shape :=
  shape

/-! ## Indexing -/

/-- Try to read a scalar using a runtime list of coordinates. -/
def getSpec {α : Type} [TorchLean.Storage α] {shape : Shape}
    (tensor : Tensor α shape) (indices : List Nat) : Option α :=
  (Shape.Coord.ofList? shape indices).map tensor

/-- The empty coordinate list reads the item of a rank-zero tensor. -/
@[simp] theorem get_spec_scalar_nil {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α (Shape.ofList [])) :
    getSpec tensor [] = some tensor.item := by
  simp [getSpec, Shape.Coord.ofList?, Tensor.item]

/-- A rank-zero tensor has no axis to index, so any nonempty coordinate list fails. -/
@[simp] theorem get_spec_scalar_cons {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α (Shape.ofList [])) (index : Nat) (indices : List Nat) :
    getSpec tensor (index :: indices) = none := by
  simp [getSpec, Shape.Coord.ofList?]

/-- A tensor with an axis is not a scalar, so the empty coordinate list fails. -/
@[simp] theorem get_spec_dim_nil {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape)) :
    getSpec tensor [] = none := by
  simp [getSpec, Shape.Coord.ofList?]

/-- One step of runtime lookup: check the leading index against the axis, then recurse into the
slice. This is the equation that turns `getSpec` on a literal coordinate list into a chain of
`unstack`s, which is how the executable and proof-facing readings are kept in step. -/
@[simp] theorem get_spec_dim_cons {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape))
    (index : Nat) (indices : List Nat) :
    getSpec tensor (index :: indices) =
      if h : index < n then getSpec (Tensor.unstack tensor ⟨index, h⟩) indices else none := by
  by_cases hIndex : index < n
  · cases hCoordinate : Shape.Coord.ofList? shape indices with
    | none =>
        simp [getSpec, Shape.Coord.ofList?, hIndex, hCoordinate]
    | some coordinate =>
        simp [getSpec, Shape.Coord.ofList?, Tensor.unstack, hIndex, hCoordinate]
  · simp [getSpec, Shape.Coord.ofList?, hIndex]

/-- A shape cast does not move any data, so runtime lookup sees straight through it. -/
@[simp] theorem get_spec_castShape {α : Type} [TorchLean.Storage α]
    {source target : Shape} (tensor : Tensor α source)
    (h : source = target) (index : List Nat) :
    getSpec (Tensor.castShape tensor h) index = getSpec tensor index := by
  cases h
  rfl

attribute [grind =] get_spec_scalar_nil get_spec_scalar_cons get_spec_dim_nil get_spec_dim_cons

end Spec

namespace TorchLean.Tensor

/-- Select one coordinate along an arbitrary axis and remove that axis. -/
def selectSpec {α : Type} [TorchLean.Storage α] :
    (axis : Nat) → {shape : Shape} → (tensor : Tensor α shape) →
      [_h : Shape.AxisInBounds axis shape] →
      Fin (Shape.axisSize shape axis) → Tensor α (shape.eraseAxis axis)
  | 0, .dim _ _, tensor, _, index => unstack tensor index
  | axis + 1, .dim _ rest, tensor, h, index =>
      dim fun outer =>
        @selectSpec α _ axis rest (unstack tensor outer)
          ⟨by
            have := h.proof
            simp only [Shape.rank] at this
            grind⟩
          (Fin.cast (by rfl) index)

/-- Selecting along axis zero is exactly the outer-axis slice. -/
@[simp] theorem selectSpec_zero_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (values : Fin n → Tensor α shape)
    (index : Fin n) :
    selectSpec 0 (dim values) index = values index := by
  exact unstack_dim values index

end TorchLean.Tensor

namespace Spec

/-- Select one entry from the outermost tensor axis. -/
def get {α : Type} [TorchLean.Storage α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (.dim n shape)) (index : Fin n) : Tensor α shape :=
  Tensor.unstack tensor index

/-- Indexing a stacked tensor returns the entry that was stacked. -/
@[simp] theorem get_dim {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (values : Fin n → Tensor α shape)
    (index : Fin n) :
    get (Tensor.dim values) index = values index :=
  Tensor.unstack_dim values index

/--
Vector indexing returns a scalar directly.

This higher-priority instance also makes chained indexing natural:
`matrix[row][column]` returns an entry, while the first lookup still returns
the statically shaped row tensor.
-/
instance (priority := high) {α : Type} [TorchLean.Storage α] {n : Nat} :
    GetElem (TorchLean.Tensor.Internal.Rep α [n]) (Fin n) α
      (fun _ _ => True) where
  getElem tensor index _ := tensor (index, PUnit.unit)

instance {α : Type} [TorchLean.Storage α] {n : Nat} {shape : List Nat} :
    GetElem (TorchLean.Tensor.Internal.Rep α (n :: shape)) (Fin n)
      (Tensor α (Shape.ofList shape)) (fun _ _ => True) where
  getElem tensor index _ :=
    TorchLean.Tensor.Internal.Rep.unstack (s := shape) tensor index

/--
Natural-number indexing of a vector returns one scalar and asks `GetElem` to
discharge the static bound.
-/
instance (priority := high) {α : Type} [TorchLean.Storage α] {n : Nat} :
    GetElem (TorchLean.Tensor.Internal.Rep α [n]) Nat α
      (fun _ index => index < n) where
  getElem tensor index hIndex :=
    tensor (⟨index, hIndex⟩, PUnit.unit)

/--
Natural-number indexing selects one slice along the outermost tensor axis.

Literal indices are checked during elaboration, so ordinary code can write
`matrix[0]` without constructing a `Fin` value explicitly.
-/
instance {α : Type} [TorchLean.Storage α] {n : Nat} {shape : List Nat} :
    GetElem (TorchLean.Tensor.Internal.Rep α (n :: shape)) Nat
      (Tensor α (Shape.ofList shape)) (fun _ index => index < n) where
  getElem tensor index hIndex :=
    TorchLean.Tensor.Internal.Rep.unstack (s := shape) tensor ⟨index, hIndex⟩

end Spec

namespace TorchLean.Tensor

/-- Runtime-list lookup is extensional. -/
@[ext] theorem ext_getSpec {α : Type} [TorchLean.Storage α]
    {shape : Shape} {left right : Tensor α shape}
    (h : ∀ index, getSpec left index = getSpec right index) :
    left = right := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simpa [getSpec] using h (Shape.Coord.toList shape coordinate)

/-- Extract one scalar from a vector. -/
def getScalar {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) : α :=
  item (get tensor index)

/-- Vector scalar lookup is ordinary coordinate evaluation. -/
theorem getScalar_eq_apply {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) :
    getScalar tensor index = tensor (index, PUnit.unit) := by
  unfold getScalar Spec.get unstack item
  exact TorchLean.Tensor.Internal.Rep.unstack_apply tensor index PUnit.unit

/-- Direct storage read used when compiling vector scalar lookup. -/
@[inline] def Internal.getScalarDirect {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α [n]) (index : Fin n) : α :=
  tensor (index, PUnit.unit)

/-- Compile scalar lookup without allocating the intermediate rank-zero slice. -/
@[csimp] theorem getScalar_eq_getScalarDirect :
    @getScalar = @Internal.getScalarDirect := by
  funext α storage n tensor index
  exact getScalar_eq_apply tensor index

/-- Reading a vector filled by the packed constant constructor returns that constant. -/
@[simp] theorem getScalar_const {α : Type} [TorchLean.Storage α]
    {n : Nat} (value : α) (index : Fin n) :
    getScalar
        (TorchLean.Tensor.Internal.Rep.const value : Tensor α [n])
        index =
      value := by
  rw [getScalar_eq_apply]
  exact TorchLean.Tensor.Internal.Rep.const_apply
    (s := [n]) value (index, PUnit.unit)

/-- Scalar lookup on a rank-one literal returns the corresponding source entry. -/
@[simp] theorem getScalar_ofList {α : Type} [TorchLean.Storage α]
    (values : List α) (index : Fin values.length) :
    getScalar (TorchLean.Tensor.Internal.Rep.ofList values) index =
      values.get index := by
  rw [getScalar_eq_apply]
  exact TorchLean.Tensor.Internal.Rep.ofList_apply values index

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
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  rcases coordinate with ⟨index, ⟨⟩⟩
  have hIndex := h index
  unfold getScalar Spec.get unstack item at hIndex
  convert hIndex using 1
  · exact (TorchLean.Tensor.Internal.Rep.unstack_apply left index PUnit.unit).symm
  · exact (TorchLean.Tensor.Internal.Rep.unstack_apply right index PUnit.unit).symm

end TorchLean.Tensor

namespace Spec

/-- Matrix element access. -/
def get2 {α : Type} [TorchLean.Storage α] {m n : Nat}
    (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) : α :=
  Tensor.getScalar (get tensor row) column

/-- Matrix access is a row slice followed by a scalar read.

True by `rfl`, but worth a name: it is the rewrite that lets a matrix proof reuse every vector lemma
about `getScalar` instead of duplicating them at rank two. -/
theorem get2_eq_getScalar_get {α : Type} [TorchLean.Storage α]
    {m n : Nat} (tensor : Tensor α [m, n])
    (row : Fin m) (column : Fin n) :
    get2 tensor row column = Tensor.getScalar (get tensor row) column :=
  rfl

/-- Matrix scalar lookup is one read at the corresponding two-axis coordinate. -/
theorem get2_eq_apply {α : Type} [TorchLean.Storage α]
    {m n : Nat} (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) :
    get2 tensor row column = tensor (row, column, PUnit.unit) := by
  rw [get2, Tensor.getScalar_eq_apply]
  exact TorchLean.Tensor.Internal.Rep.unstack_apply tensor row (column, PUnit.unit)

/-- Direct storage read used when compiling matrix scalar lookup. -/
@[inline] def Internal.get2Direct {α : Type} [TorchLean.Storage α]
    {m n : Nat} (tensor : Tensor α [m, n]) (row : Fin m) (column : Fin n) : α :=
  tensor (row, column, PUnit.unit)

/-- Compile matrix lookup without allocating an intermediate row or scalar tensor. -/
@[csimp] theorem get2_eq_get2Direct : @get2 = @Internal.get2Direct := by
  funext α storage m n tensor row column
  exact get2_eq_apply tensor row column

/-- Reading back a matrix built from a two-argument function returns the function. -/
@[simp] theorem get2_dim {α : Type} [TorchLean.Storage α]
    {m n : Nat} (values : Fin m → Fin n → α)
    (row : Fin m) (column : Fin n) :
    get2 (Tensor.dim fun i => Tensor.dim fun j => Tensor.scalar (values i j))
      row column = values row column := by
  simp [get2]

instance {α : Type} [TorchLean.Storage α] {m n : Nat} :
    GetElem (Tensor α [m, n]) (Fin m × Fin n) α (fun _ _ => True) where
  getElem tensor index _ := get2 tensor index.1 index.2

/-! ## Total indexing and shape operations -/

/-- Return zero when a runtime coordinate list is invalid. -/
def getAtOrZero {α : Type} [TorchLean.Storage α] [Zero α]
    {shape : Shape} (tensor : Tensor α shape) (indices : List Nat) : α :=
  (getSpec tensor indices).getD 0

/-- On a rank-zero tensor the empty coordinate list reads the item, no fallback needed. -/
@[simp] theorem get_at_or_zero_scalar_nil {α : Type}
    [TorchLean.Storage α] [Zero α] (tensor : Tensor α (Shape.ofList [])) :
    getAtOrZero tensor [] = tensor.item := by
  simp [getAtOrZero]

/-- An over-long coordinate list on a rank-zero tensor falls back to zero. -/
@[simp] theorem get_at_or_zero_scalar_cons {α : Type}
    [TorchLean.Storage α] [Zero α] (tensor : Tensor α (Shape.ofList []))
    (index : Nat) (indices : List Nat) :
    getAtOrZero tensor (index :: indices) = 0 := by
  simp [getAtOrZero]

/-- A too-short coordinate list falls back to zero. -/
@[simp] theorem get_at_or_zero_dim_nil {α : Type}
    [TorchLean.Storage α] [Zero α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (.dim n shape)) :
    getAtOrZero tensor [] = 0 := by
  simp [getAtOrZero]

/-- One step of total lookup, with the out-of-range branch returning zero rather than failing.

The zero is what makes this function total, and it is also the reason the `Zero α` hypothesis is
there: no shape argument can rule out a bad runtime coordinate list. -/
@[simp] theorem get_at_or_zero_dim_cons {α : Type}
    [TorchLean.Storage α] [Zero α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (.dim n shape)) (index : Nat) (indices : List Nat) :
    getAtOrZero tensor (index :: indices) =
      if h : index < n then
        getAtOrZero (Tensor.unstack tensor ⟨index, h⟩) indices
      else
        0 := by
  by_cases hIndex : index < n
  · simp [getAtOrZero, hIndex]
  · simp [getAtOrZero, hIndex]

attribute [grind =] get_at_or_zero_scalar_nil get_at_or_zero_scalar_cons
  get_at_or_zero_dim_nil get_at_or_zero_dim_cons

/-- Cast a tensor along a shape equality. -/
def tensorCast {α : Type} [TorchLean.Storage α]
    {source : Shape} (target : Shape) (h : source = target) :
    Tensor α source → Tensor α target :=
  fun tensor => Tensor.castShape tensor h

/-- `tensorCast` is `castShape` with the target shape written first. -/
@[simp] theorem tensor_cast_eq_cast_shape {α : Type}
    [TorchLean.Storage α] {source target : Shape}
    (h : source = target) (tensor : Tensor α source) :
    tensorCast target h tensor = Tensor.castShape tensor h :=
  rfl

attribute [grind =] tensor_cast_eq_cast_shape

/-- Replicate a scalar tensor to any shape. -/
def replicate {α : Type} [TorchLean.Storage α] :
    {shape : Shape} → Tensor α .scalar → Tensor α shape
  | _shape, tensor => TorchLean.Tensor.Internal.Rep.const tensor.item

/-- Every coordinate of a replicated tensor holds the source item. -/
@[simp] theorem replicate_apply {α : Type} [TorchLean.Storage α]
    {shape : Shape} (tensor : Tensor α .scalar) (coordinate : shape.Coord) :
    replicate (shape := shape) tensor coordinate = tensor.item := by
  exact TorchLean.Tensor.Internal.Rep.const_apply tensor.item coordinate

/-- Replicating to rank zero returns the original scalar tensor. -/
@[simp] theorem replicate_scalar_shape {α : Type} [TorchLean.Storage α]
    (tensor : Tensor α .scalar) :
    replicate (shape := .scalar) tensor = tensor := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  cases coordinate
  rw [replicate_apply]
  rfl

/-- Every slice of a replicated tensor is the replication of the same item. -/
@[simp] theorem unstack_replicate {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α .scalar)
    (index : Fin n) :
    Tensor.unstack (replicate (shape := .dim n shape) tensor) index =
      replicate (shape := shape) tensor := by
  exact TorchLean.Tensor.Internal.Rep.unstack_const tensor.item index

/-- Scalar lookup into a replicated vector returns the source item. -/
@[simp] theorem getScalar_replicate {α : Type} [TorchLean.Storage α]
    {n : Nat} (tensor : Tensor α .scalar) (index : Fin n) :
    Tensor.getScalar (replicate (shape := [n]) tensor) index = tensor.item := by
  rw [Tensor.getScalar_eq_apply]
  exact replicate_apply (shape := [n]) tensor (index, PUnit.unit)

end Spec

namespace TorchLean.Tensor

/-- Map an output coordinate of an axis slice back to its source coordinate. -/
def sliceAxisRangeCoordinate :
    (axis : Nat) → {shape : Shape} →
      [_h : Shape.AxisInBounds axis shape] →
      (start count : Nat) → (start + count ≤ Shape.axisSize shape axis) →
      (shape.replaceAxis axis count).Coord → shape.Coord
  | 0, .dim length _, _, start, count, hRange, coordinate =>
      have hRange' : start + count ≤ length := by
        simpa only [Shape.axisSize_zero] using hRange
      (⟨start + coordinate.1.val,
          Nat.lt_of_lt_of_le
            (Nat.add_lt_add_left coordinate.1.isLt start) hRange'⟩,
        coordinate.2)
  | axis + 1, .dim _ rest, hAxis, start, count, hRange, coordinate =>
      let innerAxis : Shape.AxisInBounds axis rest :=
        ⟨by
          have := hAxis.proof
          simp only [Shape.rank] at this
          grind⟩
      letI := innerAxis
      have hRange' : start + count ≤ Shape.axisSize rest axis := by
        simpa only [Shape.axisSize_succ] using hRange
      (coordinate.1,
        @sliceAxisRangeCoordinate axis rest innerAxis
          start count hRange' coordinate.2)

/--
Keep a contiguous coordinate range along an arbitrary axis.

The coordinate recursion above runs once per output element. The tensor itself is materialized in
one packed pass, rather than recursively constructing and then restacking every outer slice.
-/
def sliceAxisRangeSpec {α : Type} [TorchLean.Storage α]
    (axis : Nat) {shape : Shape} (tensor : Tensor α shape)
    [_h : Shape.AxisInBounds axis shape]
    (start count : Nat) (hRange : start + count ≤ Shape.axisSize shape axis) :
    Tensor α (shape.replaceAxis axis count) :=
  TorchLean.Tensor.Internal.Rep.pull
    (sliceAxisRangeCoordinate axis start count hRange) tensor

/-- An axis slice reads the source coordinate selected by its checked range map. -/
@[simp] theorem sliceAxisRangeSpec_apply {α : Type} [TorchLean.Storage α]
    (axis : Nat) {shape : Shape} (tensor : Tensor α shape)
    [_h : Shape.AxisInBounds axis shape]
    (start count : Nat) (hRange : start + count ≤ Shape.axisSize shape axis)
    (coordinate : (shape.replaceAxis axis count).Coord) :
    sliceAxisRangeSpec axis tensor start count hRange coordinate =
      tensor (sliceAxisRangeCoordinate axis start count hRange coordinate) := by
  exact TorchLean.Tensor.Internal.Rep.pull_apply
    (sliceAxisRangeCoordinate axis start count hRange) tensor coordinate

end TorchLean.Tensor

namespace Spec

/-- Slice a contiguous range along the first axis. -/
def sliceRangeSpec {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape))
    (start length : Nat) (h : start + length ≤ n) :
    Tensor α (.dim length shape) :=
  Tensor.sliceAxisRangeSpec 0 tensor start length (by simpa using h)

/-- The first index of a nonempty axis. -/
def finZero {n : Nat} (h : 0 < n) : Fin n :=
  ⟨0, h⟩

/-- First slice along the outer axis, or `none` when that axis is empty.

The `Option` is what an empty axis costs: `n` is a variable here, so no shape argument can promise
there is a first slice, and returning `none` keeps the function total. -/
def getHead {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape)) :
    Option (Tensor α shape) :=
  if h : 0 < n then some (get tensor (finZero h)) else none

/-- Everything after the first slice, or `none` when the outer axis is empty. -/
def getTail {α : Type} [TorchLean.Storage α]
    {n : Nat} {shape : Shape} (tensor : Tensor α (.dim n shape)) :
    Option (Tensor α (.dim (n - 1) shape)) :=
  if h : 0 < n then
    some (Tensor.sliceAxisRangeSpec 0 tensor 1 (n - 1) (by
      simp only [Shape.axisSize_zero]
      grind))
  else
    none

/-! ## Pointwise operations and predicates -/

end Spec

namespace TorchLean.Tensor

/-- Apply a scalar function pointwise while preserving the shape. -/
def map {α β : Type} [TorchLean.Storage α]
    [TorchLean.Storage β] {shape : Shape}
    (f : α → β) (tensor : Tensor α shape) : Tensor β shape :=
  TorchLean.Tensor.Internal.Rep.map f tensor

/-- Mapping over a scalar tensor applies the function to its value. -/
@[simp] theorem map_scalar {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    (f : α → β) (value : α) :
    map f (scalar value) = scalar (f value) := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  simp only [map, scalar, TorchLean.Tensor.Internal.Rep.map_apply,
    TorchLean.Tensor.Internal.Rep.get_ofFn]

/-- Mapping commutes with stacking, so a map pushes into every slice. -/
@[simp] theorem map_dim {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {n : Nat} {shape : Shape} (f : α → β)
    (values : Fin n → Tensor α shape) :
    map f (dim values) = dim (fun index => map f (values index)) := by
  exact TorchLean.Tensor.Internal.Rep.map_stack f values

/-- The item of a mapped scalar tensor is the function applied to the item. -/
@[simp] theorem item_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    (f : α → β) (tensor : Tensor α .scalar) :
    (map f tensor).item = f tensor.item := by
  simp [item, map, TorchLean.Tensor.Internal.Rep.map_apply]

/-- Slicing after a map is mapping after a slice. -/
@[simp] theorem unstack_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {n : Nat} {shape : Shape} (f : α → β)
    (tensor : Tensor α (.dim n shape)) (index : Fin n) :
    unstack (map f tensor) index = map f (unstack tensor index) := by
  exact (TorchLean.Tensor.Internal.Rep.map_unstack f tensor index).symm

/-- Scalar lookup after a map is the function applied to the lookup. -/
@[simp] theorem getScalar_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {n : Nat} (f : α → β) (tensor : Tensor α [n]) (index : Fin n) :
    getScalar (map f tensor) index = f (getScalar tensor index) := by
  rw [← dim_unstack tensor, map_dim]
  simp only [getScalar_dim_entry]
  rw [← scalar_item (unstack tensor index), map_scalar]
  rw [item_scalar]
  exact congrArg f (item_scalar (unstack tensor index).item).symm

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
  induction shape with
  | scalar => trivial
  | dim _ _ inductionHypothesis =>
      intro index
      exact inductionHypothesis (unstack tensor index)

/-- `Forall` transports along `map` when the function respects the two predicates.

This is the workhorse for the numeric layers: a bound proved entrywise for the input survives an
elementwise activation as long as the activation maps the input bound into the output one. -/
theorem forall_map {α β : Type}
    [TorchLean.Storage α] [TorchLean.Storage β]
    {predicate : α → Prop} {resultPredicate : β → Prop}
    {f : α → β} {shape : Shape} {tensor : Tensor α shape}
    (hTensor : Forall predicate tensor)
    (hMap : ∀ value, predicate value → resultPredicate (f value)) :
    Forall resultPredicate (map f tensor) := by
  induction shape with
  | scalar =>
      change predicate tensor.item at hTensor
      rw [← scalar_item tensor, map_scalar]
      simpa only [forall_scalar] using hMap tensor.item hTensor
  | dim _ _ inductionHypothesis =>
      have hTensor' :
          ∀ index, Forall predicate (unstack tensor index) :=
        hTensor
      rw [← dim_unstack tensor, map_dim]
      apply forall_dim.mpr
      intro index
      exact inductionHypothesis (hTensor' index)

/-- A predicate true of one value is true entrywise of the tensor replicating it. -/
theorem forall_replicate {α : Type} [TorchLean.Storage α]
    {predicate : α → Prop} {shape : Shape} {value : α}
    (hValue : predicate value) :
    Forall predicate (Spec.replicate (shape := shape) (scalar value)) := by
  change Forall predicate
    (TorchLean.Tensor.Internal.Rep.const (scalar value).item)
  rw [item_scalar]
  induction shape with
  | scalar =>
      change predicate
        ((TorchLean.Tensor.Internal.Rep.const value : Tensor α .scalar)
          PUnit.unit)
      simpa using hValue
  | dim n shape inductionHypothesis =>
      intro index
      have hSlice :
          unstack
              (TorchLean.Tensor.Internal.Rep.const value :
                Tensor α (.dim n shape))
              index =
            (TorchLean.Tensor.Internal.Rep.const value : Tensor α shape) := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [unstack]
      rw [hSlice]
      exact inductionHypothesis

/-- Multiply all vector entries in row-major order. -/
def prod {α : Type} [TorchLean.Storage α] [Mul α] [One α]
    {n : Nat} (tensor : Tensor α [n]) : α :=
  tensor.foldl (· * ·) 1

end TorchLean.Tensor

namespace Spec

/-- Render a tensor recursively using the scalar `ToString` instance. -/
def pretty {α : Type} [TorchLean.Storage α] [ToString α] :
    {shape : Shape} → Tensor α shape → String
  | .scalar, tensor => toString tensor.item
  | .dim n shape, tensor =>
      "[" ++ String.intercalate ", "
        (List.ofFn fun index : Fin n =>
          pretty (shape := shape) (Tensor.unstack tensor index)) ++ "]"

end Spec
