/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Algebra.Module.Defs
public import TorchLean.Tensor.Rep

/-!
# Pointwise and family tensor operations

Scalar maps, binary maps, finite tensor families, stacking, and rank-one list
construction over native tensor storage.

The lemmas below describe the logical definitions of `full`, `map`, `zipWith`, `zipWith3`,
`dim`, and `unstack`. Each has a native fast path proved equal to it by a `csimp` theorem, and
compiled code runs that path.
-/

@[expose] public section

namespace TorchLean.Tensor

universe u v w x

/-- Evaluate a finite family once per index in ascending order and retain indexed access.

The element type needs no `Storage` instance, so the temporary buffer also holds dependent tensor
families such as packs. Public construction APIs return the tensors or packs built from it.
-/
def sequenceFinM {m : Type → Type} [Monad m] {α : Type} {n : Nat}
    (f : Fin n → m α) : m (Fin n → α) := do
  let values ← Vector.ofFnM f
  pure values.get

namespace Rep

open Shape (Coord)

/-- Fill a tensor of arbitrary shape with one value.

PyTorch analogy: `torch.full(shape, value)`.
-/
def _root_.TorchLean.Tensor.full {α : Type u} [Storage α] (shape : Shape) (value : α) :
    Tensor α shape :=
  ofFlatFn fun _ => value

/-- Native implementation of constant tensor construction. -/
def fullFast
    {α : Type u} [Storage α] (s : Shape) (a : α) : Rep α s :=
  ofFlatNativeFn
    (fun _ _ => a)
    (fun _ => a)
    (by intros; rfl)

/-- Compiled code runs `fullFast` in place of `full`. -/
@[csimp] theorem fullFast_eq : @full = @fullFast := by
  funext α _ s a
  exact (ofFlatNativeFn_eq_ofFlatFn _ _ _).symm

/-- Apply a scalar function pointwise while preserving the shape. -/
def _root_.TorchLean.Tensor.map {α : Type u} [Storage α]
    {β : Type v} [Storage β] {shape : Shape}
    (f : α → β) (tensor : Tensor α shape) :
    Tensor β shape :=
  ofFlatFn fun index => f (tensor.getFlat index)

/-- Native implementation of pointwise scalar maps. -/
def mapFast {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s : Shape}
    (f : α → β) (x : Rep α s) :
    Rep β s :=
  ofFlatNativeFn
    (fun index hIndex => f (x.getFlatUSize index hIndex))
    (fun index => f (x.getFlat index))
    (by
      intro index hIndex
      rw [x.getFlatUSize_eq_getFlat index hIndex])

/-- Compiled code runs `mapFast` in place of `map`. -/
@[csimp] theorem mapFast_eq : @map = @mapFast := by
  funext α _ β _ s f x
  exact (ofFlatNativeFn_eq_ofFlatFn _ _ _).symm

/--
Apply a binary function at every coordinate of two tensors of the same shape.

Tensors of different shapes do not typecheck here, so an operand of another shape is first
broadcast with `Tensor.broadcast`. `zipWith3` is the ternary form.
-/
def _root_.TorchLean.Tensor.zipWith {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s : Shape}
    (f : α → β → γ) (left : Tensor α s) (right : Tensor β s) : Tensor γ s :=
  ofFlatFn fun index => f (left.getFlat index) (right.getFlat index)

/-- Native implementation of pointwise binary maps. -/
def zipWithFast {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s : Shape}
    (f : α → β → γ) (x : Rep α s) (y : Rep β s) : Rep γ s :=
  ofFlatNativeFn
    (fun index hIndex =>
      f (x.getFlatUSize index hIndex) (y.getFlatUSize index hIndex))
    (fun index => f (x.getFlat index) (y.getFlat index))
    (by
      intro index hIndex
      rw [x.getFlatUSize_eq_getFlat index hIndex,
        y.getFlatUSize_eq_getFlat index hIndex])

/-- Compiled code runs `zipWithFast` in place of `zipWith`. -/
@[csimp] theorem zipWithFast_eq : @zipWith = @zipWithFast := by
  funext α _ β _ γ _ s f x y
  exact (ofFlatNativeFn_eq_ofFlatFn _ _ _).symm

/-- Apply a ternary scalar function at every coordinate of three tensors of one shape. -/
def _root_.TorchLean.Tensor.zipWith3 {α : Type u} [Storage α]
    {β : Type v} [Storage β] {γ : Type w} [Storage γ] {δ : Type x} [Storage δ] {s : Shape}
    (f : α → β → γ → δ) (x : Tensor α s) (y : Tensor β s) (z : Tensor γ s) : Tensor δ s :=
  ofFlatFn fun index => f (x.getFlat index) (y.getFlat index) (z.getFlat index)

/-- Native implementation of pointwise ternary maps. -/
def zipWith3Fast {α : Type u} [Storage α]
    {β : Type v} [Storage β] {γ : Type w} [Storage γ] {δ : Type x} [Storage δ] {s : Shape}
    (f : α → β → γ → δ) (x : Rep α s) (y : Rep β s) (z : Rep γ s) : Rep δ s :=
  ofFlatNativeFn
    (fun index hIndex =>
      f (x.getFlatUSize index hIndex) (y.getFlatUSize index hIndex)
        (z.getFlatUSize index hIndex))
    (fun index => f (x.getFlat index) (y.getFlat index) (z.getFlat index))
    (by
      intro index hIndex
      rw [x.getFlatUSize_eq_getFlat index hIndex, y.getFlatUSize_eq_getFlat index hIndex,
        z.getFlatUSize_eq_getFlat index hIndex])

/-- Compiled code runs `zipWith3Fast` in place of `zipWith3`. -/
@[csimp] theorem zipWith3Fast_eq : @zipWith3 = @zipWith3Fast := by
  funext α _ β _ γ _ δ _ s f x y z
  exact (ofFlatNativeFn_eq_ofFlatFn _ _ _).symm

/-- Tensor addition applies scalar addition entrywise. -/
instance {α : Type u} [Storage α] {s : Shape} [Add α] : Add (Rep α s) where
  add := zipWith (· + ·)

/-- Every coordinate of a filled tensor contains its fill value. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.full_apply {α : Type u} [Storage α]
    (shape : Shape) (value : α) (coordinate : shape.Coord) :
    full shape value coordinate = value := by
  simp [full, get]

/-- Evaluating a pointwise map applies the scalar function at that coordinate. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.map_apply
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s : Shape}
    (f : α → β) (x : Rep α s) (i : Coord s) : map f x i = f (x i) := by
  change
    getFlat (ofFlatFn fun index => f (x.getFlat index))
        (Coord.linearize i) =
      f (x.getFlat (Coord.linearize i))
  exact getFlat_ofFlatFn _ _

/-- Evaluating a pointwise binary operation combines both scalar entries. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.zipWith_apply {α : Type u} [Storage α]
    {β : Type v} [Storage β] {γ : Type w} [Storage γ] {s : Shape}
    (f : α → β → γ) (x : Rep α s) (y : Rep β s) (i : Coord s) :
    zipWith f x y i = f (x i) (y i) := by
  change
    getFlat
        (ofFlatFn fun index => f (x.getFlat index) (y.getFlat index))
        (Coord.linearize i) =
      f (x.getFlat (Coord.linearize i)) (y.getFlat (Coord.linearize i))
  exact getFlat_ofFlatFn _ _

/-- A coordinate of `zipWith3 f x y z` is `f` of the three entries there. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.zipWith3_apply {α : Type u} [Storage α]
    {β : Type v} [Storage β] {γ : Type w} [Storage γ] {δ : Type x} [Storage δ] {s : Shape}
    (f : α → β → γ → δ) (x : Rep α s) (y : Rep β s) (z : Rep γ s) (i : Coord s) :
    zipWith3 f x y z i = f (x i) (y i) (z i) := by
  change
    getFlat
        (ofFlatFn fun index => f (x.getFlat index) (y.getFlat index) (z.getFlat index))
        (Coord.linearize i) =
      f (x.getFlat (Coord.linearize i)) (y.getFlat (Coord.linearize i))
        (z.getFlat (Coord.linearize i))
  exact getFlat_ofFlatFn _ _

/-- Tensor addition is coordinatewise scalar addition. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.add_apply {α : Type u} [Storage α]
    {s : Shape} [Add α] (x y : Rep α s) (i : Coord s) : (x + y) i = x i + y i := by
  exact zipWith_apply (· + ·) x y i

/-!
### Pointwise algebraic structure

Every algebraic operation on native tensors is the scalar operation applied coordinatewise. We
define the homogeneous `Zero`, `Neg`, `Sub` and `SMul R` instances on `Rep α s`. With `Add` above
they form the pointwise `AddCommMonoid`, `AddCommGroup` and `Module R` structures. The low-priority
`HAdd`, `HSub`, `HMul` and `HDiv` instances in `TorchLean.Tensor.Promotion`
combine tensors of two element types. The `*_apply` lemmas are the simp normal forms.
-/

/-- The zero tensor holds the scalar zero at every coordinate. -/
instance {α : Type u} [Storage α] {s : Shape} [Zero α] : Zero (Rep α s) where
  zero := full _ 0

/-- Tensor negation negates every coordinate. -/
instance {α : Type u} [Storage α] {s : Shape} [Neg α] : Neg (Rep α s) where
  neg := map Neg.neg

/-- Tensor subtraction subtracts coordinatewise. -/
instance {α : Type u} [Storage α] {s : Shape} [Sub α] : Sub (Rep α s) where
  sub := zipWith (· - ·)

/-- A scalar acts on a tensor by acting on every coordinate. -/
instance {R : Type v} {α : Type u} [Storage α] {s : Shape} [SMul R α] : SMul R (Rep α s) where
  smul c := map (c • ·)

/-- Every coordinate of the zero tensor is the scalar zero. -/
@[simp, grind =] theorem zero_apply {α : Type u} [Storage α] {s : Shape} [Zero α] (i : Coord s) :
    (0 : Rep α s) i = 0 :=
  full_apply _ 0 i

/-- Tensor negation is coordinatewise scalar negation. -/
@[simp, grind =] theorem neg_apply {α : Type u} [Storage α] {s : Shape} [Neg α]
    (x : Rep α s) (i : Coord s) : (-x) i = -(x i) :=
  map_apply Neg.neg x i

/-- Tensor subtraction is coordinatewise scalar subtraction. -/
@[simp, grind =] theorem hSub_apply {α : Type u} [Storage α] {s : Shape} [Sub α]
    (x y : Rep α s) (i : Coord s) : (x - y) i = x i - y i :=
  zipWith_apply (· - ·) x y i

/-- A scalar acts on a tensor coordinatewise. -/
@[simp, grind =] theorem smul_apply {R : Type v} {α : Type u} [Storage α] {s : Shape} [SMul R α]
    (c : R) (x : Rep α s) (i : Coord s) : (c • x) i = c • x i :=
  map_apply (c • ·) x i

/-- Tensor addition is a commutative monoid, coordinatewise. -/
instance {α : Type u} [Storage α] {s : Shape} [AddCommMonoid α] : AddCommMonoid (Rep α s) where
  add_assoc x y z := by
    ext i
    simp only [add_apply, add_assoc]
  zero_add x := by
    ext i
    simp only [add_apply, zero_apply, zero_add]
  add_zero x := by
    ext i
    simp only [add_apply, zero_apply, add_zero]
  add_comm x y := by
    ext i
    simp only [add_apply, add_comm]
  nsmul n x := n • x
  nsmul_zero x := by
    ext i
    simp only [smul_apply, zero_smul, zero_apply]
  nsmul_succ n x := by
    ext i
    simp only [smul_apply, add_apply, succ_nsmul]

/-- Tensor subtraction and negation form a commutative group, coordinatewise. -/
instance {α : Type u} [Storage α] {s : Shape} [AddCommGroup α] : AddCommGroup (Rep α s) where
  neg_add_cancel x := by
    ext i
    simp only [add_apply, neg_apply, zero_apply, neg_add_cancel]
  sub_eq_add_neg x y := by
    ext i
    simp only [hSub_apply, add_apply, neg_apply, sub_eq_add_neg]
  zsmul n x := n • x
  zsmul_zero' x := by
    ext i
    simp only [smul_apply, zero_smul, zero_apply]
  zsmul_succ' n x := by
    ext i
    simp only [smul_apply, add_apply, natCast_zsmul, succ_nsmul]
  zsmul_neg' n x := by
    ext i
    simp only [smul_apply, neg_apply, negSucc_zsmul, natCast_zsmul, Nat.succ_eq_add_one]

/-- Tensors over a module are a module, coordinatewise. -/
instance {R : Type v} {α : Type u} [Storage α] {s : Shape} [Semiring R] [AddCommMonoid α]
    [Module R α] : Module R (Rep α s) where
  one_smul x := by
    ext i
    simp only [smul_apply, one_smul]
  mul_smul a b x := by
    ext i
    simp only [smul_apply, mul_smul]
  smul_zero a := by
    ext i
    simp only [smul_apply, zero_apply, smul_zero]
  smul_add a x y := by
    ext i
    simp only [smul_apply, add_apply, smul_add]
  add_smul a b x := by
    ext i
    simp only [smul_apply, add_apply, add_smul]
  zero_smul x := by
    ext i
    simp only [smul_apply, zero_apply, zero_smul]

/--
Mapping the identity function leaves every tensor unchanged.

It is a `simp` and `einops_simps` lemma only. As a `grind` rule next to `map_map`, e-matching
could build arbitrarily many nested identity maps.
-/
@[simp, einops_simps] theorem map_id {α : Type u} [Storage α] {s : Shape} (x : Rep α s) :
    map id x = x := by
  ext coordinate
  simp only [map_apply, id_eq]

/--
Two pointwise maps fuse to one scalar composition.

For the reason given at `map_id`, it is a `simp` lemma without a `grind` attribute.
-/
@[simp, einops_simps] theorem map_map {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {s : Shape} (outer : β → γ) (inner : α → β) (x : Rep α s) :
    map outer (map inner x) = map (outer ∘ inner) x := by
  ext coordinate
  simp only [map_apply, Function.comp_apply]

/-- A scalar map after a pointwise binary operation fuses into that operation. -/
@[simp, einops_simps] theorem map_zipWith {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {δ : Type*} [Storage δ] {s : Shape} (outer : γ → δ) (combine : α → β → γ)
    (x : Rep α s) (y : Rep β s) :
    map outer (zipWith combine x y) =
      zipWith (fun left right => outer (combine left right)) x y := by
  ext coordinate
  simp only [map_apply, zipWith_apply]

/-- Pointwise maps on both inputs fuse into one pointwise binary operation. -/
@[simp, einops_simps] theorem zipWith_map_map {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {δ ε : Type*} [Storage δ] [Storage ε] {s : Shape}
    (combine : γ → δ → ε) (leftMap : α → γ) (rightMap : β → δ)
    (x : Rep α s) (y : Rep β s) :
    zipWith combine (map leftMap x) (map rightMap y) =
      zipWith
        (fun left right => combine (leftMap left) (rightMap right)) x y := by
  ext coordinate
  simp only [zipWith_apply, map_apply]

/--
Stack `n` tensors of shape `shape` along a leading axis of size `n`.

Proofs use this definition, and compiled code runs `dimFast`.
-/
def _root_.TorchLean.Tensor.dim {α : Type u} [Storage α] {n : Nat} {shape : Shape}
    (values : Fin n → Tensor α shape) : Tensor α (n :: shape) :=
  ofFn fun coordinate => values coordinate.1 coordinate.2

/-- Select one leading-axis slice from a tensor. -/
def _root_.TorchLean.Tensor.unstack {α : Type u} [Storage α] {n : Nat} {shape : Shape}
    (tensor : Tensor α (n :: shape)) (index : Fin n) : Tensor α shape :=
  ofFn fun coordinate => tensor (index, coordinate)

/-- Observing a stack selects the component named by its leading coordinate. -/
@[simp, grind =, einops_simps]
theorem _root_.TorchLean.Tensor.dim_apply {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (coordinate : Coord (n :: s)) :
    dim components coordinate =
      components coordinate.1 coordinate.2 := by
  simp only [dim, get_ofFn]

/-- Observing a leading-axis slice fixes that axis to the selected component. -/
@[simp, grind =, einops_simps]
theorem _root_.TorchLean.Tensor.unstack_apply {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n)
    (coordinate : Coord s) :
    unstack tensor component coordinate = tensor (component, coordinate) := by
  simp only [unstack, get_ofFn]

/-- A tensor read is the corresponding entry of the ordinary array observation. -/
theorem get_eq_toArray_getElem {α : Type u} [storage : Storage α] {s : Shape}
    (x : Rep α s) (i : Coord s) :
    x i = (storage.toArray x.buffer)[(Coord.linearize i).val]'(by
      rw [storage.toArray_size, x.size_eq]
      exact (Coord.linearize i).isLt) :=
  (storage.toArray_get x.buffer _
    (by simpa only [x.size_eq] using (Coord.linearize i).isLt) _).symm

/-- Observing a finite physical-buffer fold commutes with slice appends. -/
private theorem toArray_fin_foldl_appendSlice {α : Type u} [storage : Storage α] {n : Nat}
    (sources : Fin n → storage.Buffer) (start stop : Nat) (initial : storage.Buffer) :
    storage.toArray
        (Fin.foldl n (fun output i => storage.appendSlice (sources i) start stop output)
          initial) =
      Fin.foldl n (fun output i => output ++ (storage.toArray (sources i)).extract start stop)
        (storage.toArray initial) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Fin.foldl_succ_last, Fin.foldl_succ_last, storage.toArray_appendSlice, ih]

/-- Appending `n` arrays of length `m` adds `m * n` entries. -/
private theorem size_fin_foldl_append {α : Type u} {n m : Nat} (blocks : Fin n → Array α)
    (hBlocks : ∀ i, (blocks i).size = m) (initial : Array α) :
    (Fin.foldl n (fun output i => output ++ blocks i) initial).size = initial.size + m * n := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Fin.foldl_succ_last, Array.size_append, ih (fun i => blocks i.castSucc)
        (fun i => hBlocks _), hBlocks, Nat.mul_succ, Nat.add_assoc]

/-- Appending `n` arrays of length `m` to `#[]` puts entry `j` of array `i` at `j + m * i`. -/
private theorem getElem_fin_foldl_append {α : Type u} {n m : Nat} (blocks : Fin n → Array α)
    (hBlocks : ∀ i, (blocks i).size = m) (i : Fin n) (j : Nat) (hj : j < m)
    (h : j + m * i.val < (Fin.foldl n (fun output i => output ++ blocks i) #[]).size) :
    (Fin.foldl n (fun output i => output ++ blocks i) #[])[j + m * i.val] =
      (blocks i)[j]'(by rw [hBlocks]; exact hj) := by
  induction n with
  | zero => exact i.elim0
  | succ n ih =>
      have hPrefix := size_fin_foldl_append (fun i : Fin n => blocks i.castSucc)
        (fun i => hBlocks _) #[]
      rw [Array.size_empty, Nat.zero_add] at hPrefix
      cases i using Fin.lastCases with
      | last =>
          simp only [Fin.foldl_succ_last]
          rw [Array.getElem_append_right (by rw [hPrefix, Fin.val_last]; omega)]
          simp only [hPrefix, Fin.val_last, Nat.add_sub_cancel]
      | cast i =>
          have hStep := Nat.mul_le_mul_left m (Nat.succ_le_of_lt i.isLt)
          rw [Nat.mul_succ] at hStep
          simp only [Fin.foldl_succ_last]
          rw [Array.getElem_append_left (by rw [hPrefix, Fin.val_castSucc]; omega)]
          simp only [Fin.val_castSucc]
          exact ih (fun i => blocks i.castSucc) (fun i => hBlocks _) i _

/--
Native implementation of leading-axis stacking.

For nonempty component shapes, each component is evaluated once in ascending order, and its
buffer is copied into the output with one `appendSlice`. Empty component shapes need no
component evaluation. The size check succeeds under the storage laws because every component has
`Shape.size s` entries. Its fallback is the reference definition.
-/
def dimFast {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) : Rep α (n :: s) :=
  if Shape.size s = 0 then
    -- An empty output needs no component evaluation, even when the leading extent is huge.
    ofFn fun coordinate => components coordinate.1 coordinate.2
  else
    let buffer := Fin.foldl n
      (fun output component =>
        let tensor := components component
        storage.appendSlice tensor.buffer 0 (Shape.size s) output)
      (storage.emptyWithCapacity (n * Shape.size s))
    if hSize : storage.size buffer = Shape.size (n :: s) then
      ⟨buffer, hSize⟩
    else
      ofFn fun coordinate => components coordinate.1 coordinate.2

/-- Compiled code runs `dimFast` in place of `dim`. -/
@[csimp] theorem dimFast_eq : @dim = @dimFast := by
  funext α storage n s components
  unfold dimFast
  split
  · rfl
  · dsimp only
    split
    · rename_i hSize
      have hBlocks (i : Fin n) :
          ((storage.toArray (components i).buffer).extract 0 (Shape.size s)).size =
            Shape.size s := by
        simp [storage.toArray_size, (components i).size_eq]
      apply ext
      rintro ⟨component, coordinate⟩
      simp only [dim_apply]
      rw [get_eq_toArray_getElem (components component) coordinate,
        get_eq_toArray_getElem (⟨_, hSize⟩ : Rep α (n :: s)) (component, coordinate)]
      simp only [toArray_fin_foldl_appendSlice, storage.toArray_emptyWithCapacity,
        Array.emptyWithCapacity_eq, Coord.linearize_cons_val]
      rw [getElem_fin_foldl_append _ hBlocks component _ (Coord.linearize coordinate).isLt]
      simp only [Array.getElem_extract, Nat.zero_add]
    · rfl

/-- Component `i` of a leading-axis stack ends inside the buffer. -/
theorem slice_end_le {n : Nat} {s : Shape} (component : Fin n) :
    Shape.size s * component.val + Shape.size s ≤ Shape.size s * n := by
  have hStep := Nat.mul_le_mul_left (Shape.size s) (Nat.succ_le_of_lt component.isLt)
  rw [Nat.mul_succ] at hStep
  exact hStep

/-- The ordinary array observation of a leading-axis tensor has the product length. -/
theorem toArray_size_cons {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) :
    (storage.toArray tensor.buffer).size = Shape.size s * n := by
  rw [storage.toArray_size, tensor.size_eq]
  exact Nat.mul_comm _ _

/-- Buffer of `unstackFast`: the component's contiguous interval, copied into a fresh buffer. -/
@[inline] def sliceBuffer {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) : storage.Buffer :=
  storage.appendSlice tensor.buffer (Shape.size s * component.val)
    (Shape.size s * component.val + Shape.size s)
    (storage.emptyWithCapacity (Shape.size s))

/-- The copied interval is the matching extract of the source observation. -/
theorem toArray_sliceBuffer {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) :
    storage.toArray (sliceBuffer tensor component) =
      (storage.toArray tensor.buffer).extract (Shape.size s * component.val)
        (Shape.size s * component.val + Shape.size s) := by
  rw [sliceBuffer, storage.toArray_appendSlice, storage.toArray_emptyWithCapacity,
    Array.emptyWithCapacity_eq, Array.empty_append]

/--
Leading-axis slice that copies the component's contiguous interval with one `appendSlice`.

`unstack` reads each entry through its coordinate, which linearizes the coordinate for every
scalar. The row-major layout puts component `i` at `[i * size s, (i + 1) * size s)`, so compiled
code copies that interval in one call.
-/
def unstackFast {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) : Rep α s where
  buffer := sliceBuffer tensor component
  size_eq := by
    have hEnd := slice_end_le (s := s) component
    rw [← storage.toArray_size, toArray_sliceBuffer, Array.size_extract,
      toArray_size_cons]
    omega

/-- The contiguous-copy slice agrees with the coordinate definition of `unstack`. -/
@[csimp] theorem unstackFast_eq : @unstack = @unstackFast := by
  funext α storage n s tensor component
  apply ext
  intro coordinate
  have hSlice := toArray_sliceBuffer tensor component
  rw [unstack_apply, get_eq_toArray_getElem tensor,
    get_eq_toArray_getElem (unstackFast tensor component) coordinate]
  change _ = (storage.toArray (sliceBuffer tensor component))[_]'_
  simp only [hSlice, Array.getElem_extract, Coord.linearize_cons_val, Nat.add_comm]

/-- Every slice of a filled tensor is filled with the same value. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.unstack_full {α : Type u} [Storage α]
    {n : Nat} {s : Shape} (value : α) (component : Fin n) :
    unstack (full (n :: s) value) component =
      full s value := by
  ext coordinate
  simp

/-- Mapping commutes with stacking, so a map pushes into every slice. -/
@[simp, grind =, einops =, einops_simps]
theorem _root_.TorchLean.Tensor.map_dim {α : Type u} [Storage α] {β : Type v} [Storage β]
    {n : Nat} {s : Shape} (f : α → β)
    (components : Fin n → Rep α s) :
    map f (dim components) =
      dim (fun component => map f (components component)) := by
  ext coordinate
  simp only [map_apply, dim_apply]

/-- Pointwise binary operations distribute over the leading tensor dimension. -/
@[simp, grind =, einops =, einops_simps]
theorem _root_.TorchLean.Tensor.zipWith_dim {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {n : Nat} {s : Shape} (f : α → β → γ)
    (x : Fin n → Rep α s) (y : Fin n → Rep β s) :
    zipWith f (dim x) (dim y) =
      dim fun component => zipWith f (x component) (y component) := by
  ext coordinate
  simp only [zipWith_apply, dim_apply]

/-- Slicing after a map is mapping after a slice. -/
@[simp, grind =_, einops =_, einops_simps ←]
theorem _root_.TorchLean.Tensor.unstack_map {α : Type u} [Storage α] {β : Type v} [Storage β]
    {n : Nat} {s : Shape} (f : α → β)
    (x : Rep α (n :: s)) (component : Fin n) :
    unstack (map f x) component = map f (unstack x component) := by
  ext coordinate
  simp only [unstack, map_apply, get_ofFn]

/-- Pointwise binary operations commute with matching leading-axis slices. -/
@[grind =, einops =, einops_simps]
theorem zipWith_unstack {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {n : Nat} {s : Shape} (f : α → β → γ)
    (x : Rep α (n :: s)) (y : Rep β (n :: s))
    (component : Fin n) :
    zipWith f (unstack x component) (unstack y component) =
      unstack (zipWith f x y) component := by
  ext coordinate
  simp only [unstack, zipWith_apply, get_ofFn]

/--
Stack the tensors in a list along a leading axis.

For example, two tensors of shape `[3]` give a tensor of shape `[2, 3]`.
The first component's entries come before the second's in row-major storage.

`dim` asks for components by index, so we cache the list as an array and each lookup takes
constant time.
-/
def stackList {α : Type u} [Storage α] {s : Shape}
    (components : List (Rep α s)) :
    Tensor α (components.length :: s) :=
  let cached := components.toArray
  dim fun (component : Fin components.length) =>
    cached[component.val]'(by
      simpa only [cached, List.size_toArray] using component.isLt)

/-- The leading coordinate chooses a list entry, and the remaining coordinates index that tensor. -/
@[simp, grind =] theorem stackList_apply {α : Type u} [Storage α] {s : Shape}
    (components : List (Rep α s)) (coordinate : Coord (components.length :: s)) :
    stackList components coordinate =
      components.get coordinate.1 coordinate.2 := by
  simp only [stackList, dim_apply, List.getElem_toArray, List.get_eq_getElem]

/--
Turn a list into a one-dimensional tensor in the same order.

For example, `[x, y, z]` gives shape `[3]`, with `x` at coordinate zero.
We cache the list as an array before `Storage.ofFn` copies its entries into
the buffer selected by `Storage α`. Without the cache, requesting indices
`0, ..., n - 1` would follow `n * (n - 1) / 2` list links in total.
The array makes each indexed read constant time.
-/
def ofList {α : Type u} [Storage α] (values : List α) :
    Tensor α [values.length] :=
  let cached := values.toArray
  let getValue (index : Fin values.length) : α :=
    cached[index.val]'(by
      simpa only [cached, List.size_toArray] using index.isLt)
  { buffer := Storage.ofFn getValue
    size_eq := by simpa using Storage.size_ofFn getValue }

/-- A rank-one tensor built from a list reads the corresponding list entry. -/
@[simp, grind =] theorem ofList_apply {α : Type u} [storage : Storage α]
    (values : List α) (index : Fin values.length) :
    (ofList values : Rep α [values.length]) (index, PUnit.unit) =
      values.get index := by
  change
    storage.get (Storage.ofFn values.get)
        (Coord.linearize (s := [values.length]) (index, PUnit.unit)).val _ =
      values.get index
  have hLinear :
      (Coord.linearize (s := [values.length]) (index, PUnit.unit)).val =
        index.val := by
    have hStep := Coord.linearize_cons_val (s := []) index PUnit.unit
    have hTail :
        (Coord.linearize (s := []) PUnit.unit).val < 1 :=
      (Coord.linearize (s := []) PUnit.unit).isLt
    simp only [Shape.size_nil, Nat.one_mul] at hStep
    omega
  simp only [hLinear]
  exact Storage.get_ofFn values.get index _

/-- Observing a list-built tensor returns the original values in row-major order. -/
@[simp] theorem data_ofList {α : Type u} [storage : Storage α]
    (values : List α) :
    (ofList values).data = values.toArray := by
  calc
    (ofList values).data = Array.ofFn values.get := by
      change storage.toArray (Storage.ofFn values.get) = Array.ofFn values.get
      exact Storage.toArray_ofFn (storage := storage) values.get
    _ = (List.ofFn values.get).toArray := List.toArray_ofFn.symm
    _ = values.toArray := congrArg List.toArray (List.ofFn_get values)

/-- Slicing the outer axis of a stacked tensor returns the slice that was stacked. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.unstack_dim {α : Type u} [Storage α]
    {n : Nat} {s : Shape} (components : Fin n → Rep α s) (component : Fin n) :
    unstack (dim components) component = components component := by
  ext coordinate
  simp only [unstack, get_ofFn]
  exact dim_apply components (component, coordinate)

/-- Restacking every slice of a tensor returns the tensor.

With `unstack_dim` it makes `dim` and `unstack` mutually inverse, so a proof can eliminate a tensor
by `rw [← dim_unstack]` and reason slice by slice. -/
@[simp, grind =] theorem _root_.TorchLean.Tensor.dim_unstack {α : Type u} [Storage α]
    {n : Nat} {s : Shape} (tensor : Rep α (n :: s)) :
    dim (unstack tensor) = tensor := by
  ext coordinate
  simp only [dim_apply, unstack_apply]

/-- Build a tensor of the requested shape by running one action per row-major flat index, in
increasing order, then materializing native storage from the results.

A failing action stops construction as the monad `m` dictates.
-/
def _root_.TorchLean.Tensor.generateFlatM {m : Type → Type} [Monad m] {α : Type} [Storage α]
    (shape : Shape) (f : Fin shape.size → m α) : m (Tensor α shape) := do
  let values ← sequenceFinM f
  pure (ofFlatFn values)

end Rep

end TorchLean.Tensor
