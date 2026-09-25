/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import Mathlib.Algebra.Module.Defs
public import NN.Tensor.Internal.Representation.Basic.Core

/-!
# Pointwise and family tensor operations

Scalar maps, binary maps, finite tensor families, stacking, and rank-one list
construction over native tensor storage.
-/

@[expose] public section

namespace TorchLean.Tensor.Internal

universe u v w

/-- Evaluate a finite family once per index in ascending order and retain indexed access.

The temporary buffer also supports heterogeneous tensor packs, which have no scalar Storage
instance. Public construction APIs expose tensors or packs rather than this buffer.
-/
def sequenceFinM {m : Type → Type} [Monad m] {α : Type} {n : Nat}
    (f : Fin n → m α) : m (Fin n → α) := do
  let values ← Vector.ofFnM f
  pure values.get

namespace Rep

/-- Native implementation of constant tensor construction. -/
@[inline] unsafe def constFast
    {α : Type u} [Storage α] {s : Shape} (a : α) : Rep α s :=
  ofFlatNativeFn
    (fun _ _ => a)
    (fun _ => a)
    (by intros; rfl)

/-- The constant tensor. -/
@[implemented_by constFast]
def const {α : Type u} [Storage α] {s : Shape} (a : α) : Rep α s :=
  ofFlatFn fun _ => a

/-- Native implementation of pointwise scalar maps. -/
@[inline] unsafe def mapFast {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s : Shape}
    (f : α → β) (x : Rep α s) :
    Rep β s :=
  ofFlatNativeFn
    (fun index hIndex => f (x.getFlatUSize index hIndex))
    (fun index => f (x.getFlat index))
    (by
      intro index hIndex
      rw [x.getFlatUSize_eq_getFlat index hIndex])

/-- Apply a scalar function pointwise. -/
@[implemented_by mapFast]
def map {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s : Shape}
    (f : α → β) (x : Rep α s) :
    Rep β s :=
  ofFlatFn fun index => f (x.getFlat index)

/-- Native implementation of pointwise binary maps. -/
@[inline] unsafe def zipWithFast {α : Type u} [Storage α]
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

/-- Apply a binary scalar function pointwise. -/
@[implemented_by zipWithFast]
def zipWith {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s : Shape}
    (f : α → β → γ) (x : Rep α s) (y : Rep β s) : Rep γ s :=
  ofFlatFn fun index => f (x.getFlat index) (y.getFlat index)

/-- Tensor addition applies scalar addition entrywise. -/
instance {α : Type u} [Storage α] {s : Shape} [Add α] : Add (Rep α s) where
  add := zipWith (· + ·)

/-- Evaluating a constant tensor returns its defining scalar. -/
@[simp, grind =] theorem const_apply {α : Type u} [Storage α] {s : Shape} (a : α)
    (i : Coord s) :
    const (s := s) a i = a := by
  simp [const, get]

/-- Evaluating a pointwise map applies the scalar function at that coordinate. -/
@[simp, grind =] theorem map_apply
    {α : Type u} [Storage α]
    {β : Type v} [Storage β] {s : Shape}
    (f : α → β) (x : Rep α s) (i : Coord s) : map f x i = f (x i) := by
  change
    getFlat (ofFlatFn fun index => f (x.getFlat index))
        (Coord.linearize i) =
      f (x.getFlat (Coord.linearize i))
  exact getFlat_ofFlatFn _ _

/-- Evaluating a pointwise binary map combines the two entries at that coordinate. -/
@[simp, grind =] theorem zipWith_apply {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {s : Shape}
    (f : α → β → γ) (x : Rep α s) (y : Rep β s) (i : Coord s) :
    zipWith f x y i = f (x i) (y i) := by
  change
    getFlat
        (ofFlatFn fun index => f (x.getFlat index) (y.getFlat index))
        (Coord.linearize i) =
      f (x.getFlat (Coord.linearize i)) (y.getFlat (Coord.linearize i))
  exact getFlat_ofFlatFn _ _

/-- Homogeneous tensor addition applies scalar addition pointwise. -/
@[simp, grind =] theorem hAdd_apply {α : Type u} [Storage α] {s : Shape} [Add α]
    (x y : Rep α s) (i : Coord s) : (x + y) i = x i + y i := by
  exact zipWith_apply (· + ·) x y i

/-!
### Pointwise algebraic structure

Every algebraic operation on native tensors is the scalar operation applied coordinatewise. The
instances below are the only `0`, `-`, and `•` on `Rep α s`; together with `+` above they form the
pointwise `AddCommGroup` and `Module` structures. The `*_apply` lemmas are the simp normal forms.
-/

/-- The zero tensor holds the scalar zero at every coordinate. -/
instance {α : Type u} [Storage α] {s : Shape} [Zero α] : Zero (Rep α s) where
  zero := const 0

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
  const_apply 0 i

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
    simp only [hAdd_apply, add_assoc]
  zero_add x := by
    ext i
    simp only [hAdd_apply, zero_apply, zero_add]
  add_zero x := by
    ext i
    simp only [hAdd_apply, zero_apply, add_zero]
  add_comm x y := by
    ext i
    simp only [hAdd_apply, add_comm]
  nsmul n x := n • x
  nsmul_zero x := by
    ext i
    simp only [smul_apply, zero_smul, zero_apply]
  nsmul_succ n x := by
    ext i
    simp only [smul_apply, hAdd_apply, succ_nsmul]

/-- Tensor subtraction and negation form a commutative group, coordinatewise. -/
instance {α : Type u} [Storage α] {s : Shape} [AddCommGroup α] : AddCommGroup (Rep α s) where
  neg_add_cancel x := by
    ext i
    simp only [hAdd_apply, neg_apply, zero_apply, neg_add_cancel]
  sub_eq_add_neg x y := by
    ext i
    simp only [hSub_apply, hAdd_apply, neg_apply, sub_eq_add_neg]
  zsmul n x := n • x
  zsmul_zero' x := by
    ext i
    simp only [smul_apply, zero_smul, zero_apply]
  zsmul_succ' n x := by
    ext i
    simp only [smul_apply, hAdd_apply, natCast_zsmul, succ_nsmul]
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
    simp only [smul_apply, hAdd_apply, smul_add]
  add_smul a b x := by
    ext i
    simp only [smul_apply, hAdd_apply, add_smul]
  zero_smul x := by
    ext i
    simp only [smul_apply, zero_apply, zero_smul]

/--
Mapping the identity function leaves every tensor unchanged.

This is a deterministic simplification rule rather than a global e-matching
rule. Combined with map composition, unrestricted congruence closure could
otherwise manufacture arbitrarily many nested identity maps.
-/
@[simp] theorem map_id {α : Type u} [Storage α] {s : Shape} (x : Rep α s) :
    map id x = x := by
  ext coordinate
  simp only [map_apply, id_eq]

/--
Two pointwise maps fuse to one scalar composition.

The theorem belongs to the canonical simplifier, but intentionally not to the
global `grind` set: composition-producing e-matching rules can repeatedly
instantiate through equal identity-map terms.
-/
@[simp] theorem map_map {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {s : Shape} (outer : β → γ) (inner : α → β) (x : Rep α s) :
    map outer (map inner x) = map (outer ∘ inner) x := by
  ext coordinate
  simp only [map_apply, Function.comp_apply]

/-- A scalar map after a pointwise binary operation fuses into that operation. -/
@[simp] theorem map_zipWith {α : Type u} [Storage α]
    {β : Type v} [Storage β]
    {γ : Type w} [Storage γ]
    {δ : Type*} [Storage δ] {s : Shape} (outer : γ → δ) (combine : α → β → γ)
    (x : Rep α s) (y : Rep β s) :
    map outer (zipWith combine x y) =
      zipWith (fun left right => outer (combine left right)) x y := by
  ext coordinate
  simp only [map_apply, zipWith_apply]

/-- Pointwise maps on both inputs fuse into one pointwise binary operation. -/
@[simp] theorem zipWith_map_map {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {δ ε : Type*} [Storage δ] [Storage ε] {s : Shape}
    (combine : γ → δ → ε) (leftMap : α → γ) (rightMap : β → δ)
    (x : Rep α s) (y : Rep β s) :
    zipWith combine (map leftMap x) (map rightMap y) =
      zipWith
        (fun left right => combine (leftMap left) (rightMap right)) x y := by
  ext coordinate
  simp only [zipWith_apply, map_apply]

/--
Native implementation of leading-axis stacking.

Each component is evaluated once, in ascending order, and its buffer is copied into the output
with one `appendSlice`. The size check always succeeds because every component has
`Shape.size s` entries; its fallback is the reference definition.
-/
@[inline] unsafe def stackFast {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
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

/--
Place a finite family of identically shaped tensors along a new leading axis.

The transparent definition is the proof semantics. Compiled code uses
`stackFast`, which preserves these semantics while evaluating each component
only once.
-/
@[implemented_by stackFast]
def stack {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) : Rep α (n :: s) :=
  ofFn fun coordinate => components coordinate.1 coordinate.2

/-- Select one leading-axis slice from a stacked tensor. -/
def unstack {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) : Rep α s :=
  ofFn fun coordinate => tensor (component, coordinate)

/-- Observing a stack selects the component named by its leading coordinate. -/
@[simp, grind =] theorem stack_apply {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (coordinate : Coord (n :: s)) :
    stack components coordinate =
      components coordinate.1 coordinate.2 := by
  simp only [stack, get_ofFn]

/-- Observing a leading-axis slice fixes that axis to the selected component. -/
@[simp, grind =] theorem unstack_apply {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n)
    (coordinate : Coord s) :
    unstack tensor component coordinate = tensor (component, coordinate) := by
  simp only [unstack, get_ofFn]

/-- A tensor read is the corresponding entry of the ordinary array observation. -/
theorem get_eq_toArray_getElem {α : Type u} [storage : Storage α] {s : Shape}
    (x : Rep α s) (i : Coord s)
    (h : (Coord.linearize i).val < (storage.toArray x.buffer).size) :
    x i = (storage.toArray x.buffer)[(Coord.linearize i).val]'h :=
  (storage.toArray_get x.buffer _
    (by simpa only [x.size_eq] using (Coord.linearize i).isLt) h).symm

/-- Component `i` of a leading-axis stack ends inside the buffer. -/
theorem unstackSlice_end_le {n : Nat} {s : Shape} (component : Fin n) :
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

/-- Buffer of `unstackSlice`: the component's contiguous interval, copied into a fresh buffer. -/
@[inline] def unstackSliceBuffer {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) : storage.Buffer :=
  storage.appendSlice tensor.buffer (Shape.size s * component.val)
    (Shape.size s * component.val + Shape.size s)
    (storage.emptyWithCapacity (Shape.size s))

/-- The copied interval is the matching extract of the source observation. -/
theorem toArray_unstackSliceBuffer {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) :
    storage.toArray (unstackSliceBuffer tensor component) =
      (storage.toArray tensor.buffer).extract (Shape.size s * component.val)
        (Shape.size s * component.val + Shape.size s) := by
  rw [unstackSliceBuffer, storage.toArray_appendSlice, storage.toArray_emptyWithCapacity,
    Array.emptyWithCapacity_eq, Array.empty_append]

/--
Leading-axis slice that copies the component's contiguous interval with one `appendSlice`.

`unstack` reads each entry through its coordinate, which linearizes the coordinate for every
scalar. The row-major layout puts component `i` at `[i * size s, (i + 1) * size s)`, so compiled
code copies that interval directly.
-/
def unstackSlice {α : Type u} [storage : Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) (component : Fin n) : Rep α s where
  buffer := unstackSliceBuffer tensor component
  size_eq := by
    have hEnd := unstackSlice_end_le (s := s) component
    rw [← storage.toArray_size, toArray_unstackSliceBuffer, Array.size_extract,
      toArray_size_cons]
    omega

/-- The contiguous-copy slice agrees with the coordinate definition of `unstack`. -/
@[csimp] theorem unstack_eq_unstackSlice : @unstack = @unstackSlice := by
  funext α storage n s tensor component
  apply ext
  intro coordinate
  have hEnd := unstackSlice_end_le (s := s) component
  have hFlat := (Coord.linearize coordinate).isLt
  have hSource := toArray_size_cons tensor
  have hSlice := toArray_unstackSliceBuffer tensor component
  rw [unstack_apply,
    get_eq_toArray_getElem tensor _ (by rw [hSource, Coord.linearize_cons_val]; omega),
    get_eq_toArray_getElem (unstackSlice tensor component) coordinate
      (by
        change _ < (storage.toArray (unstackSliceBuffer tensor component)).size
        rw [hSlice, Array.size_extract, hSource]
        omega)]
  change _ = (storage.toArray (unstackSliceBuffer tensor component))[_]'_
  simp only [hSlice, Array.getElem_extract, Coord.linearize_cons_val, Nat.add_comm]

/-- Every leading-axis slice of a constant tensor is the same constant tensor. -/
@[simp, grind =] theorem unstack_const {α : Type u} [Storage α]
    {n : Nat} {s : Shape} (value : α) (component : Fin n) :
    unstack (const (s := n :: s) value) component =
      const (s := s) value := by
  ext coordinate
  simp

/-- Pointwise scalar maps commute with stacking a finite tensor family. -/
@[grind =] theorem map_stack {α : Type u} [Storage α] {β : Type v} [Storage β]
    {n : Nat} {s : Shape} (f : α → β)
    (components : Fin n → Rep α s) :
    map f (stack components) =
      stack (fun component => map f (components component)) := by
  ext coordinate
  simp only [map_apply, stack_apply]

/-- Pointwise binary operations commute with stacking matching families. -/
@[grind =] theorem zipWith_stack {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {n : Nat} {s : Shape} (f : α → β → γ)
    (x : Fin n → Rep α s) (y : Fin n → Rep β s) :
    zipWith f (stack x) (stack y) =
      stack fun component => zipWith f (x component) (y component) := by
  ext coordinate
  simp only [zipWith_apply, stack_apply]

/-- Pointwise scalar maps commute with selecting a leading-axis slice. -/
@[grind =] theorem map_unstack {α : Type u} [Storage α] {β : Type v} [Storage β]
    {n : Nat} {s : Shape} (f : α → β)
    (x : Rep α (n :: s)) (component : Fin n) :
    map f (unstack x component) = unstack (map f x) component := by
  ext coordinate
  simp only [unstack, map_apply, get_ofFn]

/-- Pointwise binary operations commute with matching leading-axis slices. -/
@[grind =] theorem zipWith_unstack {α : Type u} [Storage α] {β : Type v} [Storage β]
    {γ : Type w} [Storage γ] {n : Nat} {s : Shape} (f : α → β → γ)
    (x : Rep α (n :: s)) (y : Rep β (n :: s))
    (component : Fin n) :
    zipWith f (unstack x component) (unstack y component) =
      unstack (zipWith f x y) component := by
  ext coordinate
  simp only [unstack, zipWith_apply, get_ofFn]

/--
Stack the tensors in a list along a new leading axis.

For example, two tensors of shape `[3]` give a tensor of shape `[2, 3]`.
The first component's entries come before the second's in row-major storage.

`stack` asks for components by index. We cache the list as an array so each
lookup takes constant time. Reading the list directly would walk from its
front again for each component.
-/
def stackList {α : Type u} [Storage α] {s : Shape}
    (components : List (Rep α s)) :
    Rep α (components.length :: s) :=
  let cached := components.toArray
  stack fun (component : Fin components.length) =>
    cached[component.val]'(by
      simpa only [cached, List.size_toArray] using component.isLt)

/-- The leading coordinate chooses a list entry; the remaining coordinates index that tensor. -/
@[simp, grind =] theorem stackList_apply {α : Type u} [Storage α] {s : Shape}
    (components : List (Rep α s)) (coordinate : Coord (components.length :: s)) :
    stackList components coordinate =
      components.get coordinate.1 coordinate.2 := by
  simp only [stackList, stack_apply, List.getElem_toArray, List.get_eq_getElem]

/--
Turn a list into a one-dimensional tensor in the same order.

For example, `[x, y, z]` gives shape `[3]`, with `x` at coordinate zero.
We cache the list as an array before `Storage.ofFn` copies its entries into
the buffer selected by `Storage α`. Without the cache, requesting indices
`0, ..., n - 1` would follow `n * (n - 1) / 2` list links in total.
The array makes each indexed read constant time.
-/
def ofList {α : Type u} [Storage α] (values : List α) :
    Rep α [values.length] :=
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
  have hData := Storage.toArray_ofFn values.get
  have hBound :
      (Coord.linearize (s := [values.length]) (index, PUnit.unit)).val <
        storage.size (Storage.ofFn values.get) := by
    simpa only [Storage.size_ofFn, hLinear] using index.isLt
  have hArrayBound :
      (Coord.linearize (s := [values.length]) (index, PUnit.unit)).val <
        (storage.toArray (Storage.ofFn values.get)).size := by
    simpa only [hData, Array.size_ofFn, hLinear] using index.isLt
  have hGet := storage.toArray_get
    (Storage.ofFn values.get)
    (Coord.linearize (s := [values.length]) (index, PUnit.unit)).val
    hBound hArrayBound
  simpa only [hData, Array.getElem_ofFn, hLinear] using hGet.symm

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

/--
Ordinary list notation constructs a rank-one tensor when the literal length
matches the expected static dimension. Each list entry is copied into the
tensor's storage in order.
-/
instance instCoeDepListTensor {α : Type u} [Storage α] (values : List α) :
    CoeDep (List α) values (Rep α [values.length]) where
  coe := ofList values

/-- Selecting a component after stacking recovers the original component. -/
@[simp, grind =] theorem unstack_stack {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → Rep α s) (component : Fin n) :
    unstack (stack components) component = components component := by
  ext coordinate
  simp only [unstack, get_ofFn]
  exact stack_apply components (component, coordinate)

/-- Stacking every leading-axis slice reconstructs the original tensor. -/
@[simp, grind =] theorem stack_unstack {α : Type u} [Storage α] {n : Nat} {s : Shape}
    (tensor : Rep α (n :: s)) :
    stack (unstack tensor) = tensor := by
  ext coordinate
  simp only [stack_apply, unstack_apply]

/-- Fill a temporary finite buffer in row-major effect order, then materialize native storage. -/
def ofFlatFnM {m : Type → Type} [Monad m] {α : Type} [Storage α] {s : Shape}
    (f : Fin s.size → m α) : m (Rep α s) := do
  let values ← sequenceFinM f
  pure (ofFlatFn values)

/-- Sequence a finite family of component actions once each before stacking their native buffers. -/
def stackM {m : Type → Type} [Monad m] {α : Type} [Storage α] {n : Nat} {s : Shape}
    (components : Fin n → m (Rep α s)) : m (Rep α (n :: s)) := do
  let values ← sequenceFinM components
  pure (stack values)

end Rep

end TorchLean.Tensor.Internal
