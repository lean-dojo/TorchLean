/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.TypedGraph.Idx
public import TorchLean.Tensor.ShapeErasure

/-!
# Tensor lookups and contexts

`TensorLookup` reads the tensors of a shape context by typed index, and `TensorContext` stores them
in one array of shape-erased tensors with a proof that the array comes from a typed pack. The proof
is erased at runtime: reads select tensors without traversing a linked pack, and conversion to a
pack happens only at the explicit typed boundary `TensorContext.toPack`.

A `Contributions` value describes the output of one VJP as a uniform tensor with finitely many
exceptions, together with the dense program it agrees with. Its certificate preserves the scalar
program at every entry: adding two contributions neither discards zeros nor combines repeated
parents early.

## Main declarations

- `TorchLean.Tape.TensorLookup`, with `TensorLookup.ofPack`, `TensorLookup.toPack` and
  `TensorLookup.set`.
- `TorchLean.Tape.TensorContext`, with `TensorContext.ofPack`, `TensorContext.lookup`,
  `TensorContext.push`, `TensorContext.pop` and `TensorContext.toPack`.
- `TorchLean.Tape.Contributions`, with `Contributions.zero` and `Contributions.add`.
-/

@[expose] public section

namespace TorchLean.Tape

open TensorPack (getIdx)

/-- Tensor selection without committing a node to a linked pack representation. -/
@[ext] structure TensorLookup (α : Type) [Storage α] (shapes : List Shape) where
  /-- Select the tensor with the shape certified by its index. -/
  read : {shape : Shape} → Idx shapes shape → Tensor α shape

/-- Present a linked tensor pack through the selection interface. -/
def TensorLookup.ofPack {α : Type} [Storage α] {shapes : List Shape}
    (xs : TensorPack α shapes) : TensorLookup α shapes :=
  ⟨fun idx => getIdx xs idx⟩

/-- Typed packs can supply a node's lookup without changing tensor values. -/
instance {α : Type} [Storage α] {shapes : List Shape} :
    Coe (TensorPack α shapes) (TensorLookup α shapes) := ⟨TensorLookup.ofPack⟩

namespace TensorLookup

variable {α : Type} [Storage α]

/-- Materialize a reader at an explicit typed boundary. -/
def toPack : {shapes : List Shape} → TensorLookup α shapes → TensorPack α shapes
  | [], _ => .nil
  | _ :: shapes, lookup =>
      .cons (lookup.read ⟨0, rfl⟩)
        (toPack (shapes := shapes) ⟨fun idx => lookup.read ⟨idx.i.succ, idx.h⟩⟩)

/-- Reading the pack built by `toPack` at an index gives the tensor of the reader there. -/
@[simp] theorem getIdx_toPack {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) :
    getIdx lookup.toPack idx = lookup.read idx := by
  induction shapes with
  | nil => exact Fin.elim0 idx.i
  | cons head shapes ih =>
      obtain ⟨i, h⟩ := idx
      cases i using Fin.cases with
      | zero => cases h; rfl
      | succ i =>
          exact ih ⟨fun idx => lookup.read ⟨idx.i.succ, idx.h⟩⟩ ⟨i, h⟩

/-- Equal context positions certify equal tensor shapes. -/
theorem shape_eq_of_index_eq {shapes : List Shape} {source target : Shape}
    (left : Idx shapes source) (right : Idx shapes target)
    (same : left.i.val = right.i.val) : source = target :=
  left.h.symm.trans ((congrArg shapes.get (Fin.ext same)).trans right.h)

/-- Replace one typed entry of a reader. -/
def set {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) (value : Tensor α shape) :
    TensorLookup α shapes :=
  ⟨fun other =>
    if same : idx.i.val = other.i.val then
      Tensor.castShape value (shape_eq_of_index_eq idx other same)
    else
      lookup.read other⟩

/-- Reading an updated reader at the updated index gives `value`. -/
@[simp] theorem read_set_self {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape : Shape} (idx : Idx shapes shape) (value : Tensor α shape) :
    (lookup.set idx value).read idx = value := by
  simp only [set, dite_true]
  rfl

/-- Updating a reader at one index keeps its tensors at every other position. -/
@[simp] theorem read_set_other {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (value : Tensor α shape)
    (other : Idx shapes otherShape) (different : idx.i.val ≠ other.i.val) :
    (lookup.set idx value).read other = lookup.read other := by
  simp only [set, dite_eq_right different]

/-- Erasing shapes removes the transport in a reader update. -/
theorem ofTensor_read_set {shapes : List Shape} (lookup : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (value : Tensor α shape)
    (other : Idx shapes otherShape) :
    SomeTensor.ofTensor ((lookup.set idx value).read other) =
      if idx.i.val = other.i.val then SomeTensor.ofTensor value
      else SomeTensor.ofTensor (lookup.read other) := by
  by_cases same : idx.i.val = other.i.val
  · simp only [set, dite_eq_left same, ite_eq_left same, SomeTensor.ofTensor_castShape]
  · simp only [set, dite_eq_right same, ite_eq_right same]

/-- Updating from another reader copies exactly that reader's value at the selected position. -/
theorem read_set_from {shapes : List Shape} (lookup source : TensorLookup α shapes)
    {shape otherShape : Shape} (idx : Idx shapes shape) (other : Idx shapes otherShape) :
    (lookup.set idx (source.read idx)).read other =
      if other.i.val = idx.i.val then source.read other else lookup.read other := by
  by_cases same : other.i.val = idx.i.val
  · have shapes := shape_eq_of_index_eq idx other same.symm
    cases shapes
    have positions : idx = other := by
      cases idx
      cases other
      simp only [Idx.mk.injEq]
      exact Fin.ext same.symm
    subst other
    simp only [read_set_self, ite_true]
  · rw [read_set_other _ _ _ _ (Ne.symm same), ite_eq_right same]

end TensorLookup

/-- Runtime tensor references with an erased certificate of their shapes and order. -/
@[ext] structure TensorContext (α : Type) [Storage α] (shapes : List Shape) where
  /-- Inputs followed by node values in evaluation order. -/
  values : Array (SomeTensor α)
  /-- Every runtime entry has the shape at the same position in the typed context. -/
  valid : ∃ xs : TensorPack α shapes, values = xs.erase

namespace TensorContext

variable {α : Type} [Storage α] {shapes : List Shape}

/-- Convert input tensors once, before traversing the graph. -/
def ofPack (xs : TensorPack α shapes) : TensorContext α shapes :=
  ⟨xs.erase, xs, rfl⟩

/-- Runtime and typed contexts have the same number of entries. -/
theorem size_values (ctx : TensorContext α shapes) : ctx.values.size = shapes.length := by
  obtain ⟨xs, h⟩ := ctx.valid
  simp [h]

private theorem shape_get (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) :
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).shape = shape := by
  obtain ⟨xs, h⟩ := ctx.valid
  simpa only [h, TensorPack.get_erase,
    SomeTensor.shape_ofTensor] using idx.h

/-- Read a checked position in constant time. -/
@[no_expose] def lookup (ctx : TensorContext α shapes) : TensorLookup α shapes :=
  ⟨fun idx =>
    (ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt)).cast (shape_get ctx idx)⟩

/-- Array lookup agrees with the original typed pack lookup. -/
@[simp] theorem lookup_ofPack (xs : TensorPack α shapes) :
    (ofPack xs).lookup = TensorLookup.ofPack xs := by
  apply TensorLookup.ext
  funext shape idx
  rcases idx with ⟨i, h⟩
  simp only [lookup, ofPack, TensorLookup.ofPack, TensorPack.get_erase]
  cases h
  rfl

/-- Erasing the shape of a looked-up tensor gives the runtime array entry at the same position. -/
@[simp] theorem ofTensor_lookup (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) :
    SomeTensor.ofTensor (ctx.lookup.read idx) =
      ctx.values[idx.i.val]'(by rw [size_values]; exact idx.i.isLt) := by
  simp only [lookup, SomeTensor.ofTensor_cast]

/-- Update one tensor reference without copying the context when it is uniquely owned. -/
def set (ctx : TensorContext α shapes) {shape : Shape} (idx : Idx shapes shape)
    (value : Tensor α shape) : TensorContext α shapes :=
  ⟨ctx.values.set idx.i.val (SomeTensor.ofTensor value)
      (h := by rw [size_values]; exact idx.i.isLt), by
    refine ⟨(ctx.lookup.set idx value).toPack, ?_⟩
    apply Array.ext
    · simp only [Array.size_set, TensorPack.size_erase, size_values]
    · intro i leftBound rightBound
      have bound : i < ctx.values.size := by
        simpa only [Array.size_set] using leftBound
      let other : Idx shapes (shapes.get ⟨i, by
        simpa only [TensorPack.size_erase] using rightBound⟩) :=
        ⟨⟨i, by simpa only [TensorPack.size_erase] using rightBound⟩, rfl⟩
      rw [Array.getElem_set,
        TensorPack.get_erase _ other.i]
      change (if idx.i.val = i then SomeTensor.ofTensor value else ctx.values[i]) =
        SomeTensor.ofTensor (getIdx (ctx.lookup.set idx value).toPack other)
      rw [TensorLookup.getIdx_toPack, TensorLookup.ofTensor_read_set, ofTensor_lookup]
      ⟩

/-- The lookup of an updated context is the lookup updated at the same index. -/
@[simp] theorem lookup_set (ctx : TensorContext α shapes) {shape : Shape}
    (idx : Idx shapes shape) (value : Tensor α shape) :
    (ctx.set idx value).lookup = ctx.lookup.set idx value := by
  apply TensorLookup.ext
  funext otherShape other
  have same :
      SomeTensor.ofTensor ((ctx.set idx value).lookup.read other) =
        SomeTensor.ofTensor ((ctx.lookup.set idx value).read other) := by
    rw [ofTensor_lookup, TensorLookup.ofTensor_read_set, ofTensor_lookup]
    simp only [set, Array.getElem_set]
  simpa only [SomeTensor.ofTensor, SomeTensor.mk.injEq, heq_eq_eq, true_and] using same

/-- Transport the type index without changing the runtime array. -/
def cast {other : List Shape} (h : shapes = other) (ctx : TensorContext α shapes) :
    TensorContext α other :=
  h ▸ ctx

/-- Casting the context of a pack gives the context of the cast pack. -/
@[simp] theorem cast_ofPack {other : List Shape} (h : shapes = other)
    (xs : TensorPack α shapes) :
    cast h (ofPack xs) = ofPack (Pack.cast h xs) := by
  cases h
  rfl

/-- Append a tensor reference, amortized constant time when the context is uniquely owned.

Retaining a previous context or its lookup can make this append copy the shared array. -/
def push {shape : Shape} (ctx : TensorContext α shapes) (x : Tensor α shape) :
    TensorContext α (shapes ++ [shape]) :=
  ⟨ctx.values.push (SomeTensor.ofTensor x), by
    obtain ⟨xs, h⟩ := ctx.valid
    exact ⟨xs.snoc x, by simp [h]⟩⟩

/-- Pushing `x` onto the context of `xs` gives the context of `xs.snoc x`. -/
@[simp] theorem push_ofPack {shape : Shape} (xs : TensorPack α shapes)
    (x : Tensor α shape) :
    (ofPack xs).push x = ofPack (xs.snoc x) := by
  apply TensorContext.ext
  simp [push, ofPack]

private theorem shape_back {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    (ctx.values.back (by rw [size_values]; simp)).shape = shape := by
  obtain ⟨xs, same⟩ := ctx.valid
  have values : ctx.values = (TensorPack.erase xs.unsnoc.1).push
      (SomeTensor.ofTensor xs.unsnoc.2) := by
    rw [same, ← TensorPack.erase_snoc, Pack.snoc_unsnoc]
  simp only [values, Array.back, Array.size_push, Nat.add_sub_cancel,
    Array.getElem_push_eq, SomeTensor.shape_ofTensor]

/-- Remove the last reference in constant time, preserving the typed prefix. -/
@[no_expose] def pop {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    TensorContext α shapes × Tensor α shape :=
  (⟨ctx.values.pop, by
    obtain ⟨xs, same⟩ := ctx.valid
    refine ⟨xs.unsnoc.1, ?_⟩
    rw [same, ← Pack.snoc_unsnoc xs,
      TensorPack.erase_snoc, Array.pop_push]
    simp only [Pack.unsnoc_snoc]⟩,
   (ctx.values.back (by rw [size_values]; simp)).cast (shape_back ctx))

/-- Popping the context of `xs.snoc value` gives the context of `xs` and `value`. -/
@[simp] theorem pop_ofPack_snoc {shape : Shape} (xs : TensorPack α shapes)
    (value : Tensor α shape) :
    pop (ofPack (xs.snoc value)) = (ofPack xs, value) := by
  apply Prod.ext
  · apply TensorContext.ext
    simp only [pop, ofPack, TensorPack.erase_snoc, Array.pop_push]
  · simp only [pop, ofPack, TensorPack.erase_snoc,
      Array.back, Array.size_push, Nat.add_sub_cancel, Array.getElem_push_eq]
    rfl

/-- Popping the context of a pack splits off its last tensor as `Pack.unsnoc` does. -/
@[simp] theorem pop_ofPack {shape : Shape}
    (xs : TensorPack α (shapes ++ [shape])) :
    pop (ofPack xs) = (ofPack xs.unsnoc.1, xs.unsnoc.2) := by
  conv_lhs => rw [← Pack.snoc_unsnoc xs]
  exact pop_ofPack_snoc _ _

private theorem decode_ok (ctx : TensorContext α shapes) :
    ∃ xs, TensorPack.ofErased ctx.values (shapes := shapes) = .ok xs := by
  obtain ⟨xs, h⟩ := ctx.valid
  exact ⟨xs, by simp [h]⟩

/-- Recover the public typed result once at the execution boundary. -/
def toPack (ctx : TensorContext α shapes) : TensorPack α shapes :=
  match h : TensorPack.ofErased ctx.values (shapes := shapes) with
  | .ok xs => xs
  | .error _ => False.elim (by
      obtain ⟨xs, hx⟩ := decode_ok ctx
      rw [h] at hx
      contradiction)

/-- `toPack` recovers the pack that a context is built from. -/
@[simp] theorem toPack_ofPack (xs : TensorPack α shapes) :
    (ofPack xs).toPack = xs := by
  unfold toPack
  split
  next ys h =>
    simpa only [ofPack, TensorPack.ofErased_erase,
      Except.ok.injEq] using h.symm
  next msg h =>
    simp only [ofPack, TensorPack.ofErased_erase] at h
    cases h

/-- Rebuilding a context from its `toPack` gives the same context. -/
@[simp] theorem ofPack_toPack (ctx : TensorContext α shapes) :
    ofPack ctx.toPack = ctx := by
  obtain ⟨xs, values⟩ := ctx.valid
  have same : ctx = ofPack xs := TensorContext.ext values
  rw [same, toPack_ofPack]

/-- The reader of `ctx.toPack` is the lookup of `ctx`. -/
@[simp] theorem lookup_toPack (ctx : TensorContext α shapes) :
    TensorLookup.ofPack ctx.toPack = ctx.lookup := by
  rw [← lookup_ofPack, ofPack_toPack]

/-- Popping a context and converting the rest with `toPack` agrees with `Pack.unsnoc` of
`ctx.toPack`. -/
@[simp] theorem toPack_pop {shape : Shape} (ctx : TensorContext α (shapes ++ [shape])) :
    ((ctx.pop).1.toPack, (ctx.pop).2) = ctx.toPack.unsnoc := by
  conv_lhs => rw [← ofPack_toPack ctx]
  simp only [pop_ofPack, toPack_ofPack]

end TensorContext

/-- A typed position whose shape is available without traversing the context's shape list. -/
abbrev SomeIdx (shapes : List Shape) := (shape : Shape) × Idx shapes shape

namespace TensorLookup

variable {α : Type} [Storage α] {shapes : List Shape}

/-- A reader whose tensors all contain the same scalar. -/
def fill (value : α) : TensorLookup α shapes :=
  ⟨fun {shape} _ => Tensor.full shape value⟩

/-- Pointwise addition retains left-to-right scalar argument order. -/
def add [Add α] (left right : TensorLookup α shapes) : TensorLookup α shapes :=
  ⟨fun idx => left.read idx + right.read idx⟩

/-- The reader of a constant pack is the constant reader `fill value`. -/
@[simp] theorem ofPack_fill (value : α) :
    ofPack (TensorPack.fill value (ss := shapes)) = fill value := by
  apply TensorLookup.ext
  funext shape idx
  obtain ⟨i, rfl⟩ := idx
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      cases i using Fin.cases with
      | zero => rfl
      | succ i => exact ih i

/-- The reader of the zero pack is the constant reader `fill 0`. -/
@[simp] theorem ofPack_zero [Zero α] :
    ofPack (TensorPack.zero (α := α) (ss := shapes)) = fill 0 := by
  exact ofPack_fill 0

/-- The reader of the sum of two packs is the sum of their readers. -/
@[simp] theorem ofPack_add [Add α] (left right : TensorPack α shapes) :
    ofPack (TensorPack.add left right) = add (ofPack left) (ofPack right) := by
  apply TensorLookup.ext
  funext shape idx
  obtain ⟨i, rfl⟩ := idx
  induction shapes with
  | nil => exact Fin.elim0 i
  | cons head shapes ih =>
      cases left with
      | cons left lefts =>
        cases right with
        | cons right rights =>
          cases i using Fin.cases with
          | zero => rfl
          | succ i => exact ih lefts rights i

end TensorLookup

/-- A dense VJP program with a certified compact description of its entries. -/
structure Contributions (α : Type) [Storage α] (shapes : List Shape) where
  /-- The dense VJP program, a thunk that computes every entry of the pack. -/
  dense : Unit → TensorPack α shapes
  /-- Constant-time lookup for the small expression formed by one node's VJP. -/
  lookup : TensorLookup α shapes
  /-- Scalar present in tensors at positions outside `support`. -/
  uniform : α
  /-- Positions where the contribution can differ from the uniform tensor. -/
  support : List (SomeIdx shapes)
  /-- The compact reader computes exactly the dense VJP. -/
  ofPack_dense : TensorLookup.ofPack (dense ()) = lookup
  /-- Every omitted position really is uniform, without assuming any law about zero. -/
  outside : ∀ {shape} (idx : Idx shapes shape),
    (∀ entry ∈ support, entry.2.i.val ≠ idx.i.val) →
      lookup.read idx = Tensor.full shape uniform

namespace Contributions

variable {α : Type} [Storage α] {shapes : List Shape}

/-- The contribution of the zero pack, uniformly `0` with empty support. -/
def zero [Zero α] : Contributions α shapes :=
  { dense := fun _ => TensorPack.zero
    lookup := TensorLookup.fill 0
    uniform := 0
    support := []
    ofPack_dense := TensorLookup.ofPack_zero
    outside := fun _ _ => rfl }

private theorem add_full [Add α] (shape : Shape) (left right : α) :
    Tensor.full shape left + Tensor.full shape right =
      Tensor.full shape (left + right) := by
  apply Tensor.Rep.ext
  intro coordinate
  simp [Tensor.add_def]

/--
Add two VJP descriptions, `left` before `right`. Duplicate support entries remain: the reader
evaluates the complete local sum before the global gradient receives one contribution.
-/
def add [Add α] (left right : Contributions α shapes) : Contributions α shapes :=
  { dense := fun _ => TensorPack.add (left.dense ()) (right.dense ())
    lookup := TensorLookup.add left.lookup right.lookup
    uniform := left.uniform + right.uniform
    support := left.support ++ right.support
    ofPack_dense := by rw [TensorLookup.ofPack_add, left.ofPack_dense, right.ofPack_dense]
    outside := by
      intro shape idx absent
      change (left.lookup.read idx + right.lookup.read idx) = _
      rw [left.outside idx (fun entry member =>
        absent entry (List.mem_append_left _ member))]
      rw [right.outside idx (fun entry member =>
        absent entry (List.mem_append_right _ member))]
      exact add_full shape left.uniform right.uniform }

end Contributions

end TorchLean.Tape
