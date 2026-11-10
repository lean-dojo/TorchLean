/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the data movements and indexing

Each lemma here states one arm of `Op.rule?` for a data movement (`reshape`, `permute`, `broadcast`,
`flip`, `slice`, `pad`, `indexAt`, `cat`, `indices`) or an indexing operator (`gather`, `scatter`,
`embedding`, `fullLike`). Where an operator takes an element (`Op.Element`), its Boolean and integer
forms have lemmas of their own (`Op.rule?_reshape_bool`, `Op.rule?_reshape_integer`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.gather`. -/
theorem rule?_gather [Op.Capabilities α] {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims) :
    (Op.gather dims batchDims h).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.gather xs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h, ())
        backward := fun xs g =>
          (Tensor.scatter .sum (Tensor.full _ 0) (dims ++ batchDims.map Prod.fst)
            (Op.gatherSources h (Tensor.map (Op.axisTuples _ dims) xs.2.1)) g.1, (), ())
        jvp? := some fun xs dxs =>
          (Tensor.gather dxs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h,
            ()) } :=
  rfl

/-- The rule of `Op.gather` at Booleans. -/
theorem rule?_gather_bool [Op.Capabilities α] {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims) :
    (Op.gather dims batchDims h .bool).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.gather xs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.gather` at integer words. -/
theorem rule?_gather_integer [Op.Capabilities α] {s t : Shape} (dims : List (Fin s.rank))
    (batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.GatherValid s t dims batchDims)
    {signed : Bool} {width : Nat} :
    (Op.gather dims batchDims h (.integer signed width)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.gather xs.1 dims batchDims (Tensor.map (Op.axisTuples _ dims) xs.2.1) _ h, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.reshape`. -/
theorem rule?_reshape [Op.Capabilities α] {s t : Shape} (h : s.size = t.size) :
    (Op.reshape h).rule? (α := α) = some
      { forward := fun xs => (Tensor.reshape (source := s) xs.1 t h, ())
        backward := fun _ g => (Tensor.reshape (source := t) g.1 s h.symm, ())
        jvp? := some fun _ dxs => (Tensor.reshape (source := s) dxs.1 t h, ()) } :=
  rfl

/-- The rule of `Op.reshape` at Booleans. -/
theorem rule?_reshape_bool [Op.Capabilities α] {s t : Shape} (h : s.size = t.size) :
    (Op.reshape h .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.reshape (source := s) xs.1 t h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.reshape` at integer words. -/
theorem rule?_reshape_integer [Op.Capabilities α] {s t : Shape} (h : s.size = t.size)
    {signed : Bool} {width : Nat} :
    (Op.reshape h (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.reshape (source := s) xs.1 t h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.permute`. -/
theorem rule?_permute [Op.Capabilities α] {s : Shape} (perm : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute perm) :
    (Op.permute perm out hout).rule? (α := α) = some
      { forward := fun xs => (Tensor.permute xs.1 perm out hout, ())
        backward := fun _ g =>
          (Tensor.permute g.1 (s.unpermute perm hout) s (s.permute_unpermute perm hout).symm, ())
        jvp? := some fun _ dxs => (Tensor.permute dxs.1 perm out hout, ()) } :=
  rfl

/-- The rule of `Op.permute` at Booleans. -/
theorem rule?_permute_bool [Op.Capabilities α] {s : Shape} (perm : Equiv.Perm (Fin s.rank))
    (out : Shape) (hout : out = s.permute perm) :
    (Op.permute perm out hout .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.permute xs.1 perm out hout, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.permute` at integer words. -/
theorem rule?_permute_integer [Op.Capabilities α] {s : Shape} (perm : Equiv.Perm (Fin s.rank))
    (out : Shape) (hout : out = s.permute perm) {signed : Bool} {width : Nat} :
    (Op.permute perm out hout (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.permute xs.1 perm out hout, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.broadcast`. -/
theorem rule?_broadcast [Op.Capabilities α] {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) :
    (Op.broadcast dims h).rule? (α := α) = some
      { forward := fun xs => (Tensor.broadcast xs.1 _ dims h, ())
        backward := fun _ g => (Tensor.reduceBroadcast dims h g.1, ())
        jvp? := some fun _ dxs => (Tensor.broadcast dxs.1 _ dims h, ()) } :=
  rfl

/-- The rule of `Op.broadcast` at Booleans. -/
theorem rule?_broadcast_bool [Op.Capabilities α] {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) :
    (Op.broadcast dims h .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.broadcast xs.1 _ dims h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.broadcast` at integer words. -/
theorem rule?_broadcast_integer [Op.Capabilities α] {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) {signed : Bool} {width : Nat} :
    (Op.broadcast dims h (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.broadcast xs.1 _ dims h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.fullLike`. -/
theorem rule?_fullLike [Op.Capabilities α] {s : Shape} :
    (Op.fullLike (s := s)).rule? (α := α) = some
      { forward := fun xs => (Tensor.full s xs.2.1.item, ())
        backward := fun xs _ => (Tensor.map (fun _ => 0) xs.1, (), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.embedding`. -/
theorem rule?_embedding [Op.Capabilities α] {num dim : Nat} {t : Shape} :
    (Op.embedding (num := num) (dim := dim) (t := t)).rule? (α := α) = some
      { forward := fun xs => (Tensor.embedding xs.1 xs.2.1, ())
        backward := fun xs g => (Tensor.embeddingBackward xs.2.1 g.1, (), ())
        jvp? := some fun xs dxs => (Tensor.embedding dxs.1 xs.2.1, ()) } :=
  rfl

/-- The rule of `Op.flip`. -/
theorem rule?_flip [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank)) :
    (Op.flip dims).rule? (α := α) = some
      { forward := fun xs => (Tensor.flip xs.1 dims, ())
        backward := fun _ g => (Tensor.flip g.1 dims, ())
        jvp? := some fun _ dxs => (Tensor.flip dxs.1 dims, ()) } :=
  rfl

/-- The rule of `Op.flip` at Booleans. -/
theorem rule?_flip_bool [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank)) :
    (Op.flip dims .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.flip xs.1 dims, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.flip` at integer words. -/
theorem rule?_flip_integer [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank))
    {signed : Bool} {width : Nat} :
    (Op.flip dims (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.flip xs.1 dims, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.indices`. -/
theorem rule?_indices [Op.Capabilities α] (s : Shape) (dim : Fin s.rank) :
    (Op.indices s dim).rule? (α := α) = some
      { forward := fun _ => (Tensor.indices s dim, ())
        backward := fun _ _ => () } :=
  rfl

/-- The rule of `Op.slice`. -/
theorem rule?_slice [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank))
    (starts stops steps : List Nat) (h : Shape.SliceValid s dims starts stops steps) :
    (Op.slice dims starts stops steps h).rule? (α := α) = some
      { forward := fun xs => (Tensor.slice xs.1 dims starts stops steps h, ())
        backward := fun _ g => (Tensor.sliceBackward dims starts stops steps g.1, ())
        jvp? := some fun _ dxs => (Tensor.slice dxs.1 dims starts stops steps h, ()) } :=
  rfl

/-- The rule of `Op.slice` at Booleans. -/
theorem rule?_slice_bool [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank))
    (starts stops steps : List Nat) (h : Shape.SliceValid s dims starts stops steps) :
    (Op.slice dims starts stops steps h .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.slice xs.1 dims starts stops steps h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.slice` at integer words. -/
theorem rule?_slice_integer [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank))
    (starts stops steps : List Nat) (h : Shape.SliceValid s dims starts stops steps)
    {signed : Bool} {width : Nat} :
    (Op.slice dims starts stops steps h (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.slice xs.1 dims starts stops steps h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.pad`. -/
theorem rule?_pad [Op.Capabilities α] {s : Shape} (low high interior : Fin s.rank → Nat) :
    (Op.pad low high interior).rule? (α := α) = some
      { forward := fun xs => (Tensor.pad xs.1 low high interior xs.2.1.item, ())
        backward := fun _ g =>
          (Tensor.padBackward low high interior g.1,
            Tensor.scalar (Tensor.sum (Tensor.select
              (Tensor.pad (Tensor.full s false) low high interior true) g.1 (Tensor.full _ 0))),
            ())
        jvp? := some fun _ dxs => (Tensor.pad dxs.1 low high interior dxs.2.1.item, ()) } :=
  rfl

/-- The rule of `Op.pad` at Booleans. -/
theorem rule?_pad_bool [Op.Capabilities α] {s : Shape} (low high interior : Fin s.rank → Nat) :
    (Op.pad low high interior .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.pad xs.1 low high interior xs.2.1.item, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.pad` at integer words. -/
theorem rule?_pad_integer [Op.Capabilities α] {s : Shape} (low high interior : Fin s.rank → Nat)
    {signed : Bool} {width : Nat} :
    (Op.pad low high interior (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Tensor.pad xs.1 low high interior xs.2.1.item, ())
        backward := fun _ _ => ((), (), ()) } :=
  rfl

/-- The rule of `Op.indexAt`. -/
theorem rule?_indexAt [Op.Capabilities α] {s : Shape} (dim : Fin s.rank)
    (index : Fin (s.extent dim)) :
    (Op.indexAt dim index).rule? (α := α) = some
      { forward := fun xs => (Tensor.reshape
            (Tensor.slice xs.1 [dim] [index.val] [index.val + 1] [1]
              (Shape.sliceValid_singleton _ dim index.val 1 index.isLt)) _
            (by rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ dim), ())
        backward := fun _ g => (Tensor.sliceBackward [dim] [index.val] [index.val + 1] [1]
            (Tensor.reshape g.1 _ (by
              rw [Shape.slice_singleton]
              exact (Tensor.size_shapeAfterSumKeepDim _ dim).symm)), ())
        jvp? := some fun _ dxs => (Tensor.reshape
            (Tensor.slice dxs.1 [dim] [index.val] [index.val + 1] [1]
              (Shape.sliceValid_singleton _ dim index.val 1 index.isLt)) _
            (by rw [Shape.slice_singleton]; exact Tensor.size_shapeAfterSumKeepDim _ dim), ()) } :=
  rfl

/-- The rule of `Op.cat`. -/
theorem rule?_cat [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (sizes : List Nat) :
    (Op.cat dim sizes).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.cat dim sizes.get (Op.catElementParts dim sizes xs) _ (by simp), ())
        backward := fun _ g => Op.catGrads dim g.1 0 sizes (by simp)
        jvp? := some fun _ dxs =>
          (Tensor.cat dim sizes.get (Op.catTangentParts dim sizes dxs) _ (by simp), ()) } :=
  rfl

/-- The rule of `Op.cat` at Booleans. -/
theorem rule?_cat_bool [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (sizes : List Nat) :
    (Op.cat dim sizes .bool).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.cat dim sizes.get (Op.catElementParts dim sizes xs) _ (by simp), ())
        backward := fun _ _ => Op.catUnits dim (fun _ => ()) sizes } :=
  rfl

/-- The rule of `Op.cat` at integer words. -/
theorem rule?_cat_integer [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (sizes : List Nat)
    {signed : Bool} {width : Nat} :
    (Op.cat dim sizes (.integer signed width)).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.cat dim sizes.get (Op.catElementParts dim sizes xs) _ (by simp), ())
        backward := fun _ _ => Op.catUnits dim (fun _ => ()) sizes } :=
  rfl

/-- The rule of `Op.scatter`. -/
theorem rule?_scatter [Op.Capabilities α] {s t : Shape} (rule : Tensor.ScatterReduce)
    (dims : List (Fin s.rank)) (h : dims.Nodup) :
    (Op.scatter (t := t) rule dims h).rule? (α := α) = some
      { forward := fun xs => (Op.scatterForward rule dims xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun xs g =>
          let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
          match rule with
          | .sum => (g.1, (),
              Tensor.scatterUpdatesBackward .sum dims h (Tensor.map (Op.axisTuples _ dims) xs.2.1)
                g.1, ())
          | .replace => (Tensor.scatter .replace g.1 dims positions (Tensor.full _ 0), (),
              Tensor.scatterUpdatesBackward .replace dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) g.1, ())
          | .prod =>
              let gradients := Tensor.scatterProdBackward xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
          | .max =>
              let gradients := Tensor.scatterExtremumBackward
                (Tensor.scatter .max xs.1 dims positions xs.2.2.1) xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
          | .min =>
              let gradients := Tensor.scatterExtremumBackward
                (Tensor.scatter .min xs.1 dims positions xs.2.2.1) xs.1 dims h
                (Tensor.map (Op.axisTuples _ dims) xs.2.1) xs.2.2.1 g.1
              (gradients.1, (), gradients.2, ())
        jvp? := match rule with
          | .sum => some fun xs dxs =>
              let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
              (Tensor.scatter .sum dxs.1 dims positions dxs.2.2.1, ())
          | .replace => some fun xs dxs =>
              let positions := Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1
              (Tensor.scatter .replace dxs.1 dims positions dxs.2.2.1, ())
          | _ => none } :=
  rfl

/-- The rule of `Op.scatter` at Booleans by replacement. -/
theorem rule?_scatter_bool [Op.Capabilities α] {s t : Shape} (dims : List (Fin s.rank))
    (h : dims.Nodup) :
    (Op.scatter (t := t) .replace dims h .bool).rule? (α := α) = some
      { forward := fun xs => (Tensor.scatter .replace xs.1 dims
          (Tensor.map (fun i => some (Op.axisTuples _ dims i)) xs.2.1) xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) } :=
  rfl

/-- A Boolean `Op.scatter` that does not replace has no rule. -/
theorem rule?_scatter_bool_eq_none [Op.Capabilities α] {s t : Shape} (rule : Tensor.ScatterReduce)
    (dims : List (Fin s.rank)) (h : dims.Nodup) (hrule : rule ≠ .replace) :
    (Op.scatter (t := t) rule dims h .bool).rule? (α := α) = none := by
  cases rule <;> first | rfl | exact absurd rfl hrule

/-- The rule of `Op.scatter` at integer words: wrapping sums and products, and the maximum and
minimum in the type's order. -/
theorem rule?_scatter_integer [Op.Capabilities α] {s t : Shape} (rule : Tensor.ScatterReduce)
    (dims : List (Fin s.rank)) (h : dims.Nodup) {signed : Bool} {width : Nat} :
    (Op.scatter (t := t) rule dims h (.integer signed width)).rule? (α := α) = some
      { forward := fun xs => (Op.scatterForward (maxStep := ⟨Word.maximum signed⟩)
          (minStep := ⟨Word.minimum signed⟩) rule dims xs.1 xs.2.1 xs.2.2.1, ())
        backward := fun _ _ => ((), (), (), ()) } :=
  rfl

end Op

end TorchLean
