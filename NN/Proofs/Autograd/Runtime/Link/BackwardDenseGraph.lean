/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.BackwardDense
public import NN.Proofs.Autograd.Runtime.Link.FDeriv

/-!
# Executed Backward Pass on Lowered Graphs

`BackwardDense` shows that the executed sweep `Tape.backwardDenseAll` agrees with the proved sweep
`Tape.backwardDenseFrom` on every `ZeroPreserving` tape. This file discharges that hypothesis for
every tape produced by `lowerGraphToTape` and states the resulting corollaries for the executed
backward pass.

The zero-preservation proof rests on one algebraic fact: a proof-carrying `Node` satisfies the
adjointness law `dot (jvp x dx d) δ = dotList dx (vjp x d δ)`, and the pairing `TensorAlgebra.dot`
is nondegenerate over a commutative semiring (`eq_full_zero_of_forall_dot_eq_zero`). Hence
`vjp x d 0 = 0` for every node (`node_vjp_full_zero`), so a lowered node's `backward` sends its
zero cotangent to zero contributions of the parents' shapes (`lowerGraphToTape_zeroPreserving`).

Endpoints:

* `backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx`: the executed sweep on a lowered graph,
  seeded at any typed output index, returns `backpropAllCtx` of the one-hot seed context.
* `backwardDenseAll_lowerGraphToTape_adjoint_fderiv` (and its `_at` variant): over `ℝ`, the
  input block of that result is the adjoint of the Fréchet derivative of the forward map.

As in `FDeriv`, these are statements about the exact tape model at the given carrier; they say
nothing about `Float` rounding or the CUDA path.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Graph

open Runtime
open Runtime.Autograd

variable {α : Type} [TorchLean.Storage α]

/-! ### Nondegeneracy of the algebraic pairing -/

/-- The pairing with the all-zero tensor on the left is `0`. -/
theorem dot_full_zero_left [CommSemiring α] {s : Shape} (a : Tensor α s) :
    TensorAlgebra.dot (Tensor.full s (0 : α)) a = 0 := by
  rw [TensorAlgebra.dot_comm]
  exact TensorAlgebra.dot_full_zero_right a

/-- The pairing of a stacked tensor is the sum of the pairings of its slices. -/
theorem dot_dim_eq_sum [CommSemiring α] {n : Nat} {inner : Shape} (f : Fin n → Tensor α inner)
    (v : Tensor α (.dim n inner)) :
    TensorAlgebra.dot (Tensor.dim f) v = ∑ j, TensorAlgebra.dot (f j) (v.unstack j) := by
  simp only [TensorAlgebra.dot, Tensor.unstack_dim]
  exact List.finRange_foldl_add_eq_finset_sum _

/-- A tensor whose pairing with every tensor vanishes is the zero tensor. -/
theorem eq_full_zero_of_forall_dot_eq_zero [CommSemiring α] :
    ∀ {s : Shape} (v : Tensor α s),
      (∀ a : Tensor α s, TensorAlgebra.dot a v = 0) → v = Tensor.full s 0
  | .scalar, v, h => by
      apply Tensor.ext_scalar
      have h1 := h (Tensor.scalar 1)
      simp only [TensorAlgebra.dot, Tensor.item_scalar, one_mul] at h1
      rw [h1, Tensor.item_full_scalar]
  | .dim n inner, v, h => by
      have hslice : ∀ i : Fin n, v.unstack i = Tensor.full inner 0 := by
        intro i
        apply eq_full_zero_of_forall_dot_eq_zero
        intro b
        have hb := h (Tensor.dim fun j => if j = i then b else Tensor.full inner 0)
        rw [dot_dim_eq_sum] at hb
        have hterm : ∀ j : Fin n,
            TensorAlgebra.dot (if j = i then b else Tensor.full inner 0) (v.unstack j) =
              if j = i then TensorAlgebra.dot b (v.unstack i) else 0 := by
          intro j
          by_cases hj : j = i
          · subst hj
            simp
          · simp [hj, dot_full_zero_left]
        rw [Finset.sum_congr rfl (fun j _ => hterm j), Finset.sum_ite_eq'] at hb
        simpa using hb
      have hfill :
          Tensor.full (.dim n inner) (0 : α) =
            Tensor.dim fun _ => Tensor.full inner (0 : α) := by
        rw [← Tensor.dim_unstack (Tensor.full (.dim n inner) (0 : α))]
        congr 1
        funext i
        apply TorchLean.Tensor.Internal.Rep.ext
        intro c
        simp [Tensor.unstack, Tensor.full]
      rw [← Tensor.dim_unstack v, hfill]
      congr 1
      funext i
      exact hslice i

/-- `dotList` with the all-zero context on the left is `0`. -/
theorem dotList_zero_left [CommSemiring α] :
    ∀ {ss : List Shape} (v : TorchLean.TensorPack α ss),
      TensorPack.dotList (α := α) (TorchLean.TensorPack.zero (α := α) (ss := ss)) v = 0
  | [], .nil => by simp [TensorPack.dotList, TorchLean.TensorPack.zero]
  | _ :: ss, .cons vh vt => by
      simp [TensorPack.dotList, TorchLean.TensorPack.zero, dot_full_zero_left,
        dotList_zero_left vt]

/-- A context whose pairing with every context vanishes is the zero context. -/
theorem eq_zero_of_forall_dotList_eq_zero [CommSemiring α] :
    ∀ {ss : List Shape} (v : TorchLean.TensorPack α ss),
      (∀ dx : TorchLean.TensorPack α ss, TensorPack.dotList (α := α) dx v = 0) →
        v = TorchLean.TensorPack.zero
  | [], .nil, _ => rfl
  | s :: ss, .cons vh vt, h => by
      have hh : vh = Tensor.full s 0 := by
        apply eq_full_zero_of_forall_dot_eq_zero
        intro a
        simpa [TensorPack.dotList, dotList_zero_left] using
          h (.cons a TorchLean.TensorPack.zero)
      have ht : vt = TorchLean.TensorPack.zero := by
        apply eq_zero_of_forall_dotList_eq_zero
        intro dx
        simpa [TensorPack.dotList, dot_full_zero_left] using h (.cons (Tensor.full s 0) dx)
      rw [hh, ht]
      rfl

/--
A proof-carrying node's VJP sends the zero cotangent to the zero context. This follows from the
adjointness law alone: `dotList dx (vjp x d 0) = dot (jvp x dx d) 0 = 0` for every `dx`.
-/
theorem node_vjp_full_zero [CommSemiring α] {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : Algebra.Node (α := α) Δ Γ τ) (x : TorchLean.TensorPack α Γ) (d : Δ) :
    node.vjp x d (Tensor.full τ 0) = TorchLean.TensorPack.zero := by
  apply eq_zero_of_forall_dotList_eq_zero
  intro dx
  rw [← node.correct x dx d (Tensor.full τ 0)]
  exact TensorAlgebra.dot_full_zero_right _

/-! ### Shape erasure of the zero context and of `single` -/

/-- Entries of the erased zero context are zero tensors of the recorded shapes. -/
theorem getElem?_toShapeErasedArray_zero [Zero α] :
    ∀ {ss : List Shape} (j : Nat),
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
        (TorchLean.TensorPack.zero (α := α) (ss := ss)))[j]? =
        (ss[j]?).map fun s => Spec.SomeTensor.ofTensor (Tensor.full s (0 : α))
  | [], j => by
      simp [TorchLean.TensorPack.toShapeErasedArray, TorchLean.TensorPack.zero]
  | _ :: _, 0 => by
      simp [TorchLean.TensorPack.toShapeErasedArray, TorchLean.TensorPack.zero,
        -Spec.SomeTensor.ofTensor]
  | _ :: ss, j + 1 => by
      simp [TorchLean.TensorPack.toShapeErasedArray, TorchLean.TensorPack.zero,
        Array.getElem?_append, getElem?_toShapeErasedArray_zero (ss := ss) j,
        -Spec.SomeTensor.ofTensor]

/-- Every indexed contribution of the zero context is a zero tensor at a valid position. -/
theorem mem_toIndexedShapeErasedArray_zero [Zero α] :
    ∀ {ss : List Shape} (start : Nat) {pid : Nat} {pg : Spec.SomeTensor α},
      (pid, pg) ∈ TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := ss)
        (TorchLean.TensorPack.zero (α := α) (ss := ss)) start →
      ∃ (i : Nat) (hi : i < ss.length),
        pid = start + i ∧ pg = Spec.SomeTensor.ofTensor (Tensor.full (ss[i]'hi) (0 : α))
  | [], start, pid, pg, hmem => by
      simp [TorchLean.TensorPack.toIndexedShapeErasedArray,
        TorchLean.TensorPack.zero] at hmem
  | s :: ss, start, pid, pg, hmem => by
      simp only [TorchLean.TensorPack.toIndexedShapeErasedArray,
        TorchLean.TensorPack.zero, Array.mem_append, Array.mem_singleton,
        Prod.mk.injEq] at hmem
      rcases hmem with ⟨hpid, hpg⟩ | hmem
      · exact ⟨0, by simp, by simp [hpid], hpg⟩
      · obtain ⟨i, hi, hpid, hpg⟩ := mem_toIndexedShapeErasedArray_zero (ss := ss) (start + 1) hmem
        exact ⟨i + 1, by simpa using Nat.succ_lt_succ hi,
          by rw [hpid, Nat.add_assoc, Nat.add_comm 1 i], hpg⟩

/-- The shape recorded at position `j` of an erased context is `ss[j]`. -/
theorem shape_getElem?_toShapeErasedArray :
    ∀ {ss : List Shape} (xs : TorchLean.TensorPack α ss) (j : Nat),
      ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) xs)[j]?).map
          Spec.SomeTensor.shape = ss[j]?
  | [], .nil, j => by simp [TorchLean.TensorPack.toShapeErasedArray]
  | _ :: _, .cons _ _, 0 => by simp [TorchLean.TensorPack.toShapeErasedArray]
  | _ :: ss, .cons _ xs, j + 1 => by
      simp [TorchLean.TensorPack.toShapeErasedArray, Array.getElem?_append,
        shape_getElem?_toShapeErasedArray (ss := ss) xs j]

/-- Erasing `single idx v`: `v` at `idx`, zero tensors of the recorded shapes elsewhere. -/
theorem getElem?_toShapeErasedArray_single [Zero α] :
    ∀ {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor α s) (j : Nat),
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ)
        (TensorPack.single idx v))[j]? =
        if j = idx.i.1 then some (Spec.SomeTensor.ofTensor v)
        else (Γ[j]?).map fun sh => Spec.SomeTensor.ofTensor (Tensor.full sh (0 : α))
  | [], _, ⟨i, _⟩, _, _ => (Nat.not_lt_zero _ i.2).elim
  | _ :: _, _, ⟨⟨0, _⟩, _⟩, v, 0 => by
      simp [TensorPack.single, TorchLean.TensorPack.toShapeErasedArray,
        -Spec.SomeTensor.ofTensor]
  | _ :: Γ, _, ⟨⟨0, _⟩, _⟩, v, j + 1 => by
      simp [TensorPack.single, TorchLean.TensorPack.toShapeErasedArray,
        Array.getElem?_append, getElem?_toShapeErasedArray_zero (ss := Γ) j,
        -Spec.SomeTensor.ofTensor]
  | _ :: _, _, ⟨⟨_ + 1, _⟩, _⟩, v, 0 => by
      simp [TensorPack.single, TorchLean.TensorPack.toShapeErasedArray,
        -Spec.SomeTensor.ofTensor]
  | _ :: Γ, s, ⟨⟨j' + 1, hi⟩, h⟩, v, j + 1 => by
      have ih := getElem?_toShapeErasedArray_single (Γ := Γ) (s := s)
        ⟨⟨j', Nat.lt_of_succ_lt_succ hi⟩, by simpa using h⟩ v j
      simp [TensorPack.single, TorchLean.TensorPack.toShapeErasedArray,
        Array.getElem?_append, ih, -Spec.SomeTensor.ofTensor]

/-! ### Lowered tapes are zero preserving -/

/-- The value stored at node `j` of a lowered tape has the shape recorded at `j` in `Γ ++ ss`. -/
theorem lowerGraphToTape_getElem?_value_shape [CommSemiring α] {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d0 : Δ) (j : Nat) :
    ((lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes[j]?).map
        (fun node => node.value.shape) = (Γ ++ ss)[j]? := by
  have hmap := congrArg (fun arr : Array (Spec.SomeTensor α) => (arr[j]?).map Spec.SomeTensor.shape)
    (lowerGraphToTape_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0)
  simpa [Array.getElem?_map, Option.map_map, Function.comp_def,
    shape_getElem?_toShapeErasedArray] using hmap

/--
Every tape produced by `lowerGraphToTape` is `ZeroPreserving`: leaves have no contributions, and
a lowered node's `backward` on its zero cotangent runs the stored VJP at zero, which is the zero
context by `node_vjp_full_zero`, so every emitted contribution is a parent's zero cotangent.
-/
theorem lowerGraphToTape_zeroPreserving [CommSemiring α] {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d0 : Δ) :
    ZeroPreserving (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 := by
  induction g with
  | nil =>
    refine ⟨fun id n hn _ => ?_⟩
    have hn' : ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
        (leafNodeOfSomeTensor (α := α)))[id]? = some n := by
      simpa [lowerGraphToTape, Tape.getNode?, nodes_addLeaves, Tape.empty] using hn
    rw [Array.getElem?_map] at hn'
    cases hx : (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[id]? with
    | none => simp [hx] at hn'
    | some v =>
      rw [hx, Option.map_some, Option.some.injEq] at hn'
      subst hn'
      exact ⟨#[], rfl, fun pid pg hmem => by simp at hmem⟩
  | snoc g node ih =>
    rename_i ssPrev τ
    refine ⟨fun id n hn hreq => ?_⟩
    let prev := lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
    let tPrev := prev.1
    let ctxPrev := prev.2
    let runtimeNode : Runtime.Autograd.Node α :=
      lowerNode (some "proof-carrying-graph") node.toNodeData ctxPrev d0
    have hnNodes : (tPrev.nodes.push runtimeNode)[id]? = some n := by
      simpa [lowerGraphToTape, prev, tPrev, runtimeNode, Tape.getNode?, Tape.addNode]
        using hn
    have hsizePrev : tPrev.nodes.size = (Γ ++ ssPrev).length := by
      simp [tPrev, prev, lowerGraphToTape_nodes_size, List.length_append]
    have hgetPrev : ∀ pid, pid < tPrev.nodes.size →
        (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc g node) x
          d0).1.getNode? pid = tPrev.getNode? pid := by
      intro pid hpid
      simp [lowerGraphToTape, prev, tPrev, Tape.getNode?, Tape.addNode,
        Array.getElem?_push, Nat.ne_of_lt hpid]
    by_cases hlast : id = tPrev.nodes.size
    · subst hlast
      have hnEq : n = runtimeNode := by
        symm
        simpa [Array.getElem?_push] using hnNodes
      subst hnEq
      have hτ : (zeroCotangent runtimeNode).shape = τ := by
        simp [runtimeNode, zeroCotangent, lowerNode]
      refine ⟨_, lowerNode_backward_of_shape _ _ _ _ _ hτ, fun pid pg hmem => ?_⟩
      have hcast : (zeroCotangent runtimeNode).cast hτ = Tensor.full τ (0 : α) := rfl
      rw [hcast, node_vjp_full_zero node ctxPrev d0] at hmem
      obtain ⟨i, hi, hpid, hpg⟩ := mem_toIndexedShapeErasedArray_zero (ss := Γ ++ ssPrev) 0 hmem
      have hi' : i < tPrev.nodes.size := by
        rw [hsizePrev]
        exact hi
      refine ⟨tPrev.nodes[i], ?_, ?_⟩
      · rw [hpid, Nat.zero_add, hgetPrev i hi']
        simp [Tape.getNode?, Array.getElem?_eq_getElem hi']
      · have hshape := lowerGraphToTape_getElem?_value_shape (α := α) g x d0 i
        have hi'' : i < (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
          d0).1.nodes.size := hi'
        rw [Array.getElem?_eq_getElem hi'', Option.map_some, List.getElem?_eq_getElem hi,
          Option.some.injEq] at hshape
        have hshape' : (tPrev.nodes[i]'hi').value.shape = (Γ ++ ssPrev)[i]'hi := hshape
        rw [hpg]
        exact congrArg (fun sh => Spec.SomeTensor.ofTensor (Tensor.full sh (0 : α))) hshape'.symm
    · have hidPrev : id < tPrev.nodes.size := by
        have hlt : id < tPrev.nodes.size + 1 := by
          have := (Array.getElem?_eq_some_iff.1 hnNodes).1
          simpa [Array.size_push] using this
        exact Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) hlast
      have hnPrev : tPrev.getNode? id = some n := by
        rw [← hgetPrev id hidPrev]
        exact hn
      obtain ⟨contribs, hback, hz⟩ := ih.backward_zero id n hnPrev hreq
      refine ⟨contribs, hback, fun pid pg hmem => ?_⟩
      obtain ⟨pnode, hp, hpg⟩ := hz pid pg hmem
      refine ⟨pnode, ?_, hpg⟩
      rw [hgetPrev pid (getNode?_lt_size hp)]
      exact hp

/-! ### The one-hot seed array is the erased `single` context -/

/-- On a lowered tape, `oneHotGrads` at a typed index is the erasure of `TensorPack.single`. -/
theorem oneHotGrads_lowerGraphToTape_eq_single [CommSemiring α] {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d0 : Δ) {τ : Shape}
    (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    oneHotGrads (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 output.i.1
        (Spec.SomeTensor.ofTensor seed) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (TensorPack.single output seed) := by
  apply Array.ext_getElem?
  intro j
  rw [getElem?_oneHotGrads, getElem?_toShapeErasedArray_single]
  have hshape := lowerGraphToTape_getElem?_value_shape (α := α) g x d0 j
  have hlt : output.i.1 < (Γ ++ ss).length := output.i.2
  simp only [Tape.getNode?]
  cases hnode : (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes[j]? with
  | none =>
    rw [hnode, Option.map_none] at hshape
    have hj : j ≠ output.i.1 := by
      intro hj
      rw [hj, List.getElem?_eq_getElem hlt] at hshape
      simp at hshape
    simp [hj, ← hshape]
  | some node =>
    rw [hnode, Option.map_some] at hshape
    by_cases hj : j = output.i.1
    · simp [hj]
    · simp only [Option.map_some, if_neg hj, ← hshape, zeroCotangent]

/-! ### Corollaries for the executed backward pass -/

/--
**Executed backward pass on a lowered graph = proved backpropagation.** Running the trainer's
`Tape.backwardDenseAll` on the tape produced by `lowerGraphToTape`, seeded at any typed output
index `output` with cotangent `seed`, succeeds and returns the shape erasure of `backpropAllCtx`
of the one-hot seed context. Unlike `backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx`, this
is about the variant that skips unreached nodes; the two agree by `lowerGraphToTape_zeroPreserving`.
-/
theorem backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx [CommSemiring α] {Δ : Type}
    {Γ ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TorchLean.TensorPack α Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    Runtime.Autograd.Tape.backwardDenseAll
        (t := (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1) output.i.1
        (Spec.SomeTensor.ofTensor seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
          (TensorPack.single output seed))) := by
  have hlt : output.i.1 <
      (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes.size := by
    rw [lowerGraphToTape_nodes_size]
    simpa [List.length_append] using output.i.2
  obtain ⟨outNode, hout⟩ := getNode?_of_lt_size hlt
  have hseed : (Spec.SomeTensor.ofTensor seed).shape = outNode.value.shape := by
    have hshape := lowerGraphToTape_getElem?_value_shape (α := α) g x d0 output.i.1
    have hout' : (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x
        d0).1.nodes[output.i.1]? = some outNode := hout
    rw [hout', Option.map_some, List.getElem?_eq_getElem output.i.2, Option.some.injEq] at hshape
    have hτ := output.h
    rw [List.get_eq_getElem] at hτ
    rw [Spec.SomeTensor.shape_ofTensor, hshape]
    exact hτ.symm
  rw [backwardDenseAll_eq_backwardDenseFrom (lowerGraphToTape_zeroPreserving g x d0) hout hseed,
    oneHotGrads_lowerGraphToTape_eq_single]
  exact backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx (α := α) g x d0
    (TensorPack.single output seed)

/--
**Executed backward pass = adjoint of the Fréchet derivative.** Over `ℝ`, the trainer's
`Tape.backwardDenseAll` on a lowered graph, seeded at a typed output index, succeeds with the full
backpropagation context, whose input (`Γ`-prefix) block is the adjoint of the Fréchet derivative
of the graph's forward evaluation applied to the one-hot seed. This composes
`backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx` with
`backwardDenseFrom_lowerGraphToTape_adjoint_fderiv`.
-/
theorem backwardDenseAll_lowerGraphToTape_adjoint_fderiv {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor ℝ τ)
    (hg : GraphFDerivCorrect (Γ := Γ) (toReal g d0)) :
    Runtime.Autograd.Tape.backwardDenseAll
        (t := (lowerGraphToTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1) output.i.1
        (Spec.SomeTensor.ofTensor seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss)
        (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
          (TensorPack.single output seed)))
    ∧ flattenCtx (TensorPack.takeLeft
        (backpropAllCtx (α := ℝ) g x d0 (TensorPack.single output seed)))
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx (TensorPack.single output seed)) :=
  ⟨backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx g x d0 output seed,
    (backwardDenseFrom_lowerGraphToTape_adjoint_fderiv g x d0 (TensorPack.single output seed)
      hg).2⟩

/-- Pointwise variant of `backwardDenseAll_lowerGraphToTape_adjoint_fderiv`. -/
theorem backwardDenseAll_lowerGraphToTape_adjoint_fderiv_at {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TorchLean.TensorPack ℝ Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor ℝ τ)
    (hg : GraphFDerivCorrectAt (Γ := Γ) (toReal g d0) (flattenCtx x)) :
    Runtime.Autograd.Tape.backwardDenseAll
        (t := (lowerGraphToTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1) output.i.1
        (Spec.SomeTensor.ofTensor seed) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := ℝ) (ss := Γ ++ ss)
        (backpropAllCtx (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0
          (TensorPack.single output seed)))
    ∧ flattenCtx (TensorPack.takeLeft
        (backpropAllCtx (α := ℝ) g x d0 (TensorPack.single output seed)))
      = (fderiv ℝ (Proofs.Autograd.Graph.evalVec (toReal g d0))
          (flattenCtx x)).adjoint (flattenCtx (TensorPack.single output seed)) :=
  ⟨backwardDenseAll_lowerGraphToTape_eq_backpropAllCtx g x d0 output seed,
    (backwardDenseFrom_lowerGraphToTape_adjoint_fderiv_at g x d0 (TensorPack.single output seed)
      hg).2⟩

end Graph

end Algebra
end Autograd
end Proofs
