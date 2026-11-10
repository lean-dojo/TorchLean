/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering.BackwardDense
public import TorchLean.NN.Tape.Lowering.FDeriv

/-!
# Executed Backward Pass on Lowered Graphs

`BackwardDense` shows that, over an `AddZeroClass` scalar, the executed sweep
`Tape.backwardDenseAll` agrees with the proved sweep `Tape.backwardDenseFrom` on every
`ZeroPreserving` tape. The theorems here discharge that hypothesis for the lowered tape of every
proof-carrying graph (`Tape.GraphData.toTape` at `g.toData`) over a commutative semiring and state
the resulting corollaries for `Tape.backwardDenseAll` on those tapes. No runtime path sweeps a
lowered tape. The eager trainer runs `Tape.backwardDenseAll` on its session tape, which these
theorems do not cover.

The zero-preservation proof rests on one algebraic fact: a proof-carrying `Tape.LawfulNode`
satisfies the adjointness law `dot (jvp x dx d) δ = dotList dx (vjp x d δ)`, and the pairing `dot`
is nondegenerate over a commutative semiring (`eq_full_zero_of_forall_dot_eq_zero`). Hence
`vjp x d 0 = 0` for every node (`Tape.LawfulNode.vjp_full_zero`), so a lowered node's `backward`
sends its zero cotangent to zero contributions of the parents' shapes
(`Tape.Graph.zeroPreserving_toTape`).

Endpoints:

* `Tape.Graph.backwardDenseAll_toTape`: the executed sweep on a lowered graph,
  seeded at any typed output index, returns `GraphData.backpropAllCtx` of the one-hot seed context.
* `Tape.Graph.backwardDenseAll_eq_fderiv`: over `ℝ`, at an input where the graph's
  nodes are differentiable, the input block of that result is the adjoint of the Fréchet
  derivative of the forward map.

As in `FDeriv`, both are statements about the exact tape model at the given carrier and cover
neither `Float` rounding nor the CUDA path.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Rep (dot)

section

open Tape.Node (zeroCotangent)

variable {α : Type} [Storage α]

/-! ### Nondegeneracy of the algebraic pairing -/

/-- A tensor whose pairing with every tensor vanishes is the zero tensor. -/
theorem Tensor.eq_full_zero_of_forall_dot_eq_zero [CommSemiring α] :
    ∀ {s : Shape} (v : Tensor α s),
      (∀ a : Tensor α s, Rep.dot a v = 0) → v = Tensor.full s 0
  | .scalar, v, h => by
      apply Tensor.ext_scalar
      have h1 := h (Tensor.scalar 1)
      simp only [dot_scalar, Tensor.item_scalar, one_mul] at h1
      rw [h1, Tensor.item_full_scalar]
  | .dim n inner, v, h => by
      have hslice : ∀ i : Fin n, v.unstack i = Tensor.full inner 0 := by
        intro i
        apply eq_full_zero_of_forall_dot_eq_zero
        intro b
        have hb := h (Tensor.dim fun j => if j = i then b else Tensor.full inner 0)
        rw [dot_eq_sum_unstack] at hb
        simp only [Tensor.unstack_dim] at hb
        have hterm : ∀ j : Fin n,
            Rep.dot (if j = i then b else Tensor.full inner 0) (v.unstack j) =
              if j = i then Rep.dot b (v.unstack i) else 0 := by
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
        apply Tensor.Rep.ext
        intro c
        simp
      rw [← Tensor.dim_unstack v, hfill]
      congr 1
      funext i
      exact hslice i

/-- `dotList` with the all-zero context on the left is `0`. -/
theorem TensorPack.dotList_zero_left [CommSemiring α] :
    ∀ {ss : List Shape} (v : TensorPack α ss),
      TensorPack.dotList (α := α) (TensorPack.zero (α := α) (ss := ss)) v = 0
  | [], .nil => by simp [TensorPack.dotList, TensorPack.zero]
  | _ :: ss, .cons vh vt => by
      simpa [TensorPack.dotList, TensorPack.zero, Tensor.dot_full_zero_left] using
        dotList_zero_left vt

/-- A context whose pairing with every context vanishes is the zero context. -/
theorem TensorPack.eq_zero_of_forall_dotList_eq_zero [CommSemiring α] :
    ∀ {ss : List Shape} (v : TensorPack α ss),
      (∀ dx : TensorPack α ss, TensorPack.dotList (α := α) dx v = 0) →
        v = TensorPack.zero
  | [], .nil, _ => rfl
  | s :: ss, .cons vh vt, h => by
      have hh : vh = Tensor.full s 0 := by
        apply eq_full_zero_of_forall_dot_eq_zero
        intro a
        simpa [TensorPack.dotList, dotList_zero_left] using
          h (.cons a TensorPack.zero)
      have ht : vt = TensorPack.zero := by
        apply eq_zero_of_forall_dotList_eq_zero
        intro dx
        simpa [TensorPack.dotList, Tensor.dot_full_zero_left] using
          h (.cons (Tensor.full s 0) dx)
      rw [hh, ht]
      rfl

/--
A proof-carrying node's VJP sends the zero cotangent to the zero context. The adjointness law
alone gives it: `dotList dx (vjp x d 0) = dot (jvp x dx d) 0 = 0` for every `dx`.
-/
theorem Tape.LawfulNode.vjp_full_zero [CommSemiring α] {Δ : Type} {Γ : List Shape} {τ : Shape}
    (node : Tape.LawfulNode (α := α) Δ Γ τ) (x : TensorPack α Γ) (d : Δ) :
    node.vjp x d (Tensor.full τ 0) = TensorPack.zero := by
  apply TensorPack.eq_zero_of_forall_dotList_eq_zero
  intro dx
  rw [← node.jvp_vjp_adjoint x dx d (Tensor.full τ 0)]
  exact Tensor.dot_full_zero_right _

/-! ### Shape erasure of the zero context and of `single` -/

/-- Entries of the erased zero context are zero tensors of the recorded shapes. -/
theorem TensorPack.getElem?_erase_zero [Zero α] :
    ∀ {ss : List Shape} (j : Nat),
      (TensorPack.erase (α := α) (ss := ss)
        (TensorPack.zero (α := α) (ss := ss)))[j]? =
        (ss[j]?).map fun s => SomeTensor.ofTensor (Tensor.full s (0 : α))
  | [], j => by
      simp [TensorPack.erase, TensorPack.zero]
  | _ :: _, 0 => by
      simp [TensorPack.erase, TensorPack.zero,
        -SomeTensor.ofTensor]
  | _ :: ss, j + 1 => by
      simpa [TensorPack.erase, TensorPack.zero,
        Array.getElem?_append, -SomeTensor.ofTensor] using
        getElem?_erase_zero (ss := ss) j

/-- Every indexed contribution of the zero context is a zero tensor at a valid position. -/
theorem TensorPack.mem_eraseIndexed_zero [Zero α] :
    ∀ {ss : List Shape} (start : Nat) {pid : Nat} {pg : SomeTensor α},
      (pid, pg) ∈ TensorPack.eraseIndexed (α := α) (ss := ss)
        (TensorPack.zero (α := α) (ss := ss)) start →
      ∃ (i : Nat) (hi : i < ss.length),
        pid = start + i ∧ pg = SomeTensor.ofTensor (Tensor.full (ss[i]'hi) (0 : α))
  | [], start, pid, pg, hmem => by
      simp [TensorPack.eraseIndexed,
        TensorPack.zero] at hmem
  | s :: ss, start, pid, pg, hmem => by
      simp only [TensorPack.eraseIndexed,
        TensorPack.zero, TensorPack.fill, Pack.fill,
        Array.mem_append, Array.mem_singleton, Prod.mk.injEq] at hmem
      rcases hmem with ⟨hpid, hpg⟩ | hmem
      · exact ⟨0, by simp, by simp [hpid], hpg⟩
      · obtain ⟨i, hi, hpid, hpg⟩ := mem_eraseIndexed_zero (ss := ss) (start + 1) hmem
        exact ⟨i + 1, by simpa using Nat.succ_lt_succ hi,
          by rw [hpid, Nat.add_assoc, Nat.add_comm 1 i], hpg⟩

/-- The shape recorded at position `j` of an erased context is `ss[j]`. -/
theorem TensorPack.shape_getElem?_erase :
    ∀ {ss : List Shape} (xs : TensorPack α ss) (j : Nat),
      ((TensorPack.erase (α := α) (ss := ss) xs)[j]?).map
          SomeTensor.shape = ss[j]?
  | [], .nil, j => by simp [TensorPack.erase]
  | _ :: _, .cons _ _, 0 => by simp [TensorPack.erase]
  | _ :: ss, .cons _ xs, j + 1 => by
      simp [TensorPack.erase, Array.getElem?_append,
        shape_getElem?_erase (ss := ss) xs j]

/-- Erasing `single idx v`: `v` at `idx`, zero tensors of the recorded shapes elsewhere. -/
theorem TensorPack.getElem?_erase_single [Zero α] :
    ∀ {Γ : List Shape} {s : Shape} (idx : Idx Γ s) (v : Tensor α s) (j : Nat),
      (TensorPack.erase (α := α) (ss := Γ)
        (TensorPack.single idx v))[j]? =
        if j = idx.i.1 then some (SomeTensor.ofTensor v)
        else (Γ[j]?).map fun sh => SomeTensor.ofTensor (Tensor.full sh (0 : α))
  | [], _, ⟨i, _⟩, _, _ => (Nat.not_lt_zero _ i.2).elim
  | _ :: _, _, ⟨⟨0, _⟩, _⟩, v, 0 => by
      simp [TensorPack.single, TensorPack.erase,
        -SomeTensor.ofTensor]
  | _ :: Γ, _, ⟨⟨0, _⟩, _⟩, v, j + 1 => by
      simp [TensorPack.single, TensorPack.erase,
        Array.getElem?_append, getElem?_erase_zero (ss := Γ) j,
        -SomeTensor.ofTensor]
  | _ :: _, _, ⟨⟨_ + 1, _⟩, _⟩, v, 0 => by
      simp [TensorPack.single, TensorPack.erase,
        -SomeTensor.ofTensor]
  | _ :: Γ, s, ⟨⟨j' + 1, hi⟩, h⟩, v, j + 1 => by
      have ih := getElem?_erase_single (Γ := Γ) (s := s)
        ⟨⟨j', Nat.lt_of_succ_lt_succ hi⟩, by simpa using h⟩ v j
      simp [TensorPack.single, TensorPack.erase,
        Array.getElem?_append, ih, -SomeTensor.ofTensor]

/-! ### Lowered tapes are zero preserving -/

/-- The value stored at node `j` of a lowered tape has the shape recorded at `j` in `Γ ++ ss`. -/
theorem Tape.Graph.toTape_getElem?_value_shape {Δ : Type} {Γ ss : List Shape}
    (g : GraphData α Δ Γ ss) (x : TensorPack α Γ) (d0 : Δ) (j : Nat) :
    ((GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes[j]?).map
        (fun node => node.value.shape) = (Γ ++ ss)[j]? := by
  have hmap := congrArg (fun arr : Array (SomeTensor α) => (arr[j]?).map SomeTensor.shape)
    (GraphData.toTape_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0)
  simpa [Array.getElem?_map, Option.map_map, Function.comp_def,
    TensorPack.shape_getElem?_erase] using hmap

/--
The lowered tape of a proof-carrying graph is `ZeroPreserving`. Leaves emit no contributions, and
a lowered node's `backward` on its zero cotangent runs the stored VJP at zero, which is the zero
context by `LawfulNode.vjp_full_zero`, so every contribution it emits is a parent's zero cotangent.
-/
theorem Tape.Graph.zeroPreserving_toTape [CommSemiring α] {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := α) Δ Γ ss) (x : TensorPack α Γ) (d0 : Δ) :
    ZeroPreserving
      (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x d0).1 := by
  obtain ⟨g, hg⟩ := g
  dsimp only
  induction g with
  | nil =>
    refine ⟨fun id n hn _ => ?_⟩
    have hn' : ((TensorPack.erase (α := α) (ss := Γ) x).map Node.leaf)[id]? = some n := by
      simpa [GraphData.toTape, Tape.getNode?, nodes_addLeaves, Tape.empty] using hn
    rw [Array.getElem?_map] at hn'
    cases hx : (TensorPack.erase (α := α) (ss := Γ) x)[id]? with
    | none => simp [hx] at hn'
    | some v =>
      rw [hx, Option.map_some, Option.some.injEq] at hn'
      subst hn'
      exact ⟨#[], rfl, fun pid pg hmem => by simp at hmem⟩
  | snoc g node ih =>
    rename_i ssPrev τ
    obtain ⟨hg, hnode⟩ := hg
    refine ⟨fun id n hn hreq => ?_⟩
    let prev := GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
    let tPrev := prev.1
    let ctxPrev := prev.2
    let runtimeNode : Node (SomeTensor α) :=
      NodeData.toNode (some "typed-graph") node ctxPrev d0
    have hnNodes : (tPrev.nodes.push runtimeNode)[id]? = some n := by
      simpa [GraphData.toTape, prev, tPrev, ctxPrev, runtimeNode, Tape.getNode?,
        Tape.addNode] using hn
    have hsizePrev : tPrev.nodes.size = (Γ ++ ssPrev).length := by
      simp [tPrev, prev, GraphData.toTape_nodes_size, List.length_append]
    have hgetPrev : ∀ pid, pid < tPrev.nodes.size →
        (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc g node) x
          d0).1.getNode? pid = tPrev.getNode? pid := by
      intro pid hpid
      have hpid' : pid < (GraphData.toTape g x d0).1.nodes.size := hpid
      simp [GraphData.toTape, prev, tPrev, Tape.getNode?, Tape.addNode, Array.getElem?_push,
        Nat.ne_of_lt hpid']
    by_cases hlast : id = tPrev.nodes.size
    · subst hlast
      have hnEq : n = runtimeNode := by
        symm
        simpa [Array.getElem?_push] using hnNodes
      subst hnEq
      have hτ : (zeroCotangent runtimeNode).shape = τ := by
        simp [runtimeNode, zeroCotangent, NodeData.toNode]
      refine ⟨_, NodeData.toNode_backward_of_shape _ _ _ _ _ hτ, fun pid pg hmem => ?_⟩
      have hcast : (zeroCotangent runtimeNode).cast hτ = Tensor.full τ (0 : α) := rfl
      rw [hcast, LawfulNode.vjp_full_zero ⟨node, hnode⟩ ctxPrev d0] at hmem
      obtain ⟨i, hi, hpid, hpg⟩ := TensorPack.mem_eraseIndexed_zero (ss := Γ ++ ssPrev) 0 hmem
      have hi' : i < tPrev.nodes.size := by
        rw [hsizePrev]
        exact hi
      refine ⟨tPrev.nodes[i], ?_, ?_⟩
      · rw [hpid, Nat.zero_add, hgetPrev i hi']
        simp [Tape.getNode?, Array.getElem?_eq_getElem hi']
      · have hshape := toTape_getElem?_value_shape (α := α) g x d0 i
        have hi'' : i < (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
          d0).1.nodes.size := hi'
        rw [Array.getElem?_eq_getElem hi'', Option.map_some, List.getElem?_eq_getElem hi,
          Option.some.injEq] at hshape
        have hshape' : (tPrev.nodes[i]'hi').value.shape = (Γ ++ ssPrev)[i]'hi := hshape
        rw [hpg]
        exact congrArg (fun sh => SomeTensor.ofTensor (Tensor.full sh (0 : α))) hshape'.symm
    · have hidPrev : id < tPrev.nodes.size := by
        have hlt : id < tPrev.nodes.size + 1 := by
          have := (Array.getElem?_eq_some_iff.1 hnNodes).1
          simpa [Array.size_push] using this
        exact Nat.lt_of_le_of_ne (Nat.le_of_lt_succ hlt) hlast
      have hnPrev : tPrev.getNode? id = some n := by
        rw [← hgetPrev id hidPrev]
        exact hn
      obtain ⟨contribs, hback, hz⟩ := (ih hg).backward_zero id n hnPrev hreq
      refine ⟨contribs, hback, fun pid pg hmem => ?_⟩
      obtain ⟨pnode, hp, hpg⟩ := hz pid pg hmem
      refine ⟨pnode, ?_, hpg⟩
      rw [hgetPrev pid (getNode?_lt_size hp)]
      exact hp

/-! ### The one-hot seed array is the erased `single` context -/

/-- On a lowered tape, `oneHotGrads` at a typed index is the erasure of `TensorPack.single`. -/
theorem Tape.Graph.oneHotGrads_toTape_eq_single [Zero α] {Δ : Type} {Γ ss : List Shape}
    (g : GraphData α Δ Γ ss) (x : TensorPack α Γ) (d0 : Δ) {τ : Shape}
    (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    oneHotGrads (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1 output.i.1
        (SomeTensor.ofTensor seed) =
      TensorPack.erase (α := α) (ss := Γ ++ ss)
        (TensorPack.single output seed) := by
  apply Array.ext_getElem?
  intro j
  rw [getElem?_oneHotGrads, TensorPack.getElem?_erase_single]
  have hshape := toTape_getElem?_value_shape (α := α) g x d0 j
  have hlt : output.i.1 < (Γ ++ ss).length := output.i.2
  simp only [Tape.getNode?]
  cases hnode : (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0).1.nodes[j]? with
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
    · simp only [Option.map_some, ite_eq_right hj, ← hshape, zeroCotangent]

/-! ### Corollaries for `Tape.backwardDenseAll` on lowered tapes -/

/--
**Executed backward pass on a lowered graph = proved backpropagation.** Running
`Tape.backwardDenseAll` on the tape that `GraphData.toTape` builds from `g.toData`, seeded at any
typed output index `output` with cotangent `seed`, succeeds and returns the shape erasure of
`GraphData.backpropAllCtx` of the one-hot seed context.

The sweep here is the variant that skips unreached nodes. Zero preservation of lowered tapes
(`zeroPreserving_toTape`) reduces it to `GraphData.backwardDenseFrom_toTape`.
-/
theorem Tape.Graph.backwardDenseAll_toTape [CommSemiring α] {Δ : Type}
    {Γ ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TensorPack α Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    Tape.backwardDenseAll
        (t := (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x d0).1)
        output.i.1 (SomeTensor.ofTensor seed) =
      .ok (TensorPack.erase (α := α) (ss := Γ ++ ss)
        (g.toData.backpropAllCtx x d0 (TensorPack.single output seed))) := by
  have hlt : output.i.1 <
      (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x d0).1.nodes.size := by
    rw [GraphData.toTape_nodes_size]
    simpa [List.length_append] using output.i.2
  obtain ⟨outNode, hout⟩ := getNode?_of_lt_size hlt
  have hseed : (SomeTensor.ofTensor seed).shape = outNode.value.shape := by
    have hshape := toTape_getElem?_value_shape (α := α) g.toData x d0 output.i.1
    have hout' : (GraphData.toTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x
        d0).1.nodes[output.i.1]? = some outNode := hout
    rw [hout', Option.map_some, List.getElem?_eq_getElem output.i.2, Option.some.injEq] at hshape
    have hτ := output.h
    rw [List.get_eq_getElem] at hτ
    rw [SomeTensor.shape_ofTensor, hshape]
    exact hτ.symm
  rw [backwardDenseAll_eq_backwardDenseFrom (zeroPreserving_toTape g x d0) hout hseed,
    oneHotGrads_toTape_eq_single]
  exact GraphData.backwardDenseFrom_toTape (α := α) g.toData x d0
    (TensorPack.single output seed)

/--
**Executed backward pass on a lowered graph = adjoint of the Fréchet derivative.** Over `ℝ`,
`Tape.backwardDenseAll` on a lowered graph, seeded at a typed output index, succeeds with the full
backpropagation context, whose input (`Γ`-prefix) block is the adjoint of the Fréchet derivative
of the graph's forward evaluation applied to the one-hot seed. Differentiability is assumed only
at `x` and the values it reaches. The proof composes `backwardDenseAll_toTape` with
`backwardDenseFrom_eq_fderiv`.
-/
theorem Tape.Graph.backwardDenseAll_eq_fderiv {Δ : Type} {Γ ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TensorPack ℝ Γ) (d0 : Δ)
    {τ : Shape} (output : Idx (Γ ++ ss) τ) (seed : Tensor ℝ τ)
    (hg : RealGraph.FDerivCorrectAt (Γ := Γ) g.toData d0 (TensorPack.toVec x)) :
    Tape.backwardDenseAll
        (t := (GraphData.toTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x d0).1)
        output.i.1 (SomeTensor.ofTensor seed) =
      .ok (TensorPack.erase (α := ℝ) (ss := Γ ++ ss)
        (g.toData.backpropAllCtx x d0 (TensorPack.single output seed)))
    ∧ TensorPack.toVec (Pack.split (ss₁ := Γ)
        (g.toData.backpropAllCtx x d0 (TensorPack.single output seed))).1
      = (fderiv ℝ (g.toData.evalVec d0) (TensorPack.toVec x)).adjoint
          (TensorPack.toVec (TensorPack.single output seed)) :=
  ⟨backwardDenseAll_toTape g x d0 output seed,
    (backwardDenseFrom_eq_fderiv g x d0 (TensorPack.single output seed)
      hg).2⟩

end

end TorchLean
