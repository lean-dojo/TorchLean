/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Core
public import NN.Proofs.Autograd.Runtime.Link.Accumulation

/-!
# Dense Backward Pass: Appending One Lowered Node

This file proves the inductive step shared by the `Graph` and `GraphData` backward-link theorems.
Given a prefix tape `t` on which the dense reverse loop is already known to agree with some
reverse program `bp`, appending one lowered node yields a tape on which the dense reverse loop
computes: one `vjp` step for the new node, added into the prefix seed, followed by `bp` on the
prefix. The last gradient slot is never modified afterwards.

The proof is organised in three groups of lemmas.

- Push commutation: `addGradAll`, `backwardDenseFromStep` and `backwardDenseFromLoop` on the
  extended tape and an accumulator with one extra slot act on the prefix tape and accumulator,
  leaving the extra slot untouched, as long as all ids stay in the prefix range.
- The last-node step: processing the freshly appended node adds its `vjp` contributions into
  the prefix, which is `TensorPack.add` after shape erasure.
- `backwardDenseFrom_addNode_lowerNode`, which assembles the two.

Everything is stated for an arbitrary prefix tape characterised by its size, its `requiresGrad`
flags, its stored values and the `BackwardPidsLt` invariant, so it does not depend on which graph
representation produced the prefix, nor on the debug `name` of the appended node.
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

variable {α : Type} [TorchLean.Storage α] [Add α]

omit [Add α] in
/-- Appending a node does not change the nodes stored at existing ids. -/
theorem getNode?_addNode_of_lt (t : Tape α) (nd : Runtime.Autograd.Node α) {id : Nat}
    (hid : id < t.nodes.size) :
    (t.addNode nd).1.getNode? id = t.getNode? id := by
  simp [Tape.getNode?, Tape.addNode, Array.getElem?_push_lt hid, Array.getElem?_eq_getElem hid]

/--
`addGradAll` commutes with pushing an unused last slot.

On the tape extended by `nd` and an accumulator extended by `v`, accumulating into an id of the
prefix range acts exactly as on the prefix tape and accumulator, and `v` is carried along.
-/
theorem addGradAll_addNode_push (t : Tape α) (nd : Runtime.Autograd.Node α)
    (v : Spec.SomeTensor α) (acc : Array (Spec.SomeTensor α)) (hacc : acc.size = t.nodes.size)
    (pid : Nat) (pg : Spec.SomeTensor α) (hpid : pid < t.nodes.size) :
    Tape.addGradAll (t := (t.addNode nd).1) (acc.push v) pid pg =
      Except.map (fun a => a.push v) (Tape.addGradAll (t := t) acc pid pg) := by
  have hpidAcc : pid < acc.size := by simpa [hacc] using hpid
  let nodeAt : Runtime.Autograd.Node α := t.nodes[pid]'hpid
  have hnodePrev : Tape.getNode? (t := t) pid = some nodeAt := by
    simp [Tape.getNode?, nodeAt, Array.getElem?_eq_getElem hpid]
  have hnodeNext : Tape.getNode? (t := (t.addNode nd).1) pid = some nodeAt := by
    rw [getNode?_addNode_of_lt t nd hpid, hnodePrev]
  have hgetPrev : acc[pid]? = some (acc[pid]'hpidAcc) := Array.getElem?_eq_getElem hpidAcc
  have hgetNext : (acc.push v)[pid]? = some (acc[pid]'hpidAcc) := Array.getElem?_push_lt hpidAcc
  cases hreq : nodeAt.requiresGrad with
  | false =>
      simp [Tape.addGradAll, hnodePrev, hnodeNext, hreq, result_pure_eq_ok, result_bind_ok,
        result_map_ok]
  | true =>
      by_cases hshape : pg.shape = nodeAt.value.shape
      · by_cases hex : (acc[pid]'hpidAcc).shape = nodeAt.value.shape
        · let pg' : Spec.SomeTensor α := { shape := nodeAt.value.shape, tensor := pg.cast hshape }
          let existing' : Spec.SomeTensor α :=
            { shape := nodeAt.value.shape, tensor := (acc[pid]'hpidAcc).cast hex }
          cases hadd : Runtime.Autograd.SomeTensor.add existing' pg' with
          | error e =>
              have hprev : Tape.addGradAll (t := t) acc pid pg = .error e := by
                simp [Tape.addGradAll, hnodePrev, hreq, hshape, hgetPrev, hex, pg', existing',
                  hadd, result_throw_eq_error, result_bind_error]
              have hnext : Tape.addGradAll (t := (t.addNode nd).1) (acc.push v) pid pg =
                  .error e := by
                simp [Tape.addGradAll, hnodeNext, hreq, hshape, hgetNext, hex, pg', existing',
                  hadd, result_throw_eq_error, result_bind_error]
              simp [hprev, hnext, result_map_error]
          | ok summed =>
              have hprev :
                  Tape.addGradAll (t := t) acc pid pg =
                    .ok (acc.set pid summed (h := hpidAcc)) := by
                simp [Tape.addGradAll, hnodePrev, hreq, hshape, hex, pg', existing', hadd,
                  hpidAcc, result_throw_eq_error, result_bind_ok, result_pure_eq_ok]
              have hnext :
                  Tape.addGradAll (t := (t.addNode nd).1) (acc.push v) pid pg =
                    .ok ((acc.set pid summed (h := hpidAcc)).push v) := by
                have hpidAccPush : pid < (acc.push v).size := by
                  rw [Array.size_push]
                  exact Nat.lt_succ_of_lt hpidAcc
                have hget : (acc.push v)[pid] = acc[pid] := by
                  simpa using (Array.getElem_push_lt (xs := acc) (x := v) (i := pid) hpidAcc)
                simp [Tape.addGradAll, hnodeNext, hreq, hshape, hex, pg', existing', hadd,
                  hpidAcc, hgetNext, Array.set_push, result_throw_eq_error, result_bind_ok,
                  result_pure_eq_ok]
                omega
              simp [hprev, hnext, result_map_ok]
        · simp [Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape, hgetPrev, hgetNext, hex,
            result_map_error, result_throw_eq_error]
      · simp [Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape, result_map_error,
          result_throw_eq_error]

/--
Folding `addGradAll` over a contribution list commutes with pushing an unused last slot, as long
as every contribution targets the prefix range.
-/
theorem foldlM_addGradAll_addNode_push (t : Tape α) (nd : Runtime.Autograd.Node α)
    (v : Spec.SomeTensor α) :
    ∀ (cs : List (Nat × Spec.SomeTensor α)) (acc : Array (Spec.SomeTensor α)),
      acc.size = t.nodes.size →
      (∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ cs → pid < t.nodes.size) →
      cs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := (t.addNode nd).1) acc2 pid pg)
          (acc.push v) =
        Except.map (fun a => a.push v)
          (cs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg) acc)
  | [], acc, _, _ => rfl
  | (pid, pg) :: tl, acc, hacc, hpids => by
      have hpid : pid < t.nodes.size := hpids (List.mem_cons_self ..)
      have hhead := addGradAll_addNode_push t nd v acc hacc pid pg hpid
      cases hret : Tape.addGradAll (t := t) acc pid pg with
      | error e =>
          rw [hret, result_map_error] at hhead
          simp [List.foldlM, hret, hhead, result_bind_error, result_map_error]
      | ok acc1 =>
          rw [hret, result_map_ok] at hhead
          have hsize1 : acc1.size = t.nodes.size :=
            (addGradAll_ok_size (t := t) hret).trans hacc
          have hpids_tl : ∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ tl →
              pid < t.nodes.size :=
            fun hmem => hpids (List.mem_cons_of_mem _ hmem)
          simp only [List.foldlM, hret, hhead, result_bind_ok]
          exact foldlM_addGradAll_addNode_push t nd v tl acc1 hsize1 hpids_tl

/--
`backwardDenseFromStep` commutes with pushing an unused last slot for ids in the prefix range.

The `BackwardPidsLt` invariant guarantees that all contributions emitted by the visited node also
stay in the prefix range.
-/
theorem backwardDenseFromStep_addNode_push (t : Tape α) (nd : Runtime.Autograd.Node α)
    (v : Spec.SomeTensor α) (hpids : BackwardPidsLt t) {id : Nat} (hid : id < t.nodes.size)
    (acc : Array (Spec.SomeTensor α)) (hacc : acc.size = t.nodes.size) :
    Tape.backwardDenseFromStep (t := (t.addNode nd).1) (acc.push v) id =
      Except.map (fun a => a.push v) (Tape.backwardDenseFromStep (t := t) acc id) := by
  let nodeAt : Runtime.Autograd.Node α := t.nodes[id]'hid
  have hnodePrev : Tape.getNode? (t := t) id = some nodeAt := by
    simp [Tape.getNode?, nodeAt, Array.getElem?_eq_getElem hid]
  have hnodeNext : Tape.getNode? (t := (t.addNode nd).1) id = some nodeAt := by
    rw [getNode?_addNode_of_lt t nd hid, hnodePrev]
  have hidAcc : id < acc.size := by simpa [hacc] using hid
  have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := Array.getElem?_eq_getElem hidAcc
  have hgetAccPush : (acc.push v)[id]? = some (acc[id]'hidAcc) := Array.getElem?_push_lt hidAcc
  cases hreq : nodeAt.requiresGrad with
  | false =>
      simp [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, result_pure_eq_ok,
        result_bind_ok, result_map_ok]
  | true =>
      by_cases hshape : (acc[id]'hidAcc).shape = nodeAt.value.shape
      · cases hback : nodeAt.backward (Spec.SomeTensor.ofTensor ((acc[id]'hidAcc).cast hshape)) with
        | error e =>
            simp only [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, hgetAcc, hgetAccPush,
              hshape, hback, result_pure_eq_ok, result_bind_ok, result_bind_error, result_map_error,
              dite_true, Bool.true_eq_false, if_false]
        | ok contribs =>
            have hpids_list : ∀ {pid : Nat} {pg : Spec.SomeTensor α},
                (pid, pg) ∈ contribs.toList → pid < t.nodes.size := by
              intro pid pg hmem
              exact Nat.lt_trans
                (hpids id nodeAt hnodePrev _ contribs hback (by simpa using hmem)) hid
            have hfold := foldlM_addGradAll_addNode_push t nd v contribs.toList acc hacc hpids_list
            simp only [Array.foldlM_toList] at hfold
            simp only [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, hgetAcc, hgetAccPush,
              hshape, hback, result_pure_eq_ok, result_bind_ok, dite_true, Bool.true_eq_false,
              if_false]
            exact hfold
      · simp [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, hgetAcc, hgetAccPush,
          hshape, result_map_error, result_throw_eq_error]

/-- `backwardDenseFromLoop` over the prefix range commutes with pushing an unused last slot. -/
theorem backwardDenseFromLoop_addNode_push (t : Tape α) (nd : Runtime.Autograd.Node α)
    (v : Spec.SomeTensor α) (hpids : BackwardPidsLt t) :
    ∀ (m : Nat), m ≤ t.nodes.size → ∀ (acc : Array (Spec.SomeTensor α)),
      acc.size = t.nodes.size →
      Tape.backwardDenseFromLoop (t := (t.addNode nd).1) m (acc.push v) =
        Except.map (fun a => a.push v) (Tape.backwardDenseFromLoop (t := t) m acc)
  | 0, _, _, _ => rfl
  | m + 1, hm, acc, hacc => by
      have hmid : m < t.nodes.size := hm
      have hstep := backwardDenseFromStep_addNode_push t nd v hpids hmid acc hacc
      cases hret : Tape.backwardDenseFromStep (t := t) acc m with
      | error e =>
          rw [hret, result_map_error] at hstep
          simp [Tape.backwardDenseFromLoop, hret, hstep, result_bind_error, result_map_error]
      | ok acc1 =>
          rw [hret, result_map_ok] at hstep
          have hsize1 : acc1.size = t.nodes.size :=
            (backwardDenseFromStep_ok_size (t := t) hret).trans hacc
          simp only [Tape.backwardDenseFromLoop, hret, hstep, result_bind_ok]
          exact backwardDenseFromLoop_addNode_push t nd v hpids m (Nat.le_of_lt hmid) acc1 hsize1

omit [Add α] in
/--
The prefix of the extended tape is a well-formed dense accumulator for `seed`: every prefix id
stores a node that requires gradients and whose value has the shape of the seed entry.

This is the hypothesis of `foldlM_addGradAll_toIndexedShapeErasedArray_eq_add` with `pref = #[]`.
-/
theorem addNode_prefix_node_spec (t : Tape α) (nd : Runtime.Autograd.Node α) {ss : List Shape}
    (ctx seed : TorchLean.TensorPack α ss) (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx) :
    ∀ i (hi : i < ss.length),
      ∃ node : Runtime.Autograd.Node α,
        (t.addNode nd).1.getNode? i = some node ∧ node.requiresGrad = true ∧
          node.value.shape =
            ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seed)[i]'(by
              simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape := by
  intro i hi
  have hiT : i < t.nodes.size := hsize ▸ hi
  have hiCtx :
      i < (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx).size := by
    simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hi
  refine ⟨t.nodes[i]'hiT, ?_, hreq i hiT, ?_⟩
  · rw [getNode?_addNode_of_lt t nd hiT]
    simp [Tape.getNode?, Array.getElem?_eq_getElem hiT]
  · have hval : (t.nodes[i]'hiT).value =
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx)[i]'hiCtx := by
      have h := congrArg (fun a => a[i]?) hvals
      simpa [Array.getElem?_eq_getElem hiT, Array.getElem?_eq_getElem hiCtx] using h
    rw [hval, shape_getElem_toShapeErasedArray ctx hi, shape_getElem_toShapeErasedArray seed hi]

/--
The runtime step for a freshly appended lowered node adds the node's `vjp` contributions into the
prefix gradients, which is `TensorPack.add` after shape erasure, and leaves the node's own slot
untouched.
-/
theorem backwardDenseFromStep_addNode_lowerNode_last {Δ : Type} {ss : List Shape} {τ : Shape}
    (t : Tape α) (name : Option String) (node : NodeData α Δ ss τ)
    (ctx : TorchLean.TensorPack α ss) (d : Δ) (seedPrev : TorchLean.TensorPack α ss)
    (seedOut : Tensor α τ) (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx) :
    Tape.backwardDenseFromStep (t := (t.addNode (lowerNode (α := α) name node ctx d)).1)
        ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev).push
          (Spec.SomeTensor.ofTensor seedOut))
        t.nodes.size =
      .ok ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
        (TorchLean.TensorPack.add (α := α) (ss := ss) seedPrev
          (node.vjp ctx d seedOut))).push (Spec.SomeTensor.ofTensor seedOut)) := by
  have hspec := addNode_prefix_node_spec t (lowerNode (α := α) name node ctx d) ctx seedPrev hsize
    hreq hvals
  generalize hnd : lowerNode (α := α) name node ctx d = nd at hspec ⊢
  generalize hout : Spec.SomeTensor.ofTensor seedOut = outValue
  have hsizeArr :
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev).size =
        t.nodes.size := by
    rw [TorchLean.TensorPack.size_toShapeErasedArray, hsize]
  have hnodeLast : Tape.getNode? (t := (t.addNode nd).1) t.nodes.size = some nd := by
    simp [Tape.getNode?, Tape.addNode]
  have haccLast :
      ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev).push
        outValue)[t.nodes.size]? = some outValue := by
    rw [← hsizeArr, Array.getElem?_push, if_pos rfl]
  have hreqNd : nd.requiresGrad = true := by
    rw [← hnd]
    rfl
  have hshapeNode : outValue.shape = nd.value.shape := by
    rw [← hnd, ← hout]
    rfl
  have hbackLast : nd.backward outValue =
      .ok (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := ss)
        (node.vjp ctx d seedOut) 0) := by
    rw [← hnd, ← hout, lowerNode_backward_of_shape name node ctx d _ rfl]
    rfl
  have hfoldLast :
      (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := ss)
          (node.vjp ctx d seedOut) 0).foldlM
        (fun acc2 (pid, pg) => Tape.addGradAll (t := (t.addNode nd).1) acc2 pid pg)
        ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev).push
          outValue) =
      .ok ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
        (TorchLean.TensorPack.add (α := α) (ss := ss) seedPrev
          (node.vjp ctx d seedOut))).push outValue) := by
    have hfold := foldlM_addGradAll_toIndexedShapeErasedArray_eq_add (α := α)
      (t := (t.addNode nd).1) (ss := ss) (pref := #[]) (seed := seedPrev)
      (contrib := node.vjp ctx d seedOut)
      (suffix := #[outValue]) (by
        intro i hi
        simpa using hspec i hi)
    simpa [Array.append_assoc, Array.append_empty, Array.empty_append, Array.append_singleton]
      using hfold
  simp only [Tape.backwardDenseFromStep, hnodeLast, haccLast, hreqNd, result_pure_eq_ok,
    result_bind_ok, Bool.true_eq_false, if_false]
  rw [dif_pos hshapeNode, Spec.SomeTensor.ofTensor_cast, hbackLast, result_bind_ok]
  exact hfoldLast

/--
**Inductive step of the backward link.** Let `t` be a tape whose dense reverse loop agrees with the
reverse program `bp` on every seed (after shape erasure), and whose nodes have the sizes, flags,
values and backward-pointing invariant of a lowered prefix over the context `ss`. Appending the
lowered node `node` yields a tape whose dense reverse loop, seeded by `snoc seedPrev seedOut`,
returns `snoc (bp (seedPrev + node.vjp ctx d seedOut)) seedOut`.

This is the shared content of `backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` and its
`GraphData` variant: the `snoc` cases of `backpropAllCtx` unfold to exactly this shape.
-/
theorem backwardDenseFrom_addNode_lowerNode {Δ : Type} {ss : List Shape} {τ : Shape}
    (t : Tape α) (name : Option String) (node : NodeData α Δ ss τ)
    (ctx : TorchLean.TensorPack α ss) (d : Δ)
    (bp : TorchLean.TensorPack α ss → TorchLean.TensorPack α ss)
    (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) ctx)
    (hpids : BackwardPidsLt t)
    (hbp : ∀ s : TorchLean.TensorPack α ss,
      Tape.backwardDenseFrom (t := t) (TorchLean.TensorPack.toShapeErasedArray (α := α) s) =
        .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (bp s)))
    (seedPrev : TorchLean.TensorPack α ss) (seedOut : Tensor α τ) :
    Tape.backwardDenseFrom (t := (t.addNode (lowerNode (α := α) name node ctx d)).1)
        (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss ++ [τ])
          (TorchLean.TensorPack.snoc (α := α) (ss := ss) (τ := τ) seedPrev seedOut)) =
      .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss ++ [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := ss) (τ := τ)
          (bp (TorchLean.TensorPack.add (α := α) (ss := ss) seedPrev
            (node.vjp ctx d seedOut)))
          seedOut)) := by
  have hstepLast :=
    backwardDenseFromStep_addNode_lowerNode_last t name node ctx d seedPrev seedOut hsize hreq hvals
  generalize hnd : lowerNode (α := α) name node ctx d = nd at hstepLast ⊢
  generalize hsp :
    TorchLean.TensorPack.add (α := α) (ss := ss) seedPrev (node.vjp ctx d seedOut) =
      seedPrev' at hstepLast ⊢
  have htNextSize : (t.addNode nd).1.nodes.size = t.nodes.size + 1 := by
    simp [Tape.addNode]
  have hsizeCheck :
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss ++ [τ])
        (TorchLean.TensorPack.snoc (α := α) (ss := ss) (τ := τ) seedPrev seedOut)).size =
      (t.addNode nd).1.nodes.size := by
    simp [htNextSize, hsize, TorchLean.TensorPack.size_toShapeErasedArray]
  have hsizePrev' :
      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev').size =
        t.nodes.size := by
    simp [hsize, TorchLean.TensorPack.size_toShapeErasedArray]
  have hloopPrev :
      Tape.backwardDenseFromLoop (t := t) t.nodes.size
          (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev') =
        .ok (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
          (bp seedPrev')) := by
    have h := hbp seedPrev'
    unfold Tape.backwardDenseFrom at h
    rwa [if_pos hsizePrev'] at h
  have hloopFinal :
      Tape.backwardDenseFromLoop (t := (t.addNode nd).1) t.nodes.size
          ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedPrev').push
            (Spec.SomeTensor.ofTensor seedOut)) =
        .ok ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
          (bp seedPrev')).push (Spec.SomeTensor.ofTensor seedOut)) := by
    rw [backwardDenseFromLoop_addNode_push t nd (Spec.SomeTensor.ofTensor seedOut) hpids
      t.nodes.size le_rfl _ hsizePrev', hloopPrev, result_map_ok]
  unfold Tape.backwardDenseFrom
  rw [if_pos hsizeCheck, htNextSize, TorchLean.TensorPack.toShapeErasedArray_snoc,
    Tape.backwardDenseFromLoop, hstepLast, result_bind_ok, hloopFinal,
    TorchLean.TensorPack.toShapeErasedArray_snoc]

end Graph

end Algebra
end Autograd
end Proofs
