/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.Lowering
public import TorchLean.NN.Tape.Lowering.Accumulation

/-!
# Dense Backward Pass: Appending One Lowered Node

These lemmas prove the inductive step of `GraphData.backwardDenseFrom_toTape`,
which the `Graph` version reuses through `Graph.toData`. Suppose the dense reverse loop on a prefix
tape `t` agrees with some reverse program `bp`. After one lowered node is appended, the dense
reverse loop runs one `vjp` step for that node, adds the result into the prefix seed and then runs
`bp` on the prefix. The last gradient slot is never modified afterwards.

The proof is organised in three groups of lemmas.

- Push commutation: `addGradAll`, `backwardDenseFromStep` and `backwardDenseFromLoop` on the
  extended tape and an accumulator with one extra slot act on the prefix tape and accumulator,
  leaving the extra slot untouched, as long as all ids stay in the prefix range.
- The last-node step: processing the freshly appended node adds its `vjp` contributions into
  the prefix, which is `TensorPack.add` after shape erasure.
- `NodeData.backwardDenseFrom_toNode`, which assembles the two.

Everything is stated for an arbitrary prefix tape characterised by its size, its `requiresGrad`
flags, its stored values and the `BackwardPidsLt` invariant. Neither the graph representation
that produced the prefix nor the debug `name` of the appended node enters the proofs.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

variable {α : Type} [Storage α] [Add α]

omit [Add α] in
/-- Appending a node does not change the nodes stored at existing ids. -/
theorem getNode?_addNode_of_lt (t : Tape (SomeTensor α)) (nd : Node
    (SomeTensor α)) {id : Nat}
    (hid : id < t.nodes.size) :
    (t.addNode nd).1.getNode? id = t.getNode? id := by
  simp [Tape.getNode?, Tape.addNode, Array.getElem?_push_lt hid, Array.getElem?_eq_getElem hid]

/--
`addGradAll` commutes with pushing an unused last slot.

On the tape extended by `nd` and an accumulator extended by `v`, accumulating into an id of the
prefix range acts exactly as on the prefix tape and accumulator, and `v` is carried along.
-/
theorem addGradAll_addNode_push (t : Tape (SomeTensor α)) (nd : Node
    (SomeTensor α))
    (v : SomeTensor α) (acc : Array (SomeTensor α)) (hacc : acc.size = t.nodes.size)
    (pid : Nat) (pg : SomeTensor α) (hpid : pid < t.nodes.size) :
    Tape.addGradAll (t := (t.addNode nd).1) (acc.push v) pid pg =
      Except.map (fun a => a.push v) (Tape.addGradAll (t := t) acc pid pg) := by
  have hpidAcc : pid < acc.size := by simpa [hacc] using hpid
  let nodeAt : Node (SomeTensor α) := t.nodes[pid]'hpid
  have hnodePrev : Tape.getNode? (t := t) pid = some nodeAt := by
    simp [Tape.getNode?, nodeAt, Array.getElem?_eq_getElem hpid]
  have hnodeNext : Tape.getNode? (t := (t.addNode nd).1) pid = some nodeAt := by
    rw [getNode?_addNode_of_lt t nd hpid, hnodePrev]
  have hgetPrev : acc[pid]? = some (acc[pid]'hpidAcc) := Array.getElem?_eq_getElem hpidAcc
  have hgetNext : (acc.push v)[pid]? = some (acc[pid]'hpidAcc) := Array.getElem?_push_lt hpidAcc
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      GradValue.add_someTensor, GradValue.drop_someTensor, Tape.addGradAll, hnodePrev, hnodeNext,
      Bind.bind, Except.bind, Pure.pure, Except.pure, throw, throwThe, MonadExceptOf.throw]
  by_cases hreq : nodeAt.requiresGrad = false
  · simp [hreq, Except.map]
  · simp only [ite_eq_right hreq]
    by_cases hshape : pg.shape = nodeAt.value.shape
    · simp only [dite_eq_left hshape]
      split
      · rename_i hget
        simp [hgetNext] at hget
      · rename_i existing hget
        have heq : existing = acc[pid] := Option.some.inj (hget.symm.trans hgetNext)
        subst existing
        by_cases hex : acc[pid].shape = nodeAt.value.shape
        · simp only [dite_eq_left hex]
          cases hadd : SomeTensor.add
              (SomeTensor.ofTensor ((acc[pid]).cast hex))
              (SomeTensor.ofTensor (pg.cast hshape))
          all_goals
            dsimp only
            split
            · rename_i hget
              simp [hgetPrev] at hget
            · rename_i previous hget
              have heq : previous = acc[pid] := Option.some.inj (hget.symm.trans hgetPrev)
              subst previous
              simp only [dite_eq_left hex]
              rw [hadd]
              simp [Except.map, Array.set_push, hpidAcc]
        · simp only [dite_eq_right hex]
          split
          · rename_i hget
            simp [hgetPrev] at hget
          · rename_i previous hget
            have heq : previous = acc[pid] := Option.some.inj (hget.symm.trans hgetPrev)
            subst previous
            simp [hex, Except.map]
    · simp [hshape, Except.map]

/--
Folding `addGradAll` over a contribution list commutes with pushing an unused last slot, as long
as every contribution targets the prefix range.
-/
theorem foldlM_addGradAll_addNode_push (t : Tape (SomeTensor α)) (nd : Node
    (SomeTensor α))
    (v : SomeTensor α) :
    ∀ (cs : List (Nat × SomeTensor α)) (acc : Array (SomeTensor α)),
      acc.size = t.nodes.size →
      (∀ {pid : Nat} {pg : SomeTensor α}, (pid, pg) ∈ cs → pid < t.nodes.size) →
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
          rw [hret] at hhead
          simp only [Except.map] at hhead
          simp [List.foldlM, hret, hhead, Result.bind_error, Except.map]
      | ok acc1 =>
          rw [hret] at hhead
          simp only [Except.map] at hhead
          have hsize1 : acc1.size = t.nodes.size :=
            (addGradAll_ok_size (t := t) hret).trans hacc
          have hpids_tl : ∀ {pid : Nat} {pg : SomeTensor α}, (pid, pg) ∈ tl →
              pid < t.nodes.size :=
            fun hmem => hpids (List.mem_cons_of_mem _ hmem)
          simp only [List.foldlM, hret, hhead, Result.bind_ok]
          exact foldlM_addGradAll_addNode_push t nd v tl acc1 hsize1 hpids_tl

/--
`backwardDenseFromStep` commutes with pushing an unused last slot for ids in the prefix range.

The `BackwardPidsLt` invariant guarantees that all contributions emitted by the visited node also
stay in the prefix range.
-/
theorem backwardDenseFromStep_addNode_push (t : Tape (SomeTensor α)) (nd
    : Node (SomeTensor α))
    (v : SomeTensor α) (hpids : BackwardPidsLt t) {id : Nat} (hid : id < t.nodes.size)
    (acc : Array (SomeTensor α)) (hacc : acc.size = t.nodes.size) :
    Tape.backwardDenseFromStep (t := (t.addNode nd).1) (acc.push v) id =
      Except.map (fun a => a.push v) (Tape.backwardDenseFromStep (t := t) acc id) := by
  let nodeAt : Node (SomeTensor α) := t.nodes[id]'hid
  have hnodePrev : Tape.getNode? (t := t) id = some nodeAt := by
    simp [Tape.getNode?, nodeAt, Array.getElem?_eq_getElem hid]
  have hnodeNext : Tape.getNode? (t := (t.addNode nd).1) id = some nodeAt := by
    rw [getNode?_addNode_of_lt t nd hid, hnodePrev]
  have hidAcc : id < acc.size := by simpa [hacc] using hid
  have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := Array.getElem?_eq_getElem hidAcc
  have hgetAccPush : (acc.push v)[id]? = some (acc[id]'hidAcc) := Array.getElem?_push_lt hidAcc
  cases hreq : nodeAt.requiresGrad with
  | false =>
      simp [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, Result.pure_eq_ok,
        Result.bind_ok, Except.map]
  | true =>
      by_cases hshape : (acc[id]'hidAcc).shape = nodeAt.value.shape
      · cases hback : nodeAt.backward (SomeTensor.ofTensor ((acc[id]'hidAcc).cast hshape)) with
        | error e =>
            simp only [GradValue.shape_someTensor, GradValue.retag_someTensor,
                GradValue.check_someTensor, Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                hgetAcc, hgetAccPush, hshape, hback, Result.pure_eq_ok, Result.bind_ok,
                Result.bind_error, Except.map, dite_true, Bool.true_eq_false, ite_false]
        | ok contribs =>
            have hpids_list : ∀ {pid : Nat} {pg : SomeTensor α},
                (pid, pg) ∈ contribs.toList → pid < t.nodes.size := by
              intro pid pg hmem
              exact Nat.lt_trans
                (hpids id nodeAt hnodePrev _ contribs hback (by simpa using hmem)) hid
            have hfold := foldlM_addGradAll_addNode_push t nd v contribs.toList acc hacc hpids_list
            simp only [Array.foldlM_toList] at hfold
            simp only [GradValue.shape_someTensor, GradValue.retag_someTensor,
                GradValue.check_someTensor, Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                hgetAcc, hgetAccPush, hshape, hback, Result.pure_eq_ok, Result.bind_ok, dite_true,
                Bool.true_eq_false, ite_false]
            exact hfold
      · simp [Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq, hgetAcc, hgetAccPush,
          hshape, Except.map, Result.throw_eq_error]

/-- `backwardDenseFromLoop` over the prefix range commutes with pushing an unused last slot. -/
theorem backwardDenseFromLoop_addNode_push (t : Tape (SomeTensor α)) (nd
    : Node (SomeTensor α))
    (v : SomeTensor α) (hpids : BackwardPidsLt t) :
    ∀ (m : Nat), m ≤ t.nodes.size → ∀ (acc : Array (SomeTensor α)),
      acc.size = t.nodes.size →
      Tape.backwardDenseFromLoop (t := (t.addNode nd).1) m (acc.push v) =
        Except.map (fun a => a.push v) (Tape.backwardDenseFromLoop (t := t) m acc)
  | 0, _, _, _ => rfl
  | m + 1, hm, acc, hacc => by
      have hmid : m < t.nodes.size := hm
      have hstep := backwardDenseFromStep_addNode_push t nd v hpids hmid acc hacc
      cases hret : Tape.backwardDenseFromStep (t := t) acc m with
      | error e =>
          rw [hret] at hstep
          simp only [Except.map] at hstep
          simp [Tape.backwardDenseFromLoop, hret, hstep, Result.bind_error, Except.map]
      | ok acc1 =>
          rw [hret] at hstep
          simp only [Except.map] at hstep
          have hsize1 : acc1.size = t.nodes.size :=
            (backwardDenseFromStep_ok_size (t := t) hret).trans hacc
          simp only [Tape.backwardDenseFromLoop, hret, hstep, Result.bind_ok]
          exact backwardDenseFromLoop_addNode_push t nd v hpids m (Nat.le_of_lt hmid) acc1 hsize1

omit [Add α] in
/--
On the extended tape, every prefix id `i` stores a node that requires gradients and whose value
has the shape of entry `i` of the erased `seed`.

These are the hypotheses of `foldlM_addGradAll_eraseIndexed` with `pref = #[]`.
-/
theorem exists_getNode?_addNode_of_lt (t : Tape (SomeTensor α)) (nd : Node
    (SomeTensor α)) {ss : List Shape}
    (ctx seed : TensorPack α ss) (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TensorPack.erase (α := α) (ss := ss) ctx) :
    ∀ i (hi : i < ss.length),
      ∃ node : Node (SomeTensor α),
        (t.addNode nd).1.getNode? i = some node ∧ node.requiresGrad = true ∧
          node.value.shape =
            ((TensorPack.erase (α := α) (ss := ss) seed)[i]'(by
              simpa [TensorPack.size_erase] using hi)).shape := by
  intro i hi
  have hiT : i < t.nodes.size := hsize ▸ hi
  have hiCtx :
      i < (TensorPack.erase (α := α) (ss := ss) ctx).size := by
    simpa [TensorPack.size_erase] using hi
  refine ⟨t.nodes[i]'hiT, ?_, hreq i hiT, ?_⟩
  · rw [getNode?_addNode_of_lt t nd hiT]
    simp [Tape.getNode?, Array.getElem?_eq_getElem hiT]
  · have hval : (t.nodes[i]'hiT).value =
        (TensorPack.erase (α := α) (ss := ss) ctx)[i]'hiCtx := by
      have h := congrArg (fun a => a[i]?) hvals
      simpa [Array.getElem?_eq_getElem hiT, Array.getElem?_eq_getElem hiCtx] using h
    rw [hval, TensorPack.shape_getElem_erase ctx hi, TensorPack.shape_getElem_erase seed hi]

/--
The runtime step for a freshly appended lowered node adds the node's `vjp` contributions into the
prefix gradients, which is `TensorPack.add` after shape erasure, and leaves the node's own slot
untouched.
-/
theorem NodeData.backwardDenseFromStep_toNode {Δ : Type} {ss : List Shape} {τ : Shape}
    (t : Tape (SomeTensor α)) (name : Option String) (node : NodeData α Δ ss τ)
    (ctx : TensorPack α ss) (d : Δ) (seedPrev : TensorPack α ss)
    (seedOut : Tensor α τ) (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TensorPack.erase (α := α) (ss := ss) ctx) :
    Tape.backwardDenseFromStep (t := (t.addNode (toNode (α := α) name node ctx d)).1)
        ((TensorPack.erase (α := α) (ss := ss) seedPrev).push
          (SomeTensor.ofTensor seedOut))
        t.nodes.size =
      .ok ((TensorPack.erase (α := α) (ss := ss)
        (TensorPack.add (α := α) (ss := ss) seedPrev
          (node.vjp ctx d seedOut))).push (SomeTensor.ofTensor seedOut)) := by
  have hspec := exists_getNode?_addNode_of_lt t (toNode (α := α) name node ctx d) ctx seedPrev hsize
    hreq hvals
  generalize hnd : toNode (α := α) name node ctx d = nd at hspec ⊢
  generalize hout : SomeTensor.ofTensor seedOut = outValue
  have hsizeArr :
      (TensorPack.erase (α := α) (ss := ss) seedPrev).size =
        t.nodes.size := by
    rw [TensorPack.size_erase, hsize]
  have hnodeLast : Tape.getNode? (t := (t.addNode nd).1) t.nodes.size = some nd := by
    simp [Tape.getNode?, Tape.addNode]
  have haccLast :
      ((TensorPack.erase (α := α) (ss := ss) seedPrev).push
        outValue)[t.nodes.size]? = some outValue := by
    rw [← hsizeArr, Array.getElem?_push, ite_eq_left rfl]
  have hreqNd : nd.requiresGrad = true := by
    rw [← hnd]
    rfl
  have hshapeNode : outValue.shape = nd.value.shape := by
    rw [← hnd, ← hout]
    rfl
  have hbackLast : nd.backward outValue =
      .ok (TensorPack.eraseIndexed (α := α) (ss := ss)
        (node.vjp ctx d seedOut) 0) := by
    rw [← hnd, ← hout, toNode_backward_of_shape name node ctx d _ rfl]
    rfl
  have hfoldLast :
      (TensorPack.eraseIndexed (α := α) (ss := ss)
          (node.vjp ctx d seedOut) 0).foldlM
        (fun acc2 (pid, pg) => Tape.addGradAll (t := (t.addNode nd).1) acc2 pid pg)
        ((TensorPack.erase (α := α) (ss := ss) seedPrev).push
          outValue) =
      .ok ((TensorPack.erase (α := α) (ss := ss)
        (TensorPack.add (α := α) (ss := ss) seedPrev
          (node.vjp ctx d seedOut))).push outValue) := by
    have hfold := foldlM_addGradAll_eraseIndexed (α := α)
      (t := (t.addNode nd).1) (ss := ss) (pref := #[]) (seed := seedPrev)
      (contrib := node.vjp ctx d seedOut)
      (suffix := #[outValue]) (by
        intro i hi
        simpa using hspec i hi)
    simpa [Array.append_assoc, Array.append_empty, Array.empty_append, Array.append_singleton]
      using hfold
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      Tape.backwardDenseFromStep, hnodeLast, haccLast, hreqNd, Result.pure_eq_ok, Result.bind_ok,
      Bool.true_eq_false, ite_false]
  rw [dite_eq_left hshapeNode, SomeTensor.ofTensor_cast, hbackLast, Result.bind_ok]
  exact hfoldLast

/--
**Inductive step of the backward link.** Let `t` be a tape whose dense reverse loop agrees with the
reverse program `bp` on every seed (after shape erasure), and whose size, `requiresGrad` flags,
stored values and backward-pointing invariant are those of a lowered prefix over the context `ss`.
Appending the lowered node `node` yields a tape whose dense reverse loop, seeded by
`snoc seedPrev seedOut`, returns `snoc (bp (seedPrev + node.vjp ctx d seedOut)) seedOut`.

It is the `snoc` case of `GraphData.backwardDenseFrom_toTape`: the `snoc`
cases of `backpropAllCtx` unfold to exactly this shape.
-/
theorem NodeData.backwardDenseFrom_toNode {Δ : Type} {ss : List Shape} {τ : Shape}
    (t : Tape (SomeTensor α)) (name : Option String) (node : NodeData α Δ ss τ)
    (ctx : TensorPack α ss) (d : Δ)
    (bp : TensorPack α ss → TensorPack α ss)
    (hsize : t.nodes.size = ss.length)
    (hreq : ∀ i (hi : i < t.nodes.size), (t.nodes[i]'hi).requiresGrad = true)
    (hvals : t.nodes.map (fun n => n.value) =
      TensorPack.erase (α := α) (ss := ss) ctx)
    (hpids : BackwardPidsLt t)
    (hbp : ∀ s : TensorPack α ss,
      Tape.backwardDenseFrom (t := t) (TensorPack.erase (α := α) s) =
        .ok (TensorPack.erase (α := α) (bp s)))
    (seedPrev : TensorPack α ss) (seedOut : Tensor α τ) :
    Tape.backwardDenseFrom (t := (t.addNode (toNode (α := α) name node ctx d)).1)
        (TensorPack.erase (α := α) (ss := ss ++ [τ])
          (Pack.snoc (ss := ss) (τ := τ) seedPrev seedOut)) =
      .ok (TensorPack.erase (α := α) (ss := ss ++ [τ])
        (Pack.snoc (ss := ss) (τ := τ)
          (bp (TensorPack.add (α := α) (ss := ss) seedPrev
            (node.vjp ctx d seedOut)))
          seedOut)) := by
  have hstepLast :=
    backwardDenseFromStep_toNode t name node ctx d seedPrev seedOut hsize hreq hvals
  generalize hnd : toNode (α := α) name node ctx d = nd at hstepLast ⊢
  generalize hsp :
    TensorPack.add (α := α) (ss := ss) seedPrev (node.vjp ctx d seedOut) =
      seedPrev' at hstepLast ⊢
  have htNextSize : (t.addNode nd).1.nodes.size = t.nodes.size + 1 := by
    simp [Tape.addNode]
  have hsizeCheck :
      (TensorPack.erase (α := α) (ss := ss ++ [τ])
        (Pack.snoc (ss := ss) (τ := τ) seedPrev seedOut)).size =
      (t.addNode nd).1.nodes.size := by
    simp [htNextSize, hsize, TensorPack.size_erase]
  have hsizePrev' :
      (TensorPack.erase (α := α) (ss := ss) seedPrev').size =
        t.nodes.size := by
    simp [hsize, TensorPack.size_erase]
  have hloopPrev :
      Tape.backwardDenseFromLoop (t := t) t.nodes.size
          (TensorPack.erase (α := α) (ss := ss) seedPrev') =
        .ok (TensorPack.erase (α := α) (ss := ss)
          (bp seedPrev')) := by
    have h := hbp seedPrev'
    unfold Tape.backwardDenseFrom at h
    rwa [ite_eq_left hsizePrev'] at h
  have hloopFinal :
      Tape.backwardDenseFromLoop (t := (t.addNode nd).1) t.nodes.size
          ((TensorPack.erase (α := α) (ss := ss) seedPrev').push
            (SomeTensor.ofTensor seedOut)) =
        .ok ((TensorPack.erase (α := α) (ss := ss)
          (bp seedPrev')).push (SomeTensor.ofTensor seedOut)) := by
    rw [backwardDenseFromLoop_addNode_push t nd (SomeTensor.ofTensor seedOut) hpids
      t.nodes.size le_rfl _ hsizePrev', hloopPrev]
    rfl
  unfold Tape.backwardDenseFrom
  rw [ite_eq_left hsizeCheck, htNextSize, TensorPack.erase_snoc,
    Tape.backwardDenseFromLoop, hstepLast, Result.bind_ok, hloopFinal,
    TensorPack.erase_snoc]

end Tape
end TorchLean
