/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.ShapeErasure
public import TorchLean.NN.Tape.Backward

/-!
# Runtime Gradient Accumulation Link

Gradients on the runtime tape accumulate in a dense array of shape-erased tensors, and the proved
autograd algebra adds typed contexts with `TorchLean.TensorPack.add`. Folding the runtime updates
of an indexed `TorchLean.TensorPack` into that array leaves the shape-erased sum at the same
positions, so the two accumulations agree. A successful accumulation or backward step also keeps
the size of the array.
-/

@[expose] public section

namespace TorchLean
namespace Tape

open TorchLean.Tensor

/--
Folding `Tape.addGradAll` over the contributions of a `TorchLean.TensorPack` (indexed by
`eraseIndexed`) into the array layout `pref ++ seed ++ suffix` succeeds, and the
result holds the pointwise sum `TorchLean.TensorPack.add` of the typed contexts at the positions of
the seed. The hypothesis asks that every seed position hold a tape node that requires a gradient
and has the shape of its seed entry.

Runtime accumulation therefore matches the addition of the proved algebra.
-/
theorem foldlM_addGradAll_eraseIndexed {α : Type}
    [Storage α] [Add α]
    (t : Tape (SomeTensor α)) :
    ∀ {ss : List Shape} (pref : Array (SomeTensor α))
      (seed contrib : TensorPack α ss)
      (suffix : Array (SomeTensor α)),
      (∀ i (hi : i < ss.length),
        let id := pref.size + i
        ∃ node : Node (SomeTensor α),
          t.getNode? id = some node ∧
            node.requiresGrad = true ∧
            node.value.shape =
              ((TensorPack.erase (α := α) (ss := ss) seed)[i]'(by
                  simpa [TensorPack.size_erase] using hi)).shape) →
      (TensorPack.eraseIndexed (α := α) (ss := ss) contrib pref.size).foldlM
          (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid pg)
          (pref ++ TensorPack.erase (α := α) (ss := ss) seed ++ suffix) =
        .ok (pref ++
              TensorPack.erase (α := α) (ss := ss)
                (TensorPack.add (α := α) (ss := ss) seed contrib) ++
              suffix) := by
  intro ss pref seed contrib suffix hnodes
  rw [← Array.foldlM_toList]
  induction ss generalizing pref with
  | nil =>
      cases seed; cases contrib
      simp [TensorPack.eraseIndexed, TensorPack.erase,
        TensorPack.add]
      rfl
  | cons s ss ih =>
      cases seed with
      | cons seedHead seedTail =>
        cases contrib with
        | cons contribHead contribTail =>
          let seedHeadValue : SomeTensor α := SomeTensor.ofTensor seedHead
          let contribHeadValue : SomeTensor α := SomeTensor.ofTensor contribHead
          let newHeadValue : SomeTensor α := SomeTensor.ofTensor (seedHead +
            contribHead)

          have hseedArr :
              TensorPack.erase (α := α) (ss := s :: ss)
                  (Pack.cons seedHead seedTail) =
                #[seedHeadValue] ++
                  TensorPack.erase (α := α) (ss := ss) seedTail := by
            simpa [seedHeadValue] using
              (TensorPack.erase_cons (α := α) (s := s) (ss := ss)
                seedHead seedTail)

          have hacc0 :
              pref ++
                  TensorPack.erase (α := α) (ss := s :: ss)
                    (Pack.cons seedHead seedTail) ++
                  suffix =
                (pref.push seedHeadValue) ++
                  TensorPack.erase (α := α) (ss := ss) seedTail ++
                  suffix := by
            -- Avoid `simp` loops between `push` and `++ #[x]`.
            -- Expand the seed array, reassociate, then rewrite `pref ++ #[x]` as `pref.push x`.
            rw [hseedArr]
            simp [Array.append_assoc]

          have h0 := hnodes 0 (by simp [List.length_cons])
          rcases h0 with ⟨node0, hnode0, hreq0, hshape0'⟩

          have hshape0 : node0.value.shape = seedHeadValue.shape := by
            have : ((TensorPack.erase (α := α) (ss := s :: ss)
                (Pack.cons seedHead seedTail))[0]'(by
                  simp [TensorPack.size_erase, List.length_cons])).shape =
                  seedHeadValue.shape := by
              simp [hseedArr, seedHeadValue]
            exact hshape0'.trans this

          have hgetExisting :
              ((pref.push seedHeadValue) ++
                TensorPack.erase (α := α) (ss := ss) seedTail ++
                suffix)[pref.size]? =
                some seedHeadValue := by
            simp [Array.getElem?_append]

          have hadd0 :
              Tape.addGradAll (t := t)
                  (pref ++
                    TensorPack.erase (α := α) (ss := s :: ss)
                      (Pack.cons seedHead seedTail)
                    ++ suffix)
                  pref.size contribHeadValue =
                .ok ((pref.push newHeadValue) ++
                  TensorPack.erase (α := α) (ss := ss) seedTail ++
                  suffix) := by
              have hshapeG : contribHeadValue.shape = node0.value.shape := by
                calc
                  contribHeadValue.shape = seedHeadValue.shape := by rfl
                  _ = node0.value.shape := hshape0.symm
              have hshapeExisting : seedHeadValue.shape = node0.value.shape := by
                simpa using hshape0.symm
              have hnode0' : t.getNode? pref.size = some node0 := by
                simpa [Nat.add_zero] using hnode0
              have hgetExisting' :
                  ((pref.push seedHeadValue) ++
                    (TensorPack.erase (α := α) (ss := ss) seedTail ++
                    suffix))[pref.size]? =
                    some seedHeadValue := by
                simpa [Array.append_assoc] using hgetExisting

              have : Tape.addGradAll (t := t)
                  ((pref.push seedHeadValue) ++
                    (TensorPack.erase (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadValue =
                    .ok ((pref.push newHeadValue) ++
                      (TensorPack.erase (α := α) (ss := ss) seedTail ++
                        suffix)) := by
                -- After rewriting `node0.value.shape = seedHeadValue.shape`, all shape casts become
                -- definitional.
                cases hshape0
                -- Reduce the dependent shape-casts by eliminating the equality proofs.
                cases hshapeExisting
                cases hshapeG

                simp only [GradValue.shape_someTensor, GradValue.retag_someTensor,
                    GradValue.check_someTensor, GradValue.add_someTensor, GradValue.drop_someTensor,
                    Tape.addGradAll, hnode0', hreq0, Bool.true_eq_false, ite_false,
                    Bind.bind, Except.bind, Pure.pure, Except.pure]
                rw [dite_eq_left (show contribHeadValue.shape = node0.value.shape from rfl)]
                split
                · rename_i hget
                  simp [hgetExisting'] at hget
                · rename_i existing hget
                  have heq : existing = seedHeadValue :=
                    Option.some.inj (hget.symm.trans hgetExisting')
                  subst existing
                  simp [seedHeadValue, contribHeadValue, newHeadValue, Array.set_push,
                    SomeTensor.add, SomeTensor.cast]
                  rfl

              -- Rewrite the goal with `hacc0` and reassociate both sides into the form of `this`.
              rw [hacc0]
              rw [Array.append_assoc]
              conv_rhs => rw [Array.append_assoc]
              exact this

          have hnodesTail :
              ∀ i (hi : i < ss.length),
                let id := (pref.push newHeadValue).size + i
                ∃ node : Node (SomeTensor α),
                  t.getNode? id = some node ∧ node.requiresGrad = true ∧
                    node.value.shape =
                      ((TensorPack.erase (α := α) (ss := ss) seedTail)[i]'(by
                          simpa [TensorPack.size_erase]
                            using hi)).shape := by
            intro i hi
            have h' :=
              hnodes (i + 1) (by
                simpa [List.length_cons] using Nat.succ_lt_succ hi)
            rcases h' with ⟨node, hnode, hreq, hshape⟩
            refine ⟨node, ?_, hreq, ?_⟩
            · simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using hnode
            ·
              have hiFull :
                  i + 1 < (TensorPack.erase (α := α) (ss := s :: ss)
                    (Pack.cons seedHead seedTail)).size := by
                have : i + 1 < (s :: ss).length := by
                  simpa [List.length_cons] using Nat.succ_lt_succ hi
                simpa [TensorPack.size_erase] using this
              have hiTail :
                  i < (TensorPack.erase (α := α) (ss := ss)
                    seedTail).size := by
                simpa [TensorPack.size_erase] using hi
              have hidx :
                  (TensorPack.erase (α := α) (ss := s :: ss)
                    (Pack.cons seedHead seedTail))[i + 1]'hiFull =
                    (TensorPack.erase (α := α) (ss := ss)
                      seedTail)[i]'hiTail := by
                have hcons :=
                  (TensorPack.erase_cons (α := α) (s := s) (ss := ss)
                    seedHead seedTail)
                simp [hcons]
              have hshape' :
                  ((TensorPack.erase (α := α) (ss := s :: ss)
                    (Pack.cons seedHead seedTail))[i + 1]'hiFull).shape =
                    ((TensorPack.erase (α := α) (ss := ss)
                      seedTail)[i]'hiTail).shape := by
                simpa using congrArg SomeTensor.shape hidx
              simpa [hshape'] using hshape

          have htail :=
            ih (pref := pref.push newHeadValue) (seed := seedTail) (contrib := contribTail)
              hnodesTail

          simp only [TensorPack.eraseIndexed, Array.toList_append]
          rw [List.singleton_append, List.foldlM_cons]
          simp only [Bind.bind, Except.bind]
          rw [hadd0]
          simpa [TensorPack.add, TensorPack.erase_cons,
              Array.append_assoc, newHeadValue]
            using htail

/--
A successful `Tape.addGradAll` keeps the size of the dense gradient array, the property that lets
the runtime reverse loop preserve array sizes.
-/
theorem addGradAll_ok_size {α : Type}
    [Storage α] [Add α]
    (t : Tape (SomeTensor α)) {grads : Array (SomeTensor α)} {id : Nat}
    {g : SomeTensor α} {grads' : Array (SomeTensor α)}
    (h : Tape.addGradAll (t := t) grads id g = .ok grads') :
    grads'.size = grads.size := by
  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      GradValue.add_someTensor, GradValue.drop_someTensor, Tape.addGradAll,
      Pure.pure, Except.pure, Bind.bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals simp_all
  subst grads'
  simp

/--
If one step of the runtime dense backward loop succeeds, it preserves the accumulator array size.

The proof shows that the internal `foldlM addGradAll` preserves size and then splits on the control
flow of `backwardDenseFromStep`.
-/
theorem backwardDenseFromStep_ok_size {α : Type}
    [Storage α] [Add α]
    (t : Tape (SomeTensor α)) :
    ∀ {acc : Array (SomeTensor α)} {id : Nat} {acc' : Array (SomeTensor α)},
      Tape.backwardDenseFromStep (t := t) acc id = .ok acc' →
        acc'.size = acc.size := by
  intro acc id acc' h
  -- `foldlM` over `addGradAll` preserves `size` in the `.ok` case.
  have list_fold_ok_size :
      ∀ (contribs : List (Nat × SomeTensor α)) (acc0 accOut : Array (SomeTensor α)),
        (contribs.foldlM (fun acc2 (pid, pg) => Tape.addGradAll (t := t) acc2 pid
          pg) acc0 =
            .ok accOut) →
          accOut.size = acc0.size := by
    intro contribs acc0 accOut hfold
    induction contribs generalizing acc0 accOut with
    | nil =>
        simp [List.foldlM] at hfold
        cases hfold
        rfl
    | cons head tail ih =>
        obtain ⟨pid, pg⟩ := head
        cases hstep : Tape.addGradAll (t := t) acc0 pid pg with
        | error error => simp [List.foldlM, hstep, Bind.bind, Except.bind] at hfold
        | ok acc1 =>
            have htail := ih (acc0 := acc1) (accOut := accOut)
              (by simpa [List.foldlM, hstep, Bind.bind, Except.bind] using hfold)
            exact htail.trans (addGradAll_ok_size (t := t) hstep)

  simp only [GradValue.shape_someTensor, GradValue.retag_someTensor, GradValue.check_someTensor,
      Tape.backwardDenseFromStep, Pure.pure, Except.pure, Bind.bind, Except.bind,
      throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals simp_all
  rw [← Array.foldlM_toList] at h
  exact list_fold_ok_size _ _ _ h

end Tape
end TorchLean
