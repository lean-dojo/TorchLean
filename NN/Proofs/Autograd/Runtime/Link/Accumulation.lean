/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.ShapeErasure
public import NN.Runtime.Autograd.Engine.Core.Backward

/-!
# Runtime Gradient Accumulation Link

This file connects the executable dense-gradient array used by the runtime tape to the typed context
addition used in the proved autograd algebra. The main lemmas show that folding runtime gradient
updates over indexed tensors agrees with the proof-level `TorchLean.TensorPack` accumulation
operation.
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

/--
Key accumulation lemma for the runtime dense gradient array:

Folding `Tape.addGradAll` over the contributions corresponding to a `TorchLean.TensorPack` (via
`toIndexedShapeErasedArray`) is equivalent to pointwise addition of the typed contexts
(`TorchLean.TensorPack.add`), embedded back into the array layout `pref ++ seed ++ suffix`.

This is the “runtime accumulation matches proved addition” bridge.
-/
theorem foldlM_addGradAll_toIndexedShapeErasedArray_eq_add {α : Type}
    [TorchLean.Storage α] [Add α]
    (t : Runtime.Autograd.Tape α) :
    ∀ {ss : List Shape} (pref : Array (Spec.SomeTensor α))
      (seed contrib : TorchLean.TensorPack α ss)
      (suffix : Array (Spec.SomeTensor α)),
      (∀ i (hi : i < ss.length),
        let id := pref.size + i
        ∃ node : Runtime.Autograd.Node α,
          t.getNode? id = some node ∧
            node.requiresGrad = true ∧
            node.value.shape =
              ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seed)[i]'(by
                  simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape) →
      (TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := ss) contrib pref.size).foldlM
          (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid pg)
          (pref ++ TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seed ++ suffix) =
        .ok (pref ++
              TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
                (TorchLean.TensorPack.add (α := α) (ss := ss) seed contrib) ++
              suffix) := by
  intro ss pref seed contrib suffix hnodes
  rw [← Array.foldlM_toList]
  induction ss generalizing pref with
  | nil =>
      cases seed; cases contrib
      simp [TorchLean.TensorPack.toIndexedShapeErasedArray, TorchLean.TensorPack.toShapeErasedArray,
        TorchLean.TensorPack.add]
      rfl
  | cons s ss ih =>
      cases seed with
      | cons seedHead seedTail =>
        cases contrib with
        | cons contribHead contribTail =>
          let seedHeadValue : Spec.SomeTensor α := Spec.SomeTensor.ofTensor seedHead
          let contribHeadValue : Spec.SomeTensor α := Spec.SomeTensor.ofTensor contribHead
          let newHeadValue : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (addSpec seedHead
            contribHead)

          have hseedArr :
              TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                  (TorchLean.TensorPack.cons seedHead seedTail) =
                #[seedHeadValue] ++
                  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail := by
            simpa [seedHeadValue] using
              (TorchLean.TensorPack.toShapeErasedArray_cons (α := α) (s := s) (ss := ss)
                seedHead seedTail)

          have hacc0 :
              pref ++
                  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                    (TorchLean.TensorPack.cons seedHead seedTail) ++
                  suffix =
                (pref.push seedHeadValue) ++
                  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                  suffix := by
            -- Avoid `simp` loops between `push` and `++ #[x]`.
            -- Expand the seed array, reassociate, then rewrite `pref ++ #[x]` as `pref.push x`.
            rw [hseedArr]
            simp [Array.append_assoc]

          have h0 := hnodes 0 (by simp [List.length_cons])
          rcases h0 with ⟨node0, hnode0, hreq0, hshape0'⟩

          have hshape0 : node0.value.shape = seedHeadValue.shape := by
            have : ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                (TorchLean.TensorPack.cons seedHead seedTail))[0]'(by
                  simp [TorchLean.TensorPack.size_toShapeErasedArray, List.length_cons])).shape =
                  seedHeadValue.shape := by
              simp [hseedArr, seedHeadValue]
            exact hshape0'.trans this

          have hgetExisting :
              ((pref.push seedHeadValue) ++
                TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                suffix)[pref.size]? =
                some seedHeadValue := by
            simp [Array.getElem?_append]

          have hadd0 :
              Runtime.Autograd.Tape.addGradAll (t := t)
                  (pref ++
                    TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                      (TorchLean.TensorPack.cons seedHead seedTail)
                    ++ suffix)
                  pref.size contribHeadValue =
                .ok ((pref.push newHeadValue) ++
                  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
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
                    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                    suffix))[pref.size]? =
                    some seedHeadValue := by
                simpa [Array.append_assoc] using hgetExisting

              have : Runtime.Autograd.Tape.addGradAll (t := t)
                  ((pref.push seedHeadValue) ++
                    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                    suffix))
                  pref.size contribHeadValue =
                    .ok ((pref.push newHeadValue) ++
                      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                        suffix)) := by
                -- After rewriting `node0.value.shape = seedHeadValue.shape`, all shape casts become
                -- definitional.
                cases hshape0
                -- Reduce the dependent shape-casts by eliminating the equality proofs.
                cases hshapeExisting
                cases hshapeG

                simp only [Runtime.Autograd.Tape.addGradAll, hnode0', hreq0,
                  Bool.true_eq_false, ite_false, Bind.bind, Except.bind, Pure.pure,
                  Except.pure]
                rw [dite_eq_left (show contribHeadValue.shape = node0.value.shape from rfl)]
                split
                · rename_i hget
                  simp [hgetExisting'] at hget
                · rename_i existing hget
                  have heq : existing = seedHeadValue :=
                    Option.some.inj (hget.symm.trans hgetExisting')
                  subst existing
                  simp [seedHeadValue, contribHeadValue, newHeadValue, Array.set_push,
                    Runtime.Autograd.SomeTensor.add, Spec.SomeTensor.cast]

              -- Rewrite back to the original associative form for the outer goal.
              rw [hacc0]
              rw [Array.append_assoc]
              conv_rhs => rw [Array.append_assoc]
              exact this

          have hnodesTail :
              ∀ i (hi : i < ss.length),
                let id := (pref.push newHeadValue).size + i
                ∃ node : Runtime.Autograd.Node α,
                  t.getNode? id = some node ∧ node.requiresGrad = true ∧
                    node.value.shape =
                      ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail)[i]'(by
                          simpa [TorchLean.TensorPack.size_toShapeErasedArray]
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
                  i + 1 < (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                    (TorchLean.TensorPack.cons seedHead seedTail)).size := by
                have : i + 1 < (s :: ss).length := by
                  simpa [List.length_cons] using Nat.succ_lt_succ hi
                simpa [TorchLean.TensorPack.size_toShapeErasedArray] using this
              have hiTail :
                  i < (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
                    seedTail).size := by
                simpa [TorchLean.TensorPack.size_toShapeErasedArray] using hi
              have hidx :
                  (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                    (TorchLean.TensorPack.cons seedHead seedTail))[i + 1]'hiFull =
                    (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
                      seedTail)[i]'hiTail := by
                have hcons :=
                  (TorchLean.TensorPack.toShapeErasedArray_cons (α := α) (s := s) (ss := ss)
                    seedHead seedTail)
                simp [hcons]
              have hshape' :
                  ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := s :: ss)
                    (TorchLean.TensorPack.cons seedHead seedTail))[i + 1]'hiFull).shape =
                    ((TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss)
                      seedTail)[i]'hiTail).shape := by
                simpa using congrArg Spec.SomeTensor.shape hidx
              simpa [hshape'] using hshape

          have htail :=
            ih (pref := pref.push newHeadValue) (seed := seedTail) (contrib := contribTail)
              hnodesTail

          simp only [TorchLean.TensorPack.toIndexedShapeErasedArray, Array.toList_append]
          rw [List.singleton_append, List.foldlM_cons]
          simp only [Bind.bind, Except.bind]
          rw [hadd0]
          simpa [TorchLean.TensorPack.add, TorchLean.TensorPack.toShapeErasedArray_cons,
              Array.append_assoc, newHeadValue]
            using htail

/--
`Tape.addGradAll` never changes the size of the dense gradient array when it succeeds.

This is the structural property that lets the runtime reverse loop preserve array sizes.
-/
theorem addGradAll_ok_size {α : Type}
    [TorchLean.Storage α] [Add α]
    (t : Runtime.Autograd.Tape α) {grads : Array (Spec.SomeTensor α)} {id : Nat}
    {g : Spec.SomeTensor α} {grads' : Array (Spec.SomeTensor α)}
    (h : Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads') :
    grads'.size = grads.size := by
  simp only [Runtime.Autograd.Tape.addGradAll, Pure.pure, Except.pure, Bind.bind, Except.bind,
    throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals simp_all
  subst grads'
  simp

/--
If one step of the runtime dense backward loop succeeds, it preserves the accumulator array size.

This is proved by showing the internal `foldlM addGradAll` preserves size, then splitting on the
control flow of `backwardDenseFromStep`.
-/
theorem backwardDenseFromStep_ok_size {α : Type}
    [TorchLean.Storage α] [Add α]
    (t : Runtime.Autograd.Tape α) :
    ∀ {acc : Array (Spec.SomeTensor α)} {id : Nat} {acc' : Array (Spec.SomeTensor α)},
      Runtime.Autograd.Tape.backwardDenseFromStep (t := t) acc id = .ok acc' →
        acc'.size = acc.size := by
  intro acc id acc' h
  -- `foldlM` over `addGradAll` preserves `size` in the `.ok` case.
  have list_fold_ok_size :
      ∀ (contribs : List (Nat × Spec.SomeTensor α)) (acc0 accOut : Array (Spec.SomeTensor α)),
        (contribs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
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
        cases hstep : Runtime.Autograd.Tape.addGradAll (t := t) acc0 pid pg with
        | error error => simp [List.foldlM, hstep, Bind.bind, Except.bind] at hfold
        | ok acc1 =>
            have htail := ih (acc0 := acc1) (accOut := accOut)
              (by simpa [List.foldlM, hstep, Bind.bind, Except.bind] using hfold)
            exact htail.trans (addGradAll_ok_size (t := t) hstep)

  simp only [Runtime.Autograd.Tape.backwardDenseFromStep, Pure.pure, Except.pure, Bind.bind,
    Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals simp_all
  rw [← Array.foldlM_toList] at h
  exact list_fold_ok_size _ _ _ h

end Graph

end Algebra
end Autograd
end Proofs
