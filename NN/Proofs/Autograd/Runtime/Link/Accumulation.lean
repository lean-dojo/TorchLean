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
            have hlt : pref.size < (pref.push seedHeadValue).size := by
              simp
            simp [Array.getElem?_append]

          have hsummed :
              Runtime.Autograd.SomeTensor.add seedHeadValue contribHeadValue =
                .ok newHeadValue := by
            -- Reduce the shape-cast using definitional equality of shapes.
            have hs :
                (Spec.SomeTensor.ofTensor seedHead).shape =
                  (Spec.SomeTensor.ofTensor contribHead).shape := by
              rfl
            cases hs
            simp [Runtime.Autograd.SomeTensor.add, Spec.SomeTensor.ofTensor,
              seedHeadValue, contribHeadValue, newHeadValue]

          have hset :
              ((pref.push seedHeadValue) ++
                TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                suffix).set
                  pref.size newHeadValue
                  (by
                    simp [Array.size_append, Nat.add_assoc]
                    omega) =
                (pref.push newHeadValue) ++
                  TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                  suffix := by
            have hlt : pref.size < (pref.push seedHeadValue).size := by
              simp
            simp [Array.set_append_left (xs := pref.push seedHeadValue)
                  (ys := TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                    suffix)
                  (i := pref.size) (x := newHeadValue) hlt,
              Array.set_push]

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
              have hidAcc :
                  pref.size <
                  ((pref.push seedHeadValue) ++
                    TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                    suffix).size := by
                simp [Array.size_append, Nat.add_assoc]
                omega
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
              have hidAcc' :
                  pref.size <
                    ((pref.push seedHeadValue) ++
                      (TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := ss) seedTail ++
                      suffix)).size := by
                simpa [Array.append_assoc] using hidAcc

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

                have hid :
                    pref.size < pref.size + 1 + (ss.length + suffix.size) := by
                  -- `pref.size < pref.size + 1` and adding to the RHS preserves `<`.
                  exact Nat.lt_of_lt_of_le (Nat.lt_succ_self pref.size)
                    (Nat.le_add_right (pref.size + 1) (ss.length + suffix.size))

                -- Now `addGradAll` is a straight-line computation: fetch node, check flags/shapes,
                -- add, and overwrite the `pref.size` slot.
                -- Keep `Tensor.cast_shape` opaque here: the following tensor equalities are about
                -- the accumulated value, not the proof terms used to align shapes.
                simp [Runtime.Autograd.Tape.addGradAll, hnode0', hreq0, hid, seedHeadValue,
                  contribHeadValue, newHeadValue, Array.set_push]

                simp [Runtime.Autograd.SomeTensor.add]

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
                have hxs : (#[(Spec.SomeTensor.ofTensor seedHead)] : Array (Spec.SomeTensor
                  α)).size ≤ i + 1 := by
                  simp
                have : (i + 1) - (#[(Spec.SomeTensor.ofTensor seedHead)] : Array
                  (Spec.SomeTensor α)).size = i := by
                  simp
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
`Tape.addGradAll` never changes the size of the dense gradient array in the `.ok` case.

This is a structural property needed to show the runtime reverse loop preserves array sizes.
-/
theorem addGradAll_size_preserved {α : Type}
    [TorchLean.Storage α] [Add α]
    (t : Runtime.Autograd.Tape α) (grads : Array (Spec.SomeTensor α)) (id : Nat) (g :
      Spec.SomeTensor α) :
    match Runtime.Autograd.Tape.addGradAll (t := t) grads id g with
    | .ok grads' => grads'.size = grads.size
    | .error _ => True := by
  cases hresult : Runtime.Autograd.Tape.addGradAll (t := t) grads id g with
  | error => simp
  | ok grads' =>
      simp only [Runtime.Autograd.Tape.addGradAll, Pure.pure, Except.pure, Bind.bind, Except.bind,
        throw, throwThe, MonadExceptOf.throw] at hresult
      repeat' split at hresult
      all_goals simp_all
      subst grads'
      simp

/-- If `addGradAll` returns `.ok grads'`, then `grads'.size = grads.size`. -/
theorem addGradAll_ok_size {α : Type}
    [TorchLean.Storage α] [Add α]
    (t : Runtime.Autograd.Tape α) :
    ∀ {grads : Array (Spec.SomeTensor α)} {id : Nat} {g : Spec.SomeTensor α}
      {grads' : Array (Spec.SomeTensor α)},
      Runtime.Autograd.Tape.addGradAll (t := t) grads id g = .ok grads' →
        grads'.size = grads.size := by
  intro grads id g grads' h
  simpa [h] using addGradAll_size_preserved (t := t) grads id g

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
        cases head with
        | mk pid pg =>
            cases h1 : Runtime.Autograd.Tape.addGradAll (t := t) acc0 pid pg with
            | error e =>
                simp [List.foldlM, h1] at hfold
                cases hfold
            | ok acc1 =>
                have htail :
                    tail.foldlM
                        (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
                          pg) acc1 =
                      .ok accOut := by
                  simpa [List.foldlM, Bind.bind, Except.bind, Pure.pure, Except.pure, h1]
                    using hfold
                have hs1 : acc1.size = acc0.size :=
                  addGradAll_ok_size (t := t) (grads := acc0) (id := pid) (g := pg) (grads' := acc1)
                    (by
                    simpa using h1)
                have := ih (acc0 := acc1) (accOut := accOut) htail
                simpa [hs1] using this

  have fold_ok_size :
      ∀ (contribs : Array (Nat × Spec.SomeTensor α)) (acc0 accOut : Array (Spec.SomeTensor α)),
        (contribs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := t) acc2 pid
          pg) acc0 = .ok accOut) → accOut.size = acc0.size := by
    intro contribs acc0 accOut hfold
    apply list_fold_ok_size contribs.toList acc0 accOut
    simpa only [Array.foldlM_toList] using hfold

  simp only [Runtime.Autograd.Tape.backwardDenseFromStep, Pure.pure, Except.pure, Bind.bind,
    Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals simp_all
  exact fold_ok_size _ _ _ h


end Graph

end Algebra
end Autograd
end Proofs
