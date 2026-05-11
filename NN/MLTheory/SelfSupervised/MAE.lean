/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.SelfSupervised.Masking

/-!
# Masked Autoencoder Objective Semantics

This module formalizes the finite patch/token core of a masked autoencoder (MAE):

- an input is a finite collection of patches `Fin n → Patch`;
- an encoder/decoder stack is abstracted to a reconstruction function;
- the objective is a sum of per-patch losses over the masked indices.

The formalization is intentionally small. It captures the semantics that examples and future model
helpers should preserve, while leaving ViT blocks, convolutional patch embeddings, and image IO in
the executable API layer.

Paper anchor: “Masked Autoencoders Are Scalable Vision Learners” (He, Chen, Xie, Li, Dollár,
Girshick, 2021), arXiv:2111.06377.  The key objective-level fact we encode is that the
reconstruction loss is taken over the masked patch set.  Therefore the objective should not depend
on an arbitrary ordering of masked patch indices; `maeLoss_reverse` is the small finite theorem
capturing that property.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/-- A finite patch collection. -/
abbrev PatchBatch (n : Nat) (Patch : Type) := Fin n → Patch

/-- Reconstruct every patch using a reconstruction function. -/
def reconstruct {n : Nat} {Patch Pred : Type}
    (decode : Fin n → Pred → Patch) (pred : Fin n → Pred) : PatchBatch n Patch :=
  fun i => decode i (pred i)

/-- Exact reconstruction predicate for all patches. -/
def ExactReconstruction {n : Nat} {Patch : Type} (x y : PatchBatch n Patch) : Prop :=
  ∀ i, y i = x i

/--
MAE-style masked reconstruction loss over an explicit masked-index list.

The list is the serialized representation of the masked set. The theorems below prove that the
objective behaves like a set sum for basic reorderings/decompositions.
-/
def maeLoss {n : Nat} {Patch Pred : Type}
    (maskedIdxs : List (Fin n))
    (target : PatchBatch n Patch)
    (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) : Nat :=
  maskedLoss maskedIdxs (fun i => patchLoss (target i) (pred i))

@[simp] theorem maeLoss_nil {n : Nat} {Patch Pred : Type}
    (target : PatchBatch n Patch) (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    maeLoss ([] : List (Fin n)) target pred patchLoss = 0 := by
  simp [maeLoss]

theorem maeLoss_append {n : Nat} {Patch Pred : Type}
    (xs ys : List (Fin n)) (target : PatchBatch n Patch) (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    maeLoss (xs ++ ys) target pred patchLoss =
      maeLoss xs target pred patchLoss + maeLoss ys target pred patchLoss := by
  simp [maeLoss, maskedLoss_append]

/--
The MAE loss is invariant under reversing the order of the masked-index list.  This is the small
formal version of “masked reconstruction is a set objective, not an ordering objective.”
-/
theorem maeLoss_reverse {n : Nat} {Patch Pred : Type}
    (idxs : List (Fin n)) (target : PatchBatch n Patch) (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat) :
    maeLoss idxs.reverse target pred patchLoss =
      maeLoss idxs target pred patchLoss := by
  simp [maeLoss, maskedLoss_reverse]

/-- If every selected patch has zero reconstruction loss, the masked MAE loss is zero. -/
theorem maeLoss_eq_zero_of_patch_losses_zero {n : Nat} {Patch Pred : Type}
    (idxs : List (Fin n)) (target : PatchBatch n Patch) (pred : Fin n → Pred)
    (patchLoss : Patch → Pred → Nat)
    (h : ∀ i ∈ idxs, patchLoss (target i) (pred i) = 0) :
    maeLoss idxs target pred patchLoss = 0 := by
  exact maskedLoss_eq_zero_of_all_zero idxs (fun i => patchLoss (target i) (pred i)) h

/-- Reconstructing with the identity decoder/prediction is exact. -/
theorem exactReconstruction_identity {n : Nat} {Patch : Type} (x : PatchBatch n Patch) :
    ExactReconstruction x (reconstruct (fun _ p => p) x) := by
  intro i
  rfl

end NN.MLTheory.SelfSupervised
