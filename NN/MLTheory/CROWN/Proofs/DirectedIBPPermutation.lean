/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import all NN.MLTheory.CROWN.Graph.Engine.Base
public import NN.MLTheory.CROWN.Proofs.DirectedIBPTensor

/-!
# Directed enclosure through shape-erased coordinate operations

The runtime implements permutations by adjacent swaps. Each swap reads one input coordinate,
so the same enclosure proof applies to every accepted permutation, including empty tensors.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.DirectedBackward

open Spec TorchLean TorchLean.Tensor
open NN.MLTheory.CROWN

noncomputable section

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

/-- Shape-erased endpoint tensors have the same shape as an enclosed exact real tensor. -/
inductive SomeTensorEncloses : SomeTensor α → SomeTensor α → SomeTensor ℝ → Prop
  | mk {s : Shape} {lo hi : Tensor α s} {x : Tensor ℝ s}
      (bounds : TensorEncloses lo hi x) :
      SomeTensorEncloses ⟨s, lo⟩ ⟨s, hi⟩ ⟨s, x⟩

/-- Successful endpoint and real evaluations preserve shape and coordinate enclosure. -/
def TensorTransferEncloses
    (op : SomeTensor α → Except String (SomeTensor α))
    (exactOp : SomeTensor ℝ → Except String (SomeTensor ℝ)) : Prop :=
  ∀ {lo hi x}, SomeTensorEncloses lo hi x →
    ∀ {outLo outHi outX}, op lo = .ok outLo → op hi = .ok outHi →
      exactOp x = .ok outX → SomeTensorEncloses outLo outHi outX

theorem TensorEncloses.swapAdjacentAxes {s : Shape}
    {lo hi : Tensor α s} {x : Tensor ℝ s} (h : TensorEncloses lo hi x) (depth : Nat) :
    TensorEncloses (Tensor.swapAdjacentAxes lo depth) (Tensor.swapAdjacentAxes hi depth)
      (Tensor.swapAdjacentAxes x depth) := by
  intro c
  simpa only [Tensor.swapAdjacentAxes_apply] using
    h (Tensor.Internal.swapAdjacentAxesCoordinate s depth c)

theorem SomeTensorEncloses.swapAdjacentAtDepth {lo hi : SomeTensor α} {x : SomeTensor ℝ}
    (h : SomeTensorEncloses lo hi x) (depth : Nat) :
    SomeTensorEncloses (lo.swapAdjacentAtDepth depth) (hi.swapAdjacentAtDepth depth)
      (x.swapAdjacentAtDepth depth) := by
  cases h with
  | mk h => exact .mk (h.swapAdjacentAxes depth)

theorem SomeTensorEncloses.swapFold {lo hi : SomeTensor α} {x : SomeTensor ℝ}
    (h : SomeTensorEncloses lo hi x) (depths : List Nat) :
    SomeTensorEncloses
      (depths.foldl SomeTensor.swapAdjacentAtDepth lo)
      (depths.foldl SomeTensor.swapAdjacentAtDepth hi)
      (depths.foldl SomeTensor.swapAdjacentAtDepth x) := by
  induction depths generalizing lo hi x with
  | nil => exact h
  | cons depth depths ih => exact ih (h.swapAdjacentAtDepth depth)

/-- The checked permutation evaluator applies the same swaps to all three tensors. -/
theorem permuteSomeTensor_encloses (perm : Array Nat) :
    TensorTransferEncloses
      (fun x : SomeTensor α => NN.IR.Graph.permuteSomeTensor x perm)
      (fun x : SomeTensor ℝ => NN.IR.Graph.permuteSomeTensor x perm) := by
  intro lo hi x h outLo outHi outX hlo hhi hx
  cases h with
  | @mk s lo hi x h =>
      unfold NN.IR.Graph.permuteSomeTensor at hlo hhi hx
      cases hs : Shape.permute? s perm.toList with
      | none => simp [hs, NN.IR.throw_eq_error] at hlo
      | some shape =>
          cases hd : NN.IR.Graph.swapDepthsForPerm perm s.rank with
          | error e => simp [hs, hd] at hlo
          | ok depths =>
              simp only [hs, hd, Bind.bind, Except.bind, Pure.pure, Except.pure,
                Except.ok.injEq] at hlo hhi hx
              subst outLo outHi outX
              simpa only [← Array.foldl_toList] using
                (SomeTensorEncloses.mk h).swapFold depths.toList

/-- Exact transpose uses the runtime's validated transposition permutation. -/
def transposeTensor (axis₁ axis₂ : Nat) {β : Type} [Storage β] [Context β]
    (x : SomeTensor β) : Except String (SomeTensor β) := do
  let perm ← NN.IR.OpContracts.transposePerm x.shape.rank axis₁ axis₂
  NN.IR.Graph.permuteSomeTensor x perm

theorem transposeTensor_encloses (axis₁ axis₂ : Nat) :
    TensorTransferEncloses (transposeTensor (β := α) axis₁ axis₂)
      (transposeTensor (β := ℝ) axis₁ axis₂) := by
  intro lo hi x h outLo outHi outX hlo hhi hx
  cases h with
  | @mk s lo hi x h =>
      unfold transposeTensor at hlo hhi hx
      cases hp : NN.IR.OpContracts.transposePerm s.rank axis₁ axis₂ with
      | error e => simp only [hp, Bind.bind, Except.bind, reduceCtorEq] at hlo
      | ok perm =>
          simp only [hp, Bind.bind, Except.bind] at hlo hhi hx
          exact permuteSomeTensor_encloses perm (.mk h) hlo hhi hx

/-- Applying an enclosing shape-erased transfer to endpoint tensors encloses its exact result. -/
theorem ibpMonotoneSomeTensor?_encloses
    {s t : Shape} {B box : FlatBox α} {f : Nat → ℝ} {y : Tensor ℝ t}
    {op : SomeTensor α → Except String (SomeTensor α)}
    {exactOp : SomeTensor ℝ → Except String (SomeTensor ℝ)}
    (hop : TensorTransferEncloses op exactOp)
    (hB : RowEncloses B s.size f)
    (hy : exactOp ⟨s, realTensor s f⟩ = .ok ⟨t, y⟩)
    (hbox : ibpMonotoneSomeTensor? s t op B = some box) :
    RowEncloses box t.size (tensorValues y) := by
  unfold ibpMonotoneSomeTensor? at hbox
  split at hbox
  next hd =>
    have hin := tensorEncloses_ibpUnflatten hd hB
    generalize hl :
      op ⟨s, Tensor.reshapeSpec B.lo (by simpa [Shape.size] using hd)⟩ = lower at hbox
    generalize hu :
      op ⟨s, Tensor.reshapeSpec B.hi (by simpa [Shape.size] using hd)⟩ = upper at hbox
    cases lower with
    | error e => simp only [hl, reduceCtorEq] at hbox
    | ok lower =>
        cases upper with
        | error e => simp only [hl, hu, reduceCtorEq] at hbox
        | ok upper =>
            have hin' :
                SomeTensorEncloses
                  ⟨s, Tensor.reshapeSpec B.lo (by simpa [Shape.size] using hd)⟩
                  ⟨s, Tensor.reshapeSpec B.hi (by simpa [Shape.size] using hd)⟩
                  ⟨s, realTensor s f⟩ := by
              apply SomeTensorEncloses.mk
              obtain ⟨d, lo, hi⟩ := B
              dsimp only at hd
              subst d
              simpa only [ibpUnflatten, eq_mp_eq_cast, cast_eq, Tensor.reshapeSpec,
                Tensor.unflattenSpec] using hin
            have hout := hop hin' hl hu hy
            cases hout with
            | mk h =>
                simp only [hl, hu, dite_true, Option.some.injEq] at hbox
                subst box
                exact h.tensorValues
  next => contradiction

end

end NN.MLTheory.CROWN.Graph.DirectedBackward
