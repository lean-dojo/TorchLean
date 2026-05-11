/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Core.TensorReductionShape

/-!
# NF Reduction Operators

NF (rounded) backend: approximation lemmas for reduction operators used by LayerNorm/attention.

These lemmas are specialized to 2D tensors and avoid typeclass inference by using explicit
`Shape.reducibleAlong` proofs derived from `m>0`/`n>0`.

## PyTorch correspondence / citations
This file targets reduction patterns used by normalization/attention (sums, means, maxes along an
axis), analogous to operations like `torch.sum`, `torch.mean`, and `torch.max`.
https://pytorch.org/docs/stable/generated/torch.sum.html
https://pytorch.org/docs/stable/generated/torch.mean.html
https://pytorch.org/docs/stable/generated/torch.max.html

Current scope: 2D axis reductions. That keeps the proofs explicit and avoids hiding shape
preconditions behind automation; broader-rank reductions can reuse the same pattern later.
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

set_option maxHeartbeats 4000000

private lemma one_le_nat_coe {n : Nat} (hn : 0 < n) : (1 : ℝ) ≤ (n : ℝ) := by
  have : (1 : Nat) ≤ n := Nat.succ_le_iff.2 hn
  exact_mod_cast this

-- ---------------------------------------------------------------------------
-- Definitional unfoldings for 2D reductions (axis 0/1)
-- ---------------------------------------------------------------------------

private lemma reduce_sum_axis1_get
    {α : Type} [Add α] [Zero α]
    {m n : Nat} (x : Tensor α (.dim m (.dim n .scalar)))
    (hRed : Shape.reducibleAlong 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    (match Spec.Tensor.reduceSum (α := α) (s := .dim m (.dim n .scalar)) 1 x hRed with
      | Tensor.dim f => f i) =
      Tensor.scalar (sumSpec (α := α) (s := .dim n .scalar) (getAtSpec x i)) := by
  cases x with
  | dim rows =>
      cases hRed with
      | tail hRed' =>
          cases h : rows i with
          | dim slices =>
              aesop (add simp [Spec.Tensor.reduceSum, Spec.Tensor.reduceDim,
                Spec.Tensor.reduceDim.aux,
                Spec.Tensor.reduceFirstDim, Spec.Tensor.shapeAfterSum, getAtSpec, sumSpec, h,
                hRed', tensorFoldlSpec])

private lemma reduce_mean_axis1_get
    {α : Type} [Context α]
    {m n : Nat} (x : Tensor α (.dim m (.dim n .scalar)))
    (hRed : Shape.reducibleAlong 1 (.dim m (.dim n .scalar))) (i : Fin m) :
    (match Spec.Tensor.reduceMean (α := α) (s := .dim m (.dim n .scalar)) 1 x hRed with
      | Tensor.dim f => f i) =
      Tensor.scalar (sumSpec (α := α) (s := .dim n .scalar) (getAtSpec x i) / (n : α)) := by
  cases x with
  | dim rows =>
      cases hRed with
      | tail hRed' =>
          cases h : rows i with
          | dim slices =>
              aesop (add simp [Spec.Tensor.reduceMean, Spec.Tensor.reduceSum,
                Spec.Tensor.reduceDim,
                Spec.Tensor.reduceDim.aux, Spec.Tensor.reduceFirstDim,
                  Spec.Tensor.shapeAfterSum,
                Spec.Tensor.getDimSize, getAtSpec, sumSpec, mapSpec, h, hRed',
                  tensorFoldlSpec])

private lemma reduce_sum_axis0_get
    {α : Type} [Add α] [Zero α]
    {m n : Nat} (x : Tensor α (.dim m (.dim n .scalar)))
    (hRed : Shape.reducibleAlong 0 (.dim m (.dim n .scalar))) (j : Fin n) :
    (match Spec.Tensor.reduceSum (α := α) (s := .dim m (.dim n .scalar)) 0 x hRed with
      | Tensor.dim f => f j) =
      Tensor.scalar (sumSpec (α := α) (s := .dim m .scalar)
        (Tensor.dim (fun i : Fin m => sliceSpec (getAtSpec x i) j))) := by
  cases x with
  | dim rows =>
      simp [Spec.Tensor.reduceSum, Spec.Tensor.reduceDim, Spec.Tensor.reduceDim.aux,
        Spec.Tensor.reduceFirstDim, Spec.Tensor.shapeAfterSum, getAtSpec, sliceSpec,
        sumSpec, tensorFoldlSpec]

-- ---------------------------------------------------------------------------
-- Row-wise sum (axis=1) on a 2D tensor
-- ---------------------------------------------------------------------------

theorem approxT_reduce_sum_axis1_2d
    {m n : Nat} (hm : 0 < m) (hn : 0 < n)
    {xS : SpecTensor (.dim m (.dim n .scalar))}
    {xR : Tensor R (.dim m (.dim n .scalar))}
    {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    let s : Shape := .dim m (.dim n .scalar)
    have hm' : m ≠ 0 := Nat.ne_of_gt hm
    have hn' : n ≠ 0 := Nat.ne_of_gt hn
    have hAxis : Shape.valid_axis 1 s := (Shape.validAxisInstOne (h₁ := hm') (h₂ := hn')).proof
    let hRed : Shape.reducibleAlong 1 s := Shape.proveReducibleAlong 1 s hAxis
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.reduceSum (α := ℝ) (s := s) 1 xS hRed)
      (Spec.Tensor.reduceSum (α := R) (s := s) 1 xR hRed)
      (let boundVec : SpecTensor (.dim m .scalar) :=
        match xR with
        | .dim xRf => Tensor.dim (fun i =>
            Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) eps
              (xRf i)))
      linfNorm boundVec) := by
  intro s hm' hn' hAxis hRed
  classical
  have hε : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) (s := s) hx
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          let boundVec : SpecTensor (.dim m .scalar) :=
            Tensor.dim (fun i =>
              Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
                eps (xRf i)))
          have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)
          -- Use an explicit shape argument so later `rw` matches syntactically (Lean’s `rw`
          -- does not match definitional equal implicit args).
          have hRed' : Shape.reducibleAlong 1 (.dim m (.dim n .scalar)) := by
            simpa [s] using hRed

          refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
            (n := m) (s := .scalar)
            (xS := Spec.Tensor.reduceSum (α := ℝ) (s := .dim m (.dim n .scalar)) 1 (Tensor.dim xSf)
              hRed')
            (xR := Spec.Tensor.reduceSum (α := R) (s := .dim m (.dim n .scalar)) 1 (Tensor.dim xRf)
              hRed')
            (eps := linfNorm boundVec) hBoundNonneg ?_
          intro i
          have hxRow :=
            approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
          have hSum :=
            approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
              (xS := xSf i) (xR := xRf i) (eps := eps) hxRow
          have hSumScalar :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1
              hSum
          have hle : sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) eps (xRf
            i) ≤ linfNorm boundVec := by
            have := linf_norm_le_get_dim (t := boundVec) i
            -- `linf_norm (Tensor.scalar b) = |b|`.
            have habs : abs (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
              eps (xRf i)) ≤ linfNorm boundVec := by
              simpa [boundVec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this
            exact le_trans (le_abs_self _) habs
          have : abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := .dim n .scalar)
                (xRf i)) -
                sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i))
              ≤ linfNorm boundVec := le_trans hSumScalar hle
          have hScalarApprox :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Tensor.scalar (sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i)))
                (Tensor.scalar (sumSpec (α := R) (s := .dim n .scalar) (xRf i)))
                (linfNorm boundVec) :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
              (by
              simpa [abs_sub_comm] using this)
          -- Rewrite the component of `reduce_sum` at axis=1 to the row sum.
          have hEqS := reduce_sum_axis1_get (α := ℝ) (m := m) (n := n) (x := Tensor.dim xSf) hRed' i
          have hEqR := reduce_sum_axis1_get (α := R) (m := m) (n := n) (x := Tensor.dim xRf) hRed' i
          -- `rw` is brittle here (it does not match definitional equal implicit arguments).
          -- `convert` produces equality side-goals for each argument and avoids the brittle
          -- rewrite.
          convert hScalarApprox using 1

-- ---------------------------------------------------------------------------
-- Row-wise mean (axis=1) on a 2D tensor
-- ---------------------------------------------------------------------------

theorem approxT_reduce_mean_axis1_2d
    {m n : Nat} (hm : 0 < m) (hn : 0 < n)
    {xS : SpecTensor (.dim m (.dim n .scalar))}
    {xR : Tensor R (.dim m (.dim n .scalar))}
    {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    let s : Shape := .dim m (.dim n .scalar)
    have hm' : m ≠ 0 := Nat.ne_of_gt hm
    have hn' : n ≠ 0 := Nat.ne_of_gt hn
    have hAxis : Shape.valid_axis 1 s := (Shape.validAxisInstOne (h₁ := hm') (h₂ := hn')).proof
    let hRed : Shape.reducibleAlong 1 s := Shape.proveReducibleAlong 1 s hAxis
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.reduceMean (α := ℝ) (s := s) 1 xS hRed)
      (Spec.Tensor.reduceMean (α := R) (s := s) 1 xR hRed)
      (let boundVec : SpecTensor (.dim m .scalar) :=
        match xR with
        | .dim xRf => Tensor.dim (fun i =>
            let sumR : R := sumSpec (α := R) (s := .dim n .scalar) (xRf i)
            let epsSum := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) eps
              (xRf i)
            Tensor.scalar (
              neuralUlp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR /
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R))
                  TrainingPhase.forward / 2
                + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR) *
                    abs (1 / toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R))
                + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR)
                + epsSum))
      linfNorm boundVec) := by
  intro s hm' hn' hAxis hRed
  classical
  have hε : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) (s := s) hx
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := one_le_nat_coe (n := n) hn
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          let boundVec : SpecTensor (.dim m .scalar) :=
            Tensor.dim (fun i =>
              let sumR : R := sumSpec (α := R) (s := .dim n .scalar) (xRf i)
              let epsSum := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) eps
                (xRf i)
              Tensor.scalar (
                neuralUlp β fexp
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR /
                      toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R))
                    TrainingPhase.forward / 2
                  + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR) *
                      abs (1 / toSpec (β := β) (fexp := fexp) (rnd := rnd) (n : R))
                  + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) sumR)
                  + epsSum))
          have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)
          have hRed' : Shape.reducibleAlong 1 (.dim m (.dim n .scalar)) := by
            simpa [s] using hRed

          refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
            (n := m) (s := .scalar)
            (xS := Spec.Tensor.reduceMean (α := ℝ) (s := .dim m (.dim n .scalar)) 1 (Tensor.dim
              xSf) hRed')
            (xR := Spec.Tensor.reduceMean (α := R) (s := .dim m (.dim n .scalar)) 1 (Tensor.dim
              xRf) hRed')
            (eps := linfNorm boundVec) hBoundNonneg ?_
          intro i
          have hxRow :=
            approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
          have hSum :=
            approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar)
              (xS := xSf i) (xR := xRf i) (eps := eps) hxRow
          have hSumScalar :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1
              hSum
          -- Apply scalar division bound and weaken to the global max.
          have hMeanScalar :=
            approx_div_nf_of_one_le (β := β) (fexp := fexp) (rnd := rnd)
              (x := sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i))
              (y := (n : ℝ))
              (xR := sumSpec (α := R) (s := .dim n .scalar) (xRf i))
              (yR := (n : R))
              (epsx := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) eps (xRf
                i))
              hn1 hSumScalar
          have hle : (match boundVec with | .dim f => f i).toScalar ≤ linfNorm boundVec := by
            have := linf_norm_le_get_dim (t := boundVec) i
            have habs : abs ((match boundVec with | .dim f => f i).toScalar) ≤ linfNorm boundVec :=
              by
              simpa [linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this
            exact le_trans (le_abs_self _) habs
          have hb :
              abs
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                        (sumSpec (α := R) (s := .dim n .scalar) (xRf i) / (n : R)) -
                    (sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i) / (n : ℝ)))
                ≤ (match boundVec with | .dim f => f i).toScalar := by
            simpa [boundVec] using hMeanScalar
          have : abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (sumSpec (α := R) (s := .dim n .scalar) (xRf i) / (n : R)) -
                  (sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i) / (n : ℝ)))
              ≤ linfNorm boundVec := le_trans hb hle
          have hScalarApprox :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Tensor.scalar (sumSpec (α := ℝ) (s := .dim n .scalar) (xSf i) / (n : ℝ)))
                (Tensor.scalar (sumSpec (α := R) (s := .dim n .scalar) (xRf i) / (n : R)))
                (linfNorm boundVec) :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
              (by
              simpa [abs_sub_comm] using this)
          -- Rewrite the component of `reduce_mean` at axis=1.
          have hEqS := reduce_mean_axis1_get (α := ℝ) (m := m) (n := n) (x := Tensor.dim xSf) hRed'
            i
          have hEqR := reduce_mean_axis1_get (α := R) (m := m) (n := n) (x := Tensor.dim xRf) hRed'
            i
          convert hScalarApprox using 1

-- ---------------------------------------------------------------------------
-- Column-wise sum (axis=0) on a 2D tensor
-- ---------------------------------------------------------------------------

/-- Extract column `j` from a runtime `m×n` tensor, represented row-wise as `Fin m → Vec n`. -/
def colR {m n : Nat} (xRf : Fin m → Tensor R (.dim n .scalar)) (j : Fin n) : Tensor R (.dim m
  .scalar) :=
  Tensor.dim (fun i => sliceSpec (xRf i) j)

/-- Extract column `j` from a spec `m×n` tensor, represented row-wise as `Fin m → Vec n`. -/
def colS {m n : Nat} (xSf : Fin m → SpecTensor (.dim n .scalar)) (j : Fin n) : SpecTensor (.dim m
  .scalar) :=
  Tensor.dim (fun i => sliceSpec (xSf i) j)

theorem approxT_reduce_sum_axis0_2d
    {m n : Nat} (hm : 0 < m) (_hn : 0 < n)
    {xS : SpecTensor (.dim m (.dim n .scalar))}
    {xR : Tensor R (.dim m (.dim n .scalar))}
    {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    let s : Shape := .dim m (.dim n .scalar)
    have hAxis : Shape.valid_axis 0 s := (Shape.validAxisInstZeroAlt2 (n := m) (s := .dim n
      .scalar) hm).proof
    let hRed : Shape.reducibleAlong 0 s := Shape.proveReducibleAlong 0 s hAxis
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.reduceSum (α := ℝ) (s := s) 0 xS hRed)
      (Spec.Tensor.reduceSum (α := R) (s := s) 0 xR hRed)
      (let boundVec : SpecTensor (.dim n .scalar) :=
        match xR with
        | .dim xRf => Tensor.dim (fun j =>
            Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar) eps
              (colR (m := m) (n := n) xRf j)))
      linfNorm boundVec) := by
  intro s hAxis hRed
  classical
  have hε : 0 ≤ eps := approxT_eps_nonneg (β := β) (fexp := fexp) (rnd := rnd) (s := s) hx
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          let boundVec : SpecTensor (.dim n .scalar) :=
            Tensor.dim (fun j =>
              Tensor.scalar (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
                eps (colR (m := m) (n := n) xRf j)))
          have hBoundNonneg : 0 ≤ linfNorm boundVec := linf_norm_nonneg (t := boundVec)

          refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
            (n := n) (s := .scalar)
            (xS := Spec.Tensor.reduceSum (α := ℝ) (s := s) 0 (Tensor.dim xSf) hRed)
            (xR := Spec.Tensor.reduceSum (α := R) (s := s) 0 (Tensor.dim xRf) hRed)
            (eps := linfNorm boundVec) hBoundNonneg ?_
          intro j
          -- Approx for column tensor.
          have hcol :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (colS (m := m) (n := n) xSf j) (colR (m := m) (n := n) xRf j) eps := by
            refine approxT_dim_of_forall (β := β) (fexp := fexp) (rnd := rnd)
              (n := m) (s := .scalar) (xS := colS (m := m) (n := n) xSf j) (xR := colR (m := m) (n
                := n) xRf j)
              (eps := eps) hε ?_
            intro i
            have hrow :=
              approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
            cases hs : xSf i with
            | dim colsS =>
                cases hr : xRf i with
                | dim colsR =>
                    have hij :=
                      approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (xS := Tensor.dim colsS) (xR := Tensor.dim colsR) (eps := eps) (by simpa
                          [hs, hr] using hrow) j
                    simpa [colS, colR, hs, hr] using hij

          have hSum :=
            approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
              (xS := colS (m := m) (n := n) xSf j) (xR := colR (m := m) (n := n) xRf j) (eps := eps)
                hcol
          have hSumScalar :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1
              hSum
          have hle : sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar) eps (colR
            (m := m) (n := n) xRf j) ≤
              linfNorm boundVec := by
            have := linf_norm_le_get_dim (t := boundVec) j
            have habs : abs (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := .dim m .scalar)
              eps (colR (m := m) (n := n) xRf j)) ≤
                linfNorm boundVec := by
              simpa [boundVec, linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm] using this
            exact le_trans (le_abs_self _) habs
          have : abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (sumSpec (α := R) (s := .dim m .scalar) (colR (m := m) (n := n) xRf j)) -
                  sumSpec (α := ℝ) (s := .dim m .scalar) (colS (m := m) (n := n) xSf j))
              ≤ linfNorm boundVec := le_trans hSumScalar hle
          have hScalarApprox :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Tensor.scalar (sumSpec (α := ℝ) (s := .dim m .scalar) (colS (m := m) (n := n) xSf
                  j)))
                (Tensor.scalar (sumSpec (α := R) (s := .dim m .scalar) (colR (m := m) (n := n) xRf
                  j)))
                (linfNorm boundVec) :=
            (approxT_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
              (by
              simpa [abs_sub_comm] using this)
          simpa [colS, colR, Spec.Tensor.reduceSum, Spec.Tensor.reduceDim,
            Spec.Tensor.reduceDim.aux,
            Spec.Tensor.reduceFirstDim, Spec.Tensor.shapeAfterSum, getAtSpec, sliceSpec]
              using hScalarApprox

end NFBackend

end
end RuntimeApprox
end Proofs
