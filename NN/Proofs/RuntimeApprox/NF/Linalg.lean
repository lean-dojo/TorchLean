/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Plumbing
public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar
public import NN.Spec.Core.TensorReductionShape.ConcatSlice
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# NF Linear Algebra

Forward (runtime→spec) approximation lemmas for non-elementwise linear algebra ops over `NF`.

This extends `NN.Proofs.RuntimeApprox.NF.Ops` with bounds for the core
sum-of-products patterns that appear in linear layers and matrix multiplication.

The central trick is to separate proof-friendly scalar fold bounds for dot products from
tensor-level wrappers that turn those fold bounds into `approxTensor` theorems and graph nodes.

## PyTorch correspondence / citations
This is the proof analogue of linear algebra building blocks used throughout PyTorch models:
matrix-vector/matrix-matrix multiplication (`torch.matmul`) and linear layers
  (`torch.nn.functional.linear`).
https://pytorch.org/docs/stable/generated/torch.matmul.html
https://pytorch.org/docs/stable/generated/torch.nn.functional.linear.html
-/

@[expose] public section


namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Scalar access helpers
-- ---------------------------------------------------------------------------

/-- Extract matrix entry `(i,j)` from a runtime matrix tensor as an `NF` scalar. -/
def matGet {m n : Nat} (A : Tensor R [m, n]) (i : Fin m) (j : Fin n) : R :=
  Spec.get2 A i j

/-- Extract matrix entry `(i,j)` from a spec matrix tensor as a real scalar. -/
private def matGetS {m n : Nat} (A : SpecTensor [m, n]) (i : Fin m) (j : Fin n)
  : SpecScalar :=
  Spec.get2 A i j

-- ---------------------------------------------------------------------------
-- Exact shape ops preserve approximation (`unsqueeze`, `transpose`)
-- ---------------------------------------------------------------------------

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Inserting a singleton axis is a reindexing operation and adds no numerical error. -/
theorem approxTensor_unsqueeze_spec {shape : Shape} {xS : SpecTensor shape}
    {xR : Tensor R shape} {eps : ℝ} (axis : Nat) (hAxis : axis ≤ shape.rank)
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.unsqueezeSpec xS axis hAxis) (Tensor.unsqueezeSpec xR axis hAxis) eps := by
  induction axis generalizing shape with
  | zero =>
      have heps : 0 ≤ eps := approxTensor_eps_nonneg hx
      cases shape with
      | scalar =>
          rw [Tensor.unsqueezeSpec.eq_1, Tensor.unsqueezeSpec.eq_1]
          exact approxTensor_dim_of_forall heps (fun _ => by
            simpa only [Tensor.unstack_dim] using hx)
      | dim n rest =>
          rw [Tensor.unsqueezeSpec.eq_2, Tensor.unsqueezeSpec.eq_2]
          exact approxTensor_dim_of_forall heps (fun _ => by
            simpa only [Tensor.unstack_dim] using hx)
  | succ axis ih =>
      cases shape with
      | scalar => exact False.elim (Nat.not_succ_le_zero axis hAxis)
      | dim n rest =>
          rw [Tensor.unsqueezeSpec.eq_3, Tensor.unsqueezeSpec.eq_3]
          apply approxTensor_dim_of_forall (approxTensor_eps_nonneg hx)
          intro i
          simpa only [Tensor.unstack_dim] using
            ih (Nat.lt_succ_iff.mp (by
              simpa [Shape.rank, Nat.add_comm] using hAxis))
              (approxTensor_dim_get hx i)

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Swapping any pair of adjacent axes preserves the approximation error budget. -/
theorem approxTensor_swapAdjacentAxes {shape : Shape} {depth : Nat}
    {xS : SpecTensor shape} {xR : Tensor R shape} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.swapAdjacentAxes xS depth) (Tensor.swapAdjacentAxes xR depth) eps := by
  induction depth generalizing shape with
  | zero =>
      cases shape with
      | scalar => simpa [Tensor.swapAdjacentAxes] using hx
      | dim n rest =>
          cases rest with
          | scalar => simpa [Tensor.swapAdjacentAxes] using hx
          | dim m rest =>
              rw [Tensor.swapAdjacentAxes_zero, Tensor.swapAdjacentAxes_zero]
              apply approxTensor_dim_of_forall (approxTensor_eps_nonneg hx)
              intro j
              apply approxTensor_dim_of_forall (approxTensor_eps_nonneg hx)
              intro i
              simpa only [Tensor.unstack_dim, Spec.get] using
                (approxTensor_dim_get (approxTensor_dim_get hx i) j)
  | succ depth ih =>
      cases shape with
      | scalar => simpa [Tensor.swapAdjacentAxes] using hx
      | dim n rest =>
          simp only [Tensor.swapAdjacentAxes]
          apply approxTensor_dim_of_forall (approxTensor_eps_nonneg hx)
          intro i
          simpa only [Tensor.unstack_dim] using ih (approxTensor_dim_get hx i)

-- ---------------------------------------------------------------------------
-- Dot-product (sum of products) bound over a list of indices
-- ---------------------------------------------------------------------------

/--
One fold step for building a dot-product *and* tracking a forward error bound.

This is used to bound the error of `foldl (fun acc k => acc + aR k * bR k)` compared to the
corresponding spec (real) dot-product.
-/
def dotStep {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) :
    (R × ℝ) → Fin n → (R × ℝ)
  | (accR, epsAcc), k =>
      let akR := aR k
      let bkR := bR k
      let prodR : R := akR * bkR
      let epsProd : ℝ :=
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) akR) + epsa) * epsb +
          (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) bkR) + epsb) * epsa +
          ulp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) akR *
                toSpec (β := β) (fexp := fexp) (rnd := rnd) bkR) / 2
      let epsAcc' : ℝ :=
        epsAcc + epsProd +
          ulp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                toSpec (β := β) (fexp := fexp) (rnd := rnd) prodR) / 2
      (accR + prodR, epsAcc')

/--
Closed-form bound for a runtime dot-product over `List.finRange n`.

`dot_bound epsa epsb aR bR` is the accumulated `eps` component produced by folding `dotStep`
starting from 0.
-/
def dotBound {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) : ℝ :=
  let initEps : ℝ := ulp β fexp 0 / 2
  ((List.finRange n).foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
      ((0 : R), initEps)).2

omit [ValidRndToNearest rnd] in
/-- The `i`-th output entry of `Spec.matVecMulSpec` is the dot-product of row `i` with `v`. -/
private theorem vec_get_mat_vec_mul_spec {m n : Nat}
    (A : Tensor R [m, n]) (v : Tensor R [n]) (i : Fin m) :
    TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := R) A v) i =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGet A i k * TorchLean.Tensor.getScalar v k)
        (0 : R) := by
  change
    (TorchLean.Tensor.Internal.Rep.ofFn fun
      coordinate : TorchLean.Tensor.Internal.Coord [m] =>
      (List.finRange n).foldl
        (fun sum k => sum + Spec.get2 A coordinate.1 k * v.getScalar k) 0)
        (i, PUnit.unit) =
      _
  exact TorchLean.Tensor.Internal.Rep.get_ofFn _ _

/-- Spec (real) version of `vec_get_mat_vec_mul_spec`. -/
private theorem vec_getS_mat_vec_mul_spec {m n : Nat}
    (A : SpecTensor [m, n]) (v : SpecTensor [n]) (i : Fin m) :
    TorchLean.Tensor.getScalar (Spec.matVecMulSpec (α := SpecScalar) A v) i =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGetS A i k * TorchLean.Tensor.getScalar v k)
        (0 : SpecScalar) := by
  change
    (TorchLean.Tensor.Internal.Rep.ofFn fun
      coordinate : TorchLean.Tensor.Internal.Coord [m] =>
      (List.finRange n).foldl
        (fun sum k => sum + Spec.get2 A coordinate.1 k * v.getScalar k) 0)
        (i, PUnit.unit) =
      _
  exact TorchLean.Tensor.Internal.Rep.get_ofFn _ _

/--
Dot-product approximation bound over an arbitrary list of indices.

In words: if `aR` and `bR` approximate `aS` and `bS` entrywise (within `epsa`/`epsb`),
  then
folding `acc + aR k * bR k` approximates the corresponding spec fold, with error bounded by the
accumulated `dotStep` epsilon.
-/
private theorem approx_dot_list {n : Nat} (l : List (Fin n))
    {aS bS : Fin n → SpecScalar} {aR bR : Fin n → R}
    {accS : SpecScalar} {accR : R} {epsAcc epsa epsb : ℝ}
    (hAcc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR - accS) ≤ epsAcc)
    (ha : ∀ k, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) - aS k) ≤ epsa)
    (hb : ∀ k, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k) - bS k) ≤ epsb) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (l.foldl (fun acc k => acc + aR k * bR k) accR) -
          l.foldl (fun acc k => acc + aS k * bS k) accS) ≤
      (l.foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR) (accR, epsAcc)).2 :=
        by
  induction l generalizing accS accR epsAcc with
  | nil =>
      simpa using hAcc
  | cons k tl ih =>
      -- unfold one step
      have hProd :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k * bR k) - aS k * bS k) ≤
            ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
              (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
              ulp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) / 2) := by
        exact approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd) (x := aS k) (y := bS k)
          (xR := aR k) (yR := bR k) (epsx := epsa) (epsy := epsb) (ha k) (hb k)

      have hStep :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (accR + aR k * bR k) -
                (accS + aS k * bS k)) ≤
            epsAcc +
              ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
                (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
                ulp β fexp
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                      toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) / 2) +
              ulp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k * bR k)) / 2 := by
        -- apply the scalar add bound with `acc` and `prod`
        have := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
          (x := accS) (y := aS k * bS k) (xR := accR) (yR := aR k * bR k)
          (epsx := epsAcc)
          (epsy :=
            ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
              (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
              ulp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) / 2))
          hAcc hProd
        -- the lemma already has the correct RHS shape
        simpa [add_assoc, add_left_comm, add_comm] using this

      -- apply IH to the tail, starting from the updated accumulator
      have ih' :=
        ih (accS := accS + aS k * bS k) (accR := accR + aR k * bR k)
          (epsAcc :=
            epsAcc +
              ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
                (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
                ulp β fexp
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                      toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) / 2) +
              ulp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k * bR k)) / 2)
          hStep

      -- rewrite folds for `cons`
      simpa [List.foldl, dotStep, add_assoc, add_left_comm, add_comm] using ih'

/--
Dot-product approximation bound specialized to `List.finRange n`.

This packages `approx_dot_list` with the appropriate initial accumulator bound for `0`.
-/
private theorem approx_dot_finRange {n : Nat}
    {aS bS : Fin n → SpecScalar} {aR bR : Fin n → R} {epsa epsb : ℝ}
    (ha : ∀ k, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) - aS k) ≤ epsa)
    (hb : ∀ k, abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k) - bS k) ≤ epsb) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            ((List.finRange n).foldl (fun acc k => acc + aR k * bR k) (0 : R)) -
          (List.finRange n).foldl (fun acc k => acc + aS k * bS k) (0 : SpecScalar)) ≤
      dotBound (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR := by
    -- base approximation for the initial accumulator `0`
  have h0 :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : SpecScalar)) ≤
        ulp β fexp 0 / 2 := by
    rw [toSpec_zero (β := β) (fexp := fexp) (rnd := rnd), sub_zero, abs_zero]
    exact div_nonneg (ulp.nonneg β fexp 0) (by norm_num)
  simpa [dotBound] using
    (approx_dot_list (β := β) (fexp := fexp) (rnd := rnd) (n := n) (l := List.finRange n)
      (aS := aS) (bS := bS) (aR := aR) (bR := bR)
      (accS := (0 : SpecScalar)) (accR := (0 : R))
      (epsAcc := ulp β fexp 0 / 2)
      (epsa := epsa) (epsb := epsb) h0 ha hb)

-- ---------------------------------------------------------------------------
-- Matrix-vector multiply
-- ---------------------------------------------------------------------------

/--
Per-output bound tensor for `matVecMulSpec`.

Entry `i` is a dot-product bound for row `i` of `A` dotted with `v`, using `dotBound`.
-/
def matVecMulBoundTensor {m n : Nat} (epsA epsV : ℝ)
    (A : Tensor R [m, n]) (v : Tensor R [n]) :
    SpecTensor [m] :=
  Tensor.dim (fun i =>
    Tensor.scalar (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
      (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) A i k)
      (fun k => TorchLean.Tensor.getScalar v k)))

/--
Forward approximation bound for matrix-vector multiplication.

In words: if `A` and `v` are each approximated by runtime `AR`/`vR` within `epsA`/`epsV`,
then `mat_vec_mul_spec AS vS` is approximated by `mat_vec_mul_spec AR vR`, with error bounded by
`linf_norm (mat_vec_mul_bound_tensor epsA epsV AR vR)`.
-/
theorem approxTensor_mat_vec_mul_spec {m n : Nat} :
    ∀ {AS : SpecTensor [m, n]} {vS : SpecTensor [n]}
      {AR : Tensor R [m, n]} {vR : Tensor R [n]}
      {epsA epsV : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) AS AR epsA →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) vS vR epsV →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Spec.matVecMulSpec (α := SpecScalar) AS vS)
          (Spec.matVecMulSpec (α := R) AR vR)
          (linfNorm (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n :=
            n) epsA epsV AR vR)) := by
  intro AS vS AR vR epsA epsV hA hv
  let bnd : SpecTensor [m] :=
    matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := m) (n := n) epsA epsV AR vR
  let B : ℝ := linfNorm bnd
  have hB_nonneg : 0 ≤ B := by
    simpa [B] using linf_norm_nonneg (t := bnd)
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro i
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
  have hAik : ∀ k : Fin n,
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k) -
        matGetS AS i k) ≤ epsA := by
    intro k
    change abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) ((AR.unstack i).unstack k).item -
        ((AS.unstack i).unstack k).item) ≤ epsA
    have hEntry := approxTensor_dim_get
      (approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hA i) k
    exact (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hEntry
  have hvk : ∀ k : Fin n,
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (vR.getScalar k) -
        vS.getScalar k) ≤ epsV := by
    intro k
    change abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) (vR.unstack k).item -
        (vS.unstack k).item) ≤ epsV
    have hEntry := approxTensor_dim_get (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hv k
    exact (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hEntry
  have hdot :=
    approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
      (aS := fun k => matGetS AS i k)
      (bS := fun k => vS.getScalar k)
      (aR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
      (bR := fun k => vR.getScalar k)
      (epsa := epsA) (epsb := epsV) hAik hvk
  have hEntryAbs :
      abs (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
        (fun k => vR.getScalar k)) ≤ B := by
    simpa [bnd, matVecMulBoundTensor, dotBound, B, linfNorm,
      RuntimeApprox.linfNorm, tensorLinfNorm, Numerics.MathFunctions.abs, SpecScalar] using
      linf_norm_le_get_dim (t := bnd) i
  have hBound :
      dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
        (fun k => vR.getScalar k) ≤ B :=
    le_trans (le_abs_self _) hEntryAbs
  have hOutput := le_trans hdot hBound
  rw [← vec_get_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd) AR vR i,
    ← vec_getS_mat_vec_mul_spec AS vS i] at hOutput
  change abs
    (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        ((Spec.matVecMulSpec (α := R) AR vR).unstack i).item -
      ((Spec.matVecMulSpec (α := SpecScalar) AS vS).unstack i).item) ≤ B
  exact hOutput

-- ---------------------------------------------------------------------------
-- Matrix-matrix multiply
-- ---------------------------------------------------------------------------

/--
Per-entry bound tensor for `matMulSpec`.

Entry `(i,j)` is a dot-product bound for row `i` of `A` dotted with column `j` of `B`, using
  `dotBound`.
-/
def matMulBoundTensor {m n p : Nat} (epsA epsB : ℝ)
    (A : Tensor R [m, n]) (B : Tensor R [n, p]) :
    SpecTensor [m, p] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) A i k)
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) B k j))))

omit [ValidRndToNearest rnd] in
/-- The matrix entry `(i,j)` of `Spec.matMulSpec` is the dot-product of row `i` of `A` with column
  `j` of `B`. -/
private theorem mat_get_mat_mul_spec {m n p : Nat}
    (A : Tensor R [m, n]) (B : Tensor R [n, p])
    (i : Fin m) (j : Fin p) :
    matGet (β := β) (fexp := fexp) (rnd := rnd) (Spec.matMulSpec (α := R) A B) i j =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGet (β := β) (fexp := fexp) (rnd := rnd) A i k *
              matGet (β := β) (fexp := fexp) (rnd := rnd) B k j)
        (0 : R) := by
  simp only [matGet, Spec.matMulSpec, Spec.get2, Tensor.getScalar, Spec.get,
    Tensor.unstack, Tensor.item, TorchLean.Tensor.Internal.Rep.unstack_apply,
    TorchLean.Tensor.Internal.Rep.get_ofFn]

/-- Spec (real) version of `mat_get_mat_mul_spec`. -/
private theorem mat_getS_mat_mul_spec {m n p : Nat}
    (A : SpecTensor [m, n]) (B : SpecTensor [n, p])
    (i : Fin m) (j : Fin p) :
    matGetS (Spec.matMulSpec (α := SpecScalar) A B) i j =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGetS A i k * matGetS B k j)
        (0 : SpecScalar) := by
  simp only [matGetS, Spec.matMulSpec, Spec.get2, Tensor.getScalar, Spec.get,
    Tensor.unstack, Tensor.item, TorchLean.Tensor.Internal.Rep.unstack_apply,
    TorchLean.Tensor.Internal.Rep.get_ofFn]

/--
Forward approximation bound for matrix-matrix multiplication.

In words: if `A` and `B` are approximated by runtime matrices `AR`/`BR` within
  `epsA`/`epsB`,
then `mat_mul_spec AS BS` is approximated by `mat_mul_spec AR BR`, with error bounded by
`linf_norm (mat_mul_bound_tensor epsA epsB AR BR)`.
-/
theorem approxTensor_mat_mul_spec {m n p : Nat} :
    ∀ {AS : SpecTensor [m, n]} {BS : SpecTensor [n, p]}
      {AR : Tensor R [m, n]} {BR : Tensor R [n, p]}
      {epsA epsB : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) AS AR epsA →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) BS BR epsB →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Spec.matMulSpec (α := SpecScalar) AS BS)
          (Spec.matMulSpec (α := R) AR BR)
          (linfNorm (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p
            := p) epsA epsB AR BR)) := by
  intro AS BS AR BR epsA epsB hA hB
  let bnd : SpecTensor [m, p] :=
    matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := m) (n := n) (p := p) epsA epsB AR BR
  let B : ℝ := linfNorm bnd
  have hB_nonneg : 0 ≤ B := by
    simpa [B] using linf_norm_nonneg (t := bnd)
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro i
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro j
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
  have hAik : ∀ k : Fin n,
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k) -
        matGetS AS i k) ≤ epsA := by
    intro k
    change abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) ((AR.unstack i).unstack k).item -
        ((AS.unstack i).unstack k).item) ≤ epsA
    have hEntry := approxTensor_dim_get
      (approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hA i) k
    exact (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hEntry
  have hBkj : ∀ k : Fin n,
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
          (matGet (β := β) (fexp := fexp) (rnd := rnd) BR k j) -
        matGetS BS k j) ≤ epsB := by
    intro k
    change abs
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) ((BR.unstack k).unstack j).item -
        ((BS.unstack k).unstack j).item) ≤ epsB
    have hEntry := approxTensor_dim_get
      (approxTensor_dim_get (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hB k) j
    exact (approxTensor_scalar_iff (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).1 hEntry
  have hdot :=
    approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
      (aS := fun k => matGetS AS i k)
      (bS := fun k => matGetS BS k j)
      (aR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
      (bR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) BR k j)
      (epsa := epsA) (epsb := epsB) hAik hBkj
  have hEntryNorm :
      linfNorm ((bnd.unstack i).unstack j) ≤ B :=
    le_trans (linf_norm_le_get_dim (t := bnd.unstack i) j)
      (linf_norm_le_get_dim (t := bnd) i)
  have hEntryAbs :
      abs (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) BR k j)) ≤ B := by
    simpa [bnd, matMulBoundTensor, dotBound, B, linfNorm,
      RuntimeApprox.linfNorm, tensorLinfNorm, Numerics.MathFunctions.abs, SpecScalar]
      using hEntryNorm
  have hBound :
      dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) AR i k)
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) BR k j) ≤ B :=
    le_trans (le_abs_self _) hEntryAbs
  have hOutput := le_trans hdot hBound
  rw [← mat_get_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) AR BR i j,
    ← mat_getS_mat_mul_spec AS BS i j] at hOutput
  change abs
    (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (((Spec.matMulSpec (α := R) AR BR).unstack i).unstack j).item -
      (((Spec.matMulSpec (α := SpecScalar) AS BS).unstack i).unstack j).item) ≤ B
  exact hOutput

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for linalg ops
-- ---------------------------------------------------------------------------

/--
`FwdNode` for matrix transpose.

This lifts `approxTensor_swapAdjacentAxes` at depth zero into the `FwdGraph` interface so
transposes can be used inside larger verified graphs.
-/
def matrixTransposeNode {Γ : List Shape} {m n : Nat}
    (x : Idx Γ (.dim m (.dim n .scalar))) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim n (.dim m
      .scalar)) :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := SpecScalar)
          ctx x)
    , forwardRuntime := fun ctx =>
        Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctx x)
    , bound := fun eps _ctx =>
        getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps x
    , sound := ?_ }
  intro xS xR eps hctx
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx x
  change approxTensor _
    (Tensor.swapAdjacentAxes (getIdx (α := SpecScalar) xS x) 0)
    (Tensor.swapAdjacentAxes (getIdx (α := R) xR x) 0) _
  exact approxTensor_swapAdjacentAxes (depth := 0) hx

/--
`FwdNode` for matrix-vector multiplication.

The bound is computed by `matVecMulBoundTensor` and then reduced to a scalar budget via
  `linfNorm`.
-/
def matVecMulNode {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (v : Idx Γ (.dim n .scalar)) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m .scalar) :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Spec.matVecMulSpec (α := SpecScalar)
          (getIdx (α := SpecScalar) ctx A) (getIdx (α := SpecScalar) ctx v)
    , forwardRuntime := fun ctx =>
        Spec.matVecMulSpec (α := R)
          (getIdx (α := R) ctx A) (getIdx (α := R) ctx v)
    , bound := fun eps ctx =>
        linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
            (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
            (getIdxEps (Γ := Γ) (s := (.dim n .scalar)) eps v)
            (getIdx (α := R) ctx A)
            (getIdx (α := R) ctx v))
    , sound := ?_ }
  intro xS xR eps hctx
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A
  have hv := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    v
  simpa using
    (approxTensor_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
      (AS := getIdx (α := SpecScalar) xS A)
      (vS := getIdx (α := SpecScalar) xS v)
      (AR := getIdx (α := R) xR A)
      (vR := getIdx (α := R) xR v)
      (epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
      (epsV := getIdxEps (Γ := Γ) (s := (.dim n .scalar)) eps v)
      hA hv)

/--
`FwdNode` for matrix-matrix multiplication.

The bound is computed by `matMulBoundTensor` and then reduced to a scalar budget via `linfNorm`.
-/
def matMulNode {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m (.dim p
      .scalar)) :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Spec.matMulSpec (α := SpecScalar)
          (getIdx (α := SpecScalar) ctx A) (getIdx (α := SpecScalar) ctx B)
    , forwardRuntime := fun ctx =>
        Spec.matMulSpec (α := R)
          (getIdx (α := R) ctx A) (getIdx (α := R) ctx B)
    , bound := fun eps ctx =>
        linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p := p)
            (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
            (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) eps B)
            (getIdx (α := R) ctx A)
            (getIdx (α := R) ctx B))
    , sound := ?_ }
  intro xS xR eps hctx
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A
  have hB := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    B
  simpa using
    (approxTensor_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p := p)
      (AS := getIdx (α := SpecScalar) xS A)
      (BS := getIdx (α := SpecScalar) xS B)
      (AR := getIdx (α := R) xR A)
      (BR := getIdx (α := R) xR B)
      (epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
      (epsB := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) eps B)
      hA hB)

end NFBackend

end
end RuntimeApprox
end Proofs
