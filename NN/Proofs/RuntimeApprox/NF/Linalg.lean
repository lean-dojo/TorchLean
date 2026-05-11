/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.Utils
public import NN.Spec.Core.Tensor.Linalg

/-!
# NF Linear Algebra

Forward (runtime→spec) approximation lemmas for non-elementwise linear algebra ops over `NF`.

This extends `NN.Proofs.RuntimeApprox.NF.Ops` with bounds for the core
sum-of-products patterns that appear in linear layers and matrix multiplication.

The central trick is to separate proof-friendly scalar fold bounds for dot products from
tensor-level wrappers that turn those fold bounds into `approxT` theorems and graph nodes.

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

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Scalar access helpers (vectors / matrices)
-- ---------------------------------------------------------------------------

/-- Extract the `i`-th entry of a runtime vector tensor as an `NF` scalar. -/
def vecGet {n : Nat} (v : Tensor R (.dim n .scalar)) (i : Fin n) : R :=
  match v with
  | Tensor.dim f => Tensor.toScalar (f i)

/-- Extract the `i`-th entry of a spec vector tensor as a real scalar. -/
private def vecGetS {n : Nat} (v : SpecTensor (.dim n .scalar)) (i : Fin n) : SpecScalar :=
  match v with
  | Tensor.dim f => Tensor.toScalar (f i)

/-- Extract matrix entry `(i,j)` from a runtime matrix tensor as an `NF` scalar. -/
def matGet {m n : Nat} (A : Tensor R (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n) : R :=
  match A with
  | Tensor.dim rows =>
      match rows i with
      | Tensor.dim cols => Tensor.toScalar (cols j)

/-- Extract matrix entry `(i,j)` from a spec matrix tensor as a real scalar. -/
private def matGetS {m n : Nat} (A : SpecTensor (.dim m (.dim n .scalar))) (i : Fin m) (j : Fin n)
  : SpecScalar :=
  match A with
  | Tensor.dim rows =>
      match rows i with
      | Tensor.dim cols => Tensor.toScalar (cols j)

/-- `toScalar` is definitional on scalar tensors. -/
private lemma toScalar_scalar {α : Type} (x : α) :
    (Tensor.scalar x : Tensor α .scalar).toScalar = x := by
  rfl

-- ---------------------------------------------------------------------------
-- Exact shape ops preserve approximation (`expand_to_col`, `transpose`)
-- ---------------------------------------------------------------------------

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
/--
Componentwise lifting for `approxT` on `Tensor.dim`.

In words: if every component `xSf i` approximates `xRf i` within `eps`, then the whole
dimensioned tensor `Tensor.dim xSf` approximates `Tensor.dim xRf` within `eps` (with the same
  `eps`).
-/
private lemma approxT_dim_mk {n : Nat} {s : Shape}
    (xSf : Fin n → SpecTensor s) (xRf : Fin n → Tensor R s) {eps : ℝ}
    (heps : 0 ≤ eps)
    (h : ∀ i : Fin n,
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (xSf i) (xRf i) eps)
        :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.dim xSf) (Tensor.dim xRf) eps := by
  classical
  cases n with
  | zero =>
      -- Empty tensor: distance is `0`.
      simpa [approxT, approxWith, tensorDistance,
        NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
        linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec, Spec.mapTensor] using
          heps
  | succ n =>
      have hf :
          ∀ i ∈ List.finRange (Nat.succ n),
            tensorDistance (α := SpecScalar) linfNorm (xSf i)
                (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (xRf i))
              ≤ eps := by
        intro i _hi
        simpa [approxT, approxWith] using (h i)

      have hfold :=
        List.foldl_max_le_of_le (List.finRange (Nat.succ n))
          (fun i =>
            tensorDistance (α := SpecScalar) linfNorm (xSf i)
              (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (xRf
                i)))
          (acc := (0 : ℝ)) (eps := eps) heps hf

      have :
          tensorDistance (α := SpecScalar) linfNorm (Tensor.dim xSf)
              (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Tensor.dim xRf))
            ≤ eps := by
        simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
          linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec, Spec.mapTensor]
            using hfold
      simpa [approxT, approxWith] using this

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
/--
`expand_to_col_spec` preserves approximation.

This is a purely shape-level view operation (turn a vector into a singleton-column matrix), so it
does not introduce new numeric error beyond the input approximation.
-/
lemma approxT_expand_to_col_spec {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor R (.dim n s)} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.expandToColSpec (α := SpecScalar) (n := n) (s := s) xS)
      (Tensor.expandToColSpec (α := R) (n := n) (s := s) xR)
      eps := by
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          have heps : 0 ≤ eps := by
            have hdist :
                0 ≤ tensorDistance (α := SpecScalar) linfNorm (Tensor.dim xSf)
                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (Tensor.dim xRf)) := by
              simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub] using
                (linf_norm_nonneg
                  (t :=
                    NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub (Tensor.dim xSf)
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd)) (Tensor.dim xRf))))
            exact le_trans hdist hx

          -- Unfold `expand_to_col_spec` so the goal is pointwise on rows.
          simp [Tensor.expandToColSpec]
          refine
            approxT_dim_mk (β := β) (fexp := fexp) (rnd := rnd)
              (n := n) (s := .dim 1 s) _ _ heps ?_
          intro i
          have hrow :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (xSf i) (xRf i) eps :=
            approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (n := n) (s := s) (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
          -- Build the singleton-column tensor by reusing the row approximation.
          refine
            approxT_dim_mk (β := β) (fexp := fexp) (rnd := rnd)
              (n := 1) (s := s) (xSf := fun _j : Fin 1 => xSf i) (xRf := fun _j : Fin 1 => xRf i)
              (eps := eps) heps (by intro _j; simpa using hrow)

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
/--
`matrix_transpose_spec` preserves approximation.

Transpose is a pure reindexing/view operation on tensor entries, so `approxT` is preserved with the
same error budget.
-/
lemma approxT_matrix_transpose_spec {m n : Nat}
    {xS : SpecTensor (.dim m (.dim n .scalar))} {xR : Tensor R (.dim m (.dim n .scalar))} {eps : ℝ}
    (hx : approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) xS)
      (Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) xR)
      eps := by
  cases xS with
  | dim xSf =>
      cases xR with
      | dim xRf =>
          have heps : 0 ≤ eps := by
            have hdist :
                0 ≤ tensorDistance (α := SpecScalar) linfNorm (Tensor.dim xSf)
                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (Tensor.dim xRf)) := by
              simpa [tensorDistance, NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub] using
                (linf_norm_nonneg
                  (t :=
                    NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub (Tensor.dim xSf)
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd)) (Tensor.dim xRf))))
            exact le_trans hdist hx

          -- Unfold transpose so the goal is pointwise on rows/entries.
          simp [Tensor.matrixTransposeSpec]
          refine
            approxT_dim_mk (β := β) (fexp := fexp) (rnd := rnd)
              (n := n) (s := .dim m .scalar) _ _ heps ?_
          intro j
          refine
            approxT_dim_mk (β := β) (fexp := fexp) (rnd := rnd)
              (n := m) (s := Shape.scalar) _ _ heps ?_
          intro i

          have hrow :
              approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (xSf i) (xRf i) eps :=
            approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (n := m) (s := .dim n .scalar) (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps :=
                eps) hx i

          cases hxSi : xSf i with
          | dim xSfRow =>
              cases hxRi : xRf i with
              | dim xRfRow =>
                  have hrow' :
                      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                        (Tensor.dim xSfRow) (Tensor.dim xRfRow) eps := by
                    simpa [hxSi, hxRi] using hrow
                  have hij :=
                    approxT_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                      (n := n) (s := Shape.scalar) (xS := Tensor.dim xSfRow) (xR := Tensor.dim
                        xRfRow)
                      (eps := eps) hrow' j
                  cases hxSj : xSfRow j with
                  | scalar vSj =>
                      cases hxRj : xRfRow j with
                      | scalar vRj =>
                          simpa [hxSi, hxRi, hxSj, hxRj] using hij

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
          neuralUlp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) akR *
                toSpec (β := β) (fexp := fexp) (rnd := rnd) bkR)
              TrainingPhase.forward / 2
      let epsAcc' : ℝ :=
        epsAcc + epsProd +
          neuralUlp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                toSpec (β := β) (fexp := fexp) (rnd := rnd) prodR)
              TrainingPhase.forward / 2
      (accR + prodR, epsAcc')

/--
Closed-form bound for a runtime dot-product over `List.finRange n`.

`dot_bound epsa epsb aR bR` is the accumulated `eps` component produced by folding `dot_step`
starting from 0.
-/
def dotBound {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) : ℝ :=
  let initEps : ℝ := neuralUlp β fexp 0 TrainingPhase.forward / 2
  ((List.finRange n).foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
      ((0 : R), initEps)).2

/--
Public-facing alias of `dot_bound`.

This keeps exported tensor-bound constructors from depending directly on the local helper name that
Lean treats as non-exportable in this file.
-/
def dotBoundExport {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) : ℝ :=
  let initEps : ℝ := neuralUlp β fexp 0 TrainingPhase.forward / 2
  ((List.finRange n).foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
      ((0 : R), initEps)).2

/--
Helper lemma: rewriting a fold that pattern-matches scalar tensors into a fold on raw scalars.

This is used to relate `Spec.mat_vec_mul_spec` (defined by folding tensor scalars) to a more
familiar scalar `∑`/`foldl` expression.
-/
private lemma foldl_tensorScalar_mulAdd {α : Type} [Add α] [Mul α] {n : Nat}
    (cols : Fin n → Tensor α .scalar) (vals : Fin n → Tensor α .scalar) (l : List (Fin n)) (acc : α)
      :
    List.foldl
        (fun acc k =>
          match acc, cols k, vals k with
          | Tensor.scalar s, Tensor.scalar ak, Tensor.scalar vk => Tensor.scalar (s + ak * vk))
        (Tensor.scalar acc) l =
      Tensor.scalar
        (List.foldl
          (fun acc k =>
            acc +
              (match cols k with | Tensor.scalar x => x) *
                (match vals k with | Tensor.scalar x => x))
          acc l) := by
  induction l generalizing acc with
  | nil =>
      simp
  | cons k tl ih =>
      cases hcols : cols k with
      | scalar ak =>
          cases hvals : vals k with
          | scalar vk =>
              simpa [List.foldl, hcols, hvals] using ih (acc := acc + ak * vk)

omit [NeuralValidRndToNearest rnd] in
/-- The `i`-th output entry of `Spec.mat_vec_mul_spec` is the dot-product of row `i` with `v`. -/
private lemma vec_get_mat_vec_mul_spec {m n : Nat}
    (A : Tensor R (.dim m (.dim n .scalar))) (v : Tensor R (.dim n .scalar)) (i : Fin m) :
    vecGet (Spec.matVecMulSpec (α := R) A v) i =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGet A i k * vecGet v k)
        (0 : R) := by
  cases A with
  | dim rowsA =>
      cases v with
      | dim valsV =>
          cases hRow : rowsA i with
          | dim colsA =>
              simp [Spec.matVecMulSpec, vecGet, matGet, hRow]
              have h :=
                foldl_tensorScalar_mulAdd (cols := colsA) (vals := valsV) (l := List.finRange n)
                  (acc := (0 : R))
              have hto := congrArg Tensor.toScalar h
              exact hto

/-- Spec (real) version of `vec_get_mat_vec_mul_spec`. -/
private lemma vec_getS_mat_vec_mul_spec {m n : Nat}
    (A : SpecTensor (.dim m (.dim n .scalar))) (v : SpecTensor (.dim n .scalar)) (i : Fin m) :
    vecGetS (Spec.matVecMulSpec (α := SpecScalar) A v) i =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGetS A i k * vecGetS v k)
        (0 : SpecScalar) := by
  cases A with
  | dim rowsA =>
      cases v with
      | dim valsV =>
          cases hRow : rowsA i with
          | dim colsA =>
              simp [Spec.matVecMulSpec, vecGetS, matGetS, hRow]
              have h :=
                foldl_tensorScalar_mulAdd (cols := colsA) (vals := valsV) (l := List.finRange n)
                  (acc := (0 : SpecScalar))
              have hto := congrArg Tensor.toScalar h
              exact hto

/--
Dot-product approximation bound over an arbitrary list of indices.

In words: if `aR` and `bR` approximate `aS` and `bS` entrywise (within `epsa`/`epsb`),
  then
folding `acc + aR k * bR k` approximates the corresponding spec fold, with error bounded by the
accumulated `dot_step` epsilon.
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
              neuralUlp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k))
                  TrainingPhase.forward / 2) := by
        exact approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd) (x := aS k) (y := bS k)
          (xR := aR k) (yR := bR k) (epsx := epsa) (epsy := epsb) (ha k) (hb k)

      have hStep :
          abs
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) (accR + aR k * bR k) -
                (accS + aS k * bS k)) ≤
            epsAcc +
              ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
                (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
                neuralUlp β fexp
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                      toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k))
                    TrainingPhase.forward / 2) +
              neuralUlp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k * bR k))
                  TrainingPhase.forward / 2 := by
        -- apply the scalar add bound with `acc` and `prod`
        have := approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
          (x := accS) (y := aS k * bS k) (xR := accR) (yR := aR k * bR k)
          (epsx := epsAcc)
          (epsy :=
            ((abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k)) + epsa) * epsb +
              (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k)) + epsb) * epsa +
              neuralUlp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k))
                  TrainingPhase.forward / 2))
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
                neuralUlp β fexp
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k) *
                      toSpec (β := β) (fexp := fexp) (rnd := rnd) (bR k))
                    TrainingPhase.forward / 2) +
              neuralUlp β fexp
                  (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                    toSpec (β := β) (fexp := fexp) (rnd := rnd) (aR k * bR k))
                  TrainingPhase.forward / 2)
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
        neuralUlp β fexp 0 TrainingPhase.forward / 2 := by
    -- `toSpec 0 = roundR 0`, then apply the rounding abs-error bound.
    simpa [toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR,
      Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) (0 : ℝ))
  simpa [dotBound] using
    (approx_dot_list (β := β) (fexp := fexp) (rnd := rnd) (n := n) (l := List.finRange n)
      (aS := aS) (bS := bS) (aR := aR) (bR := bR)
      (accS := (0 : SpecScalar)) (accR := (0 : R))
      (epsAcc := neuralUlp β fexp 0 TrainingPhase.forward / 2)
      (epsa := epsa) (epsb := epsb) h0 ha hb)

-- ---------------------------------------------------------------------------
-- Matrix-vector multiply
-- ---------------------------------------------------------------------------

/--
Per-output bound tensor for `mat_vec_mul_spec`.

Entry `i` is a dot-product bound for row `i` of `A` dotted with `v`, using `dot_bound`.
-/
def matVecMulBoundTensor {m n : Nat} (epsA epsV : ℝ)
    (A : Tensor R (.dim m (.dim n .scalar))) (v : Tensor R (.dim n .scalar)) :
    SpecTensor (.dim m .scalar) :=
  Tensor.dim (fun i =>
    Tensor.scalar (dotBoundExport (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
      (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) A i k)
      (fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) v k)))

/--
Forward approximation bound for matrix-vector multiplication.

In words: if `A` and `v` are each approximated by runtime `AR`/`vR` within `epsA`/`epsV`,
then `mat_vec_mul_spec AS vS` is approximated by `mat_vec_mul_spec AR vR`, with error bounded by
`linf_norm (mat_vec_mul_bound_tensor epsA epsV AR vR)`.
-/
theorem approxT_mat_vec_mul_spec {m n : Nat} :
    ∀ {AS : SpecTensor (.dim m (.dim n .scalar))} {vS : SpecTensor (.dim n .scalar)}
      {AR : Tensor R (.dim m (.dim n .scalar))} {vR : Tensor R (.dim n .scalar)}
      {epsA epsV : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) AS AR epsA →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) vS vR epsV →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Spec.matVecMulSpec (α := SpecScalar) AS vS)
          (Spec.matVecMulSpec (α := R) AR vR)
          (linfNorm (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n :=
            n) epsA epsV AR vR)) := by
  intro AS vS AR vR epsA epsV hA hv
  -- unfold the concrete shapes
  cases AS with
  | dim ASf =>
      cases AR with
      | dim ARf =>
          cases vS with
          | dim vSf =>
              cases vR with
              | dim vRf =>
                  let bnd : SpecTensor (.dim m .scalar) :=
                    matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (m := m) (n := n) epsA epsV (Tensor.dim ARf) (Tensor.dim vRf)
                  let B : ℝ := linfNorm bnd
                  have hB_nonneg : 0 ≤ B := by
                    simpa [B] using (linf_norm_nonneg (t := bnd))

                  -- componentwise error bound
                  have hcomp : ∀ i : Fin m,
                      abs
                          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                              (vecGet (β := β) (fexp := fexp) (rnd := rnd)
                                (Spec.matVecMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim vRf))
                                  i) -
                            vecGetS
                              (Spec.matVecMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                                vSf)) i)
                        ≤ B := by
                    intro i
                    -- rewrite outputs to scalar folds
                    have hyR :
                        vecGet (β := β) (fexp := fexp) (rnd := rnd)
                            (Spec.matVecMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim vRf)) i =
                          (List.finRange n).foldl
                            (fun acc k =>
                              acc +
                                matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf) i k *
                                  vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim vRf) k)
                            (0 : R) :=
                      vec_get_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
                        (A := Tensor.dim ARf) (v := Tensor.dim vRf) i
                    have hyS :
                        vecGetS
                            (Spec.matVecMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                              vSf)) i =
                          (List.finRange n).foldl
                            (fun acc k =>
                              acc +
                                matGetS (Tensor.dim ASf) i k *
                                  vecGetS (Tensor.dim vSf) k)
                            (0 : SpecScalar) :=
                      vec_getS_mat_vec_mul_spec (A := Tensor.dim ASf) (v := Tensor.dim vSf) i

                    -- per-entry approximations from `approxT`
                    have hRowA :
                        ∀ k : Fin n,
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf) i k)
                                  -
                              matGetS (Tensor.dim ASf) i k) ≤ epsA := by
                      intro k
                      have hRow :=
                        approxT_dim_get (α := R)
                          (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hA i
                      cases hASi : ASf i with
                      | dim ASrow =>
                          cases hARi : ARf i with
                          | dim ARrow =>
                              have hRow' :
                                  approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                    rnd))
                                    (Tensor.dim ASrow) (Tensor.dim ARrow) epsA := by
                                simpa [hASi, hARi] using hRow
                              have hkT :=
                                approxT_dim_get (α := R)
                                  (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hRow' k
                              cases hASk : ASrow k with
                              | scalar aS =>
                                  cases hARk : ARrow k with
                                  | scalar aR =>
                                      have habs :
                                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) aR - aS)
                                            ≤ epsA := by
                                        exact
                                          (approxT_scalar_iff (α := R)
                                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                            (x := aS) (xR := aR) (eps := epsA)).1
                                            (by simpa [hASk, hARk] using hkT)
                                      simpa [matGet, matGetS, hASi, hARi, hASk, hARk] using habs

                    have hVecV :
                        ∀ k : Fin n,
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim vRf) k) -
                              vecGetS (Tensor.dim vSf) k) ≤ epsV := by
                      intro k
                      have hkT :=
                        approxT_dim_get (α := R)
                          (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hv k
                      cases hVS : vSf k with
                      | scalar vS =>
                          cases hVR : vRf k with
                          | scalar vR =>
                              have habs :
                                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) vR - vS) ≤ epsV
                                    := by
                                exact
                                  (approxT_scalar_iff (α := R)
                                    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                    (x := vS) (xR := vR) (eps := epsV)).1
                                    (by simpa [hVS, hVR] using hkT)
                              simpa [vecGet, vecGetS, hVS, hVR] using habs

                    have hdot :
                        abs
                            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                  ((List.finRange n).foldl
                                    (fun acc k =>
                                      acc +
                                        matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                          ARf) i k *
                                          vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                            vRf) k)
                                    (0 : R)) -
                                  (List.finRange n).foldl
                                    (fun acc k =>
                                      acc +
                                        matGetS (Tensor.dim ASf) i k *
                                          vecGetS (Tensor.dim vSf) k)
                                    (0 : SpecScalar)) ≤
                          dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf)
                              i k)
                            (fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim vRf)
                              k) :=
                      approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
                        (aS := fun k => matGetS (Tensor.dim ASf) i k)
                        (bS := fun k => vecGetS (Tensor.dim vSf) k)
                        (aR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                          ARf) i k)
                        (bR := fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                          vRf) k)
                        (epsa := epsA) (epsb := epsV) hRowA hVecV

                    have hEntryAbs :
                        abs
                            (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
                              (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                ARf) i k)
                              (fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                vRf) k)) ≤
                          B := by
                      have :=
                        linf_norm_le_get_dim (t := bnd) i
                      simpa [bnd, matVecMulBoundTensor, B, linfNorm, RuntimeApprox.linfNorm,
                        tensorLinfNorm] using
                        this

                    have hB_ge :
                        dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf)
                              i k)
                            (fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim vRf)
                              k) ≤
                          B :=
                      le_trans (le_abs_self _) hEntryAbs

                    -- rewrite using the output fold equalities
                    have hle : abs
                        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                            (vecGet (β := β) (fexp := fexp) (rnd := rnd)
                              (Spec.matVecMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim vRf)) i)
                                -
                          vecGetS
                            (Spec.matVecMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                              vSf)) i) ≤
                        dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
                          (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf) i
                            k)
                          (fun k => vecGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim vRf) k)
                            := by
                      simpa [hyR, hyS] using hdot

                    exact le_trans hle hB_ge

                  -- lift to the tensor-level Linf approximation
                  cases hOutS : (Spec.matVecMulSpec (α := SpecScalar) (Tensor.dim ASf)
                    (Tensor.dim vSf)) with
                  | dim ySf =>
                      cases hOutR : (Spec.matVecMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim
                        vRf)) with
                      | dim yRf =>
                          have hf :
                              ∀ i ∈ List.finRange m,
                                tensorDistance (α := SpecScalar) linfNorm (ySf i)
                                    (tensorToSpec (α := R)
                                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                      (yRf i))
                                  ≤ B := by
                            intro i _hi
                            have hscalar := hcomp i
                            cases hYS : ySf i with
                            | scalar yS =>
                                cases hYR : yRf i with
                                | scalar yR =>
                                    have hAbs :
                                        abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - yS) ≤
                                          B := by
                                      simpa [hOutS, hOutR, vecGet, vecGetS, hYS, hYR] using
                                        hscalar
                                    have hApprox :
                                        approxT (α := R)
                                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                            (Tensor.scalar yS) (Tensor.scalar yR) B :=
                                      (approxT_scalar_iff (α := R)
                                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                        (x := yS) (xR := yR) (eps := B)).2 hAbs
                                    simpa [approxT, approxWith, hYS, hYR, tensorToSpec,
                                      Spec.mapTensor] using hApprox

                          have hfold :=
                            List.foldl_max_le_of_le (List.finRange m)
                              (fun i =>
                                tensorDistance (α := SpecScalar) linfNorm (ySf i)
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (yRf i)))
                              (acc := (0 : ℝ)) (eps := B) hB_nonneg hf

                          have :
                              tensorDistance (α := SpecScalar) linfNorm (Tensor.dim ySf)
                                  (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp)
                                    (rnd := rnd))
                                    (Tensor.dim yRf))
                                ≤ B := by
                            simpa [tensorDistance,
                              NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                              linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                                Spec.mapTensor] using hfold

                          simpa [approxT, approxWith, B, bnd] using this

-- ---------------------------------------------------------------------------
-- Matrix-matrix multiply
-- ---------------------------------------------------------------------------

/--
Per-entry bound tensor for `mat_mul_spec`.

Entry `(i,j)` is a dot-product bound for row `i` of `A` dotted with column `j` of `B`, using
  `dot_bound`.
-/
def matMulBoundTensor {m n p : Nat} (epsA epsB : ℝ)
    (A : Tensor R (.dim m (.dim n .scalar))) (B : Tensor R (.dim n (.dim p .scalar))) :
    SpecTensor (.dim m (.dim p .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (dotBoundExport (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) A i k)
        (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) B k j))))

omit [NeuralValidRndToNearest rnd] in
/-- The matrix entry `(i,j)` of `Spec.mat_mul_spec` is the dot-product of row `i` of `A` with column
  `j` of `B`. -/
private lemma mat_get_mat_mul_spec {m n p : Nat}
    (A : Tensor R (.dim m (.dim n .scalar))) (B : Tensor R (.dim n (.dim p .scalar)))
    (i : Fin m) (j : Fin p) :
    matGet (β := β) (fexp := fexp) (rnd := rnd) (Spec.matMulSpec (α := R) A B) i j =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGet (β := β) (fexp := fexp) (rnd := rnd) A i k *
              matGet (β := β) (fexp := fexp) (rnd := rnd) B k j)
        (0 : R) := by
  cases A with
  | dim rowsA =>
      cases B with
      | dim rowsB =>
          cases hRow : rowsA i with
          | dim colsA =>
              -- Unfold spec matmul and matrix indexing; then rewrite the fold using scalar
              -- extraction.
              simp [Spec.matMulSpec, matGet, hRow, toScalar_scalar]
              refine
                foldl_congr (l := List.finRange n) (f := _) (g := _) (init := (0 : R)) ?_
              intro sum k
              cases hRowB : rowsB k with
              | dim colsB =>
                  cases hA : colsA k with
                  | scalar a =>
                      cases hBj : colsB j with
                      | scalar b =>
                          simp [hA, hBj, toScalar_scalar]

/-- Spec (real) version of `mat_get_mat_mul_spec`. -/
private lemma mat_getS_mat_mul_spec {m n p : Nat}
    (A : SpecTensor (.dim m (.dim n .scalar))) (B : SpecTensor (.dim n (.dim p .scalar)))
    (i : Fin m) (j : Fin p) :
    matGetS (Spec.matMulSpec (α := SpecScalar) A B) i j =
      (List.finRange n).foldl
        (fun acc k =>
          acc +
            matGetS A i k * matGetS B k j)
        (0 : SpecScalar) := by
  cases A with
  | dim rowsA =>
      cases B with
      | dim rowsB =>
          cases hRow : rowsA i with
          | dim colsA =>
              simp [Spec.matMulSpec, matGetS, hRow, toScalar_scalar]
              refine
                foldl_congr (l := List.finRange n) (f := _) (g := _) (init := (0 : SpecScalar)) ?_
              intro sum k
              cases hRowB : rowsB k with
              | dim colsB =>
                  cases hA : colsA k with
                  | scalar a =>
                      cases hBj : colsB j with
                      | scalar b =>
                          simp [hA, hBj, toScalar_scalar]

/--
Forward approximation bound for matrix-matrix multiplication.

In words: if `A` and `B` are approximated by runtime matrices `AR`/`BR` within
  `epsA`/`epsB`,
then `mat_mul_spec AS BS` is approximated by `mat_mul_spec AR BR`, with error bounded by
`linf_norm (mat_mul_bound_tensor epsA epsB AR BR)`.
-/
theorem approxT_mat_mul_spec {m n p : Nat} :
    ∀ {AS : SpecTensor (.dim m (.dim n .scalar))} {BS : SpecTensor (.dim n (.dim p .scalar))}
      {AR : Tensor R (.dim m (.dim n .scalar))} {BR : Tensor R (.dim n (.dim p .scalar))}
      {epsA epsB : ℝ},
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) AS AR epsA →
      approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) BS BR epsB →
        approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Spec.matMulSpec (α := SpecScalar) AS BS)
          (Spec.matMulSpec (α := R) AR BR)
          (linfNorm (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p
            := p) epsA epsB AR BR)) := by
  intro AS BS AR BR epsA epsB hA hB
  cases AS with
  | dim ASf =>
      cases AR with
      | dim ARf =>
          cases BS with
          | dim BSf =>
              cases BR with
              | dim BRf =>
                  let bnd : SpecTensor (.dim m (.dim p .scalar)) :=
                    matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (m := m) (n := n) (p := p) epsA epsB (Tensor.dim ARf) (Tensor.dim BRf)
                  let B : ℝ := linfNorm bnd
                  have hB_nonneg : 0 ≤ B := by
                    simpa [B] using (linf_norm_nonneg (t := bnd))

                  have hEntry :
                      ∀ i : Fin m, ∀ j : Fin p,
                        abs
                            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (matGet (β := β) (fexp := fexp) (rnd := rnd)
                                  (Spec.matMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim BRf)) i
                                    j) -
                              matGetS
                                (Spec.matMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                                  BSf)) i j)
                          ≤ B := by
                    intro i j
                    -- rewrite both entries to dot-fold forms
                    have hyR :=
                      mat_get_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
                        (A := Tensor.dim ARf) (B := Tensor.dim BRf) i j
                    have hyS :=
                      mat_getS_mat_mul_spec (A := Tensor.dim ASf) (B := Tensor.dim BSf) i j

                    -- per-entry approximations
                    have hAik :
                        ∀ k : Fin n,
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf) i k)
                                  -
                              matGetS (Tensor.dim ASf) i k) ≤ epsA := by
                      intro k
                      have hRow :=
                        approxT_dim_get (α := R)
                          (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hA i
                      cases hASi : ASf i with
                      | dim ASrow =>
                          cases hARi : ARf i with
                          | dim ARrow =>
                              have hRow' :
                                  approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                    rnd))
                                    (Tensor.dim ASrow) (Tensor.dim ARrow) epsA := by
                                simpa [hASi, hARi] using hRow
                              have hkT :=
                                approxT_dim_get (α := R)
                                  (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hRow' k
                              cases hASk : ASrow k with
                              | scalar aS =>
                                  cases hARk : ARrow k with
                                  | scalar aR =>
                                      have habs :
                                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) aR - aS)
                                            ≤ epsA := by
                                        exact
                                          (approxT_scalar_iff (α := R)
                                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                            (x := aS) (xR := aR) (eps := epsA)).1
                                            (by simpa [hASk, hARk] using hkT)
                                      simpa [matGet, matGetS, hASi, hARi, hASk, hARk] using habs

                    have hBkj :
                        ∀ k : Fin n,
                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim BRf) k j)
                                  -
                              matGetS (Tensor.dim BSf) k j) ≤ epsB := by
                      intro k
                      have hRow :=
                        approxT_dim_get (α := R)
                          (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hB k
                      cases hBSk : BSf k with
                      | dim BSrow =>
                          cases hBRk : BRf k with
                          | dim BRrow =>
                              have hRow' :
                                  approxT (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                    rnd))
                                    (Tensor.dim BSrow) (Tensor.dim BRrow) epsB := by
                                simpa [hBSk, hBRk] using hRow
                              have hjT :=
                                approxT_dim_get (α := R)
                                  (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hRow' j
                              cases hBSj : BSrow j with
                              | scalar bS =>
                                  cases hBRj : BRrow j with
                                  | scalar bR =>
                                      have habs :
                                          abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) bR - bS)
                                            ≤ epsB := by
                                        exact
                                          (approxT_scalar_iff (α := R)
                                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                            (x := bS) (xR := bR) (eps := epsB)).1
                                            (by simpa [hBSj, hBRj] using hjT)
                                      simpa [matGet, matGetS, hBSk, hBRk, hBSj, hBRj] using habs

                    have hdot :=
                      approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
                        (aS := fun k => matGetS (Tensor.dim ASf) i k)
                        (bS := fun k => matGetS (Tensor.dim BSf) k j)
                        (aR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                          ARf) i k)
                        (bR := fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                          BRf) k j)
                        (epsa := epsA) (epsb := epsB) hAik hBkj

                    have hAbsBound :
                        abs
                            (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
                              (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                ARf) i k)
                              (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim
                                BRf) k j)) ≤
                          B := by
                      -- two-level projection: row then column
                      have hRow :=
                        linf_norm_le_get_dim (t := bnd) i
                      have hCol :=
                        linf_norm_le_get_dim
                          (t := match bnd with | Tensor.dim f => f i) j
                      have : linfNorm
                          (match match bnd with
                            | Tensor.dim f => f i with
                            | Tensor.dim g => g j) ≤ B := by
                        exact le_trans hCol (by simpa [B] using hRow)
                      -- scalar entry simplifies to `abs (dot_bound ..)`
                      simpa [bnd, matMulBoundTensor, B, linfNorm, RuntimeApprox.linfNorm,
                        tensorLinfNorm] using this

                    have hBound :
                        dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf)
                              i k)
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim BRf)
                              k j) ≤
                          B :=
                      le_trans (le_abs_self _) hAbsBound

                    have hle :
                        abs
                            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                (matGet (β := β) (fexp := fexp) (rnd := rnd)
                                  (Spec.matMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim BRf)) i
                                    j) -
                              matGetS
                                (Spec.matMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                                  BSf)) i j) ≤
                          dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim ARf)
                              i k)
                            (fun k => matGet (β := β) (fexp := fexp) (rnd := rnd) (Tensor.dim BRf)
                              k j) := by
                      simpa [hyR, hyS] using hdot

                    exact le_trans hle hBound

                  -- lift to the tensor-level Linf approximation (matrix: max over rows, then cols)
                  cases hOutS : (Spec.matMulSpec (α := SpecScalar) (Tensor.dim ASf) (Tensor.dim
                    BSf)) with
                  | dim ySf =>
                      cases hOutR : (Spec.matMulSpec (α := R) (Tensor.dim ARf) (Tensor.dim BRf))
                        with
                      | dim yRf =>
                          have hRowDist :
                              ∀ i ∈ List.finRange m,
                                tensorDistance (α := SpecScalar) linfNorm (ySf i)
                                    (tensorToSpec (α := R)
                                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                      (yRf i))
                                  ≤ B := by
                            intro i _hi
                            cases hYSrow : ySf i with
                            | dim ySrow =>
                                cases hYRrow : yRf i with
                                | dim yRrow =>
                                    have hf :
                                        ∀ j ∈ List.finRange p,
                                          tensorDistance (α := SpecScalar) linfNorm (ySrow j)
                                              (tensorToSpec (α := R)
                                                (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                                  rnd))
                                                (yRrow j))
                                            ≤ B := by
                                      intro j _hj
                                      have hscalar := hEntry i j
                                      cases hYS : ySrow j with
                                      | scalar yS =>
                                          cases hYR : yRrow j with
                                          | scalar yR =>
                                              have hAbs :
                                                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                                                    yR - yS) ≤ B := by
                                                simpa [hOutS, hOutR, matGet, matGetS, hYSrow,
                                                  hYRrow, hYS, hYR] using hscalar
                                              have hApprox :
                                                  approxT (α := R)
                                                      (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                                                        := rnd))
                                                      (Tensor.scalar yS) (Tensor.scalar yR) B :=
                                                (approxT_scalar_iff (α := R)
                                                  (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                                    rnd))
                                                  (x := yS) (xR := yR) (eps := B)).2 hAbs
                                              simpa [approxT, approxWith, hYS, hYR, tensorToSpec,
                                                Spec.mapTensor] using hApprox

                                    have hfold :=
                                      List.foldl_max_le_of_le (List.finRange p)
                                        (fun j =>
                                          tensorDistance (α := SpecScalar) linfNorm (ySrow j)
                                            (tensorToSpec (α := R)
                                              (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                                rnd))
                                              (yRrow j)))
                                        (acc := (0 : ℝ)) (eps := B) hB_nonneg hf

                                    have :
                                        tensorDistance (α := SpecScalar) linfNorm (Tensor.dim
                                          ySrow)
                                            (tensorToSpec (α := R)
                                              (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                                                rnd))
                                              (Tensor.dim yRrow))
                                          ≤ B := by
                                      simpa [tensorDistance,
                                        NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                                        linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm,
                                          tensorToSpec, Spec.mapTensor] using hfold

                                    simpa [hYSrow, hYRrow] using this

                          have hfold :=
                            List.foldl_max_le_of_le (List.finRange m)
                              (fun i =>
                                tensorDistance (α := SpecScalar) linfNorm (ySf i)
                                  (tensorToSpec (α := R)
                                    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                    (yRf i)))
                              (acc := (0 : ℝ)) (eps := B) hB_nonneg hRowDist

                          have :
                              tensorDistance (α := SpecScalar) linfNorm (Tensor.dim ySf)
                                  (tensorToSpec (α := R)
                                    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                    (Tensor.dim yRf))
                                ≤ B := by
                            simpa [tensorDistance,
                              NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                              linfNorm, RuntimeApprox.linfNorm, tensorLinfNorm, tensorToSpec,
                                Spec.mapTensor] using hfold

                          simpa [approxT, approxWith, B, bnd] using this

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for linalg ops
-- ---------------------------------------------------------------------------

/--
`FwdNode` for matrix transpose.

This lifts `approxT_matrix_transpose_spec` into the `FwdGraph` interface so transposes can be used
inside larger verified graphs.
-/
def matrixTransposeNode {Γ : List Shape} {m n : Nat}
    (x : Idx Γ (.dim m (.dim n .scalar))) :
    FwdNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim n (.dim m
      .scalar)) :=
by
  classical
  refine
    { forwardSpec := fun ctx =>
        Tensor.matrixTransposeSpec (α := SpecScalar) (m := m) (n := n) (getIdx (α := SpecScalar)
          ctx x)
    , forwardRuntime := fun ctx =>
        Tensor.matrixTransposeSpec (α := R) (m := m) (n := n) (getIdx (α := R) ctx x)
    , bound := fun eps _ctx =>
        getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps x
    , sound := ?_ }
  intro xS xR eps hctx
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx x
  simpa using
    (approxT_matrix_transpose_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
      (xS := getIdx (α := SpecScalar) xS x)
      (xR := getIdx (α := R) xR x)
      (eps := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps x)
      hx)

/--
`FwdNode` for matrix-vector multiplication.

The bound is computed by `mat_vec_mul_bound_tensor` and then reduced to a scalar budget via
  `linf_norm`.
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
    (approxT_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
      (AS := getIdx (α := SpecScalar) xS A)
      (vS := getIdx (α := SpecScalar) xS v)
      (AR := getIdx (α := R) xR A)
      (vR := getIdx (α := R) xR v)
      (epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
      (epsV := getIdxEps (Γ := Γ) (s := (.dim n .scalar)) eps v)
      hA hv)

/--
`FwdNode` for matrix-matrix multiplication.

The bound is computed by `mat_mul_bound_tensor` and then reduced to a scalar budget via `linf_norm`.
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
    (approxT_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p := p)
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
