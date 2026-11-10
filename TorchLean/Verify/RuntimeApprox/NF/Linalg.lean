/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Scalar
public import TorchLean.Tensor.ConcatSlice
public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Contract

/-!
# NF Linear Algebra

Forward-error bounds in the noncomputable rounded-real NF model for the sum-of-products patterns
of linear layers and matrix multiplication. They extend the scalar and elementwise bounds of the
modules under `TorchLean/Verify/RuntimeApprox/`.

Scalar fold bounds for dot products come first, and tensor-level wrappers turn them into
`ApproxTensor` theorems and graph nodes. Each dot product rounds every multiplication and addition
separately, in increasing index order, so a native GEMM or fused multiply-add kernel that reorders
or fuses these steps needs its own bound.
-/

@[expose] public section


namespace TorchLean

open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TensorPack (getIdx)

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Exact shape ops preserve approximation (`transpose`)
-- ---------------------------------------------------------------------------

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Permuting the axes preserves the approximation error budget: each output entry copies an
input entry. -/
theorem ApproxTensor.permute {shape out : Shape} {σ : Equiv.Perm (Fin shape.rank)}
    {hout : out = shape.permute σ} {xS : Tensor ℝ shape} {xR : Tensor R shape} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.permute xS σ out hout) (Tensor.permute xR σ out hout) eps :=
  approxTensor_of_forall (eps_nonneg hx) fun index => by
    simpa only [Tensor.permute_apply] using abs_sub_le hx _

-- ---------------------------------------------------------------------------
-- Dot-product (sum of products) bound over a list of indices
-- ---------------------------------------------------------------------------

/--
One fold step that builds a rounded dot product and tracks its forward error bound.

`approx_dot_list` uses it to bound `foldl (fun acc k => acc + aR k * bR k)` against the
corresponding spec (real) dot product.
-/
def NF.dotStep {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) :
    (R × ℝ) → Fin n → (R × ℝ)
  | (accR, epsAcc), k =>
      let akR := aR k
      let bkR := bR k
      let prodR : R := akR * bkR
      let epsProd : ℝ :=
        (abs (NF.toReal akR) + epsa) * epsb +
          (abs (NF.toReal bkR) + epsb) * epsa +
          ulp β fexp
              (NF.toReal akR *
                NF.toReal bkR) / 2
      let epsAcc' : ℝ :=
        epsAcc + epsProd +
          ulp β fexp
              (NF.toReal accR +
                NF.toReal prodR) / 2
      (accR + prodR, epsAcc')

/--
Accumulated error budget for a sequential dot product over `List.finRange n`.
The initial zero contributes a conservative half ULP, followed by separate multiplication
and addition budgets at each index.
-/
def NF.dotBound {n : Nat} (epsa epsb : ℝ) (aR bR : Fin n → R) : ℝ :=
  let initEps : ℝ := ulp β fexp 0 / 2
  ((List.finRange n).foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
      ((0 : R), initEps)).2

/--
Dot-product approximation bound over an arbitrary list of indices.

The list determines the traversal order and may repeat indices. Each multiplication and
addition contributes its own error budget through `dotStep`.
-/
private theorem NF.approx_dot_list {n : Nat} (l : List (Fin n))
    {aS bS : Fin n → ℝ} {aR bR : Fin n → R}
    {accS : ℝ} {accR : R} {epsAcc epsa epsb : ℝ}
    (hAcc : abs (NF.toReal accR - accS) ≤ epsAcc)
    (ha : ∀ k, abs (NF.toReal (aR k) - aS k) ≤ epsa)
    (hb : ∀ k, abs (NF.toReal (bR k) - bS k) ≤ epsb) :
    abs
        (NF.toReal
            (l.foldl (fun acc k => acc + aR k * bR k) accR) -
          l.foldl (fun acc k => acc + aS k * bS k) accS) ≤
      (l.foldl (dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
        (accR, epsAcc)).2 := by
  have values := List.foldl_hom (Prod.fst : R × ℝ → R)
    (g₁ := dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
    (g₂ := fun acc k => acc + aR k * bR k) (l := l) (init := (accR, epsAcc))
    (fun _ _ => rfl)
  rw [values]
  apply List.foldl_rel (l := l)
    (f := dotStep (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR)
    (g := fun acc k => acc + aS k * bS k)
    (r := fun (state : R × ℝ) reference => |NF.toReal state.1 - reference| ≤ state.2) hAcc
  rintro k _ ⟨value, error⟩ reference hstate
  simpa only [dotStep] using
    (approx_add (β := β) (fexp := fexp) (rnd := rnd) hstate
      (approx_mul (β := β) (fexp := fexp) (rnd := rnd) (ha k) (hb k)))

/--
Dot-product approximation bound specialized to `List.finRange n`.

The fold starts from `0` with budget `ulp β fexp 0 / 2`, the initial accumulator bound that
`approx_dot_list` needs.
-/
private theorem NF.approx_dot_finRange {n : Nat}
    {aS bS : Fin n → ℝ} {aR bR : Fin n → R} {epsa epsb : ℝ}
    (ha : ∀ k, abs (NF.toReal (aR k) - aS k) ≤ epsa)
    (hb : ∀ k, abs (NF.toReal (bR k) - bS k) ≤ epsb) :
    abs
        (NF.toReal
            ((List.finRange n).foldl (fun acc k => acc + aR k * bR k) (0 : R)) -
          (List.finRange n).foldl (fun acc k => acc + aS k * bS k) (0 : ℝ)) ≤
      dotBound (β := β) (fexp := fexp) (rnd := rnd) epsa epsb aR bR := by
  have h0 :
      abs (NF.toReal (0 : R) - (0 : ℝ)) ≤
        ulp β fexp 0 / 2 := by
    rw [toReal_zero (β := β) (fexp := fexp) (rnd := rnd), sub_zero, abs_zero]
    exact div_nonneg (ulp.nonneg β fexp 0) (by norm_num)
  simpa [dotBound] using
    (approx_dot_list (β := β) (fexp := fexp) (rnd := rnd) (n := n) (l := List.finRange n)
      (aS := aS) (bS := bS) (aR := aR) (bR := bR)
      (accS := (0 : ℝ)) (accR := (0 : R))
      (epsAcc := ulp β fexp 0 / 2)
      (epsa := epsa) (epsb := epsb) h0 ha hb)

-- ---------------------------------------------------------------------------
-- Matrix-vector multiply
-- ---------------------------------------------------------------------------

/--
Per-output bound tensor for the matrix-vector contraction.

Entry `i` is a dot-product bound for row `i` of `A` dotted with `v`, using `dotBound`.
-/
def NF.matVecMulBoundTensor {m n : Nat} (epsA epsV : ℝ)
    (A : Tensor R [m, n]) (v : Tensor R [n]) :
    Tensor ℝ [m] :=
  Tensor.dim (fun i =>
    Tensor.scalar (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
      (fun k => Tensor.get2 A i k)
      (fun k => Tensor.getScalar v k)))

/--
Forward approximation bound for matrix-vector multiplication.

If `AR` and `vR` approximate `AS` and `vS` within `epsA` and `epsV`, then their contraction
approximates the exact one within `linfNorm (matVecMulBoundTensor epsA epsV AR vR)`.
-/
theorem ApproxTensor.contract_matVec {m n : Nat} :
    ∀ {AS : Tensor ℝ [m, n]} {vS : Tensor ℝ [n]}
      {AR : Tensor R [m, n]} {vR : Tensor R [n]}
      {epsA epsV : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) AS AR epsA →
      ApproxTensor (α := R) (toSpec := NF.toReal) vS vR epsV →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.contract AS vS _ _ [m] .matVec)
          (Tensor.contract AR vR _ _ [m] .matVec)
          (linfNorm (NF.matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := n) epsA epsV AR vR)) := by
  intro AS vS AR vR epsA epsV hA hv
  let bnd : Tensor ℝ [m] :=
    NF.matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := m) (n := n) epsA epsV AR vR
  let B : ℝ := linfNorm bnd
  have hB_nonneg : 0 ≤ B := by
    simpa [B] using linfNorm_nonneg (t := bnd)
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro i
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := NF.toReal)).2
  have hAik : ∀ k : Fin n,
      abs (NF.toReal (Tensor.get2 AR i k) -
        Tensor.get2 AS i k) ≤ epsA := by
    intro k
    simpa only [Tensor.get2_eq_apply] using abs_sub_le hA (i, k, PUnit.unit)
  have hvk : ∀ k : Fin n,
      abs (NF.toReal (vR.getScalar k) -
        vS.getScalar k) ≤ epsV := by
    intro k
    simpa only [Tensor.getScalar_eq_apply] using abs_sub_le hv (k, PUnit.unit)
  have hdot :=
    NF.approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
      (aS := fun k => Tensor.get2 AS i k)
      (bS := fun k => vS.getScalar k)
      (aR := fun k => Tensor.get2 AR i k)
      (bR := fun k => vR.getScalar k)
      (epsa := epsA) (epsb := epsV) hAik hvk
  have hEntryAbs :
      abs (NF.dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsV
        (fun k => Tensor.get2 AR i k)
        (fun k => vR.getScalar k)) ≤ B := by
    simpa [bnd, NF.matVecMulBoundTensor, NF.dotBound, B,
      linfNorm, Numerics.MathFunctions.abs] using
      linfNorm_unstack_le (t := bnd) i
  have hOutput := hdot.trans ((le_abs_self _).trans hEntryAbs)
  rw [← Tensor.getScalar_contract_matVec AR vR i, ← Tensor.getScalar_contract_matVec AS vS i]
    at hOutput
  change abs
    (NF.toReal
        ((Tensor.contract AR vR _ _ [m] .matVec).unstack i).item -
      ((Tensor.contract AS vS _ _ [m] .matVec).unstack i).item) ≤ B
  exact hOutput

-- ---------------------------------------------------------------------------
-- Matrix-matrix multiply
-- ---------------------------------------------------------------------------

/--
Per-entry bound tensor for `Tensor.matmul`.

Entry `(i,j)` is a dot-product bound for row `i` of `A` dotted with column `j` of `B`, using
`dotBound`.
-/
def NF.matMulBoundTensor {m n p : Nat} (epsA epsB : ℝ)
    (A : Tensor R [m, n]) (B : Tensor R [n, p]) :
    Tensor ℝ [m, p] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => Tensor.get2 A i k)
        (fun k => Tensor.get2 B k j))))

/--
Forward approximation bound for matrix-matrix multiplication.

If `AR` and `BR` approximate `AS` and `BS` within `epsA` and `epsB`, then `Tensor.matmul AR BR`
approximates `Tensor.matmul AS BS` within `linfNorm (matMulBoundTensor epsA epsB AR BR)`.
-/
theorem ApproxTensor.matmul {m n p : Nat} :
    ∀ {AS : Tensor ℝ [m, n]} {BS : Tensor ℝ [n, p]}
      {AR : Tensor R [m, n]} {BR : Tensor R [n, p]}
      {epsA epsB : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) AS AR epsA →
      ApproxTensor (α := R) (toSpec := NF.toReal) BS BR epsB →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.matmul (α := ℝ) AS BS)
          (Tensor.matmul (α := R) AR BR)
          (linfNorm (NF.matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := n) (p := p) epsA epsB AR BR)) := by
  intro AS BS AR BR epsA epsB hA hB
  let bnd : Tensor ℝ [m, p] :=
    NF.matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (m := m) (n := n) (p := p) epsA epsB AR BR
  let B : ℝ := linfNorm bnd
  have hB_nonneg : 0 ≤ B := by
    simpa [B] using linfNorm_nonneg (t := bnd)
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro i
  refine approxTensor_dim_of_forall hB_nonneg ?_
  intro j
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := NF.toReal)).2
  have hAik : ∀ k : Fin n,
      abs (NF.toReal (Tensor.get2 AR i k) -
        Tensor.get2 AS i k) ≤ epsA := by
    intro k
    simpa only [Tensor.get2_eq_apply] using abs_sub_le hA (i, k, PUnit.unit)
  have hBkj : ∀ k : Fin n,
      abs (NF.toReal (Tensor.get2 BR k j) -
        Tensor.get2 BS k j) ≤ epsB := by
    intro k
    simpa only [Tensor.get2_eq_apply] using abs_sub_le hB (k, j, PUnit.unit)
  have hdot :=
    NF.approx_dot_finRange (β := β) (fexp := fexp) (rnd := rnd) (n := n)
      (aS := fun k => Tensor.get2 AS i k)
      (bS := fun k => Tensor.get2 BS k j)
      (aR := fun k => Tensor.get2 AR i k)
      (bR := fun k => Tensor.get2 BR k j)
      (epsa := epsA) (epsb := epsB) hAik hBkj
  have hEntryNorm :
      linfNorm ((bnd.unstack i).unstack j) ≤ B :=
    le_trans (linfNorm_unstack_le (t := bnd.unstack i) j)
      (linfNorm_unstack_le (t := bnd) i)
  have hEntryAbs :
      abs (NF.dotBound (β := β) (fexp := fexp) (rnd := rnd) (n := n) epsA epsB
        (fun k => Tensor.get2 AR i k)
        (fun k => Tensor.get2 BR k j)) ≤ B := by
    simpa [bnd, NF.matMulBoundTensor, NF.dotBound, B,
      linfNorm, Numerics.MathFunctions.abs]
      using hEntryNorm
  have hOutput := hdot.trans ((le_abs_self _).trans hEntryAbs)
  rw [← Tensor.get2_matmul AR BR i j, ← Tensor.get2_matmul AS BS i j] at hOutput
  change abs
    (NF.toReal
        (((Tensor.matmul (α := R) AR BR).unstack i).unstack j).item -
      (((Tensor.matmul (α := ℝ) AS BS).unstack i).unstack j).item) ≤ B
  exact hOutput

-- ---------------------------------------------------------------------------
-- `FwdNode` constructors for linalg ops
-- ---------------------------------------------------------------------------

/--
`FwdNode` for matrix-vector multiplication.

The bound is computed by `matVecMulBoundTensor` and then reduced to a scalar budget via
`linfNorm`.
-/
def NF.matVecMulNode {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (v : Idx Γ (.dim n .scalar)) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ
      (.dim m .scalar) := by
  refine
    { forwardReal := fun ctx =>
        Tensor.contract (getIdx (α := ℝ) ctx A) (getIdx (α := ℝ) ctx
            v) _ _ [m] .matVec
    , forwardRuntime := fun ctx =>
        Tensor.contract (getIdx (α := R) ctx A) (getIdx (α := R) ctx v) _ _ [m] .matVec
    , bound := fun eps ctx =>
        linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
            (EList.getIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
            (EList.getIdx (Γ := Γ) (s := (.dim n .scalar)) eps v)
            (getIdx (α := R) ctx A)
            (getIdx (α := R) ctx v))
    , sound := ?_ }
  intro xS xR eps hctx
  have hA := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    A
  have hv := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    v
  simpa using
    (ApproxTensor.contract_matVec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n)
      (AS := getIdx (α := ℝ) xS A)
      (vS := getIdx (α := ℝ) xS v)
      (AR := getIdx (α := R) xR A)
      (vR := getIdx (α := R) xR v)
      (epsA := EList.getIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
      (epsV := EList.getIdx (Γ := Γ) (s := (.dim n .scalar)) eps v)
      hA hv)

/--
`FwdNode` for matrix-matrix multiplication.

The bound is computed by `matMulBoundTensor` and then reduced to a scalar budget via
`linfNorm`.
-/
def NF.matMulNode {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    FwdNode (α := R) (toSpec := NF.toReal) Γ
      (.dim m (.dim p .scalar)) := by
  refine
    { forwardReal := fun ctx =>
        Tensor.matmul (α := ℝ)
          (getIdx (α := ℝ) ctx A) (getIdx (α := ℝ) ctx B)
    , forwardRuntime := fun ctx =>
        Tensor.matmul (α := R)
          (getIdx (α := R) ctx A) (getIdx (α := R) ctx B)
    , bound := fun eps ctx =>
        linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p := p)
            (EList.getIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
            (EList.getIdx (Γ := Γ) (s := (.dim n (.dim p .scalar))) eps B)
            (getIdx (α := R) ctx A)
            (getIdx (α := R) ctx B))
    , sound := ?_ }
  intro xS xR eps hctx
  have hA := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    A
  have hB := ApproxCtx.getIdx (α := R) (toSpec := NF.toReal) hctx
    B
  simpa using
    (ApproxTensor.matmul (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := n) (p := p)
      (AS := getIdx (α := ℝ) xS A)
      (BS := getIdx (α := ℝ) xS B)
      (AR := getIdx (α := R) xR A)
      (BR := getIdx (α := R) xR B)
      (epsA := EList.getIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) eps A)
      (epsB := EList.getIdx (Γ := Γ) (s := (.dim n (.dim p .scalar))) eps B)
      hA hB)

end

end
end TorchLean
