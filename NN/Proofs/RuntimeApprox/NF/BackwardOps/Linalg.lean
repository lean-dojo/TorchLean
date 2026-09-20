/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Backend
public import NN.Proofs.RuntimeApprox.Graph.BackwardApprox
public import NN.Proofs.RuntimeApprox.NF.Linalg

/-!
# Linear Algebra NF Reverse Nodes

Reverse-mode approximation nodes for matrix-vector and matrix-matrix multiplication.
-/
@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec
open Proofs.Autograd.Algebra

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

namespace Impl

/-- Internal spelling of axis-one insertion used repeatedly in outer-product formulas. -/
def unsqueezeAfterLeading {α : Type} [TorchLean.Storage α] {n : Nat} {s : Shape}
    (tensor : Tensor α (.dim n s)) : Tensor α (.dim n (.dim 1 s)) :=
  Tensor.dim fun i => Tensor.dim fun _ => tensor.unstack i

end Impl

open Impl

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- The internal axis-one spelling is an exact shape operation. -/
theorem approxTensor_unsqueezeAfterLeading {n : Nat} {s : Shape}
    {xS : SpecTensor (.dim n s)} {xR : Tensor R (.dim n s)} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (unsqueezeAfterLeading xS) (unsqueezeAfterLeading xR) eps := by
  have heps : 0 ≤ eps := approxTensor_eps_nonneg hx
  apply approxTensor_dim_of_forall heps
  intro i
  apply approxTensor_dim_of_forall heps
  intro _
  simpa [unsqueezeAfterLeading] using (approxTensor_dim_get hx i)

-- ---------------------------------------------------------------------------
-- Linear algebra reverse nodes (`mat_vec_mul_spec`, `mat_mul_spec`)
-- ---------------------------------------------------------------------------

/--
Reverse node for matrix-vector multiplication (`matVecMulSpec`).

VJP uses the standard adjoint identities: `δW = δ ⊗ x` and `δx = Wᵀ δ` (expressed in tensor form),
with NF error bounds layered over the primitive ops.
-/
def matVecMulRevNode {Γ : List Shape} {m n : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (v : Idx Γ (.dim n .scalar)) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m .scalar) :=
by
  classical
  refine
    { toFwdNode := matVecMulNode (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (m := m) (n := n) A v
      vjpSpec := fun ctx δ =>
        let vS := getIdx (α := SpecScalar) ctx v
        let δcol := unsqueezeAfterLeading (α := SpecScalar) (n := m) (s := Shape.scalar) δ
        let vcol := unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) vS
        let dA :=
          Spec.matMulSpec (α := SpecScalar)
            δcol (TorchLean.Tensor.swapAdjacentAxes (depth := 0) vcol)
        let dV :=
          Spec.matVecMulSpec (α := SpecScalar)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
              SpecScalar) ctx A))
            δ
        SparseContext.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := (.dim m (.dim n .scalar)))
          (s₂ := (.dim n .scalar)) A dA v dV
      vjpRuntime := fun ctx δ =>
        let vR := getIdx (α := R) ctx v
        let δcol := unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δ
        let vcol := unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) vR
        let dA :=
          Spec.matMulSpec (α := R)
            δcol (TorchLean.Tensor.swapAdjacentAxes (depth := 0) vcol)
        let dV :=
          Spec.matVecMulSpec (α := R)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctx A))
            δ
        SparseContext.set2Idx (α := R) (Γ := Γ) (s₁ := (.dim m (.dim n .scalar)))
          (s₂ := (.dim n .scalar)) A dA v dV
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let vR := getIdx (α := R) ctxR v
        let δcolR := unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR
        let vcolR := unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) vR
        let vrowR := TorchLean.Tensor.swapAdjacentAxes (depth := 0) vcolR
        let epsV := getIdxEps (Γ := Γ) (s := (.dim n .scalar)) epsCtx v
        let epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A
        let dABound :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := 1) (p := n)
              epsδ epsV δcolR vrowR)
        let AT_R := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
          ctxR A)
        let dVBound :=
          linfNorm
            (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m)
              epsA epsδ AT_R δR)
        EList.set2Idx (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n .scalar)) A dABound
          v dVBound 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  -- Approximate `v` and `A` from the context.
  have hv := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    v
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A

  -- `dA = mat_mul (expand δ) (transpose (expand v))`.
  have hδcol :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (unsqueezeAfterLeading (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
        (unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR)
        epsδ :=
    approxTensor_unsqueezeAfterLeading hδ

  have hvcol :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α :=
          SpecScalar) ctxS v))
        (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R) ctxR
          v))
        (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) :=
    approxTensor_unsqueezeAfterLeading hv

  have hvrow :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
          (unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α
            := SpecScalar) ctxS v)))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
          (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
            ctxR v)))
        (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) :=
    approxTensor_swapAdjacentAxes (depth := 0) (β := β) (fexp := fexp) (rnd := rnd) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) hvcol

  have hdA :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar)
          (unsqueezeAfterLeading (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
            (unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx
              (α := SpecScalar) ctxS v))))
        (Spec.matMulSpec (α := R)
          (unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
            (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
              ctxR v))))
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := 1) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v)
            (unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
              (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
                ctxR v))))) := by
    simpa using
      (approxTensor_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := 1) (p := n)
        (AS := unsqueezeAfterLeading (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
        (BS := TorchLean.Tensor.swapAdjacentAxes (depth := 0)
          (unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx (α
            := SpecScalar) ctxS v)))
        (AR := unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR)
        (BR := TorchLean.Tensor.swapAdjacentAxes (depth := 0)
          (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
            ctxR v)))
        (epsA := epsδ) (epsB := getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v) hδcol hvrow)

  -- `dV = mat_vec_mul (transpose A) δ`.
  have hAT :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS A))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
        (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) :=
    approxTensor_swapAdjacentAxes (depth := 0) (β := β) (fexp := fexp) (rnd := rnd) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) hA

  have hdV :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matVecMulSpec (α := SpecScalar)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
            SpecScalar) ctxS A)) δS)
        (Spec.matVecMulSpec (α := R)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
            δR)
        (linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m)
            (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) epsδ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
              δR)) := by
    simpa using
      (approxTensor_mat_vec_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := m)
        (AS := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS A))
        (vS := δS)
        (AR := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
          A))
        (vR := δR)
        (epsA := getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) (epsV := epsδ)
        hAT hδ)

  have hne : A.i ≠ v.i := by
    intro hEq
    -- Shapes would have to coincide.
    have hshapeEq :
        Shape.dim m (Shape.dim n Shape.scalar) = Shape.dim n Shape.scalar := by
      have : Γ.get A.i = Γ.get v.i := by simp [hEq]
      calc
        Shape.dim m (Shape.dim n Shape.scalar) = Γ.get A.i := by simpa using A.h.symm
        _ = Γ.get v.i := this
        _ = Shape.dim n Shape.scalar := by simpa using v.h
    -- Contradiction by constructor discrimination.
    cases hshapeEq

  have hctx' :=
    approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n .scalar)) A v
      (t₁S :=
        Spec.matMulSpec (α := SpecScalar)
          (unsqueezeAfterLeading (α := SpecScalar) (n := m) (s := Shape.scalar) δS)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
            (unsqueezeAfterLeading (α := SpecScalar) (n := n) (s := Shape.scalar) (getIdx
              (α := SpecScalar) ctxS v))))
      (t₁R :=
        Spec.matMulSpec (α := R)
          (unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0)
            (unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) (getIdx (α := R)
              ctxR v))))
      (eps₁ :=
        let vR := getIdx (α := R) ctxR v
        let δcolR := unsqueezeAfterLeading (α := R) (n := m) (s := Shape.scalar) δR
        let vcolR := unsqueezeAfterLeading (α := R) (n := n) (s := Shape.scalar) vR
        let vrowR := TorchLean.Tensor.swapAdjacentAxes (depth := 0) vcolR
        linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := 1) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := .dim n .scalar) epsCtx v)
            δcolR vrowR))
      (t₂S :=
        Spec.matVecMulSpec (α := SpecScalar)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
            SpecScalar) ctxS A))
          δS)
      (t₂R :=
        Spec.matVecMulSpec (α := R)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
          δR)
      (eps₂ :=
        linfNorm
          (matVecMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m)
            (getIdxEps (Γ := Γ) (s := .dim m (.dim n .scalar)) epsCtx A) epsδ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
              δR))
      hdA hdV hne
  simpa using hctx'

/--
Reverse node for matrix multiplication (`matMulSpec`).

VJP uses the standard identities `δA = δC * Bᵀ` and `δB = Aᵀ * δC` (in appropriate shapes),
with NF error bounds layered over the primitive ops.
-/
def matMulRevNode {Γ : List Shape} {m n p : Nat}
    (A : Idx Γ (.dim m (.dim n .scalar))) (B : Idx Γ (.dim n (.dim p .scalar))) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ (.dim m (.dim p
      .scalar)) :=
by
  classical
  refine
    { toFwdNode := matMulNode (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ) (m := m) (n := n) (p :=
      p) A B
      vjpSpec := fun ctx δ =>
        if h : A.i = B.i then
          -- both contributions land in the same slot
          let A0 := getIdx (α := SpecScalar) ctx A
          let δA := Spec.matMulSpec (α := SpecScalar) δ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := SpecScalar) ctx B))
          let δB := Spec.matMulSpec (α := SpecScalar)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) A0) δ
          let δB' := tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) h δB
          SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := (.dim m (.dim n .scalar))) A
            (addSpec δA δB')
        else
          let δA := Spec.matMulSpec (α := SpecScalar) δ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := SpecScalar) ctx B))
          let δB := Spec.matMulSpec (α := SpecScalar)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := SpecScalar) ctx A)) δ
          SparseContext.set2Idx (α := SpecScalar) (Γ := Γ)
            (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A δA B δB
      vjpRuntime := fun ctx δ =>
        if h : A.i = B.i then
          let A0 := getIdx (α := R) ctx A
          let δA := Spec.matMulSpec (α := R) δ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctx B))
          let δB := Spec.matMulSpec (α := R) (TorchLean.Tensor.swapAdjacentAxes (depth := 0) A0) δ
          let δB' := tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) h δB
          SparseContext.setIdx (α := R) (Γ := Γ) (s := (.dim m (.dim n .scalar))) A (addSpec δA δB')
        else
          let δA := Spec.matMulSpec (α := R) δ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctx B))
          let δB := Spec.matMulSpec (α := R)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctx A)) δ
          SparseContext.set2Idx (α := R) (Γ := Γ)
            (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A δA B δB
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A
        let epsB := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B
        let BT_R := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
          ctxR B)
        let AT_R := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
          ctxR A)
        let δA_R := Spec.matMulSpec (α := R) δR BT_R
        let δB_R := Spec.matMulSpec (α := R) AT_R δR
        let epsδA :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ epsB δR BT_R)
        let epsδB :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              epsA epsδ AT_R δR)
        if h : A.i = B.i then
          let δB_R' := tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) h δB_R
          let epsSum :=
            linfNorm
              (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := (.dim m (.dim n .scalar))) epsδA epsδB δA_R δB_R')
          EList.setIdx (Γ := Γ) (s := (.dim m (.dim n .scalar))) A epsSum
        else
          EList.set2Idx (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar)))
            A epsδA B epsδB 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  classical
  have hA := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    A
  have hB := approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx
    B

  have hBT :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS B))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR B))
        (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) :=
    approxTensor_swapAdjacentAxes (depth := 0) (β := β) (fexp := fexp) (rnd := rnd) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) hB

  have hAT :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS A))
        (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
        (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) :=
    approxTensor_swapAdjacentAxes (depth := 0) (β := β) (fexp := fexp) (rnd := rnd) (xS := _)
      (xR := _) (eps := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) hA

  have hdA :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar) δS
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
            SpecScalar) ctxS B)))
        (Spec.matMulSpec (α := R) δR
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR B)))
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := m) (n := p) (p := n)
            epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
              B)))) := by
    simpa using
      (approxTensor_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := m) (n := p) (p := n)
        (AS := δS)
        (BS := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS B))
        (AR := δR)
        (BR := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
          B))
        (epsA := epsδ) (epsB := getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) hδ
          hBT)

  have hdB :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.matMulSpec (α := SpecScalar)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
            SpecScalar) ctxS A)) δS)
        (Spec.matMulSpec (α := R)
          (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
            δR)
        (linfNorm
          (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (m := n) (n := m) (p := p)
            (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
              δR)) := by
    simpa using
      (approxTensor_mat_mul_spec (β := β) (fexp := fexp) (rnd := rnd) (m := n) (n := m) (p := p)
        (AS := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
          SpecScalar) ctxS A))
        (BS := δS)
        (AR := TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
          A))
        (BR := δR)
        (epsA := getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) (epsB := epsδ) hAT
          hδ)

  by_cases hEq : A.i = B.i
  · -- contributions add in one slot
    have hdB' :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := SpecScalar)
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                SpecScalar) ctxS A)) δS))
          (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := R)
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR)) := by
      simpa [tensorCastOfIdxEq, idx_shape_eq_of_i_eq] using
        (approxTensor_tensor_cast (β := β) (fexp := fexp) (rnd := rnd)
          (h := (idx_shape_eq_of_i_eq (Γ := Γ) (a := A) (b := B) hEq).symm)
          (xS :=
            Spec.matMulSpec (α := SpecScalar)
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                SpecScalar) ctxS A)) δS)
          (xR :=
            Spec.matMulSpec (α := R)
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR)
          (eps :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := n) (n := m) (p := p)
                (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  A)) δR))
          hdB)

    have hsum :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec
            (Spec.matMulSpec (α := SpecScalar) δS
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                SpecScalar) ctxS B)))
            (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                  SpecScalar) ctxS A)) δS)))
          (addSpec
            (Spec.matMulSpec (α := R) δR
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                B)))
            (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  A)) δR)))
          (linfNorm
            (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := (.dim m (.dim n .scalar)))
              (linfNorm
                (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (m := m) (n := p) (p := n)
                  epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
                  (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
                    ctxR B))))
              (linfNorm
                (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (m := n) (n := m) (p := p)
                  (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                  (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
                    ctxR A)) δR))
              (Spec.matMulSpec (α := R) δR
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  B)))
              (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
                (Spec.matMulSpec (α := R)
                  (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R)
                    ctxR A)) δR)))) := by
      simpa using
        (approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := (.dim m (.dim n .scalar)))
          (xS :=
            Spec.matMulSpec (α := SpecScalar) δS
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                SpecScalar) ctxS B)))
          (yS :=
            tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                  SpecScalar) ctxS A)) δS))
          (xR :=
            Spec.matMulSpec (α := R) δR
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                B)))
          (yR :=
            tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  A)) δR))
          (epsx :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := m) (n := p) (p := n)
                epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  B))))
          (epsy :=
            linfNorm
              (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (m := n) (n := m) (p := p)
                (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  A)) δR))
          hdA hdB')

    let epsSum : ℝ :=
      linfNorm
        (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := (.dim m (.dim n .scalar)))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                B))))
          (linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR))
          (Spec.matMulSpec (α := R) δR
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR B)))
          (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
            (Spec.matMulSpec (α := R)
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR)))
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := (.dim m (.dim n .scalar))) A
        (tS :=
          addSpec
            (Spec.matMulSpec (α := SpecScalar) δS
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                SpecScalar) ctxS B)))
            (tensorCastOfIdxEq (α := SpecScalar) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := SpecScalar)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
                  SpecScalar) ctxS A)) δS)))
        (tR :=
          addSpec
            (Spec.matMulSpec (α := R) δR
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                B)))
            (tensorCastOfIdxEq (α := R) (Γ := Γ) (a := A) (b := B) hEq
              (Spec.matMulSpec (α := R)
                (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                  A)) δR)))
        (eps := epsSum) (by simpa [epsSum] using hsum)
    simpa [hEq, epsSum, tensorCastOfIdxEq, idx_shape_eq_of_i_eq] using hctx'
  · -- disjoint indices: use `set2Idx`
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := (.dim m (.dim n .scalar))) (s₂ := (.dim n (.dim p .scalar))) A B
        (t₁S :=
          Spec.matMulSpec (α := SpecScalar) δS
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
              SpecScalar) ctxS B)))
        (t₁R :=
          Spec.matMulSpec (α := R) δR
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR B)))
        (eps₁ :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := m) (n := p) (p := n)
              epsδ (getIdxEps (Γ := Γ) (s := (.dim n (.dim p .scalar))) epsCtx B) δR
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                B))))
        (t₂S :=
          Spec.matMulSpec (α := SpecScalar)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α :=
              SpecScalar) ctxS A)) δS)
        (t₂R :=
          Spec.matMulSpec (α := R)
            (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR A))
              δR)
        (eps₂ :=
          linfNorm
            (matMulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (m := n) (n := m) (p := p)
              (getIdxEps (Γ := Γ) (s := (.dim m (.dim n .scalar))) epsCtx A) epsδ
              (TorchLean.Tensor.swapAdjacentAxes (depth := 0) (getIdx (α := R) ctxR
                A)) δR))
        hdA hdB hEq
    simpa [hEq] using hctx'

end NFBackend

end

end RuntimeApprox
end Proofs
