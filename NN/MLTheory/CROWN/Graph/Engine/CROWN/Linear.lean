/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Affine

/-!
# Forward CROWN Bounds

This module computes lower and upper affine bounds for each graph node. The pass is
Linear nodes compose affine lower and upper forms. Operator-specific rules may retain more
dependence, while unsupported or numerically unjustified cases fall back to constant affine bounds
derived from the already-computed IBP box.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/-!
For a chosen flattened input node `ctx.inputId`, the pass computes a pair of affine forms
`loAff(x) ≤ node(x) ≤ hiAff(x)` for each supported node.  The transfer rules below use the usual
CROWN/DeepPoly ingredients:

- Linear layers use sign-splitting (`W⁺/W⁻`) to combine parent bounds.
- ReLU uses the standard triangle upper bound and a simple evidence-based lower choice (0 vs x).
- Exp/log use secant/tangent bounds (convex/concave).
- Softmax and LayerNorm use conservative last-axis relaxations.

Unsupported axes or shape mismatches fall back to constant affine bounds derived from the IBP box.
-/

/-- Exact lower and upper affine bounds for the identity node. -/
@[expose] def boundsIdentity (n : Nat) : FlatAffineBounds α :=
  { inDim := n, outDim := n, loAff := affIdentity (α:=α) n, hiAff := affIdentity (α:=α) n }

/-- Constant affine bounds with zero coefficient matrix and explicit lower/upper offsets. -/
@[expose]
def boundsConst (inputDim outDim : Nat) (lo hi : Tensor α [outDim]) :
  FlatAffineBounds α :=
  let zA := Tensor.full (α:=α) (.dim outDim (.dim inputDim .scalar)) 0
  { inDim := inputDim
    outDim := outDim
    loAff := { A := zA, c := lo }
    hiAff := { A := zA, c := hi } }

/-- Propagate affine lower/upper bounds through an affine layer `W*x + b`. -/
def propagateLinearBounds
  {n m : Nat}
  (W : Tensor α [m, n])
  (b : Tensor α [m])
  (xB : FlatAffineBounds α)
  (hout : xB.outDim = n) : FlatAffineBounds α := by
  -- Align parent affines to outDim=n.
  let xLo : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.loAff
  let xHi : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.hiAff
  let Wpos := IBP.matPos (α:=α) (m:=m) (n:=n) W
  let Wneg := IBP.matNeg (α:=α) (m:=m) (n:=n) W
  let A_hi :=
    Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xHi.A) (Spec.matMulSpec (α:=α) Wneg xLo.A)
  let c_hi :=
    Tensor.addSpec
      (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xHi.c) (Spec.matVecMulSpec (α:=α)
        Wneg xLo.c))
      b
  let A_lo :=
    Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xLo.A) (Spec.matMulSpec (α:=α) Wneg xHi.A)
  let c_lo :=
    Tensor.addSpec
      (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xLo.c) (Spec.matVecMulSpec (α:=α)
        Wneg xHi.c))
      b
  exact
    { inDim := xB.inDim
      outDim := m
      loAff := { A := A_lo, c := c_lo }
      hiAff := { A := A_hi, c := c_hi } }

/-- Compose an affine form with a diagonal affine relaxation `slope * x + bias`. -/
def affApplyDiag {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  let A' :=
    Tensor.dim fun i =>
      Tensor.dim fun j =>
        Tensor.scalar (Tensor.getScalar slopes i * Spec.get2 aff.A i j)
  let c' :=
    Tensor.dim fun i =>
      Tensor.scalar
        (Tensor.getScalar slopes i * Tensor.getScalar aff.c i + Tensor.getScalar bias i)
  { A := A', c := c' }

/-- Apply a diagonal relaxation for an upper bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedUpper {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  let A' :=
    Tensor.dim fun i =>
      let si := Tensor.getScalar slopes i
      let rows := if decide (si > 0) then xHi.A else xLo.A
      Tensor.dim fun j => Tensor.scalar (si * Spec.get2 rows i j)
  let c' :=
    Tensor.dim fun i =>
      let si := Tensor.getScalar slopes i
      let constants := if decide (si > 0) then xHi.c else xLo.c
      Tensor.scalar (si * Tensor.getScalar constants i + Tensor.getScalar bias i)
  { A := A', c := c' }

/-- Apply a diagonal relaxation for a lower bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedLower {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  let A' :=
    Tensor.dim fun i =>
      let si := Tensor.getScalar slopes i
      let rows := if decide (si > 0) then xLo.A else xHi.A
      Tensor.dim fun j => Tensor.scalar (si * Spec.get2 rows i j)
  let c' :=
    Tensor.dim fun i =>
      let si := Tensor.getScalar slopes i
      let constants := if decide (si > 0) then xLo.c else xHi.c
      Tensor.scalar (si * Tensor.getScalar constants i + Tensor.getScalar bias i)
  { A := A', c := c' }


end NN.MLTheory.CROWN.Graph
