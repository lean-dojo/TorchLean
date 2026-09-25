/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.IBP

/-!
# Forward CROWN Bounds

Shared affine forms and exact-arithmetic transfer rules for graph CROWN.
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

/--
Context for affine (CROWN/DeepPoly) propagation.

Affine bounds are computed with respect to a single designated *input* node, whose flattened
dimension is `inputDim`.
-/
structure AffineCtx where
  /-- Node id treated as the input variable for affine bounds. -/
  inputId  : Nat
  /-- Flattened input dimension. -/
  inputDim : Nat

/-- Identity affine map on a flattened vector of length `n`. -/
@[expose]
def affIdentity (n : Nat) : AffineVec α n n :=
  let A :=
    Tensor.dim (fun i =>
      Tensor.dim (fun j => Tensor.scalar (if decide (i.val = j.val) then 1 else 0)))
  let c := Tensor.full (α:=α) (.dim n .scalar) 0
  { A := A, c := c }

/-- Pointwise addition of two affine maps with the same input and output dimensions. -/
def affAdd {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.addSpec a1.A a2.A, c := Tensor.addSpec a1.c a2.c }

/-- Pointwise subtraction of two affine maps with the same input and output dimensions. -/
def affSub {n m : Nat} (a1 a2 : AffineVec α n m) : AffineVec α n m :=
  { A := Tensor.subSpec a1.A a2.A, c := Tensor.subSpec a1.c a2.c }

-- Affine helpers for linear/matmul are handled by the explicit transfer rules below.

/--
Flatten a typed convolution into the affine map it denotes.

The CROWN pass uses this when a convolution is linear in the selected input. Keeping the conversion
here lets convolution share the same affine machinery as linear and matmul nodes.

-/
@[expose] def affOfConv (config : NN.IR.ConvParams α) (leading : Shape) :
    AffineVec α (config.input leading).size (config.output leading).size :=
  let outSpatial :=
    Spec.convOutSpatialDilated config.inputSpatial config.kernel config.stride config.dilation
      config.padding config.paddingAfter
  let W :=
    NN.MLTheory.CROWN.convLinearMatrix (α := α) (inSpatial := config.inputSpatial)
      config.spec config.dilation config.paddingAfter config.groups leading
  let b := NN.MLTheory.CROWN.convBiasBroadcast (α := α) (outSpatial := outSpatial)
    config.spec.bias leading
  AffineVec.ofLinear (α:=α)
    (inDim := (config.input leading).size)
    (outDim := (config.output leading).size)
    W b

/-!
For a chosen flattened input node `ctx.inputId`, the pass computes a pair of affine forms
`loAff(x) ≤ node(x) ≤ hiAff(x)` for each supported node.  The transfer rules below use the usual
CROWN/DeepPoly ingredients:

- Linear layers use sign-splitting (`W⁺/W⁻`) to combine parent bounds.
- ReLU uses the standard triangle upper bound and a simple evidence-based lower choice (0 vs x).
- Exp/log use secant/tangent bounds (convex/concave).
- Softmax and LayerNorm use conservative value enclosures for the selected axis or suffix.

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

/-- Algebraic composition of affine bounds through `W*x + b` on exact scalar backends.

Rounded backends must account for coefficient errors as well as final evaluation errors.
The graph runner selects directed backward propagation for those backends.
-/
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
