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
DeepPoly/CROWN-style: linear nodes compose exactly, nonlinear nodes attach local relaxations, and
unsupported cases fall back to constant affine bounds derived from the already-computed IBP box.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
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
def boundsConst (inputDim outDim : Nat) (lo hi : Tensor α (.dim outDim .scalar)) :
  FlatAffineBounds α :=
  let zA := Spec.fill (α:=α) 0 (.dim outDim (.dim inputDim .scalar))
  { inDim := inputDim
    outDim := outDim
    loAff := { A := zA, c := lo }
    hiAff := { A := zA, c := hi } }

/-- Positive part of a matrix, used for sign-splitting affine bounds through linear layers. -/
def matPos {m n : Nat} (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n
  .scalar)) :=
  match W with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar w => Tensor.scalar (if w > 0 then w else 0)))

/-- Negative part of a matrix, used with `matPos` to propagate lower and upper affine forms. -/
def matNeg {m n : Nat} (W : Tensor α (.dim m (.dim n .scalar))) : Tensor α (.dim m (.dim n
  .scalar)) :=
  match W with
  | .dim rows =>
    Tensor.dim (fun i =>
      match rows i with
      | .dim cols =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar w => Tensor.scalar (if w > 0 then 0 else w)))

/-- Materialize an affine vector so later graph passes do not accumulate deep tensor closures. -/
def materializeAffineVec {inDim outDim : Nat} (a : AffineVec α inDim outDim) : AffineVec α
  inDim outDim :=
  { A := Tensor.materialize a.A, c := Tensor.materialize a.c }

/-- Propagate affine lower/upper bounds through an affine layer `W*x + b`. -/
def propagateLinearBounds
  {n m : Nat}
  (W : Tensor α (.dim m (.dim n .scalar)))
  (b : Tensor α (.dim m .scalar))
  (xB : FlatAffineBounds α)
  (hout : xB.outDim = n) : FlatAffineBounds α := by
  -- Align parent affines to outDim=n.
  let xLo : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.loAff
  let xHi : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.hiAff
  let xLo := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xLo
  let xHi := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xHi
  let Wpos := matPos (α:=α) (m:=m) (n:=n) W
  let Wneg := matNeg (α:=α) (m:=m) (n:=n) W
  let A_hi :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xHi.A) (Spec.matMulSpec (α:=α) Wneg xLo.A)
  let c_hi :=
    Tensor.materialize <|
      Tensor.addSpec
        (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xHi.c) (Spec.matVecMulSpec (α:=α)
          Wneg xLo.c))
        b
  let A_lo :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xLo.A) (Spec.matMulSpec (α:=α) Wneg xHi.A)
  let c_lo :=
    Tensor.materialize <|
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
  (slopes bias : Tensor α (.dim outDim .scalar))
  (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, aff.A, aff.c with
  | .dim sF, .dim bF, .dim rows, .dim cF =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i, rows i with
        | .scalar si, .dim cols =>
          Tensor.dim (fun j =>
            match cols j with
            | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, cF i, bF i with
        | .scalar si, .scalar ci, .scalar bi => Tensor.scalar (si * ci + bi))
    { A := A', c := c' }

/-- Apply a diagonal relaxation for an upper bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedUpper {inDim outDim : Nat}
  (slopes bias : Tensor α (.dim outDim .scalar))
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsU i else rowsL i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cU i else cL i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }

/-- Apply a diagonal relaxation for a lower bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedLower {inDim outDim : Nat}
  (slopes bias : Tensor α (.dim outDim .scalar))
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsL i else rowsU i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cL i else cU i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }

/--
One-dimensional sigmoid linear bounds on an interval `[l,u]`.

Returns `(a_lo, b_lo, a_hi, b_hi)` for lower and upper affine lines.
-/
def sigmoidLineBounds (l u : α) : α × α × α × α :=
  let σ (x : α) := Activation.Math.sigmoidSpec (α := α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α := α) x
  if u < Numbers.zero then
    -- Convex region: secant is an upper bound, tangent is a lower bound.
    let den := u - l
    let a_hi := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_hi := σ l - a_hi * l
    let a_lo := σ' u
    let b_lo := σ u - a_lo * u
    (a_lo, b_lo, a_hi, b_hi)
  else if l > Numbers.zero then
    -- Concave region: tangent is an upper bound, secant is a lower bound.
    let a_hi := σ' l
    let b_hi := σ l - a_hi * l
    let den := u - l
    let a_lo := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_lo := σ l - a_lo * l
    (a_lo, b_lo, a_hi, b_hi)
  else
    -- Crossing the inflection: fall back to constant bounds.
    (Numbers.zero, σ l, Numbers.zero, σ u)

/-- Propagate CROWN bounds through ReLU using standard per-neuron triangle relaxations. -/
def propagateReluBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α := by
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let relaxLo0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVectorLower (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

/-- Propagate ReLU bounds with externally supplied α slopes for crossing neurons. -/
def propagateReluBoundsWithAlpha
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim)
  (alpha : Tensor α (.dim preB.dim .scalar)) : FlatAffineBounds α := by
  -- Upper relaxation: standard secant/tight bounds (independent of α).
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  -- Lower relaxation: for crossing bounds l < 0 < u, use a provided per-neuron α ∈ [0,1]
  -- (line y ≥ α x), which is always sound for ReLU; stable regions override α.
  let clamp01 (x : α) : α :=
    let x0 := if x > Numbers.zero then x else Numbers.zero
    if x0 > Numbers.one then Numbers.one else x0
  let relaxLo0 : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) (.dim preB.dim .scalar) :=
    match preB.lo, preB.hi, alpha with
    | .dim lF, .dim uF, .dim aF =>
      Tensor.dim (fun i =>
        match lF i, uF i, aF i with
        | .scalar l, .scalar u, .scalar a =>
          if u > Numbers.zero then
            if l > Numbers.zero then
              Tensor.scalar { slope := Numbers.one, bias := Numbers.zero }
            else
              Tensor.scalar { slope := clamp01 a, bias := Numbers.zero }
          else
            Tensor.scalar { slope := Numbers.zero, bias := Numbers.zero })
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

/-- Propagate CROWN bounds through `exp` using tangent/secant relaxations. -/
def propagateExpBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  -- Per-component [l,u]
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        let b := MathFunctions.exp l - a * l
        Tensor.scalar b)
  -- Lower: tangent at l
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        Tensor.scalar (MathFunctions.exp l))
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        let a := MathFunctions.exp l
        Tensor.scalar (MathFunctions.exp l - a * l))
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through log on the positive-domain convention used by the verifier. -/
def propagateLogBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  -- Clamp to positive domain.
  let loSafe : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let hiSafe : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match fhi i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let floS := getDimScalarFn (α:=α) loSafe
  let fhiS := getDimScalarFn (α:=α) hiSafe
  -- Upper: tangent at loSafe (concave ⇒ tangent is over-approx).
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l => Tensor.scalar (Numbers.one / l))
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l =>
        let a := Numbers.one / l
        Tensor.scalar (MathFunctions.log l - a * l))
  -- Lower: secant on [loSafe, hiSafe] (concave ⇒ secant is under-approx).
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        let b := MathFunctions.log l - a * l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through sigmoid with convex/concave one-dimensional relaxations. -/
def propagateSigmoidBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let σ (x : α) := Activation.Math.sigmoidSpec (α:=α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α:=α) x
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        -- Convex for x ≤ 0, concave for x ≥ 0; crossing uses constant bounds.
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else if l > Numbers.zero then
            σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else if l > Numbers.zero then
            let a := σ' l
            σ l - a * l
          else
            -- Crossing: constant upper bound = σ(u)
            σ u
        Tensor.scalar b)
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            σ' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := σ' u
            σ u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else
            -- Crossing: constant lower bound = σ(l)
            σ l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through tanh with convex/concave one-dimensional relaxations. -/
def propagateTanhBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let t (x : α) := Activation.Math.tanhSpec (α:=α) x
  let t' (x : α) := Activation.Math.tanhDerivSpec (α:=α) x
  let slopes_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else if l > Numbers.zero then
            t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else if l > Numbers.zero then
            let a := t' l
            t l - a * l
          else
            t u
        Tensor.scalar b)
  let slopes_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            t' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α (.dim preB.dim .scalar) :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := t' u
            t u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else
            t l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Permute the output coordinates of an affine bound when the output shape permutation is valid. -/
def permuteAffineOut {inDim outDim : Nat}
  (perm : Fin outDim → Fin outDim) (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match aff.A, aff.c with
  | .dim rows, .dim cvec =>
    { A := Tensor.dim (fun i => rows (perm i))
      c := Tensor.dim (fun i => cvec (perm i)) }

/-- Conservative CROWN-style affine bounds for softmax along the last tensor axis. -/
def propagateSoftmaxBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each last-axis slice has length 1, so softmax is identically 1.
    let ones : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.one (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) ones ones
  else
    let dim := preB.dim
    if dim % m = 0 then
      let expLo : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.lo
      let expHi : Tensor α (.dim dim .scalar) := Tensor.expSpec preB.hi
      let groups : Nat := dim / m
      let totalExpLo : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expLo [base + j]) 0
          Tensor.scalar sum)
      let totalExpHi : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sum : α := (List.range m).foldl (fun acc j => acc + getAtOrZero expHi [base + j]) 0
          Tensor.scalar sum)
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      -- Upper bound via logistic with C = Σ_{j≠i} exp(lo_j)
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aHi
            else
              Tensor.scalar Numbers.zero)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpLo [g]
            let eLi := getAtOrZero expLo [i.val]
            let c := tot - eLi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (_aLo, _bLo, aHi, bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bHi - aHi * logC)
            else
              Tensor.scalar Numbers.one)
      -- Lower bound via logistic with C = Σ_{j≠i} exp(hi_j)
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, _bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar aLo
            else
              Tensor.scalar Numbers.zero)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let tot := getAtOrZero totalExpHi [g]
            let eUi := getAtOrZero expHi [i.val]
            let c := tot - eUi
            if c > Numbers.epsilon then
              let logC := MathFunctions.log c
              let (aLo, bLo, _aHi, _bHi) := sigmoidLineBounds (α := α) (l - logC) (u - logC)
              Tensor.scalar (bLo - aLo * logC)
            else
              Tensor.scalar Numbers.zero)
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: fall back to trivial [0,1] bounds.
      let zeros : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim dim .scalar)
      let ones : Tensor α (.dim dim .scalar) := Spec.fill (α := α) Numbers.one (.dim dim .scalar)
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) zeros ones

/-- Conservative affine bounds for layer normalization over the last tensor axis. -/
def propagateLayernormBoundsLastAxis
  (s : Shape) (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α := α) (n := xB.inDim) (m := xB.outDim) (m' := preB.dim) hout xB.hiAff
  let m := lastDimLen s
  if m = 0 then
    { inDim := xB.inDim, outDim := preB.dim, loAff := xLo, hiAff := xHi }
  else if m = 1 then
    -- Each slice has length 1: (x - mean)/sqrt(var+eps) = 0.
    let zeros : Tensor α (.dim preB.dim .scalar) := Spec.fill (α := α) Numbers.zero (.dim preB.dim
      .scalar)
    boundsConst (α := α) (inputDim := xB.inDim) (outDim := preB.dim) zeros zeros
  else
    let dim := preB.dim
    if dim % m = 0 then
      let groups : Nat := dim / m
      let mA : α := (m : Nat)
      let denLo : α := MathFunctions.sqrt Numbers.epsilon
      let muLoG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumLo : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.lo [base +
            j]) 0
          Tensor.scalar (sumLo / mA))
      let muHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let sumHi : α := (List.range m).foldl (fun acc j => acc + getAtOrZero preB.hi [base +
            j]) 0
          Tensor.scalar (sumHi / mA))
      let denHiG : Tensor α (.dim groups .scalar) :=
        Tensor.dim (fun g =>
          let base := g.val * m
          let muLo := getAtOrZero muLoG [g.val]
          let muHi := getAtOrZero muHiG [g.val]
          let loSlice : Tensor α (.dim m .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (getAtOrZero preB.lo [base + j.val]))
          let hiSlice : Tensor α (.dim m .scalar) :=
            Tensor.dim (fun j => Tensor.scalar (getAtOrZero preB.hi [base + j.val]))
          let varHi := layerNormVarianceUpper (α := α) loSlice hiSlice muLo muHi
          Tensor.scalar (MathFunctions.sqrt (varHi + Numbers.epsilon)))
      let flo := getDimScalarFn (α := α) preB.lo
      let fhi := getDimScalarFn (α := α) preB.hi
      let slopes_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (uU - uL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_hi : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muLo := getAtOrZero muLoG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let uL :=
              let num := l - muLo
              let den := if decide (l > muLo) then denLo else denHi
              num / den
            let uU :=
              let num := u - muLo
              let den := if decide (u > muLo) then denLo else denHi
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (uU - uL) / denx
              Tensor.scalar (uL - a * l)
            else
              Tensor.scalar (if uL > uU then uL else uU))
      let slopes_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            let a := if denx > Numbers.epsilon then (lU - lL) / denx else Numbers.zero
            Tensor.scalar a)
      let bias_lo : Tensor α (.dim dim .scalar) :=
        Tensor.dim (fun i =>
          let g := i.val / m
          let muHi := getAtOrZero muHiG [g]
          let denHi := getAtOrZero denHiG [g]
          match flo i, fhi i with
          | .scalar l, .scalar u =>
            let lL :=
              let num := l - muHi
              let den := if decide (l > muHi) then denHi else denLo
              num / den
            let lU :=
              let num := u - muHi
              let den := if decide (u > muHi) then denHi else denLo
              num / den
            let denx := u - l
            if denx > Numbers.epsilon then
              let a := (lU - lL) / denx
              Tensor.scalar (lL - a * l)
            else
              Tensor.scalar (if lL < lU then lL else lU))
      let loAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_lo bias_lo xLo
      let hiAff := affApplyDiag (α := α) (inDim := xB.inDim) (outDim := dim) slopes_hi bias_hi xHi
      { inDim := xB.inDim, outDim := dim, loAff := loAff, hiAff := hiAff }
    else
      -- Shape mismatch: conservative constant bounds from IBP on this op.
      let (flatLo, flatHi) :=
        ibpLayernormLastTensor (α := α) (s := .dim dim .scalar) preB.lo preB.hi
      boundsConst (α := α) (inputDim := xB.inDim) (outDim := dim) flatLo flatHi

/-- Internal matmul affine propagation using McCormick-style product planes. -/
private def propagateMatmulBounds
  (sA sB : Shape) (Bx By : FlatBox α)
  (aB bB : FlatAffineBounds α) :
  Option (FlatAffineBounds α) :=
  if hin : aB.inDim = bB.inDim then
    let inDim := aB.inDim
    let bLo : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.loAff
    let bHi : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.hiAff
    let split (a : α) : α × α :=
      if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero, a)
    let dims? : Option (Nat × Nat × Nat × Nat) :=
      match sA, sB with
      | .dim m (.dim k .scalar), .dim k' (.dim n .scalar) =>
        if k = k' then
          some (1, m, k, n)
        else
          none
      | .dim b (.dim m (.dim k .scalar)), .dim b' (.dim k' (.dim n .scalar)) =>
        if hb : b = b' then
          match hb with
          | rfl =>
            if k = k' then
              some (b, m, k, n)
            else
              none
        else
          none
      | _, _ => none
    match dims? with
    | none => none
    | some (batch, m, k, n) =>
      let dimA := batch * m * k
      let dimB := batch * k * n
      let outDim := batch * m * n
      if Bx.dim = dimA ∧ aB.outDim = dimA then
        if By.dim = dimB ∧ bB.outDim = dimB then
          let block : Nat := m * n
          let strideA : Nat := m * k
          let strideB : Nat := k * n

          let termUpperCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL

          let termUpperConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL + off

          let termLowerCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU

          let termLowerConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * Numbers.pointfive
            let cy := (ly + uy) * Numbers.pointfive
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let off := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU + off

          let A_hi : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termUpperCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_hi : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termUpperConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          let A_lo : Tensor α (.dim outDim (.dim inDim .scalar)) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range k).foldl (fun acc kk =>
                    acc + termLowerCoeff (baseA + i * k + kk) (baseB + kk * n + j) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_lo : Tensor α (.dim outDim .scalar) :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let bi := t / block
              let rem := t % block
              let i := rem / n
              let j := rem % n
              let baseA := bi * strideA
              let baseB := bi * strideB
              let coeff :=
                (List.range k).foldl (fun acc kk =>
                  acc + termLowerConst (baseA + i * k + kk) (baseB + kk * n + j)
                ) 0
              Tensor.scalar coeff)

          some
            { inDim := inDim
              outDim := outDim
              loAff := { A := A_lo, c := c_lo }
              hiAff := { A := A_hi, c := c_hi } }
        else
          none
      else
        none
  else
    none

/-- Propagate affine bounds through componentwise multiplication using per-coordinate product planes. -/
def propagateMulElemBounds
  (Bx By : FlatBox α)
  (xB yB : FlatAffineBounds α)
  (houtX : xB.outDim = Bx.dim) (houtY : yB.outDim = By.dim) :
  Option (FlatAffineBounds α) :=
  -- Require equal vector lengths and equal input widths.
  if hdim : Bx.dim = By.dim then
    if hin : xB.inDim = yB.inDim then
      let n := Bx.dim
      let hyo : yB.outDim = n := Eq.trans houtY (Eq.symm hdim)
      let hBy : By.dim = n := by simpa [n] using (Eq.symm hdim)
      let ByLo : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.lo
      let ByHi : Tensor α (.dim n .scalar) := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.hi

      let xLo : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.loAff
      let xHi : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.hiAff
      let yLo0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.loAff
      let yHi0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.hiAff
        let yLo : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yLo0
        let yHi : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yHi0

        -- Helper to split a scalar coefficient into (pos, neg).
        let split (a : α) : α × α := if a > Numbers.zero then (a, Numbers.zero) else (Numbers.zero,
          a)

        -- Build row-wise A/c for upper and lower using a single selected McCormick plane per
        -- component.
        let A_hi :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose min of two upper planes at the interval center.
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy     -- coeff for x
                let aY := if u1 < u2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    Tensor.scalar (aXpos * xu + aXneg * xl + aYpos * yu + aYneg * yl)))
        let c_hi :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let u1 := ux * cy + ly * cx - ux * ly
                let u2 := lx * cy + uy * cx - lx * uy
                let aX := if u1 < u2 then ly else uy
                let aY := if u1 < u2 then ux else lx
                let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxu + aXneg * cxl + aYpos * cyu + aYneg * cyl + off))
        let A_lo :=
          match xLo.A, xHi.A, yLo.A, yHi.A, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim AxL, .dim AxU, .dim AyL, .dim AyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, AxL i, AxU i, AyL i, AyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .dim rowXL, .dim rowXU, .dim rowYL, .dim rowYU =>
                -- Choose max of two lower planes at the interval center.
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly     -- coeff for x
                let aY := if l1 > l2 then ux else lx     -- coeff for y
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.dim (fun j =>
                  match rowXL j, rowXU j, rowYL j, rowYU j with
                  | .scalar xl, .scalar xu, .scalar yl, .scalar yu =>
                    -- For lower bound, negative coeffs use the *upper* input bound.
                    Tensor.scalar (aXpos * xl + aXneg * xu + aYpos * yl + aYneg * yu)))
        let c_lo :=
          match xLo.c, xHi.c, yLo.c, yHi.c, Bx.lo, Bx.hi, ByLo, ByHi with
          | .dim cxL, .dim cxU, .dim cyL, .dim cyU, .dim lxF, .dim uxF, .dim lyF, .dim uyF =>
            Tensor.dim (fun i =>
              match lxF i, uxF i, lyF i, uyF i, cxL i, cxU i, cyL i, cyU i with
              | .scalar lx, .scalar ux, .scalar ly, .scalar uy,
                .scalar cxl, .scalar cxu, .scalar cyl, .scalar cyu =>
                let cx := (lx + ux) * Numbers.pointfive
                let cy := (ly + uy) * Numbers.pointfive
                let l1 := ux * cy + uy * cx - ux * uy
                let l2 := lx * cy + ly * cx - lx * ly
                let aX := if l1 > l2 then uy else ly
                let aY := if l1 > l2 then ux else lx
                let off := if l1 > l2 then (-(ux * uy)) else (-(lx * ly))
                let (aXpos, aXneg) := split aX
                let (aYpos, aYneg) := split aY
                Tensor.scalar (aXpos * cxl + aXneg * cxu + aYpos * cyl + aYneg * cyu + off))

        some
          { inDim := xB.inDim
            outDim := n
            loAff := { A := A_lo, c := c_lo }
            hiAff := { A := A_hi, c := c_hi } }
    else
      none
  else
    none

/--
Propagate a single node’s *affine bounds* (lower/upper) given parent bounds.

This is the CROWN/DeepPoly-style transfer step used by `runCROWN`. For node kinds without a
dedicated rule, we fall back to the IBP enclosure (turned into a constant affine bound).
-/
def propagateCROWNNode
  (nodes : Array Node) (ps : ParamStore α)
  (ibp : Array (Option (FlatBox α)))
  (bounds : Array (Option (FlatAffineBounds α)))
  (ctx : AffineCtx) (id : Nat) : Array (Option (FlatAffineBounds α)) :=
  let node := nodes[id]!
  let getB (pid : Nat) := (bounds[pid]!)
  match node.kind with
  | .input =>
    if node.id = ctx.inputId then
      bounds.set! id (some (boundsIdentity (α:=α) ctx.inputDim))
    else bounds
  | .const _ =>
    match ps.constVals[id]? with
    | some v =>
      -- Exact constant bounds.
      bounds.set! id (some (boundsConst (α:=α) ctx.inputDim v.n v.v v.v))
    | none => bounds
  | .detach =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some b => bounds.set! id (some b)
      | none => bounds
    | _ => bounds
  | .randUniform _ | .bernoulliMask _ | .abs | .sqrt | .permute _ | .maxElem | .minElem | .sin |
    .cos
  | .maxPool2d .. | .avgPool2d .. | .maxPool2dPad .. | .avgPool2dPad ..
  | .broadcastTo .. | .reduceSum .. | .reduceMean .. =>
    -- Conservative fallback: use IBP box as a constant affine bound (A = 0).
    match ibp[id]! with
    | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
    | none => bounds
  | .add =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.loAff
                hiAff := affAdd (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.hiAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .sub =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hout : b1.outDim = b2.outDim then
          if hin : b1.inDim = b2.inDim then
            let b1Lo : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.loAff)
            let b1Hi : AffineVec α b2.inDim b2.outDim :=
              castAffineIn (α:=α) (n:=b1.inDim) (n':=b2.inDim) (m:=b2.outDim) hin
                (castAffineOut (α:=α) (n:=b1.inDim) (m:=b1.outDim) (m':=b2.outDim) hout b1.hiAff)
            let out : FlatAffineBounds α :=
              { inDim := b2.inDim
                outDim := b2.outDim
                loAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Lo b2.hiAff
                hiAff := affSub (α:=α) (n:=b2.inDim) (m:=b2.outDim) b1Hi b2.loAff }
            bounds.set! id (some out)
          else bounds
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .linear =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ps.linearWB[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .matmul =>
    match node.parents with
    | p1 :: p2 :: _ =>
      -- General (batched) matmul: use McCormick relaxations per product term.
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some aAff, some bAff, some aBox, some bBox =>
        match propagateMatmulBounds (α:=α) (sA := nodes[p1]!.outShape) (sB := nodes[p2]!.outShape)
              aBox bBox aAff bAff with
        | some out =>
          bounds.set! id (some out)
        | none =>
          match ibp[id]! with
          | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
            Bout.hi))
          | none => bounds
      | _, _, _, _ => bounds
    | p1 :: _ =>
      match getB p1, ps.matmulW[id]? with
      | some xin, some p =>
        if hout : xin.outDim = p.n then
          let zb := Spec.fill (α:=α) 0 (.dim p.m .scalar)
          let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w zb xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .relu =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          let out := propagateReluBounds (α:=α) preB xin hout
          bounds.set! id (some out)
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .exp =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateExpBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .log =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateLogBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .inv =>
    -- Reciprocal has an asymptote at 0; we conservatively fall back to a constant affine
    -- bound derived from IBP (which widens to ±1/ε when the interval crosses 0).
    match ibp[id]! with
    | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo Bout.hi))
    | none => bounds
  | .sigmoid =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateSigmoidBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .tanh =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if hout : xin.outDim = preB.dim then
          bounds.set! id (some (propagateTanhBounds (α:=α) preB xin hout))
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .mul_elem =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some xB, some yB, some Bx, some By =>
        if hxo : xB.outDim = Bx.dim then
          if hyo : yB.outDim = By.dim then
            match propagateMulElemBounds (α:=α) Bx By xB yB hxo hyo with
            | some out => bounds.set! id (some out)
            | none =>
              match ibp[id]! with
              | some Bout => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim Bout.dim Bout.lo
                Bout.hi))
              | none => bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .sum =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        let onesRow : Tensor α (.dim 1 (.dim xin.outDim .scalar)) :=
          Spec.fill (α := α) Numbers.one (.dim 1 (.dim xin.outDim .scalar))
        let loAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.loAff.A
            c := Spec.matVecMulSpec onesRow xin.loAff.c }
        let hiAff : AffineVec α xin.inDim 1 :=
          { A := Spec.matMulSpec onesRow xin.hiAff.A
            c := Spec.matVecMulSpec onesRow xin.hiAff.c }
        bounds.set! id (some { inDim := xin.inDim, outDim := 1, loAff := loAff, hiAff := hiAff })
      | none => bounds
    | _ => bounds
  | .reshape _ _ =>
    -- Flattened representation preserves order; treat as identity.
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .flatten _ =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin => bounds.set! id (some xin)
      | none => bounds
    | _ => bounds
  | .concat _ =>
    -- Exact concatenation on flattened vectors.
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2 with
      | some b1, some b2 =>
        if hin : b1.inDim = b2.inDim then
          let b2Lo : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.loAff
          let b2Hi : AffineVec α b1.inDim b2.outDim :=
            castAffineIn (α := α) (n := b2.inDim) (n' := b1.inDim) (m := b2.outDim) hin.symm
              b2.hiAff
          match b1.loAff.A, b1.hiAff.A, b1.loAff.c, b1.hiAff.c, b2Lo.A, b2Hi.A, b2Lo.c, b2Hi.c with
          | .dim A1L, .dim A1U, .dim c1L, .dim c1U, .dim A2L, .dim A2U, .dim c2L, .dim c2U =>
            let outDim := b1.outDim + b2.outDim
            let ALo : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1L i1) (fun i2 => A2L i2) i)
            let AHi : Tensor α (.dim outDim (.dim b1.inDim .scalar)) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => A1U i1) (fun i2 => A2U i2) i)
            let cLo : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1L i1) (fun i2 => c2L i2) i)
            let cHi : Tensor α (.dim outDim .scalar) :=
              Tensor.dim (fun i =>
                Fin.addCases (fun i1 => c1U i1) (fun i2 => c2U i2) i)
            bounds.set! id
              (some
                { inDim := b1.inDim
                  outDim := outDim
                  loAff := { A := ALo, c := cLo }
                  hiAff := { A := AHi, c := cHi } })
        else bounds
      | _, _ => bounds
    | _ => bounds
  | .swap_first_two =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim m (.dim n rest) =>
          let sIn : Shape := .dim m (.dim n rest)
          if xin.outDim = sIn.size then
            let restSize := Shape.size rest
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              -- Empty tensor: permutation is trivial.
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := m * restSize
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let j := t / block
                let rem := t % block
                let i := rem / restSize
                let k := rem % restSize
                let tIn := i * (n * restSize) + j * restSize + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .transpose3dLastTwo =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match nodes[p1]!.outShape with
        | .dim a (.dim b (.dim c .scalar)) =>
          let sIn : Shape := .dim a (.dim b (.dim c .scalar))
          if xin.outDim = sIn.size then
            let outDim := xin.outDim
            if h0 : outDim = 0 then
              bounds.set! id (some xin)
            else
              haveI : NeZero outDim := ⟨h0⟩
              let block := c * b
              let perm : Fin outDim → Fin outDim := fun idx =>
                let t := idx.val
                let i := t / block
                let rem := t % block
                let k := rem / b
                let j := rem % b
                let tIn := i * (b * c) + j * c + k
                Fin.ofNat outDim tIn
              let loAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.loAff
              let hiAff := permuteAffineOut (α := α) (inDim := xin.inDim) (outDim := outDim) perm
                xin.hiAff
              bounds.set! id (some { inDim := xin.inDim, outDim := outDim, loAff := loAff, hiAff :=
                hiAff })
          else
            match ibp[id]! with
            | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
            | none => bounds
        | _ =>
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | none => bounds
    | _ => bounds
  | .layernorm axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateLayernormBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .softmax axis =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ibp[p1]! with
      | some xin, some preB =>
        if axis = Shape.rank node.outShape - 1 then
          if hout : xin.outDim = preB.dim then
            bounds.set! id (some (propagateSoftmaxBoundsLastAxis (α:=α) node.outShape preB xin
              hout))
          else
            bounds
        else
          match ibp[id]! with
          | some B => bounds.set! id (some (boundsConst (α:=α) ctx.inputDim B.dim B.lo B.hi))
          | none => bounds
      | _, _ => bounds
    | _ => bounds
  | .mseLoss =>
    match node.parents with
    | p1 :: p2 :: _ =>
      match getB p1, getB p2, ibp[p1]!, ibp[p2]! with
      | some yAff, some tAff, some yB, some tB =>
        if hout : yAff.outDim = yB.dim then
          if tAff.outDim = tB.dim then
            if hdim : yB.dim = tB.dim then
              if hout2 : yAff.outDim = tAff.outDim then
                if hin : yAff.inDim = tAff.inDim then
                  let yLo : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.loAff)
                  let yHi : AffineVec α tAff.inDim tAff.outDim :=
                    castAffineIn (α:=α) (n:=yAff.inDim) (n':=tAff.inDim) (m:=tAff.outDim) hin
                      (castAffineOut (α:=α) (n:=yAff.inDim) (m:=yAff.outDim) (m':=tAff.outDim) hout2
                        yAff.hiAff)
                  let tHiVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.hi
                  let tLoVec := castDimScalar (α:=α) (n:=tB.dim) (n':=yB.dim) (h:=hdim.symm) tB.lo
                  let diffLoVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.lo tHiVec
                  let diffHiVec : Tensor α (.dim yB.dim .scalar) := Tensor.subSpec yB.hi tLoVec
                  let n := yB.dim
                  let hOutToN : tAff.outDim = n := Eq.trans (Eq.symm hout2) hout
                  let yLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yLo
                  let yHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN yHi
                  let tLoN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.loAff
                  let tHiN : AffineVec α tAff.inDim n :=
                    castAffineOut (α:=α) (n:=tAff.inDim) (m:=tAff.outDim) (m':=n) hOutToN tAff.hiAff
                  let diffLoAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yLoN tHiN
                  let diffHiAff' : AffineVec α tAff.inDim n :=
                    affSub (α:=α) (n:=tAff.inDim) (m:=n) yHiN tLoN
                  -- Square relaxation on each component of `diff`.
                  let flo := getDimScalarFn (α := α) diffLoVec
                  let fhi := getDimScalarFn (α := α) diffHiVec
                  let slopes_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (u + l))
                  let bias_hi : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u => Tensor.scalar (-(u * l)))
                  let slopes_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (Numbers.two * d))
                  let bias_lo : Tensor α (.dim n .scalar) :=
                    Tensor.dim (fun i =>
                      match flo i, fhi i with
                      | .scalar l, .scalar u =>
                        let d := if u < Numbers.zero then u else if l > Numbers.zero then l else
                          Numbers.zero
                        Tensor.scalar (-(d * d)))
                  let sqLoAff :=
                    affApplyDiagSignedLower (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_lo
                      bias_lo diffLoAff' diffHiAff'
                  let sqHiAff :=
                    affApplyDiagSignedUpper (α:=α) (inDim:=tAff.inDim) (outDim:=n) slopes_hi
                      bias_hi diffLoAff' diffHiAff'
                  if n > 0 then
                    let nA : α := (n : Nat)
                    let scale : α := Numbers.one / nA
                    let scaleRow : Tensor α (.dim 1 (.dim n .scalar)) :=
                      Spec.fill (α := α) scale (.dim 1 (.dim n .scalar))
                    let outLo : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqLoAff.A
                        c := Spec.matVecMulSpec scaleRow sqLoAff.c }
                    let outHi : AffineVec α tAff.inDim 1 :=
                      { A := Spec.matMulSpec scaleRow sqHiAff.A
                        c := Spec.matVecMulSpec scaleRow sqHiAff.c }
                    bounds.set! id (some { inDim := tAff.inDim, outDim := 1, loAff := outLo, hiAff
                      := outHi })
                  else
                    let z : Tensor α (.dim 1 .scalar) := Spec.fill (α := α) Numbers.zero (.dim 1
                      .scalar)
                    bounds.set! id (some (boundsConst (α := α) ctx.inputDim 1 z z))
                else bounds
              else bounds
            else bounds
          else bounds
        else bounds
      | _, _, _, _ => bounds
    | _ => bounds
  | .conv2d .. =>
    match node.parents with
    | p1 :: _ =>
      match getB p1 with
      | some xin =>
        match ps.conv2dCfg[id]? with
        | some cfg =>
          let convIn := cfg.inC * cfg.inH * cfg.inW
          if _hs : cfg.stride = 0 then
            bounds
          else if hout : xin.outDim = convIn then
            let outH := (cfg.inH + 2 * cfg.padding - cfg.kH) / cfg.stride + 1
            let outW := (cfg.inW + 2 * cfg.padding - cfg.kW) / cfg.stride + 1
            let convAff := affOfConv2d (α:=α) cfg
            let out := propagateLinearBounds (α:=α) (n:=convIn) (m:=cfg.outC * outH * outW)
              convAff.A convAff.c xin hout
            bounds.set! id (some out)
          else bounds
        | none =>
          match ps.linearWB[id]? with
          | some p =>
            if hout : xin.outDim = p.n then
              let out := propagateLinearBounds (α:=α) (n:=p.n) (m:=p.m) p.w p.b xin hout
              bounds.set! id (some out)
            else bounds
          | none => bounds
      | none => bounds
    | _ => bounds
  | .batchNorm2dNchwEval .. =>
    match node.parents with
    | p1 :: _ =>
      match getB p1, ps.batchNorm2dNchwEval[id]? with
      | some xin, some cfg =>
        match batchNorm2dNchwEvalLinear? (α := α) nodes[p1]!.outShape cfg with
        | some p =>
          if hout : xin.outDim = p.n then
            let out := propagateLinearBounds (α := α) (n := p.n) (m := p.m) p.w p.b xin hout
            bounds.set! id (some out)
          else
            bounds
        | none => bounds
      | _, _ => bounds
    | _ => bounds

/-- Run the basic CROWN affine-bounds pass; requires prior IBP for per-node intervals. -/
def runCROWN (g : Graph) (ps : ParamStore α) (ctx : AffineCtx)
    (ibp : Array (Option (FlatBox α))) : Array (Option (FlatAffineBounds α)) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl (fun acc i => propagateCROWNNode (α:=α) g.nodes ps ibp acc ctx
    i) init

/-- Evaluate already-computed CROWN output affine bounds on an input box. -/
def evalCROWNOutputBox? (bounds : Array (Option (FlatAffineBounds α))) (xB : FlatBox α)
    (outputId inputDim : Nat) : Except String (FlatBox α) := do
  let outAff ←
    match bounds[outputId]? with
    | some (some outAff) => pure outAff
    | some none => throw s!"CROWN produced no affine bound at output node {outputId}"
    | none => throw s!"output node {outputId} is out of bounds for {bounds.size} CROWN entries"
  if hIn : outAff.inDim = inputDim then
    if hXB : xB.dim = inputDim then
      let outB := outAff.evalOnFlatBox xB (by simpa [hXB] using hIn.symm)
      pure { dim := outAff.outDim, lo := outB.lo, hi := outB.hi }
    else
      throw s!"input box dimension mismatch: got {xB.dim}, expected {inputDim}"
  else
    throw s!"CROWN input dimension mismatch: got {outAff.inDim}, expected {inputDim}"

/--
Run IBP, run forward CROWN, and evaluate the output affine bounds on the selected input box.

This is the common "forward CROWN output box" workflow. It keeps callers from open-coding the same
output-array lookup and input-dimension proof checks around `runCROWN`.
-/
def outputBoxCROWN? (g : Graph) (ps : ParamStore α) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) := do
  let ibp := runIBP (α := α) g ps
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }
  let crown := runCROWN (α := α) g ps ctx ibp
  evalCROWNOutputBox? (α := α) crown xB outputId inputDim

namespace ParamStore

/--
Run `outputBoxCROWN?` from an input-seeded parameter store.

This method form reads naturally at call sites that already thread a `ParamStore`.
-/
def outputBoxCROWN? (ps : ParamStore α) (g : Graph) (xB : FlatBox α)
    (inputId outputId inputDim : Nat) : Except String (FlatBox α) :=
  Graph.outputBoxCROWN? (α := α) g ps xB inputId outputId inputDim

end ParamStore

end NN.MLTheory.CROWN.Graph
