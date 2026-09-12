/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.BackwardObjective
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# ResidualAffine

Helpers for assembling PDE residual bounds using affine CROWN relaxations
where available.

This module provides:
- Evaluation of affine upper/lower bounds on the network output `u(x)`.
- McCormick-style linear upper/lower envelopes for scalar products over
  independent intervals (for Burgers-type residuals).
- A compact branch-and-bound splitter on the 1D input box to tighten bounds by
  subdividing the domain and taking the envelope across sub-boxes.

Notes:
- We intentionally keep this file numeric (Float) and specialized to this workflow.

References:
- CROWN / DeepPoly-style affine bounds: `https://arxiv.org/abs/1811.00866`
- alpha,beta-CROWN (context): `https://arxiv.org/abs/2103.06624`
-/

@[expose] public section


namespace NN.Verification.PINN.ResidualAffine

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Spec TorchLean
open TorchLean.Tensor

/-- Forward CROWN/DeepPoly bounds for the scalar output `u` (or sum of outputs). -/
def crownUBoundsForward (g : Graph) (ps : ParamStore Float)
  (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  match ps.inputBoxes[0]? with
  | none => none
  | some (inB : FlatBox Float) =>
    let ctx : AffineCtx := { inputId := 0, inputDim := inB.dim }
    let crown := runCROWN (α:=Float) g ps ctx ibp
    let outId : Nat := g.nodes.size - 1
    match evalCROWNOutputBox? (α := Float) crown inB outId inB.dim with
    | .ok outB =>
        let ulo := TorchLean.Tensor.sumSpec outB.lo
        let uhi := TorchLean.Tensor.sumSpec outB.hi
        some (ulo, uhi)
    | .error _ => none

/-- Objective-dependent (backward/dual) CROWN bounds for the scalar output `u` (or sum of outputs).
  -/
def crownUBoundsBackward (g : Graph) (ps : ParamStore Float)
  (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  match ps.inputBoxes[0]? with
  | none => none
  | some (inB : FlatBox Float) =>
    let ctx : AffineCtx := { inputId := 0, inputDim := inB.dim }
    let outId : Nat := g.nodes.size - 1
    let outDim : Nat :=
      match outputBox? ibp outId with
      | .ok B => B.dim
      | .error _ => 0
    let objV : Tensor Float [outDim] := Tensor.full (α := Float) (.dim outDim
      .scalar) 1.0
    let obj : FlatTensor Float := { n := outDim, v := objV }
    match backwardObjectiveBox? (α := Float) g ps ctx ibp inB outId obj with
    | .ok outB =>
        let ulo := TorchLean.Tensor.sumSpec outB.lo
        let uhi := TorchLean.Tensor.sumSpec outB.hi
        some (ulo, uhi)
    | .error _ => none

/-- Scalar product upper envelope over rectangles using McCormick.

Given $u\in[l_x,u_x]$ and $v\in[l_y,u_y]$, this returns coefficients
$(a_x,a_y,c)$ for the affine upper bound

$$
uv\le a_xu+a_yv+c.
$$

It chooses the tighter of the two classical McCormick upper planes.
-/
def mccormickUpper (lx ux ly uy : Float) : (Float × Float × Float) :=
  let cx := (lx + ux) * 0.5
  let cy := (ly + uy) * 0.5
  let u1 := ux * cy + ly * cx - ux * ly
  let u2 := lx * cy + uy * cx - lx * uy
  if u1 ≤ u2 then
    -- plane: uv ≤ ly*u + ux*v - ux*ly
    (ly, ux, -(ux * ly))
  else
    -- plane: uv ≤ uy*u + lx*v - lx*uy
    (uy, lx, -(lx * uy))

/-- Scalar product lower envelope over rectangles using McCormick.

Given $u\in[l_x,u_x]$ and $v\in[l_y,u_y]$, this returns an affine lower bound
$uv\ge a_xu+a_yv+c$.
-/
def mccormickLower (lx ux ly uy : Float) : (Float × Float × Float) :=
  -- Lower planes: uv ≥ uy*u + ux*v - ux*uy and uv ≥ ly*u + lx*v - lx*ly
  let l1 := uy * ((ux + lx) * 0.5) + ux * ((uy + ly) * 0.5) - ux * uy
  let l2 := ly * ((ux + lx) * 0.5) + lx * ((uy + ly) * 0.5) - lx * ly
  if l1 ≥ l2 then
    (uy, ux, -(ux * uy))
  else
    (ly, lx, -(lx * ly))

/-- Evaluate $a_xu+a_yv+c$ on the box
$u\in[u_{\mathrm{lo}},u_{\mathrm{hi}}]$,
$v\in[v_{\mathrm{lo}},v_{\mathrm{hi}}]$ to produce a numeric interval bound. -/
def evalAffine2DOnBox (ax ay c ul uh vl vh : Float) : (Float × Float) :=
  let p1 := ax * ul + ay * vl + c
  let p2 := ax * ul + ay * vh + c
  let p3 := ax * uh + ay * vl + c
  let p4 := ax * uh + ay * vh + c
  let lo1 := if p1 < p2 then p1 else p2
  let lo2 := if p3 < p4 then p3 else p4
  let hi1 := if p1 > p2 then p1 else p2
  let hi2 := if p3 > p4 then p3 else p4
  ((if lo1 < lo2 then lo1 else lo2), (if hi1 > hi2 then hi1 else hi2))

end NN.Verification.PINN.ResidualAffine
