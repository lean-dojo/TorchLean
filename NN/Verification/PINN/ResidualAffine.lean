/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph

/-!
# ResidualAffine

Helpers for assembling PDE residual bounds using affine CROWN relaxations
where available.

This module provides:
- Eval of an affine upper/lower bound on the network output u(x) using runAffine
  and AffineVec.eval_on_box, given the current input box.
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
open _root_.Spec
open _root_.Spec.Tensor

/-- Cast a dimension-indexed vector box across definitional equality of Nats. -/
def castBoxDimEq {n n' : Nat}
  (h : n = n')
  (B : Box Float (.dim n .scalar)) : Box Float (.dim n' .scalar) := by
  simpa [h] using B

/-- Forward CROWN/DeepPoly bounds for the scalar output `u` (or sum of outputs). -/
def crownUBoundsForward (g : Graph) (ps : ParamStore Float)
  (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  match ps.inputBoxes[0]? with
  | none => none
  | some (inB : FlatBox Float) =>
    let ctx : AffineCtx := { inputId := 0, inputDim := inB.dim }
    let crown := runCROWN (α:=Float) g ps ctx ibp
    let outId : Nat := g.nodes.size - 1
    match crown[outId]! with
    | some bout =>
      let inB0 : Box Float (.dim inB.dim .scalar) := { lo := inB.lo, hi := inB.hi }
      if hdim : inB.dim = bout.inDim then
        let inB1 := castBoxDimEq (h:=hdim) inB0
        let outLoBox := NN.MLTheory.CROWN.AffineVec.evalOnBox (α:=Float) bout.loAff inB1
        let outHiBox := NN.MLTheory.CROWN.AffineVec.evalOnBox (α:=Float) bout.hiAff inB1
        let ulo := Spec.Tensor.sumSpec outLoBox.lo
        let uhi := Spec.Tensor.sumSpec outHiBox.hi
        some (ulo, uhi)
      else
        none
    | none => none

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
      match ibp[outId]! with
      | some B => B.dim
      | none => 0
    let objV : Tensor Float (.dim outDim .scalar) := Spec.fill (α := Float) 1.0 (.dim outDim
      .scalar)
    let obj : FlatVec Float := { n := outDim, v := objV }
    match runCROWNBackwardObjective (α := Float) g ps ctx ibp outId obj with
    | some bout =>
      let inB0 : Box Float (.dim inB.dim .scalar) := { lo := inB.lo, hi := inB.hi }
      if hdim : inB.dim = bout.inDim then
        let inB1 := castBoxDimEq (h:=hdim) inB0
        let outLoBox := NN.MLTheory.CROWN.AffineVec.evalOnBox (α:=Float) bout.loAff inB1
        let outHiBox := NN.MLTheory.CROWN.AffineVec.evalOnBox (α:=Float) bout.hiAff inB1
        let ulo := Spec.Tensor.sumSpec outLoBox.lo
        let uhi := Spec.Tensor.sumSpec outHiBox.hi
        some (ulo, uhi)
      else
        none
    | none => none

/-- Backwards-compatible name used by the PINN CLI: treat "affine" u-bounds as objective-dependent
  CROWN. -/
def affineUBounds (g : Graph) (ps : ParamStore Float)
  (ibp : Array (Option (FlatBox Float))) : Option (Float × Float) :=
  crownUBoundsBackward g ps ibp

/-- Scalar product upper envelope over rectangles using McCormick.
    Given u∈[lx,ux], v∈[ly,uy], returns an affine upper bound of the form
      uv ≤ ax*u + ay*v + c,
    as coefficients (ax, ay, c).
    We pick the tighter of the two classical McCormick upper planes.
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
    Given u∈[lx,ux], v∈[ly,uy], returns an affine lower bound of the form
      uv ≥ ax*u + ay*v + c.
-/
def mccormickLower (lx ux ly uy : Float) : (Float × Float × Float) :=
  -- Lower planes: uv ≥ uy*u + ux*v - ux*uy and uv ≥ ly*u + lx*v - lx*ly
  let l1 := uy * ((ux + lx) * 0.5) + ux * ((uy + ly) * 0.5) - ux * uy
  let l2 := ly * ((ux + lx) * 0.5) + lx * ((uy + ly) * 0.5) - lx * ly
  if l1 ≥ l2 then
    (uy, ux, -(ux * uy))
  else
    (ly, lx, -(lx * ly))

/-- Evaluate an affine ax*u + ay*v + c on intervals u∈[ul,uh], v∈[vl,vh]
    to produce a numeric interval bound. -/
def eval2OnBox (ax ay c ul uh vl vh : Float) : (Float × Float) :=
  let p1 := ax * ul + ay * vl + c
  let p2 := ax * ul + ay * vh + c
  let p3 := ax * uh + ay * vl + c
  let p4 := ax * uh + ay * vh + c
  let lo1 := if p1 < p2 then p1 else p2
  let lo2 := if p3 < p4 then p3 else p4
  let hi1 := if p1 > p2 then p1 else p2
  let hi2 := if p3 > p4 then p3 else p4
  ((if lo1 < lo2 then lo1 else lo2), (if hi1 > hi2 then hi1 else hi2))

/-- Basic 1D branch-and-bound over the input box [x-ε,x+ε]. Recursively splits the
    box up to `maxDepth` or until width ≤ `minWidth`. On each sub-box it calls
    the provided bounding function `boundOn` which must return (lo, hi).
    Returns the tightest global (min lo, max hi) across sub-boxes. -/
def bnb1D (x eps : Float) (maxDepth : Nat) (minWidth : Float)
   (boundOn : Float → Float → IO (Float × Float)) : IO (Float × Float) := do
  let rec go (a b : Float) (d : Nat) : IO (Float × Float) := do
    match d with
    | 0 =>
      boundOn ((a + b) * 0.5) ((b - a) * 0.5)
    | Nat.succ d' =>
      if (b - a) ≤ minWidth then
        boundOn ((a + b) * 0.5) ((b - a) * 0.5)
      else
        let mid := (a + b) * 0.5
        let (l1, h1) ← go a mid d'
        let (l2, h2) ← go mid b d'
        pure (if l1 < l2 then l1 else l2, if h1 > h2 then h1 else h2)
  go (x - eps) (x + eps) maxDepth

end NN.Verification.PINN.ResidualAffine
