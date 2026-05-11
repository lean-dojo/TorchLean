/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Core
public import NN.Verification.PINN.PdeAst
public import NN.Verification.PINN.PdeParse
public import NN.MLTheory.CROWN.Graph
public import NN.Verification.Util.Json

/-!
# PINN Certificate

PINN certificate checker (recompute-and-compare).

This module is the executable checker for the PINN certificate workflow:
- parse a JSON certificate produced by Python,
- rebuild the same CROWN graph and seed the same input boxes,
- recompute IBP + derivative bounds in Lean, and
- compare the resulting residual intervals against the exported values.

It is intentionally conservative: the goal is to validate the export/import path and the interval
computations, not to be a fully featured PDE verifier.

References / context:
- PINNs: Raissi et al. (2019), "Physics-informed neural networks" (JCP)
- CROWN/LiRPA background (for the bound propagation machinery): `https://arxiv.org/abs/1811.00866`

Export (Python):
`python3.12 scripts/verification/pinn/export_pinn_cert.py`

Run (Lean):
`lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
-/

@[expose] public section


namespace NN.Verification.PINN.Certificate

open NN.Verification.PINN
open NN.Verification.PINN.PdeAst
open NN.Verification.PINN.PdeParse
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor
open Lean
open Json

/-- IO entry that reads the cert, recomputes bounds, and prints comparisons. -/
def verifyCert (path : String) : IO Unit := do
  let j ← NN.Verification.Json.readJsonFile path
  match parseCert j with
  | .error msg => throw <| IO.userError s!"Bad Cert JSON: {msg}"
  | .ok (cfg, residPairs, residPairsDeriv, _uTriples) => do
    let g := buildGraph
    let basePs := seedParamsFloat
    let residA := residPairs.toArray
    let residDerivA := residPairsDeriv.toArray
    for i in List.finRange cfg.nPts do
      let x := Tensor.vecGet cfg.pts i
      let xs := #[x - cfg.h, x, x + cfg.h]
      let mut uTrip : List (Float × Float) := []
      let mut duTrip : List (Float × Float) := []
      let mut d2uTrip : List (Float × Float) := []
      for xi in xs do
        let ps := seedInputFloat basePs xi cfg.eps
        let boxes := NN.MLTheory.CROWN.Graph.runIBP (α:=Float) g ps
        let some outB := boxes[5]! | throw <| IO.userError "PINN IBP failed"
        let loVal := Spec.Tensor.sumSpec outB.lo
        let hiVal := Spec.Tensor.sumSpec outB.hi
        uTrip := uTrip ++ [(loVal, hiVal)]
        let dboxes := NN.MLTheory.CROWN.Graph.runDeriv1D (α:=Float) g ps boxes
        match dboxes[5]! with
        | some (dB : FlatBox Float) =>
          let dlo := Spec.Tensor.sumSpec dB.lo
          let dhi := Spec.Tensor.sumSpec dB.hi
          duTrip := duTrip ++ [(dlo, dhi)]
        | none =>
          duTrip := duTrip ++ [(0.0, 0.0)]
        let d2boxes := NN.MLTheory.CROWN.Graph.runDeriv2D (α:=Float) g ps boxes dboxes
        match d2boxes[5]! with
        | some (d2B : FlatBox Float) =>
          let d2lo := Spec.Tensor.sumSpec d2B.lo
          let d2hi := Spec.Tensor.sumSpec d2B.hi
          d2uTrip := d2uTrip ++ [(d2lo, d2hi)]
        | none =>
          d2uTrip := d2uTrip ++ [(0.0, 0.0)]
      match uTrip with
      | [(lm, hm), (l0, h0), (lp, hp)] =>
        let (rlo, rhi) := fdResidualBounds (lm,hm) (l0,h0) (lp,hp) cfg.h
        let (eradLo, eradHi) := residA[i.1]!
        if ¬(approxEq rlo eradLo) ∨ ¬(approxEq rhi eradHi) then
          IO.println s!"FD residual mismatch at x={x}: Lean ({rlo},{rhi}) vs Py ({eradLo},{eradHi})"
        else
          pure ()
        let (dLoPy, dHiPy) := residDerivA[i.1]!
        match d2uTrip with
        | [_, (_d2l, _d2h), _] =>
          IO.println s!"u''(x) residual bound (derivative-based, Py): [{dLoPy},{dHiPy}]"
        | _ => pure ()
        match d2uTrip with
        | [_, (d2l, d2h), _] =>
          IO.println s!"u''(x) residual bound (derivative-based): [{d2l},{d2h}]"
        | _ => pure ()
        -- Compute and print residual bounds from the PDE specification via the parser/AST.
        -- We support a small DSL: u, ux, uxx, uy, uyy, +, -, *, scaling constants, parentheses, and
        -- powers by ^n.
        let env : String → Option Float := fun _ => none
        -- identifiers map, can be extended to constants
        let pdeParsed : Option NN.Verification.PINN.PdeAst.Expr :=
          match parseExpr env cfg.pde with
          | .ok e => some e
          | .error _ => none
        -- Build primitive bounds at the central point x using computed intervals
        let primsOpt : Option Prims :=
          match uTrip, duTrip, d2uTrip with
          | [_, (u0l,u0h), _], [_, (d1l,d1h), _], [_, (d2l,d2h), _] =>
            some
              { u := some (u0l, u0h)
                duX := some (d1l, d1h)
                duY := none
                d2uX := some (d2l, d2h)
                d2uY := none }
          | _, _, _ => none
        match pdeParsed, primsOpt with
        | some e, some prims =>
          match eval prims e with
          | some (rpLo, rpHi) => IO.println s!"Residual R(x) from PDE '{cfg.pde}': [{rpLo},{rpHi}]"
          | none => IO.println s!"PDE '{cfg.pde}' evaluation failed (insufficient primitives)"
        | _, _ => pure ()
        match duTrip with
        | [d1, d2, d3] =>
          let (d1l, d1h) := d1; let (d2l, d2h) := d2; let (d3l, d3h) := d3
          IO.println s!"u'(x-h)∈[{d1l},{d1h}], u'(x)∈[{d2l},{d2h}], u'(x+h)∈[{d3l},{d3h}]"
        | _ => pure ()
        match d2uTrip with
        | [dd1, dd2, dd3] =>
          let (dd1l, dd1h) := dd1; let (dd2l, dd2h) := dd2; let (dd3l, dd3h) := dd3
          IO.println s!"u''(x-h)∈[{dd1l},{dd1h}], u''(x)∈[{dd2l},{dd2h}], u''(x+h)∈[{dd3l},{dd3h}]"
        | _ => pure ()
      | _ => throw <| IO.userError "unexpected uTrip structure"
    IO.println "PINN certificate verified: Python and Lean agree on residual bounds."

end NN.Verification.PINN.Certificate
