/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.Architecture
public import NN.Verification.Util.Json
public import NN.MLTheory.CROWN.Graph.Engine.Derivatives
public import NN.Tensor.Constructors
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# PINN Core

PINN helper library: reference graphs, seeding, derivatives, and certificate parsing.

This module is shared by the PINN verification workflows. It provides:
- a dimension-parameterized CROWN graph for a tanh MLP,
- deterministic parameters and typed input-box seeding,
- a few interval/finite-difference residual helpers,
- JSON parsing for the certificate schema used by the surrounding examples.

Run the curated entrypoints instead of importing this file directly:
- `lake exe verify -- pinn-cert [NN/Examples/Verification/PINN/pinn_cert.json]`
- `lake exe verify -- pinn-dataset-check --dataset=PATH.json [--weights=WEIGHTS.json]`

References:
- PINNs (physics-informed neural nets): `https://arxiv.org/abs/1711.10561`
- CROWN (linear bound propagation): `https://arxiv.org/abs/1811.00866`
- IBP (interval bound propagation): `https://arxiv.org/abs/1810.12715`
-/

@[expose] public section


namespace NN.Verification.PINN

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open Spec TorchLean
open TorchLean.Tensor
open Lean
open Json

/-- Parse an object-valued JSON field. -/
def parseObjectField (ctx key : String) (j : Json) :
    Except String (Std.TreeMap.Raw String Json compare) := do
  TorchLean.Json.expectObject s!"{ctx}.{key}" (← TorchLean.Json.expectField ctx key j)

/-- Parse an array-valued JSON field. -/
def parseArrayField (ctx key : String) (j : Json) : Except String (Array Json) := do
  TorchLean.Json.expectArray s!"{ctx}.{key}" (← TorchLean.Json.expectField ctx key j)

/-- Parse a finite-float-array JSON field. -/
def parseFiniteFloatArrayField (ctx key : String) (j : Json) :
    Except String (Array Float) := do
  NN.Verification.Json.parseFiniteFloatArray s!"{ctx}.{key}"
    (← TorchLean.Json.expectField ctx key j)

/-- A closed interval with finite floating-point endpoints. -/
structure FloatInterval where
  /-- Lower endpoint. -/
  lower : Float
  /-- Upper endpoint. -/
  upper : Float
  deriving Repr

/-- Parse an interval object `{ "lo": ..., "hi": ... }`. -/
def parseInterval (ctx : String) (j : Json) : Except String FloatInterval := do
  let lower ← NN.Verification.Json.parseFieldFiniteFloat ctx "lo" j
  let upper ← NN.Verification.Json.parseFieldFiniteFloat ctx "hi" j
  if lower ≤ upper then
    pure { lower, upper }
  else
    throw s!"{ctx}: lower endpoint {lower} exceeds upper endpoint {upper}"

/-- Parse parallel lower/upper arrays into checked intervals. -/
def parseIntervals (ctx : String) (j : Json) (expected : Nat) :
    Except String (Array FloatInterval) := do
  let lowerEndpoints ← parseFiniteFloatArrayField ctx "lo" j
  let upperEndpoints ← parseFiniteFloatArrayField ctx "hi" j
  if h : lowerEndpoints.size = expected ∧ upperEndpoints.size = expected then
    let intervals := Array.ofFn (fun (i : Fin expected) =>
      have hLower : i.val < lowerEndpoints.size := by
        rw [h.1]
        exact i.isLt
      have hUpper : i.val < upperEndpoints.size := by
        rw [h.2]
        exact i.isLt
      { lower := lowerEndpoints[i.val]'hLower
        upper := upperEndpoints[i.val]'hUpper })
    intervals.mapIdxM fun i interval =>
      if interval.lower ≤ interval.upper then
        pure interval
      else
        throw (s!"{ctx}[{i}]: lower endpoint {interval.lower} " ++
          s!"exceeds upper endpoint {interval.upper}")
  else
    throw s!"{ctx}: length mismatch (expected {expected})"

/-- One finite-difference sample from a PINN certificate. -/
structure SolutionBounds where
  /-- Coordinate at the center of the finite-difference stencil. -/
  point : Float
  /-- Solution enclosure at `point - spacing`. -/
  previous : FloatInterval
  /-- Solution enclosure at `point`. -/
  center : FloatInterval
  /-- Solution enclosure at `point + spacing`. -/
  next : FloatInterval
  deriving Repr

/-- Parse one finite-difference `u_bounds` entry. -/
def parseSolutionBounds (ctx : String) (j : Json) :
    Except String SolutionBounds := do
  let x ← NN.Verification.Json.parseFieldFiniteFloat ctx "x" j
  let previous ← parseInterval s!"{ctx}.u_minus" (← TorchLean.Json.expectField ctx "u_minus" j)
  let center ← parseInterval s!"{ctx}.u" (← TorchLean.Json.expectField ctx "u" j)
  let next ← parseInterval s!"{ctx}.u_plus" (← TorchLean.Json.expectField ctx "u_plus" j)
  pure { point := x, previous, center, next }

/-- Configuration parsed from a PINN certificate JSON. -/
structure CertificateConfig where
  /-- PDE identifier carried by the certificate. -/
  pde : String
  /-- Grid spacing used by the exported finite-difference residual. -/
  spacing : Float
  /-- Input perturbation radius for interval checking. -/
  radius : Float
  /-- Number of sample points encoded in `points`. -/
  pointCount : Nat
  /--
  Sample points as a length-`pointCount` 1D tensor.

  PyTorch analogue: this is the `torch.Tensor` you would keep in memory after loading a JSON/CSV
  list of sample coordinates.
  -/
  points : TorchLean.Tensor Float [pointCount]

/-- A parsed and validated PINN certificate artifact. -/
structure Certificate where
  /-- Model and sampling configuration. -/
  config : CertificateConfig
  /-- Finite-difference residual enclosures at each sample point. -/
  residualBounds : Array FloatInterval
  /-- Derivative residual enclosures at each sample point. -/
  derivativeResidualBounds : Array FloatInterval
  /-- Solution enclosures for each three-point finite-difference stencil. -/
  solutionBounds : Array SolutionBounds

/--
Tolerance for comparing a Lean-recomputed PINN bound against the decimal an external exporter
printed. It is looser than the `1e-6` the leaf-artifact checker uses because these bounds come out
of a finite-difference stencil, so the exported decimal carries more accumulated rounding than a
single subtraction does. The comparison itself is `NN.Verification.Util.approxEq`; there is one
implementation of it in the verification layer and this is only the constant it gets called with.
-/
def certTol : Float := 1e-5

/-- The reference scalar-output PINN architecture at an arbitrary input dimension. -/
def referenceArch (inputDim : Nat) : SequentialPINNArch :=
  { inputDim := inputDim
    hiddenDims := #[16, 16]
    outputDim := 1
    activation := .tanh }

/-- Build the reference PINN graph at the requested input dimension. -/
def buildReferenceGraph (inputDim : Nat) : Graph :=
  (referenceArch inputDim).buildGraph

/--
Deterministic reference parameters at an arbitrary input dimension.

These values mirror the bundled exporter. They are demonstration parameters, not trained weights;
production checks should load the exported state through `PINN.PyTorch`.
-/
def referenceParams {α : Type} [TorchLean.Storage α] [Context α] (inputDim : Nat) : ParamStore α :=
  let firstWeight : Tensor α [16, inputDim] :=
    Tensor.dim fun i => Tensor.dim fun j =>
      let base := ((i.val + 1 : Nat) : α) * ((1 / 2) * (1 / 10))
      Tensor.scalar <| if j.val = 0 then base * 2 else base
  let eight : α := 4 * 2
  let firstBias : Tensor α [16] :=
    Tensor.dim fun i =>
      Tensor.scalar <| (1 / 2) * (1 / 10) * ((i.val : Nat) : α) -
        (1 / 2) * (1 / 10) * eight
  let middleWeight : Tensor α [16, 16] :=
    Tensor.dim fun i => Tensor.dim fun j =>
      Tensor.scalar <| if i = j then 1 else (1 / 2) * (1 / 10)
  let middleBias : Tensor α [16] := Tensor.dim fun _ => Tensor.scalar 0
  let hundredth : α := (1 / 10) * (1 / 10)
  let outputWeight : Tensor α [1, 16] :=
    Tensor.dim fun _ => Tensor.dim fun j =>
      Tensor.scalar <| (1 / 10) + hundredth * ((j.val : Nat) : α)
  let outputBias : Tensor α [1] := Tensor.dim fun _ => Tensor.scalar 0
  let params : ParamStore α := {}
  let params :=
    { params with
      linearWB := params.linearWB.insert 1
        { m := 16, n := inputDim, w := firstWeight, b := firstBias } }
  let params :=
    { params with
      linearWB := params.linearWB.insert 3
        { m := 16, n := 16, w := middleWeight, b := middleBias } }
  { params with
    linearWB := params.linearWB.insert 5
      { m := 1, n := 16, w := outputWeight, b := outputBias } }

/-- Seed an $\ell_\infty$ input box centered at a typed input tensor. -/
def seedInput {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α] {inputDim : Nat}
    (ps : ParamStore α) (center : Tensor α [inputDim]) (eps : α) : ParamStore α :=
  ps.seedLInfBall 0 center eps

/--
Enclose the first and second directional derivatives of the output along one input axis.

The two come back together on purpose. The second derivative is propagated on top of the first, so
splitting them into separate entry points would repeat the same sweep. The interval boxes arrive as
a parameter instead of being recomputed here, which is what lets a two-dimensional model pay for
the IBP pass once and then ask about `x` and `y` in turn. A `none` component means the derivative
propagator produced no box at the output node, which happens when the graph contains an operator it
does not cover.
-/
def axisDerivativeBounds (g : Graph) (ps : ParamStore Float)
    (ibp : Array (Option (FlatBox Float))) (inDim : Nat) (axis : Fin inDim) :
    Option FloatInterval × Option FloatInterval :=
  let outId := SequentialPINNArch.graphOutputId g
  let direction := FlatBox.ofTensor (TorchLean.Tensor.oneHot (α := Float) inDim axis)
  let first := NN.MLTheory.CROWN.Graph.runDirectionalDerivative (α := Float) g ps ibp direction
  let second := NN.MLTheory.CROWN.Graph.runScalarSecondDerivative (α := Float) g ps ibp first
  let intervalAt (boxes : Array (Option (FlatBox Float))) : Option FloatInterval :=
    match NN.MLTheory.CROWN.Graph.outputBox? boxes outId with
    | .ok box =>
        some
          { lower := TorchLean.Tensor.sumSpec box.lo
            upper := TorchLean.Tensor.sumSpec box.hi }
    | .error _ => none
  (intervalAt first, intervalAt second)

/-- Parse the JSON certificate consumed by the PINN verification CLI. -/
def parseCertificate (j : Json) : Except String Certificate := do
  let _ ← TorchLean.Json.expectObject "PINN certificate" j
  let po ← parseObjectField "PINN certificate" "pinn" j
  let pdeStr ←
    match Std.TreeMap.Raw.get? po "pde" with
    | none => pure "u''(x) = 0"
    | some Json.null => pure "u''(x) = 0"
    | some pdeJ =>
        match pdeJ with
        | .str s => pure s
        | _ => throw "PINN certificate.pinn.pde: expected string"
  let spacing ← NN.Verification.Json.parseFieldFiniteFloat "PINN certificate.pinn" "h" (.obj po)
  let radius ← NN.Verification.Json.parseFieldFiniteFloat "PINN certificate.pinn" "eps" (.obj po)
  unless spacing > 0.0 do
    throw s!"PINN certificate.pinn.h: expected a positive spacing, got {spacing}"
  unless radius ≥ 0.0 do
    throw s!"PINN certificate.pinn.eps: expected a nonnegative radius, got {radius}"
  let pointValues ← parseFiniteFloatArrayField "PINN certificate.pinn" "points" (.obj po)
  let pointCount := pointValues.size
  unless pointCount > 0 do
    throw "PINN certificate.pinn.points: expected at least one sample point"
  let points : TorchLean.Tensor Float [pointCount] := TorchLean.Tensor.from pointValues
  let rb ← TorchLean.Json.expectField "PINN certificate" "residual_bounds" j
  let residualBounds ← parseIntervals "PINN certificate.residual_bounds" rb pointCount
  let derivJ ← TorchLean.Json.expectField "PINN certificate" "residual_bounds_deriv" j
  let derivativeResidualBounds ←
    parseIntervals "PINN certificate.residual_bounds_deriv" derivJ pointCount
  let solutionJson ← parseArrayField "PINN certificate" "u_bounds" j
  let solutionBounds ← solutionJson.mapIdxM fun i entry =>
    parseSolutionBounds s!"PINN certificate.u_bounds[{i}]" entry
  unless solutionBounds.size = pointCount do
    throw
      s!"PINN certificate.u_bounds: expected {pointCount} entries, got {solutionBounds.size}"
  pure
    { config := { pde := pdeStr, spacing, radius, pointCount, points }
      residualBounds
      derivativeResidualBounds
      solutionBounds }

/-- Finite-difference residual bounds for 1D second derivative. -/
def finiteDifferenceResidual (previous center next : FloatInterval)
    (spacing : Float) : FloatInterval :=
  let lowerNumerator := next.lower - 2.0 * center.upper + previous.lower
  let upperNumerator := next.upper - 2.0 * center.lower + previous.upper
  let scale := 1 / (spacing * spacing)
  { lower := lowerNumerator * scale
    upper := upperNumerator * scale }

end NN.Verification.PINN
