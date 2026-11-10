/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN.Architecture
public import TorchLean.NN.Json
public import TorchLean.NN.IR.Tangent
public import TorchLean.Verify.CROWN.Op.Exec
public import TorchLean.Tensor.Constructors
public import TorchLean.Tensor.Literal

/-!
# PINN Core

Helpers shared by the PINN verification workflows:
- a dimension-parameterized tanh MLP as a graph of registered operators with deterministic
  parameters,
- the graph's first and second tangents along a direction input (`Derivatives.ofGraph`, from
  `TorchLean.IR.Graph.tangent`) and their interval bounds,
- interval and finite-difference residual helpers,
- JSON parsing for the certificate schema of the bundled examples.

The checks read the tangents as directional derivatives, and no theorem relates a tangent graph to
a derivative.

Users run the workflows through the curated entrypoints:
- `scripts/lake.sh exe verify -- pinn-cert [TorchLeanExamples/Verification/PINN/pinn_cert.json]`
- `scripts/lake.sh exe verify -- pinn-dataset-check --dataset=PATH.json [--weights=WEIGHTS.json]`

References:
- PINNs (physics-informed neural nets): `https://arxiv.org/abs/1711.10561`
- CROWN (linear bound propagation): `https://arxiv.org/abs/1811.00866`
- IBP (interval bound propagation): `https://arxiv.org/abs/1810.12715`
-/

@[expose] public section


namespace TorchLean.Examples.PINN

open CROWN
open TorchLean.Tensor
open Lean
open Lean.Json

/-- Parse an interval object `{ "lo": ..., "hi": ... }` with finite, ordered endpoints. -/
def parseInterval (ctx : String) (j : Json) :
    Except String (FloatLib.Numerics.Interval Float) := do
  let o ← Json.object ctx j
  let lower ← Json.parseFiniteFloat s!"{ctx}.lo" (← Json.field ctx "lo" o)
  let upper ← Json.parseFiniteFloat s!"{ctx}.hi" (← Json.field ctx "hi" o)
  if lower ≤ upper then
    pure { lo := lower, hi := upper }
  else
    throw s!"{ctx}: lower endpoint {lower} exceeds upper endpoint {upper}"

/-- Parse the parallel finite arrays `lo` and `hi`, each of length `expected`, into ordered
intervals. -/
def parseIntervals (ctx : String) (j : Json) (expected : Nat) :
    Except String (Array (FloatLib.Numerics.Interval Float)) := do
  let o ← Json.object ctx j
  let lowerEndpoints ←
    Json.parseFiniteFloatArray s!"{ctx}.lo" (← Json.field ctx "lo" o)
  let upperEndpoints ←
    Json.parseFiniteFloatArray s!"{ctx}.hi" (← Json.field ctx "hi" o)
  if h : lowerEndpoints.size = expected ∧ upperEndpoints.size = expected then
    let intervals := Array.ofFn (fun (i : Fin expected) =>
      have hLower : i.val < lowerEndpoints.size := by
        rw [h.1]
        exact i.isLt
      have hUpper : i.val < upperEndpoints.size := by
        rw [h.2]
        exact i.isLt
      { lo := lowerEndpoints[i.val]'hLower
        hi := upperEndpoints[i.val]'hUpper })
    intervals.mapIdxM fun i interval =>
      if interval.lo ≤ interval.hi then
        pure interval
      else
        throw (s!"{ctx}[{i}]: lower endpoint {interval.lo} " ++
          s!"exceeds upper endpoint {interval.hi}")
  else
    throw s!"{ctx}: length mismatch (expected {expected})"

/-- One finite-difference sample from a PINN certificate. -/
structure SolutionBounds where
  /-- Coordinate at the center of the finite-difference stencil. -/
  point : Float
  /-- Solution enclosure at `point - spacing`. -/
  previous : FloatLib.Numerics.Interval Float
  /-- Solution enclosure at `point`. -/
  center : FloatLib.Numerics.Interval Float
  /-- Solution enclosure at `point + spacing`. -/
  next : FloatLib.Numerics.Interval Float
  deriving Repr

/-- Parse one finite-difference `u_bounds` entry. -/
def parseSolutionBounds (ctx : String) (j : Json) :
    Except String SolutionBounds := do
  let o ← Json.object ctx j
  let x ← Json.parseFiniteFloat s!"{ctx}.x" (← Json.field ctx "x" o)
  let previous ← parseInterval s!"{ctx}.u_minus" (← Json.field ctx "u_minus" o)
  let center ← parseInterval s!"{ctx}.u" (← Json.field ctx "u" o)
  let next ← parseInterval s!"{ctx}.u_plus" (← Json.field ctx "u_plus" o)
  pure { point := x, previous, center, next }

/-- Configuration parsed from a PINN certificate JSON. -/
structure CertificateConfig where
  /-- The PDE residual expression, parsed with `parseExpr` during replay. -/
  pde : String
  /-- Grid spacing used by the exported finite-difference residual. -/
  spacing : Float
  /-- Input perturbation radius for interval checking. -/
  radius : Float
  /-- Number of sample points encoded in `points`. -/
  pointCount : Nat
  /--
  Sample points as a length-`pointCount` 1D tensor, the analogue of a `torch.Tensor` of sample
  coordinates loaded from a JSON or CSV list.
  -/
  points : Tensor Float [pointCount]

/-- A parsed and validated PINN certificate artifact. -/
structure Certificate where
  /-- Model and sampling configuration. -/
  config : CertificateConfig
  /-- Finite-difference residual enclosures at each sample point. -/
  residualBounds : Array (FloatLib.Numerics.Interval Float)
  /-- Residual enclosures at each sample point from the PDE expression on derivative bounds. -/
  derivativeResidualBounds : Array (FloatLib.Numerics.Interval Float)
  /-- Solution enclosures for each three-point finite-difference stencil. -/
  solutionBounds : Array SolutionBounds

/--
Absolute tolerance for PINN artifact replay comparisons through `TorchLean.Numeric.approxEq`.
It is a comparison policy, and no proof relates it to rounding or finite-difference truncation
error.
-/
def certTol : Float := 1e-5

/-- The reference scalar-output PINN architecture at an arbitrary input dimension. -/
def referenceArch (inputDim : Nat) : Architecture :=
  { inputDim := inputDim
    hiddenDims := #[16, 16]
    outputDim := 1
    activation := .tanh }

/--
Deterministic reference layers at an arbitrary input dimension.

The values mirror the bundled exporter and serve as demonstration parameters. Production checks
should load trained weights through `Examples.PINN.Checkpoint.ofJson?`.
-/
def referenceLayers {α : Type} [Storage α] [Context α] (inputDim : Nat) :
    Array (SomeLinear α) :=
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
  #[{ inDim := inputDim, outDim := 16, spec := { weights := firstWeight, bias := firstBias } },
    { inDim := 16, outDim := 16, spec := { weights := middleWeight, bias := middleBias } },
    { inDim := 16, outDim := 1, spec := { weights := outputWeight, bias := outputBias } }]

/-- The reference PINN at the requested input dimension as a graph of registered operators, with
the reference layers as constants. -/
def referenceModel {α : Type} [Storage α] [Context α] (inputDim : Nat) :
    Except String (IR.Graph × IR.Payload α) :=
  (referenceArch inputDim).model (referenceLayers inputDim)

/--
Read the box of graph output `i` of a propagation pass as a scalar interval.

Scalar-output PINN graphs have a length-one output vector, so summing its entries is the exact
scalar readout. A missing box is reported with `ctx` as the message prefix, so callers can name the
failing pass.
-/
def outputInterval (ctx : String) (G : IR.Graph) (boxes : Array (Option (FlatBox Float)))
    (i : Nat) : Except String (FloatLib.Numerics.Interval Float) :=
  match Op.outputBox? G boxes i with
  | .ok box =>
      .ok { lo := Tensor.sum box.lo, hi := Tensor.sum box.hi }
  | .error msg => .error s!"{ctx}: {msg}"

/-- A PINN graph with its tangents along a direction: `graph` reads the PINN's input at graph
input `0` and a direction of the same size at graph input `1`. Graph output `0` is the PINN's
value, `1` its tangent along the direction and, when `hasSecond`, `3` the tangent of output `1`.
-/
structure Derivatives (α : Type) [Storage α] where
  /-- The PINN and its tangents. -/
  graph : IR.Graph
  /-- Its constants. -/
  payload : IR.Payload α
  /-- Whether `graph` has output `3`, the second tangent. -/
  hasSecond : Bool

/--
Extend the PINN graph `G` with payload `P` by graph input `1`, a direction of the shape of input
`0`, and take its tangent along that input (`IR.Graph.tangent`). The tangent of that graph,
seeded with the first tangents of `G`'s nodes, adds the second tangents, and when it fails (at an
operator without a tangent rule, say) the first tangent graph is kept with `hasSecond := false`.
-/
def Derivatives.ofGraph {α : Type} [Storage α] [Context α] (G : IR.Graph)
    (P : IR.Payload α) : Except String (Derivatives α) := do
  let some (.input _ p) := G.nodes.find? (· matches .input 0 _)
    | throw "PINN graph has no input 0"
  let direction : IR.Ref := ⟨G.nodes.size, 0⟩
  let G₀ : IR.Graph := { G with nodes := G.nodes.push (.input 1 p) }
  let seed (j : Nat) : Option IR.Ref :=
    match G₀.nodes[j]? with
    | some (.input 0 _) => some direction
    | _ => none
  let (G₁, P₁, first) ← G₀.tangent P seed
  match G₁.tangent P₁ fun j => if j < G₀.nodes.size then (first[j]?).join else none with
  | .ok (G₂, P₂, _) => pure { graph := G₂, payload := P₂, hasSecond := true }
  | .error _ => pure { graph := G₁, payload := P₁, hasSecond := false }

/--
Enclose graph outputs `1` and `3` of `D`, the first and second tangents along input axis `axis`,
with one interval pass over `D.graph` from the input box `xB`. A `none` component means the pass
produced no box at that output, or `D` has no second tangent.
-/
def axisDerivativeBounds (D : Derivatives Float) (xB : FlatBox Float) (inDim : Nat)
    (axis : Fin inDim) :
    Option (FloatLib.Numerics.Interval Float) × Option (FloatLib.Numerics.Interval Float) :=
  let direction := FlatBox.ofTensor (Tensor.oneHot (α := Float) inDim axis)
  let boxes := Op.runIBP D.graph D.payload (Op.inputSeeds D.graph [xB, direction])
  let intervalAt (i : Nat) : Option (FloatLib.Numerics.Interval Float) :=
    (outputInterval "derivative" D.graph boxes i).toOption
  (intervalAt 1, if D.hasSecond then intervalAt 3 else none)

/-- Parse a PINN certificate JSON, requiring a PDE string, a positive spacing `h`, a nonnegative
radius `eps`, at least one point and bound arrays with one entry per point. -/
def parseCertificate (j : Json) : Except String Certificate := do
  let o ← Json.object "PINN certificate" j
  let po ← Json.object "PINN certificate.pinn"
    (← Json.field "PINN certificate" "pinn" o)
  -- The PDE expression is required. A default such as `u''(x) = 0` would use prime and equality
  -- syntax that `parseExpr` rejects, so a missing field could only fail later with a parse error.
  let pdeStr ←
    match Std.TreeMap.Raw.get? po "pde" with
    | some (.str s) => pure s
    | some Json.null | none => throw "PINN certificate.pinn.pde: missing PDE expression string"
    | some _ => throw "PINN certificate.pinn.pde: expected string"
  let spacing ← Json.parseFiniteFloat "PINN certificate.pinn.h"
    (← Json.field "PINN certificate.pinn" "h" po)
  let radius ← Json.parseFiniteFloat "PINN certificate.pinn.eps"
    (← Json.field "PINN certificate.pinn" "eps" po)
  unless spacing > 0.0 do
    throw s!"PINN certificate.pinn.h: expected a positive spacing, got {spacing}"
  unless radius ≥ 0.0 do
    throw s!"PINN certificate.pinn.eps: expected a nonnegative radius, got {radius}"
  let pointValues ← Json.parseFiniteFloatArray "PINN certificate.pinn.points"
    (← Json.field "PINN certificate.pinn" "points" po)
  let pointCount := pointValues.size
  unless pointCount > 0 do
    throw "PINN certificate.pinn.points: expected at least one sample point"
  let points : Tensor Float [pointCount] := Tensor.from pointValues
  let rb ← Json.field "PINN certificate" "residual_bounds" o
  let residualBounds ← parseIntervals "PINN certificate.residual_bounds" rb pointCount
  let derivJ ← Json.field "PINN certificate" "residual_bounds_deriv" o
  let derivativeResidualBounds ←
    parseIntervals "PINN certificate.residual_bounds_deriv" derivJ pointCount
  let solutionJson ← Json.array "PINN certificate.u_bounds"
    (← Json.field "PINN certificate" "u_bounds" o)
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

/-- Bounds on the central second difference $(u(x+h)-2u(x)+u(x-h))/h^2$, with `spacing` as $h$,
from intervals for the three stencil values. -/
def finiteDifferenceResidual (previous center next : FloatLib.Numerics.Interval Float)
    (spacing : Float) : FloatLib.Numerics.Interval Float :=
  let lowerNumerator := next.lo - 2.0 * center.hi + previous.lo
  let upperNumerator := next.hi - 2.0 * center.lo + previous.hi
  let scale := 1 / (spacing * spacing)
  { lo := lowerNumerator * scale
    hi := upperNumerator * scale }

end TorchLean.Examples.PINN
