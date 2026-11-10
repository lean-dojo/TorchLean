/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json
public import TorchLean.NN.Arithmetic
public import TorchLeanExamples.Verification.ODE.Parse
public import TorchLean.NN.CLI.Parser
public import TorchLean.Verify.CROWN.Op.Imported
public import TorchLeanExamples.Verification.PINN.PyTorch.Model
public import TorchLean.NN.IR.Tangent

/-!
# Verify

ODE enclosure verification with neural-network sub- and super-solutions, an executable checking
loop inspired by arXiv:2601.19818. Candidate functions `u₋` and `u₊` that satisfy

$$
u_-(t_0)\leq u_0\leq u_+(t_0),\qquad
u_-(t)\leq u_+(t),
$$

and

$$
u_-'(t)\leq f(t,u_-(t)),\qquad
u_+'(t)\geq f(t,u_+(t)),
$$

enclose a solution in $[u_-,u_+]$ by the classical comparison theorem. The real theorem in
`TorchLeanExamples.Verification.ODE.Enclosure` assumes clamped dynamics together with continuity
and derivative hypotheses. This executable checker tests the corresponding interval inequalities
on outward-rounded endpoints, with each network's tangent graph in place of its time derivative. No
theorem connects these computations to the real theorem: that needs interval soundness, agreement
with the real corridor functions, and a theorem relating the tangent graph to the derivative.

Numbers in the certificate JSON and in the weight files are read as their nearest binary64 values,
and the checks hold for those values. Under the `ieee` arithmetic the weights are rounded again, to
the nearest binary32 value. A decimal constant in the right-hand side expression is read as its
roundings toward $-\infty$ and $+\infty$, which enclose it.

The checking pipeline:
- import the lower and upper corridor networks from PyTorch JSON weights as graphs of registered
  operators,
- append their tangent graphs along `dt = 1` (`TorchLean.IR.Graph.tangent`),
- bound `u(t)` and its tangent, read as `u'(t)`, with IBP on time boxes,
- evaluate the ODE right-hand side on those boxes in interval arithmetic,
- split the time domain recursively until every box verifies or a depth or width limit is reached.
-/

section


open FloatLib.Floats (ExecFloat Binary32)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean.Examples.ODE

open CROWN
open PINN
open TorchLean.Tensor
open Lean
open Lean.Json

/-!
Time intervals `[lo, hi]` are FloatLib's endpoint pairs `FloatLib.Numerics.Interval Float`. They
drive only the recursive partitioning of the time domain. The bounds on network outputs are
`α × α` pairs in the checking arithmetic (`Bounds`).
-/

/-- Time-interval width $\mathrm{hi}-\mathrm{lo}$. -/
@[inline] def width (I : FloatLib.Numerics.Interval Float) : Float := I.hi - I.lo
/-- Midpoint without adding same-sign endpoints or subtracting opposite-sign endpoints.
Both operations can overflow even when both endpoints are finite. -/
@[inline] def center (I : FloatLib.Numerics.Interval Float) : Float :=
  if I.lo < 0.0 && 0.0 < I.hi then I.lo * 0.5 + I.hi * 0.5
  else I.lo + (I.hi - I.lo) * 0.5

/-- Split a time interval into two halves at its center. -/
def split (I : FloatLib.Numerics.Interval Float) :
    FloatLib.Numerics.Interval Float × FloatLib.Numerics.Interval Float :=
  let m := center I
  ({ lo := I.lo, hi := m }, { lo := m, hi := I.hi })

/--
Bounds for a corridor candidate on a time interval.

`u` and `du` are lower/upper bounds for the corridor network output and its tangent along
`dt = 1`, which the checker reads as the time derivative.
-/
structure Bounds (α : Type) where
  /-- Interval bounds for the corridor value `u(t)`. -/
  u  : α × α
  /-- Interval bounds for the tangent output, read as the corridor derivative `du/dt`. -/
  du : α × α
  deriving Repr

/--
An imported corridor network and its tangent graph, as one graph of registered operators.

`graph` reads the time at graph input `0`, and no theorem relates its tangent output to a
derivative. Each certificate segment pairs a lower and an upper corridor model. The name avoids
clashing with the opened FloatLib `Model`, which is the IEEE format model used by `toBinary32`.
-/
public structure CorridorModel (α : Type) [Storage α] [Context α] where
  /-- The network and its tangent graph: `u(t)` at output `0`, its tangent along `dt = 1` at
  output `1`. -/
  graph : IR.Graph
  /-- The constants of `graph`. -/
  payload : IR.Payload α
  /-- Input dimension, which `boundsOn` requires to be 1 (scalar time). -/
  inDim : Nat

/-- The corridor model of a one-output network graph `G` with payload `P`. A constant `1` of size
`inDim` is added to `G` and seeds its tangent graph at graph input `0`. -/
private def CorridorModel.ofGraph {α : Type} [Storage α] [Context α] (G : IR.Graph)
    (P : IR.Payload α) (inDim : Nat) : IO (CorridorModel α) := do
  unless G.outputs.size = 1 do
    throw <| IO.userError s!"ODE corridor graph: expected one output, got {G.outputs.size}"
  let (b, P, one) :=
    (⟨G.nodes⟩ : IR.Graph.Builder).constant P ⟨.value [inDim], Tensor.full [inDim] 1⟩
  let G₀ : IR.Graph := { G with nodes := b.nodes }
  let seed (j : Nat) : Option IR.Ref :=
    match G₀.nodes[j]? with
    | some (.input 0 _) => some one
    | _ => none
  match G₀.tangent P seed with
  | .ok (g, P, _) => pure { graph := g, payload := P, inDim := inDim }
  | .error e => throw <| IO.userError s!"ODE derivative graph: {e}"

/-- The time interval `I` as a box at the time input, its endpoints converted outward, without a
center/radius round trip. -/
private def timeBox {α : Type} [Storage α] [Context α] [FromFloat α]
    (I : FloatLib.Numerics.Interval Float) : FlatBox α :=
  { dim := 1
    lo := Tensor.full [1] (ofFloat I.lo (mode := .down))
    hi := Tensor.full [1] (ofFloat I.hi (mode := .up)) }

/--
Bounds on `u(t)` and `du/dt` over the time interval `I`, from IBP on the corridor model's graph
started at `timeBox I`. The model must have input dimension 1.
-/
private def boundsOn {α : Type} [Storage α] [Context α] [BoundOps α]
    [FromFloat α] (m : CorridorModel α) (I : FloatLib.Numerics.Interval Float) :
    IO (Bounds α) := do
  if m.inDim ≠ 1 then
    throw <| IO.userError s!"ODE verifier expects inputDim=1, got {m.inDim}"
  let ibp := Op.runIBP m.graph m.payload (Op.inputSeeds m.graph [timeBox (α := α) I])
  let outB ←
    match Op.outputBox? m.graph ibp 0 with
    | .ok outB => pure outB
    | .error msg => throw <| IO.userError s!"IBP failed at output: {msg}"
  let uLo := Tensor.sum outB.lo
  let uHi := Tensor.sum outB.hi
  let dB ←
    match Op.outputBox? m.graph ibp 1 with
    | .ok dB => pure dB
    | .error msg => throw <| IO.userError s!"IBP failed at derivative output: {msg}"
  let duLo := Tensor.sum dB.lo
  let duHi := Tensor.sum dB.hi
  pure { u := (uLo, uHi), du := (duLo, duHi) }

/-- How to load the corridor network weights. -/
public inductive ModelBackend where
  /-- Build the graph with the PyTorch-import graph builder. -/
  | direct
  /-- Rebuild the network as a TorchLean program and lower it with TorchLean's lowering. -/
  | torchlean
  deriving DecidableEq, Repr

/-- Pretty-print the backend choice for CLI logs. -/
instance : ToString ModelBackend :=
  ⟨fun b => match b with | .direct => "direct" | .torchlean => "torchlean"⟩

/-- Parse the model backend name used by CLI flags and certificate settings. -/
private def ModelBackend.parse? (s : String) : Option ModelBackend :=
  if s = "torchlean" then some .torchlean
  else if s = "direct" then some .direct
  else none

/-- Parse a model backend name and report a CLI-friendly error on failure. -/
private def ModelBackend.parse (s : String) : Except String ModelBackend :=
  match parse? s with
  | some backend => pure backend
  | none => throw s!"--model: expected direct or torchlean; got `{s}`"

/-- Load PINN weights exported from PyTorch (JSON state dict). -/
private def loadCheckpoint (path : String) : IO PINN.Checkpoint := do
  let j ← Json.readFile path
  match Checkpoint.ofJson? j with
  | none => throw <| IO.userError s!"Could not load MLP weights from {path}"
  | some sd => pure sd

/--
Load a corridor model with the direct graph builder: build the checkpoint's graph with
`Checkpoint.model`, then append its tangent graph with `CorridorModel.ofGraph`. The network must
have one output.
-/
private def loadModelDirectWith {α : Type} [Storage α] [Context α] (ofFloat : Float → α)
    (path : String) : IO (CorridorModel α) := do
  let sd ← loadCheckpoint path
  if sd.arch.outputDim ≠ 1 then
    throw <| IO.userError
      s!"ODE verifier expects scalar outputDim=1, got {sd.arch.outputDim} in {path}"
  let (g, P) ← IO.ofExcept (Checkpoint.model ofFloat sd)
  CorridorModel.ofGraph g P sd.arch.inputDim

/--
An MLP as a chain of linear layers, indexed by its input and output dimensions.

`loadModelTorchLean` rebuilds imported PINN weights in this form before lowering them.
-/
private inductive LayerChain : Nat → Nat → Type where
  | last {inDim outDim : Nat} : Linear Float inDim outDim → LayerChain inDim outDim
  | cons {inDim hidDim outDim : Nat} : Linear Float inDim hidDim → LayerChain hidDim outDim →
    LayerChain inDim outDim

/-- Build a `LayerChain` from an array of linear layers, rejecting an empty array or a dimension
mismatch. -/
private def chainOfLayers (layers : Array (SomeLinear Float)) :
    Except String (Σ inDim outDim, LayerChain inDim outDim) :=
  match layers.back? with
  | none => .error "empty MLP (no layers)"
  | some lastLayer =>
    layers.pop.foldrM (init := ⟨lastLayer.inDim, lastLayer.outDim, .last lastLayer.spec⟩)
      fun (l : SomeLinear Float) ⟨inTail, outTail, tail⟩ =>
        if h : l.outDim = inTail then
          let tail' : LayerChain l.outDim outTail := by
            cases h
            simpa using tail
          .ok ⟨l.inDim, outTail, .cons l.spec tail'⟩
        else
          .error s!"layer dim mismatch: {l.outDim} ≠ {inTail}"

/--
The imported layer chain as an execution-polymorphic TorchLean program: alternate linear layers
with the given hidden activation. The activation is chosen once per program, so every hidden layer
uses the same nonlinearity, matching `Architecture`.
-/
private def layerChainProgram {inDim outDim : Nat} (activation : HiddenActivation)
    (chain : LayerChain inDim outDim) :
    Program Float [Shape.dim inDim .scalar] (Shape.dim outDim .scalar) :=
  fun {m} [Monad m] [Ops (m := m) (α := Float)] =>
    fun x =>
      let rec evalChain
          {inD outD : Nat}
          (ch : LayerChain inD outD)
          (x : Ops.Ref (m := m) (α := Float) (.dim inD .scalar)) :
          m (Ops.Ref (m := m) (α := Float) (.dim outD .scalar)) := do
        match ch with
        | .last l =>
          let wR ← Ops.const (m := m) (α := Float)
            (s := .dim outD (.dim inD .scalar)) l.weights
          let bR ← Ops.const (m := m) (α := Float)
            (s := .dim outD .scalar) l.bias
          Ops.apply (m := m) (α := Float)
              (.linear (lead := []) (inDim := inD) (outDim := outD)) (wR, bR, x, ())
        | .cons l tail =>
          let wR ← Ops.const (m := m) (α := Float)
            (s := .dim _ (.dim _ .scalar)) l.weights
          let bR ← Ops.const (m := m) (α := Float)
            (s := .dim _ .scalar) l.bias
          let z ← Ops.apply (m := m) (α := Float)
              (.linear (lead := []) (inDim := inD) (outDim := _)) (wR, bR, x, ())
          let a ←
            match activation with
            | .tanh =>
              Ops.apply (m := m) (α := Float)
                (.tanh (s := .dim _ .scalar)) (z, ())
            | .relu =>
              Ops.apply (m := m) (α := Float)
                (.relu (s := .dim _ .scalar)) (z, ())
            | .sin =>
              Ops.apply (m := m) (α := Float)
                (.sin (s := .dim _ .scalar)) (z, ())
          evalChain tail a
      evalChain chain x

/--
Load a corridor model through the TorchLean lowering pipeline.

The imported weights are rebuilt as a small TorchLean program, lowered to the verifier IR and
imported as a graph of registered operators, and `CorridorModel.ofGraph` appends the tangent graph.
It accepts only ReLU and tanh networks, so `sin` networks need the `direct` backend.
-/
private def loadModelTorchLean (path : String) : IO (CorridorModel Float) := do
  let sd ← loadCheckpoint path
  match chainOfLayers sd.layers with
  | .error e => throw <| IO.userError s!"Bad layer chain in {path}: {e}"
  | .ok ⟨inDim, outDim, chain⟩ =>
    if outDim ≠ 1 then
      throw <| IO.userError s!"ODE verifier expects scalar outputDim=1, got {outDim} in {path}"
    if sd.arch.activation = .sin then
      throw <| IO.userError
        ("ODE verifier: --model=torchlean accepts ReLU/Tanh activations; " ++
          "use --model=direct for sin.")
    let xShape : Shape := .dim inDim .scalar
    let yShape : Shape := .dim outDim .scalar
    let lowered ←
      match IR.lowerForward
            (α := Float) (paramShapes := []) (inShape := xShape) (outShape := yShape)
            (layerChainProgram sd.arch.activation chain) (.nil) with
      | .ok c => pure c
      | .error e => throw <| IO.userError e
    let imported := lowered.toImported
    CorridorModel.ofGraph imported.graph imported.payload inDim

/-- Load a corridor model in the default native arithmetic (`Float`), choosing the backend path. -/
public def loadModelFloat (backend : ModelBackend) (path : String) :
    IO (CorridorModel Float) := do
  match backend with
  | .direct => loadModelDirectWith (α := Float) (fun x => x) path
  | .torchlean => loadModelTorchLean path

/-- Load a corridor model in a non-`Float` arithmetic, which only the `direct` backend supports. -/
private def loadModelNonFloat {α : Type} [Storage α] [Context α] (ofFloat : Float → α)
    (backend : ModelBackend) (path : String) : IO (CorridorModel α) := do
  match backend with
  | .direct => loadModelDirectWith (α := α) ofFloat path
  | .torchlean =>
    throw <| IO.userError
      "ODE verifier: --model=torchlean is only supported with --arithmetic=native"

/-- Arithmetic used while checking an ODE certificate. -/
inductive Arithmetic where
  /-- Host `Float` with the one-ulp outward `instBoundOpsFloat` and `instNonlinearBoundOpsFloat`,
  which bound `sin` and `cos` by $[-1,1]$ and have no `exp` or `log`. -/
  | native
  /-- `ExecFloat.Binary 8 23` with FloatLib's directed operations, on the corridor networks with
  weights rounded to the nearest binary32 value (direct import only). -/
  | ieee
  deriving DecidableEq, Repr

/-- Pretty-print an ODE checker arithmetic choice for logs and certificate settings. -/
instance : ToString Arithmetic :=
  ⟨fun arithmetic => match arithmetic with | .native => "native" | .ieee => "ieee"⟩

/-- Parse an arithmetic name as used in CLI flags and certificate settings. -/
private def Arithmetic.parse? (s : String) : Option Arithmetic :=
  if s = "native" then some .native
  else if s = "ieee" then some .ieee
  else none

/-- Parse an arithmetic name and report a CLI-friendly error on failure. -/
private def Arithmetic.parse (s : String) : Except String Arithmetic :=
  match parse? s with
  | some arithmetic => pure arithmetic
  | none => throw s!"--arithmetic: expected native or ieee; got `{s}`"

/--
Verification settings for domain splitting, slack, logging, model backend and arithmetic.

A certificate JSON supplies them in its `settings` object, and the CLI flags `--model` and
`--arithmetic` override the last two.
-/
structure Settings where
  /-- Maximum recursion depth for time-domain splitting. -/
  maxDepth : Nat := 18
  /-- Boxes of width at most this are not split again, so a failure there fails the segment. -/
  minWidth : Float := 1e-3
  /-- Relaxation added to the right-hand side of each derivative and order comparison, a user
  choice with no proved rounding bound behind it. -/
  slack : Float := 0.0
  /-- Print intermediate bounds and decisions for debugging. -/
  verbose : Bool := false
  /-- How to import the corridor networks. -/
  modelBackend : ModelBackend := .direct
  /-- Which arithmetic semantics to use for the checks. -/
  arithmetic : Arithmetic := .native
  deriving Repr

/--
One time segment of a certificate: a time interval `t`, an initial value interval at `t.lo` and
weight files for the lower and upper corridor networks.
-/
structure Segment where
  /-- Time interval to verify. -/
  t : FloatLib.Numerics.Interval Float
  /-- Initial value interval at `t.lo`. -/
  init : Float × Float
  /-- JSON weights path for the lower-corridor network `u₋`. -/
  lowerWeights : String
  /-- JSON weights path for the upper-corridor network `u₊`. -/
  upperWeights : String
  deriving Repr

/--
Parsed certificate: ODE RHS expression, segments, and optional settings. Each segment is checked as
its own initial-value problem, and nothing requires a segment to start where the previous one ends.
-/
structure Certificate where
  /-- ODE RHS expression (string parsed by `parseExpr`). -/
  rhs : String
  /-- Time segments to check. -/
  segments : Array Segment
  /-- Optional settings record (defaults apply when omitted). -/
  settings : Settings := {}
  deriving Repr

/-- Require finite, nonnegative `minWidth` and `slack`. Other values could disable subdivision or
make comparisons non-finite. -/
@[noinline] private def validateSettings (config : Settings) : Except String Unit := do
  unless config.minWidth.isFinite do
    throw "settings.minWidth must be finite"
  if config.minWidth < 0.0 then
    throw "settings.minWidth must be nonnegative"
  unless config.slack.isFinite do
    throw "settings.slack must be finite"
  if config.slack < 0.0 then
    throw "settings.slack must be nonnegative"

/-- Validate the ordered, finite scalar data carried by one certificate segment. -/
@[noinline] private def validateSegment (seg : Segment) : Except String Unit := do
  unless seg.t.lo.isFinite && seg.t.hi.isFinite do
    throw "segment time interval must have finite endpoints"
  if seg.t.hi < seg.t.lo then
    throw "segment: t1 < t0"
  let (initLo, initHi) := seg.init
  unless initLo.isFinite && initHi.isFinite do
    throw "segment initial interval must have finite endpoints"
  if initHi < initLo then
    throw "segment.init: hi < lo"

/-- Parse an interval object `{ lo: ..., hi: ... }` from JSON. -/
private def parseIntervalObj (j : Json) : Except String (FloatLib.Numerics.Interval Float) := do
  let o ← Json.object "interval" j
  let lo ← Json.parseFiniteFloat "interval.lo" (← Json.field "interval" "lo" o)
  let hi ← Json.parseFiniteFloat "interval.hi" (← Json.field "interval" "hi" o)
  if hi < lo then throw "interval: hi < lo" else
  pure { lo := lo, hi := hi }

/-- Parse the optional `settings` object in the certificate JSON. Omitted fields take the
`Settings` defaults. -/
private def parseSettings (j : Json) : Except String Settings := do
  let defaults : Settings := {}
  match j with
  | .obj o =>
    let maxDepth ←
      match o.get? "maxDepth" with
      | some value => Json.expectNat "settings.maxDepth" value
      | none => pure defaults.maxDepth
    let minWidth ←
      match o.get? "minWidth" with
      | some value => Json.parseFiniteFloat "settings.minWidth" value
      | none => pure defaults.minWidth
    let slack ←
      match o.get? "slack" with
      | some value => Json.parseFiniteFloat "settings.slack" value
      | none => pure defaults.slack
    let verbose ←
      match o.get? "verbose" with
      | some (.bool b) => pure b
      | some _ => throw "settings.verbose: expected boolean"
      | none => pure defaults.verbose
    let modelBackend ←
      match o.get? "modelBackend" with
      | some (.str name) =>
          match ModelBackend.parse? name with
          | some backend => pure backend
          | none => throw s!"settings.modelBackend: expected direct or torchlean; got `{name}`"
      | some _ => throw "settings.modelBackend: expected string"
      | none => pure defaults.modelBackend
    let arithmetic ←
      match o.get? "arithmetic" with
      | some (.str name) =>
          match Arithmetic.parse? name with
          | some arithmetic => pure arithmetic
          | none => throw s!"settings.arithmetic: expected native or ieee; got `{name}`"
      | some _ => throw "settings.arithmetic: expected string"
      | none => pure defaults.arithmetic
    let config : Settings :=
      { maxDepth := maxDepth
        minWidth := minWidth
        slack := slack
        verbose := verbose
        modelBackend := modelBackend
        arithmetic := arithmetic }
    validateSettings config
    pure config
  | .null => pure defaults
  | _ => throw "settings: expected object"

/-- Parse a single segment object from the certificate JSON. -/
private def parseSegment (j : Json) : Except String Segment := do
  let o ← Json.object "segment" j
  let t0 ← Json.parseFiniteFloat "segment.t0" (← Json.field "segment" "t0" o)
  let t1 ← Json.parseFiniteFloat "segment.t1" (← Json.field "segment" "t1" o)
  let initJ ← Json.field "segment" "init" o
  let init ←
    match initJ with
    | .obj _ =>
        let I ← parseIntervalObj initJ
        pure (I.lo, I.hi)
    | .num _ =>
        let x ← Json.parseFiniteFloat "segment.init" initJ
        pure (x, x)
    | _ => throw "segment.init must be number or {lo,hi}"
  let lw ← Json.string "segment.lowerWeights"
    (← Json.field "segment" "lowerWeights" o)
  let uw ← Json.string "segment.upperWeights"
    (← Json.field "segment" "upperWeights" o)
  let seg := { t := { lo := t0, hi := t1 }, init := init,
               lowerWeights := lw, upperWeights := uw }
  validateSegment seg
  pure seg

/--
Parse the top-level certificate JSON object into an `Certificate`.

Parsing fails closed on an empty segment array, non-finite or unordered interval data, negative or
non-finite numerical settings, and unrecognized model-backend or arithmetic tags.
-/
def parseCertificate (j : Json) : Except String Certificate := do
  let o ← Json.object "ode certificate" j
  let rhs ← Json.string "ode certificate.rhs" (← Json.field "ode certificate" "rhs" o)
  let segArr ← Json.array "ode certificate.segments" <|
    ← Json.field "ode certificate" "segments" o
  let segs ← segArr.mapM parseSegment
  if segs.isEmpty then
    throw "ode certificate.segments must contain at least one segment"
  let settings ←
    match Std.TreeMap.Raw.get? o "settings" with
    | some settingsJ => parseSettings settingsJ
    | none => parseSettings Json.null
  pure { rhs := rhs, segments := segs, settings := settings }

/-- Show a closed interval pair as `(lo, hi)`. -/
private def showPair {α : Type} [ToString α] (p : α × α) : String :=
  s!"({p.1}, {p.2})"

/-- Show a time interval as `[lo,hi]`. -/
private def showInterval (I : FloatLib.Numerics.Interval Float) : String :=
  s!"[{I.lo},{I.hi}]"

/--
Subsolution check: whether the upper bound `du.2` on `du/dt` is at most the lower bound `f.1` on
`f(t, u)` plus `slack`, with the sum rounded down by `BoundOps.addDown`.
-/
def checkSub {α : Type} [Storage α] [Context α] [BoundOps α] (du : α × α) (f : α × α)
    (slack : α) : Bool :=
  let duHi := du.2
  let fLo := f.1
  Ival.leBool duHi (BoundOps.addDown fLo slack)

/--
Supersolution check: whether the upper bound `f.2` on `f(t, u)` is at most the lower bound `du.1`
on `du/dt` plus `slack`, with the sum rounded down by `BoundOps.addDown`.
-/
def checkSuper {α : Type} [Storage α] [Context α] [BoundOps α] (du : α × α) (f : α × α)
    (slack : α) : Bool :=
  let duLo := du.1
  let fHi := f.2
  Ival.leBool fHi (BoundOps.addDown duLo slack)

/-- Order check: whether the upper bound `uL.2` on `u₋` is at most the lower bound `uU.1` on `u₊`
plus `slack`, with the sum rounded down by `BoundOps.addDown`. -/
def checkOrder {α : Type} [Storage α] [Context α] [BoundOps α] (uL uU : α × α)
    (slack : α := 0) : Bool :=
  Ival.leBool uL.2 (BoundOps.addDown uU.1 slack)

/--
Verify a single time interval by bounding `u₋, u₊, du₋, du₊` and checking the corridor inequalities.

Returns `(ok, msg)` where `msg` is a short debug string explaining the first failing check.
-/
private def verifyInterval {α : Type} [Storage α] [Context α] [BoundOps α] [ToString α]
    [FromFloat α] [NonlinearBoundOps α]
    (rhs : Expr) (mL mU : CorridorModel α)
    (I : FloatLib.Numerics.Interval Float)
    (config : Settings) : IO (Bool × String) := do
  let bL ← boundsOn (α := α) mL I
  let bU ← boundsOn (α := α) mU I
  let slackA : α := ofFloat config.slack (mode := .down)
  if ¬checkOrder (α := α) bL.u bU.u slackA then
    return (false,
      s!"order failed on {showInterval I}: uL∈{showPair bL.u} not ≤ uU∈{showPair bU.u}")
  let tBox := (ofFloat (α := α) I.lo (mode := .down),
    ofFloat (α := α) I.hi (mode := .up))
  let envL : ODE.Env α := { t := tBox, u := bL.u }
  let envU : ODE.Env α := { t := tBox, u := bU.u }
  let some fL := ODE.Expr.eval (α := α) envL rhs
    | return (false, s!"RHS eval failed on {showInterval I} (sub)")
  let some fU := ODE.Expr.eval (α := α) envU rhs
    | return (false, s!"RHS eval failed on {showInterval I} (super)")
  if ¬checkSub (α := α) bL.du fL slackA then
    return (false, s!"sub inequality failed on {showInterval I}: duL∈{showPair bL.du}, " ++
      s!"f(t,uL)∈{showPair fL}")
  if ¬checkSuper (α := α) bU.du fU slackA then
    return (false, s!"super inequality failed on {showInterval I}: duU∈{showPair bU.du}, " ++
      s!"f(t,uU)∈{showPair fU}")
  return (true, "")

/--
Check `I` with `verifyInterval` and, when that fails, split `I` at its center and check both halves
with `depth - 1`.

The search fails at the first failing box that has depth `0`, width at most `minWidth`, or no
representable split point strictly inside it.
-/
private def verifySegmentRecursive {α : Type} [Storage α] [Context α] [BoundOps α]
    [ToString α] [FromFloat α] [NonlinearBoundOps α]
    (rhs : Expr) (mL mU : CorridorModel α)
    (I : FloatLib.Numerics.Interval Float)
    (config : Settings) (depth : Nat) : IO Bool := do
  let (ok, msg) ← verifyInterval (α := α) rhs mL mU I config
  if ok then
    if config.verbose then
      IO.println s!"[ODE] OK {showInterval I}"
    pure true
  else
    if _ : depth = 0 ∨ width I ≤ config.minWidth then
      let reason := if depth = 0 then "maximum subdivision depth reached"
        else "minimum time width reached"
      IO.eprintln s!"[ODE] FAIL {msg}; {reason}"
      pure false
    else
      let (I1, I2) := split I
      if !(I.lo < I1.hi && I1.hi < I.hi) then
        IO.eprintln s!"[ODE] FAIL {msg}; no representable interior time split remains"
        return false
      let ok1 ← verifySegmentRecursive (α := α) rhs mL mU I1 config (depth - 1)
      if ok1 then
        verifySegmentRecursive (α := α) rhs mL mU I2 config (depth - 1)
      else
        pure false
termination_by depth

/--
Verify one certificate segment. After loading both corridor networks, it checks that `seg.init`
lies between the corridors at `t0 = seg.t.lo` and that they are ordered there, then verifies
`seg.t` with `verifySegmentRecursive` from depth `maxDepth`.
-/
private def verifySegmentWith {α : Type} [Storage α] [Context α] [BoundOps α]
    [ToString α] [FromFloat α] [NonlinearBoundOps α]
    (loadModel : ModelBackend → String → IO (CorridorModel α))
    (rhs : Expr) (seg : Segment) (config : Settings) : IO Bool := do
  if config.verbose then
    IO.println <|
      s!"[ODE] loading models ({config.modelBackend}, arithmetic={config.arithmetic}): " ++
        s!"lower={seg.lowerWeights}, upper={seg.upperWeights}"
  let mL ← loadModel config.modelBackend seg.lowerWeights
  let mU ← loadModel config.modelBackend seg.upperWeights
  let slackA : α := ofFloat config.slack (mode := .down)
  let t0I : FloatLib.Numerics.Interval Float := { lo := seg.t.lo, hi := seg.t.lo }
  let bL0 ← boundsOn (α := α) mL t0I
  let bU0 ← boundsOn (α := α) mU t0I
  let (iLoF, iHiF) := seg.init
  let iLo : α := ofFloat iLoF (mode := .down)
  let iHi : α := ofFloat iHiF (mode := .up)
  if !Ival.leBool bL0.u.2 iLo then
    IO.eprintln s!"[ODE] FAIL initial: uL(t0)∈{showPair bL0.u} not ≤ init.lo={iLoF}"
    return false
  if !Ival.leBool iHi bU0.u.1 then
    IO.eprintln s!"[ODE] FAIL initial: uU(t0)∈{showPair bU0.u} not ≥ init.hi={iHiF}"
    return false
  if ¬checkOrder (α := α) bL0.u bU0.u slackA then
    IO.eprintln s!"[ODE] FAIL initial order: uL(t0)∈{showPair bL0.u} not ≤ uU(t0)∈{showPair bU0.u}"
    return false
  IO.println s!"[ODE] initial OK at t0={seg.t.lo}"
  verifySegmentRecursive (α := α) rhs mL mU seg.t config config.maxDepth

/-- Round a host `Float` to the binary32 value used by the IEEE reference arithmetic. -/
private def toBinary32 (x : Float) : Binary32 :=
  ExecFloat.Binary.ofModel
    (Model.cast FloatFormat.binary64 FloatFormat.binary32
      (ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat x : FloatLib.Floats.Binary64)))

/-- Verify every segment under the configured arithmetic, returning whether all succeed. A failing
segment does not stop the loop, so every failure gets reported. -/
private def verifySegments (rhs : Expr) (segments : Array Segment)
    (config : Settings) : IO Bool := do
  let mut allOk := true
  for seg in segments do
    let ok ←
      match config.arithmetic with
      | .native =>
          verifySegmentWith (α := Float) loadModelFloat rhs seg config
      | .ieee =>
          verifySegmentWith (α := Binary32)
            (loadModelNonFloat (α := Binary32) toBinary32) rhs seg config
    if ¬ok then allOk := false
  pure allOk

/--
Read, validate and verify the certificate file at `path`, as run by
`scripts/lake.sh exe verify -- ode --cert=...`. A backend or arithmetic given on the command line
replaces the certificate's setting, and a failed verification throws an `IO.userError`.
-/
def runCertificate (path : String) (backendOverride : Option ModelBackend)
    (arithmeticOverride : Option Arithmetic) : IO Unit := do
  let j ← Json.readFile path
  let cert ←
    match parseCertificate j with
    | .ok c => pure c
    | .error msg => throw <| IO.userError s!"Bad ODE cert JSON: {msg}"
  let rhsAst ←
    match parseExpr cert.rhs with
    | .ok e => pure e
    | .error msg => throw <| IO.userError s!"RHS parse error: {msg}"
  let config :=
    match backendOverride with
    | some mb => { cert.settings with modelBackend := mb }
    | none => cert.settings
  let config :=
    match arithmeticOverride with
    | some arithmetic => { config with arithmetic := arithmetic }
    | none => config
  match validateSettings config with
  | .error msg => throw <| IO.userError s!"Invalid ODE verifier settings: {msg}"
  | .ok () => pure ()
  for seg in cert.segments do
    match validateSegment seg with
    | .error msg => throw <| IO.userError s!"Invalid ODE certificate segment: {msg}"
    | .ok () => pure ()
  if ← verifySegments rhsAst cert.segments config then
    IO.println "[ODE] certificate verified: all segments succeeded."
  else
    throw <| IO.userError "[ODE] certificate verification failed."

/--
Parse CLI arguments and verify either the certificate file named by `--cert=...` or one segment
given inline by `--rhs`, `--t0`, `--t1`, `--init`, `--lower` and `--upper`. The `--model` and
`--arithmetic` flags apply in both modes.
-/
def runArgs (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let (backendName?, args) ← IO.ofExcept <|
    CLI.takeFlagValue? args "model"
  let backendOverride ←
    match backendName? with
    | none => pure none
    | some name =>
        match ModelBackend.parse name with
        | .ok backend => pure (some backend)
        | .error message => throw <| IO.userError message
  let (arithmeticName?, args) ← IO.ofExcept <|
    CLI.takeFlagValue? args "arithmetic"
  let arithmeticOverride ←
    match arithmeticName? with
    | none => pure none
    | some name =>
        match Arithmetic.parse name with
        | .ok arithmetic => pure (some arithmetic)
        | .error message => throw <| IO.userError message
  let (cert?, args) ← IO.ofExcept <| CLI.takeFlagValue? args "cert"
  match cert? with
  | some p => do
      IO.ofExcept (CLI.checkNoArgs args)
      runCertificate p backendOverride arithmeticOverride
  | none =>
    let (rhsS, args) ← IO.ofExcept <|
      CLI.requireFlagValue args "rhs" (some "missing --rhs=<expr>")
    let (t0, args) ← IO.ofExcept <|
      CLI.requireFloatFlag args "t0" (some "missing --t0=<float>")
    let (t1, args) ← IO.ofExcept <|
      CLI.requireFloatFlag args "t1" (some "missing --t1=<float>")
    let (initF, args) ← IO.ofExcept <|
      CLI.requireFloatFlag args "init" (some "missing --init=<float>")
    let (lw, args) ← IO.ofExcept <|
      CLI.requireFlagValue args "lower" (some "missing --lower=<weights.json>")
    let (uw, args) ← IO.ofExcept <|
      CLI.requireFlagValue args "upper" (some "missing --upper=<weights.json>")
    let defaults : Settings := {}
    let (maxDepth, args) ← IO.ofExcept <|
      CLI.takeNatFlag args "maxDepth" (default := defaults.maxDepth)
    let (minWidth, args) ← IO.ofExcept <|
      CLI.takeFloatFlag args "minWidth" (default := defaults.minWidth)
    let (slack, args) ← IO.ofExcept <|
      CLI.takeFloatFlag args "slack" (default := defaults.slack)
    let (verbose, args) ← IO.ofExcept <|
      CLI.takeBoolValueFlag args "verbose" (default := defaults.verbose)
    IO.ofExcept (CLI.checkNoArgs args)
    let init := (initF, initF)
    let rhsAst ←
      match parseExpr rhsS with
      | .ok e => pure e
      | .error msg => throw <| IO.userError s!"RHS parse error: {msg}"
    let config : Settings :=
      { maxDepth := maxDepth
        minWidth := minWidth
        slack := slack
        verbose := verbose
        modelBackend := backendOverride.getD defaults.modelBackend
        arithmetic := arithmeticOverride.getD defaults.arithmetic }
    let seg : Segment :=
      { t := { lo := t0, hi := t1 }, init := init, lowerWeights := lw, upperWeights := uw }
    match validateSettings config with
    | .error msg => throw <| IO.userError s!"Invalid ODE verifier settings: {msg}"
    | .ok () => pure ()
    match validateSegment seg with
    | .error msg => throw <| IO.userError s!"Invalid ODE certificate segment: {msg}"
    | .ok () => pure ()
    if ← verifySegments rhsAst #[seg] config then
      IO.println "[ODE] verification succeeded."
    else
      throw <| IO.userError "[ODE] verification failed."

/--
Entry point of the `ode` tool of `scripts/lake.sh exe verify`.

With no arguments, `--help` or `-h` it prints usage, and otherwise it passes the arguments to
`runArgs`.
-/
public def main (args : List String) : IO Unit := do
  let args :=
    match args with
    | "--" :: rest => rest
    | _ => args
  if args = [] ∨ args = ["--help"] ∨ args = ["-h"] then
    IO.println
      ("Usage:\n" ++
       ("  lake exe verify -- ode [--model=direct|torchlean] " ++
         "[--arithmetic=native|ieee] --cert=<cert.json>\n") ++
       ("  lake exe verify -- ode [--model=direct|torchlean] " ++
         "[--arithmetic=native|ieee] --rhs=\"<expr>\" --t0=<float> " ++
         "--t1=<float> --init=<float> --lower=<wL.json> --upper=<wU.json>\n") ++
       "Options:\n" ++
       "  --maxDepth=<nat>   (default 18)\n" ++
       "  --minWidth=<nonnegative finite float> (default 1e-3)\n" ++
       "  --slack=<nonnegative finite float>    (default 0)\n" ++
       "  --arithmetic=native|ieee (artifact setting unless overridden; inline default native)\n" ++
       "  --verbose=true|false\n")
  else
    runArgs args

end TorchLean.Examples.ODE
