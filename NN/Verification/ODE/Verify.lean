/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json
public import NN.Verification.ODE.Parse
public import NN.API.CLI.Command
public import NN.MLTheory.CROWN.Extras.BoundOpsIEEE32Exec
public import NN.Verification.Builtin.Lowering.API
public import NN.Verification.PINN.PyTorch.ParamStore

/-!
# Verify

ODE enclosure verification via NN sub- and super-solutions.

This module implements the core executable checking loop inspired by arXiv:2601.19818:

Given candidate functions `u₋, u₊` with

$$
u_-(t_0)\leq u_0\leq u_+(t_0),\qquad
u_-(t)\leq u_+(t),
$$

and

$$
u_-'(t)\leq f(t,u_-(t)),\qquad
u_+'(t)\geq f(t,u_+(t)),
$$

these are the corridor inequalities used by the classical comparison theorem for enclosing a
solution inside $[u_-,u_+]$. The real theorem in `NN.Proofs.Verification.ODE.Enclosure` assumes
clamped dynamics and explicit continuity and derivative hypotheses. This executable checker
tests the corresponding interval inequalities; a bridge from these computations to the theorem
still needs interval soundness and agreement with the real corridor functions.

Executable checking pipeline:
- import corridor networks (lower/upper) from PyTorch JSON weights,
- build a derivative graph `d/dt` by structural differentiation of the IR,
- use IBP bounds to bound `u(t)` and `u'(t)` on time boxes,
- evaluate the ODE RHS on those boxes (interval evaluation),
- recursively split the time domain until all boxes verify or we hit a depth/width limit.
-/

section


namespace NN.Verification.ODE.Verify

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.PINN
open NN.Verification.ODE
open NN.Verification.ODE.Parse
open Spec TorchLean
open TorchLean.Tensor
open Lean
open Json

/-!
Simple `Float` intervals used for time partitioning.

These are *not* the interval arithmetic backend; they only control domain splitting.
-/
/--
Simple time interval `[lo, hi]` used for recursive domain splitting.

These intervals are only used to drive partitioning of the time domain; they are not the main
interval-arithmetic representation used for bounding neural network outputs.
-/
structure Interval where
  /-- Interval lower endpoint. -/
  lo : Float
  /-- Interval upper endpoint. -/
  hi : Float
  deriving Repr

/-- Pretty-print intervals as `[lo,hi]`. -/
instance : ToString Interval :=
  ⟨fun I => s!"[{I.lo},{I.hi}]"⟩

namespace Interval

/-- Interval width $\mathrm{hi}-\mathrm{lo}$. -/
@[inline] def width (I : Interval) : Float := I.hi - I.lo
/-- Midpoint without adding same-sign endpoints or subtracting opposite-sign endpoints.
Both operations can overflow even when both endpoints are finite. -/
@[inline] def center (I : Interval) : Float :=
  if I.lo < 0.0 && 0.0 < I.hi then I.lo * 0.5 + I.hi * 0.5
  else I.lo + (I.hi - I.lo) * 0.5
/-- Interval radius $(\mathrm{hi}-\mathrm{lo})/2$. -/
@[inline] def radius (I : Interval) : Float := (I.hi - I.lo) * 0.5

/-- Split an interval into two halves at its center. -/
def split (I : Interval) : Interval × Interval :=
  let m := I.center
  ({ lo := I.lo, hi := m }, { lo := m, hi := I.hi })

end Interval

/-!
Bounds required from bound propagation: the enclosure of `u(t)` and `u'(t)` on a time box.
-/
/--
Bounds for a corridor candidate on a time interval.

`u` and `du` are lower/upper bounds for the corridor network output and its time derivative.
-/
structure Bounds (α : Type) where
  /-- Interval bounds for the corridor value `u(t)`. -/
  u  : α × α
  /-- Interval bounds for the corridor derivative `du/dt`. -/
  du : α × α
  deriving Repr

/-!
An imported corridor model plus its derived-graph.

We store:
- `g`: forward graph,
- `dg`: derivative-augmented graph,
- `baseParams`: parameters and constants,
- `outId` and `dOutId`: output node ids for `u` and `du/dt`,
- `inDim`: expected input dimension (1 for ODE time).
-/
/--
An imported corridor network together with its derived (time-derivative) graph.

This is the core executable artifact we verify against a parsed ODE certificate.
-/
structure Model (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Forward graph computing `u(t)`. -/
  g     : Graph
  /-- Derived graph computing both `u(t)` and `du/dt` (by structural differentiation). -/
  dg    : Graph
  /-- Parameters/constants for the graphs. -/
  baseParams   : ParamStore α
  /-- Output node id of `u(t)` in `g`. -/
  outId : Nat
  /-- Output node id of `du/dt` in `dg`. -/
  dOutId : Nat
  /-- Input dimension (expected to be 1 for scalar time). -/
  inDim : Nat

/-!
Internal state for building derivative graphs.
-/
private structure DerivBuildState (α : Type) [TorchLean.Storage α] [Context α] where
  nodes : Array Node
  ps : ParamStore α
  dId : Array Nat

/-- State monad used during derivative-graph construction. -/
private abbrev DerivBuildM (α : Type) [TorchLean.Storage α] [Context α] :=
  StateT (DerivBuildState α) IO

/-- Build a constant vector value of length `n`, filled with `x`. -/
private def constVecFill {α : Type} [TorchLean.Storage α] [Context α] (n : Nat) (x : α) :
    FlatTensor α :=
  { n := n, v := Tensor.full (α := α) (.dim n .scalar) x }

/-- Insert a constant value payload for a `.const` node id. -/
private def addConstVal {α : Type} [TorchLean.Storage α] [Context α] (ps : ParamStore α) (id : Nat)
    (v : FlatTensor α) : ParamStore α :=
  { ps with constVals := ps.constVals.insert id v }

/-- Insert matmul parameters for a `.matmul` node id. -/
private def addMatmulW {α : Type} [TorchLean.Storage α] [Context α] (ps : ParamStore α) (id : Nat)
    (p : MatParams α) : ParamStore α :=
  { ps with matmulW := ps.matmulW.insert id p }

/--
Build a derivative graph `d/dt` for a 1D-input graph `g`.

This is a structural AD pass over the IR: for each node we build a corresponding derivative node
and store its id in a side table.
-/
private def buildDerivativeGraph1D {α : Type} [TorchLean.Storage α] [Context α]
    (g : Graph) (baseParams : ParamStore α) (outId : Nat) : IO (Graph × ParamStore α × Nat) := do
  if g.nodes.isEmpty then
    throw <| IO.userError "buildDerivativeGraph1D: empty graph"
  match g.nodes[0]? with
  | some node =>
    match node.kind with
    | .input => pure ()
    | _ => throw <| IO.userError "buildDerivativeGraph1D: expected node 0 to be `.input`"
  | none =>
    throw <| IO.userError "buildDerivativeGraph1D: empty graph"

  let pushNode : Array Nat → OpKind → Shape → DerivBuildM α Nat := fun parents kind outShape => do
    let st ← get
    let id := st.nodes.size
    let nodes' := st.nodes.push { id := id, parents := parents, kind := kind, outShape := outShape }
    set { st with nodes := nodes' }
    pure id

  let mkConstFill : Shape → α → DerivBuildM α Nat := fun outShape x => do
    let id ← pushNode #[] (.const outShape) outShape
    modify fun st =>
      { st with
        ps := addConstVal (α := α) st.ps id (constVecFill (α := α) (Spec.Shape.size outShape) x) }
    pure id

  let setDerivativeId : Nat → Nat → DerivBuildM α Unit := fun i did => do
    let st ← get
    if h : i < st.dId.size then
      set { st with dId := st.dId.set i did h }
    else
      throw <| IO.userError s!"buildDerivativeGraph1D: derivative slot out of bounds at node {i}"

  let derivativeId : DerivBuildState α → Nat → Nat → DerivBuildM α Nat :=
    fun st nodeId parentId => do
      match st.dId[parentId]? with
      | some did => pure did
      | none =>
          throw <| IO.userError <|
            s!"buildDerivativeGraph1D: derivative parent {parentId} out of bounds at node {nodeId}"

  let init : DerivBuildState α :=
    { nodes := g.nodes, ps := baseParams, dId := Array.replicate g.nodes.size 0 }

  let (_, st) ← (show DerivBuildM α Unit from do
    for i in [0:g.nodes.size] do
      let node ←
        match g.nodes[i]? with
        | some node => pure node
        | none => throw <| IO.userError s!"buildDerivativeGraph1D: node {i} out of bounds"
      let outShape := node.outShape
      match node.kind with
      | .input =>
          let did ← mkConstFill outShape 1
          setDerivativeId i did
      | .const _ =>
          let did ← mkConstFill outShape 0
          setDerivativeId i did
      | .detach =>
          let did ← mkConstFill outShape 0
          setDerivativeId i did
      | .add =>
          match NN.IR.binaryParents? node.parents with
          | some (p1, p2) =>
              let st ← get
              let d1 ← derivativeId st i p1
              let d2 ← derivativeId st i p2
              let did ← pushNode #[d1, d2] .add outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at add node {i}"
      | .sub =>
          match NN.IR.binaryParents? node.parents with
          | some (p1, p2) =>
              let st ← get
              let d1 ← derivativeId st i p1
              let d2 ← derivativeId st i p2
              let did ← pushNode #[d1, d2] .sub outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sub node {i}"
      | .mulElem =>
          match NN.IR.binaryParents? node.parents with
          | some (p1, p2) =>
              let st ← get
              let d1 ← derivativeId st i p1
              let d2 ← derivativeId st i p2
              let t1 ← pushNode #[d1, p2] .mulElem outShape
              let t2 ← pushNode #[p1, d2] .mulElem outShape
              let did ← pushNode #[t1, t2] .add outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at mul_elem node {i}"
      | .linear =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              match st.ps.linearWB[i]? with
              | none =>
                  throw <| IO.userError <|
                    s!"buildDerivativeGraph1D: missing linearWB params at node {i}"
              | some p =>
                  let d1 ← derivativeId st i p1
                  let did ← pushNode #[d1] .matmul outShape
                  modify fun st =>
                    { st with ps := addMatmulW (α := α) st.ps did { m := p.m, n := p.n, w := p.w } }
                  setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at linear node {i}"
      | .matmul =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              match st.ps.matmulW[i]? with
              | none =>
                  throw <| IO.userError <|
                    s!"buildDerivativeGraph1D: missing matmulW params at node {i}"
              | some p =>
                  let d1 ← derivativeId st i p1
                  let did ← pushNode #[d1] .matmul outShape
                  modify fun st => { st with ps := addMatmulW (α := α) st.ps did p }
                  setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at matmul node {i}"
      | .tanh =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let y2 ← pushNode #[i, i] .mulElem outShape
              let one ← mkConstFill outShape 1
              let fac ← pushNode #[one, y2] .sub outShape
              let did ← pushNode #[fac, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at tanh node {i}"
      | .sigmoid =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let one ← mkConstFill outShape 1
              let oneMy ← pushNode #[one, i] .sub outShape
              let yFac ← pushNode #[i, oneMy] .mulElem outShape
              let did ← pushNode #[yFac, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sigmoid node {i}"
      | .exp =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[i, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at exp node {i}"
      | .log =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let invz ← pushNode #[p1] .inv outShape
              let did ← pushNode #[invz, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at log node {i}"
      | .sin =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let cosz ← pushNode #[p1] .cos outShape
              let did ← pushNode #[cosz, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sin node {i}"
      | .cos =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let sinz ← pushNode #[p1] .sin outShape
              let neg1 ← mkConstFill outShape (-1)
              let negsin ← pushNode #[neg1, sinz] .mulElem outShape
              let did ← pushNode #[negsin, d1] .mulElem outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at cos node {i}"
      | .sum =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] .sum outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sum node {i}"
      | .reshape inS outS =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.reshape inS outS) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reshape node {i}"
      | .flatten s =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.flatten s) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at flatten node {i}"
      | .permute perm =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.permute perm) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at permute node {i}"
      | .broadcastTo s₁ s₂ =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.broadcastTo s₁ s₂) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at broadcastTo node {i}"
      | .reduceSum axis =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.reduceSum axis) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reduce_sum node {i}"
      | .reduceMean axis =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              let st ← get
              let d1 ← derivativeId st i p1
              let did ← pushNode #[d1] (.reduceMean axis) outShape
              setDerivativeId i did
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reduce_mean node {i}"
      | k =>
          throw <| IO.userError
            s!"buildDerivativeGraph1D: unsupported op in ODE derivative graph: {repr k} at node {i}"
  ).run init

  let dg : Graph := { nodes := st.nodes }
  let dOutId ←
    match st.dId[outId]? with
    | some did => pure did
    | none =>
        throw <| IO.userError s!"buildDerivativeGraph1D: output node {outId} out of bounds"
  pure (dg, st.ps, dOutId)

/-- Seed the time endpoints directly, avoiding the rounding introduced by a center/radius
roundtrip. Conversion into the selected arithmetic backend still uses `ofFloat`. -/
private def seedInput1D {α : Type} [TorchLean.Storage α] [Context α]
    (ps : ParamStore α) (ofFloat : Float → α) (I : Interval) : ParamStore α :=
  ps.seedInputBox 0
    { dim := 1
      lo := Tensor.full [1] (ofFloat I.lo)
      hi := Tensor.full [1] (ofFloat I.hi) }

/--
Compute bounds for `u(t)` and `du/dt` on a time interval.

This evaluates IBP over the derivative graph `m.dg` after seeding the converted endpoints of `I`.
-/
private def boundsOn {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]
    (ofFloat : Float → α) (m : Model α) (I : Interval) : IO (Bounds α) := do
  if m.inDim ≠ 1 then
    throw <| IO.userError s!"ODE verifier expects inputDim=1, got {m.inDim}"
  let ps := seedInput1D (α := α) m.baseParams ofFloat I
  let ibp := runIBP (α:=α) m.dg ps
  let outB ←
    match NN.MLTheory.CROWN.Graph.outputBox? ibp m.outId with
    | .ok outB => pure outB
    | .error msg => throw <| IO.userError s!"IBP failed at output: {msg}"
  let uLo := TorchLean.Tensor.sumSpec outB.lo
  let uHi := TorchLean.Tensor.sumSpec outB.hi
  let dB ←
    match NN.MLTheory.CROWN.Graph.outputBox? ibp m.dOutId with
    | .ok dB => pure dB
    | .error msg => throw <| IO.userError s!"IBP failed at derivative output: {msg}"
  let duLo := TorchLean.Tensor.sumSpec dB.lo
  let duHi := TorchLean.Tensor.sumSpec dB.hi
  pure { u := (uLo, uHi), du := (duLo, duHi) }

/--
How to load the corridor network weights.

`direct` uses the PyTorch-import graph builder directly. `torchlean` goes through TorchLean
lowering.
-/
inductive ModelBackend where
  | direct
  | torchlean
  deriving DecidableEq, Repr

/-- Pretty-print the backend choice for CLI logs. -/
instance : ToString ModelBackend :=
  ⟨fun b => match b with | .direct => "direct" | .torchlean => "torchlean"⟩

/-- Parse the model backend name used by CLI flags and certificate settings. -/
private def parseModelBackendName (s : String) : Option ModelBackend :=
  if s = "torchlean" then some .torchlean
  else if s = "direct" then some .direct
  else none

/-- Parse a model backend name and report a CLI-friendly error on failure. -/
private def parseModelBackendNameE (s : String) : Except String ModelBackend :=
  match parseModelBackendName s with
  | some backend => pure backend
  | none => throw s!"--model: expected direct or torchlean; got `{s}`"

/-- Load PINN weights exported from PyTorch (JSON state dict). -/
private def loadPinnState (path : String) : IO Import.PINNPyTorch.PinnState := do
  let j ← NN.Verification.Json.readJsonFile path
  match Import.PINNPyTorch.loadPinnState j with
  | none => throw <| IO.userError s!"Could not load MLP weights from {path}"
  | some sd => pure sd

/--
Load a corridor model using the direct graph builder, and build its derivative graph.

This is the simplest path: import the PyTorch graph, then run `buildDerivativeGraph1D`.
-/
private def loadModelDirectWith {α : Type} [TorchLean.Storage α] [Context α] (ofFloat : Float → α)
    (path : String) : IO (Model α) := do
  let sd ← loadPinnState path
  if sd.arch.outputDim ≠ 1 then
    throw <| IO.userError
      s!"ODE verifier expects scalar outputDim=1, got {sd.arch.outputDim} in {path}"
  let g := Import.PINNPyTorch.buildGraph sd
  let baseParams := Import.PINNPyTorch.toParamStoreWith (α := α) ofFloat sd
  let outId := SequentialPINNArch.graphOutputId g
  let (dg, paramsWithDerivative, dOutId) ← buildDerivativeGraph1D (α := α) g baseParams outId
  pure { g := g, dg := dg, baseParams := paramsWithDerivative, outId := outId, dOutId := dOutId,
         inDim := sd.arch.inputDim }

/-- Linear-layer payload extracted from a PINN export (`w`, `b`). -/
private structure LinLayer (inDim outDim : Nat) where
  w : Tensor Float [outDim, inDim]
  b : Tensor Float [outDim]

/--
Simple representation of an MLP as a chain of linear layers.

This is used when lowering imported PINN weights into the TorchLean lowering pipeline.
-/
private inductive LayerChain : Nat → Nat → Type where
  | last {inDim outDim : Nat} : LinLayer inDim outDim → LayerChain inDim outDim
  | cons {inDim hidDim outDim : Nat} : LinLayer inDim hidDim → LayerChain hidDim outDim → LayerChain
    inDim outDim

/-- Convert one imported PINN layer payload into a `LinLayer`. -/
private def linOfPinn (pl : Import.PINNPyTorch.PinnLayer) : LinLayer pl.inDim pl.outDim :=
  { w := pl.weights, b := pl.bias }

/-- Convert an imported PINN layer list into a `LayerChain` (returns `none` if empty). -/
private def chainOfPinnLayers :
    List Import.PINNPyTorch.PinnLayer →
      Except String (Σ inDim outDim, LayerChain inDim outDim)
  | .nil => .error "empty MLP (no layers)"
  | .cons l .nil => .ok ⟨l.inDim, l.outDim, .last (linOfPinn l)⟩
  | .cons l rest =>
    match chainOfPinnLayers rest with
    | .error e => .error e
    | .ok ⟨inTail, outTail, tail⟩ =>
      if h : l.outDim = inTail then
        let tail' : LayerChain l.outDim outTail := by
          cases h
          simpa using tail
        .ok ⟨l.inDim, outTail, .cons (linOfPinn l) tail'⟩
      else
        .error s!"layer dim mismatch: {l.outDim} ≠ {inTail}"

/--
Load a corridor model via the TorchLean lowering pipeline.

This path reconstructs a small TorchLean program from the imported weights, lowers it to a CROWN
graph, and then builds the derivative graph from that lowered graph.
-/
private def loadModelTorchLean (path : String) : IO (Model Float) := do
  let sd ← loadPinnState path
  match chainOfPinnLayers sd.layers.toList with
  | .error e => throw <| IO.userError s!"Bad layer chain in {path}: {e}"
  | .ok ⟨inDim, outDim, chain⟩ =>
    if outDim ≠ 1 then
      throw <| IO.userError s!"ODE verifier expects scalar outputDim=1, got {outDim} in {path}"
    let xShape : Shape := .dim inDim .scalar
    let yShape : Shape := .dim outDim .scalar

    match sd.arch.activation with
    | .tanh =>
      let model : Runtime.Autograd.Model.Program Float [xShape] yShape :=
        fun {m} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := Float)] =>
          fun x =>
            let rec evalT
                {inD outD : Nat}
                (ch : LayerChain inD outD)
                (x : Runtime.Autograd.Model.RefTy (m := m) (α := Float) (.dim inD .scalar)) :
                m (Runtime.Autograd.Model.RefTy (m := m) (α := Float) (.dim outD .scalar)) := do
              match ch with
              | .last l =>
                let wR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim outD (.dim inD .scalar)) l.w
                let bR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim outD .scalar) l.b
                Runtime.Autograd.Torch.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := outD) wR bR x
              | .cons l tail =>
                let wR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim _ (.dim _ .scalar)) l.w
                let bR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim _ .scalar) l.b
                let z ← Runtime.Autograd.Torch.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := _) wR bR x
                let a ← Runtime.Autograd.Model.tanh (m := m) (α := Float) (s := .dim _ .scalar)
                  z
                evalT tail a
            evalT chain x
      let lowered ←
        match NN.Verification.Builtin.lowerForwardToIR
              (α := Float) (paramShapes := []) (inShape := xShape) (outShape := yShape)
              model (.nil) with
        | .ok c => pure c
        | .error e => throw <| IO.userError e
      let (dg, paramsWithDerivative, dOutId) ←
        buildDerivativeGraph1D (α := Float) lowered.graph lowered.ps lowered.outputId
      pure { g := lowered.graph, dg := dg, baseParams := paramsWithDerivative,
             outId := lowered.outputId, dOutId := dOutId, inDim := inDim }
    | .relu =>
      let model : Runtime.Autograd.Model.Program Float [xShape] yShape :=
        fun {m} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := Float)] =>
          fun x =>
            let rec evalR
                {inD outD : Nat}
                (ch : LayerChain inD outD)
                (x : Runtime.Autograd.Model.RefTy (m := m) (α := Float) (.dim inD .scalar)) :
                m (Runtime.Autograd.Model.RefTy (m := m) (α := Float) (.dim outD .scalar)) := do
              match ch with
              | .last l =>
                let wR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim outD (.dim inD .scalar)) l.w
                let bR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim outD .scalar) l.b
                Runtime.Autograd.Torch.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := outD) wR bR x
              | .cons l tail =>
                let wR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim _ (.dim _ .scalar)) l.w
                let bR ← Runtime.Autograd.Model.const (m := m) (α := Float)
                  (s := .dim _ .scalar) l.b
                let z ← Runtime.Autograd.Torch.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := _) wR bR x
                let a ← Runtime.Autograd.Model.relu (m := m) (α := Float) (s := .dim _ .scalar)
                  z
                evalR tail a
            evalR chain x
      let lowered ←
        match NN.Verification.Builtin.lowerForwardToIR
              (α := Float) (paramShapes := []) (inShape := xShape) (outShape := yShape)
              model (.nil) with
        | .ok c => pure c
        | .error e => throw <| IO.userError e
      let (dg, paramsWithDerivative, dOutId) ←
        buildDerivativeGraph1D (α := Float) lowered.graph lowered.ps lowered.outputId
      pure { g := lowered.graph, dg := dg, baseParams := paramsWithDerivative,
             outId := lowered.outputId, dOutId := dOutId, inDim := inDim }
    | .sin =>
      throw <| IO.userError
        ("ODE verifier: --model=torchlean accepts ReLU/Tanh/Sigmoid activations; " ++
          "use --model=direct for sin.")

/-- Load a corridor model in the default native arithmetic (`Float`), choosing the backend path. -/
private def loadModelFloat (backend : ModelBackend) (path : String) : IO (Model Float) := do
  match backend with
  | .direct => loadModelDirectWith (α := Float) (fun x => x) path
  | .torchlean => loadModelTorchLean path

/-- Load a corridor model in a non-`Float` arithmetic mode (supported in `direct`
  mode). -/
private def loadModelNonFloat {α : Type} [TorchLean.Storage α] [Context α] (ofFloat : Float → α)
    (backend : ModelBackend) (path : String) : IO (Model α) := do
  match backend with
  | .direct => loadModelDirectWith (α := α) ofFloat path
  | .torchlean =>
    throw <| IO.userError
      "ODE verifier: --model=torchlean is only supported with --arithmetic=native"

/--
Arithmetic used while checking an ODE certificate.

`native` is the host `Float` evaluator used by the direct and TorchLean-lowered paths. `ieee`
uses the independent IEEE reference evaluator and is available for direct evaluation.
-/
inductive Arithmetic where
  | native
  | ieee
  deriving DecidableEq, Repr

/-- Pretty-print an ODE checker arithmetic choice for logs and certificate settings. -/
instance : ToString Arithmetic :=
  ⟨fun arithmetic => match arithmetic with | .native => "native" | .ieee => "ieee"⟩

/-- Parse an arithmetic name as used in CLI flags and certificate settings. -/
private def parseArithmeticName (s : String) : Option Arithmetic :=
  if s = "native" then some .native
  else if s = "ieee" then some .ieee
  else none

/-- Parse an arithmetic name and report a CLI-friendly error on failure. -/
private def parseArithmeticNameE (s : String) : Except String Arithmetic :=
  match parseArithmeticName s with
  | some arithmetic => pure arithmetic
  | none => throw s!"--arithmetic: expected native or ieee; got `{s}`"

/--
Verification settings controlling domain splitting and slack.

These come from the `settings` section of a certificate JSON (and can be partially overridden by
  CLI).
-/
structure ODEVerifierSettings where
  /-- Maximum recursion depth for time-domain splitting. -/
  maxDepth : Nat := 18
  /-- Stop splitting once interval width falls below this; unresolved boxes remain undecided. -/
  minWidth : Float := 1e-3
  /-- Explicit relaxation added to corridor comparisons; this is not a proved rounding bound. -/
  slack : Float := 0.0
  /-- Print intermediate bounds and decisions for debugging. -/
  verbose : Bool := false
  /-- How to import the corridor networks. -/
  modelBackend : ModelBackend := .direct
  /-- Which arithmetic semantics to use for the checks. -/
  arithmetic : Arithmetic := .native
  deriving Repr

/--
One time segment of a certificate.

Each segment carries:
- a time interval `t`,
- an initial corridor constraint at `t.lo`,
- and weight files for the lower and upper corridor networks.
-/
structure ODECertificateSegment where
  /-- Time interval to verify. -/
  t : Interval
  /-- Initial value interval at `t.lo`. -/
  init : Float × Float
  /-- JSON weights path for the lower-corridor network `u₋`. -/
  lowerWeights : String
  /-- JSON weights path for the upper-corridor network `u₊`. -/
  upperWeights : String
  deriving Repr

/-- Parsed certificate: ODE RHS expression, segments, and optional settings. -/
structure ODECertificate where
  /-- ODE RHS expression (string parsed by `NN.Verification.ODE.Parse`). -/
  rhs : String
  /-- Time segments to check. -/
  segments : Array ODECertificateSegment
  /-- Optional settings record (defaults apply when omitted). -/
  settings : ODEVerifierSettings := {}
  deriving Repr

/-- Reject settings that could disable subdivision or make comparisons non-finite. -/
@[noinline] private def validateSettings (config : ODEVerifierSettings) : Except String Unit := do
  unless config.minWidth.isFinite do
    throw "settings.minWidth must be finite"
  if config.minWidth < 0.0 then
    throw "settings.minWidth must be nonnegative"
  unless config.slack.isFinite do
    throw "settings.slack must be finite"
  if config.slack < 0.0 then
    throw "settings.slack must be nonnegative"

/-- Validate the ordered, finite scalar data carried by one certificate segment. -/
@[noinline] private def validateSegment (seg : ODECertificateSegment) : Except String Unit := do
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
private def parseIntervalObj (j : Json) : Except String Interval := do
  let lo ← NN.Verification.Json.parseFieldFiniteFloat "interval" "lo" j
  let hi ← NN.Verification.Json.parseFieldFiniteFloat "interval" "hi" j
  if hi < lo then throw "interval: hi < lo" else
  pure { lo := lo, hi := hi }

/-- Parse an interval object as a raw pair `(lo, hi)`. -/
private def parsePairObj (j : Json) : Except String (Float × Float) := do
  let I ← parseIntervalObj j
  pure (I.lo, I.hi)

/-- Parse the optional `settings` object in the certificate JSON. -/
private def parseSettings (j : Json) : Except String ODEVerifierSettings := do
  match j with
  | .obj o =>
    let maxDepth ←
      match o.get? "maxDepth" with
      | some value => TorchLean.Json.expectNat "settings.maxDepth" value
      | none => pure 18
    let minWidth ←
      match o.get? "minWidth" with
      | some value => NN.Verification.Json.parseFiniteFloat "settings.minWidth" value
      | none => pure 1e-3
    let slack ←
      match o.get? "slack" with
      | some value => NN.Verification.Json.parseFiniteFloat "settings.slack" value
      | none => pure 0.0
    let verbose ←
      match o.get? "verbose" with
      | some (.bool b) => pure b
      | some _ => throw "settings.verbose: expected boolean"
      | none => pure false
    let modelBackend ←
      match o.get? "modelBackend" with
      | some (.str name) =>
          match parseModelBackendName name with
          | some backend => pure backend
          | none => throw s!"settings.modelBackend: expected direct or torchlean; got `{name}`"
      | some _ => throw "settings.modelBackend: expected string"
      | none => pure ModelBackend.direct
    let arithmetic ←
      match o.get? "arithmetic" with
      | some (.str name) =>
          match parseArithmeticName name with
          | some arithmetic => pure arithmetic
          | none => throw s!"settings.arithmetic: expected native or ieee; got `{name}`"
      | some _ => throw "settings.arithmetic: expected string"
      | none => pure Arithmetic.native
    let config : ODEVerifierSettings :=
      { maxDepth := maxDepth
        minWidth := minWidth
        slack := slack
        verbose := verbose
        modelBackend := modelBackend
        arithmetic := arithmetic }
    validateSettings config
    pure config
  | .null => pure {}
  | _ => throw "settings: expected object"

/-- Parse a single segment object from the certificate JSON. -/
private def parseSegment (j : Json) : Except String ODECertificateSegment := do
  let _ ← TorchLean.Json.expectObject "segment" j
  let t0 ← NN.Verification.Json.parseFieldFiniteFloat "segment" "t0" j
  let t1 ← NN.Verification.Json.parseFieldFiniteFloat "segment" "t1" j
  let initJ ← TorchLean.Json.expectField "segment" "init" j
  let init ←
    match initJ with
    | .obj _ => parsePairObj initJ
    | .num _ =>
        let x ← NN.Verification.Json.parseFiniteFloat "segment.init" initJ
        pure (x, x)
    | _ => throw "segment.init must be number or {lo,hi}"
  let lw ← NN.Verification.Json.parseFieldString "segment" "lowerWeights" j
  let uw ← NN.Verification.Json.parseFieldString "segment" "upperWeights" j
  let seg := { t := { lo := t0, hi := t1 }, init := init,
               lowerWeights := lw, upperWeights := uw }
  validateSegment seg
  pure seg

/--
Parse the top-level certificate JSON object into an `ODECertificate`.

Parsing fails closed on an empty segment array, non-finite or unordered interval data, negative or
non-finite numerical settings, and unrecognized model-backend or arithmetic tags.
-/
def parseODECertificate (j : Json) : Except String ODECertificate := do
  let o ← TorchLean.Json.expectObject "ode certificate" j
  let rhs ← NN.Verification.Json.parseFieldString "ode certificate" "rhs" j
  let segArr ← TorchLean.Json.expectArray "ode certificate.segments" <|
    ← TorchLean.Json.expectField "ode certificate" "segments" j
  let segs ← segArr.mapM parseSegment
  if segs.isEmpty then
    throw "ode certificate.segments must contain at least one segment"
  let settings ←
    match Std.TreeMap.Raw.get? o "settings" with
    | some settingsJ => parseSettings settingsJ
    | none => parseSettings Json.null
  pure { rhs := rhs, segments := segs, settings := settings }

/-- Boolean `<=` on scalars, rejecting unordered values such as NaN. -/
def leBool {α : Type} [TorchLean.Storage α] [Context α] (x y : α) : Bool :=
  Ival.leBool x y

/-- Show a closed interval pair as `(lo, hi)`. -/
private def showPair {α : Type} [ToString α] (p : α × α) : String :=
  s!"({p.1}, {p.2})"

/--
Subsolution check: ensure `du/dt <= f(t, u)` on the interval (with slack).

We compare the *upper bound* on `du` against the *lower bound* on `f`, allowing `slack` as a
nonnegative numerical tolerance.
-/
def checkSub {α : Type} [TorchLean.Storage α] [Context α] (du : α × α) (f : α × α) (slack : α) :
    Bool :=
  let duHi := du.2
  let fLo := f.1
  leBool duHi (fLo + slack)

/--
Supersolution check: ensure `du/dt >= f(t, u)` on the interval (with slack).

We compare the *lower bound* on `du` against the *upper bound* on `f`, allowing `slack` as a
nonnegative numerical tolerance.
-/
def checkSuper {α : Type} [TorchLean.Storage α] [Context α] (du : α × α) (f : α × α) (slack : α) :
    Bool :=
  let duLo := du.1
  let fHi := f.2
  leBool fHi (duLo + slack)

/-- Order check: ensure the lower corridor stays below the upper corridor (`u₋ <= u₊`), up to
the configured slack. -/
def checkOrder {α : Type} [TorchLean.Storage α] [Context α] (uL uU : α × α)
    (slack : α := 0) : Bool :=
  leBool uL.2 (uU.1 + slack)

/--
Verify a single time interval by bounding `u₋, u₊, du₋, du₊` and checking the corridor inequalities.

Returns `(ok, msg)` where `msg` is a short debug string explaining the first failing check.
-/
private def verifyInterval {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α] [ToString α]
    (ofFloat : Float → α) (rhs : Expr) (mL mU : Model α) (I : Interval)
    (config : ODEVerifierSettings) : IO (Bool × String) := do
  let bL ← boundsOn (α := α) ofFloat mL I
  let bU ← boundsOn (α := α) ofFloat mU I
  let slackA : α := ofFloat config.slack
  if ¬checkOrder (α := α) bL.u bU.u slackA then
    return (false, s!"order failed on {I}: uL∈{showPair bL.u} not ≤ uU∈{showPair bU.u}")
  let tBox := (ofFloat I.lo, ofFloat I.hi)
  let envL : NN.Verification.ODE.Env α := { t := tBox, u := bL.u }
  let envU : NN.Verification.ODE.Env α := { t := tBox, u := bU.u }
  let some fL := NN.Verification.ODE.eval (α := α) ofFloat envL rhs
    | return (false, s!"RHS eval failed on {I} (sub)")
  let some fU := NN.Verification.ODE.eval (α := α) ofFloat envU rhs
    | return (false, s!"RHS eval failed on {I} (super)")
  if ¬checkSub (α := α) bL.du fL slackA then
    return (false, s!"sub inequality failed on {I}: duL∈{showPair bL.du}, f(t,uL)∈{showPair fL}")
  if ¬checkSuper (α := α) bU.du fU slackA then
    return (false, s!"super inequality failed on {I}: duU∈{showPair bU.du}, f(t,uU)∈{showPair fU}")
  return (true, "")

/--
Recursive verifier for a segment interval.

If `verifyInterval` fails on `I`, this splits the interval and recurses until either verification
succeeds everywhere or we hit `(depth = 0)` / the `minWidth` cutoff.
-/
private partial def verifySegmentRecursive {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]
    [ToString α]
    (ofFloat : Float → α) (rhs : Expr) (mL mU : Model α) (I : Interval)
    (config : ODEVerifierSettings) (depth : Nat) : IO Bool := do
  let (ok, msg) ← verifyInterval (α := α) ofFloat rhs mL mU I config
  if ok then
    if config.verbose then
      IO.println s!"[ODE] OK {I}"
    pure true
  else
    if depth = 0 ∨ I.width ≤ config.minWidth then
      let reason := if depth = 0 then "maximum subdivision depth reached"
        else "minimum time width reached"
      IO.eprintln s!"[ODE] FAIL {msg}; {reason}"
      pure false
    else
      let (I1, I2) := I.split
      if !(I.lo < I1.hi && I1.hi < I.hi) then
        IO.eprintln s!"[ODE] FAIL {msg}; no representable interior time split remains"
        return false
      let ok1 ← verifySegmentRecursive (α := α) ofFloat rhs mL mU I1 config (depth - 1)
      if ok1 then
        verifySegmentRecursive (α := α) ofFloat rhs mL mU I2 config (depth - 1)
      else
        pure false

/--
Verify a full certificate segment: check initial conditions at `t0`, then recursively verify the
  time interval.

This loads both corridor networks and then recursively verifies `seg.t`.
-/
  private def verifySegmentWith {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α]
    [ToString α]
    (ofFloat : Float → α) (loadModel : ModelBackend → String → IO (Model α))
    (rhs : Expr) (seg : ODECertificateSegment) (config : ODEVerifierSettings) : IO Bool := do
  if config.verbose then
    IO.println <|
      s!"[ODE] loading models ({config.modelBackend}, arithmetic={config.arithmetic}): " ++
        s!"lower={seg.lowerWeights}, upper={seg.upperWeights}"
  let mL ← loadModel config.modelBackend seg.lowerWeights
  let mU ← loadModel config.modelBackend seg.upperWeights
  let slackA : α := ofFloat config.slack
  let t0I : Interval := { lo := seg.t.lo, hi := seg.t.lo }
  let bL0 ← boundsOn (α := α) ofFloat mL t0I
  let bU0 ← boundsOn (α := α) ofFloat mU t0I
  let (iLoF, iHiF) := seg.init
  let iLo : α := ofFloat iLoF
  let iHi : α := ofFloat iHiF
  if !leBool bL0.u.2 iLo then
    IO.eprintln s!"[ODE] FAIL initial: uL(t0)∈{showPair bL0.u} not ≤ init.lo={iLoF}"
    return false
  if !leBool iHi bU0.u.1 then
    IO.eprintln s!"[ODE] FAIL initial: uU(t0)∈{showPair bU0.u} not ≥ init.hi={iHiF}"
    return false
  if ¬checkOrder (α := α) bL0.u bU0.u slackA then
    IO.eprintln s!"[ODE] FAIL initial order: uL(t0)∈{showPair bL0.u} not ≤ uU(t0)∈{showPair bU0.u}"
    return false
  IO.println s!"[ODE] initial OK at t0={seg.t.lo}"
  verifySegmentRecursive (α := α) ofFloat rhs mL mU seg.t config config.maxDepth

/--
Run verification for a parsed certificate file.

This is the main executable entry used by `lake exe verify -- ode --cert=...`.
-/
def runCertificate (path : String) (backendOverride : Option ModelBackend)
    (arithmeticOverride : Option Arithmetic) : IO Unit := do
  let j ← NN.Verification.Json.readJsonFile path
  let cert ←
    match parseODECertificate j with
    | .ok c => pure c
    | .error msg => throw <| IO.userError s!"Bad ODE cert JSON: {msg}"
  let rhsAst ←
    match Parse.parseExpr cert.rhs with
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
  match config.arithmetic with
  | .native =>
    let mut allOk := true
    for seg in cert.segments do
      let ok ← verifySegmentWith (α := Float) (fun x => x) (fun mb p => loadModelFloat mb p) rhsAst
        seg config
      if ¬ok then allOk := false
    if allOk then
      IO.println "[ODE] certificate verified: all segments succeeded."
    else
      throw <| IO.userError "[ODE] certificate verification failed."
  | .ieee =>
    let αI := TorchLean.Floats.IEEE754.IEEE32Exec
    let ofF := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat
    let mut allOk := true
    for seg in cert.segments do
      let ok ← verifySegmentWith (α := αI) ofF (fun mb p => loadModelNonFloat (α := αI) ofF mb p)
        rhsAst seg config
      if ¬ok then allOk := false
    if allOk then
      IO.println "[ODE] certificate verified: all segments succeeded."
    else
      throw <| IO.userError "[ODE] certificate verification failed."

/--
Parse CLI arguments and either:
- verify a `--cert=...` certificate file, or
- verify a single segment specified inline via `--rhs`, `--t0`, `--t1`, etc.
-/
def runArgs (args : List String) : IO Unit := do
  let args := TorchLean.CLI.dropDashDash args
  let (backendName?, args) ← TorchLean.CLI.orThrowIO <|
    TorchLean.CLI.takeFlagValue? args "model"
  let backendOverride ←
    match backendName? with
    | none => pure none
    | some name =>
        match parseModelBackendNameE name with
        | .ok backend => pure (some backend)
        | .error message => throw <| IO.userError message
  let (arithmeticName?, args) ← TorchLean.CLI.orThrowIO <|
    TorchLean.CLI.takeFlagValue? args "arithmetic"
  let arithmeticOverride ←
    match arithmeticName? with
    | none => pure none
    | some name =>
        match parseArithmeticNameE name with
        | .ok arithmetic => pure (some arithmetic)
        | .error message => throw <| IO.userError message
  let (cert?, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeFlagValue? args "cert"
  match cert? with
  | some p => do
      TorchLean.CLI.orThrowIO (TorchLean.CLI.checkNoArgs args)
      runCertificate p backendOverride arithmeticOverride
  | none =>
    let (rhsS, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFlagValue args "rhs" (some "missing --rhs=<expr>")
    let (t0, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFloatFlag args "t0" (some "missing --t0=<float>")
    let (t1, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFloatFlag args "t1" (some "missing --t1=<float>")
    let (initF, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFloatFlag args "init" (some "missing --init=<float>")
    let (lw, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFlagValue args "lower" (some "missing --lower=<weights.json>")
    let (uw, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.requireFlagValue args "upper" (some "missing --upper=<weights.json>")
    let (maxDepth, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.takeNatFlag args "maxDepth" (default := 18)
    let (minWidth, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.takeFloatFlag args "minWidth" (default := 1e-3)
    let (slack, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.takeFloatFlag args "slack" (default := 0.0)
    let (verbose, args) ← TorchLean.CLI.orThrowIO <|
      TorchLean.CLI.takeBoolValueFlag args "verbose" (default := false)
    TorchLean.CLI.orThrowIO (TorchLean.CLI.checkNoArgs args)
    let init := (initF, initF)
    let rhsAst ←
      match Parse.parseExpr rhsS with
      | .ok e => pure e
      | .error msg => throw <| IO.userError s!"RHS parse error: {msg}"
    let cfg0 : ODEVerifierSettings :=
      { maxDepth := maxDepth
        minWidth := minWidth
        slack := slack
        verbose := verbose
        modelBackend := backendOverride.getD .direct }
    let config := { cfg0 with arithmetic := arithmeticOverride.getD .native }
    let seg : ODECertificateSegment :=
      { t := { lo := t0, hi := t1 }, init := init, lowerWeights := lw, upperWeights := uw }
    match validateSettings config with
    | .error msg => throw <| IO.userError s!"Invalid ODE verifier settings: {msg}"
    | .ok () => pure ()
    match validateSegment seg with
    | .error msg => throw <| IO.userError s!"Invalid ODE certificate segment: {msg}"
    | .ok () => pure ()
    match config.arithmetic with
    | .native =>
      let ok ← verifySegmentWith (α := Float) (fun x => x) (fun mb p => loadModelFloat mb p) rhsAst
        seg config
      if ok then IO.println "[ODE] verification succeeded."
      else throw <| IO.userError "[ODE] verification failed."
    | .ieee =>
      let αI := TorchLean.Floats.IEEE754.IEEE32Exec
      let ofF := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat
      let ok ← verifySegmentWith (α := αI) ofF (fun mb p => loadModelNonFloat (α := αI) ofF mb p)
        rhsAst seg config
      if ok then IO.println "[ODE] verification succeeded."
      else throw <| IO.userError "[ODE] verification failed."

/--
`lake exe verify` entry point for the ODE verifier.

Prints a short help message on `--help` / `-h`, otherwise dispatches to `runArgs`.
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

end NN.Verification.ODE.Verify
