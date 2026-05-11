/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.PyTorch
public import NN.Verification.ODE.Parse
public import NN.Verification.TorchLean.Compile
public import NN.Verification.Util.Json

/-!
# Verify

ODE enclosure verification via NN sub- and super-solutions.

This module implements the core executable checking loop inspired by arXiv:2601.19818:

Given candidate functions `u₋, u₊` with
  `u₋(t₀) ≤ u₀ ≤ u₊(t₀)`,
  `u₋(t) ≤ u₊(t)` for all `t`,
  `u₋'(t) ≤ f(t, u₋(t))` and `u₊'(t) ≥ f(t, u₊(t))` for all `t`,
these are the corridor inequalities used by the classical comparison theorem for enclosing a
solution of `u' = f(t,u)` inside `[u₋, u₊]`. The executable checker validates these sufficient
inequalities on the chosen time boxes; the mathematical existence/comparison theorem is the
external classical assumption behind the workflow.

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
open _root_.Spec
open _root_.Spec.Tensor
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

/-- Interval width `hi - lo`. -/
@[inline] def width (I : Interval) : Float := I.hi - I.lo
/-- Interval center `(lo + hi)/2`. -/
@[inline] def center (I : Interval) : Float := (I.lo + I.hi) * 0.5
/-- Interval radius `(hi - lo)/2`. -/
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
- `ps0`: parameters and constants,
- `outId` and `dOutId`: output node ids for `u` and `du/dt`,
- `inDim`: expected input dimension (1 for ODE time).
-/
/--
An imported corridor network together with its derived (time-derivative) graph.

This is the core executable artifact we verify against a parsed ODE certificate.
-/
structure Model (α : Type) [Context α] where
  /-- Forward graph computing `u(t)`. -/
  g     : Graph
  /-- Derived graph computing both `u(t)` and `du/dt` (by structural differentiation). -/
  dg    : Graph
  /-- Parameters/constants for the graphs. -/
  ps0   : ParamStore α
  /-- Output node id of `u(t)` in `g`. -/
  outId : Nat
  /-- Output node id of `du/dt` in `dg`. -/
  dOutId : Nat
  /-- Input dimension (expected to be 1 for scalar time). -/
  inDim : Nat

/-!
Internal state for building derivative graphs.
-/
private structure DerivBuildState (α : Type) [Context α] where
  nodes : Array Node
  ps : ParamStore α
  dId : Array Nat

/-- State monad used during derivative-graph construction. -/
private abbrev DerivBuildM (α : Type) [Context α] := StateT (DerivBuildState α) IO

/-- Build a constant vector value of length `n`, filled with `x`. -/
private def constVecFill {α : Type} [Context α] (n : Nat) (x : α) : FlatVec α :=
  { n := n, v := Spec.fill (α := α) x (.dim n .scalar) }

/-- Insert a constant value payload for a `.const` node id. -/
private def addConstVal {α : Type} [Context α] (ps : ParamStore α) (id : Nat) (v : FlatVec α) :
  ParamStore α :=
  { ps with constVals := ps.constVals.insert id v }

/-- Insert matmul parameters for a `.matmul` node id. -/
private def addMatmulW {α : Type} [Context α] (ps : ParamStore α) (id : Nat) (p : MatParams α) :
  ParamStore α :=
  { ps with matmulW := ps.matmulW.insert id p }

/-!
Build a derivative graph `d/dt` for a 1D-input graph `g`.

This is a structural AD pass over the IR: for each node we build a new node computing the
derivative w.r.t. the single scalar input.

Supported ops are exactly those used by the imported corridor networks.
-/
/--
Build a derivative graph `d/dt` for a 1D-input graph `g`.

This is a structural AD pass over the IR: for each node we build a corresponding derivative node
and store its id in a side table.
-/
private def buildDerivativeGraph1D {α : Type} [Context α]
    (g : Graph) (ps0 : ParamStore α) (outId : Nat) : IO (Graph × ParamStore α × Nat) := do
  if g.nodes.isEmpty then
    throw <| IO.userError "buildDerivativeGraph1D: empty graph"
  match (g.nodes[0]!).kind with
  | .input => pure ()
  | _ => throw <| IO.userError "buildDerivativeGraph1D: expected node 0 to be `.input`"

  let pushNode : List Nat → OpKind → Shape → DerivBuildM α Nat := fun parents kind outShape => do
    let st ← get
    let id := st.nodes.size
    let nodes' := st.nodes.push { id := id, parents := parents, kind := kind, outShape := outShape }
    set { st with nodes := nodes' }
    pure id

  let mkConstFill : Shape → α → DerivBuildM α Nat := fun outShape x => do
    let id ← pushNode [] (.const outShape) outShape
    modify fun st =>
      { st with ps := addConstVal (α := α) st.ps id (constVecFill (α := α) (Shape.size outShape) x) }
    pure id

  let init : DerivBuildState α :=
    { nodes := g.nodes, ps := ps0, dId := Array.replicate g.nodes.size 0 }

  let (_, st) ← (show DerivBuildM α Unit from do
    for i in [0:g.nodes.size] do
      let node := g.nodes[i]!
      let outShape := node.outShape
      match node.kind with
      | .input =>
          let did ← mkConstFill outShape Numbers.one
          modify fun st => { st with dId := st.dId.set! i did }
      | .const _ =>
          let did ← mkConstFill outShape Numbers.zero
          modify fun st => { st with dId := st.dId.set! i did }
      | .detach =>
          let did ← mkConstFill outShape Numbers.zero
          modify fun st => { st with dId := st.dId.set! i did }
      | .add =>
          match node.parents with
          | p1 :: p2 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!, st.dId[p2]!] .add outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at add node {i}"
      | .sub =>
          match node.parents with
          | p1 :: p2 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!, st.dId[p2]!] .sub outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sub node {i}"
      | .mul_elem =>
          match node.parents with
          | p1 :: p2 :: _ =>
              let st ← get
              let t1 ← pushNode [st.dId[p1]!, p2] .mul_elem outShape
              let t2 ← pushNode [p1, st.dId[p2]!] .mul_elem outShape
              let did ← pushNode [t1, t2] .add outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at mul_elem node {i}"
      | .linear =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              match st.ps.linearWB[i]? with
              | none =>
                  throw <| IO.userError <|
                    s!"buildDerivativeGraph1D: missing linearWB params at node {i}"
              | some p =>
                  let did ← pushNode [st.dId[p1]!] .matmul outShape
                  modify fun st =>
                    { st with ps := addMatmulW (α := α) st.ps did { m := p.m, n := p.n, w := p.w } }
                  modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at linear node {i}"
      | .matmul =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              match st.ps.matmulW[i]? with
              | none =>
                  throw <| IO.userError <|
                    s!"buildDerivativeGraph1D: missing matmulW params at node {i}"
              | some p =>
                  let did ← pushNode [st.dId[p1]!] .matmul outShape
                  modify fun st => { st with ps := addMatmulW (α := α) st.ps did p }
                  modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at matmul node {i}"
      | .tanh =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let y2 ← pushNode [i, i] .mul_elem outShape
              let one ← mkConstFill outShape Numbers.one
              let fac ← pushNode [one, y2] .sub outShape
              let did ← pushNode [fac, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at tanh node {i}"
      | .sigmoid =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let one ← mkConstFill outShape Numbers.one
              let oneMy ← pushNode [one, i] .sub outShape
              let yFac ← pushNode [i, oneMy] .mul_elem outShape
              let did ← pushNode [yFac, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sigmoid node {i}"
      | .exp =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [i, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at exp node {i}"
      | .log =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let invz ← pushNode [p1] .inv outShape
              let did ← pushNode [invz, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at log node {i}"
      | .sin =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let cosz ← pushNode [p1] .cos outShape
              let did ← pushNode [cosz, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sin node {i}"
      | .cos =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let sinz ← pushNode [p1] .sin outShape
              let neg1 ← mkConstFill outShape Numbers.neg_one
              let negsin ← pushNode [neg1, sinz] .mul_elem outShape
              let did ← pushNode [negsin, st.dId[p1]!] .mul_elem outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at cos node {i}"
      | .sum =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] .sum outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at sum node {i}"
      | .reshape inS outS =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.reshape inS outS) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reshape node {i}"
      | .flatten s =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.flatten s) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at flatten node {i}"
      | .permute perm =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.permute perm) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at permute node {i}"
      | .broadcastTo s₁ s₂ =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.broadcastTo s₁ s₂) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at broadcastTo node {i}"
      | .reduceSum axis =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.reduceSum axis) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reduce_sum node {i}"
      | .reduceMean axis =>
          match node.parents with
          | p1 :: _ =>
              let st ← get
              let did ← pushNode [st.dId[p1]!] (.reduceMean axis) outShape
              modify fun st => { st with dId := st.dId.set! i did }
          | _ => throw <| IO.userError s!"buildDerivativeGraph1D: bad arity at reduce_mean node {i}"
      | k =>
          throw <| IO.userError
            s!"buildDerivativeGraph1D: unsupported op in ODE derivative graph: {repr k} at node {i}"
  ).run init

  let dg : Graph := { nodes := st.nodes }
  let dOutId := st.dId[outId]!
  pure (dg, st.ps, dOutId)

/--
Seed the 1D input box (time) into a `ParamStore`.

The verifier represents a time interval `I = [lo, hi]` using its center `tCenter` and radius `tRad`,
and inserts the resulting box at input node id `0`.
-/
private def seedInput1D {α : Type} [Context α]
    (ps : ParamStore α) (ofFloat : Float → α) (tCenter tRad : Float) : ParamStore α :=
  let tC := ofFloat tCenter
  let tR := ofFloat tRad
  let t0 : Tensor α (.dim 1 .scalar) := Tensor.dim (fun _ => Tensor.scalar tC)
  let rad := Spec.fill (α:=α) tR (.dim 1 .scalar)
  let tB : Box α (.dim 1 .scalar) := { lo := Tensor.subSpec t0 rad, hi := Tensor.addSpec t0 rad }
  { ps with inputBoxes := ps.inputBoxes.insert 0 { dim := 1, lo := tB.lo, hi := tB.hi } }

/--
Compute bounds for `u(t)` and `du/dt` on a time interval.

This evaluates IBP over the derivative graph `m.dg` after seeding an input box centered at
`I.center` with radius `I.radius`.
-/
private def boundsOn {α : Type} [Context α] [DecidableEq Shape]
    (ofFloat : Float → α) (m : Model α) (I : Interval) : IO (Bounds α) := do
  if m.inDim ≠ 1 then
    throw <| IO.userError s!"ODE verifier expects inputDim=1, got {m.inDim}"
  let ps := seedInput1D (α := α) m.ps0 ofFloat I.center I.radius
  let ibp := runIBP (α:=α) m.dg ps
  let some outB := ibp[m.outId]! | throw <| IO.userError "IBP failed at output"
  let uLo := Spec.Tensor.sumSpec outB.lo
  let uHi := Spec.Tensor.sumSpec outB.hi
  let some dB := ibp[m.dOutId]! | throw <| IO.userError "IBP failed at derivative output"
  let duLo := Spec.Tensor.sumSpec dB.lo
  let duHi := Spec.Tensor.sumSpec dB.hi
  pure { u := (uLo, uHi), du := (duLo, duHi) }

/--
How to load the corridor network weights.

`direct` uses the PyTorch-import graph builder directly. `torchlean` goes through the TorchLean
compiler path (useful for testing the compilation pipeline).
-/
inductive ModelBackend where
  | direct
  | torchlean
  deriving DecidableEq, Repr

/-- Pretty-print the backend choice for CLI logs. -/
instance : ToString ModelBackend :=
  ⟨fun b => match b with | .direct => "direct" | .torchlean => "torchlean"⟩

/-- Parse a `--model=...` argument into a `ModelBackend` choice. -/
private def parseModelBackendArg (s : String) : Option ModelBackend :=
  if s = "--model=torchlean" ∨ s = "--model torchlean" then some .torchlean
  else if s = "--model=direct" ∨ s = "--model direct" then some .direct
  else none

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
private def loadModelDirectWith {α : Type} [Context α] (ofFloat : Float → α) (path : String) : IO
  (Model α) := do
  let sd ← loadPinnState path
  if sd.arch.outputDim ≠ 1 then
    throw <| IO.userError
      s!"ODE verifier expects scalar outputDim=1, got {sd.arch.outputDim} in {path}"
  let g := Import.PINNPyTorch.buildGraph sd
  let ps0 := Import.PINNPyTorch.toParamStoreWith (α := α) ofFloat sd
  let outId := SequentialPINNArch.graphOutputId g
  let (dg, ps1, dOutId) ← buildDerivativeGraph1D (α := α) g ps0 outId
  pure { g := g, dg := dg, ps0 := ps1, outId := outId, dOutId := dOutId, inDim := sd.arch.inputDim }

/-- Linear-layer payload extracted from a PINN export (`w`, `b`). -/
private structure LinLayer (inDim outDim : Nat) where
  w : Tensor Float (.dim outDim (.dim inDim .scalar))
  b : Tensor Float (.dim outDim .scalar)

/--
Simple representation of an MLP as a chain of linear layers.

This is used when lowering imported PINN weights into the TorchLean compilation pipeline.
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
  | [] => .error "empty MLP (no layers)"
  | [l] => .ok ⟨l.inDim, l.outDim, .last (linOfPinn l)⟩
  | l :: rest =>
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
Load a corridor model via the TorchLean compilation pipeline.

This path reconstructs a small TorchLean program from the imported weights, compiles it to a CROWN
graph, and then builds the derivative graph from that compiled graph.
-/
private def loadModelTorchLean (path : String) : IO (Model Float) := do
  let sd ← loadPinnState path
  match chainOfPinnLayers sd.layers with
  | .error e => throw <| IO.userError s!"Bad layer chain in {path}: {e}"
  | .ok ⟨inDim, outDim, chain⟩ =>
    if outDim ≠ 1 then
      throw <| IO.userError s!"ODE verifier expects scalar outputDim=1, got {outDim} in {path}"
    let xShape : Shape := .dim inDim .scalar
    let yShape : Shape := .dim outDim .scalar

    match sd.arch.activation with
    | .tanh =>
      let model : Runtime.Autograd.TorchLean.Program Float [xShape] yShape :=
        fun {m} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := Float)] =>
          fun x =>
            let rec evalT
                {inD outD : Nat}
                (ch : LayerChain inD outD)
                (x : Runtime.Autograd.TorchLean.RefTy (m := m) (α := Float) (.dim inD .scalar)) :
                m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := Float) (.dim outD .scalar)) := do
              match ch with
              | .last l =>
                let wR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim outD (.dim inD .scalar)) l.w
                let bR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim outD .scalar) l.b
                Runtime.Autograd.TorchLean.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := outD) wR bR x
              | .cons l tail =>
                let wR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim _ (.dim _ .scalar)) l.w
                let bR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim _ .scalar) l.b
                let z ← Runtime.Autograd.TorchLean.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := _) wR bR x
                let a ← Runtime.Autograd.TorchLean.tanh (m := m) (α := Float) (s := .dim _ .scalar)
                  z
                evalT tail a
            evalT chain x
      let compiled ←
        match NN.Verification.TorchLean.compileForward1
              (α := Float) (paramShapes := []) (inShape := xShape) (outShape := yShape)
              model (.nil) with
        | .ok c => pure c
        | .error e => throw <| IO.userError e
      let (dg, ps1, dOutId) ← buildDerivativeGraph1D (α := Float) compiled.graph compiled.ps
        compiled.outputId
      pure { g := compiled.graph, dg := dg, ps0 := ps1, outId := compiled.outputId, dOutId :=
        dOutId, inDim := inDim }
    | .relu =>
      let model : Runtime.Autograd.TorchLean.Program Float [xShape] yShape :=
        fun {m} [Monad m] [Runtime.Autograd.Torch.Ops (m := m) (α := Float)] =>
          fun x =>
            let rec evalR
                {inD outD : Nat}
                (ch : LayerChain inD outD)
                (x : Runtime.Autograd.TorchLean.RefTy (m := m) (α := Float) (.dim inD .scalar)) :
                m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := Float) (.dim outD .scalar)) := do
              match ch with
              | .last l =>
                let wR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim outD (.dim inD .scalar)) l.w
                let bR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim outD .scalar) l.b
                Runtime.Autograd.TorchLean.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := outD) wR bR x
              | .cons l tail =>
                let wR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim _ (.dim _ .scalar)) l.w
                let bR ← Runtime.Autograd.TorchLean.const (m := m) (α := Float)
                  (s := .dim _ .scalar) l.b
                let z ← Runtime.Autograd.TorchLean.linear (m := m) (α := Float)
                  (inDim := inD) (outDim := _) wR bR x
                let a ← Runtime.Autograd.TorchLean.relu (m := m) (α := Float) (s := .dim _ .scalar)
                  z
                evalR tail a
            evalR chain x
      let compiled ←
        match NN.Verification.TorchLean.compileForward1
              (α := Float) (paramShapes := []) (inShape := xShape) (outShape := yShape)
              model (.nil) with
        | .ok c => pure c
        | .error e => throw <| IO.userError e
      let (dg, ps1, dOutId) ← buildDerivativeGraph1D (α := Float) compiled.graph compiled.ps
        compiled.outputId
      pure { g := compiled.graph, dg := dg, ps0 := ps1, outId := compiled.outputId, dOutId :=
        dOutId, inDim := inDim }
    | .sin =>
      throw <| IO.userError
        "ODE verifier: --model=torchlean accepts ReLU/Tanh/Sigmoid activations; use --model=direct for sin."

/-- Load a corridor model in the default executable scalar (`Float`), choosing the backend path. -/
private def loadModelFloat (backend : ModelBackend) (path : String) : IO (Model Float) := do
  match backend with
  | .direct => loadModelDirectWith (α := Float) (fun x => x) path
  | .torchlean => loadModelTorchLean path

/-- Load a corridor model in a non-`Float` scalar backend (supported in `direct`
  mode). -/
private def loadModelNonFloat {α : Type} [Context α] (ofFloat : Float → α)
    (backend : ModelBackend) (path : String) : IO (Model α) := do
  match backend with
  | .direct => loadModelDirectWith (α := α) ofFloat path
  | .torchlean =>
    throw <| IO.userError "ODE verifier: --model=torchlean is only supported with --scalar=float"

/-- Choice of scalar backend for the verifier. -/
inductive ScalarBackend where
  | float
  | ieee32exec
  deriving DecidableEq, Repr

/-- Pretty-print the scalar backend choice for CLI logs. -/
instance : ToString ScalarBackend :=
  ⟨fun b => match b with | .float => "float" | .ieee32exec => "ieee32exec"⟩

/--
Verification settings controlling domain splitting and slack.

These come from the `settings` section of a certificate JSON (and can be partially overridden by
  CLI).
-/
structure ODEVerifierSettings where
  /-- Maximum recursion depth for time-domain splitting. -/
  maxDepth : Nat := 18
  /-- Stop splitting once interval width falls below this (even if not verified). -/
  minWidth : Float := 1e-3
  /-- Optional numerical slack added to comparisons (helps absorb small floating-point error). -/
  slack : Float := 0.0
  /-- Print intermediate bounds and decisions for debugging. -/
  verbose : Bool := false
  /-- How to import the corridor networks. -/
  modelBackend : ModelBackend := .direct
  /-- Which scalar backend to run the checks in. -/
  scalar : ScalarBackend := .float
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
  segments : List ODECertificateSegment
  /-- Optional settings record (defaults apply when omitted). -/
  settings : ODEVerifierSettings := {}
  deriving Repr

/-- Parse a `Float` from JSON (expects a JSON number). -/
private def parseFloatFromJson (j : Json) : Except String Float :=
  match j with
  | .num n => .ok n.toFloat
  | _ => .error "expected JSON number"

/-- Parse a `String` from JSON (expects a JSON string). -/
private def parseStrFromJson (j : Json) : Except String String :=
  match j with
  | .str s => .ok s
  | _ => .error "expected JSON string"

/-- Parse an interval object `{ lo: ..., hi: ... }` from JSON. -/
private def parseIntervalObj (j : Json) : Except String Interval := do
  let o ← j.getObj?
  let lo ← (o.get? "lo").getD Json.null |> parseFloatFromJson
  let hi ← (o.get? "hi").getD Json.null |> parseFloatFromJson
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
    let maxDepth :=
      match o.get? "maxDepth" with
      | some (.num n) => (n.toFloat.toUInt64).toNat
      | _ => 18
    let minWidth :=
      match o.get? "minWidth" with
      | some (.num n) => n.toFloat
      | _ => 1e-3
    let slack :=
      match o.get? "slack" with
      | some (.num n) => n.toFloat
      | _ => 0.0
    let verbose :=
      match o.get? "verbose" with
      | some (.bool b) => b
      | _ => false
    let modelBackend :=
      match o.get? "modelBackend" with
      | some (.str "torchlean") => ModelBackend.torchlean
      | some (.str "direct") => ModelBackend.direct
      | _ => ModelBackend.direct
    let scalar :=
      match o.get? "scalar" with
      | some (.str "ieee32exec") => ScalarBackend.ieee32exec
      | some (.str "float") => ScalarBackend.float
      | _ => ScalarBackend.float
    pure { maxDepth := maxDepth, minWidth := minWidth, slack := slack, verbose := verbose
           , modelBackend := modelBackend, scalar := scalar }
  | _ => pure {}

/-- Parse a single segment object from the certificate JSON. -/
private def parseSegment (j : Json) : Except String ODECertificateSegment := do
  let o ← j.getObj?
  let t0 ← (o.get? "t0").getD Json.null |> parseFloatFromJson
  let t1 ← (o.get? "t1").getD Json.null |> parseFloatFromJson
  let initJ := (o.get? "init").getD Json.null
  let init ←
    match initJ with
    | .obj _ => parsePairObj initJ
    | .num n => pure (n.toFloat, n.toFloat)
    | _ => throw "segment.init must be number or {lo,hi}"
  let lw ← (o.get? "lowerWeights").getD Json.null |> parseStrFromJson
  let uw ← (o.get? "upperWeights").getD Json.null |> parseStrFromJson
  if t1 < t0 then throw "segment: t1 < t0" else
  pure { t := { lo := t0, hi := t1 }, init := init, lowerWeights := lw, upperWeights := uw }

/-- Parse the top-level certificate JSON object into an `ODECertificate`. -/
def parseODECertificate (j : Json) : Except String ODECertificate := do
  let o ← j.getObj?
  let rhs ← (o.get? "rhs").getD Json.null |> parseStrFromJson
  let segArr ← (o.get? "segments").getD Json.null |>.getArr?
  let segs ← segArr.toList.mapM parseSegment
  let settings ← parseSettings ((o.get? "settings").getD Json.null)
  pure { rhs := rhs, segments := segs, settings := settings }

/-- Parse a `Float` literal from a CLI argument (must consume the whole string). -/
private def parseFloatArg (s : String) : Option Float :=
  match NN.Verification.ODE.Parse.parseNumber { s := s } with
  | .ok (v, st) => if st.i = s.rawEndPos then some v else none
  | .error _ => none

/-- `List.findMap` specialized to `Option` for the local flag parser below. -/
private def findMap? {α β : Type} (f : α → Option β) : List α → Option β
  | [] => none
  | x :: xs =>
    match f x with
    | some y => some y
    | none => findMap? f xs

/-- Read `--key=value` from an argument list. -/
private def getFlag? (args : List String) (key : String) : Option String :=
  let pref := s!"--{key}="
  findMap? (fun a => if a.startsWith pref then some (toString (a.drop pref.length)) else none) args

/-- Read `--key value` from an argument list. -/
private def getPositional? (args : List String) (key : String) : Option String :=
  let rec go : List String → Option String
    | [] => none
    | a :: b :: rest =>
      if a = s!"--{key}" then some b else go (b :: rest)
    | [_] => none
  go args

/-- Read either `--key=value` or `--key value` (prefers the `--key=value` form if both appear). -/
private def argOrFlag? (args : List String) (key : String) : Option String :=
  getFlag? args key <|> getPositional? args key

/-- Boolean `<=` on scalars, defined in terms of the backend's `gtBool`. -/
def leBool {α : Type} [Context α] (x y : α) : Bool :=
  not (Context.gtBool x y)

/-- Show a closed interval pair as `(lo, hi)`. -/
private def showPair {α : Type} [ToString α] (p : α × α) : String :=
  s!"({p.1}, {p.2})"

/--
Subsolution check: ensure `du/dt <= f(t, u)` on the interval (with slack).

We compare the *upper bound* on `du` against the *lower bound* on `f`, allowing `slack` as a
nonnegative numerical tolerance.
-/
def checkSub {α : Type} [Context α] (du : α × α) (f : α × α) (slack : α) : Bool :=
  let duHi := du.2
  let fLo := f.1
  leBool duHi (fLo + slack)

/--
Supersolution check: ensure `du/dt >= f(t, u)` on the interval (with slack).

We compare the *lower bound* on `du` against the *upper bound* on `f`, allowing `slack` as a
nonnegative numerical tolerance.
-/
def checkSuper {α : Type} [Context α] (du : α × α) (f : α × α) (slack : α) : Bool :=
  let duLo := du.1
  let fHi := f.2
  leBool fHi (duLo + slack)

/-- Order check: ensure the lower corridor stays below the upper corridor (`u₋ <= u₊`), up to
the configured slack. -/
def checkOrder {α : Type} [Context α] (uL uU : α × α) (slack : α := Numbers.zero) : Bool :=
  leBool uL.2 (uU.1 + slack)

/--
Verify a single time interval by bounding `u₋, u₊, du₋, du₊` and checking the corridor inequalities.

Returns `(ok, msg)` where `msg` is a short debug string explaining the first failing check.
-/
private def verifyInterval {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    (ofFloat : Float → α) (rhs : Expr) (mL mU : Model α) (I : Interval) (cfg : ODEVerifierSettings) : IO (Bool
      × String) := do
  let bL ← boundsOn (α := α) ofFloat mL I
  let bU ← boundsOn (α := α) ofFloat mU I
  let slackA : α := ofFloat cfg.slack
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
private partial def verifySegmentAux {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    (ofFloat : Float → α) (rhs : Expr) (mL mU : Model α) (I : Interval) (cfg : ODEVerifierSettings) (depth :
      Nat) : IO Bool := do
  let (ok, msg) ← verifyInterval (α := α) ofFloat rhs mL mU I cfg
  if ok then
    if cfg.verbose then
      IO.println s!"[ODE] OK {I}"
    pure true
  else
    if depth = 0 ∨ I.width ≤ cfg.minWidth then
      IO.eprintln s!"[ODE] FAIL {msg}"
      pure false
    else
      let (I1, I2) := I.split
      let ok1 ← verifySegmentAux (α := α) ofFloat rhs mL mU I1 cfg (depth - 1)
      if ok1 then
        verifySegmentAux (α := α) ofFloat rhs mL mU I2 cfg (depth - 1)
      else
        pure false

/--
Verify a full certificate segment: check initial conditions at `t0`, then recursively verify the
  time interval.

This loads both corridor networks (lower/upper) and then runs `verifySegmentAux` on `seg.t`.
-/
  private def verifySegmentWith {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    (ofFloat : Float → α) (loadModel : ModelBackend → String → IO (Model α))
    (rhs : Expr) (seg : ODECertificateSegment) (cfg : ODEVerifierSettings) : IO Bool := do
  if cfg.verbose then
    IO.println <|
      s!"[ODE] loading models ({cfg.modelBackend}, scalar={cfg.scalar}): " ++
        s!"lower={seg.lowerWeights}, upper={seg.upperWeights}"
  let mL ← loadModel cfg.modelBackend seg.lowerWeights
  let mU ← loadModel cfg.modelBackend seg.upperWeights
  let slackA : α := ofFloat cfg.slack
  let t0I : Interval := { lo := seg.t.lo, hi := seg.t.lo }
  let bL0 ← boundsOn (α := α) ofFloat mL t0I
  let bU0 ← boundsOn (α := α) ofFloat mU t0I
  let (iLoF, iHiF) := seg.init
  let iLo : α := ofFloat iLoF
  let iHi : α := ofFloat iHiF
  if Context.gtBool bL0.u.2 iLo then
    IO.eprintln s!"[ODE] FAIL initial: uL(t0)∈{showPair bL0.u} not ≤ init.lo={iLoF}"
    return false
  if Context.gtBool iHi bU0.u.1 then
    IO.eprintln s!"[ODE] FAIL initial: uU(t0)∈{showPair bU0.u} not ≥ init.hi={iHiF}"
    return false
  if ¬checkOrder (α := α) bL0.u bU0.u slackA then
    IO.eprintln s!"[ODE] FAIL initial order: uL(t0)∈{showPair bL0.u} not ≤ uU(t0)∈{showPair bU0.u}"
    return false
  IO.println s!"[ODE] initial OK at t0={seg.t.lo}"
  verifySegmentAux (α := α) ofFloat rhs mL mU seg.t cfg cfg.maxDepth

/-- Parse a scalar backend name as used in CLI flags and certificate settings. -/
private def parseScalarName (s : String) : Option ScalarBackend :=
  if s = "float" then some .float
  else if s = "ieee32exec" then some .ieee32exec
  else none

/--
Run verification for a parsed certificate file.

This is the main executable entry used by `lake exe verify -- ode --cert=...`.
-/
def runCertificate (path : String) (backendOverride : Option ModelBackend) (scalarOverride : Option
  ScalarBackend) : IO Unit := do
  let j ← NN.Verification.Json.readJsonFile path
  let cert ←
    match parseODECertificate j with
    | .ok c => pure c
    | .error msg => throw <| IO.userError s!"Bad ODE cert JSON: {msg}"
  let rhsAst ←
    match Parse.parseExpr cert.rhs with
    | .ok e => pure e
    | .error msg => throw <| IO.userError s!"RHS parse error: {msg}"
  let cfg :=
    match backendOverride with
    | some mb => { cert.settings with modelBackend := mb }
    | none => cert.settings
  let cfg :=
    match scalarOverride with
    | some sc => { cfg with scalar := sc }
    | none => cfg
  match cfg.scalar with
  | .float =>
    let mut allOk := true
    for seg in cert.segments do
      let ok ← verifySegmentWith (α := Float) (fun x => x) (fun mb p => loadModelFloat mb p) rhsAst
        seg cfg
      if ¬ok then allOk := false
    if allOk then
      IO.println "[ODE] certificate verified: all segments succeeded."
    else
      throw <| IO.userError "[ODE] certificate verification failed."
  | .ieee32exec =>
    let αI := TorchLean.Floats.IEEE754.IEEE32Exec
    let ofF := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat
    let mut allOk := true
    for seg in cert.segments do
      let ok ← verifySegmentWith (α := αI) ofF (fun mb p => loadModelNonFloat (α := αI) ofF mb p)
        rhsAst seg cfg
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
  let (backendOverride, args) :=
    match args with
    | a :: rest =>
      match parseModelBackendArg a with
      | some mb => (some mb, rest)
      | none => (none, args)
    | [] => (none, [])
  let scalarOverride :=
    match argOrFlag? args "scalar" with
    | some s => parseScalarName s
    | none => none
  match argOrFlag? args "cert" with
  | some p => runCertificate p backendOverride scalarOverride
  | none =>
    let rhsS ←
      match argOrFlag? args "rhs" with
      | some s => pure s
      | none => throw <| IO.userError "missing --rhs=<expr>"
    let t0S ←
      match argOrFlag? args "t0" with
      | some s => pure s
      | none => throw <| IO.userError "missing --t0=<float>"
    let t1S ←
      match argOrFlag? args "t1" with
      | some s => pure s
      | none => throw <| IO.userError "missing --t1=<float>"
    let initS ←
      match argOrFlag? args "init" with
      | some s => pure s
      | none => throw <| IO.userError "missing --init=<float>"
    let lw ←
      match argOrFlag? args "lower" with
      | some s => pure s
      | none => throw <| IO.userError "missing --lower=<weights.json>"
    let uw ←
      match argOrFlag? args "upper" with
      | some s => pure s
      | none => throw <| IO.userError "missing --upper=<weights.json>"
    let t0 ←
      match parseFloatArg t0S with
      | some x => pure x
      | none => throw <| IO.userError s!"bad --t0: {t0S}"
    let t1 ←
      match parseFloatArg t1S with
      | some x => pure x
      | none => throw <| IO.userError s!"bad --t1: {t1S}"
    let init ←
      match parseFloatArg initS with
      | some x => pure (x, x)
      | none => throw <| IO.userError s!"bad --init: {initS}"
    let maxDepth :=
      match argOrFlag? args "maxDepth" with
      | some s => (parseFloatArg s).map (fun f => f.toUInt64.toNat) |>.getD 18
      | none => 18
    let minWidth :=
      match argOrFlag? args "minWidth" with
      | some s => (parseFloatArg s).getD 1e-3
      | none => 1e-3
    let slack :=
      match argOrFlag? args "slack" with
      | some s => (parseFloatArg s).getD 0.0
      | none => 0.0
    let verbose :=
      match argOrFlag? args "verbose" with
      | some "true" => true
      | some "1" => true
      | some "false" => false
      | some "0" => false
      | _ => false
    let rhsAst ←
      match Parse.parseExpr rhsS with
      | .ok e => pure e
      | .error msg => throw <| IO.userError s!"RHS parse error: {msg}"
    let mb := backendOverride.getD .direct
    let cfg0 : ODEVerifierSettings :=
      { maxDepth := maxDepth
        minWidth := minWidth
        slack := slack
        verbose := verbose
        modelBackend := mb }
    let cfg :=
      match scalarOverride with
      | some sc => { cfg0 with scalar := sc }
      | none => cfg0
    let seg : ODECertificateSegment :=
      { t := { lo := t0, hi := t1 }, init := init, lowerWeights := lw, upperWeights := uw }
    match cfg.scalar with
    | .float =>
      let ok ← verifySegmentWith (α := Float) (fun x => x) (fun mb p => loadModelFloat mb p) rhsAst
        seg cfg
      if ok then IO.println "[ODE] verification succeeded."
      else throw <| IO.userError "[ODE] verification failed."
    | .ieee32exec =>
      let αI := TorchLean.Floats.IEEE754.IEEE32Exec
      let ofF := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat
      let ok ← verifySegmentWith (α := αI) ofF (fun mb p => loadModelNonFloat (α := αI) ofF mb p)
        rhsAst seg cfg
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
         "[--scalar=float|ieee32exec] --cert=<cert.json>\n") ++
       ("  lake exe verify -- ode [--model=direct|torchlean] " ++
         "[--scalar=float|ieee32exec] --rhs=\"<expr>\" --t0=<float> " ++
         "--t1=<float> --init=<float> --lower=<wL.json> --upper=<wU.json>\n") ++
       "Options:\n" ++
       "  --maxDepth=<nat>   (default 18)\n" ++
       "  --minWidth=<float> (default 1e-3)\n" ++
       "  --slack=<float>    (default 0)\n" ++
       "  --scalar=float|ieee32exec\n" ++
       "  --verbose=true|false\n")
  else
    runArgs args

end NN.Verification.ODE.Verify
