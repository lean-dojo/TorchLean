/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Lowering.Builder
public import NN.IR.Infer
public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Run

/-!
# Verification IR Lowering

Public entry points for lowering TorchLean programs to the shared verifier IR and querying IBP and
CROWN bounds. Graph construction lives in `Lowering.Builder`; this module contains the stable
result type and the operations available on a lowered graph.

The broad `lowerForwardToIR` entry point runs a `TorchLean.Program` with the IR-building
interpreter, then checks the graph's structure and shapes. A successful result is executable and
ready for verifier passes, but does not by itself prove equality with the source program. The
theorem-backed source fragment lives under `NN.Verification.Builtin.Proved`.
-/

@[expose] public section

namespace NN.Verification.Builtin

open Spec TorchLean
open TorchLean.Tensor
open NN.IR

/-! ## Public lowering entry points -/

/--
Result of lowering a TorchLean forward model to verifier IR.

This bundles:
- the produced IR graph (`NN.IR.Graph`),
- a CROWN/LiRPA-style `ParamStore` containing constants and layer parameters, and
- the distinguished input/output node ids (used by bound propagation and certificate checkers).
-/
structure LoweredIR (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Lowered IR graph. -/
  graph    : Graph
  /-- Parameters/constants for verifier algorithms (IBP, CROWN, etc.). -/
  ps       : NN.MLTheory.CROWN.Graph.ParamStore α
  /-- Distinguished input node id (kept stable as `0`). -/
  inputId  : Nat
  /-- Output node id. -/
  outputId : Nat

/-- Seed the distinguished verifier input with an explicit flat input box. -/
def LoweredIR.seedInputBox {α : Type} [TorchLean.Storage α] [Context α]
    (lowered : LoweredIR α) (xB : NN.MLTheory.CROWN.FlatBox α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  lowered.ps.seedInputBox lowered.inputId xB

/-- Flatten a shaped center/radius pair into the verifier input-box representation. -/
def lInfBox {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (center radius : Tensor α s) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBox (α := α) center radius

/-- Uniform $\ell^\infty$ box around a shaped TorchLean input tensor. -/
def lInfBall {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (center : Tensor α s) (eps : α) : NN.MLTheory.CROWN.FlatBox α :=
  NN.MLTheory.CROWN.FlatBox.lInfBall (α := α) center eps

/-- Flattening an $\ell^\infty$ ball preserves the number of tensor entries as its dimension. -/
@[simp] theorem lInfBall_dim {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (center : Tensor α s) (eps : α) :
    (lInfBall center eps).dim = s.size := by
  simp [lInfBall, NN.MLTheory.CROWN.FlatBox.lInfBall,
    NN.MLTheory.CROWN.FlatBox.lInfBox]

/-- Seed the distinguished verifier input with a uniform $\ell^\infty$ ball. -/
def LoweredIR.seedLInfBall {α : Type} [TorchLean.Storage α] [Context α] {s : Shape}
    (lowered : LoweredIR α) (center : Tensor α s) (eps : α) :
    NN.MLTheory.CROWN.Graph.ParamStore α :=
  lowered.ps.seedLInfBall lowered.inputId center eps

/-- Shape of the distinguished verifier input node. -/
def LoweredIR.inputShape? {α : Type} [TorchLean.Storage α] [Context α] (lowered : LoweredIR α) :
    Except String Shape := do
  match lowered.graph.nodes[lowered.inputId]? with
  | some node => pure node.outShape
  | none =>
      throw (s!"lowered verifier input node {lowered.inputId} is out of bounds " ++
        s!"for {lowered.graph.nodes.size} graph nodes")

/-- Flattened dimension of the distinguished verifier input node. -/
def LoweredIR.inputDim? {α : Type} [TorchLean.Storage α] [Context α] (lowered : LoweredIR α) :
    Except String Nat := do
  pure (Spec.Shape.size (← lowered.inputShape?))

/-- Affine/CROWN context for the distinguished verifier input. -/
def LoweredIR.affineCtx? {α : Type} [TorchLean.Storage α] [Context α] (lowered : LoweredIR α) :
    Except String NN.MLTheory.CROWN.Graph.AffineCtx := do
  pure { inputId := lowered.inputId, inputDim := ← lowered.inputDim? }

/-- Run IBP on a lowered verifier graph. -/
def LoweredIR.runIBP {α : Type} [TorchLean.Storage α] [Context α] [NN.MLTheory.CROWN.BoundOps α]
    [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α)
    (ps : NN.MLTheory.CROWN.Graph.ParamStore α) :
    Array (Option (NN.MLTheory.CROWN.FlatBox α)) :=
  NN.MLTheory.CROWN.Graph.runIBP (α := α) lowered.graph ps

/-- Read the verifier output box from an IBP result array. -/
def LoweredIR.outputBox? {α : Type} [TorchLean.Storage α] [Context α]
    (lowered : LoweredIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  NN.MLTheory.CROWN.Graph.outputBox? boxes lowered.outputId

/-- Read the lowered verifier output box, throwing an `IO.userError` if it is missing. -/
def LoweredIR.outputBoxOrThrow {α : Type} [TorchLean.Storage α] [Context α]
    (lowered : LoweredIR α)
    (boxes : Array (Option (NN.MLTheory.CROWN.FlatBox α))) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match lowered.outputBox? boxes with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Read the verifier output affine form from a forward affine result array. -/
def LoweredIR.outputAffine? {α : Type} [TorchLean.Storage α] [Context α]
    (lowered : LoweredIR α)
    (affs : Array (Option (NN.MLTheory.CROWN.Graph.FlatAffine α))) :
    Except String (NN.MLTheory.CROWN.Graph.FlatAffine α) := do
  match affs[lowered.outputId]? with
  | some (some outAff) => pure outAff
  | some none => throw s!"verification output affine missing at node {lowered.outputId}"
  | none =>
      throw (s!"verification output node {lowered.outputId} is out of bounds " ++
        s!"for {affs.size} affine entries")

/-- Run forward CROWN and evaluate the lowered verifier output on a selected input box. -/
def LoweredIR.outputBoxCROWN? {α : Type} [TorchLean.Storage α] [Context α]
    [NN.MLTheory.CROWN.BoundOps α] [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let inputDim ← lowered.inputDim?
  NN.MLTheory.CROWN.Graph.outputBoxCROWN? (α := α) lowered.graph ps xB
    lowered.inputId lowered.outputId inputDim

/-- Run forward CROWN for a lowered verifier graph, throwing an `IO.userError` on failure. -/
def LoweredIR.outputBoxCROWNOrThrow {α : Type} [TorchLean.Storage α] [Context α]
    [NN.MLTheory.CROWN.BoundOps α] [NN.MLTheory.CROWN.NonlinearBoundOps α]
    (lowered : LoweredIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (xB : NN.MLTheory.CROWN.FlatBox α) :
    IO (NN.MLTheory.CROWN.FlatBox α) := do
  match lowered.outputBoxCROWN? ps xB with
  | .ok outB => pure outB
  | .error msg => throw <| IO.userError msg

/-- Run objective-dependent backward CROWN and evaluate the scalar objective on the input box. -/
def LoweredIR.backwardObjectiveBox? {α : Type} [TorchLean.Storage α] [Context α]
    [NN.MLTheory.CROWN.BoundOps α]
    (lowered : LoweredIR α) (ps : NN.MLTheory.CROWN.Graph.ParamStore α)
    (ibp : Array (Option (NN.MLTheory.CROWN.FlatBox α)))
    (xB : NN.MLTheory.CROWN.FlatBox α) (obj : NN.MLTheory.CROWN.Graph.FlatTensor α) :
    Except String (NN.MLTheory.CROWN.FlatBox α) := do
  let ctx ← lowered.affineCtx?
  NN.MLTheory.CROWN.Graph.backwardObjectiveBox? (α := α) lowered.graph ps ctx
    ibp xB lowered.outputId obj

/-- Convert a parameter `TorchLean.TensorPack` into constant references for IR lowering. -/
def refListConstOfPack {α : Type} [TorchLean.Storage α] [Context α] :
    {ss : List Shape} → TorchLean.TensorPack α ss →
      Runtime.Autograd.Torch.RefList (fun s => Ref α s) ss
  | .nil, .nil => .nil
  | .cons _s ss, .cons t ts => .cons (.const t) (refListConstOfPack (ss := ss) ts)

/--
Lower a TorchLean forward model with one distinguished input, supplied as its last argument.

Success means that every encountered operation was supported and that the produced IR passed
structural and shape validation. It does not attach a source-to-IR semantic theorem.
-/
@[noinline, nospecialize]
def lowerForwardToIR
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.Model.Program α (paramShapes ++ [inShape]) outShape)
    (params : TorchLean.TensorPack α paramShapes) :
    Except String (LoweredIR α) :=
  let build : BuildM α Nat := do
    let x : Ref α inShape ← emitInput (α := α)
    let psRefs : Runtime.Autograd.Torch.RefList (Ref α) paramShapes :=
      refListConstOfPack (α := α) (ss := paramShapes) params
    let allRefs : Runtime.Autograd.Torch.RefList (Ref α) (paramShapes ++ [inShape]) :=
      Runtime.Autograd.Torch.RefList.append (ss₁ := paramShapes) (ss₂ := [inShape]) psRefs (.cons x
        .nil)
    let outRef ← Runtime.Autograd.Torch.CurriedRef.uncurry
      (Ref := fun s => Ref α s) (ss := paramShapes ++ [inShape]) (model (m := BuildM α)) allRefs
    ensureNode (α := α) outRef
  match StateT.run build { nodes := #[], ps := {} } with
  | Except.error e => Except.error e
  | Except.ok (outId, st) =>
      let g : Graph := { nodes := st.nodes }
      match (g.checkWellFormed *> g.checkShapes) with
      | Except.error e => Except.error s!"TorchLean→IR: produced an ill-formed graph: {e}"
      | Except.ok _ =>
          Except.ok { graph := g, ps := st.ps, inputId := 0, outputId := outId }

end NN.Verification.Builtin
