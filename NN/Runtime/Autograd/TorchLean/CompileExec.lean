/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec
public import NN.Verification.TorchLean.Compile
public import NN.Verification.TorchLean.Correctness

/-!
# CompileExec

TorchLean → shared IR → executable SSA graph (runtime-facing wrapper).

This module wraps the shared-IR compilation pipeline for runtime use:

1. compile a TorchLean `Program` to the shared op-tagged IR (`NN.IR.Graph`),
2. convert verifier-style parameter stores to an IR `Payload`, and
3. lower the IR graph to an executable forward SSA graph
  (`Runtime.Autograd.Compiled.ExecGraphData`).

This is the “one shared IR” path described in the TorchLean paper: the exact same IR artifact is
usable for execution and verification.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Autodiff

open NN.IR

/-- Result of compiling a TorchLean forward graph through the shared IR and then to executable SSA.
  -/
structure CompiledIRExec (α : Type) [Context α] where
  /-- graph. -/
  graph    : NN.IR.Graph
  /-- payload. -/
  payload  : NN.IR.Payload α
  /-- input Id. -/
  inputId  : Nat
  /-- output Id. -/
  outputId : Nat
  /-- exec. -/
  exec     : Runtime.Autograd.Compiled.ExecGraphData α

/--
Compile a TorchLean forward model (single distinguished input) to:
- the shared op-tagged IR (`NN.IR.Graph`),
- a concrete IR `Payload` (parameters/constants),
- and an executable forward SSA graph (`ExecGraphData`).
-/
def compileForward1IRExec
    {α : Type} [Context α] [DecidableEq Shape] [Inhabited α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.TorchLean.Program α (paramShapes ++ [inShape]) outShape)
    (params : Runtime.Autograd.Torch.TList α paramShapes) :
    Except String (CompiledIRExec α) := do
  let c ← NN.Verification.TorchLean.compileForward1
    (α := α) (paramShapes := paramShapes) (inShape := inShape) (outShape := outShape) model params
  let payload : NN.IR.Payload α :=
    NN.Verification.TorchLean.payloadOfParamStore (α := α) c.ps
  let exec ← Runtime.Autograd.Compiled.execGraphOfIR (α := α) c.graph payload
  pure { graph := c.graph, payload := payload, inputId := c.inputId, outputId := c.outputId, exec :=
    exec }

end Autodiff

end TorchLean
end Autograd
end Runtime
