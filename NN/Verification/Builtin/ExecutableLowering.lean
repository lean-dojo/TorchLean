/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Builtin.Correctness
public import NN.Runtime.Autograd.IRExec.API

/-!
# Executable IR Lowering

Connects the verification IR to its forward executor:

1. lower a TorchLean `Program` to `NN.IR.Graph` and a verifier `ParamStore`;
2. convert the parameter store to an IR `Payload`; and
3. lower the IR graph to `Runtime.Autograd.IRExec.ForwardGraph`.

The returned `LoweredIR` remains available for verification. The accompanying forward graph runs
that same IR artifact; it is not the differentiable graph returned by `nn.lowerToTypedGraph`.

This function composes two executable lowerings. It does not strengthen the broad program-to-IR
pass with the theorem for `NN.Verification.Builtin.Proved.ForwardProgram`.
-/

@[expose] public section

namespace NN.Verification.Builtin

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.IR

/--
Lower a TorchLean forward model with one distinguished input to shared IR and its
forward-executable graph.

Success establishes that both checked lowerings accepted the concrete program and payload. Use a
named lowering theorem when a claim also requires equality with a source evaluator.
-/
@[noinline, nospecialize]
def lowerForwardExecutable
    {α : Type} [TorchLean.Storage α] [Context α]
    {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Runtime.Autograd.Model.Program α (paramShapes ++ [inShape]) outShape)
    (params : TorchLean.TensorPack α paramShapes) :
    Except String (LoweredIR α × Runtime.Autograd.IRExec.ForwardGraph α) := do
  let lowered ← lowerForwardToIR (α := α) (paramShapes := paramShapes) (inShape := inShape)
    (outShape := outShape) model params
  let payload : Payload α := payloadOfParamStore (α := α) lowered.ps
  let graph ← Runtime.Autograd.IRExec.lowerToForwardGraph (α := α) lowered.graph payload
  pure (lowered, graph)

end NN.Verification.Builtin
