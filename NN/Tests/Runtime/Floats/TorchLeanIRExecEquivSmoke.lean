/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.IR.Semantics
public import NN.Tests.Runtime.Floats.Utils
public import NN.Verification.TorchLean.CompileExec
public import Std

/-!
# TorchLeanIRExecEquivSmoke

Smoke test: IR denotation agrees with the executable `IRExec` bridge.

We compile a small TorchLean model to `NN.IR.Graph` (plus payload), then compile that IR to an
executable `ExecGraphData` and check that both evaluators produce the same output tensor.
-/

@[expose] public section


open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIRExecEquivSmoke

def run : IO Unit := do
  IO.println "torchlean_ir_exec_equiv_smoke: begin"

  let inDim : Nat := 2
  let hidDim : Nat := 3
  let outDim : Nat := 1
  let xShape : Shape := NN.Tensor.Shape.Vec inDim
  let yShape : Shape := NN.Tensor.Shape.Vec outDim

  -- A small deterministic TorchLean MLP (weights initialized by explicit seeds).
  let model :=
    NN.GraphSpec.Models.TorchLean.mlp
      (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      (seedW1 := 0) (seedB1 := 1) (seedW2 := 2) (seedB2 := 3)

  let paramShapes := Runtime.Autograd.TorchLean.NN.Seq.paramShapes model
  let params : Runtime.Autograd.Torch.TList Float paramShapes :=
    Runtime.Autograd.TorchLean.NN.Seq.initParams (m := model)

  -- One input vector.
  let x : Tensor Float xShape :=
    Tensor.dim (fun i => Tensor.scalar ([0.5, 0.8][i.val]!))

  -- TorchLean program for the model.
  let prog :
      Runtime.Autograd.TorchLean.Program Float (paramShapes ++ [xShape]) yShape :=
    Runtime.Autograd.TorchLean.NN.Seq.program (model := model) (α := Float)

  -- Compile to IR and executable `ExecGraphData`.
  let (c, exec) ←
    match NN.Verification.TorchLean.compileForward1Exec
        (α := Float) (paramShapes := paramShapes) (inShape := xShape) (outShape := yShape) prog
          params with
    | .error e => throw <| IO.userError s!"torchlean_ir_exec_equiv_smoke: compile failed: {e}"
    | .ok r => pure r

  let payload : NN.IR.Payload Float :=
    NN.Verification.TorchLean.payloadOfParamStore (α := Float) c.ps

  -- Cast the test input into the executable graph's expected input shape.
  let xExec : Tensor Float exec.inShape ←
    if hIn : exec.inShape = xShape then
      pure <| Tensor.castShape x (Eq.symm hIn)
    else
      let msg :=
        s!"torchlean_ir_exec_equiv_smoke: exec input shape mismatch: got {repr exec.inShape}" ++
          s!", expected {repr xShape}"
      throw <| IO.userError msg

  -- IR denotation at the compiled output node.
  let yIR : Tensor Float yShape ←
    match NN.IR.Graph.denote (α := Float) (g := c.graph) (payload := payload)
        (input := NN.IR.DVal.mk (α := Float) xShape x) (outputId := c.outputId) with
    | .error e => throw <| IO.userError s!"torchlean_ir_exec_equiv_smoke: IR denote failed: {e}"
    | .ok out =>
        match NN.IR.Graph.expectShape (α := Float) (expected := yShape) out with
        | .ok t => pure t
        | .error e =>
            throw <| IO.userError s!"torchlean_ir_exec_equiv_smoke: IR output shape mismatch: {e}"

  -- Executable `GraphData` evaluation, then read the IR output id from the full value table.
  let execVals := Runtime.Autograd.Compiled.ExecGraphData.denoteAll (α := Float) exec xExec
  let yExec : Tensor Float yShape ←
    match execVals[c.outputId]? with
    | none =>
        let msg :=
          "torchlean_ir_exec_equiv_smoke: exec outputId out of bounds: " ++
            s!"{c.outputId}"
        throw <| IO.userError msg
    | some out =>
        match NN.IR.Graph.expectShape (α := Float) (expected := yShape) out with
        | .ok t => pure t
        | .error e =>
            throw <| IO.userError s!"torchlean_ir_exec_equiv_smoke: exec output shape mismatch: {e}"

  for i in List.finRange outDim do
    assertApprox s!"ir/exec forward[{i.val}]" (vecVal yIR i) (vecVal yExec i) 1e-6

  IO.println "torchlean_ir_exec_equiv_smoke: ok"

end TorchLeanIRExecEquivSmoke
end Floats
end Tests
