/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Runtime.PyTorch.Export.IRPyTorch
public import NN.Verification.TorchLean.Compile

/-!
# TorchLean IR to PyTorch

Tutorial: TorchLean → IR (`NN.IR.Graph`) → emitted PyTorch code.

Run:
  `lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch cnn > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch conv-mlp > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py`
  `lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py`
Then:
  `python3 exported_model.py`
-/

@[expose] public section


namespace NN.Examples.Advanced.TorchIRPyTorch

open Spec
open Tensor
open NN.Tensor
open NN.API

/-! ## Architectures -/

def archLinear : nn.M (nn.Sequential (Shape.Vec 2) (Shape.Vec 1)) :=
  nn.linear 2 1 (pfx := Spec.Shape.scalar)

def archMLP : nn.M (nn.Sequential (Shape.Vec 2) (Shape.Vec 1)) :=
  nn.sequential![
    nn.linear 2 3 (pfx := Spec.Shape.scalar),
    nn.relu,
    nn.linear 3 1 (pfx := Spec.Shape.scalar)
  ]

def archSumReduce : nn.M (nn.Sequential (NN.Tensor.Shape.Vec 4) Spec.Shape.scalar) :=
  nn.sum (s := NN.Tensor.Shape.Vec 4)

def archAutoencoder : nn.M (nn.Sequential (Shape.Vec 3) (Shape.Vec 3)) :=
  nn.sequential![
    nn.linear 3 2 (pfx := Spec.Shape.scalar),
    nn.tanh,
    nn.linear 2 3 (pfx := Spec.Shape.scalar)
  ]

def archCNN : nn.M (nn.Sequential (NN.Tensor.Shape.Images 1 1 4 4) (shape![1, 3])) :=
  let featDim : Nat := Spec.Shape.size (Shape.CHW 2 2 2)
  nn.sequential![
    nn.conv (n := 1) (inC := 1) (inH := 4) (inW := 4)
      { outC := 2, kH := 3, kW := 3, stride := 1, padding := 0 },
    nn.relu,
    nn.flattenBatch (n := 1) (s := Shape.CHW 2 2 2),
    nn.linear featDim 3 (pfx := Shape.Vec 1)
  ]

def archConvMLP :
    nn.M (nn.Sequential (NN.Tensor.Shape.Images 1 1 3 3) (shape![1, 1])) :=
  -- Conv output: `Images 1 1 2 2`, so `flattenBatch` produces `Mat 1 4`.
  let featDim : Nat := Spec.Shape.size (Shape.CHW 1 2 2)
  nn.sequential![
    nn.conv (n := 1) (inC := 1) (inH := 3) (inW := 3)
      { outC := 1, kH := 2, kW := 2, stride := 1, padding := 0 },
    nn.relu,
    nn.flattenBatch (n := 1) (s := Shape.CHW 1 2 2),
    nn.linear featDim 3 (pfx := Shape.Vec 1),
    nn.relu,
    nn.linear 3 1 (pfx := Shape.Vec 1)
  ]

def archMHA :
    nn.M (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.multiheadAttention (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 }

def archMHAMask : Spec.Tensor Bool (NN.Tensor.Shape.Mat 4 4) :=
  text.causalMask 4

def archMHAMasked :
    nn.M (nn.Sequential (shape![1, 4, 8]) (shape![1, 4, 8])) :=
  nn.multiheadAttention (batch := 1) (n := 4) (dModel := 8)
    { numHeads := 2, headDim := 4 } (mask := some archMHAMask)

def archTransformer :
    nn.M (nn.Sequential (shape![1, 2, 2]) (shape![1, 2, 2])) :=
  nn.transformerEncoderBlock (batch := 1) (n := 2) (dModel := 2)
    { numHeads := 1
    , headDim := 2
    , ffnHidden := 2 }

/-! ## CLI parsing -/

def usage : String :=
  String.intercalate "\n"
    [ "TorchLean → IR → PyTorch exporter"
    , ""
    , "Usage:"
    , "  lake exe torchlean torch_ir_pytorch --arch linear > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mlp --seed 123 > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch sum > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch autoencoder > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch cnn > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch conv-mlp > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch mha-mask > exported_model.py"
    , "  lake exe torchlean torch_ir_pytorch --arch transformer > exported_model.py"
    , ""
    , "Then: python3 exported_model.py"
    ]

/-! ## Export driver -/

def emitSeq {σ τ : Spec.Shape} (className : String) (model : nn.Sequential σ τ) : IO Unit := do
  let ps := nn.paramShapes model
  let prog : TorchLean.Program Float (ps ++ [σ]) τ :=
    nn.program (model := model) (α := Float)
  let params := nn.initParams (m := model)
  let compiled ←
    match NN.Verification.TorchLean.compileForward1
        (α := Float) (paramShapes := ps) (inShape := σ) (outShape := τ)
        (model := prog) (params := params) with
    | .error e => throw <| IO.userError e
    | .ok c => pure c

  let code ←
    match Export.IRPyTorch.emit
        (g := compiled.graph) (ps := compiled.ps) (inputId := compiled.inputId) (outputId :=
          compiled.outputId)
        (opts := { className := className }) with
    | .error e => throw <| IO.userError e
    | .ok s => pure s

  IO.println code

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  let help := args.contains "--help" || args.contains "-h"
  if help then
    IO.println usage
  else
    let (seed, args) ← Common.orThrow "TorchIRPyTorch" <| CLI.takeSeed args 0
    let (arch?, rest) ← Common.orThrow "TorchIRPyTorch" <| CLI.takeFlagValueOnce args "arch"
    Common.orThrow "TorchIRPyTorch" <| CLI.requireNoArgs rest
    let arch := arch?.getD "mlp"
    if arch == "linear" then
      emitSeq (className := "TorchLeanLinear") (nn.build seed archLinear)
    else if arch == "mlp" then
      emitSeq (className := "TorchLeanMLP") (nn.build seed archMLP)
    else if arch == "sum" then
      emitSeq (className := "TorchLeanSumReduce") (nn.build seed archSumReduce)
    else if arch == "autoencoder" then
      emitSeq (className := "TorchLeanAutoencoder") (nn.build seed archAutoencoder)
    else if arch == "cnn" then
      emitSeq (className := "TorchLeanCNN") (nn.build seed archCNN)
    else if arch == "conv-mlp" then
      emitSeq (className := "TorchLeanConvMLP") (nn.build seed archConvMLP)
    else if arch == "mha" then
      emitSeq (className := "TorchLeanMHA") (nn.build seed archMHA)
    else if arch == "mha-mask" then
      emitSeq (className := "TorchLeanMHAMasked") (nn.build seed archMHAMasked)
    else if arch == "transformer" then
      emitSeq (className := "TorchLeanTransformerBlock") (nn.build seed archTransformer)
    else
      throw <| IO.userError
        (s!"unknown --arch {arch} (supported: linear | mlp | sum | autoencoder | " ++
          s!"cnn | conv-mlp | mha | mha-mask | transformer)")

end NN.Examples.Advanced.TorchIRPyTorch
