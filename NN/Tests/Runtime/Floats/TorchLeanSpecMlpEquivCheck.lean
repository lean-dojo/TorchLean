/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Mlp
public import NN.Tests.Runtime.Floats.Utils
public import NN.API.Seeded

/-!
# TorchLeanSpecMlpEquivCheck

Runtime check: TorchLean MLP forward agrees with Spec MLP forward.

Goal: make the "Spec vs TorchLean" relationship concrete with an executable check:
given the *same* initialized parameters, both front-ends produce the same output tensor.

This is not a performance test; it is a regression guard for:
- parameter ordering conventions (W,b pairs),
- vector and matrix tensor layout conventions,
- and the meaning of `Linear → ReLU → Linear`.
-/

@[expose] public section


open Spec TorchLean
open TorchLean TorchLean.Tensor
open Tests.Utils
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanSpecMLPEquivCheck

def run : IO Unit := do
  IO.println "torchlean_spec_mlp_equiv_check: begin"

  let inputWidth : Nat := 2
  let hiddenWidth : Nat := 3
  let outputWidth : Nat := 1
  let xShape : Shape := [inputWidth]
  let yShape : Shape := [outputWidth]

  -- TorchLean MLP built through the public deterministic builder.
  let model := nn.build 0 <|
    nn.mlp inputWidth outputWidth { hiddenWidths := [hiddenWidth] }

  -- One input vector.
  let x : Tensor Float xShape :=
    Tensor.ofFn fun i => [0.5, 0.8][i.val]!

  -- Extract TorchLean parameters and reinterpret them as Spec `LinearSpec`s.
  let ps := Runtime.Autograd.Model.Layers.Seq.initState (m := model)
  let (w1, b1, w2, b2) :=
    match ps with
    | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) => (w1, b1, w2, b2)

  let l1 : Spec.LinearSpec Float inputWidth hiddenWidth := { weights := w1, bias := b1 }
  let l2 : Spec.LinearSpec Float hiddenWidth outputWidth := { weights := w2, bias := b2 }

  -- Spec forward reference.
  let ySpec : Tensor Float yShape := Examples.mlpForward (α := Float) l1 l2 x

  -- TorchLean forward reference (typed graph evaluation of the TorchLean forward program).
  let graph ← Runtime.Autograd.Model.Autodiff.lowerToTypedGraph (α := Float)
    (paramShapes := Runtime.Autograd.Model.Layers.Seq.stateShapes model)
    (inputShapes := [xShape]) (τ := yShape)
    (fun {β} _ _ => Runtime.Autograd.Model.Layers.Seq.forward model (α := β))

  let args : TorchLean.TensorPack Float
      (Runtime.Autograd.Model.Layers.Seq.stateShapes model ++ [xShape]) :=
    .cons w1 (.cons b1 (.cons w2 (.cons b2 (.cons x .nil))))

  let yTorch : Tensor Float yShape :=
    Runtime.Autograd.Torch.TypedGraph.forward graph args

  -- Since `outputWidth = 1`, check the single coordinate. (Kept structured so it scales to
  -- `outputWidth >
  -- 1`.)
  for i in List.finRange outputWidth do
    assertApprox s!"mlp forward[{i.val}] spec/torchlean" (vecVal ySpec i) (vecVal yTorch i) 1e-6

  IO.println "torchlean_spec_mlp_equiv_check: ok"

end TorchLeanSpecMLPEquivCheck
end Floats
end Tests
