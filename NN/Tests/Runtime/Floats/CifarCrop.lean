/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Models.Common.RealData
public import NN.Tests.Runtime.Floats.Utils

/-!
# Packed CIFAR Crop Check

Checks that the model-example crop preserves channel-first row-major layout while selecting the
top-left spatial window.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace CifarCrop

open TorchLean
open NN.Examples.Models
open Tests.Utils
open Tests.Floats.Utils

def input : Tensor Float [1, RealData.cifarChannels, RealData.cifarHeight, RealData.cifarWidth] :=
  Tensor.generateFlat [1, RealData.cifarChannels, RealData.cifarHeight, RealData.cifarWidth]
    Float.ofNat

def expected : Array Float :=
  #[0.0, 1.0, 2.0, 32.0, 33.0, 34.0,
    1024.0, 1025.0, 1026.0, 1056.0, 1057.0, 1058.0,
    2048.0, 2049.0, 2050.0, 2080.0, 2081.0, 2082.0]

def run : IO Unit := do
  IO.println "cifar_crop: begin"
  let cropped ← IO.ofExcept <| RealData.cropCifarImages 1 2 3 input
  let actual :=
    Tensor.to cropped (Array Float)
  if actual.size != expected.size then
    throw <| IO.userError
      s!"cifar_crop: expected {expected.size} values, got {actual.size}"
  for index in List.finRange expected.size do
    assertApprox s!"cifar_crop[{index.val}]" actual[index.val]! expected[index] 0.0
  IO.println "cifar_crop: ok"

end CifarCrop
end Floats
end Tests
