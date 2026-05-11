/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# CNN Model Helpers (API)

Small config-style CNN constructors for runnable examples.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a compact convolutional image classifier. -/
structure CnnConfig where
  batch : Nat
  inC : Nat
  inH : Nat
  inW : Nat
  outDim : Nat
  conv : nn.Conv := { outC := 16, kH := 3, kW := 3, stride := 2, padding := 1 }
  pool : nn.MaxPool := { kH := 2, kW := 2, stride := 2 }

def CnnConfig.outH1 (cfg : CnnConfig) : Nat :=
  (cfg.inH + 2 * cfg.conv.padding - cfg.conv.kH) / cfg.conv.stride + 1

def CnnConfig.outW1 (cfg : CnnConfig) : Nat :=
  (cfg.inW + 2 * cfg.conv.padding - cfg.conv.kW) / cfg.conv.stride + 1

def CnnConfig.outH2 (cfg : CnnConfig) : Nat :=
  (cfg.outH1 - cfg.pool.kH) / cfg.pool.stride + 1

def CnnConfig.outW2 (cfg : CnnConfig) : Nat :=
  (cfg.outW1 - cfg.pool.kW) / cfg.pool.stride + 1

def CnnConfig.featSize (cfg : CnnConfig) : Nat :=
  Spec.Shape.size (NN.Tensor.Shape.CHW cfg.conv.outC cfg.outH2 cfg.outW2)

abbrev cnnInShape (cfg : CnnConfig) : Shape :=
  NN.Tensor.Shape.NCHW cfg.batch cfg.inC cfg.inH cfg.inW

abbrev cnnOutShape (cfg : CnnConfig) : Shape :=
  NN.Tensor.Shape.Mat cfg.batch cfg.outDim

/--
Build a compact CNN classifier:
`conv -> relu -> maxPool -> flatten -> linear`.
-/
def cnn (cfg : CnnConfig)
    (h_inC : cfg.inC ≠ 0 := by decide)
    (h_kH : cfg.conv.kH ≠ 0 := by decide)
    (h_kW : cfg.conv.kW ≠ 0 := by decide)
    (h_poolKH : cfg.pool.kH ≠ 0 := by decide)
    (h_poolKW : cfg.pool.kW ≠ 0 := by decide) :
    nn.M (nn.Sequential (cnnInShape cfg) (cnnOutShape cfg)) :=
  letI : NeZero cfg.inC := ⟨h_inC⟩
  letI : NeZero cfg.conv.kH := ⟨h_kH⟩
  letI : NeZero cfg.conv.kW := ⟨h_kW⟩
  letI : NeZero cfg.pool.kH := ⟨h_poolKH⟩
  letI : NeZero cfg.pool.kW := ⟨h_poolKW⟩
  nn.sequential![
    nn.conv cfg.conv,
    nn.relu,
    nn.maxPool cfg.pool,
    nn.flattenBatch,
    nn.linear cfg.featSize cfg.outDim (pfx := NN.Tensor.Shape.Vec cfg.batch)
  ]

end models
end nn

end API
end NN
