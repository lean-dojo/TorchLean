/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Runtime.Autograd.TorchLean.Fno1d

/-!
# Fourier Neural Operator Model Helpers (API)

Reusable configuration for the real-valued 1D FNO path used by the runnable Burgers example.

The public example is intentionally about data loading, training, evaluation, and plotting.  The
model constructor lives here so other examples can reuse the same typed FNO shape contract without
copying the wiring.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a real-valued 1D FNO on a fixed grid. -/
structure Fno1dConfig where
  grid : Nat
  width : Nat
  modes : Nat
  blocks : Nat
  seed : Nat := 0
deriving Repr

/-- Input shape for a scalar field sampled on `cfg.grid` points. -/
abbrev fno1dInShape (cfg : Fno1dConfig) : Shape :=
  NN.Tensor.Shape.Vec cfg.grid

/-- Output shape for a scalar field sampled on `cfg.grid` points. -/
abbrev fno1dOutShape (cfg : Fno1dConfig) : Shape :=
  NN.Tensor.Shape.Vec cfg.grid

/--
Real-valued FNO1D constructor.

The backend may later choose a dense-DFT or cuFFT-backed spectral implementation, but the model
shape and parameter contract are the same from the API side.
-/
def fno1dReal (cfg : Fno1dConfig)
    (hModesFit : 2 * cfg.modes ≤ cfg.grid := by decide) :
    nn.M (nn.Sequential (fno1dInShape cfg) (fno1dOutShape cfg)) :=
  pure <|
    _root_.Runtime.Autograd.TorchLean.NN.FNO1D.Real.model
      (grid := cfg.grid) (width := cfg.width) (modes := cfg.modes) (blocks := cfg.blocks)
      (seed := cfg.seed) (hModes := hModesFit)

end models
end nn

end API
end NN
