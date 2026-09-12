/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.SelfSupervised.BlockMask
public import NN.Tensor

/-!
# Block-Mask API Tests

These checks exercise the runtime coordinate plumbing behind the general block-mask theorem. They
cover a one-dimensional signal and a three-dimensional volume so a later implementation change
cannot accidentally restore image-specific rank assumptions.
-/

@[expose] public section

namespace NN.Tests.API.SelfSupervised.BlockMask

open TorchLean

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"block-mask check failed: {tag}"

def signal : TorchLean.Tensor Float [8] :=
  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def volume : TorchLean.Tensor Float [2, 2, 2] :=
  [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]

def batchedSignal : TorchLean.Tensor Float [1, 4] :=
  [[1.0, 2.0, 3.0, 4.0]]

def run : IO Unit := do
  let signalMasked := TorchLean.ssl.BlockMask.apply signal
    ([some 2] : TorchLean.Tensor (Option Nat) [1]) 2 0
  expect "signal hidden block"
    (TorchLean.Tensor.at? signalMasked #[0] == some 0.0)
  expect "signal visible block"
    (TorchLean.Tensor.at? signalMasked #[2] == some 3.0)

  let volumeMasked := TorchLean.ssl.BlockMask.apply volume
    ([some 1, some 1, some 1] : TorchLean.Tensor (Option Nat) [3]) 2 0
  expect "volume hidden block"
    (TorchLean.Tensor.at? volumeMasked #[0, 0, 0] == some 0.0)
  expect "volume visible block"
    (TorchLean.Tensor.at? volumeMasked #[0, 0, 1] == some 2.0)

  let blocks : TorchLean.Tensor (Option Nat) [1] := [some 2]
  match TorchLean.ssl.BlockMAE.sample [1] (dataShape := [4]) 4 blocks 2 0 batchedSignal with
  | .error message =>
      throw <| IO.userError s!"valid block-MAE sample was rejected: {message}"
  | .ok sample =>
      expect "block-MAE masks the input"
        (TorchLean.Tensor.at? sample.input #[0, 0] == some 0.0 &&
          TorchLean.Tensor.at? sample.input #[0, 2] == some 3.0)
      expect "block-MAE preserves the reconstruction target"
        (TorchLean.Tensor.at? sample.target #[0, 0] == some 1.0 &&
          TorchLean.Tensor.at? sample.target #[0, 3] == some 4.0)

  match TorchLean.ssl.BlockMAE.sample [1] (dataShape := [4]) 5 blocks 2 0 batchedSignal with
  | .ok _ =>
      throw <| IO.userError "oversized block-MAE reconstruction was accepted"
  | .error _ =>
      pure ()

end NN.Tests.API.SelfSupervised.BlockMask
