/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Buffer-Update Detection Tests

Checks that composition wrappers report mutable buffers exactly. Stateless residual and branch
models must not trigger the reference evaluator after an accelerated training step, while wrappers
around BatchNorm must preserve its running-statistics update.
-/

@[expose] public section

namespace NN.Tests.API.BufferUpdates

open TorchLean

def expect (tag : String) (expected actual : Bool) : IO Unit := do
  unless actual == expected do
    throw <| IO.userError
      s!"buffer-update detection failed: {tag} (expected {expected}, got {actual})"

def hasUpdates {σ τ : Spec.Shape} (model : nn.Sequential σ τ) : Bool :=
  Runtime.Autograd.Model.Layers.Seq.hasBufferUpdates model

def affine : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)

def normalized : nn.Sequential [1, 1, 1] [1, 1, 1] :=
  nn.Sequential.fromLayer <| Runtime.Autograd.Model.Layers.batchNorm 1 1 [1] (by decide)

def run : IO Unit := do
  expect "affine" false (hasUpdates affine)
  expect "stateless residual" false (hasUpdates (nn.residual affine))
  expect "stateless branches" false (hasUpdates (nn.addBranches affine affine))
  expect "BatchNorm" true (hasUpdates normalized)
  expect "stateful residual" true (hasUpdates (nn.residual normalized))
  expect "stateful branch" true (hasUpdates (nn.addBranches normalized normalized))

end NN.Tests.API.BufferUpdates
