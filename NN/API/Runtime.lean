/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- This module supplies public namespace exports used by downstream consumers. Import shaking
-- cannot see those downstream lookups, so keep the marked imports.
module -- shake: keep-downstream

public import NN.Backend.Report -- shake: keep
public import NN.Runtime.Autograd.Model.Program -- shake: keep
public import NN.Runtime.Autograd.Torch.Core.Types -- shake: keep
public import NN.Runtime.Autograd.Model.Functional.ShapeOps -- shake: keep
public import NN.API.Arithmetic -- shake: keep
public import NN.Runtime.Autograd.Torch.Core.TensorTransfer -- shake: keep

/-!
# Runtime Selection

Arithmetic semantics, execution mode, device, and backend-contract inspection.
-/

@[expose] public section

/-- Arithmetic semantics, execution mode, and device for a run. The record is defined with the
Torch runtime; this is the name the public API uses, so user code never spells
`Runtime.Autograd.Torch` to configure a run. -/
abbrev TorchLean.Runtime.Config := Runtime.Autograd.Torch.Config

namespace TorchLean

namespace Runtime

export Runtime.Autograd.Torch (Ops)
export Runtime.Autograd.Model (Program)
export Runtime.Autograd.Torch (TensorTransfer)
export Runtime.Autograd.Torch.TensorTransfer (readFloatTensor toFloatTensor)

open Spec TorchLean

/--
A shape-indexed handle to a value owned by a runtime program.

Unlike `Tensor`, a `ValueRef` does not contain tensor elements. It names an intermediate value in
an eager session or typed graph and is valid only in the program that created it.
-/
abbrev ValueRef (m : Type → Type) (α : Type)
    [TorchLean.Storage α] [Context α] [Monad m] [Ops (m := m) (α := α)]
    (shape : Shape) :=
  Runtime.Autograd.Model.RefTy (m := m) (α := α) shape

/-- Apply an affine map to the final axis, independently over every index in `batchShape`. -/
def linear {α : Type} [TorchLean.Storage α] [Context α]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (batchShape : Shape := []) {inputWidth outputWidth : Nat}
    (weight : ValueRef (m := m) (α := α) [outputWidth, inputWidth])
    (bias : ValueRef (m := m) (α := α) [outputWidth])
    (input : ValueRef (m := m) (α := α) (batchShape.appendDim inputWidth)) :
    m (ValueRef (m := m) (α := α) (batchShape.appendDim outputWidth)) := by
  let input' : ValueRef (m := m) (α := α) (batchShape.concat [inputWidth]) := by
    simpa only [Shape.appendDim_eq_concat] using input
  simpa only [Shape.appendDim_eq_concat] using
    Runtime.Autograd.Model.linearEach
      (m := m) (α := α) (leadingShape := batchShape) weight bias input'

/-!
## Operation-Polymorphic Programs

These operations build or execute runtime programs. They consume shape-indexed references rather
than materialized `Tensor` values, so they live under `Runtime` instead of the pure tensor API.
Ordinary models should use `nn` and `Trainer`; this lower-level surface is useful for custom losses,
verification programs, and graph-lowering tools.
-/

export Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape reduceSum reduceMean select indexSelect scatterAdd
   matmul
   relu silu gelu sigmoid tanh softplus exp sin cos log inv safeLog
   sum flatten mseLoss)
export Runtime.Autograd.Model
  (mapLeading maxPool avgPool smoothMaxPool layerNorm multiHeadAttention
   multiHeadAttentionOutputBias conv convTranspose)
export Runtime.Autograd.Model.F (permute softmax logSoftmax)

export Runtime.Autograd.Torch (ExecutionMode)

namespace ExecutionMode

export Runtime.Autograd.Torch.ExecutionMode (eager typedGraph)

/-- Stable command-line spelling of an execution mode. -/
def cliName : ExecutionMode → String
  | .eager => "eager"
  | .typedGraph => "typed-graph"

/-- Parse the stable command-line spelling of an execution mode. -/
def parse (value : String) : Except String ExecutionMode :=
  match value with
  | "eager" => pure .eager
  | "typed-graph" => pure .typedGraph
  | _ => throw s!"unknown execution mode `{value}` (supported: eager | typed-graph)"

end ExecutionMode

export NN.Backend (Device)

namespace Device

export NN.Backend.Device (cpu cuda rocm metal wasm tpu trainium custom external)

/--
Parse a public device selector. `auto` chooses the portable CPU runtime; every other value is
validated against the devices known to the backend registry.
-/
def parse (value : String) : Except String Device :=
  if value == "auto" then pure .cpu else NN.Backend.Device.parse value

end Device

namespace BackendContracts

/-- Plan operations under the runtime-selected backend-contract profile. -/
def planReport (config : Config) (ops : Array NN.Backend.BackendOp) : Except String String := do
  let profile ← config.effectiveBackendProfile
  profile.planReport ops

end BackendContracts

end Runtime


end TorchLean
