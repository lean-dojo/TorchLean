/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Session

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec TorchLean TorchLean.Tensor

namespace Internal

namespace EagerSession

/-!
## Tensor ops (eager tape wrappers)

The following definitions are the eager front-end for `Runtime.Autograd.Tape.*` primitives.
Each one:
- reads the current tape from `s.tape`,
- appends a new node/leaf via a `Tape.*` constructor,
- writes the updated tape back, and
- returns a fresh `TensorRef` pointing to the new node id.

PyTorch comparison: this is the standard eager autograd mechanism (a dynamic tape of ops).
-/

/--
Dispatch an eager operation through its selected CPU or CUDA capsule.

The selected capsule is bound to the matching handler before any implementation runs.
Returning `none` means the operation has no implementation in this CUDA runtime;
there is no per-operation CPU fallback.
-/
def execute {α : Type} [TorchLean.Storage α] {sh : Shape} (s : EagerSession α)
    (op : NN.Backend.BackendOp) (refs : Array (Option RefIdentity))
    (cpu : IO (TensorRef α sh)) (cuda : IO (Option (TensorRef α sh))) : IO (TensorRef α sh) := do
  s.validateRefIdentities refs
  let cpuHandler : NN.Backend.KernelHandler (TensorRef α sh) :=
    { name := "TorchLean reference CPU"
      op
      provider := .reference
      device := .cpu
      execute := fun _ => cpu }
  let cudaHandlers : Array (NN.Backend.KernelHandler (TensorRef α sh)) :=
    #[{ name := "LibTorch CUDA executor"
        op
        provider := .libTorch
        device := .cuda
        execute := fun _ => do
          let tape ← s.cudaTape.get
          if tape.nodes[0]?.any (fun node => (LibTorch.Buffer.format? node.value.buf).isSome) &&
              !([.add, .sub, .mul, .div, .neg, .reshape, .slice, .concat] :
                List NN.Backend.BackendOp).contains op then
            throw <| IO.userError
              s!"autograd: configured GPU arithmetic does not support `{op.name}`; use CPU"
          match ← cuda with
          | some result => pure result
          | none =>
              throw <| IO.userError <|
                s!"torch: cuda: `{op.name}` is unsupported by LibTorch" }]
  let result ← s.executeSelected op (#[cpuHandler] ++ cudaHandlers)
  pure { result with identity? := some (← s.currentRefIdentity) }

/--
Dispatch two recording actions that return node ids.

Both actions remain delayed until `execute` validates the references and selects a handler,
including any scalar transfers performed by the CUDA action.
-/
def executeRecorded {α : Type} [TorchLean.Storage α] {sh : Shape} (s : EagerSession α)
    (op : NN.Backend.BackendOp) (refs : Array (Option RefIdentity))
    (cpu cuda : IO Nat) : IO (TensorRef α sh) :=
  execute s op refs
    (do pure { id := ← cpu })
    (do pure (some { id := ← cuda }))

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
