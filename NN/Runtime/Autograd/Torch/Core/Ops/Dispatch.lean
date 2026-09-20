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

`cudaProviders` names the providers implemented by the supplied CUDA handler. The selected capsule
is bound to the matching handler before any implementation runs. Returning `none` still means that
the operation has no implementation in this CUDA runtime; there is no per-operation CPU fallback.
-/
def dispatchCudaCapsuleOpt {α : Type} [TorchLean.Storage α] {sh : Shape} (s : EagerSession α)
    (op : NN.Backend.BackendOp) (refs : Array (Option RefIdentity))
    (cudaProviders : Array NN.Backend.Provider) (cpu : IO (TensorRef α sh))
    (cuda : NN.Backend.KernelCapsule → IO (Option (TensorRef α sh))) : IO (TensorRef α sh) := do
  s.validateRefIdentities refs
  let cpuHandler : NN.Backend.KernelHandler (TensorRef α sh) :=
    { name := "TorchLean reference CPU"
      op
      provider := .reference
      device := .cpu
      execute := fun _ => cpu }
  let cudaHandlers : Array (NN.Backend.KernelHandler (TensorRef α sh)) :=
    cudaProviders.map fun provider =>
      { name := s!"CUDA executor for {reprStr provider}"
        op
        provider
        device := .cuda
        execute := fun capsule => do
          match ← cuda capsule with
          | some result => pure result
          | none =>
              throw <| IO.userError <|
                s!"torch: cuda: `{op.name}` is unsupported by `{reprStr provider}`" }
  let result ← s.executeSelected op (#[cpuHandler] ++ cudaHandlers)
  pure { result with identity? := some (← s.currentRefIdentity) }

/--
Dispatch an eager operation implemented by the reference CPU and TorchLean native CUDA runtimes.
-/
def dispatchCudaOpt {α : Type} [TorchLean.Storage α] {sh : Shape} (s : EagerSession α)
    (op : NN.Backend.BackendOp)
    (refs : Array (Option RefIdentity)) (cpu : IO (TensorRef α sh))
    (cuda : IO (Option (TensorRef α sh))) : IO (TensorRef α sh) :=
  dispatchCudaCapsuleOpt s op refs #[.nativeCuda] cpu (fun _ => cuda)

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
