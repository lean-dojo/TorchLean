/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.Runtime.Autograd.Torch.Core.Session.State

/-!
# Eager Backend Dispatch

Cache each accepted backend capsule, then require an executable handler for that exact provider
and device. Reports describe the handler we actually run.
-/

public section

namespace Runtime.Autograd.Torch.Internal

open Spec TorchLean
namespace EagerSession

/--
Select the accepted capsule for one operation.

The profile stays fixed for the session, so cache the first choice for each operation.
`executeSelected` binds that contract to a handler for the same provider and device.
-/
def selectedCapsule {α : Type} [Storage α]
    (s : EagerSession α) (op : NN.Backend.BackendOp) :
    IO NN.Backend.KernelCapsule := do
  let selected ← s.selectedBackends.get
  match selected.find? (fun accepted => accepted.op == op) with
  | some accepted => return accepted.capsule
  | none => pure ()
  let planned ←
    match s.options.planBackendOp op with
    | .ok planned => pure planned
    | .error msg => throw <| IO.userError s!"torch: kernel selection failed for `{op.name}`: {msg}"
  let capsule := planned.capsule
  match capsule.validateEagerRequest op s.options.device with
  | .ok () => pure ()
  | .error msg =>
    throw <| IO.userError s!"torch: invalid backend selection during eager execution: {msg}"
  s.selectedBackends.set (selected.push planned)
  if s.options.showBackend then
    let row := NN.Backend.KernelAudit.ofPlannedKernel
      { op := planned.op, capsule := planned.capsule }
    for line in row.detailedReportLines do
      IO.println line
  pure capsule

/--
Execute an operation through a handler that matches the selected capsule.

Fail if the chosen provider is not linked into this build. `ExecutableKernel` certifies dispatch
identity; the selected capsule states the numerical evidence.
-/
def executeSelected {α β : Type} [Storage α]
    (s : EagerSession α) (op : NN.Backend.BackendOp)
    (handlers : Array (NN.Backend.KernelHandler β)) : IO β := do
  let capsule ← s.selectedCapsule op
  match handlers.find? (fun handler => handler.matchesCapsule capsule) with
  | none =>
      let available := String.intercalate ", " <|
        (handlers.map fun handler =>
          s!"{handler.name} ({reprStr handler.provider}/{handler.device.cliName})").toList
      throw <| IO.userError <|
        s!"torch: `{op.name}` selected capsule `{capsule.name}` " ++
          s!"({reprStr capsule.provider}/{capsule.device.cliName}), but no matching executable " ++
          s!"handler is linked" ++
          (if available.isEmpty then "" else s!"; available handlers: {available}")
  | some handler =>
      match capsule.bind handler with
      | .ok kernel => kernel.run
      | .error msg =>
          throw <| IO.userError s!"torch: invalid executable backend binding: {msg}"

/--
Execute an operation implemented by the reference CPU and LibTorch CUDA runtimes.

The selected handler computes tensor values. TorchLean retains tape ownership in either case.
-/
def executeReferenceOrLibTorch {α β : Type} [Storage α] (s : EagerSession α)
    (op : NN.Backend.BackendOp) (cpu cuda : IO β) : IO β :=
  s.executeSelected op
    #[({ name := "TorchLean reference CPU"
         op
         provider := .reference
         device := .cpu
         execute := fun _ => cpu } : NN.Backend.KernelHandler β),
      ({ name := "LibTorch CUDA"
         op
         provider := .libTorch
         device := .cuda
         execute := fun _ => cuda } : NN.Backend.KernelHandler β)]

/-- Accepted capsules actually selected by this eager session. -/
def backendSelections {α : Type} [Storage α]
    (s : EagerSession α) : IO (Array NN.Backend.AcceptedKernel) :=
  s.selectedBackends.get

end EagerSession

end Runtime.Autograd.Torch.Internal
