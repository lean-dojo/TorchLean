/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.NormNum.Inv
import Mathlib.Tactic.NormNum.Pow
import Mathlib.Tactic.Positivity.Finset
public import NN.Runtime.Autograd.Torch.Core.Types
public import NN.Runtime.Autograd.Model -- shake: keep

/-!
# Training Memory Monitoring

CUDA allocator sampling and drift warnings shared by TorchLean training loops.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Memory

/-- State carried by the CUDA-memory drift detector used by sustained training runs. -/
structure State where
  firstStep : Nat
  firstFreeBytes : Nat
  warned : Bool
deriving Repr

/-- Resolve an explicit CUDA-memory cadence, or enable periodic sampling for very long runs. -/
def cadence (options : Runtime.Autograd.Torch.Config)
    (steps requested : Nat) : Nat :=
  if requested != 0 then
    requested
  else if options.usesCuda && steps >= 1000 then
    Nat.max 1 (steps / 10)
  else
    0

/--
Sample the CUDA allocator and warn when sustained free-memory loss projects exhaustion before the
requested run completes.
-/
def sample (options : Runtime.Autograd.Torch.Config)
    (watchEvery totalSteps done : Nat) (state? : Option State) : IO (Option State) := do
  if !options.usesCuda || watchEvery = 0 || (done != 0 && done % watchEvery != 0) then
    pure state?
  else
    let stats ← Runtime.Autograd.Cuda.Buffer.allocatorStatsWithToken (UInt32.ofNat done)
    IO.println s!"  cuda_mem step={done}: {stats.format}"
    let freeNow := stats.deviceFreeBytes.toNat
    match state? with
    | none =>
        pure (some { firstStep := done, firstFreeBytes := freeNow, warned := false })
    | some st =>
        if st.warned || done <= st.firstStep || st.firstFreeBytes <= freeNow then
          pure (some st)
        else
          let span := done - st.firstStep
          let drop := st.firstFreeBytes - freeNow
          let dropPerStep := drop / Nat.max 1 span
          if dropPerStep = 0 then
            pure (some st)
          else
            let projectedFailure := done + freeNow / dropPerStep
            if projectedFailure < totalSteps then
              IO.println <|
                s!"  cuda_mem warning: free device memory is dropping by ~{dropPerStep} " ++
                  s!"bytes/step; projected allocation failure before requested step count " ++
                  s!"(around step {projectedFailure})."
              pure (some { st with warned := true })
            else
              pure (some st)

end Memory
end Trainer
end TorchLean
