/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.LibTorch.Controls
public import TorchLean.NN.Session.Config

/-!
# Training Memory Monitoring

CUDA allocator sampling and drift warnings shared by the TorchLean training loops. `cadence` picks
how often to sample, and `sample` reads the allocator, prints its statistics and warns when the
estimated headroom is on course to run out before the last step.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Memory

/-- State carried by the CUDA-memory drift detector used by sustained training runs. -/
structure State where
  /-- Completed update count at the first sample. -/
  firstStep : Nat
  /-- Allocator headroom estimated at the first sample, in bytes. -/
  firstAvailableBytes : Nat
  /-- Whether the depletion warning has been printed, which stops further warnings. -/
  warned : Bool
deriving Repr

/-- CUDA-memory sampling cadence: `requested` when it is nonzero, otherwise every `steps / 10`
updates for a CUDA run of at least 1000 steps, and `0` (no sampling) for any other run. -/
def cadence (options : Session.Config)
    (steps requested : Nat) : Nat :=
  if requested != 0 then
    requested
  else if options.usesCUDA && steps >= 1000 then
    Nat.max 1 (steps / 10)
  else
    0

/--
Sample the CUDA allocator every `watchEvery` updates and warn once if the estimated allocation
headroom is falling fast enough to run out before step `totalSteps`.

The estimate adds driver-free bytes to reserved-minus-allocated bytes and caps the sum by the
unallocated part of the configured memory-fraction budget, so growth of the allocator cache alone
does not look like a leak. It is rough: native allocations include library workspaces that can
outlive TorchLean's buffer owners, reserved storage is not always reclaimable, samples are not
atomic and the fraction-to-byte conversion is approximate. The warning extrapolates a linear trend
from the first sample, so the step it names is an estimate.
-/
def sample (options : Session.Config)
    (watchEvery totalSteps done : Nat) (state? : Option State) : IO (Option State) := do
  if !options.usesCUDA || watchEvery = 0 || (done != 0 && done % watchEvery != 0) then
    pure state?
  else
    let stats ← LibTorch.Buffer.Memory.read
    IO.println s!"  cuda_mem step={done}: {stats.format}"
    let allocated := stats.allocatedBytes.toNat
    let unusedReserved := stats.reservedBytes.toNat - allocated
    let fraction ← LibTorch.getMemoryFraction
    let total := stats.deviceTotalBytes.toNat
    let budget := Nat.min total ((Float.ofNat total * fraction).toUInt64.toNat)
    let availableNow := Nat.min
      (stats.deviceFreeBytes.toNat + unusedReserved) (budget - allocated)
    match state? with
    | none =>
        pure (some { firstStep := done, firstAvailableBytes := availableNow, warned := false })
    | some st =>
        if st.warned || done <= st.firstStep || st.firstAvailableBytes <= availableNow then
          pure (some st)
        else
          let span := done - st.firstStep
          let drop := st.firstAvailableBytes - availableNow
          let dropPerStep := drop / Nat.max 1 span
          if dropPerStep = 0 then
            pure (some st)
          else
            let projectedDepletion := done + availableNow / dropPerStep
            if projectedDepletion < totalSteps then
              IO.println <|
                "  cuda_mem warning: estimated allocator headroom is dropping by " ++
                  s!"~{dropPerStep} bytes/step; continuing this trend would consume " ++
                  s!"the estimate around step {projectedDepletion}. Actual allocation " ++
                  "failures depend on fragmentation and other device activity."
              pure (some { st with warned := true })
            else
              pure (some st)

end Memory
end Trainer
end TorchLean
