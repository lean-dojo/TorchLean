/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

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
  /-- Estimated allocator headroom at the first sample, not guaranteed free memory. -/
  firstAvailableBytes : Nat
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
Sample the CUDA allocator and warn when estimated allocation headroom keeps decreasing.

The estimate adds driver-free bytes to reserved-minus-allocated bytes, then caps that sum by the
remaining configured memory-fraction budget. This avoids treating cache growth alone as a leak.
Native allocations include library workspaces that can outlive TorchLean's buffer owners.
Reserved storage is not necessarily reclaimable, samples are not atomic, and the fraction-to-byte
conversion is approximate. The forecast is a drift warning, not a guarantee of successful
allocation or a prediction of the exact step where allocation fails.
-/
def sample (options : Runtime.Autograd.Torch.Config)
    (watchEvery totalSteps done : Nat) (state? : Option State) : IO (Option State) := do
  if !options.usesCuda || watchEvery = 0 || (done != 0 && done % watchEvery != 0) then
    pure state?
  else
    let stats ← Runtime.Autograd.Cuda.Buffer.allocatorStats
    IO.println s!"  cuda_mem step={done}: {stats.format}"
    let allocated := stats.allocatedBytes.toNat
    let unusedReserved := stats.reservedBytes.toNat - allocated
    let fraction ← Runtime.Autograd.Cuda.LibTorch.getMemoryFraction
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
