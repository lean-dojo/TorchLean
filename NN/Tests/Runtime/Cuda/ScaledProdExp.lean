/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import Std

/-!
# CUDA Kernel Coverage: scaledProdExp (fused scaled product exponential)

`Buffer.scaledProdExp x y c` computes `exp((c · x) · y)` in a single fused device kernel. This test
pins its defining property: element for element it must be **bit-identical** to the composed form built
from the elementwise `full` / `mul` / `exp` kernels — `exp(((full c) · x) · y)`, the same
left-association and the same `expf`. A silent divergence (a fast-math `__expf`, a reassociated
product, a lost float32 cast) is a regression this catches.

Like the rest of this suite it runs on the CPU stub (`lake build`) and on the GPU (`-K cuda`) alike.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ScaledProdExp

open Runtime.Autograd.Cuda

def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: scaledProdExp (fused vs composed) ==="
  -- Varied, signed, finite fixtures, kept in a range where `exp` stays finite.
  let xs : FloatArray := FloatArray.mk #[0.10, -0.20, 0.35, -0.50, 0.75, -0.90, 0.00, 1.00, -1.00, 0.42]
  let ys : FloatArray := FloatArray.mk #[0.90, -0.75, 0.50, -0.30, 0.15, -0.05, 1.00, -1.00, 0.25, -0.60]
  let n : UInt32 := xs.size.toUInt32
  let x := Buffer.ofFloatArray xs
  let y := Buffer.ofFloatArray ys
  -- Scalars spanning sign and magnitude, including the identity `c = 0`.
  for c in (#[-2.0, 0.5, 3.25, -0.125, 1.0, 0.0] : Array Float) do
    let fused    := Buffer.scaledProdExp x y c
    let composed := Buffer.exp (Buffer.mul (Buffer.mul (Buffer.full n c) x) y)
    let af := Buffer.toFloatArray fused
    let ac := Buffer.toFloatArray composed
    if af.size != ac.size then
      throw <| IO.userError s!"scaledProdExp c={c}: size mismatch ({af.size} vs {ac.size})"
    let mut mism : Nat := 0
    let mut maxDiff : Float := 0.0
    for i in [:af.size] do
      let vf := af.get! i
      let vc := ac.get! i
      -- Bit-level comparison: `toBits` distinguishes results that `==` would call equal.
      if vf.toBits != vc.toBits then mism := mism + 1
      let d := Float.abs (vf - vc)
      if d > maxDiff then maxDiff := d
    if mism != 0 then
      throw <| IO.userError
        s!"scaledProdExp c={c}: {mism}/{af.size} elements differ from composed exp((c·x)·y) (max |Δ|={maxDiff})"
  IO.println "  fused scaledProdExp bit-identical to composed exp((c·x)·y) over all fixtures ✓"

end ScaledProdExp
end Cuda
end Tests
