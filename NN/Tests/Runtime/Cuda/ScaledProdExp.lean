/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Buffer
public import Std

/-!
# Scaled Product Exponential Composition Parity

`Buffer.scaledProdExp x y c` computes `exp((c · x) · y)`. Its native implementation casts `c`
to float32, then applies two ATen multiplications followed by the exponential.

This test checks bit-identical results against `exp(((full c) · x) · y)` on finite fixtures.
Both expressions use the same left association. The comparison checks the scalar conversion
and composition through the buffer API, without assuming a particular device kernel.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace ScaledProdExp

open Runtime.Autograd.Cuda

def run : IO Unit := do
  IO.println "=== scaledProdExp composition parity ==="
  -- Varied, signed, finite fixtures, kept in a range where `exp` stays finite.
  let xs : FloatArray := FloatArray.mk
    #[0.10, -0.20, 0.35, -0.50, 0.75, -0.90, 0.00, 1.00, -1.00, 0.42]
  let ys : FloatArray := FloatArray.mk
    #[0.90, -0.75, 0.50, -0.30, 0.15, -0.05, 1.00, -1.00, 0.25, -0.60]
  let n : UInt32 := xs.size.toUInt32
  let x := Buffer.ofFloatArray xs
  let y := Buffer.ofFloatArray ys
  -- Scalars spanning sign and magnitude, including the identity `c = 0`.
  for c in (#[-2.0, 0.5, 3.25, -0.125, 1.0, 0.0] : Array Float) do
    let actual := Buffer.scaledProdExp x y c
    let composed := Buffer.exp (Buffer.mul (Buffer.mul (Buffer.full n c) x) y)
    let af := Buffer.toFloatArray actual
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
      throw <| IO.userError <|
        s!"scaledProdExp c={c}: {mism}/{af.size} elements differ from composed exp((c·x)·y) " ++
          s!"(max |Δ|={maxDiff})"
  IO.println "  scaledProdExp bit-identical to composed exp((c·x)·y) over all fixtures ✓"

end ScaledProdExp
end Cuda
end Tests
