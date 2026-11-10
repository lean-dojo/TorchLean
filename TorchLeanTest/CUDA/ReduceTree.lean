/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Buffer
public import TorchLean.CUDA.LibTorch.Controls
public import TorchLean.Verify.LibTorch.TreeSum

/-!
# The deterministic CUDA sum against its tree specification

With deterministic algorithms enabled, the native flat sum `Buffer.reduceSum` returns the bits of
`LibTorch.reduceSumTree` on each fixture below: empty and one-element inputs, a cancellation the
left fold rounds away, lengths around one and two blocks, operand order across the halves of the
tree, signed zeros, overflow, opposite infinities, subnormals and NaN. These are regression checks
on fixtures, not a proof that the native sum agrees with `LibTorch.reduceSumTree` on every input.
A NaN result is compared as a NaN only.
-/

@[expose] public section

namespace TorchLean.Test
namespace CUDA
namespace ReduceTree

open LibTorch

/-- Checks that the native sum of `xs`, each entry rounded to binary32, has the bits of
`reduceSumTree` on the same entries, or that both sums are NaN. -/
def checkFixture (label : String) (xs : Array Float) : IO Unit := do
  let inputBits := xs.map fun x => x.toFloat32.toBits
  let expected := FloatLib.Floats.ExecFloat.Binary.toBits32 <|
    LibTorch.reduceSumTree (inputBits.toList.map FloatLib.Floats.ExecFloat.Binary.ofBits32)
  let input ← Buffer.ofFloatArrayIO <|
    FloatArray.mk (inputBits.map fun b => (Float32.ofBits b).toFloat)
  let sum := Buffer.reduceSum input
  let host ← Buffer.toFloatArrayIO sum
  discard <| Buffer.releaseIO sum
  discard <| Buffer.releaseIO input
  unless host.size == 1 do
    throw <| IO.userError s!"{label}: native sum returned {host.size} values"
  let native := host[0]!.toFloat32
  let expectedValue := Float32.ofBits expected
  unless native.toBits == expected || (native.isNaN && expectedValue.isNaN) do
    throw <| IO.userError
      s!"{label}: native sum bits {native.toBits} differ from the tree specification {expected}"

/-- The fixtures, run with deterministic algorithms enabled and the previous setting restored. -/
def run : IO Unit := do
  let previous ← getDeterministic
  setDeterministic true
  unless ← getDeterministic do
    throw <| IO.userError "reduce tree: deterministic algorithms could not be enabled"
  try
    let big : Float := 67108864.0
    checkFixture "empty" #[]
    checkFixture "negative zero alone" #[-0.0]
    checkFixture "cancellation the left fold loses" #[big, 1.0, -big]
    let ramp (n : Nat) : Array Float :=
      (Array.range n).map fun i => (Float.ofNat (i % 7) - 3.0) * 0.1 + Float.ofNat i * 1e-3
    for n in [255, 256, 257, 513] do
      checkFixture s!"ramp of {n}" (ramp n)
    -- Lane 0 holds `2^24` and lane 128 holds `-2^24`, among copies of `1/3`, and the first
    -- halving of the 256-lane tree adds these two, one from each half.
    let order := (Array.replicate 256 (1.0 / 3.0)).set! 0 16777216.0 |>.set! 128 (-16777216.0)
    checkFixture "operand order across the halves" order
    checkFixture "all negative zeros" (Array.replicate 300 (-0.0))
    checkFixture "overflow to infinity" #[3.0e38, 3.0e38, 1.0]
    checkFixture "subnormals" (Array.replicate 513 1.0e-45)
    checkFixture "opposite infinities" #[1.0 / 0.0, 1.0, -1.0 / 0.0]
    checkFixture "NaN input" #[1.0, 0.0 / 0.0, 2.0]
    IO.println "  reduce tree: deterministic native sums match LibTorch.reduceSumTree"
  finally
    setDeterministic previous

end ReduceTree
end CUDA
end TorchLean.Test
