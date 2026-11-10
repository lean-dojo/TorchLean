/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.CUDA.LibTorch.Kernels
public import TorchLean.NN.LibTorch.Tape
public import TorchLean.Tensor.Datatype

/-!
# Axis moves for the CUDA row kernels

Adjacent swaps that move an axis to the front of a CUDA buffer and back, for the entries of
`TorchLean.LibTorch.Kernel.ofOp` and the graph executor that read or write whole rows along that
axis.
-/

@[expose] public section

namespace TorchLean
namespace LibTorch

open TorchLean.Tensor

namespace Tape

/-!
## Arbitrary-axis bounded indexing
-/

namespace Indexing

/-- An adjacent-swap depth paired with its checked CUDA representation. -/
abbrev SwapStep := Nat × UInt32

/-- Adjacent swaps that move `axis` to the outermost position. -/
def moveAxisToFrontDepths (axis : Nat) : List Nat :=
  (List.range axis).reverse

/-- Check each adjacent-swap depth before it crosses the CUDA ABI. -/
def checkedSwapSteps (depths : List Nat) : Result (List SwapStep) :=
  depths.mapM fun depth => do
    let some depth32 := Bytes.toUInt32? depth | throw SomeBuffer.tooLarge
    pure (depth, depth32)

/-- Apply swaps to an owned buffer, releasing each consumed intermediate. -/
def permuteOwned (x : Buffer) (s : Shape) : List SwapStep → Buffer
  | [] => x
  | (depth, depth32) :: steps =>
      let y := Buffer.swapAdjacentAtDepth x s.toArray depth32
      permuteOwned (Buffer.releaseThen x y) (s.swapAdjacentAtDepth depth) steps

/-- Apply swaps to a borrowed buffer, owning only the buffers created by the swaps. -/
def permuteBorrowed (x : Buffer) (s : Shape) : List SwapStep → Buffer
  | [] => x
  | (depth, depth32) :: steps =>
      let y := Buffer.swapAdjacentAtDepth x s.toArray depth32
      permuteOwned y (s.swapAdjacentAtDepth depth) steps

/-- Release a permuted buffer exactly when `permuteBorrowed` allocated it. -/
def releasePermutedThen (steps : List SwapStep) (permuted keep : Buffer) : Buffer :=
  match steps with
  | [] => keep
  | _ :: _ => Buffer.releaseThen permuted keep

end Indexing

end Tape

end LibTorch
end TorchLean
