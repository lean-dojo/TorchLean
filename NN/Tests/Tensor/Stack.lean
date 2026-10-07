/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Tensor

/-!
# Native stacking agrees with coordinate construction

`stack` uses the unproved `stackFast` implementation in compiled code. Compare it with a separately
constructed coordinate reference on packed Float and boxed Nat storage, including empty outputs.
Float comparisons use bits so signed zeros and NaN payloads cannot hide a copy error.
-/

public section

namespace NN.Tests.Tensor.Stack

open TorchLean

/-- Compare the compiled stack with its coordinate reference, without converting tensor values. -/
def run : IO Unit := do
  let components : Fin 5 → Tensor Float [2, 3] := fun i =>
    Tensor.generateFlat [2, 3] fun index =>
      if index % 3 = 0 then
        if index = 0 then -0.0 else Float.ofBits 0x7ff8000000000017
      else (i.val * 10 + index).toFloat
  let actual := TorchLean.Tensor.Internal.Rep.stack components
  let expected : Tensor Float [5, 2, 3] :=
    TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => components coordinate.1 coordinate.2
  unless Tensor.foldl (· && ·) true
      (TorchLean.Tensor.Internal.Rep.zipWith (fun x y => x.toBits == y.toBits)
        actual expected) do
    throw <| IO.userError "stack: Float fast path differs from coordinate construction"
  let boxed : Fin 4 → Tensor Nat [3] := fun i =>
    Tensor.ofFn fun j => i.val * 100 + j.val
  let boxedActual := TorchLean.Tensor.Internal.Rep.stack boxed
  let boxedExpected : Tensor Nat [4, 3] :=
    TorchLean.Tensor.Internal.Rep.ofFn fun coordinate => boxed coordinate.1 coordinate.2
  unless Tensor.foldl (· && ·) true
      (TorchLean.Tensor.Internal.Rep.zipWith (· == ·) boxedActual boxedExpected) do
    throw <| IO.userError "stack: Nat fast path differs from coordinate construction"
  let emptyTail := TorchLean.Tensor.Internal.Rep.stack
    (fun (_ : Fin 6) => Tensor.full [0] (0 : Float))
  let emptyHead := TorchLean.Tensor.Internal.Rep.stack
    (fun (i : Fin 0) => (nomatch i : Tensor Float [3]))
  unless Tensor.foldl (fun count _ => count + 1) (0 : Nat) emptyTail == 0 &&
      Tensor.foldl (fun count _ => count + 1) (0 : Nat) emptyHead == 0 do
    throw <| IO.userError "stack: empty output has entries"
  IO.println "tensor stack: ok"

end NN.Tests.Tensor.Stack
