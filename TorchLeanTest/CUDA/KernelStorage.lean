/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.NN.Kernel.Tensor
public import TorchLeanTest.Harness
import TorchLean.CUDA.Kernel.Elab

/-!
# Binary64 arithmetic on binary32 buffers

`Program.runBinary32` runs a `Kernel.Program Float` on `Float32` tensors: each element is widened
exactly to binary64, the operation computes in binary64 and each result is rounded once to
binary32. We check it on the linear interpolation `a + f * (b - a)` between the ends `a` and `b` of
a segment at the weight `f`, on six segments. Two have finite ends of opposite sign whose binary32
difference overflows, so the binary32 interpolation gives `-∞`, and `NaN` at `f = 0`. Two lose
their small result to cancellation in binary32 arithmetic, one is flat and one is ordinary.

On the CPU, `Program.evalBinary32` gives the binary64 interpolation rounded to binary32, bit for
bit, including the exact `2^126`, `2^127`, `-0.25` and `0.5` of the four delicate segments. The
generated source declares `float` buffers and `double` arithmetic. On CUDA the GPU run has the bits
of the CPU run.
-/

@[expose] public section

namespace TorchLean.Test.KernelStorage

open Backend.Device TorchLean.Tensor
open scoped Kernel

/-- The linear interpolation `a + f * (b - a)` of inputs `a`, `b` and `f`, in binary64. -/
def lerp64 : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    let f ← read 2 i
    pure (a + f * (b - a)) : Except Kernel.Error Float))

/-- The linear interpolation `a + f * (b - a)` of inputs `a`, `b` and `f`, in binary32. -/
def lerp32 : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    let f ← read 2 i
    pure (a + f * (b - a)) : Except Kernel.Error Float32))

/-- The six segments `(a, b, f)` and the exact interpolation, every value a binary32 number:
opposite ends `±2^127` at `f = 1/4` and `f = 0`, the cancellation cases `4194299, -6990499` at
`3/8` and `2^23, 1 - 2^23` at `1/2`, the flat segment of `0.1` and the segment `1, 2` at `1/2`. -/
def segments : List (Float × Float × Float × Float) :=
  let top := Float.scaleB 1 127
  let tenth := (0.1 : Float32).toFloat
  [(top, -top, 0.25, Float.scaleB 1 126), (top, -top, 0, top),
    (4194299, -6990499, 0.375, -0.25), (8388608, -8388607, 0.5, 0.5),
    (tenth, tenth, 0.375, tenth), (1, 2, 0.5, 1.5)]

/-- Column `k` of `segments` as a binary32 tensor. -/
def column (k : Fin 4) : Tensor Float32 [6] :=
  Tensor.ofFn fun i =>
    let (a, b, f, exact) := segments[i.val]!
    Float.toFloat32 (if k = 0 then a else if k = 1 then b else if k = 2 then f else exact)

/-- The inputs `a`, `b` and `f`. -/
def inputs : TensorPack Float32 [[6], [6], [6]] :=
  .cons (column 0) (.cons (column 1) (.cons (column 2) .nil))

/-- The CPU checks: the binary64 interpolation rounded to binary32 bit for bit, the exact values,
the binary32 overflow and the types of the generated source. -/
def checkHost : IO (Tensor Float32 [6]) := do
  let result ← lerp64.runBinary32 inputs [6]
  let exact := column 3
  let (a, b, f) := (column 0, column 1, column 2)
  for i in List.finRange 6 do
    let wide := a[i].toFloat + f[i].toFloat * (b[i].toFloat - a[i].toFloat)
    unless result[i].toBits == wide.toFloat32.toBits && result[i].toBits == exact[i].toBits do
      throw <| IO.userError s!"binary32 storage [{i}]: {result[i]}, expected {exact[i]}"
  let narrow ← lerp32.run inputs [6]
  unless narrow[0] == -(1 / 0 : Float32) && narrow[1].isNaN do
    throw <| IO.userError s!"binary32 interpolation: {narrow[0]} and {narrow[1]}, \
      expected -inf and NaN"
  let bits := fun x => (lerp64.scalar.toBits x).toNat.toUInt64
  let source ← IO.ofExcept (CUDA.emitStored .binary32 .binary64 bits 3 lerp64.expression)
  let has (part : String) := (source.text.splitOn part).length > 1
  unless has "const float* input0" && has "const float* input2" && has "float* output" &&
      has "double " && !has "const double*" && !has "double* output" &&
      source.storage == .binary32 && source.format == .binary64 do
    throw <| IO.userError s!"binary32 storage: unexpected source\n{source.text}"
  pure result

/-- Run the checks on the CPU and, when `device` is CUDA, compare the GPU run with the CPU bits. -/
def run (device : Backend.Device) : IO Unit := do
  let expected ← checkHost
  if device == .cuda then
    let actual ← lerp64.runBinary32 inputs [6] (device := .cuda)
    for i in List.finRange 6 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError s!"binary32 storage [{i}]: CUDA bits {actual[i].toBits}, \
          CPU bits {expected[i].toBits}"
  IO.println s!"binary64 arithmetic on binary32 buffers ({repr device}): \
    rounded once, exact cancellations, float buffers: ok"

end TorchLean.Test.KernelStorage
