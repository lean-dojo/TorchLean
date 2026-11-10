/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public meta import TorchLean.Tensor.Literal
public import TorchLean.CUDA.Kernel.Gradient
public import TorchLean.NN.Kernel.Tensor
public import TorchLean.Tensor.Literal
public import TorchLeanTest.Harness
import TorchLean.CUDA.Kernel.Elab

/-!
# Tangent and gradient kernels

`Program.tangent` and `Program.gradient` differentiate a custom operation written in Lean. The
derivative kernels are expressions of the kernel language, so they run on the CPU through
`Program.eval` and on the GPU through the CUDA emitter of the operation itself. We check four
operations a user would write, each in binary32 and in binary64: a rational function with a branch
applied to every entry, a quotient of two inputs, a matrix-vector product written as a recursion
along each row, and the product of each row, a recursion that multiplies its accumulator.

On the CPU the tangents agree with the analytic derivatives to within `1e-5` away from the kink
`x = 0` of the rational function, and each gradient pairs with a direction as the tangent does
(`⟪gradient cot, dx⟫ = ⟪cot, tangent dx⟫`). At the kink, where the left derivative is `-1` and the
right derivative is `1`, the tangent is `1`, the derivative of the branch `x / (x + 1)` that `0`
selects. With these cotangents, none of them zero, a gradient entry of the entrywise function is
the cotangent times the tangent at that entry, bit for bit. A zero cotangent can give `+0` where
that product is `-0`, since the gradient adds the zero terms of the other outputs. The inputs are
finite: an infinite or NaN partial derivative makes every gradient entry NaN, as the kernel README
explains. On small integers every operation of the matrix-vector product and of the row products
is exact, and so are their tangents and gradients (`Wᵀ cot` and `cot vᵀ` for the product with a
vector, `cot` times the product of the other entries of the row for the row products). On CUDA
every derivative kernel has the bits of the CPU run.
-/

@[expose] public section

namespace TorchLean.Test.KernelGradient

open Backend.Device TorchLean.Tensor
open scoped Kernel

/-- A scalar format of the checks: its name, its conversions from and to binary64, and its bits. -/
structure CheckFormat (α : Type) where
  /-- The name of the format in messages. -/
  name : String
  /-- The value of the format nearest to a binary64 number. -/
  ofFloat : Float → α
  /-- The binary64 value of a number of the format. -/
  toFloat : α → Float
  /-- The bits of a number of the format. -/
  toBits : α → UInt64

/-- Binary32, through `Float.toFloat32` and `Float32.toFloat`. -/
def CheckFormat.binary32 : CheckFormat Float32 :=
  ⟨"binary32", Float.toFloat32, Float32.toFloat, fun x => x.toBits.toUInt64⟩

/-- Binary64, the format of `Float`. -/
def CheckFormat.binary64 : CheckFormat Float := ⟨"binary64", id, id, Float.toBits⟩

/-- The entrywise function `x² - x` below zero and `x / (x + 1)` from zero. -/
def rational : Kernel.Program Float32 :=
  Program.of (fun (x : Float32) => if x < 0 then x * x - x else x / (x + 1))

/-- The entrywise function `x² - x` below zero and `x / (x + 1)` from zero, in binary64. -/
def rational64 : Kernel.Program Float :=
  Program.of (fun (x : Float) => if x < 0 then x * x - x else x / (x + 1))

/-- The product of two inputs divided by one plus the square of the first, `a b / (a² + 1)`. -/
def quotient : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    pure (a * b / (a * a + 1)) : Except Kernel.Error Float32))

/-- The product of two inputs divided by one plus the square of the first, in binary64. -/
def quotient64 : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (i : UInt64) => (do
    let a ← read 0 i
    let b ← read 1 i
    pure (a * b / (a * a + 1)) : Except Kernel.Error Float))

/-- The dot product of the first `n` entries of row `row` of the `4 × 3` matrix in input zero with
the vector in input one. -/
def rowDot (read : Kernel.Reader Float32) (row : UInt64) : Nat → Except Kernel.Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← rowDot read row n
      let w ← read 0 (row * 3 + n.toUInt64)
      let v ← read 1 n.toUInt64
      pure (previous + w * v)

/-- The dot product of the first `n` entries of row `row` of the `4 × 3` matrix in input zero with
the vector in input one, in binary64. -/
def rowDot64 (read : Kernel.Reader Float) (row : UInt64) : Nat → Except Kernel.Error Float
  | 0 => pure 0
  | n + 1 => do
      let previous ← rowDot64 read row n
      let w ← read 0 (row * 3 + n.toUInt64)
      let v ← read 1 n.toUInt64
      pure (previous + w * v)

/-- The product of the `4 × 3` matrix in input zero with the vector in input one. -/
def matVec : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (row : UInt64) => rowDot read row 3)

/-- The product of the `4 × 3` matrix in input zero with the vector in input one, in binary64. -/
def matVec64 : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (row : UInt64) => rowDot64 read row 3)

/-- The product of the first `n` entries of row `row` of the `4 × 3` matrix in input zero. -/
def rowProduct (read : Kernel.Reader Float32) (row : UInt64) : Nat → Except Kernel.Error Float32
  | 0 => pure 1
  | n + 1 => do
      let previous ← rowProduct read row n
      let w ← read 0 (row * 3 + n.toUInt64)
      pure (previous * w)

/-- The product of the first `n` entries of row `row` of the `4 × 3` matrix in input zero, in
binary64. -/
def rowProduct64 (read : Kernel.Reader Float) (row : UInt64) : Nat → Except Kernel.Error Float
  | 0 => pure 1
  | n + 1 => do
      let previous ← rowProduct64 read row n
      let w ← read 0 (row * 3 + n.toUInt64)
      pure (previous * w)

/-- The product of each row of the `4 × 3` matrix in input zero. -/
def rowProducts : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (row : UInt64) => rowProduct read row 3)

/-- The product of each row of the `4 × 3` matrix in input zero, in binary64. -/
def rowProducts64 : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (row : UInt64) => rowProduct64 read row 3)

variable {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] [Add α] [Mul α] [BEq α] [ToString α]

/-- Run `program` with `n` outputs on the CPU and, when `device` is CUDA, on the GPU, whose result
must have the bits of the CPU run. -/
def runOn (format : CheckFormat α) (label : String) (device : Backend.Device)
    (program : Kernel.Program α) {shapes : List Shape} (inputs : TensorPack α shapes) (n : Nat) :
    IO (Tensor α [n]) := do
  let expected ← program.run inputs [n]
  if device == .cuda then
    let actual ← program.run inputs [n] (device := .cuda)
    for i in List.finRange n do
      unless format.toBits actual[i] == format.toBits expected[i] do
        throw <| IO.userError s!"{label} ({format.name})[{i}]: CUDA bits \
          {format.toBits actual[i]}, CPU bits {format.toBits expected[i]}"
  pure expected

/-- Sixteen points `(i - 8) / 4` from `-2` to `1.75`, exact in binary32 and binary64. -/
def points (format : CheckFormat α) : Tensor α [16] :=
  Tensor.ofFn fun i => format.ofFloat ((Float.ofNat i.val - 8) / 4)

/-- Second inputs `(i % 5 - 2) / 2`, from `-1` to `1`. -/
def factors (format : CheckFormat α) : Tensor α [16] :=
  Tensor.ofFn fun i => format.ofFloat ((Float.ofNat (i.val % 5) - 2) / 2)

/-- Cotangents `i % 5 - 2.5`, none of them zero. -/
def cotangents (format : CheckFormat α) : Tensor α [16] :=
  Tensor.ofFn fun i => format.ofFloat (Float.ofNat (i.val % 5) - 2.5)

/-- In the direction of all ones, the tangent of `program`, which computes `rational`, is the
derivative `2x - 1` below zero and `1 / (x + 1)²` above zero, and at the kink `x = 0` it is `1`,
the right derivative, where the left derivative is `-1`. With the nonzero cotangents of
`cotangents`, its gradient entry is the cotangent times that tangent, bit for bit. -/
def checkRational (format : CheckFormat α) (program : Kernel.Program α)
    (device : Backend.Device) : IO Unit := do
  let ones : Tensor α [16] := Tensor.ofFn fun _ => format.ofFloat 1
  let tangent ← runOn format "rational tangent" device (program.tangent 1)
    (.cons (points format) (.cons ones .nil)) 16
  let gradient ← runOn format "rational gradient" device (program.gradient 1 0 16)
    (.cons (points format) (.cons (cotangents format) .nil)) 16
  for i in List.finRange 16 do
    let x := format.toFloat (points format)[i]
    let t := format.toFloat tangent[i]
    if x == 0 then
      unless t == 1 do
        throw <| IO.userError s!"rational tangent at the kink ({format.name}): {t}, expected 1"
    else
      let derivative := if x < 0 then 2 * x - 1 else 1 / ((x + 1) * (x + 1))
      Harness.assertApprox s!"rational tangent[{i}] ({format.name})" t derivative
    let expected := (cotangents format)[i] * tangent[i]
    unless format.toBits gradient[i] == format.toBits expected do
      throw <| IO.userError
        s!"rational gradient[{i}] ({format.name}): {gradient[i]}, expected {expected}"

/-- The tangent and both gradients of `program`, which computes `quotient`, agree with its partial
derivatives `b (1 - a²) / (a² + 1)²` and `a / (a² + 1)`, and the gradients pair with a direction
as the tangent does. -/
def checkQuotient (format : CheckFormat α) (program : Kernel.Program α)
    (device : Backend.Device) : IO Unit := do
  let da : Tensor α [16] := Tensor.ofFn fun i => format.ofFloat (Float.ofNat (i.val % 3) - 1)
  let db : Tensor α [16] := Tensor.ofFn fun i => format.ofFloat (Float.ofNat (i.val % 4) - 1.5)
  let tangent ← runOn format "quotient tangent" device (program.tangent 2)
    (.cons (points format) (.cons (factors format) (.cons da (.cons db .nil)))) 16
  let inputs : TensorPack α [[16], [16], [16]] :=
    .cons (points format) (.cons (factors format) (.cons (cotangents format) .nil))
  let ga ← runOn format "quotient gradient a" device (program.gradient 2 0 16) inputs 16
  let gb ← runOn format "quotient gradient b" device (program.gradient 2 1 16) inputs 16
  let f := format.toFloat
  let mut pairing : Float := 0
  let mut expected : Float := 0
  for i in List.finRange 16 do
    let (a, b, c) := (f (points format)[i], f (factors format)[i], f (cotangents format)[i])
    let q := a * a + 1
    let (pa, pb) := (b * (1 - a * a) / (q * q), a / q)
    Harness.assertApprox s!"quotient tangent[{i}] ({format.name})" (f tangent[i])
      (pa * f da[i] + pb * f db[i])
    Harness.assertApprox s!"quotient gradient a[{i}] ({format.name})" (f ga[i]) (c * pa)
    Harness.assertApprox s!"quotient gradient b[{i}] ({format.name})" (f gb[i]) (c * pb)
    pairing := pairing + f ga[i] * f da[i] + f gb[i] * f db[i]
    expected := expected + c * f tangent[i]
  Harness.assertApprox s!"quotient pairing ({format.name})" pairing expected

/-- On small integers the tangent of `program`, which computes `matVec`, is `dW v + W dv`, its
gradient in the vector is `Wᵀ cot`, and its gradient in the matrix, in row-major order, is
`cot vᵀ`, all exact. -/
def checkMatVec (format : CheckFormat α) (program : Kernel.Program α)
    (device : Backend.Device) : IO Unit := do
  let w := Tensor.map format.ofFloat
    ([[1, -2, 0], [2, 1, -1], [0, 3, 2], [-1, 0, 1]] : Tensor Float [4, 3])
  let v := Tensor.map format.ofFloat ([1, -2, 3] : Tensor Float [3])
  let dw := Tensor.map format.ofFloat
    ([[0, 1, 0], [1, 0, 0], [0, 0, -1], [2, 0, 0]] : Tensor Float [4, 3])
  let dv := Tensor.map format.ofFloat ([1, 0, -1] : Tensor Float [3])
  let cot := Tensor.map format.ofFloat ([1, 2, -1, 3] : Tensor Float [4])
  let tangent ← runOn format "matrix-vector tangent" device (program.tangent 2)
    (.cons w (.cons v (.cons dw (.cons dv .nil)))) 4
  let inputs : TensorPack α [[4, 3], [3], [4]] := .cons w (.cons v (.cons cot .nil))
  let gw ← runOn format "matrix-vector gradient W" device (program.gradient 2 0 4) inputs 12
  let gv ← runOn format "matrix-vector gradient v" device (program.gradient 2 1 4) inputs 3
  for r in List.finRange 4 do
    let expected := (List.finRange 3).foldl
      (fun acc c => acc + (dw[r][c] * v[c] + w[r][c] * dv[c])) (format.ofFloat 0)
    unless tangent[r] == expected do
      throw <| IO.userError
        s!"matrix-vector tangent[{r}] ({format.name}): {tangent[r]}, expected {expected}"
  for c in List.finRange 3 do
    let expected := (List.finRange 4).foldl (fun acc r => acc + cot[r] * w[r][c])
      (format.ofFloat 0)
    unless gv[c] == expected do
      throw <| IO.userError
        s!"matrix-vector gradient v[{c}] ({format.name}): {gv[c]}, expected {expected}"
  for a in List.finRange 12 do
    let r : Fin 4 := ⟨a.val / 3, by have := a.isLt; omega⟩
    let c : Fin 3 := ⟨a.val % 3, by omega⟩
    unless gw[a] == cot[r] * v[c] do
      throw <| IO.userError s!"matrix-vector gradient W[{r}, {c}] ({format.name}): {gw[a]}, \
        expected {cot[r] * v[c]}"

/-- On small integers the tangent of `program`, which computes `rowProducts`, is
`∑ c, dW[r][c] ∏ c' ≠ c, W[r][c']` at row `r`, and its gradient at entry `(r, c)`, in row-major
order, is `cot[r] ∏ c' ≠ c, W[r][c']`, all exact. -/
def checkRowProducts (format : CheckFormat α) (program : Kernel.Program α)
    (device : Backend.Device) : IO Unit := do
  let w := Tensor.map format.ofFloat
    ([[1, -2, 3], [2, 1, -1], [3, 2, 2], [-1, 1, 2]] : Tensor Float [4, 3])
  let dw := Tensor.map format.ofFloat
    ([[0, 1, 0], [1, 0, 2], [-1, 0, 1], [2, -1, 1]] : Tensor Float [4, 3])
  let cot := Tensor.map format.ofFloat ([1, 2, -1, 3] : Tensor Float [4])
  let tangent ← runOn format "row product tangent" device (program.tangent 1)
    (.cons w (.cons dw .nil)) 4
  let gradient ← runOn format "row product gradient" device (program.gradient 1 0 4)
    (.cons w (.cons cot .nil)) 12
  let f := format.toFloat
  for r in List.finRange 4 do
    let others (c : Fin 3) : Float := (List.finRange 3).foldl
      (fun acc c' => if c' == c then acc else acc * f w[r][c']) 1
    let expected := (List.finRange 3).foldl (fun acc c => acc + f dw[r][c] * others c) 0
    unless f tangent[r] == expected do
      throw <| IO.userError
        s!"row product tangent[{r}] ({format.name}): {tangent[r]}, expected {expected}"
    for c in List.finRange 3 do
      let a : Fin 12 := ⟨r.val * 3 + c.val, by omega⟩
      unless f gradient[a] == f cot[r] * others c do
        throw <| IO.userError s!"row product gradient[{r}, {c}] ({format.name}): \
          {gradient[a]}, expected {f cot[r] * others c}"

/-- Run the derivative checks in binary32 and in binary64 on the CPU and, when `device` is CUDA,
compare the GPU runs with the CPU bits. -/
def run (device : Backend.Device) : IO Unit := do
  checkRational .binary32 rational device
  checkQuotient .binary32 quotient device
  checkMatVec .binary32 matVec device
  checkRowProducts .binary32 rowProducts device
  checkRational .binary64 rational64 device
  checkQuotient .binary64 quotient64 device
  checkMatVec .binary64 matVec64 device
  checkRowProducts .binary64 rowProducts64 device
  IO.println s!"tangent and gradient kernels ({repr device}): derivatives, pairing, exact sums: ok"

end TorchLean.Test.KernelGradient
