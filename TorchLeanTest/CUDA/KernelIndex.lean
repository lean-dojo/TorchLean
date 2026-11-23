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
public import TorchLeanTest.CUDA.KernelGradient
import TorchLean.CUDA.Kernel.Elab

/-!
# Conversions between scalars and indices

A custom operation can convert a scalar to an index (`Float32.toUInt64`, `Float.toUInt64`, or
`IndexCast.toIndex` in any scalar type) and an index to a scalar (`UInt64.toFloat32`,
`UInt64.toFloat`, `IndexCast.ofIndex`), so it can read an input at a position it computes. We check
the two conversions at the values where their rules differ: NaN, the infinities, negative values,
signed zeros, fractions, the ties of rounding an index to binary32 and binary64, and the ends of the
`UInt64` range. A scalar converted to an index is observed rounded back to a scalar and through
the remainder of the index modulo 7, which keeps its low bits exact. Configured binary formats
convert through the device template in `csrc/libtorch/binary.h`, checked on exceptional and
pseudorandom words of each format.

The application is the table lookup of a coordinate `u` into a uniform table of `width` samples by
linear interpolation, `np.interp` on the grid `0, 1, …, width - 1` with the end samples held
outside it. The coordinate's segment `⌊u⌋` is an index computed from the input, so the lookup is one
kernel. Its value is checked at interior, integral, boundary, out-of-range and NaN coordinates. Its
gradient in the coordinate is the slope of the segment inside the grid and zero outside it, and its
gradient in the table gives each sample the interpolation weights of the coordinates that read it.
These values are dyadic, so the gradients are exact. On CUDA every result has the bits of the CPU
run.
-/

@[expose] public section

namespace TorchLean.Test.KernelIndex

open Backend.Device TorchLean.Tensor
open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange
open TorchLean.Test.KernelGradient (CheckFormat runOn)
open scoped Kernel

/-! ## Native formats -/

/-- The index of each entry of input zero, rounded back to binary32. -/
def roundTrip32 : Kernel.Program Float32 :=
  Program.of (fun (x : Float32) => x.toUInt64.toFloat32)

/-- The index of each entry of input zero, rounded back to binary64. -/
def roundTrip64 : Kernel.Program Float :=
  Program.of (fun (x : Float) => x.toUInt64.toFloat)

/-- Entry `index % 7` of input one, at the index of each entry of input zero. -/
def lowBits32 : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let x ← read 0 i
    read 1 (x.toUInt64 % 7) : Except Kernel.Error Float32))

/-- Entry `index % 7` of input one, at the index of each entry of input zero, in binary64. -/
def lowBits64 : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (i : UInt64) => (do
    let x ← read 0 i
    read 1 (x.toUInt64 % 7) : Except Kernel.Error Float))

/-- Binary32 rounding of an index from a pseudorandom sweep of the `UInt64` range. -/
def sweep32 : Kernel.Program Float32 :=
  Program.of (fun (_ : Kernel.Reader Float32) (i : UInt64) =>
    (pure (i * 6364136223846793005 + 1442695040888963407).toFloat32 :
      Except Kernel.Error Float32))

/-- Binary64 rounding of an index from a pseudorandom sweep of the `UInt64` range. -/
def sweep64 : Kernel.Program Float :=
  Program.of (fun (_ : Kernel.Reader Float) (i : UInt64) =>
    (pure (i * 6364136223846793005 + 1442695040888963407).toFloat : Except Kernel.Error Float))

/-- Binary32 rounding of the indices from `2^24 - 4`, where consecutive integers start to tie. -/
def ties32 : Kernel.Program Float32 :=
  Program.of (fun (_ : Kernel.Reader Float32) (i : UInt64) =>
    (pure (i + 16777212).toFloat32 : Except Kernel.Error Float32))

/-- Binary64 rounding of the indices from `2^53 - 4`, where consecutive integers start to tie. -/
def ties64 : Kernel.Program Float :=
  Program.of (fun (_ : Kernel.Reader Float) (i : UInt64) =>
    (pure (i + 9007199254740988).toFloat : Except Kernel.Error Float))

/-- The values where the conversion rules differ, in binary64. -/
def specials : Array Float :=
  #[0.0 / 0.0, -1.0 / 0.0, -1.5, -0.5, -0.0, 0.0, 0.5, 1.0, 1.9, 2.0, 16777217.0, 3.0e9,
    9223372036854775808.0, 18446742974197923840.0, 18446744073709551616.0, 1.0e30, 1.0 / 0.0]

/-- The special values as a tensor of `format`. -/
def specialTensor {α : Type} [Storage α] (format : CheckFormat α) : Tensor α [17] :=
  Tensor.ofFn fun i => format.ofFloat specials[i.val]!

/-- The table `0, 1, …, 6` that `lowBits32` and `lowBits64` read. -/
def residues {α : Type} [Storage α] (format : CheckFormat α) : Tensor α [7] :=
  Tensor.ofFn fun i => format.ofFloat (Float.ofNat i.val)

/-- Each native check against Lean's own conversions, then on CUDA against the CPU bits. -/
def checkNative {α : Type} [Storage α] (format : CheckFormat α) (toIndex : α → UInt64)
    (ofIndex : UInt64 → α) (roundTrip lowBits sweep ties : Kernel.Program α)
    (tieBase : UInt64) (device : Backend.Device) : IO Unit := do
  let x := specialTensor format
  let back ← runOn format "index round trip" device roundTrip (.cons x .nil) 17
  let low ← runOn format "index low bits" device lowBits (.cons x (.cons (residues format) .nil)) 17
  for i in List.finRange 17 do
    let index := toIndex x[i]
    unless format.toBits back[i] == format.toBits (ofIndex index) do
      throw <| IO.userError s!"index round trip ({format.name}) at {specials[i.val]!}: \
        {format.toFloat back[i]}, expected {format.toFloat (ofIndex index)}"
    unless format.toFloat low[i] == Float.ofNat (index % 7).toNat do
      throw <| IO.userError s!"index low bits ({format.name}) at {specials[i.val]!}: \
        {format.toFloat low[i]}, expected {(index % 7).toNat}"
  -- Lean's conversion: zero at a NaN and below zero, truncation, saturation at `2^64 - 1`.
  let expected : List Nat := [0, 0, 0, 0, 0, 0, 0, 1, 1, 2]
  for h : i in [:expected.length] do
    let got := toIndex x[(⟨i, by have := h.upper; simp [expected] at this; omega⟩ : Fin 17)]
    unless got.toNat == expected[i] do
      throw <| IO.userError s!"index of {specials[i]!} ({format.name}): {got.toNat}, \
        expected {expected[i]}"
  for i in [14, 15, 16] do
    unless toIndex (format.ofFloat specials[i]!) == UInt64.ofNat (2 ^ 64 - 1) do
      throw <| IO.userError s!"index of {specials[i]!} ({format.name}) does not saturate"
  let swept ← runOn format "index sweep" device sweep (.nil) 64
  let tied ← runOn format "index ties" device ties (.nil) 8
  for i in List.finRange 64 do
    let index := UInt64.ofNat i.val * 6364136223846793005 + 1442695040888963407
    unless format.toBits swept[i] == format.toBits (ofIndex index) do
      throw <| IO.userError s!"index sweep ({format.name})[{i}]: {format.toFloat swept[i]}"
  for i in List.finRange 8 do
    unless format.toBits tied[i] == format.toBits (ofIndex (UInt64.ofNat i.val + tieBase)) do
      throw <| IO.userError s!"index ties ({format.name})[{i}]: {format.toFloat tied[i]}"

/-! ## Configured formats -/

/-- The scalar of the index of `x`, in a configured binary format, on the CPU and on CUDA, and the
scalars of a sweep of indices. The words are the exceptional and boundary patterns of the format
and 64 pseudorandom words. -/
@[noinline, nospecialize] def checkConfigured {format : FloatFormat}
    {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]
    (_ : ExecFloat (Configured.Family format code plan)) (label : String) : IO Unit := do
  let α := ExecFloat (Configured.Family format code plan)
  let fractionBits := format.fracWidth
  let sign := 2 ^ (format.expWidth + fractionBits)
  let inf := (2 ^ format.expWidth - 1) * 2 ^ fractionBits
  let one := format.exponentBias * 2 ^ fractionBits
  let patterns : List Nat := [0, sign, 1, sign + 1, 2 ^ fractionBits, one, sign + one,
    one - 1, one + 1, one + 2 ^ fractionBits, inf - 1, inf, sign + inf, inf + 1]
  let count := patterns.length + 64
  let word (i : Nat) : Nat := (((i + 1) * 6364136223846793005) % 2 ^ 64) % (2 * sign)
  let words : Tensor α [count] := Tensor.generateFlat [count] fun i =>
    ExecFloat.Binary.ofNatBits (patterns.getD i (word i))
  let roundTrip : Kernel.Program α :=
    Program.of (fun (x : α) => (Kernel.IndexCast.ofIndex (Kernel.IndexCast.toIndex x) : α))
  let sweep : Kernel.Program α :=
    Program.of (fun (_ : Kernel.Reader α) (i : UInt64) =>
      (pure (Kernel.IndexCast.ofIndex (i * 6364136223846793005 + 1442695040888963407)) :
        Except Kernel.Error α))
  let small : Kernel.Program α :=
    Program.of (fun (_ : Kernel.Reader α) (i : UInt64) =>
      (pure (Kernel.IndexCast.ofIndex i) : Except Kernel.Error α))
  let compare (operation : String) {n : Nat} (cpu gpu : Tensor α [n]) : IO Unit := do
    for h : i in [:n] do
      let index : Fin n := ⟨i, h.upper⟩
      let observed := ExecFloat.Binary.toNatBits (gpu.getScalar index)
      let target := ExecFloat.Binary.toNatBits (cpu.getScalar index)
      unless observed == target do
        throw <| IO.userError s!"custom {label} {operation}: encoding mismatch at {i}: \
          got {observed}, expected {target}"
  compare "index round trip" (← roundTrip.run (.cons words .nil) [count])
    (← roundTrip.run (.cons words .nil) [count] (device := .cuda))
  compare "index sweep" (← sweep.run .nil [64]) (← sweep.run .nil [64] (device := .cuda))
  compare "small indices" (← small.run .nil [600]) (← small.run .nil [600] (device := .cuda))
  IO.println s!"  custom {label}: index conversions matched FloatLib"

/-- The configured checks over the narrow, standard-width and wide formats and the encodings. -/
def checkConfiguredFormats : IO Unit := do
  checkConfigured (0 : ExecFloat.Binary 4 3) "IEEE e4m3"
  checkConfigured (0 : ExecFloat.Binary 5 2) "IEEE e5m2"
  checkConfigured (0 : ExecFloat.Binary 4 3 (encoding := .finiteMaxNaN)) "e4m3 finite/NaN"
  checkConfigured (0 : ExecFloat.Binary 4 3 (encoding := .finiteUnsignedZero))
    "e4m3 unsigned zero"
  checkConfigured (0 : ExecFloat.Binary 4 3 (encoding := .finite)) "fully finite e4m3"
  checkConfigured (0 : ExecFloat.Binary 5 10) "binary16"
  checkConfigured (0 : ExecFloat.Binary 8 7) "bfloat16"
  checkConfigured (0 : ExecFloat.Binary 8 23) "configured binary32"
  checkConfigured (0 : ExecFloat.Binary 11 52) "configured binary64"
  checkConfigured (0 : ExecFloat.Binary 15 112) "binary128"

/-! ## Table lookup -/

/-- Linear interpolation into the uniform table of `width` samples in input one at each coordinate
of input zero: the first sample at or below `0`, the last at or above `width - 1`, and inside the
grid `v₀ + (v₁ - v₀) (u - ⌊u⌋)` with `v₀ = table[⌊u⌋]` and `v₁ = table[⌊u⌋ + 1]`. A NaN coordinate
selects the segment `0` and gives NaN. -/
def lookup (width : UInt64) : Kernel.Program Float32 :=
  Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => (do
    let u ← read 0 i
    if (width - 1).toFloat32 ≤ u then read 1 (width - 1)
    else do
      let k := u.toUInt64
      let v₀ ← read 1 k
      if u ≤ 0 then pure v₀
      else do
        let v₁ ← read 1 (k + 1)
        pure (v₀ + (v₁ - v₀) * (u - k.toFloat32)) : Except Kernel.Error Float32))

/-- Linear interpolation into a uniform table, in binary64. -/
def lookup64 (width : UInt64) : Kernel.Program Float :=
  Program.of (fun (read : Kernel.Reader Float) (i : UInt64) => (do
    let u ← read 0 i
    if (width - 1).toFloat ≤ u then read 1 (width - 1)
    else do
      let k := u.toUInt64
      let v₀ ← read 1 k
      if u ≤ 0 then pure v₀
      else do
        let v₁ ← read 1 (k + 1)
        pure (v₀ + (v₁ - v₀) * (u - k.toFloat)) : Except Kernel.Error Float))

/-- Coordinates inside, at and outside the grid of five samples, all dyadic. -/
def coordinates : Array Float := #[-1.0, 0.0, 0.5, 1.25, 2.0, 2.75, 3.5, 4.0, 7.0]

/-- The samples `1, 2, 4, 8, 16`. -/
def samples : Array Float := #[1.0, 2.0, 4.0, 8.0, 16.0]

/-- The cotangents `1, …, 9` of the nine lookups. -/
def cotangents : Array Float := (Array.range 9).map fun i => Float.ofNat (i + 1)

/-- The value, the coordinate gradient and the table gradient of the lookup, exact on these
dyadic inputs, and NaN at a NaN coordinate. -/
def checkLookup {α : Type} [Storage α] [OfNat α 0] [OfNat α 1] (format : CheckFormat α)
    (program : Kernel.Program α) (device : Backend.Device) : IO Unit := do
  let u : Tensor α [9] := Tensor.ofFn fun i => format.ofFloat coordinates[i.val]!
  let table : Tensor α [5] := Tensor.ofFn fun i => format.ofFloat samples[i.val]!
  let cot : Tensor α [9] := Tensor.ofFn fun i => format.ofFloat cotangents[i.val]!
  let value ← runOn format "lookup" device program (.cons u (.cons table .nil)) 9
  let inputs : TensorPack α [[9], [5], [9]] := .cons u (.cons table (.cons cot .nil))
  let du ← runOn format "lookup gradient in the coordinate" device (program.gradient 2 0 9)
    inputs 9
  let dtable ← runOn format "lookup gradient in the table" device (program.gradient 2 1 9)
    inputs 5
  let mut expectedTable : Array Float := Array.replicate 5 0
  for i in List.finRange 9 do
    let x := coordinates[i.val]!
    let c := cotangents[i.val]!
    let (v, slope) :=
      if x ≥ 4 then (samples[4]!, (0 : Float))
      else if x ≤ 0 then (samples[0]!, (0 : Float))
      else
        let k := x.floor.toUInt64.toNat
        let w := x - x.floor
        (samples[k]! + (samples[k + 1]! - samples[k]!) * w, samples[k + 1]! - samples[k]!)
    unless format.toFloat value[i] == v do
      throw <| IO.userError s!"lookup ({format.name}) at {x}: {format.toFloat value[i]}, \
        expected {v}"
    unless format.toFloat du[i] == c * slope do
      throw <| IO.userError s!"lookup gradient in the coordinate ({format.name}) at {x}: \
        {format.toFloat du[i]}, expected {c * slope}"
    if x ≥ 4 then expectedTable := expectedTable.modify 4 (· + c)
    else if x ≤ 0 then expectedTable := expectedTable.modify 0 (· + c)
    else
      let k := x.floor.toUInt64.toNat
      let w := x - x.floor
      expectedTable := (expectedTable.modify k (· + c * (1 - w))).modify (k + 1) (· + c * w)
  for i in List.finRange 5 do
    unless format.toFloat dtable[i] == expectedTable[i.val]! do
      throw <| IO.userError s!"lookup gradient in the table ({format.name})[{i}]: \
        {format.toFloat dtable[i]}, expected {expectedTable[i.val]!}"
  let nan : Tensor α [1] := Tensor.ofFn fun _ => format.ofFloat (0.0 / 0.0)
  let atNaN ← runOn format "lookup at NaN" device program (.cons nan (.cons table .nil)) 1
  unless (format.toFloat atNaN[(0 : Fin 1)]).isNaN do
    throw <| IO.userError s!"lookup at NaN ({format.name}): {format.toFloat atNaN[(0 : Fin 1)]}"

/-- Run the checks on the CPU and, when `device` is CUDA, compare every result with the GPU's. -/
def run (device : Backend.Device) : IO Unit := do
  checkNative .binary32 Float32.toUInt64 UInt64.toFloat32 roundTrip32 lowBits32 sweep32 ties32
    16777212 device
  checkNative .binary64 Float.toUInt64 UInt64.toFloat roundTrip64 lowBits64 sweep64 ties64
    9007199254740988 device
  checkLookup .binary32 (lookup 5) device
  checkLookup .binary64 (lookup64 5) device
  if device == .cuda then checkConfiguredFormats
  IO.println s!"  kernel index conversions ({if device == .cuda then "CUDA" else "CPU"}): passed"

end TorchLean.Test.KernelIndex
