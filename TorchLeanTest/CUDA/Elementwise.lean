/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public meta import TorchLean.Tensor.Einops.Elab.Einsum
public meta import TorchLean.Tensor.Einops.Elab.Pack
public meta import TorchLean.Tensor.Literal
public meta import TorchLean.Tensor.Einops.Elab.Transform
public meta import TorchLean.Tensor.Tactic.Einops.Report
public import TorchLean.Tensor.Literal
public import TorchLeanTest.Harness.CUDA
import TorchLean.NN.Kernel.Function
import TorchLean.NN.Kernel.Graph
public import TorchLean.Tensor.Tactic.Init
import TorchLean.NN.Sequential.Execution
import TorchLean.NN.Optimizer.Constructor
import TorchLean.NN.Builder
import TorchLean.NN.Training.Reporting

/-!
# CUDA Kernel Coverage: Elementwise Ops

Activation regressions cover `relu`, `tanh`, `sigmoid` and `silu` on device buffers. A composite
forward and backward pass compares the CUDA eager tape with the CPU tape through `add`, `sub`,
`mul`, `scale`, `abs`, `sqrt`, `clamp`, `max`, `min`, `relu`, `sigmoid`, `tanh`, `softplus`, `exp`,
`log`, `inv`, `safe_log` and `sum`, and a second pass does the same for `gelu`.
The VJP kernels of `abs`, `relu`, `sqrt` and `tanh` in binary32 and binary64 and of `sigmoid` in
binary32 have the expected bits, any two NaNs equal, at edge cases such as signed zeros, NaNs,
infinities and subnormals.
`Buffer.scaledProdExp` has the bits of the composed `exp ((c * x) * y)`, the `max` and `min`
pullbacks those of their binary32 CPU rules, and `dropout` those of its specification at binary32.
The Huber terms and slopes have the bits of `Huber.term` and `Tensor.huberLossBackward` at
signed zeros, infinities, NaNs, rounding ties and overflow. Generated custom computations also
exercise NVRTC, scalar precision, lazy branches, sequential accumulation, empty tensors and checked
native reads, and the fused AdamW step is compared with the staged computation it replaces.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Test
namespace CUDA
namespace Elementwise

open TorchLean.Tensor
open scoped Kernel

/-- Fail unless `got` agrees with `expected`: a NaN when `expected` is a NaN, the same bits when
`expected` is a zero or an infinity, and otherwise finite within relative tolerance `rtol`. -/
def assertActivationValue (label : String) (got expected : Float)
    (rtol : Float := 3e-6) : IO Unit := do
  if expected.isNaN then
    unless got.isNaN do
      throw <| IO.userError s!"{label}: expected NaN, got {got}"
  else if expected == 0.0 || expected.isInf then
    unless got.toBits == expected.toBits do
      throw <| IO.userError
        s!"{label}: expected bits {expected.toBits}, got {got.toBits}"
  else
    -- No absolute floor: replacing a tiny representable result with zero must fail.
    unless !got.isNaN && !got.isInf && (got - expected).abs ≤ rtol * expected.abs do
      throw <| IO.userError
        (s!"{label}: got {got} (bits {got.toBits}), expected {expected} " ++
          s!"(bits {expected.toBits}, rtol {rtol})")

/-- Fail unless the arrays have one size and `assertActivationValue` holds at every index. -/
def assertActivationArray (label : String) (got expected : FloatArray)
    (rtol : Float := 3e-6) : IO Unit := do
  unless got.size == expected.size do
    throw <| IO.userError s!"{label}: size {got.size}, expected {expected.size}"
  for i in [:expected.size] do
    assertActivationValue s!"{label}[{i}]" (got.get! i) (expected.get! i) rtol

/-- Checks `forward` on `inputs` against `expected`, then the local and taped VJPs of the node that
`record` adds, at `seeds` and over two passes that retain no temporary payload. The reference slope
`derivative` is evaluated at the input, which keeps the rounding of the activation output out of
it. -/
def checkActivation
    (label : String)
    (forward : LibTorch.Buffer → LibTorch.Buffer)
    (record : (s : Shape) → Tape LibTorch.SomeBuffer → Nat →
      IO (Tape LibTorch.SomeBuffer × Nat))
    (derivative : Float32 → Float)
    (inputs expected seeds : Array Float) : IO Unit := do
  unless inputs.size == expected.size && inputs.size == seeds.size do
    throw <| IO.userError s!"{label}: inconsistent test data"
  let s : Shape := [inputs.size]
  let input ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk inputs)
  let original ← LibTorch.Buffer.toFloatArrayIO input
  let direct ← IO.lazyPure fun _ => forward input
  assertActivationArray s!"{label} buffer"
    (← LibTorch.Buffer.toFloatArrayIO direct) (FloatArray.mk expected)
  discard <| LibTorch.Buffer.releaseIO direct
  assertActivationArray s!"{label} borrowed input"
    (← LibTorch.Buffer.toFloatArrayIO input) original

  let (t1, xId) := Tape.leaf
    Tape.empty { s := s, buf := input : LibTorch.SomeBuffer }
  let (t2, yId) ← record s t1 xId
  let some node := t2.getNode? yId
    | throw <| IO.userError s!"{label}: missing activation node"
  unless t2.nodes.size == 2 && node.parents == #[xId] &&
      node.requiresGrad && node.ownsValue do
    throw <| IO.userError s!"{label}: expected one owned, differentiable unary node"
  assertActivationArray s!"{label} tape"
    (← LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
  let expectedGrad := FloatArray.mk <| inputs.mapIdx fun i x =>
    let seed := seeds[i]!.toFloat32
    let slope := derivative x.toFloat32
    -- A finite cotangent is weighted by the binary64 slope and the product rounded to binary32. A
    -- nonfinite one is multiplied in binary32 by the slope rounded to binary32.
    if seed.isNaN || seed.isInf then (seed * slope.toFloat32).toFloat
    else (seed.toFloat * slope).toFloat32.toFloat
  let retained ← LibTorch.Buffer.Memory.read
  for pass in [:2] do
    let seed ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk seeds)
    -- The recorded closure runs on its own before any accumulation, with cotangents that include
    -- signed zeros.
    let localResult ← IO.lazyPure fun _ => node.backward { s := s, buf := seed }
    let contributions ← IO.ofExcept localResult
    let some (parentId, contribution) := contributions[0]?
      | throw <| IO.userError s!"{label}: missing VJP contribution"
    unless contributions.size == 1 && parentId == xId && contribution.s == s do
      throw <| IO.userError s!"{label}: malformed VJP contribution"
    assertActivationArray s!"{label} local VJP, pass {pass}"
      (← LibTorch.Buffer.toFloatArrayIO contribution.buf) expectedGrad
        (rtol := 3e-5)
    discard <| LibTorch.Buffer.releaseIO contribution.buf
    assertActivationArray s!"{label} borrowed cotangent"
      (← LibTorch.Buffer.toFloatArrayIO seed) (FloatArray.mk seeds)
    let grads ← LibTorch.Tape.backwardSparse t2 yId
      { s := s, buf := seed } (fun id => id == xId)
    let some grad := grads.get? xId
      | throw <| IO.userError s!"{label}: missing tape gradient"
    assertActivationArray s!"{label} tape VJP, pass {pass}"
      (← LibTorch.Buffer.toFloatArrayIO grad.buf) expectedGrad (rtol := 3e-5)
    LibTorch.Tape.releaseSparseGrads grads
    assertActivationArray s!"{label} input after backward"
      (← LibTorch.Buffer.toFloatArrayIO input) original
    assertActivationArray s!"{label} output after backward"
      (← LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
    let after ← LibTorch.Buffer.Memory.read
    unless after.liveBytes == retained.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads on pass {pass}"
  discard <| LibTorch.Buffer.releaseIO node.value.buf
  discard <| LibTorch.Buffer.releaseIO input

/-- Device activations against their specifications: `relu` at signed zeros, infinities and NaN in
binary32 and binary64, and `tanh`, `sigmoid` and `silu` with their VJPs on tiny, central,
saturated, nonfinite and empty inputs and at nonfinite cotangents. A tiny result may not be flushed
to zero. -/
def runActivationNumerics : IO Unit := do
  IO.println "== direct activation numerical regressions =="
  let inf := Float.ofBits 0x7ff0000000000000
  let nan := Float.ofBits 0x7ff8000000000000
  -- The ordered maximum keeps a NaN input. Selecting the strictly positive entries would give `0`
  -- for it.
  let reluInputs : Array Float := #[nan, -inf, -1.0, -0.0, 0.0, 1.0, inf]
  let reluValues := FloatArray.mk <| reluInputs.map Activation.relu
  for dtype in #[LibTorch.Dtype.float32,
      LibTorch.Dtype.float64] do
    let input ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk reluInputs) dtype
    let output ← IO.lazyPure fun _ => LibTorch.Buffer.relu input
    assertActivationArray s!"relu nonfinite/zero ({repr dtype})"
      (← LibTorch.Buffer.toFloatArrayIO output) reluValues
    discard <| LibTorch.Buffer.releaseIO output
    discard <| LibTorch.Buffer.releaseIO input
  let tiny : Array Float :=
    #[1e-30, -1e-30, 1e-12, -1e-12, 1e-8, -1e-8, 1e-5, -1e-5]
  let central : Array Float := #[-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]
  let inputs := (tiny ++ #[0.0, -0.0] ++ central ++ #[-100.0, 100.0, -inf, inf, nan]).map
    (fun (x : Float) => x.toFloat32.toFloat)
  let seedPattern : Array Float := #[2.0, -0.5, 1.5, -3.0, 0.25]
  let seeds := inputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  let tanhValues := inputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat
  let tanhNode := fun s t id => Harness.record t "tanh" (.tanh (s := s)) (id, ())
  let sigmoidNode := fun s t id => Harness.record t "sigmoid" (.sigmoid (s := s)) (id, ())
  -- The reference slopes are evaluated in binary64 at the input, from `exp (-|x|)`, so they avoid
  -- the cancellation of a formula in the rounded activation such as `1 - tanh x ^ 2`.
  let tanhDerivative := fun (x : Float32) =>
    let z := Float.exp (-x.toFloat.abs)
    let c := (z + z) / (1 + z * z)
    c * c
  let sigmoidDerivative := fun (x : Float32) =>
    let z := Float.exp (-x.toFloat.abs)
    z / ((1 + z) * (1 + z))
  checkActivation "tanh tiny/center/tails" LibTorch.Buffer.tanh tanhNode
    tanhDerivative inputs tanhValues seeds

  let sigmoidInputs := inputs ++ #[-80.0, -20.0, 20.0, 80.0]
  let sigmoidValues := sigmoidInputs.map fun x => (Context.sigmoid x).toFloat32.toFloat
  let sigmoidSeeds := sigmoidInputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  checkActivation "sigmoid tiny/center/tails" LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative sigmoidInputs sigmoidValues sigmoidSeeds

  let siluNode := fun s t id => Harness.record t "silu" (.silu (s := s)) (id, ())
  -- At the infinities the slope takes its limits `1` and `-0`, as `Activation.siluDeriv`
  -- does.
  let siluDerivative := fun (x : Float32) =>
    if x.isInf then (if x > 0 then 1 else -0.0) else
    let z := Float.exp (-x.toFloat.abs)
    let d := 1 + z
    let gate := if x > 0 then 1 / d else z / d
    gate + x.toFloat * (z / (d * d))
  let siluInputs := (tiny ++ #[0.0, -0.0] ++ central ++
    #[-104.0, -80.0, 20.0, 100.0, -inf, inf, nan]).map
      (fun (x : Float) => x.toFloat32.toFloat)
  let siluValues := siluInputs.map fun x => (Activation.silu x).toFloat32.toFloat
  -- With unit seeds each gradient is the slope itself, so the subnormal slope at `-104` is checked
  -- without a further underflow in the cotangent product.
  checkActivation "silu tiny/center/tails" LibTorch.Buffer.silu siluNode
    siluDerivative siluInputs siluValues (siluInputs.map fun _ => 1)

  -- Infinite seeds multiply the slope computed from the input. At `100` the sigmoid slope is a
  -- binary32 subnormal, and a slope recovered from the rounded output `1` would be zero.
  let specialInputs : Array Float :=
    #[0.0, -0.0, 1.0, -1.0, 100.0, -100.0, inf, -inf, nan, 0.5, -100.0]
  let specialSeeds : Array Float :=
    #[-0.0, 0.0, inf, -inf, inf, -inf, -0.0, -2.0, 0.0, nan, nan]
  checkActivation "tanh nonfinite VJP" LibTorch.Buffer.tanh tanhNode tanhDerivative
    specialInputs (specialInputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat)
    specialSeeds
  checkActivation "sigmoid nonfinite VJP" LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative specialInputs
    (specialInputs.map fun x => (Context.sigmoid x).toFloat32.toFloat)
    specialSeeds
  checkActivation "tanh empty" LibTorch.Buffer.tanh tanhNode
    tanhDerivative #[] #[] #[]
  checkActivation "sigmoid empty" LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative #[] #[] #[]
  checkActivation "silu nonfinite VJP" LibTorch.Buffer.silu siluNode
    siluDerivative specialInputs
    (specialInputs.map fun x => (Activation.silu x).toFloat32.toFloat)
    specialSeeds
  checkActivation "silu empty" LibTorch.Buffer.silu siluNode
    siluDerivative #[] #[] #[]

/-- The VJP kernels of `abs`, `relu`, `sqrt` and `tanh` in binary32 and binary64 and of `sigmoid`
in binary32, bitwise with any two NaNs equal: `abs` multiplies `Context.sign x` (`x` itself at a
zero or NaN) by the upstream value, `relu` and `sqrt` multiply a zero factor by it where `x > 0`
fails, `sqrt` rounds `g / (2 r)` once at the rounded root `r` (`7 / 3` at `x = 2.25`, ties to
even, overflow), at `x = ±0` `tanh` keeps a subnormal `g`, and `sigmoid` rounds a tie `g / 4` to
even at `x = ±0` and toward zero at a nonzero `x` whose half `|x| / 2` rounds below the least
normal. -/
def checkPointwiseVJPBits : IO Unit := do
  IO.println "== abs, relu, sqrt, sigmoid and tanh VJP bits =="
  let inf := Float.ofBits 0x7ff0000000000000
  let nan := Float.ofBits 0x7ff8000000000000
  for dtype in #[LibTorch.Dtype.float32,
      LibTorch.Dtype.float64] do
    let third : Float := if dtype == .float32 then ((7 : Float32) / 3).toFloat else 7 / 3
    -- `5 / 6` rounded once at `x = 9` and `g = 5`, which differs in both formats from `5`
    -- times the rounded `1 / 6`.
    let atNine : Float := if dtype == .float32 then (Float32.ofBits 0x3F555555).toFloat
      else Float.ofBits 0x3FEAAAAAAAAAAAAB
    -- `3 / (2 √3)` rounded once at the rounded root, the smallest subnormal, a subnormal with an
    -- exact root, and a seed whose quotient by twice that root overflows.
    let atThree : Float := if dtype == .float32 then (Float32.ofBits 0x3F5DB3D8).toFloat
      else Float.ofBits 0x3FEBB67AE8584CAB
    let least : Float := if dtype == .float32 then (Float32.ofBits 1).toFloat else Float.ofBits 1
    let tiny : Float := if dtype == .float32 then (Float32.ofBits 2).toFloat else Float.ofBits 1
    let big : Float := if dtype == .float32 then 1e30 else 1e300
    let cases : List (String × (LibTorch.Buffer →
        LibTorch.Buffer → LibTorch.Buffer) ×
        Array Float × Array Float × Array Float) :=
      [("abs", LibTorch.Buffer.absBackward,
          #[nan, -0.0, -0.0, 0.0, 0.0, -1, -1, -1, 2,
            0.0, -0.0, -0.0, 0.0, 2, -1, 2, -1, -1, inf, -inf, nan, least, -least],
          #[1, 1, -1, 1, inf, -1, inf, nan, -inf,
            -1, -inf, -0.0, -0.0, inf, -inf, -0.0, -0.0, 0.0, 3, 3, 0.0, 5, 5],
          #[nan, -0.0, 0.0, 0.0, nan, 1, -inf, nan, -inf,
            -0.0, nan, 0.0, -0.0, inf, inf, -0.0, 0.0, -0.0, 3, -3, nan, 5, -5]),
        ("relu", LibTorch.Buffer.reluBackward,
          #[-1, -1, -1, -0.0, nan, 2, 2,
            0.0, -1, 0.0, 2, 2, 2, -1, -1, nan, nan, inf, -inf, -inf, least, -least],
          #[-1, inf, nan, -1, 1, -3, nan,
            -1, -inf, inf, inf, -inf, -0.0, -0.0, 0.0, -1, inf, -2, 2, -2, 5, 5],
          #[-0.0, nan, nan, -0.0, 0, -3, nan,
            -0.0, nan, nan, inf, -inf, -0.0, -0.0, 0.0, -0.0, nan, -2, 0.0, -0.0, 5, 0.0]),
        ("sqrt", LibTorch.Buffer.sqrtClampNegativeBackward,
          #[-1, 0, -0.0, nan, 2.25, 3, 1, 1, tiny, tiny,
            9, 0.0, -1, 0.0, 4, 4, 4, 4, -1, nan, nan, inf, inf, inf, -inf, -inf],
          #[-1, inf, 1, 1, 7, 3, least, 3 * least, big, -big,
            5, -1, -inf, nan, nan, inf, -inf, -0.0, -0.0, -1, inf, 2, -2, inf, 2, -2],
          #[-0.0, nan, 0, 0, third, atThree, 0, 2 * least, inf, -inf,
            atNine, -0.0, nan, nan, nan, inf, -inf, -0.0, -0.0, -0.0, nan, 0.0, -0.0, nan,
            0.0, -0.0]),
        ("tanh", LibTorch.Buffer.tanhBackward,
          #[0, -0.0], #[6 * least, -3 * least], #[6 * least, -3 * least])] ++
      -- Binary64 rounds its sigmoid repair in stages, which misses these ties at `x = 0`. At
      -- `x = ±2^-130` and the smallest subnormal, `-|x| / 2` rounds below the least normal, and
      -- the exact slope, below `1 / 4`, takes a tie of `g / 4` toward zero.
      (if dtype == .float32 then
        let small : Float := (Float32.ofBits 0x00080000).toFloat
        [("sigmoid", LibTorch.Buffer.sigmoidBackward,
          #[0, -0.0, 0, 0, -200, -200, small, least, -small, small, small],
          #[6 * least, -6 * least, 10 * least, 14 * least, 1e30, -1e30, 6 * least, 6 * least,
            6 * least, -6 * least, 14 * least],
          #[2 * least, -2 * least, 2 * least, 4 * least, 0, -0.0, least, least, least, -least,
            3 * least])]
      else [])
    for (label, backward, xs, gs, expected) in cases do
      let x ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk xs) dtype
      let g ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk gs) dtype
      let dx ← IO.lazyPure fun _ => backward x g
      let got ← LibTorch.Buffer.toFloatArrayIO dx
      unless got.size == expected.size do
        throw <| IO.userError s!"{label} VJP ({repr dtype}): size {got.size}"
      for i in [:expected.size] do
        let (a, b) := (got.get! i, expected[i]!)
        unless a.toBits == b.toBits || (a.isNaN && b.isNaN) do
          throw <| IO.userError
            s!"{label} VJP ({repr dtype})[{i}]: got bits {a.toBits}, expected {b.toBits}"
      for buffer in #[dx, g, x] do
        discard <| LibTorch.Buffer.releaseIO buffer

private def square := fun (x : Float32) => x * x
private def positiveSquare := fun (x : Float32) => if x < 0 then 0 else x * x
private def squareWide := fun (x : Float) => x * x

private abbrev WideBinary := FloatLib.Floats.ExecFloat.Binary 15 112

/-- `x ^ n` by plain structural recursion, lowered from its equations without a fold helper such as
`Kernel.iterate`. -/
private def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

/-- `n` steps of `y ↦ y * y + 1` from `x`. Each step reads the previous state twice, which the
lowering has to share without repeating the recurrence. -/
private def evolve {α : Type} [One α] [Add α] [Mul α] (x : α) : Nat → α
  | 0 => x
  | n + 1 =>
      let y := evolve x n
      y * y + 1

private def refine (depth : UInt64) (x : Float32) : Nat → Float32
  | 0 => x
  | n + 1 => power (refine depth x n) depth.toNat

/-- `refine depth x rounds` at each entry `x` of `input` on the GPU, with both recurrence bounds as
parameters of the compiled function. -/
private def refinement (depth rounds : UInt64) (input : Tensor Float32 [257]) :
    IO (Tensor Float32 [257]) :=
  (fun x => refine depth x rounds.toNat).run input (device := .cuda)

private def sumPrefix (read : Kernel.Reader Float32) (start : UInt64) :
    Nat → Except Kernel.Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

/-- The inputs of the recursion checks: `-0` at each index divisible by `7` and `i % 5 - 2` at the
others, all exact in binary32. -/
private def recursionInput : Tensor Float32 [257] := Tensor.ofFn fun i =>
  if i.val % 7 == 0 then Float32.ofBits 0x80000000 else Float32.ofNat (i.val % 5) - 2

/-- Runtime depths and the long literal depth `4096` compile to loops. At each runtime depth the
GPU results have the bits of the same function run on the CPU. -/
private def checkRecursiveDepths : IO Unit := do
  let input := recursionInput
  for n in [:9] do
    let depth := n.toUInt64
    let f := fun (x : Float32) => power x depth.toNat
    let expected ← f.run input
    let actual ← f.run input (device := .cuda)
    for i in List.finRange 257 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError s!"custom recursion: changed result at depth {n}, entry {i}"
  for n in [:11] do
    let depth := n.toUInt64
    let f := fun (x : Float32) => evolve x depth.toNat
    let expected ← f.run input
    let actual ← f.run input (device := .cuda)
    for i in List.finRange 257 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError s!"custom recurrence: changed result at depth {n}, entry {i}"
  let three := fun (x : Float32) => evolve x 3
  let sample ← three.run ([1, 2] : Tensor Float32 [2]) (device := .cuda)
  unless sample[0] == 26 && sample[1] == 677 do
    throw <| IO.userError "custom recurrence: incorrect repeated square-plus-one update"
  let deep := fun (x : Float32) => power x 4096
  let deepResult ← deep.run ([1, -1] : Tensor Float32 [2]) (device := .cuda)
  unless deepResult[0] == 1 && deepResult[1] == 1 do
    throw <| IO.userError "custom recursion: failed the long loop"

/-- Each entry takes its own depth `i % 9`, and `refine` calls `power` inside its own recurrence,
both with the bits of the Lean evaluation. -/
private def checkRecursiveNesting : IO Unit := do
  let input := recursionInput
  let varying := Program.of (fun (read : Kernel.Reader Float32) (i : UInt64) => do
    let x ← read 0 i
    pure (power x (i % 9).toNat))
  let varied ← varying.run (.singleton input) [257] (device := .cuda)
  let nested ← refinement 2 3 input
  for i in List.finRange 257 do
    unless varied[i].toBits == (power input[i] (i.val % 9)).toBits &&
        nested[i].toBits == (refine 2 input[i] 3).toBits do
      throw <| IO.userError "custom recursion: changed per-entry depth or nested recurrence"

/-- Recursive source keeps native binary64 and configured binary128 precision on the GPU, where
`power x 7` has the bits of the CPU run in both formats. -/
private def checkRecursivePrecision : IO Unit := do
  let nativeWide : Tensor Float [3] := [1.0000000000000002, -3.5, -0.0]
  let nativeFunction := fun (x : Float) => power x 7
  let nativeResult ← nativeFunction.run nativeWide (device := .cuda)
  let nativeExpected ← nativeFunction.run nativeWide
  for i in List.finRange 3 do
    unless nativeResult[i].toBits == nativeExpected[i].toBits do
      throw <| IO.userError "custom recursion: changed native binary64 arithmetic"
  let a : WideBinary := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let precise : Tensor WideBinary [3] := [a, -a, 1]
  let f := fun (x : WideBinary) => power x 7
  let wide ← f.run precise (device := .cuda)
  let expected ← f.run precise
  for i in List.finRange 3 do
    unless Numeric.Word.toBits wide[i] == Numeric.Word.toBits expected[i] do
      throw <| IO.userError "custom recursion: narrowed configured arithmetic"

/-- Recursive reads keep the order of addition, so `2^24 + 1 - 2^24` added from the left is `0` in
binary32. They make no read at depth zero and reject an index past the end of the input. -/
private def checkRecursiveReads : IO Unit := do
  let rows : Tensor Float32 [2, 3] := [[16777216, 1, -16777216], [1, 2, 3]]
  let sum (width : UInt64) := Program.of
    (fun (read : Kernel.Reader Float32) (row : UInt64) =>
      sumPrefix read (row * width) width.toNat)
  let result ← (sum 3).run (.singleton rows) [2] (device := .cuda)
  unless result[0] == 0 && result[1] == 6 do
    throw <| IO.userError "custom recursive read: changed addition order"
  let empty : Tensor Float32 [0] := Tensor.zeros [0]
  let zero ← (sum 0).run (.singleton empty) [2] (device := .cuda)
  unless zero[0] == 0 && zero[1] == 0 do
    throw <| IO.userError "custom recursive read: evaluated a zero-depth step"
  let rejected ← try
    let _ ← (sum 4).run (.singleton rows) [2] (device := .cuda)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom recursive read: accepted an out-of-bounds access"

/-- Runs the recursion checks on the generated GPU loops themselves, which the proofs about their
Lean source do not cover. -/
private def checkRecursion : IO Unit := do
  checkRecursiveDepths
  checkRecursiveNesting
  checkRecursivePrecision
  checkRecursiveReads
  IO.println "  custom recursion: runtime depths, nested calls, precision and failed reads passed"

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- Compares full encodings, including signed zeros and NaN payloads, across the foreign boundary.
The inputs are all 400 pairs of 20 exceptional and rounding-boundary patterns and 512 pseudorandom
words. The reference is FloatLib's proved model, and the check covers the choice between native and
software arithmetic and the foreign arithmetic itself. -/
@[noinline, nospecialize] private def checkBinary {format : FloatFormat}
    {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code]
    (zero : ExecFloat (Configured.Family format code plan)) (label : String) : IO Unit := do
  let α := ExecFloat (Configured.Family format code plan)
  let exponentBits := format.expWidth
  let fractionBits := format.fracWidth
  let model (x : α) := ExecFloat.Binary.toModel x
  let scalar (x : Model format) : α := ExecFloat.Binary.ofModel x
  let sign := 2 ^ (exponentBits + fractionBits)
  let inf := (2 ^ exponentBits - 1) * 2 ^ fractionBits
  let quiet := 2 ^ (fractionBits - 1)
  let one := format.exponentBias * 2 ^ fractionBits
  let maximum := format.encoding.maxFiniteExponent exponentBits * 2 ^ fractionBits +
    (2 ^ fractionBits - if format.encoding == .finiteMaxNaN then 2 else 1)
  let patterns : Tensor Nat [20] :=
    [0, sign, 1, sign + 1, 2 ^ fractionBits - 1, 2 ^ fractionBits,
      one, sign + one, one - 1, one + 1, inf - 1, inf,
      sign + inf, inf + quiet, inf + 1, sign + inf + quiet + 1,
      maximum, sign + maximum, maximum - 1, sign + maximum - 1]
  let count := 912
  let word (i : Nat) : Nat := Id.run do
    let mut bits := 0
    for j in [:((exponentBits + fractionBits) / 64 + 1)] do
      let limb := ((i + 1) * 6364136223846793005 + (j + 1) * 1442695040888963407) % 2 ^ 64
      bits := bits + limb * 2 ^ (64 * j)
    return bits % (2 * sign)
  let left : Tensor α [count] := Tensor.generateFlat [count] fun i =>
    ExecFloat.Binary.ofNatBits (if h : i < 400 then patterns.getScalar ⟨i / 20, by omega⟩
      else word i)
  let right : Tensor α [count] := Tensor.generateFlat [count] fun i =>
    ExecFloat.Binary.ofNatBits (if i < 400 then patterns.getScalar ⟨i % 20, by omega⟩
      else word (i + 7919))
  -- The reference applies the scalar model to each pair, since complete FP8 lookup tables would add
  -- work unrelated to the foreign implementation under test.
  let reference (op : Model format → Model format → Model format) : Tensor α [count] :=
    Tensor.zipWith (fun x y => scalar (op (model x) (model y))) left right
  let check (operation : String) (actual expected : Tensor α [count]) : IO Unit := do
    for h : i in [:count] do
      let index : Fin count := ⟨i, h.upper⟩
      let observed := ExecFloat.Binary.toNatBits (actual.getScalar index)
      let target := ExecFloat.Binary.toNatBits (expected.getScalar index)
      unless observed == target do
        throw <| IO.userError
          (s!"custom {label} {operation}: encoding mismatch at {i}: " ++
            s!"got {observed}, expected {target}")
  check "add" (← (fun (x y : α) => x + y).zip left right (device := .cuda))
    (reference Model.add)
  check "sub" (← (fun (x y : α) => x - y).zip left right (device := .cuda))
    (reference Model.sub)
  check "mul" (← (fun (x y : α) => x * y).zip left right (device := .cuda))
    (reference Model.mul)
  check "div" (← (fun (x y : α) => x / y).zip left right (device := .cuda))
    (reference Model.div)
  check "mul-add" (← (fun (x y : α) => x * y + x).zip left right (device := .cuda))
    (reference fun x y => Model.add (Model.mul x y) x)
  check "neg" (← (fun (x : α) => -x).run left (device := .cuda))
    (Tensor.map (fun x => scalar (Model.neg (model x))) left)
  check "branch" (← (fun (x y : α) => if x < y then x else y).zip left right (device := .cuda))
    (reference fun x y => if Model.compare x y == some .lt then x else y)
  check "equality" (← (fun (x y : α) => if x == y then x else y).zip left right (device := .cuda))
    (reference fun x y => if Model.compare x y == some .eq then x else y)
  check "weak order" (← (fun (x y : α) => if x ≤ y then x else y).zip left right (device := .cuda))
    (reference fun x y =>
      if Model.compare x y == some .lt || Model.compare x y == some .eq then x else y)
  let _ ← (fun (x : α) => x * x).run (Tensor.full [0] zero) (device := .cuda)
  IO.println s!"  custom {label}: complete encodings matched FloatLib"

private def checkSmallPrecisions : IO Unit := do
  checkBinary (0 : ExecFloat.Binary 5 10) "binary16"
  checkBinary (0 : ExecFloat.Binary 8 7) "bfloat16"

private def checkTinyPrecisions : IO Unit := do
  checkBinary (0 : ExecFloat.Binary 4 3) "IEEE e4m3"
  checkBinary (0 : ExecFloat.Binary 5 2) "IEEE e5m2"

private def checkEncodings : IO Unit := do
  checkBinary (0 : ExecFloat.Binary 4 3 (encoding := .finiteMaxNaN))
    "e4m3 finite/NaN"
  checkBinary (0 : ExecFloat.Binary 4 3 (encoding := .finiteUnsignedZero))
    "e4m3 unsigned zero"
  checkBinary (0 : ExecFloat.Binary 4 3 (encoding := .finite))
    "fully finite e4m3"
  checkBinary (0 : ExecFloat.Binary 6 13 (bias := 17))
    "custom width and bias"
  -- These have the same widths as hardware formats but different arithmetic conventions.
  checkBinary (0 : ExecFloat.Binary 5 10 (bias := 14))
    "binary16 custom bias"
  checkBinary (0 : ExecFloat.Binary 8 7 (encoding := .finite))
    "fully finite bfloat16 width"

private def checkStandardPrecisions : IO Unit := do
  checkBinary (0 : ExecFloat.Binary 8 23) "configured binary32"
  checkBinary (0 : ExecFloat.Binary 11 52) "configured binary64"

private def checkWidePrecisions : IO Unit := do
  checkBinary (0 : WideBinary) "binary128"
  checkBinary (0 : ExecFloat.Binary 15 240) "256-bit storage"
  -- The low bits of `1 + 2^-100` cannot survive a binary64 upload. A captured constant and a
  -- sequential fold are checked as well.
  let a : WideBinary := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let exact : Tensor WideBinary [3] := [a, 1, -a]
  let squared ← (fun (x : WideBinary) => x * x).run exact (device := .cuda)
  unless Numeric.Word.toBits squared[0] == Numeric.Word.toBits (a * a) do
    throw <| IO.userError "custom binary128: lost precision beyond binary64"
  let sum := Program.of
    (fun (read : Kernel.Reader WideBinary) (_ : UInt64) =>
      Kernel.iterate (fun i acc => do
        let x ← read 0 i
        pure (acc + x)) 3 0 a)
  let expected ← sum.run (.singleton exact) [1]
  let actual ← sum.run (.singleton exact) [1] (device := .cuda)
  unless Numeric.Word.toBits actual[0] == Numeric.Word.toBits expected[0] do
    throw <| IO.userError "custom binary128: changed captured precision or sequential fold"

/-- Generated computations on the GPU: binary32 and binary64 maps and zips, binary32 folds, captured
parameters, a lazy branch, empty inputs, and the rejection of an out-of-bounds read and of an
unsupported function, followed by the precision and recursion checks. Results are compared with the
same Lean functions on the CPU or with known values. These native runs complement the proofs of
source equivalence at the NVRTC and FFI boundary. -/
private def checkCustomComputations : IO Unit := do
  let input : Tensor Float32 [513] := Tensor.ofFn fun i => Float32.ofNat (i.val % 17) - 8
  let squared ← square.run input (device := .cuda)
  let activated ← positiveSquare.run input (device := .cuda)
  for i in List.finRange 513 do
    unless squared[i].toBits == (square input[i]).toBits &&
        activated[i].toBits == (positiveSquare input[i]).toBits do
      throw <| IO.userError s!"custom binary32: incorrect result at {i}"
  let wide : Tensor Float [2] := [1.0000000000000002, -3.5]
  let squaredWide ← squareWide.run wide (device := .cuda)
  for i in List.finRange 2 do
    unless squaredWide[i].toBits == (squareWide wide[i]).toBits do
      throw <| IO.userError s!"custom binary64: incorrect result at {i}"
  let _ ← square.run (Tensor.zeros [0]) (device := .cuda)
  let rows : Tensor Float32 [2, 3] := [[16777216, 1, -16777216], [1, 2, 3]]
  let rowSum := Program.of
    (fun (read : Kernel.Reader Float32) (row : UInt64) =>
      Kernel.iterate (fun column acc => do
        let x ← read 0 (row * 3 + column)
        pure (acc + x)) 3 0 0)
  let sums ← rowSum.run (.singleton rows) [2] (device := .cuda)
  unless sums[0] == 0 && sums[1] == 6 do
    throw <| IO.userError "custom fold: changed sequential addition order"
  let guarded := Program.of
    (fun (read : Kernel.Reader Float32) (i : UInt64) =>
      if i == 0 then pure 7 else read 0 i)
  let noInput : Tensor Float32 [0] := Tensor.zeros [0]
  let guardedOutput ← guarded.run (.singleton noInput) [1] (device := .cuda)
  unless guardedOutput[0] == 7 do
    throw <| IO.userError "custom conditional: incorrect branch"
  let rejected ← try
    let _ ← guarded.run (.singleton noInput) [2] (device := .cuda)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom read: accepted an out-of-bounds native access"
  -- The weight changes between calls, so it must not be frozen at elaboration time. These
  -- comparisons test code generation and native execution, beyond the source-equivalence proofs.
  let weights : Tensor Float32 [4] := [-0.25, 0, 0.5, 1.25]
  for parameter in List.finRange 4 do
    let weight := weights[parameter]
    let f := fun (x y : Float32) => weight * x + (1 - weight) * y
    let expected ← f.zip input squared
    let actual ← f.zip input squared (device := .cuda)
    let activation := fun (x : Float32) => if x < 0 then weight * x else x
    let expectedActivation ← activation.run input
    let actualActivation ← activation.run input (device := .cuda)
    for i in List.finRange 513 do
      unless actual[i].toBits == expected[i].toBits &&
          actualActivation[i].toBits == expectedActivation[i].toBits do
        throw <| IO.userError s!"custom parameters: incorrect result at {i}"
  let sum := fun (x y : Float) => x + y
  let combined ← sum.zip wide wide (device := .cuda)
  for i in List.finRange 2 do
    unless combined[i].toBits == (sum wide[i] wide[i]).toBits do
      throw <| IO.userError s!"custom binary64 zip: incorrect result at {i}"
  let _ ← sum.zip (Tensor.zeros [0]) (Tensor.zeros [0]) (device := .cuda)
  for n in [:4] do
    let count := n.toUInt64
    let program := Program.of
      (fun (read : Kernel.Reader Float32) (row : UInt64) =>
        Kernel.iterate (fun col acc => do
          let x ← read 0 (row * 3 + col)
          pure (acc + x)) count.toNat 0 0)
    let expected ← program.run (.singleton rows) [2]
    let actual ← program.run (.singleton rows) [2] (device := .cuda)
    for i in List.finRange 2 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError "custom loop parameter: changed accumulation order"
  let unsupported := fun (x : Float) => x.sin
  let _ ← unsupported.run wide
  let rejected ← try
    let _ ← unsupported.run wide (device := .cuda)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom source: silently accepted an unsupported GPU function"
  checkSmallPrecisions
  checkTinyPrecisions
  checkEncodings
  checkStandardPrecisions
  checkWidePrecisions
  checkRecursion
  IO.println "  custom computations: precision, parameters, zip, branches, folds and bounds passed"

open LibTorch in
/-- `Buffer.scaledProdExp x y c` against `exp ((c * x) * y)` composed from separate ATen calls, bit
for bit, which checks the C ABI's cast of the scalar `c` and its order of multiplication. -/
private def checkScaledProductExp : IO Unit := do
  IO.println "=== scaledProdExp composition parity ==="
  -- Varied, signed, finite fixtures, kept in a range where `exp` stays finite.
  let xs : FloatArray := FloatArray.mk
    #[0.10, -0.20, 0.35, -0.50, 0.75, -0.90, 0.00, 1.00, -1.00, 0.42]
  let ys : FloatArray := FloatArray.mk
    #[0.90, -0.75, 0.50, -0.30, 0.15, -0.05, 1.00, -1.00, 0.25, -0.60]
  let n : UInt32 := xs.size.toUInt32
  let x ← Buffer.ofFloatArrayIO xs
  let y ← Buffer.ofFloatArrayIO ys
  -- Scalars spanning sign and magnitude, including the constant-one case `c = 0`.
  for c in (#[-2.0, 0.5, 3.25, -0.125, 1.0, 0.0] : Array Float) do
    let actual := Buffer.scaledProdExp x y c
    let composed := Buffer.exp (Buffer.mul (Buffer.mul (← Buffer.fullIO n c) x) y)
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

open LibTorch in
/-- The CUDA pullback `native` of `op` against its binary32 CPU rule, bit for bit up to a NaN's
payload. -/
private def checkSelectionPullback (name : String)
    (op : Op [.value [16], .value [16]] [.value [16]])
    (native : Buffer → Buffer → Buffer → Buffer × Buffer) (left right seed : Tensor Float32 [16]) :
    IO Unit := do
  let some rule := op.rule? (α := Float32) | throw <| IO.userError s!"{name}: no binary32 rule"
  let expected := rule.backward (left, right, ()) (seed, ())
  let upload := fun (t : Tensor Float32 [16]) =>
    Buffer.ofFloatArrayIO (FloatArray.mk ((t.to (Array Float32)).map Float32.toFloat))
  let (da, db) := native (← upload left) (← upload right) (← upload seed)
  for (port, buffer, want) in [("first", da, expected.1), ("second", db, expected.2.1)] do
    let got ← Buffer.toFloatArrayIO buffer
    let want := want.to (Array Float32)
    unless got.size == want.size do
      throw <| IO.userError s!"{name} pullback to the {port} input: size {got.size}"
    for i in [:want.size] do
      let actual := (got.get! i).toFloat32
      unless actual.toBits == want[i]!.toBits || (actual.isNaN && want[i]!.isNaN) do
        throw <| IO.userError <|
          s!"{name} pullback to the {port} input [{i}]: bits {actual.toBits}, " ++
            s!"CPU rule {want[i]!.toBits}"

/-- The CUDA `max` and `min` pullbacks select as the CPU rules: the strict loser receives `+0` at
every cotangent, the winner the cotangent, and a tie or a NaN operand half of it. -/
private def checkSelectionPullbacks : IO Unit := do
  IO.println "=== max/min pullback selection parity ==="
  let left : Tensor Float32 [16] := [2, 2, 2, 2, 2, 2, 1 / 0, 1, -0.0, 1, 1, 1, 1, 1, 0 / 0, 1]
  let right : Tensor Float32 [16] := [1, 1, 1, 1, 1, 1, -1 / 0, 1, 0, 1, 1, 1, 1, 1, 1, 0 / 0]
  let seed : Tensor Float32 [16] := [1 / 0, -1 / 0, 0 / 0, -1, -0.0, Float32.ofBits 0x7F7FFFFF,
    -1, 3, -6, Float32.ofBits 1, Float32.ofBits 3, Float32.ofBits 0x7F7FFFFF, -0.0, 1 / 0, 6, 6]
  checkSelectionPullback "max" (Op.max (s := [16]))
    LibTorch.Buffer.maxBackward left right seed
  checkSelectionPullback "min" (Op.min (s := [16]))
    LibTorch.Buffer.minBackward left right seed
  IO.println "  max/min pullbacks select as the CPU rules ✓"

open LibTorch in
/-- `Buffer.huberTerms` and `Buffer.huberSlope` at the pairs `cases` and the threshold `delta` in
`dtype`, against `Huber.term` and the slope of `Tensor.huberLossBackward` at the scalar `α`, bit
for bit up to a NaN's payload. -/
private def checkHuberAt {α : Type} [Storage α] [Context α] (name : String)
    (dtype : LibTorch.Dtype) (widen : α → Float) (delta : α)
    (cases : Array (α × α)) : IO Unit := do
  let n := cases.size
  let predictions := Tensor.generateFlat [n] fun i => (cases[i]!).1
  let targets := Tensor.generateFlat [n] fun i => (cases[i]!).2
  let upload := fun (xs : Array α) => Buffer.ofFloatArrayIO (FloatArray.mk (xs.map widen)) dtype
  let p ← upload (cases.map (·.1))
  let t ← upload (cases.map (·.2))
  let terms := cases.map fun (x, y) => Huber.term x y delta
  let slopes := (Tensor.huberLossBackward predictions targets delta false).to (Array α)
  for (output, buffer, want) in
      [("terms", Buffer.huberTerms p t (widen delta), terms),
        ("slopes", Buffer.huberSlope (Buffer.sub p t) (widen delta), slopes)] do
    let got ← Buffer.toFloatArrayIO buffer
    unless got.size == n && want.size == n do
      throw <| IO.userError s!"Huber {name} {output}: size {got.size}"
    for i in [:n] do
      let actual := got.get! i
      let expected := widen want[i]!
      unless actual.toBits == expected.toBits || (actual.isNaN && expected.isNaN) do
        throw <| IO.userError <|
          s!"Huber {name} {output} at delta {widen delta} [{i}]: bits {actual.toBits}, " ++
            s!"CPU rule {expected.toBits}"

/-- The CUDA Huber terms and slopes at signed zeros, infinities, a NaN, the branch joins, rounding
ties and overflow, and at zero, negative, NaN and infinite thresholds, in binary32 and binary64. -/
private def checkHuber : IO Unit := do
  IO.println "=== Huber terms and slopes parity ==="
  -- The smallest subnormal halves to a tie, rounded to +0.
  let pairs32 : Array (Float32 × Float32) := #[(0, 0), (-0.0, 0), (0, -0.0), (1 / 0, 0),
    (-1 / 0, 0), (1 / 0, 1 / 0), (0 / 0, 1), (1, 0), (-1, 0), (0.5, 0), (0.25, 0), (2, 0),
    (Float32.ofBits 1, 0)]
  let pairs64 : Array (Float × Float) := #[(0, 0), (-0.0, 0), (0, -0.0), (1 / 0, 0),
    (-1 / 0, 0), (1 / 0, 1 / 0), (0 / 0, 1), (1, 0), (-1, 0), (0.5, 0), (0.25, 0), (2, 0),
    (Float.ofBits 1, 0)]
  for delta in (#[1, 0.5, 0, -0.0, -1, 0 / 0, 1 / 0, -1 / 0] : Array Float32) do
    checkHuberAt "binary32" .float32 Float32.toFloat delta pairs32
  for delta in (#[1, 0.5, 0, -0.0, -1, 0 / 0, 1 / 0, -1 / 0] : Array Float) do
    checkHuberAt "binary64" .float64 id delta pairs64
  -- Ties: `x - delta / 2` at 2 ^ 24 - 1 / 2 (binary32) and 2 ^ 53 - 1 / 2 (binary64), and the
  -- half-square `(x / 2) * x` at 8392704.5 and 2 ^ 53 - 2 ^ 27 + 1 / 2. Overflow: the difference
  -- of finite operands (the weighted difference finite at delta 1 / 2, infinite at delta 1) and
  -- the half-square below an infinite threshold.
  checkHuberAt "binary32" .float32 Float32.toFloat 1 #[(16777216, 0), (3e38, -3e38)]
  checkHuberAt "binary32" .float32 Float32.toFloat 8192 #[(4097, 0)]
  checkHuberAt "binary32" .float32 Float32.toFloat 0.5 #[(3e38, -3e38)]
  checkHuberAt "binary32" .float32 Float32.toFloat (1 / 0) #[(3e19, 0)]
  checkHuberAt "binary64" .float64 id 1 #[(9007199254740992, 0), (1.5e308, -1.5e308)]
  checkHuberAt "binary64" .float64 id 134217728 #[(134217727, 0)]
  checkHuberAt "binary64" .float64 id 0.5 #[(1.5e308, -1.5e308)]
  checkHuberAt "binary64" .float64 id (1 / 0) #[(1e155, 0)]
  IO.println "  Huber terms and slopes match the CPU rules ✓"

private def customPolynomial (x : Tensor Float32 [3]) : Tensor Float32 [3] :=
  let y := x * x
  y * y + Tensor.full [3] 1

private def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (x * x)

private def weights : Tensor Float32 [3, 2] := [[1, 2], [3, 4], [5, 6]]

private def project (x : Tensor Float32 [2, 3]) : Tensor Float32 [2, 2] := x.matmul weights

/-- The tensor form of `evolve`, polymorphic in the shape, which also tests the homogeneous
tensor-addition bridge. -/
private def update {s : Shape} (x : Tensor Float32 s) : Nat → Tensor Float32 s
  | 0 => x
  | n + 1 => let y := update x n; y * y + Tensor.full s 1

private def checkRecordedRecurrence (steps : Nat) : IO Unit := do
  let input : Tensor Float32 [2] := [1, 2]
  let f := fun x => evolve x steps
  let cpuResult ← f.run input (grad := true)
  let gpuResult ← f.run input (device := .cuda) (grad := true)
  let cpuGradient ← cpuResult.backward
  let gpuGradient ← gpuResult.backward
  unless gpuResult.device == .cuda do
    throw <| IO.userError "custom recurrence: GPU recording silently fell back"
  for i in List.finRange 2 do
    unless cpuResult.value[i].toBits == gpuResult.value[i].toBits &&
        cpuGradient[i].toBits == gpuGradient[i].toBits do
      throw <| IO.userError "custom recurrence: GPU forward or recorded gradient disagrees"
  let tensor := fun (x : Tensor Float32 [2]) => update x steps
  let result ← tensor.run input (device := .cuda) (grad := true)
  let gradient ← result.backward
  unless result.device == .cuda do
    throw <| IO.userError "custom recurrence: tensor recording silently fell back"
  for i in List.finRange 2 do
    unless result.value[i].toBits == cpuResult.value[i].toBits &&
        gradient[i].toBits == cpuGradient[i].toBits do
      throw <| IO.userError "custom recurrence: tensor recording changed values or gradients"

/-- The scalar frontend records on the existing CUDA tape and returns no detached values. The shared
intermediate of `customPolynomial` uses the ordinary multiplication VJP twice, and the session owns
the cleanup. -/
@[noinline] private def checkNativeAutograd : IO Unit := do
  -- At `x = 0` the unused branch divides by zero. Recording has to choose the branch before
  -- execution and must not evaluate both and then mask an invalid result or cotangent.
  let guarded := fun (x : Float32) => if x == 0 then x else -(x * x / x)
  let values : Tensor Float32 [3] := [0, 2, -3]
  let reference ← guarded.run values (grad := true)
  let recorded ← guarded.run values (device := .cuda) (grad := true)
  let referenceGradient ← reference.backward
  let recordedGradient ← recorded.backward
  unless recorded.device == .cuda do
    throw <| IO.userError "custom branch: GPU recording silently fell back"
  for i in List.finRange 3 do
    unless reference.value[i].toBits == recorded.value[i].toBits &&
        referenceGradient[i].toBits == recordedGradient[i].toBits do
      throw <| IO.userError "custom branch: selected path or recorded gradient disagrees"
  let empty : Tensor Float32 [0] := Tensor.ofFn fun i => Fin.elim0 i
  let result ← guarded.run empty (device := .cuda) (grad := true)
  let _ : Tensor Float32 [0] ← result.backward
  unless result.device == .cuda do
    throw <| IO.userError "custom branch: empty tensor recording failed"
  let input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)
  let result ← customPolynomial.run input (device := .cuda) (grad := true)
  try
    unless result.device == .cuda do
      throw <| IO.userError "custom calculation: native CUDA recording silently fell back"
    let value := result.value
    let gradient ← result.backward
    unless value[0] == 2 && value[1] == 17 && value[2] == 82 &&
        gradient[0] == 4 && gradient[1] == 32 && gradient[2] == 108 do
      throw <| IO.userError "custom calculation: CUDA forward or recorded VJP disagrees"
  finally
    result.close
  let loss ← energy.run input (device := .cuda) (grad := true)
  let gradient ← loss.backward
  unless loss.value.item == (14 : Float32) &&
      gradient[0] == 2 && gradient[1] == 4 && gradient[2] == 6 do
    throw <| IO.userError "custom calculation: CUDA whole-tensor reduction disagrees"
  let rows : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]
  let forward ← project.run rows (device := .cuda)
  let result ← project.run rows (device := .cuda) (grad := true)
  let gradient ← result.backward
  unless forward[0][0] == 22 && forward[1][1] == 64 &&
      result.value[0][0] == 22 && result.value[1][1] == 64 &&
      gradient[0][0] == 3 && gradient[0][1] == 7 && gradient[0][2] == 11 &&
      gradient[1][0] == 3 && gradient[1][1] == 7 && gradient[1][2] == 11 do
    throw <| IO.userError "custom calculation: CUDA matrix product or recorded VJP disagrees"

  -- The input `1 + 2^-40` rounds to `1` in binary32. The forward, the accumulation at the shared
  -- node and the zero cotangent must all use the input's binary64 dtype, since a binary32 result
  -- widened after download loses these digits.
  let wide : Tensor Float [1] := Tensor.full [1] (Float.ofBits 0x3ff0000000001000)
  let square := fun (x : Float) => x * x
  let result ← square.run wide (device := .cuda) (grad := true)
  let gradient ← result.backward
  unless result.device == .cuda && result.value[0].toBits == (wide[0] * wide[0]).toBits &&
      gradient[0].toBits == (2 * wide[0]).toBits do
    throw <| IO.userError "custom calculation: binary64 forward or gradient narrowed"
  let constant := fun (_ : Float) => (1 : Float)
  let result ← constant.run wide (device := .cuda) (grad := true)
  let gradient ← result.backward
  unless result.value[0] == 1 && gradient[0] == 0 do
    throw <| IO.userError "custom calculation: disconnected binary64 cotangent failed"

  let quotient := fun (x : Float32) => -(x / (x + 1))
  let quotientCpu ← quotient.run input (grad := true)
  let quotientGpu ← quotient.run input (device := .cuda) (grad := true)
  let cpuGradient ← quotientCpu.backward
  let gpuGradient ← quotientGpu.backward
  for i in [:3] do
    unless quotientCpu.value[i]!.toBits == quotientGpu.value[i]!.toBits &&
        cpuGradient[i]!.toBits == gpuGradient[i]!.toBits do
      throw <| IO.userError "custom calculation: division/negation VJP disagrees"

/-- Encoded tensors keep all their configured digits through recorded forward and reverse steps. -/
@[noinline] private def checkConfiguredAutograd : IO Unit := do
  -- The low-order bit in this input is not representable in a native double. Compare the
  -- complete forward and gradient words, including accumulation of the shared input's VJPs.
  let a : WideBinary :=
    Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let configured : Tensor WideBinary [1] := Tensor.full [1] a
  let calculate := fun (x : WideBinary) => -(x * x / (x + 1))
  let cpuResult ← calculate.run configured (grad := true)
  let gpuResult ← calculate.run configured (device := .cuda) (grad := true)
  let cpuGradient ← cpuResult.backward
  let gpuGradient ← gpuResult.backward
  unless gpuResult.device == .cuda &&
      Numeric.Word.toBits gpuResult.value[0] == Numeric.Word.toBits cpuResult.value[0] &&
      Numeric.Word.toBits gpuGradient[0] == Numeric.Word.toBits cpuGradient[0] do
    throw <| IO.userError "custom calculation: configured GPU forward or VJP changed precision"
  let disconnected := fun (_ : WideBinary) => a
  let result ← disconnected.run configured (device := .cuda) (grad := true)
  let gradient ← result.backward
  unless result.device == .cuda && gradient[0] == 0 do
    throw <| IO.userError "custom calculation: configured disconnected cotangent failed"

  -- Scalar branch traversal slices and concatenates encoded words as well as native values.
  let guarded := fun (x : WideBinary) => if x == 0 then x else -(x * x / x)
  let values : Tensor WideBinary [3] := [0, a, -a]
  let reference ← guarded.run values (grad := true)
  let recorded ← guarded.run values (device := .cuda) (grad := true)
  let referenceGradient ← reference.backward
  let recordedGradient ← recorded.backward
  unless recorded.device == .cuda do
    throw <| IO.userError "custom branch: configured GPU recording silently fell back"
  for i in List.finRange 3 do
    unless Numeric.Word.toBits reference.value[i] == Numeric.Word.toBits recorded.value[i] &&
        Numeric.Word.toBits referenceGradient[i] == Numeric.Word.toBits recordedGradient[i] do
      throw <| IO.userError "custom branch: encoded traversal or VJP changed precision"

/-- Resident buffers reject mixed dtypes and preserve binary64 words through byte transfers. -/
@[noinline] private def checkRecordingBuffers : IO Unit := do
  let wide : Tensor Float [1] := Tensor.full [1] (Float.ofBits 0x3ff0000000001000)
  let buffer ← LibTorch.Buffer.ofFloatArrayIO
    (FloatArray.mk #[wide[0]]) .float64
  let wrong ← LibTorch.Buffer.fullIO 1 1 .float32
  try
    let value : LibTorch.SomeBuffer := { s := [1], buf := buffer }
    -- The resident graph runner holds a Float32 payload, while the tape is general in the dtype.
    -- It rejects input storage of another dtype before any graph operation can consume it.
    let opGraph : IR.Graph := { nodes := #[.input 0 (.value [1])], outputs := #[⟨0, 0⟩] }
    let refused ← try
      let _ ← opGraph.runBuffers {} #[⟨⟨.value, [1]⟩, buffer⟩]
      pure false
    catch error => pure (error.toString.contains "input 0 dtype must be binary32")
    unless refused do
      throw <| IO.userError "custom calculation: binary32 operator graph accepted binary64 storage"
    -- A binary64 value supplied where a frozen input is declared is refused before it is read.
    let frozenGraph : IR.Graph :=
      { nodes := #[.input 0 (.frozen [1])], outputs := #[⟨0, 0⟩] }
    let refused ← try
      let _ ← frozenGraph.runBuffers {} #[⟨⟨.value, [1]⟩, buffer⟩]
      pure false
    catch error => pure (error.toString.contains "input 0 dtype must be binary32")
    unless refused do
      throw <| IO.userError "custom calculation: a frozen input accepted binary64 storage"
    let seed : LibTorch.SomeBuffer := { s := [1], buf := wrong }
    let (tape, id) :=
      Tape.leaf .empty value (requiresGrad := true)
    if let .ok _ := Tape.backwardDenseAll tape id seed then
      throw <| IO.userError "custom calculation: mixed-dtype cotangent was accepted"
    let bytes ← LibTorch.Buffer.toBytesIO buffer
    let restored ← LibTorch.Buffer.ofBytesIO bytes .float64
    try
      let values ← LibTorch.Buffer.toFloatArrayIO restored
      unless bytes.size == 8 && values[0]!.toBits == wide[0].toBits do
        throw <| IO.userError "custom calculation: binary64 checkpoint narrowed"
    finally
      discard <| LibTorch.Buffer.releaseIO restored
  finally
    discard <| LibTorch.Buffer.releaseIO wrong
    discard <| LibTorch.Buffer.releaseIO buffer

private def checkCustomAutograd : IO Unit := do
  checkRecordedRecurrence 0
  checkRecordedRecurrence 3
  checkNativeAutograd
  checkConfiguredAutograd
  checkRecordingBuffers

/-- Whether `got` has the bits of `expected`, any NaN matching a NaN. -/
private def sameBits (got expected : Array Float) : Bool :=
  got.size == expected.size &&
    (got.zip expected).all fun (a, b) => if b.isNaN then a.isNaN else a.toBits == b.toBits

/-- `Op.dropout`'s CUDA entry against `Dropout.masked` at binary32 in both passes, bit for
bit with any NaN matching a NaN: an entry kept iff its sample is at least `p` (ties, a NaN sample),
signed zeros, infinities, NaNs, a subnormal, the one-division case `1 + 2^-22` at `p = 1/4`, and
`p` at `0`, `1/4`, `1/2`, `1 - 2^-24`, `1`, `3/2` and NaN. The cotangents are nonnegative or NaN,
so no gradient is `-0`. -/
private def checkDropout : IO Unit := do
  IO.println "== dropout against the specification =="
  let inf := Float.ofBits 0x7ff0000000000000
  let nan := Float.ofBits 0x7ff8000000000000
  let below := (Float32.ofBits 0x3F7FFFFF).toFloat
  let s : Shape := [9]
  let xs : Array Float :=
    #[-1.0, -0.0, 0.0, inf, -inf, nan, (Float32.ofBits 0x3F800002).toFloat, 3.0, 1e-45]
  let samples : Array Float := #[0.0, 0.25, 0.5, 0.75, below, nan, 0.25, 0.5, 0.1]
  let seeds : Array Float := #[1.0, 2.0, inf, nan, 1.0, 2.0, 1.0, 1.0, 1.0]
  let single (v : Array Float) : Tensor Float32 s :=
    Tensor.generateFlat s fun i => (v[i]!).toFloat32
  let widen (t : Tensor Float32 s) : Array Float := (t.to (Array Float32)).map Float32.toFloat
  for p in [0.0, 0.25, 0.5, below, 1.0, 1.5, nan] do
    let x ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk xs)
    let sample ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk samples)
    let (t1, xId) := Tape.leaf Tape.empty
      { s, buf := x : LibTorch.SomeBuffer }
    let (t2, sampleId) := Tape.leaf t1 { s, buf := sample }
      (requiresGrad := false)
    let (t3, yId) ← Harness.record t2 "dropout" (.dropout (s := s))
      (xId, sampleId, Tensor.scalar p, ())
    let y ← IO.ofExcept <| LibTorch.Tape.requireValue t3 yId s
    let seed : LibTorch.SomeBuffer :=
      { s, buf := ← LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk seeds) }
    let grads ← IO.ofExcept <| Tape.backwardDenseAll (t := t3) yId seed
    let gradient ← Harness.cudaGrad (s := s) grads xId
    let p32 := p.toFloat32
    let keep := Tensor.ge (single samples) (Tensor.full s p32)
    unless sameBits (← LibTorch.Buffer.toFloatArrayIO y).data
        (widen (Dropout.masked p32 keep (single xs))) do
      throw <| IO.userError s!"dropout forward at p = {p}"
    unless sameBits (gradient.to (Array Float))
        (widen (Dropout.masked p32 keep (single seeds))) do
      throw <| IO.userError s!"dropout backward at p = {p}"

/-- Runs the activation, pointwise VJP bit, `Buffer.scaledProdExp`, selection, Huber, custom
computation and dropout checks, then compares the CPU and CUDA tapes on the composite pass and on
GELU, and the fused AdamW step with its staged computation. -/
@[no_expose] def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: elementwise ==="
  runActivationNumerics
  checkPointwiseVJPBits
  checkScaledProductExp
  checkSelectionPullbacks
  checkHuber
  checkCustomComputations
  checkCustomAutograd
  checkDropout

  let s : Shape := [5]
  let a : Tensor Float s :=
    (Tensor.from #[0.10, -0.20, 0.30, -0.15, 0.05]).reshape [5] (by dsimp; decide)
  let b : Tensor Float s :=
    (Tensor.from #[0.20,  0.10, -0.25, 0.40, -0.05]).reshape [5] (by dsimp; decide)

  let scaleC : Float := 0.3
  let clampLo : Float := 1e-3
  let clampHi : Float := 10.0
  let eps : Float := 1e-6

  -- CPU tape
  let t0 : Tape (SomeTensor Float) := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) (.ofTensor a) (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) (.ofTensor b) (name := some "b")
  let (t3, u1, _) ← IO.ofExcept (Tape.apply t2 "add" (Op.add (s := s)) (aId, bId, ()))
  let (t4, u2, _) ← IO.ofExcept <|
    Tape.apply t3 "scale" (Op.scale (s := s)) (aId, Tensor.scalar scaleC, ())
  let (t5, u3, _) ← IO.ofExcept (Tape.apply t4 "sub" (Op.sub (s := s)) (u1, u2, ()))
  let (t6, u4, _) ← IO.ofExcept (Tape.apply t5 "mul" (Op.mul (s := s)) (u3, bId, ()))
  let (t7, u5, _) ← IO.ofExcept (Tape.apply t6 "max" (Op.max (s := s)) (u4, aId, ()))
  let (t8, u6, _) ← IO.ofExcept (Tape.apply t7 "min" (Op.min (s := s)) (u5, bId, ()))
  let (t9, u7, _) ← IO.ofExcept (Tape.apply t8 "relu" (Op.relu (s := s)) (u6, ()))
  let (t10, u8, _) ← IO.ofExcept (Tape.apply t9 "sigmoid" (Op.sigmoid (s := s)) (u7, ()))
  let (t11, u9, _) ← IO.ofExcept (Tape.apply t10 "tanh" (Op.tanh (s := s)) (u8, ()))
  let (t12, u10, _) ← IO.ofExcept <|
    Tape.apply t11 "softplus" (Op.softplus (s := s)) (u9, ())
  let (t13, u11, _) ← IO.ofExcept (Tape.apply t12 "exp" (Op.exp (s := s)) (u10, ()))
  let (t14, u12, _) ← IO.ofExcept (Tape.apply t13 "abs" (Op.abs (s := s)) (u11, ()))
  let (t15, u13, _) ← IO.ofExcept <|
    Tape.apply t14 "clamp" (Op.clamp (s := s))
      (u12, Tensor.scalar clampLo, Tensor.scalar clampHi, ())
  let (t16, u14, _) ← IO.ofExcept (Tape.apply t15 "sqrt" (Op.sqrt (s := s)) (u13, ()))
  let (t17, u15, _) ← IO.ofExcept <|
    Tape.apply t16 "inv" (Op.reciprocal (s := s)) (u14, ())
  let (t18, u16, _) ← IO.ofExcept (Tape.apply t17 "log" (Op.log (s := s)) (u14, ()))
  let (t19, u17, _) ← IO.ofExcept <|
    Tape.apply t18 "safe_log" (Op.safeLog (s := s)) (u14, Tensor.scalar eps, ())
  let (t20, u18, _) ← IO.ofExcept (Tape.apply t19 "add" (Op.add (s := s)) (u15, u16, ()))
  let (t21, u19, _) ← IO.ofExcept (Tape.apply t20 "add" (Op.add (s := s)) (u18, u17, ()))
  let (t22, outId, _) ← IO.ofExcept <|
    Tape.apply t21 "sum" (Op.reduce (s := s) .sum (List.finRange s.rank) false
      (fun h => absurd h (by decide))) (u19, ())

  let outCpu ← Harness.cpuValue (s := Shape.scalar) t22 outId
  let seedCpu : SomeTensor Float := SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpu ← IO.ofExcept (Tape.backwardDenseAll (V := SomeTensor Float) (t
      := t22) outId seedCpu)
  let dA_cpu ← Harness.cpuGrad (s := s) gradsCpu aId
  let dB_cpu ← Harness.cpuGrad (s := s) gradsCpu bId

  -- CUDA tape
  let t0c : Tape LibTorch.SomeBuffer :=
    Tape.empty
  let (t1c, aIdc) :=
    Tape.leaf (t := t0c) (Harness.tensorToSomeBuffer a) (name := some "a")
  let (t2c, bIdc) :=
    Tape.leaf (t := t1c) (Harness.tensorToSomeBuffer b) (name := some "b")
  let (t3c, u1c) ← Harness.record t2c "add" (.add (s := s)) (aIdc, bIdc, ())
  let (t4c, u2c) ← Harness.record t3c "scale" (.scale (s := s)) (aIdc, Tensor.scalar scaleC, ())
  let (t5c, u3c) ← Harness.record t4c "sub" (.sub (s := s)) (u1c, u2c, ())
  let (t6c, u4c) ← Harness.record t5c "mul" (.mul (s := s)) (u3c, bIdc, ())
  let (t7c, u5c) ← Harness.record t6c "max" (.max (s := s)) (u4c, aIdc, ())
  let (t8c, u6c) ← Harness.record t7c "min" (.min (s := s)) (u5c, bIdc, ())
  let (t9c, u7c) ← Harness.record t8c "relu" (.relu (s := s)) (u6c, ())
  let (t10c, u8c) ← Harness.record t9c "sigmoid" (.sigmoid (s := s)) (u7c, ())
  let (t11c, u9c) ← Harness.record t10c "tanh" (.tanh (s := s)) (u8c, ())
  let (t12c, u10c) ← Harness.record t11c "softplus" (.softplus (s := s)) (u9c, ())
  let (t13c, u11c) ← Harness.record t12c "exp" (.exp (s := s)) (u10c, ())
  let (t14c, u12c) ← Harness.record t13c "abs" (.abs (s := s)) (u11c, ())
  let (t15c, u13c) ← Harness.record t14c "clamp" (.clamp (s := s))
    (u12c, Tensor.scalar clampLo, Tensor.scalar clampHi, ())
  let (t16c, u14c) ← Harness.record t15c "sqrt" (.sqrt (s := s)) (u13c, ())
  let (t17c, u15c) ← Harness.record t16c "inv" (.reciprocal (s := s)) (u14c, ())
  let (t18c, u16c) ← Harness.record t17c "log" (.log (s := s)) (u14c, ())
  let (t19c, u17c) ← Harness.record t18c "safe_log" (.safeLog (s := s))
    (u14c, Tensor.scalar eps, ())
  let (t20c, u18c) ← Harness.record t19c "add" (.add (s := s)) (u15c, u16c, ())
  let (t21c, u19c) ← Harness.record t20c "add" (.add (s := s)) (u18c, u17c, ())
  let (t22c, outIdc) ← Harness.record t21c "sum" (Harness.sumAll s) (u19c, ())

  let outCuda ← Harness.cudaValue (s := Shape.scalar) t22c outIdc
  let seedCuda : LibTorch.SomeBuffer :=
    { s := Shape.scalar, buf := ← LibTorch.Buffer.fullIO 1 1.0 }
  let gradsCuda ← IO.ofExcept
    (Tape.backwardDenseAll (t := t22c) outIdc seedCuda)
  let dA_cuda ← Harness.cudaGrad (s := s) gradsCuda aIdc
  let dB_cuda ← Harness.cudaGrad (s := s) gradsCuda bIdc

  Harness.assertTensorApprox (s := Shape.scalar) "elementwise forward" outCuda outCpu (tol := 2e-3)
  Harness.assertTensorApprox (s := s) "elementwise backward dA" dA_cuda dA_cpu (tol := 2e-3)
  Harness.assertTensorApprox (s := s) "elementwise backward dB" dB_cuda dB_cpu (tol := 2e-3)

  -- GELU is one semantic tape node: ATen's tanh `gelu` forward and a staged ATen backward. Check
  -- both against the spec-backed CPU tape over the nonlinear center and saturated tails.
  let geluShape : Shape := [7]
  let geluInput : Tensor Float geluShape :=
    (Tensor.from #[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]).reshape [7] (by dsimp; decide)
  let geluCpu0 : Tape (SomeTensor Float) := Tape.empty
  let (geluCpu1, geluCpuInputId) :=
    Tape.leaf (t := geluCpu0) (.ofTensor geluInput) (name := some "gelu input")
  let (geluCpu2, geluCpuOutputId, _) ← IO.ofExcept <|
    Tape.apply geluCpu1 "gelu" (Op.gelu (s := geluShape)) (geluCpuInputId, ())
  let geluCpuOutput ← Harness.cpuValue (s := geluShape) geluCpu2 geluCpuOutputId
  let geluSeedCpu : SomeTensor Float :=
    SomeTensor.ofTensor (Tensor.full (α := Float) geluShape 1.0)
  let geluCpuGrads ← IO.ofExcept <|
    Tape.backwardDenseAll (V := SomeTensor Float) (t := geluCpu2) geluCpuOutputId geluSeedCpu
  let geluCpuGrad ← Harness.cpuGrad (s := geluShape) geluCpuGrads geluCpuInputId

  let geluCuda0 : Tape LibTorch.SomeBuffer
      := Tape.empty
  let (geluCuda1, geluCudaInputId) :=
    Tape.leaf
      (t := geluCuda0) (Harness.tensorToSomeBuffer geluInput) (name := some "gelu input")
  let (geluCuda2, geluCudaOutputId) ←
    Harness.record geluCuda1 "gelu" (.gelu (s := geluShape)) (geluCudaInputId, ())
  let geluCudaOutput ← Harness.cudaValue (s := geluShape) geluCuda2 geluCudaOutputId
  let geluSeedCuda : LibTorch.SomeBuffer :=
    { s := geluShape, buf := ← LibTorch.Buffer.fullIO 7 1.0 }
  let geluCudaGrads ← IO.ofExcept <|
    Tape.backwardDenseAll
      (t := geluCuda2) geluCudaOutputId geluSeedCuda
  let geluCudaGrad ← Harness.cudaGrad (s := geluShape) geluCudaGrads geluCudaInputId

  Harness.assertTensorApprox
    (s := geluShape) "GELU forward" geluCudaOutput geluCpuOutput (tol := 3e-5)
  Harness.assertTensorApprox
    (s := geluShape) "GELU backward" geluCudaGrad geluCpuGrad (tol := 3e-5)

  -- The fused `Buffer.adamStep` must agree within `1e-6` with the staged AdamW computation it
  -- replaces.
  let optimShape : Shape := [4]
  let params : Tensor Float optimShape :=
    (Tensor.from #[1.0, -2.0, 0.5, 4.0]).reshape [4] (by dsimp; decide)
  let gradient : Tensor Float optimShape :=
    (Tensor.from #[0.2, -0.1, 0.4, -0.3]).reshape [4] (by dsimp; decide)
  let firstMoment : Tensor Float optimShape :=
    (Tensor.from #[0.01, -0.02, 0.03, -0.04]).reshape [4] (by dsimp; decide)
  let secondMoment : Tensor Float optimShape :=
    (Tensor.from #[0.2, 0.1, 0.4, 0.3]).reshape [4] (by dsimp; decide)
  let paramsBuf := Harness.tensorToBuffer params
  let gradientBuf := Harness.tensorToBuffer gradient
  let firstMomentBuf := Harness.tensorToBuffer firstMoment
  let secondMomentBuf := Harness.tensorToBuffer secondMoment
  let beta1 : Float := 0.9
  let beta2 : Float := 0.999
  let oneMinusBeta1 : Float := 1.0 - beta1
  let oneMinusBeta2 : Float := 1.0 - beta2
  let firstMomentCorrection : Float := 1.0 / (1.0 - beta1)
  let secondMomentCorrection : Float := 1.0 / (1.0 - beta2)
  let epsilon : Float := 1e-8
  let lr : Float := 3e-4
  let weightDecay : Float := 0.1

  let mScaled := LibTorch.Buffer.scale firstMomentBuf beta1
  let expectedM := LibTorch.Buffer.axpy mScaled gradientBuf oneMinusBeta1
  let gradientSquared := LibTorch.Buffer.mul gradientBuf gradientBuf
  let vScaled := LibTorch.Buffer.scale secondMomentBuf beta2
  let expectedV := LibTorch.Buffer.axpy vScaled gradientSquared oneMinusBeta2
  let mHat := LibTorch.Buffer.scale expectedM firstMomentCorrection
  let vHat := LibTorch.Buffer.scale expectedV secondMomentCorrection
  let sqrtVHat := LibTorch.Buffer.sqrtClampNegative vHat
  let epsilonBuf ← LibTorch.Buffer.fullIO 4 epsilon
  let denominator := LibTorch.Buffer.add sqrtVHat epsilonBuf
  let normalizedUpdate := LibTorch.Buffer.div mHat denominator
  let decayedParams :=
    LibTorch.Buffer.axpy paramsBuf paramsBuf (-(lr * weightDecay))
  let expectedParams :=
    LibTorch.Buffer.axpy decayedParams normalizedUpdate (-lr)
  let (gotParams, gotM, gotV) := LibTorch.Buffer.adamStep
    paramsBuf gradientBuf firstMomentBuf secondMomentBuf
    beta1 oneMinusBeta1 beta2 oneMinusBeta2
    (1.0 - beta1) (1.0 - beta2) epsilon
    (some (-(lr * weightDecay))) (-lr)

  Harness.assertTensorApprox (s := optimShape) "fused AdamW parameters"
    (← Harness.bufferToTensor gotParams) (← Harness.bufferToTensor expectedParams) (tol := 1e-6)
  Harness.assertTensorApprox (s := optimShape) "fused AdamW first moment"
    (← Harness.bufferToTensor gotM) (← Harness.bufferToTensor expectedM) (tol := 1e-6)
  Harness.assertTensorApprox (s := optimShape) "fused AdamW second moment"
    (← Harness.bufferToTensor gotV) (← Harness.bufferToTensor expectedV) (tol := 1e-6)

end Elementwise
end CUDA
end TorchLean.Test
