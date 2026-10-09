/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops
public import NN.Tensor
public import NN.Tests.Runtime.Cuda.Utils
import NN.Kernel
import NN.API.Precision

/-!
# CUDA Kernel Coverage: Elementwise Ops

Focused activation value/VJP regressions and a composite forward/backward test cover
(`add/sub/mul/scale/abs/sqrt/clamp/max/min/relu/sigmoid/tanh/gelu/softplus/exp/log/inv/safe_log`)
plus `sum`. Generated custom computations also exercise NVRTC, scalar precision, lazy branches,
sequential accumulation, empty tensors and checked native reads.
-/

@[expose] public section

namespace Tests
namespace Cuda
namespace Elementwise

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd
open scoped NN.Kernel

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
      throw <| IO.userError s!"{label}: got {got}, expected {expected} (rtol {rtol})"

def assertActivationArray (label : String) (got expected : FloatArray)
    (rtol : Float := 3e-6) : IO Unit := do
  unless got.size == expected.size do
    throw <| IO.userError s!"{label}: size {got.size}, expected {expected.size}"
  for i in [:expected.size] do
    assertActivationValue s!"{label}[{i}]" (got.get! i) (expected.get! i) rtol

def checkActivation
    (label : String)
    (forward : Runtime.Autograd.LibTorch.Buffer → Runtime.Autograd.LibTorch.Buffer)
    (record : (s : Shape) → Runtime.Autograd.LibTorch.Tape → Nat →
      Result (Runtime.Autograd.LibTorch.Tape × Nat))
    (derivative : Float32 → Float32)
    (inputs expected seeds : Array Float) : IO Unit := do
  unless inputs.size == expected.size && inputs.size == seeds.size do
    throw <| IO.userError s!"{label}: inconsistent test data"
  let s : Shape := [inputs.size]
  let input ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk inputs)
  let original ← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input
  let direct ← IO.lazyPure fun _ => forward input
  assertActivationArray s!"{label} buffer"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO direct) (FloatArray.mk expected)
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO direct
  assertActivationArray s!"{label} borrowed input"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input) original

  let (t1, xId) := Runtime.Autograd.LibTorch.Tape.leaf
    Runtime.Autograd.LibTorch.Tape.empty { s := s, buf := input }
  let result ← IO.lazyPure fun _ => record s t1 xId
  let (t2, yId) ← IO.ofExcept result
  let some node := t2.getNode? yId
    | throw <| IO.userError s!"{label}: missing activation node"
  unless t2.nodes.size == 2 && node.parents == #[xId] &&
      node.requiresGrad && node.ownsValue do
    throw <| IO.userError s!"{label}: expected one owned, differentiable unary node"
  assertActivationArray s!"{label} tape"
    (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
  let expectedGrad := FloatArray.mk <| expected.mapIdx fun i y =>
    (seeds[i]!.toFloat32 * derivative y.toFloat32).toFloat
  let retained ← Runtime.Autograd.LibTorch.Buffer.memory
  for pass in [:2] do
    let seed ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO (FloatArray.mk seeds)
    -- Exercise the recorded closure before accumulation, including signed-zero cotangents.
    let localResult ← IO.lazyPure fun _ => node.backward { s := s, buf := seed }
    let contributions ← IO.ofExcept localResult
    let some (parentId, contribution) := contributions[0]?
      | throw <| IO.userError s!"{label}: missing VJP contribution"
    unless contributions.size == 1 && parentId == xId && contribution.s == s do
      throw <| IO.userError s!"{label}: malformed VJP contribution"
    assertActivationArray s!"{label} local VJP, pass {pass}"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO contribution.buf) expectedGrad
        (rtol := 3e-5)
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO contribution.buf
    assertActivationArray s!"{label} borrowed cotangent"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO seed) (FloatArray.mk seeds)
    let grads ← Runtime.Autograd.LibTorch.Tape.backwardSparse t2 yId
      { s := s, buf := seed } (fun id => id == xId)
    let some grad := grads.get? xId
      | throw <| IO.userError s!"{label}: missing tape gradient"
    assertActivationArray s!"{label} tape VJP, pass {pass}"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO grad.buf) expectedGrad (rtol := 3e-5)
    Runtime.Autograd.LibTorch.Tape.releaseSparseGrads grads
    assertActivationArray s!"{label} input after backward"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO input) original
    assertActivationArray s!"{label} output after backward"
      (← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO node.value.buf) (FloatArray.mk expected)
    let after ← Runtime.Autograd.LibTorch.Buffer.memory
    unless after.liveBytes == retained.liveBytes do
      throw <| IO.userError s!"{label}: backward retained temporary payloads on pass {pass}"
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO node.value.buf
  discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO input

/-- Direct activations retain tiny values, selected nonfinite behavior, and TorchLean's VJPs. -/
def runActivationNumerics : IO Unit := do
  IO.println "== direct tanh/sigmoid numerical regressions =="
  let inf := Float.ofBits 0x7ff0000000000000
  let nan := Float.ofBits 0x7ff8000000000000
  let tiny : Array Float :=
    #[1e-30, -1e-30, 1e-12, -1e-12, 1e-8, -1e-8, 1e-5, -1e-5]
  let central : Array Float := #[-3.0, -1.0, -0.5, 0.5, 1.0, 3.0]
  let inputs := (tiny ++ #[0.0, -0.0] ++ central ++ #[-100.0, 100.0, -inf, inf, nan]).map
    (fun (x : Float) => x.toFloat32.toFloat)
  let seedPattern : Array Float := #[2.0, -0.5, 1.5, -3.0, 0.25]
  let seeds := inputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  let tanhValues := inputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat
  let tanhNode := fun s t id => Runtime.Autograd.LibTorch.Tape.tanh (s := s) t id
  let sigmoidNode := fun s t id => Runtime.Autograd.LibTorch.Tape.sigmoid (s := s) t id
  let tanhDerivative := fun (y : Float32) => 1 - y * y
  let sigmoidDerivative := fun (y : Float32) => y * (1 - y)
  checkActivation "tanh tiny/center/tails" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode
    tanhDerivative inputs tanhValues seeds

  -- Test normal negative-tail values separately from ATen's saturated float32 tail.
  let sigmoidInputs := inputs ++ #[-80.0, -20.0, 20.0, 80.0]
  let sigmoidValues := sigmoidInputs.map fun x =>
    if x == -100.0 then 0.0 else (Activation.Math.sigmoidSpec x).toFloat32.toFloat
  let sigmoidSeeds := sigmoidInputs.mapIdx fun i _ => seedPattern[i % seedPattern.size]!
  checkActivation "sigmoid tiny/center/tails" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative sigmoidInputs sigmoidValues sigmoidSeeds

  -- Infinite cotangents at saturated outputs still multiply by zero and produce NaN.
  let specialInputs : Array Float :=
    #[0.0, -0.0, 1.0, -1.0, 100.0, -100.0, inf, -inf, nan, 0.5, -100.0]
  let specialSeeds : Array Float :=
    #[-0.0, 0.0, inf, -inf, inf, -inf, -0.0, -2.0, 0.0, nan, nan]
  checkActivation "tanh nonfinite VJP" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode tanhDerivative
    specialInputs (specialInputs.map fun x => (MathFunctions.tanh x).toFloat32.toFloat)
    specialSeeds
  checkActivation "sigmoid nonfinite VJP" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative specialInputs
    (specialInputs.map fun x =>
      if x == -100.0 then 0.0 else (Activation.Math.sigmoidSpec x).toFloat32.toFloat)
    specialSeeds
  checkActivation "tanh empty" Runtime.Autograd.LibTorch.Buffer.tanh tanhNode
    tanhDerivative #[] #[] #[]
  checkActivation "sigmoid empty" Runtime.Autograd.LibTorch.Buffer.sigmoid sigmoidNode
    sigmoidDerivative #[] #[] #[]

private def square := fun (x : Float32) => x * x
private def positiveSquare := fun (x : Float32) => if x < 0 then 0 else x * x
private def squareWide := fun (x : Float) => x * x

private abbrev WideBinary := FloatLib.Floats.ExecFloat.Binary 15 112

/-- Ordinary recursive source exercises equation-based lowering, not an explicit fold helper. -/
private def power {α : Type} [One α] [Mul α] (x : α) : Nat → α
  | 0 => 1
  | n + 1 => power x n * x

/-- Reusing an intermediate state must neither duplicate the recurrence nor lose its sharing. -/
private def evolve {α : Type} [One α] [Add α] [Mul α] (x : α) : Nat → α
  | 0 => x
  | n + 1 =>
      let y := evolve x n
      y * y + 1

private def refine (depth : UInt64) (x : Float32) : Nat → Float32
  | 0 => x
  | n + 1 => power (refine depth x n) depth.toNat

/-- Keep both recurrence bounds as parameters of the compiled function. -/
private def refinement (depth rounds : UInt64) (input : Tensor Float32 [257]) :
    IO (Tensor Float32 [257]) :=
  (fun x => refine depth x rounds.toNat).run input (device := gpu)

private def sumPrefix (read : NN.Kernel.Reader Float32) (start : UInt64) :
    Nat → Except NN.Kernel.Error Float32
  | 0 => pure 0
  | n + 1 => do
      let previous ← sumPrefix read start n
      let x ← read 0 (start + n.toUInt64)
      pure (previous + x)

/-- Include negative zero among the exact inputs to the recursive functions. -/
private def recursionInput : Tensor Float32 [257] := Tensor.ofFn fun i =>
  if i.val % 7 == 0 then Float32.ofBits 0x80000000 else Float32.ofNat (i.val % 5) - 2

/-- Runtime and long literal depths use loops; the original function is the bitwise oracle. -/
private def checkRecursiveDepths : IO Unit := do
  let input := recursionInput
  for n in [:9] do
    let depth := n.toUInt64
    let f := fun (x : Float32) => power x depth.toNat
    let expected ← f.run input
    let actual ← f.run input (device := gpu)
    for i in List.finRange 257 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError s!"custom recursion: changed result at depth {n}, entry {i}"
  for n in [:11] do
    let depth := n.toUInt64
    let f := fun (x : Float32) => evolve x depth.toNat
    let expected ← f.run input
    let actual ← f.run input (device := gpu)
    for i in List.finRange 257 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError s!"custom recurrence: changed result at depth {n}, entry {i}"
  let three := fun (x : Float32) => evolve x 3
  let sample ← three.run ([1, 2] : Tensor Float32 [2]) (device := gpu)
  unless sample[0] == 26 && sample[1] == 677 do
    throw <| IO.userError "custom recurrence: incorrect repeated square-plus-one update"
  let deep := fun (x : Float32) => power x 4096
  let deepResult ← deep.run ([1, -1] : Tensor Float32 [2]) (device := gpu)
  unless deepResult[0] == 1 && deepResult[1] == 1 do
    throw <| IO.userError "custom recursion: failed the long loop"

/-- Each lane can choose its depth, and recursive calls can occur inside another recurrence. -/
private def checkRecursiveNesting : IO Unit := do
  let input := recursionInput
  let varying := Program.of (fun (read : NN.Kernel.Reader Float32) (i : UInt64) => do
    let x ← read 0 i
    pure (power x (i % 9).toNat))
  let varied ← varying.run (Arguments.empty.push input) [257] (device := gpu)
  let nested ← refinement 2 3 input
  for i in List.finRange 257 do
    unless varied[i].toBits == (power input[i] (i.val % 9)).toBits &&
        nested[i].toBits == (refine 2 input[i] 3).toBits do
      throw <| IO.userError "custom recursion: changed per-entry depth or nested recurrence"

/-- Recursive source retains native binary64 and configured binary128 precision. -/
private def checkRecursivePrecision : IO Unit := do
  let nativeWide : Tensor Float [3] := [1.0000000000000002, -3.5, -0.0]
  let nativeFunction := fun (x : Float) => power x 7
  let nativeResult ← nativeFunction.run nativeWide (device := gpu)
  let nativeExpected ← nativeFunction.run nativeWide
  for i in List.finRange 3 do
    unless nativeResult[i].toBits == nativeExpected[i].toBits do
      throw <| IO.userError "custom recursion: changed native binary64 arithmetic"
  let a : WideBinary := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let precise : Tensor WideBinary [3] := [a, -a, 1]
  let f := fun (x : WideBinary) => power x 7
  let wide ← f.run precise (device := gpu)
  let expected ← f.run precise
  for i in List.finRange 3 do
    unless NN.Kernel.Scalar.encode wide[i] == NN.Kernel.Scalar.encode expected[i] do
      throw <| IO.userError "custom recursion: narrowed configured arithmetic"

/-- Recursive reads retain addition order, avoid zero-depth reads and reject invalid indices. -/
private def checkRecursiveReads : IO Unit := do
  let rows : Tensor Float32 [2, 3] := [[16777216, 1, -16777216], [1, 2, 3]]
  let sum (width : UInt64) := Program.of
    (fun (read : NN.Kernel.Reader Float32) (row : UInt64) =>
      sumPrefix read (row * width) width.toNat)
  let result ← (sum 3).run (Arguments.empty.push rows) [2] (device := gpu)
  unless result[0] == 0 && result[1] == 6 do
    throw <| IO.userError "custom recursive read: changed addition order"
  let empty : Tensor Float32 [0] := Tensor.zeros [0]
  let zero ← (sum 0).run (Arguments.empty.push empty) [2] (device := gpu)
  unless zero[0] == 0 && zero[1] == 0 do
    throw <| IO.userError "custom recursive read: evaluated a zero-depth step"
  let rejected ← try
    let _ ← (sum 4).run (Arguments.empty.push rows) [2] (device := gpu)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom recursive read: accepted an out-of-bounds access"

/-- Exercise the generated GPU loops, rather than only their Lean-source proofs. -/
private def checkRecursion : IO Unit := do
  checkRecursiveDepths
  checkRecursiveNesting
  checkRecursivePrecision
  checkRecursiveReads
  IO.println "  custom recursion: runtime depths, nested calls, precision and failed reads passed"

open FloatLib.Floats.Formats.BinaryInterchange (Model)

/-- Compare full encodings, including signed zeros and NaN payloads, across the foreign boundary.
The input includes every pair of exceptional/rounding-boundary patterns plus distributed words.
This checks native/software selection and foreign arithmetic, not FloatLib's proved model. -/
@[noinline, nospecialize] private def checkBinary {α : Type} [Storage α] [NN.Kernel.Scalar α]
    [Zero α] [Add α] [Sub α] [Mul α] [Div α] [Neg α] [BEq α]
    [LT α] [LE α] [DecidableLT α] [DecidableLE α]
    (label : String) : IO Unit := do
  let .binary format := (inferInstance : NN.Kernel.Scalar α).precision |
    throw <| IO.userError "configured precision check requires a binary descriptor"
  let exponentBits := format.expWidth
  let fractionBits := format.fracWidth
  let model (x : α) := Model.ofNatBits (fmt := format) (NN.Kernel.Scalar.encode x)
  let scalar (x : Model format) : α :=
    NN.Kernel.Scalar.decode x.toNatBits
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
    NN.Kernel.Scalar.decode (if h : i < 400 then patterns.getScalar ⟨i / 20, by omega⟩
      else word i)
  let right : Tensor α [count] := Tensor.generateFlat [count] fun i =>
    NN.Kernel.Scalar.decode (if i < 400 then patterns.getScalar ⟨i % 20, by omega⟩
      else word (i + 7919))
  -- Compare directly with the scalar model; initializing complete FP8 lookup tables is unrelated
  -- to the foreign implementation under test.
  let reference (op : Model format → Model format → Model format) : Tensor α [count] :=
    Tensor.map2Spec (fun x y => scalar (op (model x) (model y))) left right
  let check (operation : String) (actual expected : Tensor α [count]) : IO Unit := do
    for h : i in [:count] do
      let index : Fin count := ⟨i, h.upper⟩
      let observed := NN.Kernel.Scalar.encode (actual.getScalar index)
      let target := NN.Kernel.Scalar.encode (expected.getScalar index)
      unless observed == target do
        throw <| IO.userError
          (s!"custom {label} {operation}: encoding mismatch at {i}: " ++
            s!"got {observed}, expected {target}")
  check "add" (← (fun (x y : α) => x + y).zip left right (device := gpu))
    (reference Model.add)
  check "sub" (← (fun (x y : α) => x - y).zip left right (device := gpu))
    (reference Model.sub)
  check "mul" (← (fun (x y : α) => x * y).zip left right (device := gpu))
    (reference Model.mul)
  check "div" (← (fun (x y : α) => x / y).zip left right (device := gpu))
    (reference Model.div)
  check "mul-add" (← (fun (x y : α) => x * y + x).zip left right (device := gpu))
    (reference fun x y => Model.add (Model.mul x y) x)
  check "neg" (← (fun (x : α) => -x).run left (device := gpu))
    (Tensor.map (fun x => scalar (Model.neg (model x))) left)
  check "branch" (← (fun (x y : α) => if x < y then x else y).zip left right (device := gpu))
    (reference fun x y => if Model.compare x y == some .lt then x else y)
  check "equality" (← (fun (x y : α) => if x == y then x else y).zip left right (device := gpu))
    (reference fun x y => if Model.compare x y == some .eq then x else y)
  check "weak order" (← (fun (x y : α) => if x ≤ y then x else y).zip left right (device := gpu))
    (reference fun x y =>
      if Model.compare x y == some .lt || Model.compare x y == some .eq then x else y)
  let _ ← (fun (x : α) => x * x).run (Tensor.zeros [0]) (device := gpu)
  IO.println s!"  custom {label}: complete encodings matched FloatLib"

private def checkSmallPrecisions : IO Unit := do
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 5 10) "binary16"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 8 7) "bfloat16"

private def checkTinyPrecisions : IO Unit := do
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 4 3) "IEEE e4m3"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 5 2) "IEEE e5m2"

private def checkEncodings : IO Unit := do
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 4 3 (encoding := .finiteMaxNaN))
    "e4m3 finite/NaN"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 4 3 (encoding := .finiteUnsignedZero))
    "e4m3 unsigned zero"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 4 3 (encoding := .finite))
    "fully finite e4m3"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 6 13 (bias := 17))
    "custom width and bias"
  -- These have the same widths as hardware formats but different arithmetic conventions.
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 5 10 (bias := 14))
    "binary16 custom bias"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 8 7 (encoding := .finite))
    "fully finite bfloat16 width"

private def checkStandardPrecisions : IO Unit := do
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 8 23) "configured binary32"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 11 52) "configured binary64"

private def checkWidePrecisions : IO Unit := do
  checkBinary (α := WideBinary) "binary128"
  checkBinary (α := FloatLib.Floats.ExecFloat.Binary 15 240) "256-bit storage"
  -- These low bits cannot survive a binary64 upload. Check captured constants and a fold too.
  let a : WideBinary := Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let exact : Tensor WideBinary [3] := [a, 1, -a]
  let squared ← (fun (x : WideBinary) => x * x).run exact (device := gpu)
  unless NN.Kernel.Scalar.encode squared[0] == NN.Kernel.Scalar.encode (a * a) do
    throw <| IO.userError "custom binary128: lost precision beyond binary64"
  let sum := Program.of
    (fun (read : NN.Kernel.Reader WideBinary) (_ : UInt64) =>
      NN.Kernel.iterate (fun i acc => do
        let x ← read 0 i
        pure (acc + x)) 3 0 a)
  let expected ← sum.run (Arguments.empty.push exact) [1]
  let actual ← sum.run (Arguments.empty.push exact) [1] (device := gpu)
  unless NN.Kernel.Scalar.encode actual[0] == NN.Kernel.Scalar.encode expected[0] do
    throw <| IO.userError "custom binary128: changed captured precision or sequential fold"

/-- Native execution checks complement source-equivalence proofs at the NVRTC/FFI boundary. -/
private def checkCustomComputations : IO Unit := do
  let input : Tensor Float32 [513] := Tensor.ofFn fun i => Float32.ofNat (i.val % 17) - 8
  let squared ← square.run input (device := gpu)
  let activated ← positiveSquare.run input (device := gpu)
  for i in List.finRange 513 do
    unless squared[i].toBits == (square input[i]).toBits &&
        activated[i].toBits == (positiveSquare input[i]).toBits do
      throw <| IO.userError s!"custom binary32: incorrect result at {i}"
  let wide : Tensor Float [2] := [1.0000000000000002, -3.5]
  let squaredWide ← squareWide.run wide (device := gpu)
  for i in List.finRange 2 do
    unless squaredWide[i].toBits == (squareWide wide[i]).toBits do
      throw <| IO.userError s!"custom binary64: incorrect result at {i}"
  let _ ← square.run (Tensor.zeros [0]) (device := gpu)
  let rows : Tensor Float32 [2, 3] := [[16777216, 1, -16777216], [1, 2, 3]]
  let rowSum := Program.of
    (fun (read : NN.Kernel.Reader Float32) (row : UInt64) =>
      NN.Kernel.iterate (fun column acc => do
        let x ← read 0 (row * 3 + column)
        pure (acc + x)) 3 0 0)
  let sums ← rowSum.run (Arguments.empty.push rows) [2] (device := gpu)
  unless sums[0] == 0 && sums[1] == 6 do
    throw <| IO.userError "custom fold: changed sequential addition order"
  let guarded := Program.of
    (fun (read : NN.Kernel.Reader Float32) (i : UInt64) =>
      if i == 0 then pure 7 else read 0 i)
  let noInput : Tensor Float32 [0] := Tensor.zeros [0]
  let guardedOutput ← guarded.run (Arguments.empty.push noInput) [1] (device := gpu)
  unless guardedOutput[0] == 7 do
    throw <| IO.userError "custom conditional: incorrect branch"
  let rejected ← try
    let _ ← guarded.run (Arguments.empty.push noInput) [2] (device := gpu)
    pure false
  catch _ => pure true
  unless rejected do
    throw <| IO.userError "custom read: accepted an out-of-bounds native access"
  -- Parameters change between calls; they must not be frozen at elaboration time. These
  -- comparisons exercise code generation and native execution, beyond source-equivalence proofs.
  let weights : Tensor Float32 [4] := [-0.25, 0, 0.5, 1.25]
  for parameter in List.finRange 4 do
    let weight := weights[parameter]
    let f := fun (x y : Float32) => weight * x + (1 - weight) * y
    let expected ← f.zip input squared
    let actual ← f.zip input squared (device := gpu)
    let activation := fun (x : Float32) => if x < 0 then weight * x else x
    let expectedActivation ← activation.run input
    let actualActivation ← activation.run input (device := gpu)
    for i in List.finRange 513 do
      unless actual[i].toBits == expected[i].toBits &&
          actualActivation[i].toBits == expectedActivation[i].toBits do
        throw <| IO.userError s!"custom parameters: incorrect result at {i}"
  let sum := fun (x y : Float) => x + y
  let combined ← sum.zip wide wide (device := gpu)
  for i in List.finRange 2 do
    unless combined[i].toBits == (sum wide[i] wide[i]).toBits do
      throw <| IO.userError s!"custom binary64 zip: incorrect result at {i}"
  let _ ← sum.zip (Tensor.zeros [0]) (Tensor.zeros [0]) (device := gpu)
  for n in [:4] do
    let count := n.toUInt64
    let program := Program.of
      (fun (read : NN.Kernel.Reader Float32) (row : UInt64) =>
        NN.Kernel.iterate (fun col acc => do
          let x ← read 0 (row * 3 + col)
          pure (acc + x)) count.toNat 0 0)
    let expected ← program.run (Arguments.empty.push rows) [2]
    let actual ← program.run (Arguments.empty.push rows) [2] (device := gpu)
    for i in List.finRange 2 do
      unless actual[i].toBits == expected[i].toBits do
        throw <| IO.userError "custom loop parameter: changed accumulation order"
  let unsupported := fun (x : Float) => x.sin
  let _ ← unsupported.run wide
  let rejected ← try
    let _ ← unsupported.run wide (device := gpu)
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

open Runtime.Autograd.LibTorch in
/-- Check the C ABI's scalar cast and multiplication order against separate ATen calls. -/
private def checkScaledProductExponential : IO Unit := do
  IO.println "=== scaledProdExp composition parity ==="
  -- Varied, signed, finite fixtures, kept in a range where `exp` stays finite.
  let xs : FloatArray := FloatArray.mk
    #[0.10, -0.20, 0.35, -0.50, 0.75, -0.90, 0.00, 1.00, -1.00, 0.42]
  let ys : FloatArray := FloatArray.mk
    #[0.90, -0.75, 0.50, -0.30, 0.15, -0.05, 1.00, -1.00, 0.25, -0.60]
  let n : UInt32 := xs.size.toUInt32
  let x := Buffer.ofFloatArray xs
  let y := Buffer.ofFloatArray ys
  -- Scalars spanning sign and magnitude, including the constant-one case `c = 0`.
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

private def customPolynomial (x : Tensor Float32 [3]) : Tensor Float32 [3] :=
  let y := x * x
  y * y + Tensor.full [3] 1

private def energy (x : Tensor Float32 [3]) : Float32 := Tensor.sum (x * x)

private def weights : Tensor Float32 [3, 2] := [[1, 2], [3, 4], [5, 6]]

private def project (x : Tensor Float32 [2, 3]) : Tensor Float32 [2, 2] := x.matmul weights

/-- A shape-polymorphic recurrence also exercises the homogeneous tensor-addition bridge. -/
private def update {s : Shape} (x : Tensor Float32 s) : Nat → Tensor Float32 s
  | 0 => x
  | n + 1 => let y := update x n; y * y + Tensor.full s 1

private def checkRecordedRecurrence (steps : Nat) : IO Unit := do
  let input : Tensor Float32 [2] := [1, 2]
  let f := fun x => evolve x steps
  let cpuResult ← f.run input (grad := true)
  let gpuResult ← f.run input (device := gpu) (grad := true)
  let cpuGradient ← cpuResult.backward
  let gpuGradient ← gpuResult.backward
  unless gpuResult.device == gpu do
    throw <| IO.userError "custom recurrence: GPU recording silently fell back"
  for i in List.finRange 2 do
    unless cpuResult.value[i].toBits == gpuResult.value[i].toBits &&
        cpuGradient[i].toBits == gpuGradient[i].toBits do
      throw <| IO.userError "custom recurrence: GPU forward or recorded gradient disagrees"
  let tensor := fun (x : Tensor Float32 [2]) => update x steps
  let result ← tensor.run input (device := gpu) (grad := true)
  let gradient ← result.backward
  unless result.device == gpu do
    throw <| IO.userError "custom recurrence: tensor recording silently fell back"
  for i in List.finRange 2 do
    unless result.value[i].toBits == cpuResult.value[i].toBits &&
        gradient[i].toBits == cpuGradient[i].toBits do
      throw <| IO.userError "custom recurrence: tensor recording changed values or gradients"

/-- The scalar frontend must record on the existing CUDA tape, not return detached values.
Its shared intermediate uses the ordinary multiplication VJP twice; cleanup stays session-owned. -/
private def checkCustomAutograd : IO Unit := do
  checkRecordedRecurrence 0
  checkRecordedRecurrence 3
  -- The unused branch divides by zero. Recording must choose the branch before execution,
  -- rather than evaluate both and try to mask an invalid result or cotangent afterwards.
  let guarded := fun (x : Float32) => if x == 0 then x else -(x * x / x)
  let values : Tensor Float32 [3] := [0, 2, -3]
  let reference ← guarded.run values (grad := true)
  let recorded ← guarded.run values (device := gpu) (grad := true)
  let referenceGradient ← reference.backward
  let recordedGradient ← recorded.backward
  unless recorded.device == gpu do
    throw <| IO.userError "custom branch: GPU recording silently fell back"
  for i in List.finRange 3 do
    unless reference.value[i].toBits == recorded.value[i].toBits &&
        referenceGradient[i].toBits == recordedGradient[i].toBits do
      throw <| IO.userError "custom branch: selected path or recorded gradient disagrees"
  let empty : Tensor Float32 [0] := Tensor.ofFn fun i => Fin.elim0 i
  let result ← guarded.run empty (device := gpu) (grad := true)
  let _ : Tensor Float32 [0] ← result.backward
  unless result.device == gpu do
    throw <| IO.userError "custom branch: empty tensor recording failed"
  let input : Tensor Float32 [3] := Tensor.ofFn fun i => Float32.ofNat (i.val + 1)
  let result ← customPolynomial.run input (device := gpu) (grad := true)
  try
    unless result.device == gpu do
      throw <| IO.userError "custom calculation: native CUDA recording silently fell back"
    let value := result.value
    let gradient ← result.backward
    unless value[0] == 2 && value[1] == 17 && value[2] == 82 &&
        gradient[0] == 4 && gradient[1] == 32 && gradient[2] == 108 do
      throw <| IO.userError "custom calculation: CUDA forward or recorded VJP disagrees"
  finally
    result.close
  let loss ← energy.run input (device := gpu) (grad := true)
  let gradient ← loss.backward
  unless loss.value.item == (14 : Float32) &&
      gradient[0] == 2 && gradient[1] == 4 && gradient[2] == 6 do
    throw <| IO.userError "custom calculation: CUDA whole-tensor reduction disagrees"
  let rows : Tensor Float32 [2, 3] := [[1, 2, 3], [4, 5, 6]]
  let forward ← project.run rows (device := gpu)
  let result ← project.run rows (device := gpu) (grad := true)
  let gradient ← result.backward
  unless forward[0][0] == 22 && forward[1][1] == 64 &&
      result.value[0][0] == 22 && result.value[1][1] == 64 &&
      gradient[0][0] == 3 && gradient[0][1] == 7 && gradient[0][2] == 11 &&
      gradient[1][0] == 3 && gradient[1][1] == 7 && gradient[1][2] == 11 do
    throw <| IO.userError "custom calculation: CUDA matrix product or recorded VJP disagrees"

  -- These digits disappear in FP32. Forward, shared-node accumulation and zero cotangents
  -- must all use the input's binary64 dtype, not merely widen the downloaded result.
  let wide : Tensor Float [1] := Tensor.full [1] (Float.ofBits 0x3ff0000000001000)
  let square := fun (x : Float) => x * x
  let result ← square.run wide (device := gpu) (grad := true)
  let gradient ← result.backward
  unless result.device == gpu && result.value[0].toBits == (wide[0] * wide[0]).toBits &&
      gradient[0].toBits == (2 * wide[0]).toBits do
    throw <| IO.userError "custom calculation: binary64 forward or gradient narrowed"
  let constant := fun (_ : Float) => (1 : Float)
  let result ← constant.run wide (device := gpu) (grad := true)
  let gradient ← result.backward
  unless result.value[0] == 1 && gradient[0] == 0 do
    throw <| IO.userError "custom calculation: disconnected binary64 cotangent failed"

  let quotient := fun (x : Float32) => -(x / (x + 1))
  let quotientCpu ← quotient.run input (grad := true)
  let quotientGpu ← quotient.run input (device := gpu) (grad := true)
  let cpuGradient ← quotientCpu.backward
  let gpuGradient ← quotientGpu.backward
  for i in [:3] do
    unless quotientCpu.value[i]!.toBits == quotientGpu.value[i]!.toBits &&
        cpuGradient[i]!.toBits == gpuGradient[i]!.toBits do
      throw <| IO.userError "custom calculation: division/negation VJP disagrees"

  -- The low-order bit in this input is not representable in a native double. Compare the
  -- complete forward and gradient words, including accumulation of the shared input's VJPs.
  let a : WideBinary :=
    Rat.cast (1 + 1 / (2 ^ 100 : Nat) : Rat)
  let configured : Tensor WideBinary [1] := Tensor.full [1] a
  let calculate := fun (x : WideBinary) => -(x * x / (x + 1))
  let cpuResult ← calculate.run configured (grad := true)
  let gpuResult ← calculate.run configured (device := gpu) (grad := true)
  let cpuGradient ← cpuResult.backward
  let gpuGradient ← gpuResult.backward
  unless gpuResult.device == gpu &&
      NN.Kernel.Scalar.encode gpuResult.value[0] == NN.Kernel.Scalar.encode cpuResult.value[0] &&
      NN.Kernel.Scalar.encode gpuGradient[0] == NN.Kernel.Scalar.encode cpuGradient[0] do
    throw <| IO.userError "custom calculation: configured GPU forward or VJP changed precision"
  let disconnected := fun (_ : WideBinary) => a
  let result ← disconnected.run configured (device := gpu) (grad := true)
  let gradient ← result.backward
  unless result.device == gpu && gradient[0] == 0 do
    throw <| IO.userError "custom calculation: configured disconnected cotangent failed"

  -- Scalar branch traversal slices and concatenates encoded words as well as native values.
  let guarded := fun (x : WideBinary) => if x == 0 then x else -(x * x / x)
  let values : Tensor WideBinary [3] := [0, a, -a]
  let reference ← guarded.run values (grad := true)
  let recorded ← guarded.run values (device := gpu) (grad := true)
  let referenceGradient ← reference.backward
  let recordedGradient ← recorded.backward
  unless recorded.device == gpu do
    throw <| IO.userError "custom branch: configured GPU recording silently fell back"
  for i in List.finRange 3 do
    unless NN.Kernel.Scalar.encode reference.value[i] ==
          NN.Kernel.Scalar.encode recorded.value[i] &&
        NN.Kernel.Scalar.encode referenceGradient[i] ==
          NN.Kernel.Scalar.encode recordedGradient[i] do
      throw <| IO.userError "custom branch: encoded traversal or VJP changed precision"

  let buffer ← Runtime.Autograd.LibTorch.Buffer.ofFloatArrayIO
    (FloatArray.mk #[wide[0]]) .float64
  let wrong ← Runtime.Autograd.LibTorch.Buffer.fullIO 1 1 .float32
  try
    let value : Runtime.Autograd.LibTorch.AnyBuffer := { s := [1], buf := buffer }
    -- The canonical forward runner has a Float32 payload, unlike the dtype-general tape.
    -- Reject mismatched input storage before any graph operation can consume it.
    let graph : NN.IR.Graph :=
      { nodes := #[{ id := 0, parents := #[], kind := .input, outShape := [1] }] }
    let rejected ← try
      let _ ← graph.runBuffers {} (fun _ => some value)
      pure false
    catch error => pure (error.toString.contains "input dtype must be binary32")
    unless rejected do
      throw <| IO.userError "custom calculation: binary32 graph accepted binary64 storage"
    let seed : Runtime.Autograd.LibTorch.AnyBuffer := { s := [1], buf := wrong }
    let (tape, id) :=
      Runtime.Autograd.LibTorch.Tape.leaf .empty value (requiresGrad := true)
    if let .ok _ := Runtime.Autograd.LibTorch.Tape.backwardDenseAll tape id seed then
      throw <| IO.userError "custom calculation: mixed-dtype cotangent was accepted"
    let bytes ← Runtime.Autograd.LibTorch.Buffer.toBytesIO buffer
    let restored ← Runtime.Autograd.LibTorch.Buffer.ofBytesIO bytes .float64
    try
      let values ← Runtime.Autograd.LibTorch.Buffer.toFloatArrayIO restored
      unless bytes.size == 8 && values[0]!.toBits == wide[0].toBits do
        throw <| IO.userError "custom calculation: binary64 checkpoint narrowed"
    finally
      discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO restored
  finally
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO wrong
    discard <| Runtime.Autograd.LibTorch.Buffer.releaseIO buffer

/-- Exercise native elementwise VJPs, generated computations, and composed buffer operations. -/
@[no_expose] def run : IO Unit := do
  IO.println "=== CUDA kernel coverage: elementwise ==="
  runActivationNumerics
  checkScaledProductExponential
  checkCustomComputations
  checkCustomAutograd

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
  let t0 : Tape Float := Tape.empty
  let (t1, aId) := Tape.leaf (t := t0) a (name := some "a")
  let (t2, bId) := Tape.leaf (t := t1) b (name := some "b")
  let (t3, u1) ← IO.ofExcept (Tape.add (α := Float) (t := t2) (s := s) aId bId)
  let (t4, u2) ← IO.ofExcept (Tape.scale (α := Float) (t := t3) (s := s) aId scaleC)
  let (t5, u3) ← IO.ofExcept (Tape.sub (α := Float) (t := t4) (s := s) u1 u2)
  let (t6, u4) ← IO.ofExcept (Tape.mul (α := Float) (t := t5) (s := s) u3 bId)
  let (t7, u5) ← IO.ofExcept (Tape.max (α := Float) (t := t6) (s := s) u4 aId)
  let (t8, u6) ← IO.ofExcept (Tape.min (α := Float) (t := t7) (s := s) u5 bId)
  let (t9, u7) ← IO.ofExcept (Tape.relu (α := Float) (t := t8) (s := s) u6)
  let (t10, u8) ← IO.ofExcept (Tape.sigmoid (α := Float) (t := t9) (s := s) u7)
  let (t11, u9) ← IO.ofExcept (Tape.tanh (α := Float) (t := t10) (s := s) u8)
  let (t12, u10) ← IO.ofExcept (Tape.softplus (α := Float) (t := t11) (s := s) u9)
  let (t13, u11) ← IO.ofExcept (Tape.exp (α := Float) (t := t12) (s := s) u10)
  let (t14, u12) ← IO.ofExcept (Tape.abs (α := Float) (t := t13) (s := s) u11)
  let (t15, u13) ← IO.ofExcept (Tape.clamp (α := Float) (t := t14) (s := s) u12 clampLo clampHi)
  let (t16, u14) ← IO.ofExcept (Tape.sqrt (α := Float) (t := t15) (s := s) u13)
  let (t17, u15) ← IO.ofExcept (Tape.inv (α := Float) (t := t16) (s := s) u14)
  let (t18, u16) ← IO.ofExcept (Tape.log (α := Float) (t := t17) (s := s) u14)
  let (t19, u17) ← IO.ofExcept (Tape.safeLog (α := Float) (t := t18) (s := s) u14 (ε := eps))
  let (t20, u18) ← IO.ofExcept (Tape.add (α := Float) (t := t19) (s := s) u15 u16)
  let (t21, u19) ← IO.ofExcept (Tape.add (α := Float) (t := t20) (s := s) u18 u17)
  let (t22, outId) ← IO.ofExcept (Tape.sum (α := Float) (t := t21) (s := s) u19)

  let outCpu ← Utils.cpuValue (s := Shape.scalar) t22 outId
  let seedCpu : Spec.SomeTensor Float := Spec.SomeTensor.ofTensor (Tensor.scalar 1.0)
  let gradsCpu ← IO.ofExcept (Tape.backwardDenseAll (α := Float) (t := t22) outId seedCpu)
  let dA_cpu ← Utils.cpuGrad (s := s) gradsCpu aId
  let dB_cpu ← Utils.cpuGrad (s := s) gradsCpu bId

  -- CUDA tape
  let t0c : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (t1c, aIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t0c) (Utils.tensorToAnyBuffer a) (name := some "a")
  let (t2c, bIdc) :=
    Runtime.Autograd.LibTorch.Tape.leaf (t := t1c) (Utils.tensorToAnyBuffer b) (name := some "b")
  let (t3c, u1c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t2c) (s := s)
    aIdc bIdc)
  let (t4c, u2c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.scale (t := t3c) (s := s) aIdc scaleC)
  let (t5c, u3c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sub (t := t4c) (s := s) u1c u2c)
  let (t6c, u4c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.mul (t := t5c) (s := s) u3c bIdc)
  let (t7c, u5c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.max (t := t6c) (s := s) u4c aIdc)
  let (t8c, u6c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.min (t := t7c) (s := s) u5c bIdc)
  let (t9c, u7c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.relu (t := t8c) (s := s) u6c)
  let (t10c, u8c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sigmoid (t := t9c) (s := s) u7c)
  let (t11c, u9c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.tanh (t := t10c) (s := s) u8c)
  let (t12c, u10c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.softplus (t := t11c)
    (s := s) u9c)
  let (t13c, u11c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.exp (t := t12c) (s := s) u10c)
  let (t14c, u12c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.abs (t := t13c) (s := s) u11c)
  let (t15c, u13c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.clamp (t := t14c) (s := s) u12c clampLo clampHi)
  let (t16c, u14c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sqrt (t := t15c) (s := s) u13c)
  let (t17c, u15c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.inv (t := t16c) (s := s) u14c)
  let (t18c, u16c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.log (t := t17c) (s := s) u14c)
  let (t19c, u17c) ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.safeLog (t := t18c) (s := s) u14c eps)
  let (t20c, u18c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t19c) (s :=
    s) u15c u16c)
  let (t21c, u19c) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.add (t := t20c) (s :=
    s) u18c u17c)
  let (t22c, outIdc) ← IO.ofExcept (Runtime.Autograd.LibTorch.Tape.sum (t := t21c) (s
    := s) u19c)

  let outCuda ← Utils.cudaValue (s := Shape.scalar) t22c outIdc
  let seedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := Shape.scalar, buf := Runtime.Autograd.LibTorch.Buffer.full 1 1.0 }
  let gradsCuda ← IO.ofExcept
    (Runtime.Autograd.LibTorch.Tape.backwardDenseAll (t := t22c) outIdc seedCuda)
  let dA_cuda ← Utils.cudaGrad (s := s) gradsCuda aIdc
  let dB_cuda ← Utils.cudaGrad (s := s) gradsCuda bIdc

  Utils.assertTensorApprox (s := Shape.scalar) "elementwise forward" outCuda outCpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dA" dA_cuda dA_cpu (tol := 2e-3)
  Utils.assertTensorApprox (s := s) "elementwise backward dB" dB_cuda dB_cpu (tol := 2e-3)

  -- GELU is one semantic tape node and one pointwise kernel in each direction. Check both against
  -- the spec-backed CPU tape over the nonlinear center and saturated tails.
  let geluShape : Shape := [7]
  let geluInput : Tensor Float geluShape :=
    (Tensor.from #[-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0]).reshape [7] (by dsimp; decide)
  let geluCpu0 : Tape Float := Tape.empty
  let (geluCpu1, geluCpuInputId) :=
    Tape.leaf (t := geluCpu0) geluInput (name := some "gelu input")
  let (geluCpu2, geluCpuOutputId) ← IO.ofExcept <|
    Tape.gelu (α := Float) (t := geluCpu1) (s := geluShape) geluCpuInputId
  let geluCpuOutput ← Utils.cpuValue (s := geluShape) geluCpu2 geluCpuOutputId
  let geluSeedCpu : Spec.SomeTensor Float :=
    Spec.SomeTensor.ofTensor (Tensor.full (α := Float) geluShape 1.0)
  let geluCpuGrads ← IO.ofExcept <|
    Tape.backwardDenseAll (α := Float) (t := geluCpu2) geluCpuOutputId geluSeedCpu
  let geluCpuGrad ← Utils.cpuGrad (s := geluShape) geluCpuGrads geluCpuInputId

  let geluCuda0 : Runtime.Autograd.LibTorch.Tape := Runtime.Autograd.LibTorch.Tape.empty
  let (geluCuda1, geluCudaInputId) :=
    Runtime.Autograd.LibTorch.Tape.leaf
      (t := geluCuda0) (Utils.tensorToAnyBuffer geluInput) (name := some "gelu input")
  let (geluCuda2, geluCudaOutputId) ← IO.ofExcept <|
    Runtime.Autograd.LibTorch.Tape.gelu (t := geluCuda1) (s := geluShape) geluCudaInputId
  let geluCudaOutput ← Utils.cudaValue (s := geluShape) geluCuda2 geluCudaOutputId
  let geluSeedCuda : Runtime.Autograd.LibTorch.AnyBuffer :=
    { s := geluShape, buf := Runtime.Autograd.LibTorch.Buffer.full 7 1.0 }
  let geluCudaGrads ← IO.ofExcept <|
    Runtime.Autograd.LibTorch.Tape.backwardDenseAll
      (t := geluCuda2) geluCudaOutputId geluSeedCuda
  let geluCudaGrad ← Utils.cudaGrad (s := geluShape) geluCudaGrads geluCudaInputId

  Utils.assertTensorApprox
    (s := geluShape) "fused GELU forward" geluCudaOutput geluCpuOutput (tol := 3e-5)
  Utils.assertTensorApprox
    (s := geluShape) "fused GELU backward" geluCudaGrad geluCpuGrad (tol := 3e-5)

  -- The fused optimizer primitive must preserve the staged AdamW computation it replaces.
  let optimShape : Shape := [4]
  let params : Tensor Float optimShape :=
    (Tensor.from #[1.0, -2.0, 0.5, 4.0]).reshape [4] (by dsimp; decide)
  let gradient : Tensor Float optimShape :=
    (Tensor.from #[0.2, -0.1, 0.4, -0.3]).reshape [4] (by dsimp; decide)
  let firstMoment : Tensor Float optimShape :=
    (Tensor.from #[0.01, -0.02, 0.03, -0.04]).reshape [4] (by dsimp; decide)
  let secondMoment : Tensor Float optimShape :=
    (Tensor.from #[0.2, 0.1, 0.4, 0.3]).reshape [4] (by dsimp; decide)
  let paramsBuf := Utils.tensorToBuffer params
  let gradientBuf := Utils.tensorToBuffer gradient
  let firstMomentBuf := Utils.tensorToBuffer firstMoment
  let secondMomentBuf := Utils.tensorToBuffer secondMoment
  let beta1 : Float := 0.9
  let beta2 : Float := 0.999
  let oneMinusBeta1 : Float := 1.0 - beta1
  let oneMinusBeta2 : Float := 1.0 - beta2
  let firstMomentCorrection : Float := 1.0 / (1.0 - beta1)
  let secondMomentCorrection : Float := 1.0 / (1.0 - beta2)
  let epsilon : Float := 1e-8
  let learningRate : Float := 3e-4
  let weightDecay : Float := 0.1

  let mScaled := Runtime.Autograd.LibTorch.Buffer.scale firstMomentBuf beta1
  let expectedM := Runtime.Autograd.LibTorch.Buffer.axpy mScaled gradientBuf oneMinusBeta1
  let gradientSquared := Runtime.Autograd.LibTorch.Buffer.mul gradientBuf gradientBuf
  let vScaled := Runtime.Autograd.LibTorch.Buffer.scale secondMomentBuf beta2
  let expectedV := Runtime.Autograd.LibTorch.Buffer.axpy vScaled gradientSquared oneMinusBeta2
  let mHat := Runtime.Autograd.LibTorch.Buffer.scale expectedM firstMomentCorrection
  let vHat := Runtime.Autograd.LibTorch.Buffer.scale expectedV secondMomentCorrection
  let sqrtVHat := Runtime.Autograd.LibTorch.Buffer.sqrt vHat
  let epsilonBuf := Runtime.Autograd.LibTorch.Buffer.full 4 epsilon
  let denominator := Runtime.Autograd.LibTorch.Buffer.add sqrtVHat epsilonBuf
  let normalizedUpdate := Runtime.Autograd.LibTorch.Buffer.div mHat denominator
  let decayedParams :=
    Runtime.Autograd.LibTorch.Buffer.axpy paramsBuf paramsBuf (-(learningRate * weightDecay))
  let expectedParams :=
    Runtime.Autograd.LibTorch.Buffer.axpy decayedParams normalizedUpdate (-learningRate)
  let (gotParams, gotM, gotV) := Runtime.Autograd.LibTorch.Buffer.adamStep
    paramsBuf gradientBuf firstMomentBuf secondMomentBuf
    beta1 oneMinusBeta1 beta2 oneMinusBeta2
    firstMomentCorrection secondMomentCorrection epsilon
    (-(learningRate * weightDecay)) (-learningRate)

  Utils.assertTensorApprox (s := optimShape) "fused AdamW parameters"
    (← Utils.bufferToTensor gotParams) (← Utils.bufferToTensor expectedParams) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW first moment"
    (← Utils.bufferToTensor gotM) (← Utils.bufferToTensor expectedM) (tol := 1e-6)
  Utils.assertTensorApprox (s := optimShape) "fused AdamW second moment"
    (← Utils.bufferToTensor gotV) (← Utils.bufferToTensor expectedV) (tol := 1e-6)

end Elementwise
end Cuda
end Tests
