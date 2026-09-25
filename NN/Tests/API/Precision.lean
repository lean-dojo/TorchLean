/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- This is a downstream consumer: the application umbrella must supply the whole typed workflow.
public import NN.API

/-!
# Configured precision through the public tensor/model API

The expectations are exact rationals, not comparisons with another invocation of the same model.
The binary128 and binary256 witnesses retain fractions that native binary64 loses. A nested-dual
quotient also rescues a representable mixed coefficient after an intermediate product underflows.
-/

@[expose] public section

namespace NN.Tests.API.Precision

open TorchLean
open TorchLean.Tensor
open FloatLib.Floats
open Runtime.Autograd.Model

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do throw <| IO.userError s!"configured precision: {label}"

def affine : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)

/--
Give both affine parameters the same scalar `a`: the model is `x ↦ a*x + a`.
At `x=2`, its output is `3*a`, input derivative is `a`, and an all-one parameter tangent gives 3.
-/
def checkModel {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) (gapExponent : Nat)
    (retainsGap : Bool) : IO Unit := do
  let source : Rat := 1 + 1 / (2 ^ gapExponent : Nat)
  let a : α := Rat.cast source
  let expected : Rat := if retainsGap then source else 1
  expect (label ++ "/rational cast") (decode a == some expected)
  let input : Tensor α [1] := Tensor.full [1] 2
  let state : nn.State α (nn.stateShapes affine) := nn.State.full a
  let graph ← nn.lowerToTypedGraph affine (α := α)
  let output := nn.TypedGraphModel.forward graph state input
  expect (label ++ "/forward")
    (decode (Tensor.getScalar output ⟨0, by decide⟩) == some (3 * expected))
  let inputJvp := nn.TypedGraphModel.jvp graph state nn.State.zeros
    input (Tensor.full [1] 1)
  expect (label ++ "/input JVP")
    (decode (Tensor.getScalar inputJvp ⟨0, by decide⟩) == some expected)
  let stateJvp := nn.TypedGraphModel.jvp graph state (nn.State.full 1)
    input (Tensor.zeros [1])
  expect (label ++ "/state JVP")
    (decode (Tensor.getScalar stateJvp ⟨0, by decide⟩) == some 3)
  let (stateVjp, inputVjp) := nn.TypedGraphModel.vjp graph state input (Tensor.full [1] 1)
  expect (label ++ "/input VJP")
    (decode (Tensor.getScalar inputVjp ⟨0, by decide⟩) == some expected)
  let weightVjp := stateVjp.get ⟨0, by decide⟩
  let biasVjp := stateVjp.get ⟨1, by decide⟩
  expect (label ++ "/weight VJP")
    ((weightVjp.to (Array α)).map decode == #[some 2])
  expect (label ++ "/bias VJP")
    ((biasVjp.to (Array α)).map decode == #[some 1])
  let matrix : Tensor α [1, 2] := Tensor.full [1, 2] a
  let column : Tensor α [2, 1] := Tensor.full [2, 1] 1
  let product : Tensor α [1, 1] :=
    einsum matrix, column "row k, k column -> row column"
  expect (label ++ "/tensor contraction")
    ((product.to (Array α)).map decode == #[some (2 * expected)])

abbrev Binary16 := ExecFloat.Binary (exponentBits := 5) (fractionBits := 10)
abbrev Tiny := ExecFloat.Binary (exponentBits := 3) (fractionBits := 2)
abbrev Binary128 := ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)
abbrev Binary256 := ExecFloat.Binary (exponentBits := 19) (fractionBits := 236)

/-- A constant row must normalize to finite zeros with the default stabilizer. -/
def checkConstantNormalization {α : Type} [Storage α] [Context α]
    (label : String) (decode : α → Option Rat) : IO Unit := do
  let input : Tensor α [1, 2] := Tensor.full [1, 2] 1
  let scale : Tensor α [2] := Tensor.full [2] 1
  let bias : Tensor α [2] := Tensor.zeros [2]
  let output := Spec.layerNorm input scale bias
  expect (label ++ "/constant layer norm")
    ((output.to (Array α)).map decode == #[some 0, some 0])

/-- Tiny formats require an explicit representable tolerance, including in typed graphs. -/
def checkTinyNormalization : IO Unit := do
  let input : Tensor Tiny [1, 2] := Tensor.full [1, 2] 1
  let scale : Tensor Tiny [2] := Tensor.full [2] 1
  let bias : Tensor Tiny [2] := Tensor.zeros [2]
  let defaultOutput := Spec.layerNorm input scale bias
  expect "tiny/spec normalization default produces NaNs"
    ((defaultOutput.to (Array Tiny)).map ExecFloat.Binary.isNaN == #[true, true])
  let positive : Tiny := Rat.cast (1 / 16 : Rat)
  let explicitOutput := Spec.layerNorm input scale bias (epsilon := positive)
  expect "tiny/spec normalization explicit epsilon"
    ((explicitOutput.to (Array Tiny)).map ExecFloat.Binary.toRat? == #[some 0, some 0])
  let defaultModel : nn.Sequential [2] [2] :=
    nn.build 0 (nn.layerNorm (width := 2) (affine := false))
  let defaultGraph ← nn.lowerToTypedGraph defaultModel (α := Tiny)
  let vector : Tensor Tiny [2] := Tensor.full [2] 1
  let defaultGraphOutput := nn.TypedGraphModel.forward defaultGraph nn.State.zeros vector
  expect "tiny/typed normalization default produces NaNs"
    ((defaultGraphOutput.to (Array Tiny)).map ExecFloat.Binary.isNaN == #[true, true])
  let explicitModel : nn.Sequential [2] [2] :=
    nn.build 0 (nn.layerNorm (width := 2) (eps := (1 / 16 : Rat)) (affine := false))
  let explicitGraph ← nn.lowerToTypedGraph explicitModel (α := Tiny)
  let explicitGraphOutput := nn.TypedGraphModel.forward explicitGraph nn.State.zeros vector
  expect "tiny/typed normalization explicit epsilon"
    ((explicitGraphOutput.to (Array Tiny)).map ExecFloat.Binary.toRat? == #[some 0, some 0])

/-- Exact expected encodings catch overflow while forming the old denominator. -/
def checkNormalizationEpsilon : IO Unit := do
  let half : Binary16 := normalizationEpsilon
  expect "binary16/normalization epsilon encoding" (ExecFloat.Binary.toNatBits half == 0x00a8)
  expect "binary16/normalization epsilon value"
    (ExecFloat.Binary.toRat? half == some (168 / (2 ^ 24 : Nat) : Rat))
  let dualHalf : Dual Binary16 := normalizationEpsilon
  expect "binary16/dual normalization epsilon"
    (ExecFloat.Binary.toNatBits dualHalf.re == 0x00a8 &&
      ExecFloat.Binary.toNatBits dualHalf.du == 0)
  checkConstantNormalization (α := Binary16) "binary16" ExecFloat.Binary.toRat?
  -- With only two fraction bits but binary16's exponent range, 1e-5 rounds to the first subnormal.
  let coarse : ExecFloat.Binary (exponentBits := 5) (fractionBits := 2) := normalizationEpsilon
  expect "coarse/normalization epsilon encoding" (ExecFloat.Binary.toNatBits coarse == 1)
  expect "coarse/normalization epsilon value"
    (ExecFloat.Binary.toRat? coarse == some (1 / (2 ^ 16 : Nat) : Rat))
  checkConstantNormalization (α := ExecFloat.Binary (exponentBits := 5) (fractionBits := 2))
    "coarse" ExecFloat.Binary.toRat?
  -- This format cannot represent the tolerance: normalization does not silently change it.
  let tiny : Tiny := normalizationEpsilon
  expect "tiny/normalization epsilon underflows" (ExecFloat.Binary.toNatBits tiny == 0)
  checkTinyNormalization
  let native32 : Float32 := normalizationEpsilon
  let native64 : Float := normalizationEpsilon
  expect "native32/normalization epsilon encoding" (native32.toBits == 0x3727c5ac)
  expect "native64/normalization epsilon encoding" (native64.toBits == 0x3ee4f8b588e368f1)
  expect "native32/normalization epsilon preserves previous result"
    (native32.toBits == (1 / ((100000 : Nat) : Float32)).toBits)
  expect "native64/normalization epsilon preserves previous result"
    (native64.toBits == (1 / ((100000 : Nat) : Float)).toBits)

/-- A mixed quotient coefficient survives even when a constituent product rounds to zero. -/
def checkNestedQuotientUnderflow : IO Unit := do
  expect "binary128/checked quotient support"
    (Numeric.QuotientArithmetic.supported (α := Binary128))
  let a : Binary128 := Rat.cast (1 / (2 ^ 9000 : Nat) : Rat)
  let b : Binary128 := Rat.cast ((2 ^ 15000 : Nat) : Rat)
  expect "binary128/underflow witness" (ExecFloat.Binary.isZero (a * a))
  let x : Dual (Dual Binary128) := ⟨⟨a, 0⟩, ⟨0, 0⟩⟩
  let y : Dual (Dual Binary128) := ⟨⟨1, a⟩, ⟨b, 0⟩⟩
  let quotient := x / y
  -- Coefficients of a/(1 + a*ε + b*η): a, -a², -a*b, 2*a²*b.
  expect "binary128/rescued mixed quotient coefficient"
    (ExecFloat.Binary.toRat? quotient.du.du ==
      some (1 / (2 ^ 2999 : Nat) : Rat))
  expect "binary128/large finite derivative"
    (ExecFloat.Binary.toRat? quotient.du.re ==
      some (-((2 ^ 6000 : Nat) : Rat)))

/-- Every nested-dual coefficient retains the configured wide significand. -/
def checkNestedQuotientPrecision : IO Unit := do
  let high : Binary256 := Rat.cast (1 + 1 / (2 ^ 140 : Nat) : Rat)
  let numerator : Dual (Dual Binary256) := ⟨⟨high, high⟩, ⟨high, high⟩⟩
  let denominator : Dual (Dual Binary256) := ⟨⟨2, 0⟩, ⟨0, 0⟩⟩
  let result := numerator / denominator
  let expected : Rat := (1 + 1 / (2 ^ 140 : Nat)) / 2
  expect "binary256/nested precision"
    ((#[result.re.re, result.re.du, result.du.re, result.du.du]).map
      ExecFloat.Binary.toRat? == #[some expected, some expected, some expected, some expected])

def checkNestedQuotient : IO Unit := do
  checkNestedQuotientUnderflow
  checkNestedQuotientPrecision

def checkInputBoundaries : IO Unit := do
  let literal : Binary128 :=
    1.0000000000000000000008470329472543003390683225006796419620513916015625
  let expected : Rat := 1 + 1 / (2 ^ 70 : Nat)
  expect "binary128/direct decimal literal" (ExecFloat.Binary.toRat? literal == some expected)
  let parsed : Except _ Binary128 :=
    ExecFloat.Binary.parse
      "1.0000000000000000000008470329472543003390683225006796419620513916015625"
  match parsed with
  | .error _ => throw <| IO.userError "configured precision: exact decimal parsing failed"
  | .ok value =>
    expect "binary128/exact decimal parser"
      (ExecFloat.Binary.toRat? value == some expected)
  let generated : Tensor Binary128 [2] :=
    Tensor.ofFn fun i => Rat.cast (expected + (i.val : Rat))
  expect "binary128/tensor ofFn"
    ((generated.to (Array Binary128)).map ExecFloat.Binary.toRat? ==
      #[some expected, some (expected + 1)])
  let bracket : Tensor Binary128 [2] :=
    [1.0000000000000000000008470329472543003390683225006796419620513916015625,
     Rat.cast (expected + 1)]
  expect "binary128/tensor literal"
    ((bracket.to (Array Binary128)).map ExecFloat.Binary.toRat? ==
      #[some expected, some (expected + 1)])
  expect "binary128/power preserves wide base"
    (ExecFloat.Binary.toRat? (literal ^ (1 : Binary128)) == some expected)
  expect "binary128/square root preserves wide fraction"
    (ExecFloat.Binary.toRat? (MathFunctions.sqrt (literal * literal)) == some expected)
  let delta : Binary128 := Rat.cast (1 / (2 ^ 70 : Nat) : Rat)
  -- At this input the correctly rounded exponential is 1+2^-70. This checks one approximation
  -- result; it does not assert an error theorem for FloatLib's exponential algorithm.
  expect "binary128/exponential avoids binary64 result conversion"
    (ExecFloat.Binary.toRat? (MathFunctions.exp delta) == some expected)
  match ExecFloat.Binary.toRat? (MathFunctions.log literal) with
  | none => throw <| IO.userError "configured precision: logarithm returned a non-finite value"
  | some value =>
    expect "binary128/logarithm retains a small nonzero result"
      (decide (1 / (2 ^ 71 : Nat) ≤ value ∧ value ≤ 1 / (2 ^ 69 : Nat)))
  let coarseEpsilon :
      ExecFloat.Binary (exponentBits := 3) (fractionBits := 2) := Context.defaultEpsilon
  expect "coarse format/nonzero safeguard"
    (Context.gtBool coarseEpsilon 0 && ExecFloat.Binary.toRat? coarseEpsilon == some (1 / 16))
  let tinyEpsilon :
      ExecFloat.Binary (exponentBits := 2) (fractionBits := 1) := Context.defaultEpsilon
  expect "minimal format/smallest positive safeguard"
    (ExecFloat.Binary.toNatBits tinyEpsilon == 1 &&
      ExecFloat.Binary.toRat? tinyEpsilon == some (1 / 2))
  let wideEpsilon : Binary128 := Context.defaultEpsilon
  expect "binary128/safeguard retains positive rounded value"
    (Context.gtBool wideEpsilon 0 && ExecFloat.Binary.toNatBits wideEpsilon != 1)
  -- An exact fraction must not overflow while independently casting its numerator/denominator.
  let largeFraction : Rat := ((2 ^ 20000 : Nat) + 1) / ((2 ^ 20000 : Nat) : Rat)
  let rounded : Binary128 := Rat.cast largeFraction
  expect "binary128/one-round rational conversion" (ExecFloat.Binary.toRat? rounded == some 1)
  let tooPrecise : Tensor Binary128 [1] := Tensor.full [1] literal
  let transferRejected ← try
      let _ ← Runtime.toFloatTensor tooPrecise
      pure false
    catch _ => pure true
  expect "binary128/native tensor transfer rejected" transferRejected

/-- Independent IEEE encodings exercise rounding, sign, range and exact-fraction conversion. -/
def checkNativeRationalCasts : IO Unit := do
  let cases32 : Array (Rat × UInt32) := #[
    (0, 0), (1 / 3, 0x3eaaaaab), (-1 / 3, 0xbeaaaaab),
    (1 + 1 / (2 ^ 24 : Nat), 0x3f800000),
    (1 + 3 / (2 ^ 24 : Nat), 0x3f800002),
    (1 / (2 ^ 149 : Nat), 0x00000001),
    (-1 / (2 ^ 149 : Nat), 0x80000001),
    (1 / (2 ^ 150 : Nat), 0x00000000),
    (-1 / (2 ^ 150 : Nat), 0x80000000),
    (((2 ^ 128 - 2 ^ 104 : Nat) : Rat), 0x7f7fffff),
    (((2 ^ 128 : Nat) : Rat), 0x7f800000)]
  for (value, expected) in cases32 do
    let actual : Float32 := Rat.cast value
    expect s!"native32/exact rational {value}" (actual.toBits == expected)
  let cases64 : Array (Rat × UInt64) := #[
    (0, 0), (1 / 3, 0x3fd5555555555555), (-1 / 3, 0xbfd5555555555555),
    (1 + 1 / (2 ^ 53 : Nat), 0x3ff0000000000000),
    (1 + 3 / (2 ^ 53 : Nat), 0x3ff0000000000002),
    (1 / (2 ^ 1074 : Nat), 0x0000000000000001),
    (-1 / (2 ^ 1074 : Nat), 0x8000000000000001),
    (1 / (2 ^ 1075 : Nat), 0x0000000000000000),
    (-1 / (2 ^ 1075 : Nat), 0x8000000000000000),
    (((2 ^ 1024 - 2 ^ 971 : Nat) : Rat), 0x7fefffffffffffff),
    (((2 ^ 1024 : Nat) : Rat), 0x7ff0000000000000)]
  for (value, expected) in cases64 do
    let actual : Float := Rat.cast value
    expect s!"native64/exact rational {value}" (actual.toBits == expected)
  let largeFraction : Rat := ((2 ^ 20000 : Nat) + 1) / ((2 ^ 20000 : Nat) : Rat)
  let wide32 : Float32 := Rat.cast largeFraction
  let wide64 : Float := Rat.cast largeFraction
  expect "native32/large exact numerator and denominator" (wide32.toBits == 0x3f800000)
  expect "native64/large exact numerator and denominator" (wide64.toBits == 0x3ff0000000000000)

/-- Maintained execution checks for several configured widths and both native scalar types. -/
def run : IO Unit := do
  checkInputBoundaries
  checkNativeRationalCasts
  checkNormalizationEpsilon
  checkModel (α := Binary16)
    "binary16" ExecFloat.Binary.toRat? 80 false
  checkModel (α := ExecFloat.Binary (exponentBits := 8) (fractionBits := 23))
    "binary32" ExecFloat.Binary.toRat? 80 false
  checkModel (α := ExecFloat.Binary (exponentBits := 11) (fractionBits := 52))
    "binary64" ExecFloat.Binary.toRat? 80 false
  checkModel (α := Binary128) "binary128" ExecFloat.Binary.toRat? 100 true
  checkModel (α := Binary256) "binary256" ExecFloat.Binary.toRat? 140 true
  checkModel (α := Float) "native64"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat value)) 80 false
  checkModel (α := Float32) "native32"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat32 value)) 80 false
  checkNestedQuotient

end NN.Tests.API.Precision
