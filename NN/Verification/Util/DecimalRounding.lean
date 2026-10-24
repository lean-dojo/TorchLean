/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.FloatInstances

public import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Runtime
public import FloatLib.Floats.Formats.IEEE754.Native
public import NN.Tensor.Conversion

/-!
# Rounding JSON decimals into binary formats

A JSON number is an exact decimal `mantissa / 10^exponent`. Reading it through
`JsonNumber.toFloat` and then casting to binary32 rounds twice, and the first rounding is not
directed. The helpers here round the exact decimal once, in the requested direction, directly into
binary32 or binary64.

Certificate parsers use `.down` for lower endpoints and
`.up` for upper endpoints, so the parsed box contains the decimal box that the
artifact states. Transcript entries that must equal a replayed value use `.nearest`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Floats.Formats.BinaryInterchange.Model (IEEERoundingMode)

namespace NN.Verification.Util.DecimalRounding

open Lean

/-- Round an exact JSON decimal once into `fmt`. -/
def roundDecimal (fmt : FloatFormat) (mode : IEEERoundingMode) (n : JsonNumber) : Model fmt :=
  Model.ofRat fmt ((n.mantissa : Rat) / (10 ^ n.exponent : Rat)) (mode := mode)

/-- Round an exact JSON decimal once into the executable format `fmt`. -/
def binary (fmt : FloatFormat) (mode : IEEERoundingMode) (n : JsonNumber) :
    ExecFloat.Binary fmt.expWidth fmt.fracWidth fmt.encoding fmt.exponentBias
      fmt.expWidth_ge_two fmt.fracWidth_pos fmt.exponentBias_pos fmt.exponentBias_le_maxFinite :=
  ExecFloat.Binary.ofModel (roundDecimal fmt mode n)

/-- Round an exact JSON decimal once into host binary64. -/
def float (mode : IEEERoundingMode) (n : JsonNumber) : Float :=
  ExecFloat.Binary.toFloat (ExecFloat.Binary.ofModel
    (roundDecimal FloatFormat.binary64 mode n) : FloatLib.Floats.Binary64)

/-- Read a JSON array of exactly `n` numbers. -/
def numbers? (n : Nat) (j : Json) : Option (Fin n → JsonNumber) := do
  let .arr xs := j | none
  let nums ← xs.mapM fun
    | .num x => some x
    | _ => none
  if h : nums.size = n then
    pure fun i => nums[i.val]'(by rw [h]; exact i.isLt)
  else
    none

/-- Read a JSON `rows × cols` matrix of numbers. -/
def numberMatrix? (rows cols : Nat) (j : Json) : Option (Fin rows → Fin cols → JsonNumber) := do
  let .arr xs := j | none
  let parsed ← xs.mapM (numbers? cols)
  if h : parsed.size = rows then
    pure fun i => parsed[i.val]'(by rw [h]; exact i.isLt)
  else
    none

/-- Read a length-`n` JSON vector, rounding every entry once into `fmt` with `mode`. -/
def vector? (fmt : FloatFormat) (mode : IEEERoundingMode) (n : Nat) (j : Json) :
    Option (TorchLean.Tensor
      (ExecFloat.Binary fmt.expWidth fmt.fracWidth fmt.encoding fmt.exponentBias
        fmt.expWidth_ge_two fmt.fracWidth_pos fmt.exponentBias_pos fmt.exponentBias_le_maxFinite)
      [n]) := do
  let xs ← numbers? n j
  pure (TorchLean.Tensor.ofFn fun i => binary fmt mode (xs i))

/-- Read a `rows × cols` JSON matrix, rounding every entry once into `fmt` with `mode`. -/
def matrix? (fmt : FloatFormat) (mode : IEEERoundingMode) (rows cols : Nat) (j : Json) :
    Option (TorchLean.Tensor
      (ExecFloat.Binary fmt.expWidth fmt.fracWidth fmt.encoding fmt.exponentBias
        fmt.expWidth_ge_two fmt.fracWidth_pos fmt.exponentBias_pos fmt.exponentBias_le_maxFinite)
      [rows, cols]) := do
  let xs ← numberMatrix? rows cols j
  pure (TorchLean.Tensor.matrix fun i k => binary fmt mode (xs i k))

end NN.Verification.Util.DecimalRounding
