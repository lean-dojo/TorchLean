/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Json
public import TorchLeanTest.Harness

/-!
# Strict JSON integers at the RL boundary

`run` checks that `TorchLean.Json.parseNatStrict`, the reader of integer fields in rollout JSON
written by an external process, accepts a number only when its decimal value is the integer it
returns. `1.00`, `1e3`, `2^53` and the largest `UInt64` pass. Fractions, a fractional tail that
`Float` loses, `2^53 + 1`, `2^64`, negative numbers and tiny exponents are refused. It also checks
that Lean's `Json.parse`, which `TorchLean.Json.readFile` and the RL clients use, refuses an object
that repeats a key, at the top level and nested, where Lean 4.35 kept the last value.
-/

@[expose] public section

namespace TorchLean.Test
namespace RL

/-- Assert a boolean runtime condition with a labeled failure message. -/
def assertBool (msg : String) (b : Bool) : IO Unit := do
  if !b then
    throw <| IO.userError msg

/-- Strict parsing compares the original decimal, including fractional tails lost by Float. -/
def checkStrictNaturalJson : IO Unit := do
  for (source, expected) in [("0", 0), ("1.00", 1), ("1e3", 1000),
      ("9007199254740992", 9007199254740992),
      ("18446744073709549568", 18446744073709549568),
      ("18446744073709551615", 18446744073709551615)] do
    let json ← IO.ofExcept (Lean.Json.parse source)
    let parsed ← IO.ofExcept (Json.parseNatStrict "strict natural JSON" json)
    assertBool s!"strict natural JSON {source}" (parsed == expected)
  for source in ["1.5", "1.00000000000000001", "9007199254740993",
      "18446744073709551616", "-1", "-1e-400", "1e-400"] do
    let json ← IO.ofExcept (Lean.Json.parse source)
    match Json.parseNatStrict "strict natural JSON" json with
    | .ok value => throw <| IO.userError s!"strict natural JSON accepted {source} as {value}"
    | .error _ => pure ()

/-- An object that repeats a key does not parse, so no reader picks one of the two values. -/
def checkRepeatedKey : IO Unit := do
  for source in [r#"{"action": 1, "action": 2}"#, r#"{"step": {"reward": 0, "reward": 1}}"#] do
    match Lean.Json.parse source with
    | .ok json => throw <| IO.userError s!"JSON with a repeated key parsed as {json.compress}"
    | .error _ => pure ()

/-- Run the strict JSON integer and repeated key checks. -/
def run : IO Unit := do
  checkStrictNaturalJson
  checkRepeatedKey
  IO.println "rl_check: ok"

end RL
end TorchLean.Test
