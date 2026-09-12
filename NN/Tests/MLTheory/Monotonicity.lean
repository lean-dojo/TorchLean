/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Monotonicity.Json

/-!
# Exact monotonicity checker tests

Check acceptance and rejection of exact rational certificates at the JSON boundary.
-/

@[expose] public section

namespace NN.Tests.MLTheory.Monotonicity

open NN.Verification.Monotonicity

def expect (name text : String) (expected : Bool) : IO Unit := do
  unless acceptsText text == expected do
    throw <| IO.userError s!"monotonicity: {name}: unexpected acceptance result"

def run : IO Unit := do
  expect "exact fractions and negative bias"
    r#"{"format":"monotonicity_v1",
      "input_dim":2,
      "layers":[{"kind":"linear",
        "weights":[["1/3","0"]],
        "bias":["-7"]},{"kind":"relu"}]}"# true
  expect "negative weight"
    r#"{"format":"monotonicity_v1",
      "input_dim":1,
      "layers":[{"kind":"linear",
        "weights":[["-1/1000000000000000000"]],
        "bias":["0"]}]}"# false
  expect "ragged matrix"
    r#"{"format":"monotonicity_v1",
      "input_dim":2,
      "layers":[{"kind":"linear",
        "weights":[["1","2"],["3"]],
        "bias":["0","0"]}]}"# false
  expect "wrong bias length"
    r#"{"format":"monotonicity_v1",
      "input_dim":1,
      "layers":[{"kind":"linear",
        "weights":[["1"]],
        "bias":[]}]}"# false
  expect "wrong next-layer dimension"
    r#"{"format":"monotonicity_v1",
      "input_dim":1,
      "layers":[{"kind":"linear",
        "weights":[["1"],["2"]],
        "bias":["0","0"]},{"kind":"linear",
        "weights":[["1"]],
        "bias":["0"]}]}"# false
  expect "zero denominator"
    r#"{"format":"monotonicity_v1",
      "input_dim":1,
      "layers":[{"kind":"linear",
        "weights":[["1/0"]],
        "bias":["0"]}]}"# false
  expect "rounded JSON numbers are not rational strings"
    r#"{"format":"monotonicity_v1",
      "input_dim":1,
      "layers":[{"kind":"linear",
        "weights":[[0.1]],
        "bias":["0"]}]}"# false
  expect "unsupported operation"
    r#"{"format":"monotonicity_v1","input_dim":1,"layers":[{"kind":"sigmoid"}]}"# false
  expect "unsupported version"
    r#"{"format":"crown_v1","input_dim":1,"layers":[{"kind":"relu"}]}"# false
  expect "empty chain"
    r#"{"format":"monotonicity_v1","input_dim":1,"layers":[]}"# false
  expect "syntax error" "{" false
  expect "ReLU-only chain"
    r#"{"format":"monotonicity_v1","input_dim":3,"layers":[{"kind":"relu"}]}"# true
  expect "zero-dimensional ReLU"
    r#"{"format":"monotonicity_v1","input_dim":0,"layers":[{"kind":"relu"}]}"# true
  IO.println "  Exact monotonicity: 13 acceptance/rejection tests passed"

end NN.Tests.MLTheory.Monotonicity
