/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.CROWNQuery.Json

/-! # Exact CROWN output-query tests -/

@[expose] public section

namespace NN.Tests.MLTheory.CROWNQuery

open NN.Verification.CROWNQuery

/-- `relu(x) + relu(-x) < 3/2` on `[-1, 1]`; retaining affine dependence proves the query. -/
def exampleText : String :=
  r#"{"format":"crown_query_v1","input_dim":1,"input":{"lo":["-1"],"hi":["1"]},"layer"# ++
  r#"s":[{"kind":"linear","weights":[["1"],["-1"]],"bias":["0","0"]},{"kind":"relu",""# ++
  r#"alpha":["1/2","1/2"]},{"kind":"linear","weights":[["1","1"]],"bias":["0"]}],"que"# ++
  r#"ry":{"weights":[["1"]],"bias":["-3/2"],"strict":true}}"#

def expect (name text : String) (expected : Bool) : IO Unit := do
  unless acceptsText text == expected do
    throw <| IO.userError s!"CROWN query: {name}: unexpected acceptance result"

def run : IO Unit := do
  expect "mixed signs and affine dependence" exampleText true
  expect "strict boundary" (exampleText.replace "-3/2" "-1") false
  expect "non-strict boundary"
    ((exampleText.replace "-3/2" "-1").replace "true" "false") true
  expect "unsafe stronger query" (exampleText.replace "-3/2" "-1/2") false
  expect "negative alpha" (exampleText.replace "1/2" "-1/2") false
  expect "alpha above one" (exampleText.replace "1/2" "3/2") false
  expect "alpha zero" (exampleText.replace "1/2" "0") true
  expect "alpha one" (exampleText.replace "1/2" "1") true
  expect "wrong alpha dimension" (exampleText.replace "[\"1/2\",\"1/2\"]" "[\"1/2\"]") false
  expect "zero denominator" (exampleText.replace "1/2" "1/0") false
  expect "rounded numeric alpha" (exampleText.replace "\"1/2\"" "0.5") false
  expect "reversed box" (exampleText.replace "\"lo\":[\"-1\"]" "\"lo\":[\"2\"]") false
  expect "point box" (exampleText.replace "\"lo\":[\"-1\"]" "\"lo\":[\"1\"]") true
  expect "wrong input dimension" (exampleText.replace "\"input_dim\":1" "\"input_dim\":2") false
  expect "ragged matrix" (exampleText.replace "[[\"1\"],[\"-1\"]]" "[[\"1\"],[]]") false
  expect "wrong query dimension" (exampleText.replace "\"query\":{\"weights\":[[\"1\"]]"
    "\"query\":{\"weights\":[[\"1\",\"1\"]]") false
  expect "empty query" (exampleText.replace "\"query\":{\"weights\":[[\"1\"]],\"bias\":[\"-3/2\"]"
    "\"query\":{\"weights\":[],\"bias\":[]") false
  expect "unsupported activation" (exampleText.replace "\"relu\"" "\"sigmoid\"") false
  expect "legacy format rejected" (exampleText.replace "crown_query_v1" "crown_v1") false
  expect "syntax error" "{" false
  IO.println "  Exact CROWN query: 20 acceptance/rejection tests passed"

end NN.Tests.MLTheory.CROWNQuery
