/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.CLI.Training

/-!
# CLI API Tests

Regression checks for optional, defaulted, required, and typed command-line values.
-/

@[expose] public section

namespace NN.Tests.API.CLI

open TorchLean

def expectEqual {α : Type} [BEq α] [Repr α]
    (label : String) (expected actual : α) : IO Unit := do
  unless expected == actual do
    throw <| IO.userError
      s!"CLI API check failed: {label} (expected {repr expected}, got {repr actual})"

def expectOk {α : Type} (label : String) (result : Except String α) : IO α :=
  match result with
  | .ok value => pure value
  | .error message =>
      throw <| IO.userError s!"CLI API check failed: {label} returned `{message}`"

def expectError {α : Type}
    (label expected : String) (result : Except String α) : IO Unit :=
  match result with
  | .ok _ =>
      throw <| IO.userError s!"CLI API check failed: {label} unexpectedly succeeded"
  | .error actual => expectEqual label expected actual

def run : IO Unit := do
  let optionalAbsent ← expectOk "optional absent" <|
    TorchLean.CLI.takeFlagValue? ["keep"] "name"
  expectEqual "optional absent value" none optionalAbsent.1
  expectEqual "optional absent rest" ["keep"] optionalAbsent.2

  let splitValue ← expectOk "split value" <|
    TorchLean.CLI.takeFlagValue? ["before", "--name", "alice", "after"] "name"
  expectEqual "split value result" (some "alice") splitValue.1
  expectEqual "split value rest" ["before", "after"] splitValue.2

  let equalsValue ← expectOk "equals value" <|
    TorchLean.CLI.takeFlagValue? ["--name=bob", "--other"] "name"
  expectEqual "equals value result" (some "bob") equalsValue.1
  expectEqual "equals value rest" ["--other"] equalsValue.2

  expectError "duplicate value" "--name: duplicate flag" <|
    TorchLean.CLI.takeFlagValue? ["--name=a", "--name", "b"] "name"
  expectError "missing split value" "--name: expected a value" <|
    TorchLean.CLI.takeFlagValue? ["--name"] "name"

  let defaulted ← expectOk "string default" <|
    TorchLean.CLI.takeFlagValue ["--other"] "name" (default := "fallback")
  expectEqual "string default value" "fallback" defaulted.1
  expectEqual "string default rest" ["--other"] defaulted.2

  expectError "required value" "custom missing value" <|
    TorchLean.CLI.requireFlagValue [] "name" (missing? := some "custom missing value")

  let parsed ← expectOk "parsed default" <|
    TorchLean.CLI.takeParsedFlag [] "count" (default := "12") fun value =>
      match value.toNat? with
      | some count => pure count
      | none => throw "not a natural number"
  expectEqual "parsed default value" 12 parsed.1

  let boolFlag ← expectOk "bare boolean" <|
    TorchLean.CLI.takeBoolFlag ["before", "--verbose", "after"] "verbose"
  expectEqual "bare boolean value" true boolFlag.1
  expectEqual "bare boolean rest" ["before", "after"] boolFlag.2
  expectError "duplicate bare boolean" "--verbose: duplicate flag" <|
    TorchLean.CLI.takeBoolFlag ["--verbose", "--verbose"] "verbose"

  let positional ← expectOk "optional positional" <|
    TorchLean.CLI.takePositional? ["--flag", "artifact.bin"]
  expectEqual "optional positional value" (some "artifact.bin") positional.1
  expectEqual "optional positional rest" ["--flag"] positional.2
  let defaultPosition ← expectOk "default positional" <|
    TorchLean.CLI.takePositional ["--flag"] (default := "default.bin")
  expectEqual "default positional value" "default.bin" defaultPosition.1
  expectError "duplicate positional" "unexpected positional argument: second" <|
    TorchLean.CLI.takePositional? ["first", "second"]

  let natural ← expectOk "natural value" <|
    TorchLean.CLI.takeNatFlag? ["--count=7"] "count"
  expectEqual "natural value result" (some 7) natural.1
  let naturalDefault ← expectOk "natural default" <|
    TorchLean.CLI.takeNatFlag [] "count" (default := 9)
  expectEqual "natural default value" 9 naturalDefault.1
  expectError "invalid natural" "--count: expected a natural number, got `seven`" <|
    TorchLean.CLI.takeNatFlag? ["--count=seven"] "count"

  let floatValue ← expectOk "float value" <|
    TorchLean.CLI.takeFloatFlag? ["--rate=1e-3"] "rate"
  expectEqual "float value result" (some 0.001) floatValue.1
  let floatDefault ← expectOk "float default" <|
    TorchLean.CLI.takeFloatFlag [] "rate" (default := 0.25)
  expectEqual "float default value" 0.25 floatDefault.1
  expectError "required float" "missing --rate=<float>" <|
    TorchLean.CLI.requireFloatFlag [] "rate"

  let boolValue ← expectOk "boolean value" <|
    TorchLean.CLI.takeBoolValueFlag? ["--enabled=0"] "enabled"
  expectEqual "boolean value result" (some false) boolValue.1
  expectError "invalid boolean" "--enabled: expected true, false, 1, or 0; got `maybe`" <|
    TorchLean.CLI.takeBoolValueFlag? ["--enabled=maybe"] "enabled"

  let bareSwitch ← expectOk "bare switch" <|
    TorchLean.CLI.takeSwitch? ["--color", "output.txt"] "color"
  expectEqual "bare switch value" (some true) bareSwitch.1
  expectEqual "bare switch rest" ["output.txt"] bareSwitch.2
  let falseSwitch ← expectOk "false switch" <|
    TorchLean.CLI.takeSwitch ["--color=false"] "color" (default := true)
  expectEqual "false switch value" false falseSwitch.1

  let path ← expectOk "optional path" <|
    TorchLean.CLI.takePathFlag? ["--data=sample.csv"] "data"
  expectEqual "optional path value" (some ("sample.csv" : System.FilePath)) path.1
  let defaultPath ← expectOk "default path" <|
    TorchLean.CLI.takePathFlag [] "data" (default := "default.csv")
  expectEqual "default path value" ("default.csv" : System.FilePath) defaultPath.1
  expectError "required path" "tool: missing required --data <path>" <|
    TorchLean.CLI.requirePathFlag [] "data" (exeName := "tool")

  let pairedPaths ← expectOk "paired paths" <|
    TorchLean.CLI.takePairedPathFlags ["--vocab=v.json", "--merges=m.txt"] "vocab" "merges"
  expectEqual "paired first path"
    (some ("v.json" : System.FilePath)) pairedPaths.1.1
  expectEqual "paired second path"
    (some ("m.txt" : System.FilePath)) pairedPaths.1.2
  expectEqual "paired path rest" [] pairedPaths.2
  expectError "incomplete paired paths" "--vocab requires --merges" <|
    TorchLean.CLI.takePairedPathFlags ["--vocab=v.json"] "vocab" "merges"

  let trainingDefaults ← expectOk "training defaults" <|
    TorchLean.CLI.Training.RunOptions.parse
      "trainer" [] "training.json"
      (defaultSteps := 7) (defaultBatchSize := 64)
  expectEqual "training default steps" 7 trainingDefaults.1.steps
  expectEqual "training default batch size" 64 trainingDefaults.1.batchSize
  expectEqual "training default remaining arguments" [] trainingDefaults.2

  let trainingOverrides ← expectOk "training overrides" <|
    TorchLean.CLI.Training.RunOptions.parse
      "trainer" ["--steps=3", "--batch-size", "5"] "training.json"
      (defaultSteps := 7) (defaultBatchSize := 64)
  expectEqual "training override steps" 3 trainingOverrides.1.steps
  expectEqual "training override batch size" 5 trainingOverrides.1.batchSize
  expectEqual "training override remaining arguments" [] trainingOverrides.2

  expectError "invalid default batch size" "trainer: --batch-size must be > 0" <|
    TorchLean.CLI.Training.RunOptions.parse
      "trainer" [] "training.json" (defaultBatchSize := 0)

end NN.Tests.API.CLI
