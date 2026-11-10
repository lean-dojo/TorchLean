/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Boundary
public import TorchLean.NN.PyTorch.Import
public import TorchLean.NN.Json

/-!
# RL Trust Boundary JSON Loader

The JSON side of the RL trust boundary reads a small rollout schema that is easy to write from
Python and validates each transition against the contract in
`TorchLean.NN.RL.Boundary`.

Proof files need the contract and checker semantics but not the parser or the PyTorch import layer
it pulls in, so the loader has its own module. Runtime bridges import it when they load external
rollouts.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace Boundary

open TorchLean.Tensor
open Lean
open Lean.Json

/-!
## JSON loading (one simple interchange format)

The schema is small and easy to emit from Python:

```json
{
  "transitions": [
    {
      "obs": <nested arrays matching obsShape>,
      "action": <nat>,
      "reward": <float>,
      "terminated": <bool>,
      "truncated": <bool>,
      "next_obs": <nested arrays matching obsShape>
    },
    ...
  ]
}
```

The schema is one supported integration path, a practical “lowest common denominator” that is easy
to diff and validate.

We wrote the schema by hand around the values of Gymnasium's `Env.step`
(https://gymnasium.farama.org/api/env/), minus `info`, and `scripts/rl/gymnasium_server.py`
emits it with `allow_nan=False` and zero-based actions. `reward` must be a JSON number. The action
goes through `Json.parseNatStrict`, so `1.00` and `1e3` read as 1 and 1000 and `1.5` is refused,
cases that `TorchLean.Test.RL.checkStrictNaturalJson` runs.
-/

/-!
### Tensor parser

The trust-boundary loader and runtime bridges such as the Gymnasium subprocess client read
naturals, Booleans and floats through `Json` and tensors through `parseTensorE`. It returns
`Except String ...` for pure loaders, and `IO` code lifts the result with `IO.ofExcept`.
-/

/--
Parse a JSON value as a typed tensor of shape `s`.

The value uses TorchLean's nested-array tensor encoding (`PyTorch.parseTensor`), the one the
Python bridges write, and a value that does not parse at shape `s` gives an error naming `field`.
-/
def parseTensorE (field : String) (s : Shape) (j : Json) : Except String (Tensor Float s) :=
  match PyTorch.parseTensor s j with
  | some t => .ok t
  | none =>
    .error s!"RL boundary: `{field}` did not match the expected shape {s}."

/-- Parse one transition object of the rollout schema and validate it with `checkTransition`. -/
def parseTransitionJson {obsShape : Shape} {nActions : Nat}
    (c : Contract obsShape nActions) (j : Json) :
    Except String (Transition obsShape nActions) := do
  let ctx := "RL boundary: transition"
  let o ← Json.object ctx j

  let obsJ ← Json.field ctx "obs" o
  let nextObsJ ← Json.field ctx "next_obs" o
  let action ← Json.parseNatStrict s!"{ctx}.action" (← Json.field ctx "action" o)
  let reward ← Json.floatField ctx "reward" o
  let terminated ← Json.boolField ctx "terminated" o
  let truncated ← Json.boolField ctx "truncated" o

  let obs ← parseTensorE (field := "obs") obsShape obsJ
  let nextObs ← parseTensorE (field := "next_obs") obsShape nextObsJ

  checkTransition (obsShape := obsShape) (nActions := nActions) c obs nextObs action reward
    terminated truncated

/-- Read the JSON object in the rollout file at `path` and return its `transitions` array. -/
def readRolloutTransitions (path : String) : IO (Array Json) := do
  let o ← IO.ofExcept <| Json.object "RL boundary: rollout" (← Json.readFile path)
  IO.ofExcept <| Json.field "RL boundary: rollout" "transitions" o >>=
    Json.array "RL boundary: rollout.transitions"

/--
Load a rollout file and validate every transition, returning one result per entry of the input
`transitions` array: `ok t` when the transition decoded and passed the contract check, and
`error msg` when decoding or checking failed.

A failed transition does not stop the others, so widget-style debugging can show a *summary* of
all violations.
-/
def loadRolloutAll {obsShape : Shape} {nActions : Nat}
    (path : String)
    (c : Contract obsShape nActions) :
    IO (Array (Except String (Transition obsShape nActions))) := do
  let transitionsArr ← readRolloutTransitions path
  pure (transitionsArr.map (parseTransitionJson c))

end Boundary
end RL
end TorchLean
