/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Common
public import NN.Runtime.Training.Log

/-!
# RL CLI Helpers (API)

TorchLean's runnable RL examples (`NN/Examples/Models/RL/*`) intentionally share one CLI shape:

- `--updates <n>`: how many update iterations to run,
- `--eval-every <n>`: evaluate every `n` updates,
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint,
- `--eval-max-steps <n>`: max steps per evaluation episode,
- `--log <path|off|none|false>`: where to write the widget-friendly TrainLog JSON.

This module centralizes that parsing so we don't duplicate the same flag boilerplate across
CartPole/Pong/GridWorld examples.
-/

@[expose] public section

namespace NN
namespace API

namespace rl
namespace cli

/-- Parsed PPO-style training flags shared by multiple runnable examples. -/
structure PpoFlags where
  updates : Nat
  evalEvery : Nat
  evalEpisodes : Nat
  evalMaxSteps : Nat
  log : _root_.Runtime.Training.LogDestination
  logPath : System.FilePath
deriving Repr

/--
Parse PPO-style shared flags.

Notes:
- `--log off|none|false` disables writing the JSON artifact but still returns the resolved default
  `logPath` (useful for printing consistent banners).
- We treat `0` as invalid for the update/eval counts because a “no-op” run usually indicates a CLI
  mistake.
-/
def parsePpoFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultUpdates defaultEvalEvery defaultEvalEpisodes defaultEvalMaxSteps : Nat) :
    Except String (PpoFlags × List String) := do
  let (logRaw?, args) ← CLI.takeFlagValueOnce args "log"
  let (updates?, args) ← CLI.takeNatFlagOnce args "updates"
  let (evalEvery?, args) ← CLI.takeNatFlagOnce args "eval-every"
  let (evalEpisodes?, args) ← CLI.takeNatFlagOnce args "eval-episodes"
  let (evalMaxSteps?, args) ← CLI.takeNatFlagOnce args "eval-max-steps"

  let updates := updates?.getD defaultUpdates
  let evalEvery := evalEvery?.getD defaultEvalEvery
  let evalEpisodes := evalEpisodes?.getD defaultEvalEpisodes
  let evalMaxSteps := evalMaxSteps?.getD defaultEvalMaxSteps

  if updates = 0 then throw s!"{exeName}: --updates must be > 0"
  if evalEvery = 0 then throw s!"{exeName}: --eval-every must be > 0"
  if evalEpisodes = 0 then throw s!"{exeName}: --eval-episodes must be > 0"
  if evalMaxSteps = 0 then throw s!"{exeName}: --eval-max-steps must be > 0"

  let log := _root_.Runtime.Training.LogDestination.parse? defaultLogPath logRaw?
  pure ({ updates := updates
          evalEvery := evalEvery
          evalEpisodes := evalEpisodes
          evalMaxSteps := evalMaxSteps
          log := log
          logPath := log.pathD defaultLogPath }, args)

end cli
end rl

end API
end NN

