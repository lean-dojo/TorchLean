/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log
public import NN.API.CLI.Parser
public import NN.API.CLI -- shake: keep

/-!
# RL Command-Line Options

TorchLean's runnable RL examples (`NN/Examples/Models/RL/*`) share one CLI shape:

- `--updates <n>`: how many update iterations to run,
- `--eval-every <n>`: evaluate every `n` updates,
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint,
- `--eval-max-steps <n>`: max steps per evaluation episode,
- `--log <path|off|none|false>`: where to write the widget-friendly TrainLog JSON.

This module centralizes that parsing so we don't duplicate the same flag boilerplate across
CartPole/Pong/GridWorld examples.
-/

@[expose] public section

namespace TorchLean
namespace rl
namespace cli

/-- Parsed PPO command options shared by multiple runnable examples. -/
structure PPOOptions where
  updateCount : Nat
  evaluationInterval : Nat
  evaluationEpisodes : Nat
  maximumEvaluationSteps : Nat
  logDestination : Runtime.Training.LogDestination
deriving Repr

namespace PPOOptions

/-- Help text for PPO commands, with optional environment-specific artifact flags. -/
def usage (exeName : String) (artifactOptions : Array String := #[]) : String :=
  String.intercalate "\n" <| (#[
    s!"Usage: lake exe torchlean {exeName} [options]",
    "",
    "Training and evaluation:",
    "  --updates N         PPO update iterations",
    "  --eval-every N      updates between policy evaluations",
    "  --eval-episodes N   episodes in each evaluation",
    "  --eval-max-steps N  step limit for each evaluation episode",
    "  --log PATH|false    write a TrainLog JSON, or disable logging"
  ] ++ artifactOptions ++ #[
    "",
    "Runtime:",
    "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external",
    "  --execution eager|typed-graph",
    "  --arithmetic native",
    "  --seed N --show-backend"
  ]).toList

/--
Parse shared PPO command options.

Notes:
- `--log off|none|false` selects `LogDestination.disabled`.
- We treat `0` as invalid for the update/eval counts because a “no-op” run usually indicates a CLI
  mistake.
-/
def parse (exeName : String) (arguments : List String)
    (defaultLogPath : System.FilePath)
    (defaultUpdateCount defaultEvaluationInterval defaultEvaluationEpisodes
      defaultMaximumEvaluationSteps : Nat) :
    Except String (PPOOptions × List String) := do
  let (logRaw?, arguments) ← TorchLean.CLI.takeFlagValue? arguments "log"
  let (updateCount, arguments) ←
    TorchLean.CLI.takePositiveNatFlag arguments exeName "updates" (default := defaultUpdateCount)
  let (evaluationInterval, arguments) ←
    TorchLean.CLI.takePositiveNatFlag
      arguments exeName "eval-every" (default := defaultEvaluationInterval)
  let (evaluationEpisodes, arguments) ←
    TorchLean.CLI.takePositiveNatFlag
      arguments exeName "eval-episodes" (default := defaultEvaluationEpisodes)
  let (maximumEvaluationSteps, arguments) ←
    TorchLean.CLI.takePositiveNatFlag
      arguments exeName "eval-max-steps" (default := defaultMaximumEvaluationSteps)

  let logDestination :=
    Runtime.Training.LogDestination.resolve (.json defaultLogPath) logRaw?
  pure
    ({ updateCount
       evaluationInterval
       evaluationEpisodes
       maximumEvaluationSteps
       logDestination },
     arguments)

end PPOOptions
end cli
end rl
end TorchLean
