/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Training
public import TorchLean.NN.CLI.Command -- shake: keep
public import TorchLean.NN.Backend

/-!
# RL Command-Line Options

The PPO examples for CartPole and GridWorld in `TorchLeanExamples/Models/RL` share these flags:

- `--updates <n>`: how many update iterations to run,
- `--eval-every <n>`: evaluate every `n` updates,
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint,
- `--eval-max-steps <n>`: max steps per evaluation episode,
- `--log <path|off|none|false>`: where to write the widget-friendly TrainLog JSON.

`Options.parse` reads them for both examples, so we don't duplicate the flag handling, and
`Options.usage` builds their help text.
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace CLI

/-- Shared training, evaluation, and logging options for RL commands. -/
structure Options where
  /-- Number of training updates, set by `--updates`. -/
  updateCount : Nat
  /-- Number of updates between policy evaluations, set by `--eval-every`. -/
  evaluationInterval : Nat
  /-- Number of episodes in each evaluation, set by `--eval-episodes`. -/
  evaluationEpisodes : Nat
  /-- Step limit for each evaluation episode, set by `--eval-max-steps`. -/
  maximumEvaluationSteps : Nat
  /-- Destination of the training log, set by `--log`. -/
  logDestination : Training.LogDestination
deriving Repr

namespace Options

/-- Help text for RL commands, with optional environment-specific artifact flags. -/
def usage (exeName : String) (artifactOptions : Array String := #[]) : String :=
  String.intercalate "\n" <| (#[
    s!"Usage: scripts/lake.sh exe torchlean {exeName} [options]",
    "",
    "Training and evaluation:",
    "  --updates N         training update iterations",
    "  --eval-every N      updates between policy evaluations",
    "  --eval-episodes N   episodes in each evaluation",
    "  --eval-max-steps N  step limit for each evaluation episode",
    "  --log PATH|false    write a TrainLog JSON, or disable logging"
  ] ++ artifactOptions ++ #[
    "",
    "Runtime:",
    s!"  --device auto|{Backend.Device.help}",
    "  --arithmetic native",
    "  --seed N --show-backend"
  ]).toList

/--
Parse the shared RL options out of `arguments` and return them with the arguments left over.

A count flag that is absent takes the matching default argument, such as `defaultUpdateCount`.
Zero is rejected for every count, because a run that does nothing usually comes from a mistake on
the command line. `--log` with `false`, `off`, `none`, `no` or `disabled`, in any letter case,
selects `LogDestination.disabled`, another value names the JSON log file, and without the flag the
log goes to `defaultLogPath`.
-/
def parse (exeName : String) (arguments : List String)
    (defaultLogPath : System.FilePath)
    (defaultUpdateCount defaultEvaluationInterval defaultEvaluationEpisodes
      defaultMaximumEvaluationSteps : Nat) :
    Except String (Options × List String) := do
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
    Training.LogDestination.resolve (.jsonFile defaultLogPath) logRaw?
  pure
    ({ updateCount
       evaluationInterval
       evaluationEpisodes
       maximumEvaluationSteps
       logDestination },
     arguments)

end Options
end CLI
end RL
end TorchLean
