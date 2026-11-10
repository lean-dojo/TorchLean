/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Trainer.Memory
public import TorchLeanExamples.Models.Options

/-!
# Runnable Example Training Support

Artifact paths, runtime metadata and banners used by the runnable model examples.
-/

@[expose] public section

namespace TorchLean.Examples.Training

/-- Training-log note recording the selected CUDA allocator-reporting cadence. -/
def cudaMemoryNote (runtime : Session.Config) (steps requested : Nat) : String :=
  s!"cuda_mem_watch={Trainer.Memory.cadence runtime steps requested}"

/-!
Shared CLI and logging names for the built-in runnable examples.

The names are public so that the examples stay short and readable. They live under
`Examples.Training`, and ordinary library code should usually use `Trainer`, `Data` and `optim`.
-/

/--
JSON artifact path `data/{dir}/{run}_{artifact}.json` of a runnable example, by default its
training log under `data/examples`. For example, `artifactPath "ppo_gridworld" "policy" "rl"` names
the GridWorld policy snapshot. The components are filename parts chosen by the caller, not
sanitized user input, and CLI flags can override the resulting default path.
-/
def artifactPath (run : String) (artifact : String := "trainlog") (dir : String := "examples") :
    System.FilePath :=
  System.FilePath.mk s!"data/{dir}/{run}_{artifact}.json"

/-- `device=...` note string used by example logs. -/
def deviceNote (runtime : Session.Config) : String :=
  s!"device={runtime.device.cliName}"

/-- Example headline with the selected device and an optional detail line. -/
def banner (exeName description : String) (details : Option String := none)
    (runtime : Session.Config) : String :=
  let headline := s!"{exeName}: {description} (device={runtime.device.cliName})"
  match details with
  | none => headline
  | some line => headline ++ "\n" ++ line

end TorchLean.Examples.Training
