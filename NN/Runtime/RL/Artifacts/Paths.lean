/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# RL Artifact Paths

Trainers and editor-side viewers share the `data/rl/{run}_{artifact}.json` convention.
The run name is explicit so another example can reuse it without adding a filename declaration.
-/

@[expose] public section

namespace Runtime.RL.Artifacts

/-- JSON artifact path for a named run, defaulting to its training log.

For example, `path "ppo_gridworld" "policy"` names the saved policy snapshot.
Arguments are filename components supplied by the caller, not sanitized user input.
CLI flags can override the resulting default path.
-/
def path (run : String) (artifact : String := "trainlog") : System.FilePath :=
  s!"data/rl/{run}_{artifact}.json"

end Runtime.RL.Artifacts
