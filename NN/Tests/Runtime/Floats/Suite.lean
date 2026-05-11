/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Floats.AllAutogradTests
public import NN.Tests.Runtime.Floats.KnnSmoke
public import NN.Tests.Runtime.Floats.ModelsSmoke
public import NN.Tests.Runtime.Floats.PINNDerivResidual
public import NN.Tests.Runtime.Floats.RLSmoke
public import NN.Tests.Runtime.Floats.RnnGruLstmBpttSmoke
public import NN.Tests.Runtime.Floats.TorchLeanIRExecEquivSmoke
public import NN.Tests.Runtime.Floats.TorchLeanIndexShapeSmoke
public import NN.Tests.Runtime.Floats.TorchLeanOpsSmoke
public import NN.Tests.Runtime.Floats.TorchLeanSpecMlpEquivSmoke

/-!
# Suite

Aggregates the float runtime smoke and autograd test suites.

These are fast sanity checks intended to catch runtime regressions in the executable float backends
and keep public examples from silently breaking. They complement the proof modules: tests cover
runtime wiring, floating-point behavior, parser glue, and execution paths that are intentionally
outside the kernel of Lean theorems.
-/

@[expose] public section

namespace Tests
namespace Floats

/-- Unified Float test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  Tests.Floats.runAllAutogradTests
  Tests.Floats.ModelsSmoke.run
  Tests.Floats.KNN.run
  Tests.Floats.RLSmoke.run
  Tests.Floats.BPTT.run
  Tests.Floats.PinnDerivResidual.run
  Tests.Floats.TorchLeanOpsSmoke.run
  Tests.Floats.TorchLeanIndexShapeSmoke.run
  Tests.Floats.TorchLeanSpecMLPEquivSmoke.run
  Tests.Floats.TorchLeanIRExecEquivSmoke.run

end Floats
end Tests
