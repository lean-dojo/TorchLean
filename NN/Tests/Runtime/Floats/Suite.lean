/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.ParameterAliases
public import NN.Tests.Runtime.IRExecScalingRegression
public import NN.Tests.Floats.IEEE32IntervalBounds
public import NN.Tests.Runtime.Floats.AllAutogradTests
public import NN.Tests.Runtime.Floats.CertificatePreconditions
public import NN.Tests.Runtime.Floats.PyTorchRoundtripParity
public import NN.Tests.Runtime.Floats.RLCheck
public import NN.Tests.Runtime.Floats.SessionRefIdentity
public import NN.Tests.Runtime.Floats.ONNXBridge

/-!
# Suite

Aggregates the float runtime and autograd test suites.

These are runtime checks for regressions in the executable float backends
and keep public examples from silently breaking. They complement the proof modules: tests cover
runtime wiring, floating-point behavior, parser glue, and execution paths that sit outside the
kernel of Lean theorems.
-/

@[expose] public section

namespace Tests
namespace Floats

/-- Unified Float test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  NN.Tests.Runtime.ParameterAliases.runCpu
  Tests.Floats.runAllAutogradTests
  Tests.Floats.CertificatePreconditions.run
  Tests.Floats.PyTorchRoundtripParity.run
  Tests.Floats.RLCheck.run
  Tests.Floats.SessionRefIdentity.run
  Tests.Floats.ONNXBridge.run
  Tests.IRExecScalingRegression.check
  NN.Tests.Floats.IEEE32IntervalBounds.run

end Floats
end Tests
