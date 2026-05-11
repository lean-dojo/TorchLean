/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.Runtime.Rationals.AutogradEngineTest
public import NN.Tests.Runtime.Rationals.MlpTest

/-!
# Suite

Aggregates the rational runtime test modules.

The rational suite is a lightweight semantic backstop: it exercises the same high-level model and
autograd code in a scalar setting that avoids floating-point roundoff, making regressions easier to
localize.
-/

@[expose] public section

namespace Tests
namespace Rationals

namespace Suite

/-- Unified Rational test entrypoint (called by `NN/Tests/Suite.lean`). -/
def run : IO Unit := do
  Tests.Rationals.AutogradEngine.run
  Tests.Rationals.run

end Suite

end Rationals
end Tests
