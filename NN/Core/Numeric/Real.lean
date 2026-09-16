/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Core.Numeric
public import NN.Core.Numeric.Angle.Real

/-!
# Exact-real instances for the foundational numeric interfaces

FloatLib supplies the canonical real `MathFunctions` instance through `NN.Core.Numeric`.
This facade additionally exports TorchLean's real polar-angle instance.
-/

@[expose] public section
