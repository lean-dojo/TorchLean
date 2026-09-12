/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Scalar.Representable

/-!
# Rounded Scalar Interface

`NF` is the scalar-facing interface to generic rounding.  This folder keeps the carrier and its
representability invariant together.  The raw carrier may contain any real value; statements that
require membership in the declared grid use `NF.IsRepresentable` explicitly.
-/

@[expose] public section
