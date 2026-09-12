/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import NN.Tests.Interop.PyTorch

/-! Run the PyTorch graph-capture and state-dict regression suite. -/

/-- Report subprocess or validation failures as a nonzero process exit. -/
public def main (args : List String) : IO UInt32 :=
  TorchLean.CLI.exitOnError (NN.Tests.Interop.PyTorch.main args)
