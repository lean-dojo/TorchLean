/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/

module

import Nerodia
import NN.Verification.IR

open scoped Nerodia

/-!
# Python access to the IR checker

This optional extension calls the same checker as `verify ir`. JSON keeps the boundary independent
of Python's tensor representations; graph execution and the scope of each finding stay in the
existing Lean checker. Calling the compiled checker retains the native compiler and FFI boundary.
-/

py_module "_torchlean"

namespace TorchLeanPython

/-- Check an artifact using a JSON array of request names and return the unchanged JSON report.
Malformed JSON or request names of the wrong type raise Python `ValueError`. -/
@[py_module_fn "check"]
def check (artifact requests : String) : Nerodia.PyIO String := do
  let artifact ←
    match Lean.Json.parse artifact with
    | .ok value => pure value
    | .error message => Nerodia.raisePyValueError s!"invalid artifact JSON: {message}"
  let requests ←
    match Lean.Json.parse requests >>= Lean.fromJson? (α := Array String) with
    | .ok values => pure values
    | .error message => Nerodia.raisePyValueError s!"invalid check requests: {message}"
  return (NN.Verification.IR.check artifact requests).compress

/-- Return the PyTorch adapter emitted by the shared exporter. -/
@[py_module_fn "pytorch_adapter"]
def pytorchAdapter : String :=
  Export.PyTorch.TorchExport.script

end TorchLeanPython
