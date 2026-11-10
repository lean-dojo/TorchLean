/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Nerodia
import TorchLean.Verify.Capture
import TorchLean.NN.IR.Builder
import TorchLean.NN.IR.Table
import TorchLean.Tensor.Op.Pointwise.Real

open scoped Nerodia

/-!
# Python access to the IR checker and the op table

This optional extension calls the same checker as `verify ir` and returns the op table that the
Python captures read, the JSON of `verify ir --op-table` (`TorchLean.IR.tableOfNames`). JSON keeps
the boundary independent of Python's tensor representations. Graph execution and the scope of each
finding stay in the Lean checker. Python calls the compiled checker, so a result also relies on the
native compiler and the FFI boundary.
-/

py_module "_torchlean"

namespace TorchLeanPython

/-- `check artifact requests` runs the IR checker on the JSON `artifact` with the requests named in
the JSON array `requests` and returns the compact JSON report that `verify ir` prints. Malformed
JSON, or requests that are not an array of strings, raise Python `ValueError`. -/
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
  return (TorchLean.Capture.check artifact requests).compress

/-- Return the op table of the dtypes named in the JSON array `dtypes`: the JSON of
`TorchLean.IR.tableOfNames`, which `TorchLean.IR.tableOfNames_ok` describes. Malformed JSON, an
empty array, a name of no element type or a failed `TorchLean.IR.checkTable` raise Python
`ValueError`. -/
@[py_module_fn "op_table"]
def opTable (dtypes : String) : Nerodia.PyIO String := do
  let names ←
    match Lean.Json.parse dtypes >>= Lean.fromJson? (α := Array String) with
    | .ok values => pure values
    | .error message => Nerodia.raisePyValueError s!"invalid dtype names: {message}"
  match TorchLean.IR.tableOfNames names.toList with
  | .ok table => return table.compress
  | .error message => Nerodia.raisePyValueError message

end TorchLeanPython
