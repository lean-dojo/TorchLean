/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Table

/-!
# Running the Python captures

The tests that run a Python capture (`python3 -m torchlean.export`, `python3 -m torchlean.onnx`)
write Lean's op table to a file of their work directory with `writeOpTable`. They run Python with
`runPython`, which sets `TORCHLEAN_OP_TABLE` to that file and puts the repository's `python`
directory first on `PYTHONPATH`. The element type names of the table come from the Python package
(`DTYPES` of `python/torchlean/_table.py`), and the table is computed when the test runs.
-/

@[expose] public section

namespace TorchLean.Test.IR.Capture

/-- Run `python3` with the arguments `args` from the current directory, with `TORCHLEAN_OP_TABLE`
naming the table file `table` and the current directory's `python` directory first on
`PYTHONPATH`, and return its output. It sets only the environment and checks nothing. -/
def python (args : Array String) (table : System.FilePath) : IO IO.Process.Output := do
  let cwd ← IO.currentDir
  let rest : System.SearchPath := match ← IO.getEnv "PYTHONPATH" with
    | some p => if p.isEmpty then [] else System.SearchPath.parse p
    | none => []
  let env := #[("TORCHLEAN_OP_TABLE", some (cwd / table).toString),
    ("PYTHONPATH", some (System.SearchPath.toString ((cwd / "python") :: rest)))]
  IO.Process.output { cmd := "python3", args := args, cwd := some ".", env := env }

/-- `python args table`, failing with the context `ctx`, the arguments and the standard error when
Python exits nonzero, and returning the standard output otherwise. -/
def runPython (ctx : String) (args : Array String) (table : System.FilePath) : IO String := do
  let out ← python args table
  if out.exitCode != 0 then
    throw <| IO.userError
      (s!"{ctx}: subprocess failed.\n" ++
        s!"cmd=python3\nargs={args.toList}\nexit={out.exitCode}\nstderr:\n{out.stderr}\n")
  pure out.stdout

/-- Write to `path` the JSON of `IR.tableOfNames` at the element type names that `DTYPES` of
`python/torchlean/_table.py` lists, read by running Python. `IR.tableOfNames_ok` shows that
this is `IR.table` at the element types read from those names. -/
def writeOpTable (path : System.FilePath) : IO Unit := do
  let out ← runPython "Op table names"
    #["-c", "from torchlean._table import DTYPES; print('\\n'.join(DTYPES))"] path
  let names := (out.splitOn "\n").filter (!·.isEmpty)
  IO.FS.writeFile path (← IO.ofExcept (IR.tableOfNames names)).compress

end TorchLean.Test.IR.Capture
