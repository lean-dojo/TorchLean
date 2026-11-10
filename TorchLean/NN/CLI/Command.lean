/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.CLI.Parser

/-!
# Command Process Boundaries

Helpers that carry pure parser failures across an executable's `IO` boundary. `CLI.orThrow`
turns a parser result into an `IO` error, `CLI.requireNoArgs` rejects leftover arguments, and
`CLI.exitOnError` wraps an entry point so that a failure prints as `error: …` on stderr and the
program exits with status 1.
-/

@[expose] public section

namespace TorchLean.CLI

/--
Lift a parser result into `IO`, prefixing failures with the executable name.

Some shared parsers, `takePositiveNatFlag` among them, take `exeName` and already name the
program in their message. The prefix is added only when the message does not start with it, so a
failure never reads `quickstart_mlp: quickstart_mlp: --steps must be > 0`. Because the check is
made here, one `parse` function can mix both kinds of parser and its callers need not track which
kind each one is.
-/
def orThrow {α : Type} (exeName : String) : Except String α → IO α
  | .ok value => pure value
  | .error message =>
      let label := s!"{exeName}: "
      throw <| IO.userError (if label.isPrefixOf message then message else label ++ message)

/-- Fail with an `IO` error prefixed with `exeName` when arguments remain after parsing. -/
def requireNoArgs (exeName : String) (arguments : List String) : IO Unit :=
  orThrow exeName <| checkNoArgs arguments

/--
Run an executable's entry point and report a failure the way a command-line tool should.

The parsers reject bad input by throwing an `IO` error. If that error escapes `main`, the Lean
runtime prints it as `uncaught exception: quickstart_mlp: --steps must be > 0`, which reads like
a crash in the middle of a run although the program rejected its command line on purpose. This
wrapper catches the error, prints it as `error: …` on stderr, and returns exit status 1. That is
the status the runtime would have produced, so only the wording changes.

Example:
```lean
def main (arguments : List String) : IO UInt32 := CLI.exitOnError do
  let run ← Trainer.RunConfig.parseCommandLine "train" arguments
  IO.println s!"device: {run.device.cliName}"
  pure 0
```
-/
def exitOnError (body : IO UInt32) : IO UInt32 := do
  try
    body
  catch e =>
    IO.eprintln s!"error: {e}"
    pure 1

end TorchLean.CLI
