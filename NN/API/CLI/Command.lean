/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Parser

/-!
# Command Process Boundaries

Helpers for reporting pure parser failures through an executable's `IO` boundary.
-/

@[expose] public section

namespace TorchLean.CLI

/-- Lift a shared CLI parser result into `IO.userError`. -/
def orThrowIO {α : Type} (x : Except String α) : IO α :=
  match x with
  | .ok value => pure value
  | .error message => throw <| IO.userError message

/--
Lift a parser result into `IO`, prefixing failures with the executable name.

Several of the shared parsers take `exeName` themselves and have already named the program in their
message, `takePositiveNatFlag` among them. Prefixing again gives
`quickstart_mlp: quickstart_mlp: --steps must be > 0`, so the name is only added when it is not
there yet. Deciding that here rather than at every call site is what lets one `parse` function mix
the two parser styles without the caller having to remember which is which.
-/
def orThrow {α : Type} (exeName : String) : Except String α → IO α
  | .ok value => pure value
  | .error message =>
      let label := s!"{exeName}: "
      throw <| IO.userError (if label.isPrefixOf message then message else label ++ message)

/-- Parse `--seed N`, returning the selected seed and remaining arguments. -/
def seed (exeName : String) (arguments : List String) (default : Nat := 0) :
    IO (Nat × List String) :=
  orThrow exeName <| takeSeed arguments (default := default)

/-- Parse a positive natural-number flag, using `default` when it is absent. -/
def positiveNatFlag
    (exeName : String)
    (arguments : List String)
    (name : String)
    (default : Nat) :
    IO (Nat × List String) :=
  orThrow exeName <|
    takePositiveNatFlag arguments exeName name (default := default)

/-- Fail when command-specific arguments remain after parsing. -/
def requireNoArgs (exeName : String) (arguments : List String) : IO Unit :=
  orThrow exeName <| checkNoArgs arguments

/--
Run an executable's entry point and report a failure the way a command-line tool should.

Every parser above refuses bad input by throwing an `IO` error, which is the right thing for the
parser to do. Left alone, that error escapes `main` and the Lean runtime prints it as
`uncaught exception: quickstart_mlp: --steps must be > 0`, which reads like a crash in the middle
of a run even though the program rejected the command line on purpose. This wrapper catches the
error at the last possible moment, prints it as `error: …` on stderr, and returns exit status 1.
The status is what the runtime would have produced anyway; only the wording changes.
-/
def exitOnError (body : IO UInt32) : IO UInt32 := do
  try
    body
  catch e =>
    IO.eprintln s!"error: {e}"
    pure 1

end TorchLean.CLI
