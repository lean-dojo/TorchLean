/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Data.Parsing

/-!
# TorchLean CLI Parser

Pure command-line parsers shared by examples, verification tools and application API helpers.
The definitions live directly under `TorchLean.CLI`. The `take` and `require` parsers return
their result together with the arguments they did not consume, in their original order.

* String flags: `takeFlagValue?`, `takeFlagValue`, `requireFlagValue`, `takeParsedFlag` and
  `hasFlagValue`.
* Number and path flags: `takeNatFlag`, `takePositiveNatFlag`, `takeFloatFlag`,
  `requireFloatFlag`, `takePositiveFloatFlag`, `takeNonnegativeFloatFlag`, `takeSeed`,
  `takePathFlag`, `requirePathFlag` and `takePairedPathFlags`.
* Boolean flags: `takeBoolFlag`, `takeBoolValueFlag` and `takeSwitch`.
* Positional arguments and the rest of the command line: `takePositional`,
  `normalizePathFlag`, `dropDashDash`, `hasHelp` and `checkNoArgs`.

The module depends on no tensor or runtime modules, so lightweight artifact checkers can reuse
the CLI surface without importing the full public API.
-/

@[expose] public section

namespace TorchLean
namespace CLI

/--
Take an optional `--key` value from an argument list.

Accepted forms:
- `--key value`
- `--key=value`

Duplicate occurrences are rejected, and so is a final `--key` with no value after it. The
returned argument list contains every unconsumed argument in its original order.
-/
def takeFlagValue? (arguments : List String) (key : String) :
    Except String (Option String × List String) :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  let rec
    /-- Scan the remaining arguments, carrying the value found so far and the unconsumed
    arguments in reverse order. -/
    go :
        List String → Option String → List String → Except String (Option String × List String)
      | [], found, acc => .ok (found, acc.reverse)
      | a :: rest, found, acc =>
          if a == keyTok then
            match rest with
            | [] => .error s!"{keyTok}: expected a value"
            | v :: rest' =>
                if found.isSome then
                  .error s!"{keyTok}: duplicate flag"
                else
                  go rest' (some v) acc
          else if a.startsWith eqPrefix then
            let v := (a.drop eqPrefix.length).toString
            if found.isSome then
              .error s!"{keyTok}: duplicate flag"
            else
              go rest (some v) acc
          else
            go rest found (a :: acc)
  go arguments none []

/-- Select a default when an optional parser result contains no value. -/
private def parsedWithDefault
    {α : Type}
    (result : Except String (Option α × List String))
    (default : α) :
    Except String (α × List String) := do
  let (value?, rest) ← result
  pure (value?.getD default, rest)

/-- Require an optional parser result to contain a value, failing with `missing` otherwise. -/
private def requireParsed
    {α : Type}
    (result : Except String (Option α × List String))
    (missing : String) :
    Except String (α × List String) := do
  let (value?, rest) ← result
  match value? with
  | some value => pure (value, rest)
  | none => throw missing

/-- Parse an optional string-valued flag with a caller-supplied value decoder. -/
private def takeParsedFlag?
    {α : Type}
    (arguments : List String)
    (key : String)
    (parse : String → Except String α) :
    Except String (Option α × List String) := do
  let (raw?, rest) ← takeFlagValue? arguments key
  match raw? with
  | none => pure (none, rest)
  | some raw => pure (some (← parse raw), rest)

/--
Parse an optional string-valued flag and fall back to a provided default.

It suits a command parser that needs a concrete string. `takeFlagValue?` keeps an absent flag
visible as `none`.
-/
opaque takeFlagValue
    (arguments : List String)
    (key : String)
    (default : String) :
    Except String (String × List String) :=
  parsedWithDefault (takeFlagValue? arguments key) default

/-- Parse a required string-valued flag and return the remaining arguments. A missing flag fails
with `missing?`, or with `missing --key=<value>` when `missing?` is `none`. -/
opaque requireFlagValue
    (arguments : List String)
    (key : String)
    (missing? : Option String := none) :
    Except String (String × List String) :=
  requireParsed (takeFlagValue? arguments key) (missing?.getD s!"missing --{key}=<value>")

/--
Parse an optional string-valued flag, fall back to a provided default spelling when absent, and
decode the selected spelling with a caller-supplied parser.

Commands use it for enum-like flags, whose valid strings differ from command to command.
-/
opaque takeParsedFlag
    {α : Type}
    (arguments : List String)
    (key : String)
    (default : String)
    (parse : String → Except String α) :
    Except String (α × List String) := do
  let (value?, rest) ← takeParsedFlag? arguments key parse
  match value? with
  | some value => pure (value, rest)
  | none => pure (← parse default, rest)

/-- Return true when `arguments` contains `--key` or an argument that starts with `--key=`, the two
spellings of a string-valued flag. Whether a value follows `--key` is not checked. -/
def hasFlagValue (arguments : List String) (key : String) : Bool :=
  let eqPrefix := s!"--{key}="
  let keyTok := s!"--{key}"
  arguments.any (fun a => a == keyTok || a.startsWith eqPrefix)

/-- Take a no-value boolean flag, returning whether it appeared. A repeated flag is an error. -/
def takeBoolFlag (arguments : List String) (key : String) :
    Except String (Bool × List String) := do
  let keyTok := s!"--{key}"
  let rec
    /-- Scan the remaining arguments, carrying whether the flag appeared and the other arguments
    in reverse order. -/
    go : List String → Bool → List String → Except String (Bool × List String)
      | [], seen, acc => pure (seen, acc.reverse)
      | a :: rest, seen, acc =>
          if a == keyTok then
            if seen then
              throw s!"{keyTok}: duplicate flag"
            else
              go rest true acc
          else
            go rest seen (a :: acc)
  go arguments false []

/-- Drop the leading `--` separator commonly used with `lean --run`. -/
def dropDashDash (arguments : List String) : List String :=
  match arguments with
  | "--" :: rest => rest
  | xs => xs

/-- Return true when the argument list contains `--help` or `-h`. -/
def hasHelp (arguments : List String) : Bool :=
  arguments.contains "--help" || arguments.contains "-h"

/-- Fail with `unexpected arguments: …` if any CLI arguments remain unconsumed. -/
def checkNoArgs (arguments : List String) : Except String Unit :=
  if arguments.isEmpty then
    .ok ()
  else
    .error s!"unexpected arguments: {arguments}"

/--
Take at most one positional argument, leaving flags untouched.

It suits commands with a single optional artifact path plus named flags. A second positional
argument is reported as an error. Every argument that does not start with `--` counts as
positional, so the value of a flag written as `--key value` must be consumed first.
-/
opaque takePositional? (arguments : List String) :
    Except String (Option String × List String) :=
  let rec go :
      List String → Option String → List String → Except String (Option String × List String)
    | [], found, acc => .ok (found, acc.reverse)
    | a :: rest, found, acc =>
        if a.startsWith "--" then
          go rest found (a :: acc)
        else if found.isSome then
          .error s!"unexpected positional argument: {a}"
        else
          go rest (some a) acc
  go arguments none []

/-- Take one optional positional argument and fall back to `default` when it is absent. -/
opaque takePositional (arguments : List String) (default : String) :
    Except String (String × List String) :=
  parsedWithDefault (takePositional? arguments) default

/--
Normalize commands that accept either a positional path or a named path flag.

A leading `--` separator is dropped first. If `--key` or `--key=...` is then present, the list is
returned as it is. Otherwise a leading positional argument is rewritten to `--key=<path>`, and an
empty list or a list that starts with a flag gets `--key=<default>` in front.
-/
def normalizePathFlag
    (arguments : List String)
    (key default : String) :
    List String :=
  let arguments := dropDashDash arguments
  if hasFlagValue arguments key then
    arguments
  else
    match arguments with
    | [] => [s!"--{key}={default}"]
    | a :: rest =>
        if a.startsWith "--" then
          s!"--{key}={default}" :: a :: rest
        else
          s!"--{key}={a}" :: rest

/-- Take an optional natural-number flag. -/
opaque takeNatFlag? (arguments : List String) (key : String) :
    Except String (Option Nat × List String) :=
  takeParsedFlag? arguments key fun value =>
    match value.toNat? with
    | some n => pure n
    | none => throw s!"--{key}: expected a natural number, got `{value}`"

/--
Parse an optional natural-number flag and fall back to the provided default.
-/
opaque takeNatFlag
    (arguments : List String)
    (key : String)
    (default : Nat) :
    Except String (Nat × List String) :=
  parsedWithDefault (takeNatFlag? arguments key) default

/--
Parse an optional natural-number flag, fall back to a default, and require that the selected value
is strictly positive. A zero value fails with `exeName: --key must be > 0`.
-/
def takePositiveNatFlag
    (arguments : List String)
    (exeName : String)
    (key : String)
    (default : Nat) :
    Except String (Nat × List String) := do
  let (value, rest) ← takeNatFlag arguments key (default := default)
  if value = 0 then
    throw s!"{exeName}: --{key} must be > 0"
  pure (value, rest)

/--
Parse a signed decimal float literal with `Data.parseFloatString`: an optional sign,
digits with an optional fraction, and an optional exponent, rounded once to the nearest `Float`.
-/
def parseFloatLit? (s : String) : Option Float :=
  (Data.parseFloatString "float" s).toOption

/-- Take an optional floating-point flag. -/
opaque takeFloatFlag? (arguments : List String) (key : String) :
    Except String (Option Float × List String) :=
  takeParsedFlag? arguments key fun value =>
    match parseFloatLit? value with
    | some x => pure x
    | none => throw s!"--{key}: expected a float literal, got `{value}`"

/--
Parse an optional floating-point flag and fall back to the provided default.
-/
opaque takeFloatFlag
    (arguments : List String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) :=
  parsedWithDefault (takeFloatFlag? arguments key) default

/-- Parse a required floating-point flag and return the remaining arguments. A missing flag fails
with `missing?`, or with `missing --key=<float>` when `missing?` is `none`. -/
opaque requireFloatFlag
    (arguments : List String)
    (key : String)
    (missing? : Option String := none) :
    Except String (Float × List String) :=
  requireParsed (takeFloatFlag? arguments key)
    (missing?.getD s!"missing --{key}=<float>")

/-- Parse a CLI boolean value: `true`, `false`, `1` or `0`, with letter case ignored. -/
private def parseBoolLit (s : String) : Option Bool :=
  match s.toLower with
  | "true" => some true
  | "1" => some true
  | "false" => some false
  | "0" => some false
  | _ => none

/-- Take an optional boolean flag written with its value, as in `--key true` or `--key=0`. -/
opaque takeBoolValueFlag? (arguments : List String) (key : String) :
    Except String (Option Bool × List String) :=
  takeParsedFlag? arguments key fun value =>
    match parseBoolLit value with
    | some b => pure b
    | none => throw s!"--{key}: expected true, false, 1, or 0; got `{value}`"

/-- Parse an optional boolean-valued flag and fall back to the provided default. -/
opaque takeBoolValueFlag (arguments : List String) (key : String) (default : Bool) :
    Except String (Bool × List String) :=
  parsedWithDefault (takeBoolValueFlag? arguments key) default

/--
Remove a boolean flag that may be written either as a bare switch or with a value.

Accepted forms:
- `--key`
- `--key=true`
- `--key=false`
- `--key true`
- `--key false`

The values `1` and `0` stand for `true` and `false`, and case is ignored. When `--key` is followed
by a non-boolean token, the flag is treated as a bare switch and the next token is left for the
caller. Duplicate occurrences are rejected.
-/
opaque takeSwitch? (arguments : List String) (key : String) :
    Except String (Option Bool × List String) := do
  let keyTok := s!"--{key}"
  let eqPrefix := s!"--{key}="
  let rec go (arguments : List String) (seen : Option Bool) (acc : List String) :
      Except String (Option Bool × List String) := do
    match arguments with
    | [] => pure (seen, acc.reverse)
    | a :: rest =>
        if a == keyTok then
          if seen.isSome then
            throw s!"{keyTok}: duplicate flag"
          else
            match _hRest : rest with
            | v :: rest' =>
                match parseBoolLit v with
                | some b => go rest' (some b) acc
                | none => go rest (some true) acc
            | [] => go rest (some true) acc
        else if a.startsWith eqPrefix then
          if seen.isSome then
            throw s!"{keyTok}: duplicate flag"
          else
            let raw := (a.drop eqPrefix.length).toString
            match parseBoolLit raw with
            | some b => go rest (some b) acc
            | none => throw s!"{keyTok}: expected true, false, 1, or 0; got `{raw}`"
        else
          go rest seen (a :: acc)
    termination_by arguments.length
    decreasing_by
      all_goals
        try subst rest
        simp
      exact Nat.lt_succ_of_lt (Nat.lt_succ_self _)
  go arguments none []

/-- Parse a bare-or-valued boolean flag and fall back to the provided default. -/
opaque takeSwitch (arguments : List String) (key : String) (default : Bool) :
    Except String (Bool × List String) :=
  parsedWithDefault (takeSwitch? arguments key) default

/--
Parse an optional floating-point flag, fall back to the provided default, and require that the
selected value is finite and strictly positive. Other values fail with a message that starts with
`exeName`.
-/
def takePositiveFloatFlag
    (arguments : List String)
    (exeName : String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) := do
  let (value, rest) ← takeFloatFlag arguments key (default := default)
  unless value.isFinite && 0.0 < value do
    throw s!"{exeName}: --{key} must be finite and > 0"
  pure (value, rest)

/--
Parse an optional floating-point flag, fall back to the provided default, and require that the
selected value is finite and nonnegative. Other values fail with a message that starts with
`exeName`.
-/
def takeNonnegativeFloatFlag
    (arguments : List String)
    (exeName : String)
    (key : String)
    (default : Float) :
    Except String (Float × List String) := do
  let (value, rest) ← takeFloatFlag arguments key (default := default)
  unless value.isFinite && 0.0 <= value do
    throw s!"{exeName}: --{key} must be finite and >= 0"
  pure (value, rest)

/-- Take an optional path flag. -/
opaque takePathFlag? (arguments : List String) (key : String) :
    Except String (Option System.FilePath × List String) :=
  takeParsedFlag? arguments key fun value => pure value

/--
Parse an optional path flag and fall back to the provided default path.

It suits an example parser that needs a concrete path. `takePathFlag?` keeps an absent flag
visible as `none`.
-/
opaque takePathFlag
    (arguments : List String)
    (key : String)
    (default : System.FilePath) :
    Except String (System.FilePath × List String) :=
  parsedWithDefault (takePathFlag? arguments key) default

/--
Parse a required path flag such as `--data-file corpus.txt`.

The error message starts with `exeName` when it is nonempty.
-/
opaque requirePathFlag
    (arguments : List String)
    (key : String)
    (exeName : String := "") :
    Except String (System.FilePath × List String) :=
  let messagePrefix := if exeName.isEmpty then "" else s!"{exeName}: "
  requireParsed
    (takePathFlag? arguments key)
    s!"{messagePrefix}missing required --{key} <path>"

/--
Parse two optional path flags that must appear together if either one is present.

It suits paired artifacts such as a tokenizer's vocabulary and merge files, where one path alone
is not meaningful.
-/
def takePairedPathFlags
    (arguments : List String)
    (firstKey secondKey : String) :
    Except String ((Option System.FilePath × Option System.FilePath) × List String) := do
  let (firstPath?, arguments) ← takePathFlag? arguments firstKey
  let (secondPath?, arguments) ← takePathFlag? arguments secondKey
  match firstPath?, secondPath? with
  | some _, none => throw s!"--{firstKey} requires --{secondKey}"
  | none, some _ => throw s!"--{secondKey} requires --{firstKey}"
  | _, _ => pure ((firstPath?, secondPath?), arguments)

/-- Parse an optional `--seed` flag, falling back to `default` when it is absent. -/
def takeSeed (arguments : List String) (default : Nat := 0) :
    Except String (Nat × List String) :=
  takeNatFlag arguments "seed" (default := default)

end CLI
end TorchLean
