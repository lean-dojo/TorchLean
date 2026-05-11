/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.ODE.Ast

/-!
# Parse

Hand-rolled parser for ODE RHS expressions.

Grammar (informal):
  expr   := term (('+' | '-') term)*
  term   := factor (('*' | '/') factor)*
  factor := unary ('^' nat)?
  unary  := '-' unary | primary
  primary:= number | 't' | 'u' | ident '(' expr ')' | ident | '(' expr ')'

Supported unary functions: sin, cos, exp, log.
Supported identifiers: pi.

Exponentiation is expanded into repeated multiplication when `^ n` is given.
-/

@[expose] public section


namespace NN.Verification.ODE.Parse

open NN.Verification.ODE

/-!
This parser is part of the executable ODE verifier.  We keep the grammar direct and hand-written so
that the accepted certificate syntax is visible in one file, with predictable errors and no hidden
parser-combinator behavior.

The output AST is `NN.Verification.ODE.Ast.Expr`.
-/

/-! ## Parser state and low-level helpers -/

/--
Parser state for scanning a `String` by byte-position.

We track the source string `s` and the current raw byte index `i`.
-/
structure State where
  /-- Source text being parsed. -/
  s : String
  /-- Current raw byte offset in `s`. -/
  i : String.Pos.Raw := 0

/-- Peek at the current character, if any, without advancing. -/
@[inline] def peek (st : State) : Option Char := String.Pos.Raw.get? st.s st.i
/-- Advance the current position by one character. -/
@[inline] def bump (st : State) : State := { st with i := String.Pos.Raw.next st.s st.i }
/-- A fuel budget derived from the remaining input length (used to guarantee termination). -/
@[inline] def fuelOf (st : State) : Nat := (st.s.rawEndPos.byteIdx - st.i.byteIdx) + 1

/- ASCII whitespace predicate used by `skipWs`. -/
namespace Internal

/-- Internal: ASCII whitespace predicate used by `skipWs`. -/
def isWs (c : Char) : Bool :=
  c = ' ' || c = '\t' || c = '\n'

/-- Fuel-bounded whitespace skipping (implementation of `skipWs`). -/
def skipWsFuel : Nat → State → State
  | 0, st => st
  | Nat.succ fuel, st =>
    match peek st with
    | some c => if isWs c then skipWsFuel fuel (bump st) else st
    | none => st

end Internal

/-- Skip ASCII whitespace (`' '`, `'\t'`, `'\n'`). -/
def skipWs (st : State) : State := Internal.skipWsFuel (fuelOf st) st

/- Fuel-bounded implementation of `takeWhile`. -/
namespace Internal

/-- Internal: fuel-bounded implementation of `takeWhile`. -/
def takeWhileFuel (fuel : Nat) (p : Char → Bool) (acc : String) (st : State) : String × State :=
  match fuel with
  | 0 => (acc, st)
  | Nat.succ fuel =>
    match peek st with
    | some c =>
      if p c then takeWhileFuel fuel p (acc.push c) (bump st) else (acc, st)
    | none => (acc, st)

end Internal

/--
Consume consecutive characters satisfying `p`, accumulating them into `acc`.

Returns the consumed text and the updated parser state.
-/
def takeWhile (p : Char → Bool) (acc : String) (st : State) : String × State :=
  Internal.takeWhileFuel (fuelOf st) p acc st

/-- Parse a signed decimal number without exponent, e.g. `-12.34`. -/
def parseNumber (st : State) : Except String (Float × State) := do
  let st0 := skipWs st
  let (sgn, st1) :=
    match peek st0 with
    | some '-' => (-1.0, bump st0)
    | _ => (1.0, st0)
  let (intTxt, st2) := takeWhile (fun c => c.isDigit) "" st1
  if intTxt = "" then
    .error "expected number"
  let intVal : Float :=
    Float.ofNat (intTxt.toList.foldl (fun (acc : Nat) (c : Char) => acc * 10 + (c.toNat -
      '0'.toNat)) 0)
  let (fracVal, st3) :=
    match peek st2 with
    | some '.' =>
      let st2' := bump st2
      let (fracTxt, st2'') := takeWhile (fun c => c.isDigit) "" st2'
      if fracTxt = "" then (0.0, st2'')
      else
        let num : Nat := fracTxt.toList.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0
        let den : Nat := Nat.pow 10 fracTxt.length
        ((Float.ofNat num) / (Float.ofNat den), st2'')
    | _ => (0.0, st2)
  .ok (sgn * (intVal + fracVal), st3)

/-- Parse a natural number (decimal digits) used for exponents `^ n`. -/
def parseNat (st : State) : Except String (Nat × State) := do
  let st0 := skipWs st
  let (txt, st1) := takeWhile (fun c => c.isDigit) "" st0
  if txt = "" then .error "expected natural number"
  let n : Nat := txt.toList.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0
  .ok (n, st1)

/-- Parse an identifier consisting of letters/digits/underscore. -/
def parseIdent (st : State) : Except String (String × State) := do
  let st0 := skipWs st
  let (txt, st1) := takeWhile (fun c => c.isAlpha || c.isDigit || c = '_' ) "" st0
  if txt = "" then .error "expected identifier" else .ok (txt, st1)

/-! ## Built-in constants and unary functions -/

/- Interpret built-in constants like `pi` / `π`. -/
namespace Internal

/-- Internal: interpret built-in constants like `pi` / `π`. -/
def constOfIdent? (id : String) : Option Float :=
  if id = "pi" ∨ id = "π" then some 3.14159265358979323846 else none

/-- Interpret supported unary function names (e.g. `sin`, `cos`, `exp`, `log`). -/
def applyFunc? (id : String) (arg : Expr) : Option Expr :=
  if id = "sin" then some (.sin arg)
  else if id = "cos" then some (.cos arg)
  else if id = "exp" then some (.exp arg)
  else if id = "log" then some (.log arg)
  else none

end Internal

/-! ## Recursive descent: expression/term/factor/unary/primary -/

namespace Internal

mutual
  /--
  Internal: parse an `expr` (addition/subtraction chain), with an explicit fuel budget.

  This is a plain `def` (not `private`) because the module is in a `public` section for
  export/doc tooling, and Lean 4.29 disallows referring to `private` declarations from
  public ones.
  -/
  def parseExprFuel (fuel : Nat) (st : State) : Except String (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (term0, st1) ← parseTermFuel fuel st
      let rec loop (fuel : Nat) (acc : Expr) (st : State) : Except String (Expr × State) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match peek st' with
          | some '+' =>
            let (t2, st2) ← parseTermFuel fuel (bump st')
            loop fuel (.add acc t2) st2
          | some '-' =>
            let (t2, st2) ← parseTermFuel fuel (bump st')
            loop fuel (.sub acc t2) st2
          | _ => .ok (acc, st')
      loop fuel term0 st1

  /-- Parse a `term` (multiplication/division chain), with an explicit fuel budget. -/
  def parseTermFuel (fuel : Nat) (st : State) : Except String (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (f, st1) ← parseFactorFuel fuel st
      let rec loop (fuel : Nat) (acc : Expr) (st : State) : Except String (Expr × State) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match peek st' with
          | some '*' =>
            let (f2, st2) ← parseFactorFuel fuel (bump st')
            loop fuel (.mul acc f2) st2
          | some '/' =>
            let (f2, st2) ← parseFactorFuel fuel (bump st')
            loop fuel (.div acc f2) st2
          | _ => .ok (acc, st')
      loop fuel f st1

  /-- Parse a `factor` (unary with optional exponentiation), with an explicit fuel budget. -/
  def parseFactorFuel (fuel : Nat) (st : State) : Except String (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (unary0, st1) ← parseUnaryFuel fuel st
      let st1' := skipWs st1
      match peek st1' with
      | some '^' =>
        let (n, st2) ← parseNat (bump st1')
        if n ≤ 1 then
          .ok (unary0, st2)
        else
          let rec powMul (base : Expr) (k : Nat) (acc : Expr) : Expr :=
            match k with
            | 0 => acc
            | Nat.succ m => powMul base m (.mul acc base)
          .ok (powMul unary0 (n - 1) unary0, st2)
      | _ => .ok (unary0, st1')

  /-- Parse a `unary` (leading negations), with an explicit fuel budget. -/
  def parseUnaryFuel (fuel : Nat) (st : State) : Except String (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match peek st' with
      | some '-' =>
        let (e, st1) ← parseUnaryFuel fuel (bump st')
        .ok (.neg e, st1)
      | _ =>
        parsePrimaryFuel fuel st'

  /-- Parse a `primary` atom (number/variable/function-call/parentheses), with an explicit fuel
    budget. -/
  def parsePrimaryFuel (fuel : Nat) (st : State) : Except String (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match peek st' with
      | some '(' =>
        let (e, st1) ← parseExprFuel fuel (bump st')
        let st2 := skipWs st1
        match peek st2 with
        | some ')' => .ok (e, bump st2)
        | _ => .error "expected ')'"
      | some c =>
        if c.isDigit || c = '-' then
          let (v, st1) ← parseNumber st'
          .ok (.const v, st1)
        else if c = 't' then
          .ok (.t, bump st')
        else if c = 'u' then
          .ok (.u, bump st')
        else if c.isAlpha || c = 'π' then
          let (id, st1) ← parseIdent st'
          let st1' := skipWs st1
          match peek st1' with
          | some '(' =>
            let (arg, st2) ← parseExprFuel fuel (bump st1')
            let st3 := skipWs st2
            match peek st3 with
            | some ')' =>
              match applyFunc? id arg with
              | some app => .ok (app, bump st3)
              | none => .error s!"unknown function: {id}"
            | _ => .error "expected ')'"
          | _ =>
            match constOfIdent? id with
            | some v => .ok (.const v, st1')
            | none => .error s!"unknown identifier: {id}"
        else
          .error s!"unexpected char: {c}"
      | none =>
        .error "unexpected end of input"
end

end Internal

/--
Parse an ODE RHS expression string into an AST.

This is the user-facing entrypoint for the ODE verifier: it parses a string like
`"sin(t) + u^2"` into an `Expr` (`NN.Verification.ODE.Ast.Expr`).
-/
def parseExpr (s : String) : Except String Expr :=
  let st0 : State := { s := s }
  -- Generous fuel: parsing depth is not proportional to input length in bytes.
  let fuel := (fuelOf st0) * 16 + 32
  match Internal.parseExprFuel fuel st0 with
  | .ok (e, st) =>
    let st' := skipWs st
    if st'.i = st'.s.rawEndPos then .ok e else .error "trailing input"
  | .error msg => .error msg

end NN.Verification.ODE.Parse
