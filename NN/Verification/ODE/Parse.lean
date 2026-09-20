/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.ODE.Ast
public import NN.Verification.Util.TextCursor

/-!
# Parse

Hand-rolled parser for ODE RHS expressions.

Grammar (informal):
  expr   := term (('+' | '-') term)*
  term   := unary (('*' | '/') unary)*
  unary  := '-' unary | factor
  factor := primary ('^' nat)?
  primary:= number | 't' | 'u' | ident '(' expr ')' | ident | '(' expr ')'

Supported unary functions: sin, cos, exp, log.
Supported identifiers: pi.

Exponentiation is expanded into repeated multiplication when `^ n` is given; `x^0` is one.
Exponentiation binds more tightly than unary minus, so `-u^2` means `-(u^2)`.
-/

@[expose] public section


namespace NN.Verification.ODE.Parse

open NN.Verification.ODE
open NN.Verification.Util
open TextCursor (Cursor)

/-!
This parser is part of the executable ODE verifier.  We keep the grammar direct and hand-written so
that the accepted certificate syntax is visible in one file, with predictable errors and no hidden
parser-combinator behavior.

The output AST is `NN.Verification.ODE.Ast.Expr`.
-/

/-! ## Parser state and low-level helpers -/

namespace Internal

/-!
The cursor itself lives in `NN.Verification.Util.TextCursor`, shared with the PINN PDE parser, and
so do the scanning primitives. What remains here is a thin layer whose only job is to supply this
grammar's fuel budget to those primitives, so the rest of the file can call `skipWs` and friends
without repeating `fuelOf st` at every site. The PDE parser keeps the same four names for the same
reason; the budget behind them is what differs.
-/

/--
A fuel budget derived from the remaining input length, which is what guarantees termination.

Unlike the PDE parser this grammar bottoms out within the remaining bytes, so the raw remaining
count is enough and no headroom factor is needed.
-/
@[inline] def fuelOf (st : Cursor) : Nat := TextCursor.remainingFuel st

/-- Skip ASCII whitespace (`' '`, `'\t'`, `'\n'`). -/
def skipWs (st : Cursor) : Cursor :=
  TextCursor.skipWhileFuel TextCursor.isWhitespace (fuelOf st) st

/--
Consume consecutive characters satisfying `p`, accumulating them into `acc`.

Returns the consumed text and the updated parser state.
-/
def takeWhile (p : Char → Bool) (acc : String) (st : Cursor) :
    String × Cursor :=
  TextCursor.takeWhileFuel (fuelOf st) p acc st

/-- Parse a signed decimal number without exponent, e.g. `-12.34`. -/
def parseNumber (st : Cursor) : Except String (Float × Cursor) :=
  TextCursor.parseFloat (fuelOf st) st

/-- Parse a natural number (decimal digits) used for exponents `^ n`. -/
def parseNat (st : Cursor) : Except String (Nat × Cursor) :=
  TextCursor.parseNat (fuelOf st) st

/-- Parse an identifier consisting of letters/digits/underscore. -/
def parseIdent (st : Cursor) : Except String (String × Cursor) := do
  let st0 := skipWs st
  let (txt, st1) := takeWhile (fun c => c.isAlpha || c.isDigit || c = '_' ) "" st0
  if txt = "" then .error "expected identifier" else .ok (txt, st1)

/-! ## Built-in constants and unary functions -/

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

/-! ## Recursive descent: expression/term/factor/unary/primary -/

mutual
  /--
  Internal: parse an `expr` (addition/subtraction chain), with an explicit fuel budget.

  This is a plain `def` (not `private`) because the module is in a `public` section for
  export/doc tooling, and public declarations should not depend on private helper definitions.
  -/
  def parseExprFuel (fuel : Nat) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (term0, st1) ← parseTermFuel fuel st
      let rec loop (fuel : Nat) (acc : Expr) (st : Cursor) :
          Except String (Expr × Cursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match TextCursor.peek st' with
          | some '+' =>
            let (t2, st2) ← parseTermFuel fuel (TextCursor.bump st')
            loop fuel (.add acc t2) st2
          | some '-' =>
            let (t2, st2) ← parseTermFuel fuel (TextCursor.bump st')
            loop fuel (.sub acc t2) st2
          | _ => .ok (acc, st')
      loop fuel term0 st1

  /-- Parse a `term` (multiplication/division chain), with an explicit fuel budget. -/
  def parseTermFuel (fuel : Nat) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (f, st1) ← parseUnaryFuel fuel st
      let rec loop (fuel : Nat) (acc : Expr) (st : Cursor) :
          Except String (Expr × Cursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match TextCursor.peek st' with
          | some '*' =>
            let (f2, st2) ← parseUnaryFuel fuel (TextCursor.bump st')
            loop fuel (.mul acc f2) st2
          | some '/' =>
            let (f2, st2) ← parseUnaryFuel fuel (TextCursor.bump st')
            loop fuel (.div acc f2) st2
          | _ => .ok (acc, st')
      loop fuel f st1

  /-- Parse a primary expression with optional natural-number exponentiation. -/
  def parseFactorFuel (fuel : Nat) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (base, st1) ← parsePrimaryFuel fuel st
      let st1' := skipWs st1
      match TextCursor.peek st1' with
      | some '^' =>
        let (n, st2) ← parseNat (TextCursor.bump st1')
        if n = 0 then
          .ok (.const 1.0, st2)
        else if n = 1 then
          .ok (base, st2)
        else
          let rec powMul (base : Expr) (k : Nat) (acc : Expr) : Expr :=
            match k with
            | 0 => acc
            | Nat.succ m => powMul base m (.mul acc base)
          .ok (powMul base (n - 1) base, st2)
      | _ => .ok (base, st1')

  /-- Parse a `unary` (leading negations), with an explicit fuel budget. -/
  def parseUnaryFuel (fuel : Nat) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match TextCursor.peek st' with
      | some '-' =>
        let (e, st1) ← parseUnaryFuel fuel (TextCursor.bump st')
        .ok (.neg e, st1)
      | _ =>
        parseFactorFuel fuel st'

  /-- Parse a `primary` atom (number/variable/function-call/parentheses), with an explicit fuel
    budget. -/
  def parsePrimaryFuel (fuel : Nat) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match TextCursor.peek st' with
      | some '(' =>
        let (e, st1) ← parseExprFuel fuel (TextCursor.bump st')
        let st2 := skipWs st1
        match TextCursor.peek st2 with
        | some ')' => .ok (e, TextCursor.bump st2)
        | _ => .error "expected ')'"
      | some c =>
        if c.isDigit || c = '.' then
          let (v, st1) ← parseNumber st'
          .ok (.const v, st1)
        else if c = 't' then
          .ok (.t, TextCursor.bump st')
        else if c = 'u' then
          .ok (.u, TextCursor.bump st')
        else if c.isAlpha || c = 'π' then
          let (id, st1) ← parseIdent st'
          let st1' := skipWs st1
          match TextCursor.peek st1' with
          | some '(' =>
            let (arg, st2) ← parseExprFuel fuel (TextCursor.bump st1')
            let st3 := skipWs st2
            match TextCursor.peek st3 with
            | some ')' =>
              match applyFunc? id arg with
              | some app => .ok (app, TextCursor.bump st3)
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
  let st0 : Cursor := { source := s }
  -- Generous fuel: parsing depth is not proportional to input length in bytes.
  let fuel := (Internal.fuelOf st0) * 16 + 32
  match Internal.parseExprFuel fuel st0 with
  | .ok (e, st) =>
    let st' := Internal.skipWs st
    if st'.position = st'.source.rawEndPos then .ok e else .error "trailing input"
  | .error msg => .error msg

end NN.Verification.ODE.Parse
