/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.PdeAst
public import NN.Verification.Util.TextCursor

/-!
# PdeParse

A compact hand-rolled parser from strings to PDE AST (`Expr`).

Grammar (informal):
  expr   := term (('+' | '-') term)*
  term   := unary ('*' unary)*
  unary  := '-' unary | factor
  factor := primary ('^' nat)?
  primary:= derivative | number | ident | '(' expr ')'

Derivative names accept both compact and subscript-style spellings:
`u`, `ux`, `uy`, `ut`, `uxx`, `uyy`, `utt`, `u_x`, `u_y`, `u_t`, `u_xx`, `u_yy`, and `u_tt`.
For 1D-in-time PINN examples, the parser treats `t` as the second axis, so `u_t` is the same
primitive as `u_y`.

Numbers are parsed as Floats. Idents look up a value from `env : String → Option Float`.
Unsupported tokens produce an error.

Natural powers use the usual identities ($x^0=1$), and powers bind more tightly than unary
minus, so `-u^2` means `-(u^2)`.

Implementation note:
The parser is total by threading a simple `fuel : Nat` through the recursive descent; `fuel` is
initialized from the remaining bytes in the input and decreases on every recursive descent step.

References:
- PINNs (motivation for residual expressions): `https://arxiv.org/abs/1711.10561`
-/

@[expose] public section


namespace NN.Verification.PINN.PdeParse

open NN.Verification.PINN.PdeAst
open NN.Verification.Util
open TextCursor (Cursor)

namespace Internal

/-!
`NN.Verification.Util.TextCursor` is the shared byte-position cursor, and this parser and the ODE
parser both scan with it. The four forwarders below read exactly like the ODE parser's, but each one
closes over `fuelOf`, and the fuel policy is where the two grammars part company. Binding it once
here is what keeps the recursive descent underneath free of budget arithmetic.
-/

/--
Recursion budget with additional headroom for the mutually recursive PDE grammar.

This is the one place where the two hand-written parsers genuinely differ, so it is worth stating
why. `fuel` is a recursion budget, not a token count. Even a one-character input like `u` walks
`expr → term → unary → factor → primary`, so a budget equal to the remaining bytes would run out
before the grammar bottoms out. Scaling by 8 and adding 16 covers the deepest descent that any
accepted input can force.
-/
@[inline] def fuelOf (st : Cursor) : Nat :=
  16 + 8 * TextCursor.remainingFuel st

/-- Skip whitespace from the current parser state. -/
def skipWs (st : Cursor) : Cursor :=
  TextCursor.skipWhileFuel TextCursor.isWhitespace (fuelOf st) st

/-- Consume characters satisfying `p`, accumulating into `acc`. -/
def takeWhile (p : Char → Bool) (acc : String) (st : Cursor) :
    String × Cursor :=
  TextCursor.takeWhileFuel (fuelOf st) p acc st

/-- Parse a signed decimal number without exponent, e.g. `-12.34`. -/
def parseNumber (st : Cursor) : Except String (Float × Cursor) :=
  TextCursor.parseFloat (fuelOf st) st

/-- Parse a natural number at the current parser state. -/
def parseNat (st : Cursor) : Except String (Nat × Cursor) :=
  TextCursor.parseNat (fuelOf st) st

/-- Parse an identifier used for environment lookup. -/
def parseIdent (st : Cursor) : Except String (String × Cursor) := do
  let (txt, st1) := takeWhile (fun c => c.isAlpha || c.isDigit || c = '_' ) "" st
  if txt = "" then .error "expected identifier" else .ok (txt, st1)

/-- Parse the built-in PDE primitive names before falling back to external constants. -/
def builtinIdent? : String → Option Expr
  | "u" => some .u
  | "ux" | "u_x" => some (.du .X)
  | "uy" | "ut" | "u_y" | "u_t" => some (.du .Y)
  | "uxx" | "u_xx" => some (.d2u .X)
  | "uyy" | "utt" | "u_yy" | "u_tt" => some (.d2u .Y)
  | _ => none

mutual
  /-- Parse an additive/subtractive expression with an explicit recursion budget. -/
  def parseExprCoreFuel (fuel : Nat) (env : String → Option Float) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (t, st1) ← parseTermFuel fuel env st
      let rec loop (fuel : Nat) (acc : Expr) (st : Cursor) :
          Except String (Expr × Cursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match TextCursor.peek st' with
          | some '+' =>
            let st'' := TextCursor.bump st'
            let (t2, st3) ← parseTermFuel fuel env st''
            loop fuel (.add acc t2) st3
          | some '-' =>
            let st'' := TextCursor.bump st'
            let (t2, st3) ← parseTermFuel fuel env st''
            loop fuel (.sub acc t2) st3
          | _ => .ok (acc, st')
      loop fuel t st1

  /-- Parse a multiplicative term with an explicit recursion budget. -/
  def parseTermFuel (fuel : Nat) (env : String → Option Float) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (f, st1) ← parseUnaryFuel fuel env st
      let rec loop (fuel : Nat) (acc : Expr) (st : Cursor) :
          Except String (Expr × Cursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match TextCursor.peek st' with
          | some '*' =>
            let st'' := TextCursor.bump st'
            let (f2, st3) ← parseUnaryFuel fuel env st''
            loop fuel (.mul acc f2) st3
          | _ => .ok (acc, st')
      loop fuel f st1

  /-- Parse a primary expression plus an optional natural-number power, with $x^0=1$. -/
  def parseFactorFuel (fuel : Nat) (env : String → Option Float) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (p, st1) ← parsePrimaryFuel fuel env st
      let st1' := skipWs st1
      match TextCursor.peek st1' with
      | some '^' =>
        let st2 := TextCursor.bump st1'
        let (n, st3) ← parseNat (skipWs st2)
        if n = 0 then
          .ok (.const 1.0, st3)
        else if n = 1 then
          .ok (p, st3)
        else
          -- expand p^n as repeated multiplication
          let rec powMul (base : Expr) (k : Nat) (acc : Expr) : Expr :=
            match k with
            | 0 => acc
            | Nat.succ m => powMul base m (.mul acc base)
          .ok (powMul p (n - 1) p, st3)
      | _ => .ok (p, st1')

  /-- Parse leading negations. Exponentiation binds more tightly than unary minus. -/
  def parseUnaryFuel (fuel : Nat) (env : String → Option Float) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match TextCursor.peek st' with
      | some '-' =>
        let (e, st1) ← parseUnaryFuel fuel env (TextCursor.bump st')
        .ok (.neg e, st1)
      | _ => parseFactorFuel fuel env st'

  /-- Parse atoms: parenthesized expressions, `u`/derivative names, numerals, or environment
  identifiers. -/
  def parsePrimaryFuel (fuel : Nat) (env : String → Option Float) (st : Cursor) :
      Except String (Expr × Cursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match TextCursor.peek st' with
      | some '(' =>
        let st1 := TextCursor.bump st'
        let (e, st2) ← parseExprCoreFuel fuel env st1
        let st3 := skipWs st2
        match TextCursor.peek st3 with
        | some ')' => .ok (e, TextCursor.bump st3)
        | _ => .error "expected ')'"
      | some c =>
        if c.isDigit || c = '.' then
          let (v, st2) ← parseNumber st'
          .ok (.const v, st2)
        else if c.isAlpha then
          let (id, st2) ← parseIdent st'
          match builtinIdent? id with
          | some e => .ok (e, st2)
          | none =>
              match env id with
              | some v => .ok (.const v, st2)
              | none => .error s!"unknown identifier: {id}"
        else
          .error s!"unexpected char: {c}"
      | none => .error "unexpected end of input"
end

/-- Parse a full expression from the current parser state. -/
def parseExprCore (env : String → Option Float) (st : Cursor) : Except String (Expr × Cursor) :=
  parseExprCoreFuel (fuelOf st) env st

end Internal

/-- Entry point: parse a string to Expr using `env` for identifiers. -/
def parseExpr (env : String → Option Float) (s : String) : Except String Expr :=
  match Internal.parseExprCore env { source := s } with
  | .ok (e, st) =>
      let st := Internal.skipWs st
      if TextCursor.atEnd st then
        .ok e
      else
        match TextCursor.peek st with
        | some c => .error s!"unexpected trailing input near '{c}'"
        | none => .error "unexpected trailing input"
  | .error msg => .error msg

end NN.Verification.PINN.PdeParse
