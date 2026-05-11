/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.PINN.PdeAst

/-!
# PdeParse

A compact hand-rolled parser from strings to PDE AST (`Expr`).

Grammar (informal):
  expr   := term (('+' | '-') term)*
  term   := factor ('*' factor)*
  factor := primary ('^' int)?
  primary:= 'u' | 'ux' | 'uy' | 'uxx' | 'uyy' | number | ident | '(' expr ')'

Numbers are parsed as Floats. Idents look up a value from `env : String → Option Float`.
Unsupported tokens produce an error.

Implementation note:
The parser is total by threading a simple `fuel : Nat` through the recursive descent; `fuel` is
initialized from the remaining bytes in the input and decreases on every recursive descent step.

References:
- PINNs (motivation for residual expressions): `https://arxiv.org/abs/1711.10561`
-/

@[expose] public section


namespace NN.Verification.PINN.PdeParse

open NN.Verification.PINN.PdeAst

structure State where
  /-- s. -/
  s : String
  /-- i. -/
  i : String.Pos.Raw := 0

@[inline] def eof (st : State) : Bool := st.i ≥ st.s.rawEndPos

@[inline] def peek (st : State) : Option Char := String.Pos.Raw.get? st.s st.i

@[inline] def bump (st : State) : State := { st with i := String.Pos.Raw.next st.s st.i }

@[inline] def fuelOf (st : State) : Nat :=
  -- `fuel` is a recursion budget (not a token count). Even very small inputs like "u"
  -- require several mutually-recursive descent steps, so we scale the remaining-byte
  -- budget by a small constant and add a fixed headroom.
  let remaining := (st.s.rawEndPos.byteIdx - st.i.byteIdx) + 1
  16 + 8 * remaining

def isWs (c : Char) : Bool :=
  c = ' ' || c = '\t' || c = '\n'

def skipWsFuel : Nat → State → State
  | 0, st => st
  | Nat.succ fuel, st =>
    match peek st with
    | some c =>
      if isWs c then
        skipWsFuel fuel (bump st)
      else
        st
    | none => st

def skipWs (st : State) : State :=
  skipWsFuel (fuelOf st) st

def takeWhileFuel (fuel : Nat) (p : Char → Bool) (acc : String) (st : State) : String × State :=
  match fuel with
  | 0 => (acc, st)
  | Nat.succ fuel =>
    match peek st with
    | some c =>
      if p c then
        takeWhileFuel fuel p (acc.push c) (bump st)
      else
        (acc, st)
    | none => (acc, st)

def takeWhile (p : Char → Bool) (acc : String) (st : State) : String × State :=
  takeWhileFuel (fuelOf st) p acc st

/-- Parse a signed decimal number without exponent, e.g. `-12.34`. -/
def parseNumber (st : State) : Except String (Float × State) := do
  let st0 := skipWs st
  -- sign
  let (sgn, st1) :=
    match peek st0 with
    | some '-' => (-1.0, bump st0)
    | _ => (1.0, st0)
  -- integer part (at least one digit)
  let (intTxt, st2) := takeWhile (fun c => c.isDigit) "" st1
  if intTxt = "" then .error "expected number"
  let intVal : Float :=
    Float.ofNat (intTxt.toList.foldl (fun (acc : Nat) (c : Char) => acc * 10 + (c.toNat -
      '0'.toNat)) 0)
  -- optional fractional part
  let (fracVal, st3) :=
    match peek st2 with
    | some '.' =>
      let st2' := bump st2
      let (fracTxt, st2'') := takeWhile (fun c => c.isDigit) "" st2'
      if fracTxt = "" then (0.0, st2'')
      else
        let num : Nat := fracTxt.toList.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0
        let den : Nat := (Nat.pow 10 fracTxt.length)
        let fv : Float := (Float.ofNat num) / (Float.ofNat den)
        (fv, st2'')
    | _ => (0.0, st2)
  .ok (sgn * (intVal + fracVal), st3)

def parseNat (st : State) : Except String (Nat × State) := do
  let st0 := skipWs st
  let (txt, st1) := takeWhile (fun c => c.isDigit) "" st0
  if txt = "" then .error "expected natural number"
  let n : Nat := txt.toList.foldl (fun acc c => acc * 10 + (c.toNat - '0'.toNat)) 0
  .ok (n, st1)

def parseIdent (st : State) : Except String (String × State) := do
  let (txt, st1) := takeWhile (fun c => c.isAlpha || c.isDigit || c = '_' ) "" st
  if txt = "" then .error "expected identifier" else .ok (txt, st1)

mutual
  def parseExprCoreFuel (fuel : Nat) (env : String → Option Float) (st : State) : Except String
    (Expr × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (t, st1) ← parseTermFuel fuel env st
      let rec loop (fuel : Nat) (acc : Expr) (st : State) : Except String (Expr × State) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match peek st' with
          | some '+' =>
            let st'' := bump st'
            let (t2, st3) ← parseTermFuel fuel env st''
            loop fuel (.add acc t2) st3
          | some '-' =>
            let st'' := bump st'
            let (t2, st3) ← parseTermFuel fuel env st''
            loop fuel (.sub acc t2) st3
          | _ => .ok (acc, st')
      loop fuel t st1

  def parseTermFuel (fuel : Nat) (env : String → Option Float) (st : State) : Except String (Expr ×
    State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (f, st1) ← parseFactorFuel fuel env st
      let rec loop (fuel : Nat) (acc : Expr) (st : State) : Except String (Expr × State) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := skipWs st
          match peek st' with
          | some '*' =>
            let st'' := bump st'
            let (f2, st3) ← parseFactorFuel fuel env st''
            loop fuel (.mul acc f2) st3
          | _ => .ok (acc, st')
      loop fuel f st1

  def parseFactorFuel (fuel : Nat) (env : String → Option Float) (st : State) : Except String (Expr
    × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (p, st1) ← parsePrimaryFuel fuel env st
      let st1' := skipWs st1
      match peek st1' with
      | some '^' =>
        let st2 := bump st1'
        let (n, st3) ← parseNat (skipWs st2)
        if n ≤ 1 then
          .ok (p, st3)
        else
          -- expand p^n as repeated multiplication
          let rec powMul (base : Expr) (k : Nat) (acc : Expr) : Expr :=
            match k with
            | 0 => acc
            | Nat.succ m => powMul base m (.mul acc base)
          .ok (powMul p (n - 1) p, st3)
      | _ => .ok (p, st1')

  def parsePrimaryFuel (fuel : Nat) (env : String → Option Float) (st : State) : Except String (Expr
    × State) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := skipWs st
      match peek st' with
      | some '(' =>
        let st1 := bump st'
        let (e, st2) ← parseExprCoreFuel fuel env st1
        let st3 := skipWs st2
        match peek st3 with
        | some ')' => .ok (e, bump st3)
        | _ => .error "expected ')'"
      | some c =>
        if c = 'u' then
          let st1 := bump st'
          -- accept u / ux / uy / uxx / uyy
          match peek st1 with
          | some 'x' =>
            let st2 := bump st1
            match peek st2 with
            | some 'x' => .ok (.d2u .X, bump st2)
            | _ => .ok (.du .X, st2)
          | some 'y' =>
            let st2 := bump st1
            match peek st2 with
            | some 'y' => .ok (.d2u .Y, bump st2)
            | _ => .ok (.du .Y, st2)
          | _ => .ok (.u, st1)
        else if c.isDigit || c = '.' || c = '-' then
          let (v, st2) ← parseNumber st'
          .ok (.const v, st2)
        else if c.isAlpha then
          let (id, st2) ← parseIdent st'
          match env id with
          | some v => .ok (.const v, st2)
          | none => .error s!"unknown identifier: {id}"
        else
          .error s!"unexpected char: {c}"
      | none => .error "unexpected end of input"
end

def parseExprCore (env : String → Option Float) (st : State) : Except String (Expr × State) :=
  parseExprCoreFuel (fuelOf st) env st

/-- Entry point: parse a string to Expr using `env` for identifiers. -/
def parseExpr (env : String → Option Float) (s : String) : Except String Expr :=
  match parseExprCore env { s := s } with
  | .ok (e, _) => .ok e
  | .error msg => .error msg

end NN.Verification.PINN.PdeParse
