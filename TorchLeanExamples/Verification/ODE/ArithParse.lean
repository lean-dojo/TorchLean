/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.TextCursor

/-!
# Arithmetic expression parser

ODE right-hand sides and PINN residuals are written in one small arithmetic language. This module
parses it into any expression type: a `Grammar` gives the type's constructors, the atom each known
identifier stands for and, when the language has them, division and named unary functions. The
parser is hand-written recursive descent, so the accepted syntax is all in this file.

Grammar (informal):
  expr    := term (('+' | '-') term)*
  term    := unary (('*' | '/') unary)*
  unary   := '-' unary | factor
  factor  := primary ('^' nat)?
  primary := number | ident '(' expr ')' | ident | '(' expr ')'

`/` is an operator only when `Grammar.div?` is given, and `ident '(' expr ')'` a call only when
`Grammar.call?` is. An identifier is an ASCII letter followed by ASCII letters, digits and `_`. A
number is a decimal literal read by `TextCursor.parseFloatBounds` as `Grammar.const lo hi`, with
`lo` and `hi` its binary64 roundings toward $-\infty$ and $+\infty$. `x ^ n` expands into `n - 1`
multiplications (`Grammar.pow`), so the expression grows with `n`, and `x ^ 0` is the literal `1`.
`hasPowAbove` finds an exponent above a bound without parsing, so a caller can refuse such a text.
Powers bind more tightly than unary minus: `-u^2` is `-(u^2)`. Recursion is bounded by a fuel
budget of 16 per byte of the source plus 48.

The theorems about `Grammar.pow` give its expansion. In
`TorchLeanExamples.Verification.ODE.ArithParseLanguage` we prove four facts about `parse`: the fuel
budget always suffices, the printed text of every well-formed `Ast` with a value parses back to that
value, every value `parse` gives is the value of such a tree, and inserting whitespace at either end
or between two characters that are not both letters, digits, `_` or `.` leaves the result unchanged.
-/

@[expose] public section

namespace TorchLean.Examples.ArithParse

open Examples (TextCursor)

/-- The constructors and atoms of an expression type `E` that `parse` builds. -/
structure Grammar (E : Type) where
  /-- `const lo hi`: a real constant known to lie between the binary64 values `lo` and `hi`. -/
  const : Float → Float → E
  /-- Negation `-a`. -/
  neg : E → E
  /-- The sum `a + b`. -/
  add : E → E → E
  /-- The difference `a - b`. -/
  sub : E → E → E
  /-- The product `a * b`. -/
  mul : E → E → E
  /-- The quotient `a / b`, when the language has division. -/
  div? : Option (E → E → E) := none
  /-- The expression a known identifier stands for, `none` for any other identifier. -/
  ident? : String → Option E
  /-- The application of a known unary function to its argument, `none` for any other name, when
  the language has function calls. -/
  call? : Option (String → E → Option E) := none

namespace Grammar

variable {E : Type} (g : Grammar E)

/-- `base ^ n`: the literal `1` for `n = 0`, otherwise `n` copies of `base` multiplied from the
left. -/
def pow (base : E) : Nat → E
  | 0 => g.const 1.0 1.0
  | n + 1 => Nat.repeat (fun acc => g.mul acc base) n base

/-- The zeroth power is the literal `1`. -/
theorem pow_zero (base : E) : g.pow base 0 = g.const 1.0 1.0 := rfl

/-- The first power is `base` itself. -/
theorem pow_one (base : E) : g.pow base 1 = base := rfl

/-- Each higher power multiplies the one below it by `base` on the right. -/
theorem pow_succ_succ (base : E) (n : Nat) :
    g.pow base (n + 2) = g.mul (g.pow base (n + 1)) base := rfl

end Grammar

section

variable {E : Type}

mutual
  /-- `expr`: terms joined by `+` and `-`, with `fuel` bounding the recursion. -/
  def exprFuel (g : Grammar E) (fuel : Nat) (st : TextCursor) : Except String (E × TextCursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (term0, st1) ← termFuel g fuel st
      let rec loop (fuel : Nat) (acc : E) (st : TextCursor) : Except String (E × TextCursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := TextCursor.skipWs st
          match TextCursor.peek st' with
          | some '+' =>
            let (t2, st2) ← termFuel g fuel (TextCursor.bump st')
            loop fuel (g.add acc t2) st2
          | some '-' =>
            let (t2, st2) ← termFuel g fuel (TextCursor.bump st')
            loop fuel (g.sub acc t2) st2
          | _ => .ok (acc, st')
      loop fuel term0 st1

  /-- `term`: unary expressions joined by `*` and, with `Grammar.div?`, `/`. -/
  def termFuel (g : Grammar E) (fuel : Nat) (st : TextCursor) : Except String (E × TextCursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (f, st1) ← unaryFuel g fuel st
      let rec loop (fuel : Nat) (acc : E) (st : TextCursor) : Except String (E × TextCursor) := do
        match fuel with
        | 0 => .ok (acc, st)
        | Nat.succ fuel =>
          let st' := TextCursor.skipWs st
          match TextCursor.peek st', g.div? with
          | some '*', _ =>
            let (f2, st2) ← unaryFuel g fuel (TextCursor.bump st')
            loop fuel (g.mul acc f2) st2
          | some '/', some div =>
            let (f2, st2) ← unaryFuel g fuel (TextCursor.bump st')
            loop fuel (div acc f2) st2
          | _, _ => .ok (acc, st')
      loop fuel f st1

  /-- `factor`: a primary expression with an optional natural power, expanded by `Grammar.pow`. -/
  def factorFuel (g : Grammar E) (fuel : Nat) (st : TextCursor) : Except String
      (E × TextCursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let (base, st1) ← primaryFuel g fuel st
      let st1' := TextCursor.skipWs st1
      match TextCursor.peek st1' with
      | some '^' =>
        let (n, st2) ← TextCursor.parseNat (TextCursor.bump st1')
        .ok (g.pow base n, st2)
      | _ => .ok (base, st1')

  /-- `unary`: leading negations of a factor. -/
  def unaryFuel (g : Grammar E) (fuel : Nat) (st : TextCursor) : Except String
      (E × TextCursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := TextCursor.skipWs st
      match TextCursor.peek st' with
      | some '-' =>
        let (e, st1) ← unaryFuel g fuel (TextCursor.bump st')
        .ok (g.neg e, st1)
      | _ => factorFuel g fuel st'

  /-- `primary`: a parenthesized expression, a number, a call or an identifier. -/
  def primaryFuel (g : Grammar E) (fuel : Nat) (st : TextCursor) : Except String
      (E × TextCursor) := do
    match fuel with
    | 0 => .error "parser: out of fuel"
    | Nat.succ fuel =>
      let st' := TextCursor.skipWs st
      match TextCursor.peek st' with
      | some '(' =>
        let (e, st1) ← exprFuel g fuel (TextCursor.bump st')
        let st2 := TextCursor.skipWs st1
        match TextCursor.peek st2 with
        | some ')' => .ok (e, TextCursor.bump st2)
        | _ => .error "expected ')'"
      | some c =>
        if c.isDigit then
          let ((lo, hi), st1) ← TextCursor.parseFloatBounds st'
          .ok (g.const lo hi, st1)
        else if c.isAlpha then
          let (id, st1) :=
            TextCursor.takeWhile (fun c => c.isAlpha || c.isDigit || c = '_') "" st'
          let st1' := TextCursor.skipWs st1
          match g.call?, TextCursor.peek st1' with
          | some call, some '(' =>
            let (arg, st2) ← exprFuel g fuel (TextCursor.bump st1')
            let st3 := TextCursor.skipWs st2
            match TextCursor.peek st3 with
            | some ')' =>
              match call id arg with
              | some e => .ok (e, TextCursor.bump st3)
              | none => .error s!"unknown function: {id}"
            | _ => .error "expected ')'"
          | _, _ =>
            match g.ident? id with
            | some e => .ok (e, st1')
            | none => .error s!"unknown identifier: {id}"
        else
          .error s!"unexpected char: {c}"
      | none => .error "unexpected end of input"
end

end

variable {E : Type}

/-- Parse all of `source` in the language of `g`, with whitespace allowed between tokens and at
either end. Text left after the expression is an error. -/
def parse (g : Grammar E) (source : String) : Except String E := do
  let st : TextCursor := { source := source }
  let (e, st1) ← exprFuel g (16 * TextCursor.remainingFuel st + 32) st
  match TextCursor.peek (TextCursor.skipWs st1) with
  | none => .ok e
  | some c => .error s!"unexpected trailing input near '{c}'"

/-- Whether `source` has a power `^ n` with `n > maxPow`. The digits after each `^` are read with
their value capped at `maxPow + 1`, so a 256-digit exponent costs no more than a small one. -/
def hasPowAbove (source : String) (maxPow : Nat) : Bool :=
  go source.toList
where
  /-- The scan over the characters left. -/
  go : List Char → Bool
    | [] => false
    | '^' :: rest =>
      let digits := (rest.dropWhile Char.isWhitespace).takeWhile Char.isDigit
      let n := digits.foldl (fun n c => min (maxPow + 1) (10 * n + (c.toNat - '0'.toNat))) 0
      n > maxPow || go rest
    | _ :: rest => go rest

end TorchLean.Examples.ArithParse
