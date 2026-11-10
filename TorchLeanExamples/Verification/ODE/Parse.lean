/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.ODE.AST
public import TorchLeanExamples.Verification.ODE.ArithParse

/-!
# Parse

`parseExpr` reads ODE right-hand sides into `TorchLean.Examples.ODE.Expr` with the arithmetic parser
of `TorchLean.Examples.ArithParse`.
-/

@[expose] public section

namespace TorchLean.Examples.ODE

/-- The node count of `e` as a tree plus `n`, or a number above `cap` once the count passes `cap`.
From there the walk skips every right operand, so it visits at most `cap` nodes and one path of
`e`, even when the powers of `e` share one subtree many times. -/
def Expr.sizeUpTo (cap : Nat) : Expr → Nat → Nat
  | .add a b, n | .sub a b, n | .mul a b, n | .div a b, n =>
    let n := a.sizeUpTo cap (n + 1)
    if n > cap then n else b.sizeUpTo cap n
  | .neg a, n | .sin a, n | .cos a, n | .exp a, n | .log a, n => a.sizeUpTo cap (n + 1)
  | _, n => n + 1

/-- Parse an ODE right-hand side such as `"sin(t) + u^2"` into an `Expr`: the arithmetic of
`ArithParse.parse` with the atoms `t`, `u` and `pi` (the binary64 words `0x400921FB54442D18`
below $\pi$ and `0x400921FB54442D19` above), division, and the functions `sin`, `cos`, `exp` and
`log`. A text with an exponent above 512, deeper than `Expr.eval` evaluates, is an error before
parsing, and a parsed expression of more than 100000 nodes once its powers are expanded is an
error too. -/
def parseExpr (s : String) : Except String Expr := do
  if ArithParse.hasPowAbove s 512 then
    throw "parser: exponent above 512, deeper than the evaluator accepts"
  let e ← ArithParse.parse
    { const := Expr.const, neg := Expr.neg, add := Expr.add, sub := Expr.sub, mul := Expr.mul
      div? := some Expr.div
      ident? := fun
        | "t" => some Expr.t
        | "u" => some Expr.u
        | "pi" =>
          some (Expr.const (Float.ofBits 0x400921FB54442D18) (Float.ofBits 0x400921FB54442D19))
        | _ => none
      call? := some fun
        | "sin", arg => some (Expr.sin arg)
        | "cos", arg => some (Expr.cos arg)
        | "exp", arg => some (Expr.exp arg)
        | "log", arg => some (Expr.log arg)
        | _, _ => none }
    s
  if e.sizeUpTo 100000 0 > 100000 then
    throw "parser: more than 100000 nodes once powers are expanded"
  pure e

end TorchLean.Examples.ODE
