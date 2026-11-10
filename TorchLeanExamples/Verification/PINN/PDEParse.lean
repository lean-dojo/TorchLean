/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLeanExamples.Verification.PINN.PDEAST
public import TorchLeanExamples.Verification.ODE.ArithParse

/-!
# PdeParse

The PINN residual language: the arithmetic of `TorchLean.Examples.ArithParse` over the PDE AST
(`Expr`), without division or function calls.

Derivative names accept both compact and subscript-style spellings:
`u`, `ux`, `uy`, `ut`, `uxx`, `uyy`, `utt`, `u_x`, `u_y`, `u_t`, `u_xx`, `u_yy`, and `u_tt`.
For 1D-in-time PINN examples, the parser treats `t` as the second axis, so `u_t` is the same
primitive as `u_y`. Any other identifier takes its value `v` from `env : String → Option Float`
and becomes `Expr.const v v`.

References:
- PINNs (motivation for residual expressions): `https://arxiv.org/abs/1711.10561`
-/

@[expose] public section


namespace TorchLean.Examples.PINN


/-- The PDE primitive named by a built-in identifier, consulted before the constants of `env`. -/
def builtinIdent? : String → Option Expr
  | "u" => some .u
  | "ux" | "u_x" => some (.du .x)
  | "uy" | "ut" | "u_y" | "u_t" => some (.du .y)
  | "uxx" | "u_xx" => some (.d2u .x)
  | "uyy" | "utt" | "u_yy" | "u_tt" => some (.d2u .y)
  | _ => none

/-- The node count of `e` as a tree plus `n`, or a number above `cap` once the count passes `cap`.
From there the walk skips every right operand, so it visits at most `cap` nodes and one path of
`e`, even when the powers of `e` share one subtree many times. -/
def Expr.sizeUpTo (cap : Nat) : Expr → Nat → Nat
  | .add a b, n | .sub a b, n | .mul a b, n =>
    let n := a.sizeUpTo cap (n + 1)
    if n > cap then n else b.sizeUpTo cap n
  | .scale _ a, n | .neg a, n => a.sizeUpTo cap (n + 1)
  | _, n => n + 1

/-- Parse a residual string such as `"uxx + uyy"` into an `Expr`: the arithmetic of
`ArithParse.parse` with the primitives of `builtinIdent?`, and the constant `env id` for
any other identifier `id`. A text with an exponent above 256, deeper than `Expr.eval` evaluates,
is an error before parsing, and a parsed expression of more than 100000 nodes once its powers are
expanded is an error too. -/
def parseExpr (env : String → Option Float) (s : String) : Except String Expr := do
  if ArithParse.hasPowAbove s 256 then
    throw "parser: exponent above 256, deeper than the evaluator accepts"
  let e ← ArithParse.parse
    { const := Expr.const, neg := Expr.neg, add := Expr.add, sub := Expr.sub, mul := Expr.mul
      ident? := fun id => builtinIdent? id <|> (env id).map fun v => Expr.const v v }
    s
  if e.sizeUpTo 100000 0 > 100000 then
    throw "parser: more than 100000 nodes once powers are expanded"
  pure e

end TorchLean.Examples.PINN
