/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.IEEE32.Arithmetic

/-!
# Expression Refinement

Compositional refinement lemmas on top of `NN/Proofs/RuntimeApprox/IEEE32/Arithmetic.lean`.

The arithmetic adapter transfers FloatLib’s refinement theorems one operation at a time.
The expression theorem composes those results along the scalar expression tree.

This file provides:

- a compact scalar expression language `Expr`, and
- an “all intermediates are finite” witness `FiniteEval`,

so we can state and prove a single theorem that looks like:

$$
\operatorname{toReal}(\operatorname{evalRuntime}(\mathrm{env},e))
=\operatorname{evalSpec}(\operatorname{toReal}\circ\mathrm{env},e).
$$

where:
- `evalRuntime` evaluates the AST using the executable IEEE-754 kernel `ExecFloat.Binary 8 23`, and
- `evalSpec` evaluates the AST in `ℝ`, rounding after every operation using `fp32Round`.

This mirrors the standard mathematical model of float32 evaluation: compute the real operation, then
round-to-float32 at each node, provided we stay on the finite path (no NaNs/Infs, no division by
zero, no overflow).

References / background (for the rounding model itself, not this AST wrapper):
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat.Binary (isFinite toModel)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-!
## A small scalar expression language

`Expr` is a small, scalar-only AST. TorchLean's main IRs live elsewhere; this wrapper exists to
state expression-level refinement theorems for straight-line float32 computations.
-/

/-- A compact AST for scalar float32 expressions evaluated using `ExecFloat.Binary 8 23`. -/
inductive Expr where
  | var : Nat → Expr
  | const : ExecFloat.Binary 8 23 → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | fma : Expr → Expr → Expr → Expr
  | sqrt : Expr → Expr
  deriving Repr

/-- Evaluate an `Expr` using the executable float32 kernel (`ExecFloat.Binary 8 23`). -/
def evalRuntime (env : Nat → (ExecFloat.Binary 8 23)) : Expr → (ExecFloat.Binary 8 23)
  | .var i => env i
  | .const x => x
  | .add a b => ExecFloat.add (evalRuntime env a) (evalRuntime env b)
  | .sub a b => ExecFloat.sub (evalRuntime env a) (evalRuntime env b)
  | .mul a b => ExecFloat.mul (evalRuntime env a) (evalRuntime env b)
  | .div a b => ExecFloat.div (evalRuntime env a) (evalRuntime env b)
  | .fma a b c => (ExecFloat.Binary.fma (rounding := .nearestEven)) (evalRuntime env a) (evalRuntime
    env b) (evalRuntime env c)
  | .sqrt a => (ExecFloat.Binary.sqrt (rounding := .nearestEven)) (evalRuntime env a)

/-- Real semantics for the compact scalar expression language. -/
def evalSpec (env : Nat → ℝ) : Expr → ℝ
  | .var i => env i
  | .const x => (toModel x).toReal
  | .add a b => fp32Round (evalSpec env a + evalSpec env b)
  | .sub a b => fp32Round (evalSpec env a - evalSpec env b)
  | .mul a b => fp32Round (evalSpec env a * evalSpec env b)
  | .div a b => fp32Round (evalSpec env a / evalSpec env b)
  | .fma a b c => fp32Round (evalSpec env a * evalSpec env b + evalSpec env c)
  | .sqrt a => fp32Round (Real.sqrt (evalSpec env a))

/-!
## "Finite evaluation" witnesses (finite at every intermediate node)

The arithmetic adapter bridges finite `ExecFloat.Binary 8 23` behavior to binary32 rounding on the
reals.
For expression-level statements, every intermediate evaluation must remain finite.

`FiniteEval env e d` is a proof object that says:
- evaluating `e` under `env` is finite, and
- its decoded dyadic value is `d`.

We store the dyadic because it gives us a convenient “finiteness certificate”:
`toDyadic? x = some d` immediately rules out NaN/Inf and unlocks the op-level bridge lemmas.
-/

/-- Finite-evaluation witness for the compact scalar expression language. -/
inductive FiniteEval (env : Nat → (ExecFloat.Binary 8 23)) : Expr → FloatLib.Numerics.Dyadic → Prop
  where
  | var (i : Nat) (d : FloatLib.Numerics.Dyadic) (h : (toModel (env i)).toDyadic? = some d) :
      FiniteEval env (.var i) d
  | const (x : ExecFloat.Binary 8 23) (d : FloatLib.Numerics.Dyadic) (h : (toModel x).toDyadic? =
    some d) :
      FiniteEval env (.const x) d
  | add {a b : Expr} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.add (evalRuntime env a) (evalRuntime env b))).toDyadic? = some
        dout) :
      FiniteEval env (.add a b) dout
  | sub {a b : Expr} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.sub (evalRuntime env a) (evalRuntime env b))).toDyadic? = some
        dout) :
      FiniteEval env (.sub a b) dout
  | mul {a b : Expr} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : (toModel (ExecFloat.mul (evalRuntime env a) (evalRuntime env b))).toDyadic? = some
        dout) :
      FiniteEval env (.mul a b) dout
  | div {a b : Expr} {da db dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hden : db.significand ≠ 0)
      (hout : (toModel (ExecFloat.div (evalRuntime env a) (evalRuntime env b))).toDyadic? = some
        dout) :
      FiniteEval env (.div a b) dout
  | fma {a b c : Expr} {da db dc dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db) (hc : FiniteEval env c dc)
      (hout : (toModel ((ExecFloat.Binary.fma (rounding := .nearestEven)) (evalRuntime env a)
        (evalRuntime env b) (evalRuntime env c))).toDyadic?
        = some dout) :
      FiniteEval env (.fma a b c) dout
  | sqrt {a : Expr} {da dout : FloatLib.Numerics.Dyadic}
      (ha : FiniteEval env a da)
      (hout : (toModel ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) (evalRuntime env
        a))).toDyadic? = some dout) :
      FiniteEval env (.sqrt a) dout

namespace FiniteEval

/-- Extract the decoded dyadic of the runtime evaluation from a `FiniteEval` witness. -/
theorem toDyadic? {env : Nat → (ExecFloat.Binary 8 23)} {e : Expr} {d : FloatLib.Numerics.Dyadic} :
    FiniteEval env e d → (toModel (evalRuntime env e)).toDyadic? = some d := by
  intro h
  cases h with
  | var i d h =>
      simpa [evalRuntime] using h
  | const x d h =>
      simpa [evalRuntime] using h
  | add ha hb hout =>
      simpa [evalRuntime] using hout
  | sub ha hb hout =>
      simpa [evalRuntime] using hout
  | mul ha hb hout =>
      simpa [evalRuntime] using hout
  | div ha hb hden hout =>
      simpa [evalRuntime] using hout
  | fma ha hb hc hout =>
      simpa [evalRuntime] using hout
  | sqrt ha hout =>
      simpa [evalRuntime] using hout

end FiniteEval

/-!
## Whole-expression refinement

This is the main result of the file: a compositional refinement theorem that follows the
AST shape and invokes the corresponding op-level bridge theorem at each node.

If you are thinking in PyTorch terms: `Expr` is a compact “forward graph”. Its executable float32
evaluation agrees with the standard float32 mathematical model (real arithmetic plus rounding)
on the finite path.
-/

/-- Main expression-level refinement theorem for IEEEExec. -/
theorem toReal_evalRuntime_eq_evalSpec (env : Nat → (ExecFloat.Binary 8 23)) :
    ∀ {e : Expr} {d : FloatLib.Numerics.Dyadic}, FiniteEval env e d →
      (toModel (evalRuntime env e)).toReal = evalSpec (fun i => (toModel (env i)).toReal) e := by
  intro e d h
  let envS : Nat → ℝ := fun i => (toModel (env i)).toReal
  induction h with
  | var i d h =>
      simp [evalRuntime, evalSpec]
  | const x d h =>
      simp [evalRuntime, evalSpec]
  | add ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hfin : isFinite (ExecFloat.add xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := ExecFloat.add xa xb) (d := dout) hout
      have href :
          (toModel (ExecFloat.add xa xb)).toReal = fp32Round ((toModel xa).toReal +
            (toModel xb).toReal) :=
        IEEE32Exec.toReal_add_eq_fp32Round_of_isFinite hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | sub ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : (toModel xa).toDyadic? = some da := FiniteEval.toDyadic? ha
      have hxb : (toModel xb).toDyadic? = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (ExecFloat.sub xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := ExecFloat.sub xa xb) (d := dout) hout
      have href :
          (toModel (ExecFloat.sub xa xb)).toReal = fp32Round ((toModel xa).toReal -
            (toModel xb).toReal) :=
        IEEE32Exec.toReal_sub_eq_fp32Round_of_isFinite
          (isFinite_eq_true_of_toDyadic?_some hxa) (isFinite_eq_true_of_toDyadic?_some hxb) hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | mul ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hfin : isFinite (ExecFloat.mul xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := ExecFloat.mul xa xb) (d := dout) hout
      have href :
          (toModel (ExecFloat.mul xa xb)).toReal = fp32Round ((toModel xa).toReal *
            (toModel xb).toReal) :=
        IEEE32Exec.toReal_mul_eq_fp32Round_of_isFinite hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | div ha hb hden hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : (toModel xa).toDyadic? = some da := FiniteEval.toDyadic? ha
      have hxb : (toModel xb).toDyadic? = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (ExecFloat.div xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := ExecFloat.div xa xb) (d := dout) hout
      have href :
          (toModel (ExecFloat.div xa xb)).toReal = fp32Round ((toModel xa).toReal /
            (toModel xb).toReal) :=
        IEEE32Exec.toReal_div_eq_fp32Round (x := xa) (y := xb) (dx := da) (dy := db) hxa hxb hden
          hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | fma ha hb hc hout iha ihb ihc =>
      rename_i a b c da db dc dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      let xc := evalRuntime env c
      have hxa : (toModel xa).toDyadic? = some da := FiniteEval.toDyadic? ha
      have hxb : (toModel xb).toDyadic? = some db := FiniteEval.toDyadic? hb
      have hxc : (toModel xc).toDyadic? = some dc := FiniteEval.toDyadic? hc
      have hfin : isFinite ((ExecFloat.Binary.fma (rounding := .nearestEven)) xa xb xc) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := (ExecFloat.Binary.fma (rounding := .nearestEven))
          xa xb xc) (d := dout) hout
      have href :
          (toModel ((ExecFloat.Binary.fma (rounding := .nearestEven)) xa xb xc)).toReal =
            fp32Round ((toModel xa).toReal * (toModel xb).toReal + (toModel xc).toReal) :=
        IEEE32Exec.toReal_fma_eq_fp32Round (x := xa) (y := xb) (z := xc) (dx := da) (dy := db)
          (dz := dc) hxa hxb hxc hfin
      simpa [envS, xa, xb, xc, evalRuntime, evalSpec, iha, ihb, ihc] using href
  | sqrt ha hout iha =>
      rename_i a da dout
      let xa := evalRuntime env a
      have hxa : (toModel xa).toDyadic? = some da := FiniteEval.toDyadic? ha
      have hfin : isFinite ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) xa) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := (ExecFloat.Binary.sqrt (rounding := .nearestEven))
          xa) (d := dout) hout
      have href :
          (toModel ((ExecFloat.Binary.sqrt (rounding := .nearestEven)) xa)).toReal = fp32Round
            (Real.sqrt ((toModel xa).toReal)) :=
        IEEE32Exec.toReal_sqrt_eq_fp32Round (x := xa) (dx := da) hxa hfin
      simpa [envS, xa, evalRuntime, evalSpec, iha] using href

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
