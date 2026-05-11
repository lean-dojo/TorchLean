/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32

/-!
# BridgeFP32Expr

Compositional refinement lemmas on top of `NN/Floats/IEEEExec/BridgeFP32.lean`.

In `BridgeFP32.lean` we prove refinement theorems *one op at a time* (add/mul/div/sqrt, etc.). In
practice, though, we often want to talk about a whole scalar expression and not re-run the same
“op-level” proof script at every node.

This file provides:

- a compact scalar expression language `Expr`, and
- an “all intermediates are finite” witness `FiniteEval`,

so we can state and prove a single theorem that looks like:

```
toReal (evalRuntime env e) = evalSpec (toReal ∘ env) e
```

where:
- `evalRuntime` evaluates the AST using the executable IEEE-754 kernel `IEEE32Exec`, and
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


namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

noncomputable section

/-!
## A small scalar expression language

`Expr` is a small, scalar-only AST. TorchLean's main IRs live elsewhere; this wrapper exists to
state expression-level refinement theorems for straight-line float32 computations.
-/

/-- A compact AST for scalar float32 expressions evaluated using `IEEE32Exec`. -/
inductive Expr where
  | var : Nat → Expr
  | const : IEEE32Exec → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr
  | fma : Expr → Expr → Expr → Expr
  | sqrt : Expr → Expr
  deriving Repr

/-- Evaluate an `Expr` using the executable float32 kernel (`IEEE32Exec`). -/
def evalRuntime (env : Nat → IEEE32Exec) : Expr → IEEE32Exec
  | .var i => env i
  | .const x => x
  | .add a b => IEEE32Exec.add (evalRuntime env a) (evalRuntime env b)
  | .sub a b => IEEE32Exec.sub (evalRuntime env a) (evalRuntime env b)
  | .mul a b => IEEE32Exec.mul (evalRuntime env a) (evalRuntime env b)
  | .div a b => IEEE32Exec.div (evalRuntime env a) (evalRuntime env b)
  | .fma a b c => IEEE32Exec.fma (evalRuntime env a) (evalRuntime env b) (evalRuntime env c)
  | .sqrt a => IEEE32Exec.sqrt (evalRuntime env a)

/-- Real semantics for the compact scalar expression language. -/
def evalSpec (env : Nat → ℝ) : Expr → ℝ
  | .var i => env i
  | .const x => IEEE32Exec.toReal x
  | .add a b => fp32Round (evalSpec env a + evalSpec env b)
  | .sub a b => fp32Round (evalSpec env a - evalSpec env b)
  | .mul a b => fp32Round (evalSpec env a * evalSpec env b)
  | .div a b => fp32Round (evalSpec env a / evalSpec env b)
  | .fma a b c => fp32Round (evalSpec env a * evalSpec env b + evalSpec env c)
  | .sqrt a => fp32Round (Real.sqrt (evalSpec env a))

/-!
## "Finite evaluation" witnesses (finite at every intermediate node)

`BridgeFP32.lean` is explicit about the trust boundary: it bridges only the **finite** behavior of
`IEEE32Exec` to `FP32`. For expression-level statements, we therefore need an assumption that every
intermediate evaluation remains finite.

`FiniteEval env e d` is a proof object that says:
- evaluating `e` under `env` is finite, and
- its decoded dyadic value is `d`.

We store the dyadic because it gives us a convenient “finiteness certificate”:
`toDyadic? x = some d` immediately rules out NaN/Inf and unlocks the op-level bridge lemmas.
-/

private lemma isFinite_eq_true_of_toDyadic?_some {x : IEEE32Exec} {d : Dyadic}
    (hx : toDyadic? x = some d) : isFinite x = true := by
  unfold IEEE32Exec.isFinite
  apply (bne_iff_ne).2
  intro hEq
  have hEqb : (expField x == expAllOnes) = true := (beq_iff_eq).2 hEq
  by_cases hFrac : fracField x = 0
  · have hFracB : (fracField x == 0) = true := (beq_iff_eq).2 hFrac
    have hinfTrue : isInf x = true := by
      simp [IEEE32Exec.isInf, hEqb, hFracB]
    have hinfFalse : isInf x = false := isInf_eq_false_of_toDyadic?_some (hx := hx)
    have : False := by
      simp [hinfTrue] at hinfFalse
    exact this.elim
  · have hFracNeB : (fracField x != 0) = true := (bne_iff_ne).2 hFrac
    have hnanTrue : isNaN x = true := by
      simp [IEEE32Exec.isNaN, hEqb, hFracNeB]
    have hnanFalse : isNaN x = false := isNaN_eq_false_of_toDyadic?_some (hx := hx)
    have : False := by
      simp [hnanTrue] at hnanFalse
    exact this.elim

/-- Finite-evaluation witness for the compact scalar expression language. -/
inductive FiniteEval (env : Nat → IEEE32Exec) : Expr → Dyadic → Prop where
  | var (i : Nat) (d : Dyadic) (h : toDyadic? (env i) = some d) :
      FiniteEval env (.var i) d
  | const (x : IEEE32Exec) (d : Dyadic) (h : toDyadic? x = some d) :
      FiniteEval env (.const x) d
  | add {a b : Expr} {da db dout : Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : toDyadic? (IEEE32Exec.add (evalRuntime env a) (evalRuntime env b)) = some dout) :
      FiniteEval env (.add a b) dout
  | sub {a b : Expr} {da db dout : Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : toDyadic? (IEEE32Exec.sub (evalRuntime env a) (evalRuntime env b)) = some dout) :
      FiniteEval env (.sub a b) dout
  | mul {a b : Expr} {da db dout : Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hout : toDyadic? (IEEE32Exec.mul (evalRuntime env a) (evalRuntime env b)) = some dout) :
      FiniteEval env (.mul a b) dout
  | div {a b : Expr} {da db dout : Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db)
      (hden : db.mant ≠ 0)
      (hout : toDyadic? (IEEE32Exec.div (evalRuntime env a) (evalRuntime env b)) = some dout) :
      FiniteEval env (.div a b) dout
  | fma {a b c : Expr} {da db dc dout : Dyadic}
      (ha : FiniteEval env a da) (hb : FiniteEval env b db) (hc : FiniteEval env c dc)
      (hout : toDyadic? (IEEE32Exec.fma (evalRuntime env a) (evalRuntime env b) (evalRuntime env c))
        = some dout) :
      FiniteEval env (.fma a b c) dout
  | sqrt {a : Expr} {da dout : Dyadic}
      (ha : FiniteEval env a da)
      (hout : toDyadic? (IEEE32Exec.sqrt (evalRuntime env a)) = some dout) :
      FiniteEval env (.sqrt a) dout

namespace FiniteEval

/-- Extract the decoded dyadic of the runtime evaluation from a `FiniteEval` witness. -/
theorem toDyadic? {env : Nat → IEEE32Exec} {e : Expr} {d : Dyadic} :
    FiniteEval env e d → IEEE32Exec.toDyadic? (evalRuntime env e) = some d := by
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

This is the main result of the file: a compositional refinement theorem that follows the AST
AST shape and invokes the corresponding op-level bridge theorem at each node.

If you are thinking in PyTorch terms: `Expr` is a compact “forward graph”, and the result says that the
executable float32 evaluation agrees with the standard float32 mathematical model (real arithmetic +
rounding) on the finite path.
-/

/-- Main expression-level refinement theorem for IEEEExec. -/
theorem toReal_evalRuntime_eq_evalSpec (env : Nat → IEEE32Exec) :
    ∀ {e : Expr} {d : Dyadic}, FiniteEval env e d →
      IEEE32Exec.toReal (evalRuntime env e) = evalSpec (fun i => IEEE32Exec.toReal (env i)) e := by
  intro e d h
  let envS : Nat → ℝ := fun i => IEEE32Exec.toReal (env i)
  induction h with
  | var i d h =>
      simp [evalRuntime, evalSpec]
  | const x d h =>
      simp [evalRuntime, evalSpec]
  | add ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hxb : IEEE32Exec.toDyadic? xb = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (IEEE32Exec.add xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.add xa xb) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.add xa xb) = fp32Round (IEEE32Exec.toReal xa +
            IEEE32Exec.toReal xb) :=
        IEEE32Exec.toReal_add_eq_fp32Round (x := xa) (y := xb) (dx := da) (dy := db) hxa hxb hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | sub ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hxb : IEEE32Exec.toDyadic? xb = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (IEEE32Exec.sub xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.sub xa xb) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.sub xa xb) = fp32Round (IEEE32Exec.toReal xa -
            IEEE32Exec.toReal xb) :=
        IEEE32Exec.toReal_sub_eq_fp32Round (x := xa) (y := xb) (dx := da) (dy := db) hxa hxb hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | mul ha hb hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hxb : IEEE32Exec.toDyadic? xb = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (IEEE32Exec.mul xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.mul xa xb) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.mul xa xb) = fp32Round (IEEE32Exec.toReal xa *
            IEEE32Exec.toReal xb) :=
        IEEE32Exec.toReal_mul_eq_fp32Round (x := xa) (y := xb) (dx := da) (dy := db) hxa hxb hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | div ha hb hden hout iha ihb =>
      rename_i a b da db dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hxb : IEEE32Exec.toDyadic? xb = some db := FiniteEval.toDyadic? hb
      have hfin : isFinite (IEEE32Exec.div xa xb) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.div xa xb) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.div xa xb) = fp32Round (IEEE32Exec.toReal xa /
            IEEE32Exec.toReal xb) :=
        IEEE32Exec.toReal_div_eq_fp32Round (x := xa) (y := xb) (dx := da) (dy := db) hxa hxb hden
          hfin
      simpa [envS, xa, xb, evalRuntime, evalSpec, iha, ihb] using href
  | fma ha hb hc hout iha ihb ihc =>
      rename_i a b c da db dc dout
      let xa := evalRuntime env a
      let xb := evalRuntime env b
      let xc := evalRuntime env c
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hxb : IEEE32Exec.toDyadic? xb = some db := FiniteEval.toDyadic? hb
      have hxc : IEEE32Exec.toDyadic? xc = some dc := FiniteEval.toDyadic? hc
      have hfin : isFinite (IEEE32Exec.fma xa xb xc) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.fma xa xb xc) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.fma xa xb xc) =
            fp32Round (IEEE32Exec.toReal xa * IEEE32Exec.toReal xb + IEEE32Exec.toReal xc) :=
        IEEE32Exec.toReal_fma_eq_fp32Round (x := xa) (y := xb) (z := xc) (dx := da) (dy := db) (dz
          := dc)
          hxa hxb hxc hfin
      simpa [envS, xa, xb, xc, evalRuntime, evalSpec, iha, ihb, ihc] using href
  | sqrt ha hout iha =>
      rename_i a da dout
      let xa := evalRuntime env a
      have hxa : IEEE32Exec.toDyadic? xa = some da := FiniteEval.toDyadic? ha
      have hfin : isFinite (IEEE32Exec.sqrt xa) = true :=
        isFinite_eq_true_of_toDyadic?_some (x := IEEE32Exec.sqrt xa) (d := dout) hout
      have href :
          IEEE32Exec.toReal (IEEE32Exec.sqrt xa) = fp32Round (Real.sqrt (IEEE32Exec.toReal xa)) :=
        IEEE32Exec.toReal_sqrt_eq_fp32Round (x := xa) (dx := da) hxa hfin
      simpa [envS, xa, evalRuntime, evalSpec, iha] using href

end

end IEEE32Exec

end TorchLean.Floats.IEEE754
