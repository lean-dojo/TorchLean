/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Rounding
public import NN.Spec.Core.Context

/-!
## `NF`: a rounded scalar type (rounding-on-`ℝ`)

`NeuralFloat` (the record with a mantissa/exponent) is useful for *talking about the grid* and for
stating format predicates like `FLT_format`. In many places, though, we want something closer to a
“numeric scalar type” that we can plug into higher-level specs and examples.

`NF β fexp rnd` is that scalar:

- it stores a semantic value `val : ℝ`,
- and *by construction* every primitive operation rounds back to the format using `neural_round`.

So when you write `a + b` in `NF`, what you get is:

`val(a + b) = round( val(a) + val(b) )`

This is the standard textbook model used for floating-point error analysis: compute in reals, then
incur a rounding error at each step (Higham/Goldberg style).

Trust boundary:
- `NF`/`NeuralFloat` are proof-relevant *Lean models* of rounded arithmetic (built on `ℝ`).
- Instantiating `NF` with IEEE single parameters + round-to-nearest-even models the **finite**
  fragment of IEEE-754 binary32 (subnormals/rounding via `fexp`, but no NaN/Inf).
- Correspondence to hardware float32 / Lean's builtin `Float` is not proved in this file; that
  connection is an external assumption/interface boundary (or requires a separate verified kernel).
-/

@[expose] public section

namespace TorchLean.Floats

/--
Rounded scalar value at a given radix/format/rounding mode.

`β` is the radix (typically `2`), `fexp` selects the exponent grid, and `rnd` rounds the scaled
mantissa to an integer.
-/
structure NF (β : NeuralRadix) (fexp : ℤ → ℤ) (rnd : ℝ → ℤ) where
  /-- val. -/
  val : ℝ

namespace NF

variable {β : NeuralRadix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [NeuralValidExp fexp] [NeuralValidRnd rnd]

/-- The rounding operator associated with the format: `roundR x = neural_round … x`. -/
@[inline] noncomputable def roundR (x : ℝ) : ℝ := neuralRound (β := β) (fexp := fexp) rnd x

/-- Inject a real into `NF` by rounding it onto the target grid. -/
@[inline] noncomputable def ofReal (x : ℝ) : NF β fexp rnd := ⟨roundR (β := β) (fexp := fexp) (rnd
  := rnd) x⟩

/-- Forgetful projection (semantic view): treat an `NF` as a real number. -/
@[inline] noncomputable def toReal (x : NF β fexp rnd) : ℝ := x.val

omit [NeuralValidRnd rnd] in
/-- `toReal (ofReal x)` is definitionally the rounded real `roundR x`. -/
@[simp] lemma toReal_ofReal (x : ℝ) :
    toReal (β := β) (fexp := fexp) (rnd := rnd) (ofReal (β := β) (fexp := fexp) (rnd := rnd) x) =
      roundR (β := β) (fexp := fexp) (rnd := rnd) x := rfl

omit [NeuralValidRnd rnd] in
/-- The underlying `val` field of `ofReal x` is `roundR x`. -/
@[simp] lemma val_ofReal (x : ℝ) :
    (ofReal (β := β) (fexp := fexp) (rnd := rnd) x).val =
      roundR (β := β) (fexp := fexp) (rnd := rnd) x := rfl

/-- A default inhabitant (rounded zero). -/
noncomputable instance : Inhabited (NF β fexp rnd) where
  default := ofReal (β := β) (fexp := fexp) (rnd := rnd) 0

/-- Coerce natural literals into `NF` by rounding `(n : ℝ)` onto the grid. -/
noncomputable instance : Coe Nat (NF β fexp rnd) where
  coe n := ofReal (β := β) (fexp := fexp) (rnd := rnd) (n : ℝ)

/-- `0` and `1` for `NF` are defined via `ofReal`, so they live on the target grid. -/
noncomputable instance : Zero (NF β fexp rnd) where
  zero := ofReal (β := β) (fexp := fexp) (rnd := rnd) 0

/-- `1 : NF` is `ofReal 1`, i.e. the rounded real `1` on the target grid. -/
noncomputable instance : One (NF β fexp rnd) where
  one := ofReal (β := β) (fexp := fexp) (rnd := rnd) 1

/--
Arithmetic on `NF` is “compute in `ℝ`, then round”.

This is the key choice that makes many error bounds compositional: each primitive incurs at most
`ulp/2` of rounding error (under round-to-nearest assumptions), so long compositions can be bounded
by accumulating per-op bounds.
-/
noncomputable instance : Neg (NF β fexp rnd) where
  neg x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (-x.val)

/-- Rounded addition: `val(a + b) = roundR (val a + val b)`. -/
noncomputable instance : Add (NF β fexp rnd) where
  add a b := ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val + b.val)

/-- Rounded subtraction: `val(a - b) = roundR (val a - val b)`. -/
noncomputable instance : Sub (NF β fexp rnd) where
  sub a b := ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val - b.val)

/-- Rounded multiplication: `val(a * b) = roundR (val a * val b)`. -/
noncomputable instance : Mul (NF β fexp rnd) where
  mul a b := ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val * b.val)

/-- Rounded division: `val(a / b) = roundR (val a / val b)`. -/
noncomputable instance : Div (NF β fexp rnd) where
  div a b := ofReal (β := β) (fexp := fexp) (rnd := rnd) (a.val / b.val)

/--
Boolean equality on `NF` values (semantic equality of reals).

This is *not* intended as a fast runtime check (it relies on classical decidability for `ℝ`), but
it is convenient for specs that want a `BEq` instance for logging or small examples.
-/
noncomputable instance : BEq (NF β fexp rnd) where
  beq a b := decide (a.val = b.val)

/-- Strict order on `NF` induced by the strict order on `ℝ` via the `val` field. -/
noncomputable instance : LT (NF β fexp rnd) where
  lt a b := a.val < b.val

/-- Non-strict order on `NF` induced by `≤` on `ℝ` via the `val` field. -/
noncomputable instance : LE (NF β fexp rnd) where
  le a b := a.val ≤ b.val

/-- Min/max in the semantic order on `ℝ`, lifted to `NF`. -/
noncomputable instance : Min (NF β fexp rnd) where
  min x y := if x ≤ y then x else y

/-- `max` on `NF`, defined by comparing the underlying real values. -/
noncomputable instance : Max (NF β fexp rnd) where
  max x y := if x ≥ y then x else y

/--
Exponentiation, rounded back to the grid.

We define `a^b` via `exp(b * log a)` when `a > 0`. If `a ≤ 0` we return `0` to keep the operation
total in `ℝ`. (In practical ML code this is usually guarded or avoided; here we prefer a total
spec-level function over partiality.)
-/
noncomputable instance : Pow (NF β fexp rnd) (NF β fexp rnd) where
  pow a b :=
    if a.val > 0 then
      ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.exp (b.val * Real.log a.val))
    else
      ofReal (β := β) (fexp := fexp) (rnd := rnd) 0

/--
Common math functions lifted to `NF` by “evaluate in `ℝ`, then round”.

This matches the same modeling decision as `Add`/`Mul`: the spec says what real function we intend,
and the rounding model accounts for discretization.
-/
noncomputable instance : MathFunctions (NF β fexp rnd) where
  exp  x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.exp x.val)
  tanh x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.tanh x.val)
  cosh x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.cosh x.val)
  sinh x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.sinh x.val)
  sqrt x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.sqrt x.val)
  abs  x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (|x.val|)
  log  x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.log x.val)
  pi      := ofReal (β := β) (fexp := fexp) (rnd := rnd) Real.pi
  cos  x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.cos x.val)
  sin  x := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.sin x.val)

/-- Numeric constants for NF via rounded reals. -/
noncomputable instance : Numbers (NF β fexp rnd) where
  neg_point_five := ofReal (β := β) (fexp := fexp) (rnd := rnd) (-0.5)
  neg_one        := ofReal (β := β) (fexp := fexp) (rnd := rnd) (-1)
  pointone       := ofReal (β := β) (fexp := fexp) (rnd := rnd) 0.1
  pointfive      := ofReal (β := β) (fexp := fexp) (rnd := rnd) 0.5
  zero           := ofReal (β := β) (fexp := fexp) (rnd := rnd) 0
  one            := ofReal (β := β) (fexp := fexp) (rnd := rnd) 1
  two            := ofReal (β := β) (fexp := fexp) (rnd := rnd) 2
  three          := ofReal (β := β) (fexp := fexp) (rnd := rnd) 3
  four           := ofReal (β := β) (fexp := fexp) (rnd := rnd) 4
  five           := ofReal (β := β) (fexp := fexp) (rnd := rnd) 5
  ten            := ofReal (β := β) (fexp := fexp) (rnd := rnd) 10
  log10          := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.log 10)
  log10000       := ofReal (β := β) (fexp := fexp) (rnd := rnd) (Real.log 10000)
  epsilon        := ofReal (β := β) (fexp := fexp) (rnd := rnd) (1e-6)
  neg_thousand   := ofReal (β := β) (fexp := fexp) (rnd := rnd) (-1000)

/--
`Context` instance used by TorchLean specs.

We provide a decidable `>` relation by classical reasoning on `ℝ` (noncomputable, but fine for the
spec layer).
-/
noncomputable instance : Context (NF β fexp rnd) := {
  decidable_gt := Classical.decRel _
}

/--
Extract an approximate radix-`β` mantissa/exponent pair for debugging.

We compute:

- `e := cexp(x)` from the format (`fexp`),
- `m := rnd( scaled_mantissa(x) )`,

so that `x ≈ m · β^e` (with the approximation coming from rounding).

This is meant for logs / human inspection; it is not used by the core proofs.
-/
noncomputable def mantExp (x : NF β fexp rnd) : Int × Int :=
  let e : Int := neuralCexp β fexp x.val
  let m : Int := (rnd (neuralScaledMantissa β fexp x.val))
  (m, e)

/-- Format an integer in base 10. -/
@[inline] def fmtInt (n : Int) : String := toString n

/--
Format an `NF` value as a radix-`β` scientific string `"m * β^e"`.

Example (β = 2): `"-123 * 2^7"`.
-/
noncomputable def formatRadix (x : NF β fexp rnd) : String :=
  let (m, e) := mantExp (β := β) (fexp := fexp) (rnd := rnd) x
  if m = 0 then "0"
  else s!"{fmtInt m} * {β.base}^{fmtInt e}"

/-- Format an interval [lo, hi] for NF values using `formatRadix`. -/
noncomputable def formatIntervalRadix (lo hi : NF β fexp rnd) : String :=
  (s!"[ {formatRadix (β := β) (fexp := fexp) (rnd := rnd) lo}, " ++
    s!"{formatRadix (β := β) (fexp := fexp) (rnd := rnd) hi} ]")

end NF

end TorchLean.Floats
