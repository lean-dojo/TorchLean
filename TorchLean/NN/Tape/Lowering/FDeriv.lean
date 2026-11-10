/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tape.RealGraph.FDeriv
public import TorchLean.NN.Tape.Lowering.BackwardGraph

/-!
# Analytic upgrade of the runtime link: reverse mode computes `(fderiv eval)†`

Two developments meet in this file, both stated on the algebraic graphs
`Tape.GraphData α Δ Γ ss` (executable) and `Tape.Graph (α := α) Δ Γ ss` (proof-carrying), with
contexts `TorchLean.TensorPack α Γ`.

* The **runtime link** (`TorchLean.NN.Tape.Lowering.BackwardGraph`) proves that the
  tape engine's dense reverse pass (`Tape.backwardDenseFrom`) agrees with the algebraic model's full
  backpropagation `backpropAllCtx`, over any commutative semiring carrier and an arbitrary
  non-differentiable environment `Δ`.
* The **analytic tape model** (`TorchLean.NN.Tape.RealGraph.FDeriv`) proves that
  reverse-mode accumulation computes the adjoint of the Fréchet derivative of the forward
  evaluation, at `α := ℝ` and a fixed environment `d : Δ`.

This file connects them:

* **Input-prefix extraction.** The left half of `TorchLean.Pack.split` is the input (`Γ`-prefix)
  block of a full context, and `GraphData.takeLeft_backpropAllCtx` identifies the input block of
  the full backpropagation with the inputs-only `backpropCtx`.
* **Vectorization transport.** In `TorchLean.NN.Tape.RealGraph.FDeriv`, the
  `TensorPack.toVec` lemmas commute context vectorization with `cast`/`snoc`/`unsnoc`/`add`, so the
  `TensorPack`-level graph semantics coincide with the Euclidean `CtxVec` semantics
  (`GraphData.evalVec_flattenCtx`, `GraphData.jvpVec_flattenCtx`,
  `GraphData.backpropVec_flattenCtx`).
* **The composed endpoints.** `backpropCtx_eq_fderiv` upgrades the algebraic backpropagation at `ℝ`
  to the Fréchet-adjoint characterization. `backwardDenseFrom_eq_fderiv` combines it with the
  runtime link: on a lowered graph at `α := ℝ`, the tape model's dense reverse pass succeeds with
  the full backpropagation context, whose input prefix is `(fderiv ℝ eval x)† seed`. Both theorems
  assume differentiability only at the actual execution point (`RealGraph.FDerivCorrectAt`), which
  admits primitives that are not differentiable everywhere (`relu`, `abs`, `min`/`max`, `log`,
  `sqrt`, …). A global certificate `hg : RealGraph.FDerivCorrect g.toData d` enters as
  `hg.at (TensorPack.toVec x)`.

Throughout, "reverse pass" means the exact tape model: the program `Tape.backwardDenseFrom` at
the exact carrier `α := ℝ`. Native `Float` evaluation and the CUDA execution path lie outside
these statements, and relating them to the exact model is a separate approximation question.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

-- ---------------------------------------------------------------------------
-- Input-prefix extraction from a full context
-- ---------------------------------------------------------------------------

namespace TensorPack

/-- On a context with no intermediates, the left half of `Pack.split` is the cast along
`Γ ++ [] = Γ`. -/
theorem takeLeft_append_nil {α : Type} [Storage α] {Γ : List Shape}
    (w : TensorPack α (Γ ++ [])) :
    (Pack.split (ss₁ := Γ) (ss₂ := []) w).1 =
      Pack.cast (List.append_nil Γ) w := by
  induction Γ with
  | nil =>
    cases w
    rfl
  | cons s Γ ih =>
    cases w with
    | cons x w' =>
      show .cons x (Pack.split (ss₁ := Γ) (ss₂ := []) w').1 =
          Pack.cast
            (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (.cons x w')
      rw [Pack.cast_cons (s := s) (ss₁ := Γ ++ []) (ss₂ := Γ)
        (show s :: (Γ ++ []) = s :: Γ from List.append_nil (s :: Γ)) (List.append_nil Γ) x w', ih]

/-- The left half of `Pack.split` ignores a snoc-ed final block (after reassociating the
context). -/
theorem takeLeft_cast_snoc {α : Type} [Storage α] {Γ : List Shape} :
    ∀ {ss : List Shape} {τ : Shape} (h : (Γ ++ ss) ++ [τ] = Γ ++ (ss ++ [τ]))
      (w : TensorPack α (Γ ++ ss)) (y : Tensor α τ),
      (Pack.split (ss₁ := Γ) (ss₂ := ss ++ [τ]) (Pack.cast h (Pack.snoc w y))).1
        = (Pack.split (ss₁ := Γ) (ss₂ := ss) w).1 := by
  induction Γ with
  | nil =>
    intro ss τ h w y
    rfl
  | cons s Γ ih =>
    intro ss τ h w y
    cases w with
    | cons x w' =>
      show (Pack.split (ss₁ := s :: Γ) (ss₂ := ss ++ [τ])
            (Pack.cast h (.cons x (Pack.snoc w' y)))).1 =
          .cons x (Pack.split (ss₁ := Γ) (ss₂ := ss) w').1
      rw [Pack.cast_cons (s := s)
        (ss₁ := (Γ ++ ss) ++ [τ]) (ss₂ := Γ ++ (ss ++ [τ])) h
        (List.append_assoc Γ ss [τ]) x (Pack.snoc w' y)]
      show Pack.cons x (Pack.split (ss₁ := Γ) (ss₂ := ss ++ [τ])
            (Pack.cast (List.append_assoc Γ ss [τ]) (Pack.snoc w' y))).1 =
          Pack.cons x (Pack.split (ss₁ := Γ) (ss₂ := ss) w').1
      rw [ih]

end TensorPack

namespace Tape.GraphData

variable {α : Type} [Storage α] [Add α]
variable {Δ : Type}
variable {Γ : List Shape}

/--
The input (`Γ`-prefix) block of the full backpropagation is the inputs-only backpropagation.

Here `backpropAllCtx` keeps a cotangent for every value, as the tape engine's dense reverse pass
does, while `backpropCtx` returns the input block alone.
-/
theorem takeLeft_backpropAllCtx {ss : List Shape} (g : GraphData α Δ Γ ss)
    (x : TensorPack α Γ) (d : Δ) (seed : TensorPack α (Γ ++ ss)) :
    (Pack.split (ss₁ := Γ) (backpropAllCtx (α := α) g x d seed)).1
      = backpropCtx (α := α) g x d seed := by
  induction g with
  | nil =>
    exact TensorPack.takeLeft_append_nil (α := α) seed
  | snoc g node ih =>
    simp only [backpropAllCtx, backpropCtx]
    rw [TensorPack.takeLeft_cast_snoc]
    exact ih _

end Tape.GraphData

-- ---------------------------------------------------------------------------
-- Composed endpoints: the algebraic reverse pass at `ℝ` is `(fderiv eval)†`
-- ---------------------------------------------------------------------------

namespace Tape.Graph

variable {Δ : Type}
variable {Γ : List Shape}

/--
**Algebraic reverse pass = adjoint of the derivative.** At carrier `ℝ` and a fixed environment,
the inputs-only backpropagation computes the adjoint of the Fréchet derivative of the graph's
(vectorized) forward evaluation. Differentiability is assumed only at the values actually
encountered, admitting non-smooth primitives away from their kinks.
-/
theorem backpropCtx_eq_fderiv {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (d : Δ) (x : TensorPack ℝ Γ)
    (hg : RealGraph.FDerivCorrectAt (Γ := Γ) g.toData d (TensorPack.toVec x))
    (seed : TensorPack ℝ (Γ ++ ss)) :
    TensorPack.toVec (g.toData.backpropCtx x d seed)
      = (fderiv ℝ (g.toData.evalVec d) (TensorPack.toVec x)).adjoint (TensorPack.toVec seed) := by
  rw [← GraphData.backpropVec_flattenCtx]
  exact backpropVec_eq_fderiv g d _ _ hg

/--
**Tape-model reverse pass = adjoint of the Fréchet derivative.** On a lowered graph at `α := ℝ`,
the tape engine's dense reverse pass (`Tape.backwardDenseFrom`) succeeds and returns the full
backpropagation context, whose input (`Γ`-prefix) block is the adjoint of the Fréchet derivative
of the graph's forward evaluation applied to the seed. Differentiability is assumed only at `x`
and the values it reaches (`RealGraph.FDerivCorrectAt`).

The proof composes the runtime link (`GraphData.backwardDenseFrom_toTape`) with
`backpropCtx_eq_fderiv`. Because the carrier is the exact `ℝ`, native `Float` rounding
and the CUDA path lie outside the statement.
-/
theorem backwardDenseFrom_eq_fderiv {ss : List Shape}
    (g : Graph (α := ℝ) (Δ := Δ) (Γ := Γ) ss) (x : TensorPack ℝ Γ) (d0 : Δ)
    (seed : TensorPack ℝ (Γ ++ ss))
    (hg : RealGraph.FDerivCorrectAt (Γ := Γ) g.toData d0 (TensorPack.toVec x)) :
    Tape.backwardDenseFrom
        (t := (GraphData.toTape (α := ℝ) (Δ := Δ) (Γ := Γ) (ss := ss) g.toData x d0).1)
        (grads0 := TensorPack.erase (α := ℝ) (ss := Γ ++ ss) seed)
      = .ok (TensorPack.erase (α := ℝ) (ss := Γ ++ ss)
          (g.toData.backpropAllCtx x d0 seed))
    ∧ TensorPack.toVec (Pack.split (ss₁ := Γ) (g.toData.backpropAllCtx x d0 seed)).1
      = (fderiv ℝ (g.toData.evalVec d0) (TensorPack.toVec x)).adjoint (TensorPack.toVec seed) := by
  refine ⟨GraphData.backwardDenseFrom_toTape g.toData x d0 seed, ?_⟩
  rw [GraphData.takeLeft_backpropAllCtx]
  exact backpropCtx_eq_fderiv g d0 x hg seed

end Tape.Graph

end
end TorchLean
