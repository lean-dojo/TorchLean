/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Gradient.FDeriv
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Real
public import FloatLib.Floats.Formats.IEEE754.Native.Integer.Constructors

/-!
# Real readings of kernel expressions

`Expr.map f` replaces every scalar constant `c` of a kernel expression by `f c` and keeps every
other node. The tangent, seed and gradient transformations insert only the constants `0` and `1`,
so they commute with `map f` when `f 0 = 0` and `f 1 = 1` (`Expr.tangent_map`, `Expr.seed_map`,
`Expr.gradient_map`). For a map `f : α → ℝ`, the real reading `(e.gradient k j n).map f` of the
gradient kernel of a program over `α` is therefore the gradient kernel of the real expression
`e.map f`, and `Expr.map_gradient_eq_adjoint` applies `Expr.gradient_eq_adjoint` to it. Its
variants apply `Expr.exists_gradient_eq_adjoint`, which proves that the evaluated function exists,
and the forms of both for several inputs. The tangent theorems carry over in the same way.

For a configured binary scalar, FloatLib's `ExecFloat.Binary.toReal` keeps zero
(`ExecFloat.Binary.toReal_zero`), and it keeps one when the format is IEEE
(`ExecFloat.Binary.toReal_one`), as binary32 and binary64 are. Lean's native `Float32` and
`Float` read as reals through their binary32 and binary64 values, and we prove that this reading
keeps zero and one (`toReal_ofFloat32_zero`, `toReal_ofFloat32_one`, `toReal_ofFloat_zero`,
`toReal_ofFloat_one`). These results concern the expression with real arithmetic. They say nothing
about the rounding of a floating-point run, and the agreement of compiled native code with Lean's
logical model of `Float32` and `Float` is trusted.
-/

@[expose] public section

open Filter Topology

namespace TorchLean.Kernel

namespace Expr

variable {α β : Type}

/-- The expression `e` with each scalar constant `c` replaced by `f c` and all other nodes kept. -/
def map (f : α → β) : {Γ : List Ty} → {t : Ty} → Expr α Γ t → Expr β Γ t
  | _, _, .scalar c => .scalar (f c)
  | _, _, .index i => .index i
  | _, _, .predicate p => .predicate p
  | _, _, .var v => .var v
  | _, _, .load o i => .load o (i.map f)
  | _, _, .binary op x y => .binary op (x.map f) (y.map f)
  | _, _, .neg x => .neg (x.map f)
  | _, _, .indexBinary op x y => .indexBinary op (x.map f) (y.map f)
  | _, _, .compare op x y => .compare op (x.map f) (y.map f)
  | _, _, .indexCompare op x y => .indexCompare op (x.map f) (y.map f)
  | _, _, .cond p yes no => .cond (p.map f) (yes.map f) (no.map f)
  | _, _, .letIn value body => .letIn (value.map f) (body.map f)
  | _, _, .fold count initial body => .fold (count.map f) (initial.map f) (body.map f)

variable (f : α → β) {Γ : List Ty}

/-- Mapping a scalar constant applies `f` to it. -/
@[simp] theorem map_scalar (c : α) : (scalar c : Expr α Γ .scalar).map f = scalar (f c) := rfl

/-- Mapping keeps an index constant. -/
@[simp] theorem map_index (i : UInt64) : (index i : Expr α Γ .index).map f = index i := rfl

/-- Mapping keeps a predicate constant. -/
@[simp] theorem map_predicate (p : Bool) :
    (predicate p : Expr α Γ .predicate).map f = predicate p := rfl

/-- Mapping keeps a variable. -/
@[simp] theorem map_var {t : Ty} (v : Var Γ t) : (var v : Expr α Γ t).map f = var v := rfl

/-- Mapping a read maps its index. -/
@[simp] theorem map_load (o : Nat) (i : Expr α Γ .index) :
    (load o i).map f = load o (i.map f) := rfl

/-- Mapping a scalar operation maps its operands. -/
@[simp] theorem map_binary (op : ScalarOp) (x y : Expr α Γ .scalar) :
    (binary op x y).map f = binary op (x.map f) (y.map f) := rfl

/-- Mapping a negation maps its operand. -/
@[simp] theorem map_neg (x : Expr α Γ .scalar) : (neg x).map f = neg (x.map f) := rfl

/-- Mapping an index operation maps its operands. -/
@[simp] theorem map_indexBinary (op : IndexOp) (x y : Expr α Γ .index) :
    (indexBinary op x y).map f = indexBinary op (x.map f) (y.map f) := rfl

/-- Mapping a scalar comparison maps its operands. -/
@[simp] theorem map_compare (op : Compare) (x y : Expr α Γ .scalar) :
    (compare op x y).map f = compare op (x.map f) (y.map f) := rfl

/-- Mapping an index comparison maps its operands. -/
@[simp] theorem map_indexCompare (op : Compare) (x y : Expr α Γ .index) :
    (indexCompare op x y).map f = indexCompare op (x.map f) (y.map f) := rfl

/-- Mapping a conditional maps its predicate and both branches. -/
@[simp] theorem map_cond {t : Ty} (p : Expr α Γ .predicate) (yes no : Expr α Γ t) :
    (cond p yes no).map f = cond (p.map f) (yes.map f) (no.map f) := rfl

/-- Mapping a local binding maps its value and its body. -/
@[simp] theorem map_letIn {s t : Ty} (value : Expr α Γ s) (body : Expr α (s :: Γ) t) :
    (letIn value body).map f = letIn (value.map f) (body.map f) := rfl

/-- Mapping a fold maps its count, its initial value and its body. -/
@[simp] theorem map_fold (count : Expr α Γ .index) (initial : Expr α Γ .scalar)
    (body : Expr α (.scalar :: .index :: Γ) .scalar) :
    (fold count initial body).map f = fold (count.map f) (initial.map f) (body.map f) := rfl

/-- Renaming the variables of `e` commutes with mapping its scalar constants. -/
theorem rename_map {Δ : List Ty} {t : Ty} (ρ : Renaming Γ Δ) (e : Expr α Γ t) :
    (e.map f).rename ρ = (e.rename ρ).map f := by
  induction e generalizing Δ with
  | scalar | index | predicate | var => rfl
  | load operand i ih => simp only [map, rename, ih]
  | binary op x y hx hy => simp only [map, rename, hx, hy]
  | neg x hx => simp only [map, rename, hx]
  | indexBinary op x y hx hy => simp only [map, rename, hx, hy]
  | compare op x y hx hy => simp only [map, rename, hx, hy]
  | indexCompare op x y hx hy => simp only [map, rename, hx, hy]
  | cond p yes no hp hy hn => simp only [map, rename, hp, hy, hn]
  | letIn value body hv hb => simp only [map, rename, hv, hb]
  | fold count initial body hc hi hb => simp only [map, rename, hc, hi, hb]

/-- Mapping the scalar constants of `e` keeps the variables it reads. -/
theorem reads_map {s t : Ty} (v : Var Γ s) (e : Expr α Γ t) : (e.map f).reads v = e.reads v := by
  induction e with
  | scalar | index | predicate | var => rfl
  | load operand i ih => simp only [map, reads, ih]
  | binary op x y hx hy => simp only [map, reads, hx, hy]
  | neg x hx => simp only [map, reads, hx]
  | indexBinary op x y hx hy => simp only [map, reads, hx, hy]
  | compare op x y hx hy => simp only [map, reads, hx, hy]
  | indexCompare op x y hx hy => simp only [map, reads, hx, hy]
  | cond p yes no hp hy hn => simp only [map, reads, hp, hy, hn]
  | letIn value body hv hb => simp only [map, reads, hv, hb]
  | fold count initial body hc hi hb => simp only [map, reads, hc, hi, hb]

/-- Mapping a dual binding maps its value, its tangent and its body. -/
theorem bindDual_map {t : Ty} (s : Ty) (value : Expr α (dualScope Γ) s)
    (tangent : Expr α (s :: dualScope Γ) s) (body : Expr α (dualScope (s :: Γ)) t) :
    (bindDual s value tangent body).map f =
      bindDual s (value.map f) (tangent.map f) (body.map f) := by
  cases s <;> rfl

/-- The primal copy of the mapped expression is the mapped primal copy. -/
theorem primal_map {t : Ty} (e : Expr α Γ t) : (e.map f).primal = e.primal.map f := by
  induction e with
  | scalar | index | predicate | var => rfl
  | load operand i ih => simp only [map, primal, ih]
  | binary op x y hx hy => simp only [map, primal, hx, hy]
  | neg x hx => simp only [map, primal, hx]
  | indexBinary op x y hx hy => simp only [map, primal, hx, hy]
  | compare op x y hx hy => simp only [map, primal, hx, hy]
  | indexCompare op x y hx hy => simp only [map, primal, hx, hy]
  | cond p yes no hp hy hn => simp only [map, primal, hp, hy, hn]
  | letIn value body hv hb => simp only [map, primal, hv, hb, bindDual_map]
  | fold count initial body hc hi hb =>
      simp only [map, primal, hc, hi, hb]
      rfl

/-- The tangent of a fold built from mapped parts is the mapped tangent of the fold. -/
theorem foldTangent_map (initial : Expr α (dualScope Γ) .scalar)
    (body body' : Expr α (.scalar :: .scalar :: .index :: dualScope Γ) .scalar) :
    foldTangent (initial.map f) (body.map f) (body'.map f) =
      (foldTangent initial body body').map f := by
  unfold foldTangent
  rw [reads_map]
  split_ifs <;> simp only [map, rename_map]

variable {f}

/-- The tangent of the mapped expression is the mapped tangent when `f` keeps zero, the only
constant that the tangent inserts. -/
theorem tangent_map [OfNat α 0] [OfNat β 0] (hf0 : f 0 = 0) (k : Nat) {t : Ty} (e : Expr α Γ t) :
    (e.map f).tangent k = (e.tangent k).map f := by
  induction e with
  | scalar => simp only [map, tangent, hf0]
  | index | predicate => rfl
  | @var Γ t v => cases t <;> rfl
  | load operand i _ => simp only [map, tangent, primal_map]
  | binary op x y hx hy => cases op <;> simp only [map, tangent, primal_map, hx, hy]
  | neg x hx => simp only [map, tangent, hx]
  | indexBinary op x y _ _ => simp only [map, tangent, primal_map]
  | compare op x y _ _ => simp only [map, tangent, primal_map]
  | indexCompare op x y _ _ => simp only [map, tangent, primal_map]
  | cond p yes no _ hy hn => simp only [map, tangent, primal_map, hy, hn]
  | letIn value body hv hb =>
      simp only [map, tangent, primal_map, rename_map, bindDual_map, hv, hb]
  | fold count initial body _ hi hb =>
      simp only [map, tangent, primal_map, hi, hb]
      congr 1
      exact foldTangent_map f _ _ _

/-- The seeded mapped expression is the mapped seeded expression when `f` keeps zero and one, the
constants that the seed inserts. -/
theorem seed_map [OfNat α 0] [OfNat α 1] [OfNat β 0] [OfNat β 1] (hf0 : f 0 = 0) (hf1 : f 1 = 1)
    (k j : Nat) {t : Ty} (a : Var Γ .index) (e : Expr α Γ t) :
    (e.map f).seed k j a = (e.seed k j a).map f := by
  induction e with
  | scalar | index | predicate | var => rfl
  | load o i ih =>
      simp only [map, seed]
      split_ifs <;> simp only [map, ih, hf0, hf1]
  | binary op x y hx hy => simp only [map, seed, hx, hy]
  | neg x hx => simp only [map, seed, hx]
  | indexBinary op x y hx hy => simp only [map, seed, hx, hy]
  | compare op x y hx hy => simp only [map, seed, hx, hy]
  | indexCompare op x y hx hy => simp only [map, seed, hx, hy]
  | cond p yes no hp hy hn => simp only [map, seed, hp, hy, hn]
  | letIn value body hv hb => simp only [map, seed, hv, hb]
  | fold count initial body hc hi hb => simp only [map, seed, hc, hi, hb]

/-- The gradient kernel of the mapped expression is the mapped gradient kernel when `f` keeps zero
and one. -/
theorem gradient_map [OfNat α 0] [OfNat α 1] [OfNat β 0] [OfNat β 1] (hf0 : f 0 = 0)
    (hf1 : f 1 = 1) (k j : Nat) (n : UInt64) (e : Expr α [.index] .scalar) :
    (e.map f).gradient k j n = (e.gradient k j n).map f := by
  simp only [gradient, map, tangent_map hf0, rename_map, seed_map hf0 hf1, hf0]

/-- Let `f` read the constants of `α` as reals and keep zero and one. Under the hypotheses of
`gradient_eq_adjoint` for the real reading `e.map f`, `F` is differentiable at `θ₀` and the real
reading of the gradient kernel of `e` computes the adjoint of its derivative at the cotangent
`cot`. -/
theorem map_gradient_eq_adjoint [OfNat α 0] [OfNat α 1] {f : α → ℝ} (hf0 : f 0 = 0)
    (hf1 : f 1 = 1) (e : Expr α [.index] .scalar) {k j : Nat} {m n : UInt64} {read : Reader ℝ}
    (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Fin m.toNat)}
    {F : EuclideanSpace ℝ (Fin m.toNat) → EuclideanSpace ℝ (Fin n.toNat)}
    (hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.map f)
      (read.withValues (withInput x j θ)) r.val.toUInt64 = .ok (F θ r))
    (hreg : ∀ r : Fin n.toNat,
      (e.map f).Regular (read.withValues (withInput x j θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ} (hR : ∀ o i y, read.withValues (withInput x j θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    DifferentiableAt ℝ F θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps ((e.gradient k j n).map f) R a.val.toUInt64 =
        .ok ((fderiv ℝ F θ₀).adjoint cot a) := by
  simpa only [gradient_map hf0 hf1] using gradient_eq_adjoint (e.map f) hk x hF hreg hR hcot

/-- Let `f` read the constants of `α` as reals and keep zero and one. Under the hypotheses of
`exists_gradient_eq_adjoint` for the real reading `e.map f`, near `θ₀` the outputs of `e.map f`
are the entries of a map `F` that is differentiable at `θ₀`, and the real reading of the gradient
kernel of `e` computes the adjoint of its derivative at `cot`. -/
theorem exists_map_gradient_eq_adjoint [OfNat α 0] [OfNat α 1] {f : α → ℝ} (hf0 : f 0 = 0)
    (hf1 : f 1 = 1) (e : Expr α [.index] .scalar) {k j : Nat} {m n : UInt64} {read : Reader ℝ}
    (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Fin m.toNat)}
    (hy : ∀ r : Fin n.toNat, ∃ y, evaluate Arithmetic.ofOps (e.map f)
      (read.withValues (withInput x j θ₀)) r.val.toUInt64 = .ok y)
    (hreg : ∀ r : Fin n.toNat,
      (e.map f).Regular (read.withValues (withInput x j θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ} (hR : ∀ o i y, read.withValues (withInput x j θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    ∃ F : EuclideanSpace ℝ (Fin m.toNat) → EuclideanSpace ℝ (Fin n.toNat),
      (∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.map f)
        (read.withValues (withInput x j θ)) r.val.toUInt64 = .ok (F θ r)) ∧
      DifferentiableAt ℝ F θ₀ ∧ ∀ a : Fin m.toNat,
        evaluate Arithmetic.ofOps ((e.gradient k j n).map f) R a.val.toUInt64 =
          .ok ((fderiv ℝ F θ₀).adjoint cot a) := by
  simpa only [gradient_map hf0 hf1] using
    exists_gradient_eq_adjoint (e.map f) hk x hy hreg hR hcot

/-- Let `f` read the constants of `α` as reals and keep zero and one. Under the hypotheses of
`gradient_eq_adjoint_withInputs` for the real reading `e.map f`, with the inputs `o < k` moving
with the blocks of `θ`, `F` is differentiable at `θ₀` and the real reading of the gradient kernel
of each input `j < k` of `e` computes block `j` of the adjoint of its derivative at `cot`. -/
theorem map_gradient_eq_adjoint_withInputs [OfNat α 0] [OfNat α 1] {f : α → ℝ} (hf0 : f 0 = 0)
    (hf1 : f 1 = 1) (e : Expr α [.index] .scalar) {k : Nat} {m : Fin k → UInt64} {n : UInt64}
    {read : Reader ℝ} (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)}
    {F : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) → EuclideanSpace ℝ (Fin n.toNat)}
    (hF : ∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.map f)
      (read.withValues (withInputs x θ)) r.val.toUInt64 = .ok (F θ r))
    (hreg : ∀ r : Fin n.toNat,
      (e.map f).Regular (read.withValues (withInputs x θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ}
    (hR : ∀ o i y, read.withValues (withInputs x θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    DifferentiableAt ℝ F θ₀ ∧ ∀ (j : Fin k) (a : Fin (m j).toNat),
      evaluate Arithmetic.ofOps ((e.gradient k j n).map f) R a.val.toUInt64 =
        .ok ((fderiv ℝ F θ₀).adjoint cot ⟨j, a⟩) := by
  simpa only [gradient_map hf0 hf1] using
    gradient_eq_adjoint_withInputs (e.map f) hk x hF hreg hR hcot

/-- Let `f` read the constants of `α` as reals and keep zero and one. Under the hypotheses of
`exists_gradient_eq_adjoint_withInputs` for the real reading `e.map f`, near `θ₀` the outputs of
`e.map f` are the entries of a map `F` that is differentiable at `θ₀`, and the real reading of the
gradient kernel of each input `j < k` of `e` computes block `j` of the adjoint of its derivative at
`cot`. -/
theorem exists_map_gradient_eq_adjoint_withInputs [OfNat α 0] [OfNat α 1] {f : α → ℝ}
    (hf0 : f 0 = 0) (hf1 : f 1 = 1) (e : Expr α [.index] .scalar) {k : Nat}
    {m : Fin k → UInt64} {n : UInt64} {read : Reader ℝ}
    (hk : ∀ o i y, read o i = .ok y → o < k) (x : Nat → UInt64 → ℝ)
    {θ₀ : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat)}
    (hy : ∀ r : Fin n.toNat, ∃ y, evaluate Arithmetic.ofOps (e.map f)
      (read.withValues (withInputs x θ₀)) r.val.toUInt64 = .ok y)
    (hreg : ∀ r : Fin n.toNat,
      (e.map f).Regular (read.withValues (withInputs x θ₀)) (Env.output r.val.toUInt64))
    {R : Reader ℝ}
    (hR : ∀ o i y, read.withValues (withInputs x θ₀) o i = .ok y → R o i = .ok y)
    {cot : EuclideanSpace ℝ (Fin n.toNat)}
    (hcot : ∀ r : Fin n.toNat, R k r.val.toUInt64 = .ok (cot r)) :
    ∃ F : EuclideanSpace ℝ (Σ o : Fin k, Fin (m o).toNat) → EuclideanSpace ℝ (Fin n.toNat),
      (∀ᶠ θ in 𝓝 θ₀, ∀ r : Fin n.toNat, evaluate Arithmetic.ofOps (e.map f)
        (read.withValues (withInputs x θ)) r.val.toUInt64 = .ok (F θ r)) ∧
      DifferentiableAt ℝ F θ₀ ∧ ∀ (j : Fin k) (a : Fin (m j).toNat),
        evaluate Arithmetic.ofOps ((e.gradient k j n).map f) R a.val.toUInt64 =
          .ok ((fderiv ℝ F θ₀).adjoint cot ⟨j, a⟩) := by
  simpa only [gradient_map hf0 hf1] using
    exists_gradient_eq_adjoint_withInputs (e.map f) hk x hy hreg hR hcot

/-- Let `f` read the constants of `α` as reals and keep zero. Under the hypotheses of
`hasFDerivAt_evaluate_tangent` for `e.map f`, the value `g` has a derivative `D` at `θ₀` and the
real reading of the tangent kernel of `e` computes `D u`. -/
theorem hasFDerivAt_evaluate_map_tangent [OfNat α 0] {f : α → ℝ} (hf0 : f 0 = 0)
    (e : Expr α [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) {P : Type} [NormedAddCommGroup P]
    [NormedSpace ℝ P] {θ₀ : P} {x : P → Nat → UInt64 → ℝ} {x' : Nat → UInt64 → P →L[ℝ] ℝ}
    (hx : ∀ j i y, read j i = .ok y → HasFDerivAt (fun θ => x θ j i) (x' j i) θ₀)
    {index : UInt64} {g : P → ℝ}
    (hg : ∀ᶠ θ in 𝓝 θ₀,
      evaluate Arithmetic.ofOps (e.map f) (read.withValues (x θ)) index = .ok (g θ))
    (hreg : (e.map f).Regular (read.withValues (x θ₀)) (Env.output index)) :
    ∃ D : P →L[ℝ] ℝ, HasFDerivAt g D θ₀ ∧
      ∀ u, evaluate Arithmetic.ofOps ((e.tangent k).map f)
        (read.dual k (x θ₀) fun j i => x' j i u) index = .ok (D u) := by
  simpa only [tangent_map hf0] using hasFDerivAt_evaluate_tangent (e.map f) hk hx hg hreg

/-- Let `f` read the constants of `α` as reals and keep zero. Under the hypotheses of
`exists_hasFDerivAt_tangent` for `e.map f`, the value of `e.map f` has a derivative `D` at `θ₀` and
the real reading of the tangent kernel of `e` computes `D u`. -/
theorem exists_hasFDerivAt_map_tangent [OfNat α 0] {f : α → ℝ} (hf0 : f 0 = 0)
    (e : Expr α [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) {P : Type} [NormedAddCommGroup P]
    [NormedSpace ℝ P] {θ₀ : P} {x : P → Nat → UInt64 → ℝ} {x' : Nat → UInt64 → P →L[ℝ] ℝ}
    (hx : ∀ j i y, read j i = .ok y → HasFDerivAt (fun θ => x θ j i) (x' j i) θ₀)
    {index : UInt64} {y : ℝ}
    (hy : evaluate Arithmetic.ofOps (e.map f) (read.withValues (x θ₀)) index = .ok y)
    (hreg : (e.map f).Regular (read.withValues (x θ₀)) (Env.output index)) :
    ∃ (g : P → ℝ) (D : P →L[ℝ] ℝ),
      (∀ᶠ θ in 𝓝 θ₀,
        evaluate Arithmetic.ofOps (e.map f) (read.withValues (x θ)) index = .ok (g θ)) ∧
      HasFDerivAt g D θ₀ ∧
      ∀ u, evaluate Arithmetic.ofOps ((e.tangent k).map f)
        (read.dual k (x θ₀) fun j i => x' j i u) index = .ok (D u) := by
  simpa only [tangent_map hf0] using exists_hasFDerivAt_tangent (e.map f) hk hx hy hreg

/-- Let `f` read the constants of `α` as reals and keep zero. On the line of inputs `x + s dx`,
the real reading of the tangent kernel of `e` computes the derivative at `0` of the value `g s` of
`e.map f`, under the hypotheses of `hasDerivAt_evaluate_tangent`. -/
theorem hasDerivAt_evaluate_map_tangent [OfNat α 0] {f : α → ℝ} (hf0 : f 0 = 0)
    (e : Expr α [.index] .scalar) {k : Nat} {read : Reader ℝ}
    (hk : ∀ j i y, read j i = .ok y → j < k) (x dx : Nat → UInt64 → ℝ) {index : UInt64}
    {g : ℝ → ℝ}
    (hg : ∀ᶠ s in 𝓝 0, evaluate Arithmetic.ofOps (e.map f)
      (read.withValues fun j i => x j i + s * dx j i) index = .ok (g s))
    (hreg : (e.map f).Regular (read.withValues x) (Env.output index)) :
    ∃ d, evaluate Arithmetic.ofOps ((e.tangent k).map f) (read.dual k x dx) index = .ok d ∧
      HasDerivAt g d 0 := by
  simpa only [tangent_map hf0] using hasDerivAt_evaluate_tangent (e.map f) hk x dx hg hreg

end Expr

open FloatLib.Floats

/-- Lean's native `Float32` zero reads as the real `0` through its binary32 value. -/
theorem toReal_ofFloat32_zero :
    ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 (0 : Float32) : Binary32) = 0 := by
  have h : ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat32 (0 : Float32) : Binary32) =
      ExecFloat.Binary.toModel (0 : Binary32) := by
    rw [show (0 : Float32) = Float32.ofNat 0 from rfl, ExecFloat.Binary.toModel_ofFloat32_ofNat]
    exact (ExecFloat.Binary.toModel_ofModel _).symm
  rw [ExecFloat.Binary.toReal, h]
  exact ExecFloat.Binary.toReal_zero

/-- Lean's native `Float32` one reads as the real `1` through its binary32 value. -/
theorem toReal_ofFloat32_one :
    ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat32 (1 : Float32) : Binary32) = 1 := by
  have h : ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat32 (1 : Float32) : Binary32) =
      ExecFloat.Binary.toModel (1 : Binary32) := by
    rw [show (1 : Float32) = Float32.ofNat 1 from rfl, ExecFloat.Binary.toModel_ofFloat32_ofNat]
    exact (ExecFloat.Binary.toModel_ofModel _).symm
  rw [ExecFloat.Binary.toReal, h]
  exact ExecFloat.Binary.toReal_one

/-- Lean's native `Float` zero reads as the real `0` through its binary64 value. -/
theorem toReal_ofFloat_zero :
    ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat (0 : Float) : Binary64) = 0 := by
  have h : ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat (0 : Float) : Binary64) =
      ExecFloat.Binary.toModel (0 : Binary64) := by
    rw [show (0 : Float) = Float.ofNat 0 from rfl, ExecFloat.Binary.toModel_ofFloat_ofNat]
    exact (ExecFloat.Binary.toModel_ofModel _).symm
  rw [ExecFloat.Binary.toReal, h]
  exact ExecFloat.Binary.toReal_zero

/-- Lean's native `Float` one reads as the real `1` through its binary64 value. -/
theorem toReal_ofFloat_one :
    ExecFloat.Binary.toReal (ExecFloat.Binary.ofFloat (1 : Float) : Binary64) = 1 := by
  have h : ExecFloat.Binary.toModel (ExecFloat.Binary.ofFloat (1 : Float) : Binary64) =
      ExecFloat.Binary.toModel (1 : Binary64) := by
    rw [show (1 : Float) = Float.ofNat 1 from rfl, ExecFloat.Binary.toModel_ofFloat_ofNat]
    exact (ExecFloat.Binary.toModel_ofModel _).symm
  rw [ExecFloat.Binary.toReal, h]
  exact ExecFloat.Binary.toReal_one

end TorchLean.Kernel
