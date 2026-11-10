/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN

/-!
# Lyapunov certificate semantics

A Lyapunov certificate stores a box region and interval bounds on a candidate function `V` and
its orbital derivative `V̇`. The proposition `Cert.ValidFor` says that both bounds hold at
every point of the box, and bounds reported by a producer become a theorem only through it. A
checked workflow proves it from a verified graph evaluation or certificate checker, while an
external workflow states it as an assumption at its own trust boundary.

References:
- Lyapunov stability is the classical certificate pattern: prove `V > 0` and `Vdot < 0` on a
  region.
- The numeric bounds come from CROWN-style affine/interval propagation, as in Zhang et al.
  (CROWN, NeurIPS 2018) and Xu et al. (auto_LiRPA/α-CROWN).
-/

@[expose] public section

namespace TorchLean.Examples.Lyapunov

open CROWN

/-- Certificate for Lyapunov verification over a boxed region. -/
structure Cert (α : Type) [Storage α] [Context α] (n : Nat) where
  /-- Region on which the bounds are claimed. -/
  region : Box α (.dim n .scalar)
  /-- Lower bound for `V`. -/
  vLower : α
  /-- Upper bound for `V`. -/
  vUpper : α
  /-- Lower bound for `V̇`. -/
  derivativeLower : α
  /-- Upper bound for `V̇`. -/
  derivativeUpper : α

/-- A neural Lyapunov candidate: a scalar function `V` on states and its orbital derivative `V̇`.

The application supplies `V̇`, and nothing in this file derives it from dynamics. -/
structure Candidate (α : Type) [Storage α] [Context α] (n : Nat) where
  /-- Candidate Lyapunov scalar field. -/
  value : Tensor α [n] → α
  /-- Orbital derivative or decay witness associated with `V`. -/
  orbitalDerivative : Tensor α [n] → α

variable {α : Type} [Storage α] [Context α] {n : Nat}

/--
`cert.ValidFor lyap` holds when the bounds in `cert` enclose `lyap.value` and
`lyap.orbitalDerivative` at every point of `cert.region`.

A semantic certificate checker produces this proof. Parsing a certificate or checking the signs of
its endpoints cannot build it, because both fields are enclosure proofs about the functions in
`lyap` on the region stored in `cert`.
-/
structure Cert.ValidFor (cert : Cert α n) (lyap : Candidate α n) : Prop where
  /-- On the region, `V` lies between `vLower` and `vUpper`. -/
  value_bounds : ∀ x, Box.Contains cert.region x →
    cert.vLower ≤ lyap.value x ∧ lyap.value x ≤ cert.vUpper
  /-- On the region, `V̇` lies between `derivativeLower` and `derivativeUpper`. -/
  orbitalDerivative_bounds : ∀ x, Box.Contains cert.region x →
    cert.derivativeLower ≤ lyap.orbitalDerivative x ∧
      lyap.orbitalDerivative x ≤ cert.derivativeUpper

end TorchLean.Examples.Lyapunov
