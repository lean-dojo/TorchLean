/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Lean.LawfulMonad
import all Init.System.ST
import all Init.System.IO

/-!
# Return values of `IO` actions

`MonadAttach.CanReturn x a` says that some run of the action `x` returns `a`. Lean core proves
`LawfulMonadAttach.canReturn_bind_imp'` (a value of `x >>= f` comes from a value of `x`) and
`LawfulMonadAttach.eq_of_canReturn_pure` for every monad with a `LawfulMonadAttach` instance, but
gives that instance to `EStateM` only. This module gives it to `EST`, and through `EST` to `EIO`
and `IO`, so statements about what an `IO` reader returns can be proved from its code.
`TorchLean.EIO.not_canReturn_throw` adds that `throw` returns no value.

Primitives such as `IO.FS.Handle.read` are opaque, so these facts say nothing about which values
they return, only how a program's result depends on them.
-/

public section

namespace TorchLean

instance {ε σ : Type} : WeaklyLawfulMonadAttach (EST ε σ) where
  map_attach {α} {x} := by
    funext s
    show EST.bind (MonadAttach.attach x) (fun a => EST.pure a.val) s = x s
    simp only [EST.bind, EST.pure, MonadAttach.attach]
    split
    · next a s' h =>
      split at h
      · next a₀ s₀ hx =>
        injection h with ha hs
        subst hs
        cases ha
        exact hx.symm
      · cases h
    · next e s' h =>
      split at h
      · cases h
      · next e₀ s₀ hx =>
        injection h with he hs
        subst he
        subst hs
        exact hx.symm

instance {ε σ : Type} : LawfulMonadAttach (EST ε σ) where
  canReturn_map_imp {α P x a} h := by
    obtain ⟨s, s', h⟩ := h
    change EST.bind x (fun a => EST.pure a.val) s = _ at h
    simp only [EST.bind, EST.pure] at h
    split at h
    · next b s₁ _ =>
      injection h with ha
      exact ha ▸ b.property
    · cases h

instance {ε : Type} : WeaklyLawfulMonadAttach (EIO ε) :=
  inferInstanceAs (WeaklyLawfulMonadAttach (EST ε IO.RealWorld))

instance {ε : Type} : LawfulMonadAttach (EIO ε) :=
  inferInstanceAs (LawfulMonadAttach (EST ε IO.RealWorld))

/-- `throw e` returns no value. -/
theorem EIO.not_canReturn_throw {ε α : Type} {e : ε} {a : α} :
    ¬ MonadAttach.CanReturn (throw e : EIO ε α) a := by
  rintro ⟨s, s', h⟩
  change EST.Out.error e s = _ at h
  cases h

/-- In `EIO ε`, `throw e >>= f = throw e`. -/
@[simp] theorem EIO.throw_bind {ε α β : Type} {e : ε} {f : α → EIO ε β} :
    (throw e : EIO ε α) >>= f = throw e := by
  change EST.bind (EST.throw e) f = EST.throw e
  rfl

/-- `IO.ofExcept (.ok a) = pure a`. -/
@[simp] theorem IO.ofExcept_ok {ε α : Type} [ToString ε] {a : α} :
    IO.ofExcept (.ok a : Except ε α) = pure a := by
  unfold IO.ofExcept
  rfl

/-- `IO.ofExcept (.error e) = throw (IO.userError (toString e))`. -/
@[simp] theorem IO.ofExcept_error {ε α : Type} [ToString ε] {e : ε} :
    IO.ofExcept (.error e : Except ε α) = throw (IO.userError (toString e)) := by
  unfold IO.ofExcept
  rfl

universe u v

/-- Continuations that agree on every value `x` can return give the same bind:
`(∀ a, CanReturn x a → f a = g a) → x >>= f = x >>= g`. -/
theorem WeaklyLawfulMonadAttach.bind_congr {m : Type u → Type v} [Monad m] [LawfulMonad m]
    [MonadAttach m] [WeaklyLawfulMonadAttach m] {α β : Type u} {x : m α} {f g : α → m β}
    (h : ∀ a, MonadAttach.CanReturn x a → f a = g a) : x >>= f = x >>= g := by
  rw [← WeaklyLawfulMonadAttach.attach_bind_val (f := f),
    ← WeaklyLawfulMonadAttach.attach_bind_val (f := g)]
  exact _root_.bind_congr fun a => h a.val a.property

end TorchLean

end
