/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Linalg
public import NN.Spec.Core.TensorOps

/-!
# Reductions, flatten/unflatten, and shape-changing helpers

This module is where “shape-aware” operations live:

- `flattenSpec` / `unflattenSpec` (convert between a tensor and a flat vector of length
  `Shape.size`)
- broadcasting maps that change the output shape

Because shapes are indexed in types, many of these definitions necessarily carry equalities like
`Shape.size s = ...` under the hood.

Tip: when you need to transport a tensor across a proved shape equality, use:

- `Tensor.cast_shape` (defined in `NN/Spec/Core/Tensor/Core.lean`)

Prefer `abbrev`s in user-facing code so common shape equalities remain definitional rather than
requiring transport proofs.

PyTorch mental model:
- `flattenSpec` / `unflattenSpec` correspond to `torch.flatten` and `view`/`reshape` on a
  contiguous tensor.
- broadcasting (`broadcastTo` / `broadcastMapTo`) corresponds to `expand`/`broadcast_to` plus
  elementwise ops.
- reductions (`reduceSum`, `reduceMean`, `reduceVar`, `reduceMax`, and `reduceMin`) correspond to
  `sum`/`mean`/`var`/`amax`/`amin` along a chosen axis.

The difference is that our shapes live in types, so the spec definitions must be explicit about:
- what the target/output shape is,
- and why the axis is valid / reducible.

Naming note (sequence concatenation):
- This file defines `Spec.Tensor.concatSequenceSpec` for concatenating along the **time axis**
  (axis 0), producing a longer sequence.
- `NN.Spec.Core.Sequence` defines `Spec.concatSequenceSpec` for concatenating along the
  **feature axis** (inner axis) for same-length sequences.
  The names are intentionally similar, but they are different operations living in different
  namespaces (`Spec.Tensor` vs `Spec`).

References / analogies (shape intuition, not semantics):
- PyTorch `torch.flatten`: https://pytorch.org/docs/stable/generated/torch.flatten.html
- PyTorch `torch.Tensor.reshape`:
  https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
- PyTorch `torch.Tensor.view`: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- PyTorch `torch.Tensor.expand`: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html
- PyTorch `torch.sum`: https://pytorch.org/docs/stable/generated/torch.sum.html
- PyTorch `torch.mean`: https://pytorch.org/docs/stable/generated/torch.mean.html
-/

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

-- Shape‑changing operations

/-- Flatten a tensor into a 1‑D vector (length = `Shape.size s`).

The order is outermost‑dimension major (row‑major w.r.t. the shape tree).
For proofs, the key invariant is that the output length matches `Shape.size`.

Why this exists: a lot of shape-changing ops are easiest to specify as "flatten, then rebuild",
and this is also the bridge we use for some runtime interop where we want a plain sequence of
scalars (e.g. importing weights or serializing test vectors).
-/
def flattenSpec {α : Type} [Inhabited α] : ∀ {s : Shape}, Tensor α s → Tensor α (.dim (Shape.size
  s) .scalar)
| Shape.scalar, Tensor.scalar x =>
  Tensor.dim (fun i =>
    have _ : i.val < 1 := i.isLt
    if i.val = 0 then Tensor.scalar x else Tensor.scalar (Inhabited.default))
| Shape.dim n s', Tensor.dim f =>
  let _ := n * Shape.size s'
  Tensor.dim (fun i =>
    let outerIdx := i.val / (Shape.size s')
    let innerIdx := i.val % (Shape.size s')
    if h1 : outerIdx < n then
      if _ : innerIdx < (Shape.size s') then
        let innerTensor := flattenSpec (f ⟨outerIdx, h1⟩)
        match innerTensor with
        | Tensor.dim g =>
          if h3 : innerIdx < (Shape.size s') then
            g ⟨innerIdx, h3⟩
          else
            Tensor.scalar (Inhabited.default)
      else
        Tensor.scalar (Inhabited.default)
    else
      Tensor.scalar (Inhabited.default))

/-- Unflatten a 1‑D vector back into a tensor of a given shape.

PyTorch analogy: `flat.view(shape)` (assuming the element count matches).
This is the inverse of `flattenSpec` up to the ordering convention. -/
def unflattenSpec {α : Type} [Inhabited α] : ∀ (s : Shape), Tensor α (.dim (Shape.size s) .scalar)
  → Tensor α s
| Shape.scalar, Tensor.dim f =>
  -- `Shape.size Shape.scalar = 1`, so the input always has an element at index `0`.
  -- This keeps the definition simple and avoids extra proof obligations downstream.
  match f ⟨0, by simp [Shape.size]⟩ with
  | Tensor.scalar x => Tensor.scalar x
| Shape.dim n s', Tensor.dim f =>
  Tensor.dim (fun i =>
    -- For each position i in the outer dimension, extract a sub-tensor
    let startIdx := i.val * (Shape.size s')
    let subTensor : Tensor α (.dim (Shape.size s') .scalar) :=
      Tensor.dim (fun j =>
        let globalIdx := startIdx + j.val
        if h : globalIdx < n * (Shape.size s') then
          f ⟨globalIdx, h⟩
        else
          Tensor.scalar (Inhabited.default))
    unflattenSpec s' subTensor)

/-!
## `flattenSpec` / `unflattenSpec` round-trip lemmas

These are shape-transport facts: they justify treating `flattenSpec`/`unflattenSpec` like
`reshape`/`view` in PyTorch, provided you keep the element count consistent.

PyTorch references:
- `torch.flatten`: https://pytorch.org/docs/stable/generated/torch.flatten.html
- `Tensor.view` / `torch.reshape`: https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- `torch.reshape`: https://pytorch.org/docs/stable/generated/torch.reshape.html

Important nuance:
- PyTorch allows zero-sized dimensions, and its reshape/flatten semantics remain total.
- Our spec definitions are also total (they use `Inhabited.default` for unreachable branches),
  which keeps everything executable, but can make “inverse” proofs a bit index-heavy.
  The theorems below show that the round-trips do work for the spec definitions as written.
-/

namespace Private

/--
Helper lemma: `flattenSpec` on an outer `Tensor.dim` agrees with flattening a chosen slice.

This is used to prove `unflattenSpec s (flattenSpec t) = t` by reducing the statement to the
induction hypothesis on each slice.
-/
private lemma flattenSpec_dim_apply {α : Type} [Inhabited α] {n : Nat} {s : Shape}
    (f : Fin n → Tensor α s) (i : Fin n) (j : Fin (Shape.size s))
    (hmpos : 0 < Shape.size s)
    (hidx : i.val * Shape.size s + j.val < n * Shape.size s) :
    (match flattenSpec (Tensor.dim f) with
      | Tensor.dim g => g ⟨i.val * Shape.size s + j.val, hidx⟩) =
    (match flattenSpec (f i) with
      | Tensor.dim g => g j) := by
  have hdiv : (i.val * Shape.size s + j.val) / Shape.size s = i.val := by
    calc
      (i.val * Shape.size s + j.val) / Shape.size s
          = (Shape.size s * i.val + j.val) / Shape.size s := by
              simp [Nat.mul_comm]
      _ = i.val + j.val / Shape.size s := by
            simpa using (Nat.mul_add_div (m := Shape.size s) hmpos i.val j.val)
      _ = i.val := by
            simp [Nat.div_eq_of_lt j.isLt]
  have hmod : (i.val * Shape.size s + j.val) % Shape.size s = j.val :=
    Nat.mul_add_mod_of_lt (a := i.val) (b := Shape.size s) (c := j.val) j.isLt

  have houter : (i.val * Shape.size s + j.val) / Shape.size s < n := by
    simp [hdiv]
  have hinner : (i.val * Shape.size s + j.val) % Shape.size s < Shape.size s := by
    simp [hmod]

  have hfin_outer : (⟨(i.val * Shape.size s + j.val) / Shape.size s, houter⟩ : Fin n) = i := by
    apply Fin.ext
    simp [hdiv]
  have hfin_inner :
      (⟨(i.val * Shape.size s + j.val) % Shape.size s, hinner⟩ : Fin (Shape.size s)) = j := by
    apply Fin.ext
    simp [hmod]

  simp [flattenSpec, hdiv, hmod]

/-!
If a shape has `Shape.size s = 0`, then it contains **no scalar leaves** (it has a `0`-length
dimension somewhere). In that case, there is essentially only one possible tensor value of shape
`s` (up to definitional equality), because at the `0`-length dimension the indexing function has
domain `Fin 0`.

We use this as a “vacuity” lemma to avoid needing division/modulo arithmetic when `Shape.size s =
  0`.
-/
/-- If `Shape.size s = 0`, then any two tensors of shape `s` are equal (vacuity via `Fin 0`). -/
private theorem tensor_eq_of_size_zero {α : Type} :
    ∀ {s : Shape}, Shape.size s = 0 → (x y : Tensor α s) → x = y
  | .scalar, h, _x, _y => by
      simp [Shape.size] at h
  | .dim n s, h, x, y => by
      cases x with
      | dim fx =>
          cases y with
          | dim fy =>
              cases n with
              | zero =>
                  apply congrArg Tensor.dim
                  funext i
                  exact Fin.elim0 i
              | succ n =>
                  have hs0 : Shape.size s = 0 := by
                    have : (Nat.succ n = 0) ∨ (Shape.size s = 0) := Nat.mul_eq_zero.mp h
                    exact this.resolve_left (Nat.succ_ne_zero n)
                  apply congrArg Tensor.dim
                  funext i
                  exact tensor_eq_of_size_zero (α := α) (s := s) hs0 (fx i) (fy i)

end Private

/--
Round-trip `unflatten ∘ flatten = id`.

This is the spec-layer analogue of `reshape`/`view` round-tripping in PyTorch when the element
count matches.
-/
theorem flatten_unflatten_inverse {α : Type} [Inhabited α] :
    ∀ {s : Shape}, (t : Tensor α s) → unflattenSpec s (flattenSpec t) = t
  | .scalar, t => by
      cases t with
      | scalar x =>
          -- `simp` no longer unfolds `unflattenSpec` reliably on Lean 4.29, so we do the
          -- computation step-by-step.
          simp [flattenSpec, Shape.size]
          unfold unflattenSpec
          rfl
  | .dim n s, t => by
      cases t with
      | dim f =>
          cases hflat : flattenSpec (Tensor.dim f) with
          | dim flat =>
              simp [unflattenSpec]
              funext i
              by_cases hm : Shape.size s = 0
              ·
                exact
                  Private.tensor_eq_of_size_zero (α := α) (s := s) hm
                    (unflattenSpec s
                      (Tensor.dim (fun j : Fin (Shape.size s) =>
                        if hIdx : i.val * Shape.size s + j.val < n * Shape.size s then
                          flat ⟨i.val * Shape.size s + j.val, hIdx⟩
                        else
                          Tensor.scalar Inhabited.default)))
                    (f i)
              ·
                have hmpos : 0 < Shape.size s := Nat.pos_of_ne_zero hm
                have sub_eq :
                    (Tensor.dim (fun j : Fin (Shape.size s) =>
                      if hIdx : i.val * Shape.size s + j.val < n * Shape.size s then
                        flat ⟨i.val * Shape.size s + j.val, hIdx⟩
                      else
                        Tensor.scalar Inhabited.default))
                      = flattenSpec (f i) := by
                  cases hfi : flattenSpec (f i) with
                  | dim gfi =>
                      apply congrArg Tensor.dim
                      funext j
                      have hidx : i.val * Shape.size s + j.val < n * Shape.size s := by
                        have hisucc : i.val + 1 ≤ n := Nat.succ_le_of_lt i.isLt
                        have hlt :
                            i.val * Shape.size s + j.val < (i.val + 1) * Shape.size s := by
                          have := Nat.add_lt_add_left j.isLt (i.val * Shape.size s)
                          simp [Nat.succ_mul]
                        have hle : (i.val + 1) * Shape.size s ≤ n * Shape.size s :=
                          Nat.mul_le_mul_right (Shape.size s) hisucc
                        exact Nat.lt_of_lt_of_le hlt hle
                      simp [hidx]
                      have :=
                        Private.flattenSpec_dim_apply (α := α) (f := f) (i := i) (j := j)
                          (hmpos := hmpos) (hidx := hidx)
                      simpa [hflat, hfi] using this
                simpa [sub_eq] using
                  (flatten_unflatten_inverse (α := α) (s := s) (t := f i))

/--
Round-trip `flatten ∘ unflatten = id`.

This is the spec-layer analogue of flattening a reshaped/viewed tensor in PyTorch.
-/
theorem unflatten_flatten_inverse {α : Type} [Inhabited α] :
    ∀ {s : Shape}, (v : Tensor α (.dim (Shape.size s) .scalar)) → flattenSpec (unflattenSpec s v)
      = v
  | .scalar, v => by
      cases v with
      | dim f =>
          let idx0 : Fin Shape.scalar.size := ⟨0, by simp [Shape.size]⟩
          cases h0 : f idx0 with
          | scalar x =>
              have hunflat : unflattenSpec Shape.scalar (Tensor.dim f) = Tensor.scalar x := by
                simp [unflattenSpec, idx0, h0]
              rw [hunflat]
              simp [flattenSpec, Shape.size]
              funext i
              have hival : i.val = 0 := by
                have : i.val < 1 := by simp
                have : i.val ≤ 0 := by simp
                exact Nat.eq_zero_of_le_zero this
              have hi : i = idx0 := by
                apply Fin.ext
                simp [idx0]
              simp [hi, h0]
  | .dim n s, v => by
      cases v with
      | dim g =>
          by_cases hm : Shape.size s = 0
          ·
            cases hflat : flattenSpec (unflattenSpec (Shape.dim n s) (Tensor.dim g)) with
            | dim gf =>
                apply congrArg Tensor.dim
                funext idx
                have : False := by
                  have : idx.val < 0 := by simpa [Shape.size, hm] using idx.isLt
                  exact Nat.not_lt_zero _ this
                exact False.elim this
          ·
            let m : Nat := Shape.size s
            have hmpos : 0 < m := by
              have : m ≠ 0 := by simpa [m] using hm
              exact Nat.pos_of_ne_zero this
            have hunflat :
                unflattenSpec (Shape.dim n s) (Tensor.dim g) =
                  Tensor.dim (fun i : Fin n =>
                    let startIdx := i.val * m
                    let subTensor : Tensor α (.dim m .scalar) :=
                      Tensor.dim (fun j : Fin m =>
                        let globalIdx := startIdx + j.val
                        if h : globalIdx < n * m then
                          g ⟨globalIdx, h⟩
                        else
                          Tensor.scalar (Inhabited.default))
                    unflattenSpec s subTensor) := by
              rfl
            cases hflat : flattenSpec (unflattenSpec (Shape.dim n s) (Tensor.dim g)) with
            | dim gf =>
                have hflat' :
                    flattenSpec
                        (Tensor.dim (fun i : Fin n =>
                          let startIdx := i.val * m
                          let subTensor : Tensor α (.dim m .scalar) :=
                            Tensor.dim (fun j : Fin m =>
                              let globalIdx := startIdx + j.val
                              if h : globalIdx < n * m then
                                g ⟨globalIdx, h⟩
                              else
                                Tensor.scalar (Inhabited.default))
                          unflattenSpec s subTensor))
                      = Tensor.dim gf := by
                  simpa [hunflat] using hflat
                apply congrArg Tensor.dim
                funext idx
                let oi : Nat := idx.val / m
                let ij : Nat := idx.val % m
                have hoi : oi < n := by
                  have : idx.val < n * m := idx.isLt
                  exact (Nat.div_lt_iff_lt_mul hmpos).2 (by simpa [oi] using this)
                have hij : ij < m := by
                  simpa [ij] using Nat.mod_lt idx.val hmpos
                let i : Fin n := ⟨oi, hoi⟩
                let j : Fin m := ⟨ij, hij⟩
                have hrecomp : i.val * m + j.val = idx.val := by
                  simp [i, j, oi, ij, Nat.div_add_mod']
                have hidx : i.val * m + j.val < n * m :=
                  lt_of_eq_of_lt hrecomp idx.isLt
                have hfin : (⟨i.val * m + j.val, hidx⟩ : Fin (n * m)) = idx := by
                  apply Fin.ext
                  simp [hrecomp]

                have hcoord :=
                  Private.flattenSpec_dim_apply
                    (α := α)
                    (f := fun i : Fin n =>
                      let startIdx := i.val * m
                      let subTensor : Tensor α (.dim m .scalar) :=
                        Tensor.dim (fun j : Fin m =>
                          let globalIdx := startIdx + j.val
                          if h : globalIdx < n * m then
                            g ⟨globalIdx, h⟩
                          else
                            Tensor.scalar (Inhabited.default))
                      unflattenSpec s subTensor)
                    (i := i) (j := j) (hmpos := hmpos) (hidx := hidx)

                have hsub :
                    flattenSpec
                        (unflattenSpec s
                          (Tensor.dim (fun j : Fin m =>
                            let globalIdx := i.val * m + j.val
                            if h : globalIdx < n * m then
                              g ⟨globalIdx, h⟩
                            else
                              Tensor.scalar (Inhabited.default))))
                      =
                    Tensor.dim (fun j : Fin m =>
                      let globalIdx := i.val * m + j.val
                      if h : globalIdx < n * m then
                        g ⟨globalIdx, h⟩
                      else
                        Tensor.scalar (Inhabited.default)) := by
                  simpa [i, unflattenSpec] using
                    (unflatten_flatten_inverse (α := α) (s := s)
                      (v := Tensor.dim (fun j : Fin m =>
                        let globalIdx := i.val * m + j.val
                        if h : globalIdx < n * m then
                          g ⟨globalIdx, h⟩
                        else
                          Tensor.scalar (Inhabited.default))))

                have hcoord' :
                    gf ⟨i.val * m + j.val, hidx⟩ =
                      (match flattenSpec
                          (unflattenSpec s
                            (Tensor.dim (fun j : Fin m =>
                              let globalIdx := i.val * m + j.val
                              if h : globalIdx < n * m then
                                g ⟨globalIdx, h⟩
                              else
                                Tensor.scalar (Inhabited.default))))
                        with
                        | Tensor.dim g' => g' j) := by
                  simpa [hflat'] using hcoord

                have hgf0 : gf ⟨i.val * m + j.val, hidx⟩ = g ⟨i.val * m + j.val, hidx⟩ := by
                  simpa [hsub, hidx] using hcoord'
                have hgf' : gf ⟨i.val * m + j.val, hidx⟩ = g idx := by
                  exact hgf0.trans (congrArg g hfin)

                calc
                  gf idx = gf ⟨i.val * m + j.val, hidx⟩ := by
                    exact (congrArg gf hfin).symm
                  _ = g idx := hgf'

/--
Convenience corollary: the `unflatten ∘ flatten` round-trip in the common well-formed regime.
-/
theorem flatten_unflatten_inverse_wf {α : Type} [Inhabited α] {s : Shape}
    [Shape.WellFormed s] (t : Tensor α s) :
    unflattenSpec s (flattenSpec t) = t := by
  simpa using (flatten_unflatten_inverse (α := α) (s := s) (t := t))

/-- Reshape a tensor, given a proof that the number of elements matches. -/
def reshapeSpec {α : Type} [Inhabited α]
  {s₁ s₂ : Shape} (t : Tensor α s₁) (h : s₁.size = s₂.size) : Tensor α s₂ :=
  let flattened := flattenSpec t
  let retyped : Tensor α (.dim (Shape.size s₂) .scalar) :=
    Eq.recOn h flattened
  unflattenSpec s₂ retyped

/-- Reshape with an explicit equality rewrite (sometimes easier for the elaborator). -/
def reshapeExplicitSpec {α : Type} [Inhabited α] {s₁ s₂ : Shape} (t : Tensor α s₁)
  (h : s₁.size = s₂.size) : Tensor α s₂ :=
  let flattened := flattenSpec t
  let retyped : Tensor α (.dim (Shape.size s₂) .scalar) :=
    by rw [h.symm]; exact flattened
  unflattenSpec s₂ retyped

/-- Given a partial function `Fin n → Option (Tensor α s)`, build a tensor if all succeed. -/
def sequenceFin {s : Shape} {n : Nat}
  (f : Fin n → Option (Tensor α s)) : Option (Tensor α (.dim n s)) :=
  -- This is basically `Option`-sequencing for `Fin n → _`.
  -- We use it when a shape-level construction can fail (e.g. dynamic runtime checks),
  -- but we still want a *total* spec API (failure is explicit in the `Option`).
  --
  -- Implementation note: we avoid `Array.get!` / `arr[i]!` by building the result function
  -- directly via recursion on `n` (using `Fin.cases`).
  match n with
  | 0 =>
      -- A 0-length `Tensor.dim` is inhabited by the empty function.
      some (Tensor.dim (fun i => nomatch i))
  | n' + 1 =>
      match f ⟨0, Nat.succ_pos n'⟩ with
      | none => none
      | some t0 =>
          match sequenceFin (n := n') (fun j => f j.succ) with
          | none => none
          | some (Tensor.dim g) =>
              some (Tensor.dim (fun i => Fin.cases t0 g i))
              -- Note: the `Tensor.scalar` case is impossible since the shape is `.dim _ _`.

/-- Build a tensor filled with a constant, without using `fill` (used in broadcasts). -/
def broadcastFill {α : Type} [Inhabited α] : ∀ (s : Shape), α → Tensor α s
| .scalar, v => scalar v
| Shape.dim _ s', v => dim (fun _ => broadcastFill s' v)

/-! ## Broadcasting -/

/-
Broadcasting in the spec layer is defined in terms of `Shape.CanBroadcastTo`:

- you pick an explicit target shape `t`,
- and provide evidence that each operand can broadcast to `t`.

TorchLean standardizes on this explicit target style throughout core.

We intentionally do *not* provide a second "implicit" broadcasting API that tries to infer a common
output shape from two operands, because that would split the codebase into two parallel styles.
Instead, core code names the output shape and carries broadcast evidence explicitly (often inferred
by typeclass search via `Shape.BroadcastTo`).

This choice also matches the backward pass semantics: broadcasting duplicates values, so the adjoint is
a sum-reduction along the broadcasted axes (see `reduceFromBroadcastTo` below).
-/
/-- Broadcast a tensor along a `Shape.CanBroadcastTo` proof (spec-level analogue of
  `torch.broadcast_to`). -/
def broadcastTo {α : Type} [Inhabited α] :
  {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Tensor α s₁ → Tensor α s₂
| _, _, Shape.CanBroadcastTo.scalar_to_any _, t =>
    replicate t
| _, _, Shape.CanBroadcastTo.dim_eq tail, Tensor.dim xs =>
    Tensor.dim (fun i => broadcastTo tail (xs i))
| _, _, Shape.CanBroadcastTo.dim_1_to_n tail, Tensor.dim xs =>
    Tensor.dim (fun _ => broadcastTo tail (xs 0))
| _, _, Shape.CanBroadcastTo.expand_dims tail, t =>
    Tensor.dim (fun _ => broadcastTo tail t)

/-! ## Broadcasted maps -/

/--
Broadcast a scalar tensor to match a template tensor's shape.

This is a small convenience wrapper used by specs that want "like" broadcasting without spelling
out the `Shape.CanBroadcastTo` evidence.
-/
def broadcastLike {α : Type} [Inhabited α]
  {s : Shape} (_template : Tensor α s) (t : Tensor α .scalar) : Tensor α s :=
  replicate t

/-- Helper: map a scalar on the left over any tensor shape. -/
def mapScalarLeft {α : Type} (f : α → α → α) (x : α) :
  ∀ {s : Shape}, Tensor α s → Tensor α s
| _, Tensor.scalar y => Tensor.scalar (f x y)
| _, Tensor.dim g => Tensor.dim (fun i => mapScalarLeft f x (g i))

/-- Helper: map a scalar on the right over any tensor shape. -/
def mapScalarRight {α : Type} (f : α → α → α) (y : α) :
  ∀ {s : Shape}, Tensor α s → Tensor α s
| _, Tensor.scalar x => Tensor.scalar (f x y)
| _, Tensor.dim g => Tensor.dim (fun i => mapScalarRight f y (g i))

/--
Binary element-wise operation with broadcasting to an explicit target shape.

This is the helper you typically want in spec code:
- pick the output shape `t`,
- broadcast each operand to `t`,
- then `map2_spec` the pointwise operation.

PyTorch analogy: `f(x, y)` where `x` and/or `y` are broadcastable to a common shape.
We make the common shape explicit instead of "discovering" it, because at the spec layer we want:
- predictable typing,
- a single source of truth for what the output shape is.
-/
def broadcastMapTo {α} [Inhabited α] (f : α → α → α)
    {s₁ s₂ t : Shape} (cbx : Shape.CanBroadcastTo s₁ t) (cby : Shape.CanBroadcastTo s₂ t) :
    Tensor α s₁ → Tensor α s₂ → Tensor α t :=
  fun x y => map2Spec f (broadcastTo cbx x) (broadcastTo cby y)


/-! ## Reductions -/

/-- Left fold over all tensor elements. -/
def tensorFoldlSpec {α β : Type} (f : β → α → β) (init : β) : ∀ {s : Shape}, Tensor α s → β
  | Shape.scalar, Tensor.scalar value => f init value
  | Shape.dim n s, Tensor.dim values =>
    let rec go (i : Nat) (acc : β) : β :=
      if h : i < n then
        go (i + 1) (tensorFoldlSpec f acc (values ⟨i, h⟩))
      else acc
    go 0 init

/-- Right fold over all tensor elements. -/
def tensorFoldrSpec {α β : Type} (f : α → β → β) (init : β) : ∀ {s : Shape}, Tensor α s → β
  | Shape.scalar, Tensor.scalar value => f value init
  | Shape.dim n s, Tensor.dim values =>
    let rec go (i : Nat) (acc : β) : β :=
      if h : i < n then
        if h_last : (n - 1 - i) < n then
          let idx := ⟨n - 1 - i, h_last⟩
          go (i + 1) (tensorFoldrSpec f acc (values idx))
        else acc
      else acc
    go 0 init

-- Tensor reduction operations
/-- Sum all elements of a tensor. -/
def sumSpec {α : Type} [Add α] [Zero α] {s : Shape} (t : Tensor α s) : α :=
  tensorFoldlSpec (· + ·) 0 t

/-- Product of all elements of a tensor. -/
def prodSpec {s : Shape} (t : Tensor α s) : α :=
  tensorFoldlSpec (· * ·) 1 t

/-- Short name for `prodSpec`. -/
abbrev productSpec {s : Shape} (t : Tensor α s) : α :=
  prodSpec t

/-- Count the number of scalar entries in a tensor (= `Shape.size`). -/
def countSpec {s : Shape} (t : Tensor α s) : Nat :=
  tensorFoldlSpec (fun acc _ => acc + 1) 0 t

/-- `true` if any entry satisfies `p`. -/
def anySpec {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  tensorFoldlSpec (fun acc x => acc || p x) false t

/-- `true` if all entries satisfy `p`. -/
def allSpec {s : Shape} (p : α → Bool) (t : Tensor α s) : Bool :=
  tensorFoldlSpec (fun acc x => acc && p x) true t

/-- Dot product: `sum (a ⊙ b)`. -/
def dotSpec {s : Shape} (a b : Tensor α s) : α :=
  sumSpec (mulSpec a b)

-- Statistical operations
/-- Mean of all elements (treats nested dims as one big collection). -/
def meanSpec : ∀ {s : Shape}, Tensor α s → α
  | .scalar, Tensor.scalar value => value
  | .dim n _, Tensor.dim values =>
      let sum := (List.finRange n).foldl (fun acc i => acc + meanSpec (values i)) 0
      sum / ↑n

/-- Variance of all elements (population variance, divides by `n`). -/
def varianceSpec : ∀ {s : Shape}, Tensor α s → α
  | .scalar, Tensor.scalar _ => 0
  | .dim n _, Tensor.dim values =>
      let m := meanSpec (Tensor.dim values)
      let sum_sq_diff := (List.finRange n).foldl (fun acc i =>
        let diff := meanSpec (values i) - m
        acc + diff * diff) 0
      sum_sq_diff / ↑n

-- Shape update rule
/-- Output shape after summing along `axis` (drops that dimension). -/
def shapeAfterSum : Shape → Nat → Shape
  | .scalar, _ => .scalar
  | .dim _ inner, 0 => inner
  | .dim n inner, Nat.succ k => .dim n (shapeAfterSum inner k)

/-- `simp` lemma: dropping axis 1 from a 2D `(nQ+1)×(nK+1)` shape yields `(nQ+1)`. -/
@[simp]
theorem shape_after_sum_dim_1 (nQ nK : Nat) :
  shapeAfterSum (Shape.dim (nQ + 1) (Shape.dim (nK + 1) Shape.scalar)) 1 =
    Shape.dim (nQ + 1) Shape.scalar := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 1 from a 2D `nQ×nK` shape yields `nQ`. -/
@[simp]
theorem shape_after_sum_dim_1_alt (nQ nK : Nat) :
  shapeAfterSum (Shape.dim nQ (Shape.dim nK Shape.scalar)) 1 =
    Shape.dim nQ Shape.scalar := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 3 from a 4D `b×h×w×c` shape yields `b×h×w`. -/
@[simp]
theorem shape_after_sum_dim_3_alt (b h w c : Nat) :
  shapeAfterSum (Shape.dim b (Shape.dim h (Shape.dim w (Shape.dim c Shape.scalar)))) 3 =
    Shape.dim b (Shape.dim h (Shape.dim w Shape.scalar)) := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from a positive `.dim (n+1) s` yields `s`. -/
@[simp]
theorem shape_after_sum_zero {n s} :
  shapeAfterSum (.dim (n+1) s) 0 = s := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis `k+1` recurses into the tail shape. -/
@[simp]
theorem shape_after_sum_succ {n s k} :
  shapeAfterSum (.dim (n+1) s) (k+1) = .dim (n+1) (shapeAfterSum s k) := by
  simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from a 2D `(kH+1)×(kW+1)` yields `(kW+1)`. -/
@[simp]
theorem shape_after_sum_twice_zero {kH kW : Nat} :
  shapeAfterSum (Shape.dim (kH + 1) (Shape.dim (kW + 1) Shape.scalar)) 0
    = .dim (kW + 1) Shape.scalar := by simp [shapeAfterSum]

/-- `simp` lemma: dropping axis 0 from `.dim n inner` yields `inner` (even when `n=0`). -/
@[simp]
theorem shape_after_sum_zero_alt
  (n : Nat) (inner : Shape) :
  shapeAfterSum (.dim n inner) 0 = inner := by
  simp [shapeAfterSum]

-- Helper function for reflexivity
/-- Reflexive broadcast proof (`s` can broadcast to itself). -/
def canBroadcastToRefl (s : Shape) : Shape.CanBroadcastTo s s :=
  match s with
  | .scalar => Shape.CanBroadcastTo.scalar_to_any .scalar
  | .dim _ inner => Shape.CanBroadcastTo.dim_eq (canBroadcastToRefl inner)

/-- Build a broadcast proof from the reduced shape back to the original shape.

We use this when a backward pass computes something in the reduced shape (e.g. a mean/variance) and
we need to broadcast it back to match the original tensor shape.
-/
def shapeAfterSumBroadcastBack
  {s : Shape} (dim : Nat)
  (valid : Shape.valid_axis_inst dim s)
  (wf : Shape.WellFormed s) :
  Shape.CanBroadcastTo (shapeAfterSum s dim) s :=
match s, dim with
| .scalar, _ =>
  -- For scalar, shape_after_sum returns scalar
  Shape.CanBroadcastTo.scalar_to_any .scalar
| .dim n inner, 0 =>
  -- When dim = 0, shape_after_sum (.dim n inner) 0 = inner
  -- We need CanBroadcastTo inner (.dim n inner)
  Shape.CanBroadcastTo.expand_dims (canBroadcastToRefl inner)
| .dim n inner, Nat.succ k =>
  -- When dim = k+1, shape_after_sum (.dim n inner) (k+1) = .dim n (shape_after_sum inner k)
  -- We need CanBroadcastTo (.dim n (shape_after_sum inner k)) (.dim n inner)
  let valid_inner : Shape.valid_axis_inst k inner := by
    cases valid.proof with
    | valid_succ h => exact ⟨h⟩

  let inner_wf : Shape.WellFormed inner := ⟨wf.proof.right⟩

  Shape.CanBroadcastTo.dim_eq (shapeAfterSumBroadcastBack k valid_inner inner_wf)

-- The compact proof below uses the product-shape lemmas already established above.


-- Generic reducers

/-- Reduce a tensor of shape `(n, innerShape)` by applying `f` across the first axis.

This is the basic “reduce over axis 0” primitive that we reuse to implement broadcast-adjoints and
multi-axis reducers.
-/
def reduceFirstDim {α : Type} {innerShape : Shape} {n : Nat}
    (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
    (t : Tensor α (.dim n innerShape)) : Tensor α innerShape :=
    match innerShape with
    | .scalar =>
        match t with
        | .dim slices =>
            let collected := .dim (fun i => slices i)
            .scalar (f collected)
    | .dim _ _ =>
        match t with
        | .dim slices =>
            .dim (fun j =>
              let slice_at_j := .dim (fun i => sliceSpec (slices i) j)
              reduceFirstDim f slice_at_j)

/-!
Reduce a gradient from a broadcast target shape back to the original input shape.

This is the adjoint of `broadcastTo` for sum-reduction: broadcast duplicates values, so the
backward pass sums contributions across broadcasted dimensions.

PyTorch analogy: this is the logic behind "sum over broadcasted dimensions" that happens in
autograd for `expand` + elementwise ops.
-/
/-- Adjoint of `broadcastTo` under sum-reduction: collapse broadcasted dimensions by summing. -/
def reduceFromBroadcastTo {α : Type} [Add α] [Zero α] :
  {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Tensor α s₂ → Tensor α s₁
| .scalar, s₂, Shape.CanBroadcastTo.scalar_to_any .(s₂), t =>
    Tensor.scalar (sumSpec (α := α) (s := s₂) t)
| .dim n s₁, .dim .(n) s₂, Shape.CanBroadcastTo.dim_eq tail, Tensor.dim xs =>
    Tensor.dim (fun i => reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail (xs i))
| .dim 1 s₁, .dim n s₂, Shape.CanBroadcastTo.dim_1_to_n tail, t =>
    match t with
    | Tensor.dim xs =>
        let summed : Tensor α s₂ :=
          reduceFirstDim (α := α) (innerShape := s₂) (n := n)
            (fun {sliceShape} => sumSpec (α := α) (s := sliceShape)) (Tensor.dim xs)
        let reduced : Tensor α s₁ := reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail summed
        Tensor.dim (fun _ => reduced)
| s₁, .dim n s₂, Shape.CanBroadcastTo.expand_dims tail, t =>
    match t with
    | Tensor.dim xs =>
        let summed : Tensor α s₂ :=
          reduceFirstDim (α := α) (innerShape := s₂) (n := n)
            (fun {sliceShape} => sumSpec (α := α) (s := sliceShape)) (Tensor.dim xs)
        reduceFromBroadcastTo (s₁ := s₁) (s₂ := s₂) tail summed

/-- Generic reduction along a (provably reducible) axis.

`reduce_dim f axis x` applies `f` to the slices along `axis`, and returns a tensor whose shape is
`shape_after_sum s axis` (i.e. that axis is dropped).
-/
def reduceDim
  {α : Type}
  {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (axis : Nat)
  (x : Tensor α s)
  (_h : Shape.reducibleAlong axis s) : Tensor α (shapeAfterSum s axis) :=

  -- Design note:
  -- We implement `reduce_dim` by recursing down the shape tree until we hit the axis,
  -- then using `reduce_first_dim` at that level. This mirrors how you would implement
  -- `torch.sum(x, dim=axis)` via indexing/slicing, but keeps everything total and
  -- shape-correct by construction.
  let rec aux
    {inShape outShape : Shape} (axisAdjusted : Nat)
    (h_eq : outShape = shapeAfterSum inShape axisAdjusted)
    (t : Tensor α inShape) : Tensor α outShape :=

    match inShape, axisAdjusted with
    | .scalar, _ =>
      cast (congrArg (Tensor α) h_eq.symm) t

    | .dim n innerIn, 0 =>
      let reduced := reduceFirstDim f t
      cast (congrArg (Tensor α) h_eq.symm) reduced

    | .dim n innerIn, Nat.succ k =>
      let innerOut := shapeAfterSum innerIn k
      let recFun : Fin n → Tensor α innerOut := fun i =>
        aux k (by rfl) (getAtSpec t i)
      Tensor.dim recFun |> cast (congrArg (Tensor α) h_eq.symm)

  aux axis (by rfl) x

-- Reduce sum along axis
/-- Sum-reduction along a given axis. -/
def reduceSum {α : Type} [Add α] [Zero α] {s : Shape} (axis : Nat) (t : Tensor α s) (h :
  Shape.reducibleAlong axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim sumSpec axis t h

-- Need to add instance for proof, something like have _ : Shape.valid_axis_inst 0 (Shape.dim p
-- Shape.scalar) :=
--    Shape.validAxisInstZeroAlt h
/-- Sum-reduction along `axis`, with axis validity inferred via `valid_axis_inst`. -/
def reduceSumAuto {α : Type} [Add α] [Zero α] {s : Shape} (axis : Nat) [h : Shape.valid_axis_inst
  axis s] (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceSum axis t (Shape.proveReducibleAlong axis s h.proof)

-- Reduce product along axis
/-- Product-reduction along a given axis. -/
def reduceProd {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceDim prodSpec axis t h

/-- Product-reduction along `axis` when you already have a `valid_axis` proof. -/
def reduceProdAuto {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.valid_axis axis s) :
    Tensor α (shapeAfterSum s axis) :=
  reduceProd axis t (Shape.proveReducibleAlong axis s h)

/-- Get the runtime size of the `k`-th dimension (0-based), if it exists. -/
def getDimSize : Shape → Nat → Option Nat
  | .scalar, _ => none
  | .dim n _, 0 => some n
  | .dim _ inner, k+1 => getDimSize inner k

/-- Mean-reduction along a given axis. -/
def reduceMean {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => t
  | .dim n inner =>
    match axis with
    | 0 =>
      let summed := reduceSum 0 t h
      mapSpec (fun x => x / (n : α)) summed
    | Nat.succ k =>
      let summed := reduceSum (Nat.succ k) t h
      -- When reducing along an *inner* axis, divide by the size of the axis being reduced,
      -- not the size of the output shape's leading dimension.
      --
      -- Example (2D): reducing axis=1 on shape `(seqLen, embedDim)` must divide by `embedDim`,
      -- but `shape_after_sum inner 0 = scalar` has `dim_size = 1`.
      --
      -- PyTorch analogy: `torch.mean(x, dim=k)` divides by the length of that `dim`, even when
      -- you reduce an inner axis of a higher-rank tensor.
      let denomNat :=
        match getDimSize inner k with
        | some m => m
        | none => 1
      mapSpec (fun x => x / (denomNat : α)) summed

/-- Mean-reduction along `axis`, with axis validity provided as a typeclass argument. -/
def reduceMeanAuto {s : Shape} (axis : Nat) (h : Shape.valid_axis_inst axis s) (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceMean axis t (Shape.proveReducibleAlong axis s h.proof)

-- Reduce sum of squares (for variance)
/-- Sum of squares reduced along an axis (helper for variance). -/
def reduceSumSquared {n s} (axis : Nat) (t : Tensor α (.dim n s)) (h : Shape.reducibleAlong axis
  (.dim n s)) :
    Tensor α (shapeAfterSum (.dim n s) axis) :=
  reduceSum axis (mapSpec (fun x => x * x) t) h

/-- Variance-reduction along a given axis (population variance, divides by `n`). -/
def reduceVar
  {s : Shape} (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar =>
    mapSpec (fun _ => 0) t
  | .dim n inner =>
    match axis with
    | 0 =>
      -- Reducing along the first axis
      -- Compute E[X²] - E[X]² directly without broadcasting
      --
      -- PyTorch analogy: `torch.var(x, dim=0, unbiased=False)` (population variance).
      let mean := reduceMean 0 t h
      let mean_squared := mapSpec (fun x => x * x) mean

      -- Compute E[X²] by first squaring, then taking mean
      let squares := mapSpec (fun x => x * x) t
      let mean_of_squares := reduceMean 0 squares h

      -- Variance = E[X²] - E[X]²
      subSpec mean_of_squares mean_squared

    | Nat.succ k =>
      -- Reducing along axis k+1 in the inner dimensions
      -- Apply reduce_var recursively to each slice along the first dimension
      match t with
      | Tensor.dim f =>
        -- Extract the proof that inner is reducible along axis k
        let inner_reducible : Shape.reducibleAlong k inner := by
          -- We know h : Shape.reducibleAlong (k + 1) (Shape.dim n inner)
          -- This means reducibleAlong.tail (reducibleAlong k inner)
          -- So we can extract the inner proof
          cases h with
          | tail inner_h => exact inner_h

        -- For each slice along the first dimension, compute variance along axis k
        let variance_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceVar k (f i) inner_reducible
        Tensor.dim variance_slices

/-- Variance-reduction along `axis`, with axis validity provided as a typeclass argument. -/
def reduceVarAuto {s : Shape} (axis : Nat) (h : Shape.valid_axis_inst axis s) (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceVar axis t (Shape.proveReducibleAlong axis s h.proof)

/-- Min-reduction along a given axis. -/
def reduceMin {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar =>
    -- Min of a single value is the value itself
    t

  | .dim n inner =>
    match axis with
    | 0 =>
      -- Reducing along the first axis - find min across the n slices
      --
      -- PyTorch analogy: `torch.amin(x, dim=0)` (or `torch.min` along a dim).
      match n, t with
      | 0, _ => nomatch h
      | Nat.succ n', Tensor.dim f =>
        -- We have at least one element, so we can safely reduce
        let rec loop (i : Nat) (acc : Tensor α inner) (hi : i ≤ n') : Tensor α inner :=
          if h_lt : i < n' then
            let next_idx : Fin (Nat.succ n') := ⟨i + 1, Nat.succ_lt_succ h_lt⟩
            loop (i + 1) (minSpec acc (f next_idx)) (Nat.le_of_succ_le_succ (Nat.succ_le_of_lt
              (Nat.succ_lt_succ h_lt)))
          else
            acc
        -- Start with first element (index 0) and loop through the rest
        let first_idx : Fin (Nat.succ n') := ⟨0, Nat.succ_pos n'⟩
        loop 0 (f first_idx) (Nat.zero_le n')

    | Nat.succ k =>
      -- Reducing along axis k+1 in the inner dimensions
      match t with
      | Tensor.dim f =>
        -- Extract the proof that inner is reducible along axis k
        let inner_reducible : Shape.reducibleAlong k inner := by
          cases h with
          | tail inner_h => exact inner_h

        -- For each slice along the first dimension, compute min along axis k
        let min_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMin k (f i) inner_reducible
        Tensor.dim min_slices

/-- Max-reduction along a given axis. -/
def reduceMax {s : Shape}
  (axis : Nat) (t : Tensor α s) (h : Shape.reducibleAlong axis s) :
  Tensor α (shapeAfterSum s axis) :=
  match s with
  | .scalar => t
  | .dim n inner =>
    match axis with
    | 0 =>
      -- PyTorch analogy: `torch.amax(x, dim=0)`.
      match n, t with
      | 0, _ => nomatch h
      | Nat.succ n', Tensor.dim f =>
        let rec loop (i : Nat) (acc : Tensor α inner) : Tensor α inner :=
          if h_lt : i < n' then
            let next_idx : Fin (Nat.succ n') := ⟨i + 1, Nat.succ_lt_succ h_lt⟩
            loop (i + 1) (maxSpec acc (f next_idx))
          else
            acc
        let first_idx : Fin (Nat.succ n') := ⟨0, Nat.succ_pos n'⟩
        loop 0 (f first_idx)
    | Nat.succ k =>
      match t with
      | Tensor.dim f =>
        let inner_reducible : Shape.reducibleAlong k inner := by
          cases h with
          | tail inner_h => exact inner_h
        let max_slices : Fin n → Tensor α (shapeAfterSum inner k) :=
          fun i => reduceMax k (f i) inner_reducible
        Tensor.dim max_slices

/-- Max-reduction along `axis`, with axis validity inferred via `valid_axis_inst`. -/
def reduceMaxAuto {s : Shape} (axis : Nat) [h : Shape.valid_axis_inst axis s] (t : Tensor α s) :
  Tensor α (shapeAfterSum s axis) :=
  reduceMax axis t (Shape.proveReducibleAlong axis s h.proof)

/-- Reduce along the last axis of `s` (i.e. axis `rank s - 1`). -/
def reduceLastDim {α : Type} [Context α] {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (x : Tensor α s) (h : Shape.reducibleAlong (Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceDim f (Shape.rank s - 1) x h

/-- Like `reduce_last_dim`, but infers axis validity via `valid_axis_inst`. -/
def reduceLastDimAuto {α : Type} [Context α] {s : Shape}
  (f : ∀ {sliceShape : Shape}, Tensor α sliceShape → α)
  (x : Tensor α s) [h : Shape.valid_axis_inst (Shape.rank s - 1) s] :
  Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceLastDim f x (Shape.proveReducibleAlong (Shape.rank s - 1) s h.proof)

-- Reduce mean along the last dimension of any tensor shape
/-- Mean-reduce along the last axis. -/
def reduceMeanLast {α : Type} [Context α] {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong
  (Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceDim meanSpec (Shape.rank s - 1) x h

-- Reduce sum along the last dimension of a 2D tensor (specialized version)
/-- Sum-reduce along the last axis of a 2D tensor `(seqLen, embedDim)`. -/
def reduceSumLast {seqLen embedDim : Nat} (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h
  : Shape.reducibleAlong (Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim seqLen (.dim
  embedDim .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  reduceLastDim sumSpec x h

/-- Product-reduce along the last axis of a 2D tensor `(seqLen, embedDim)`. -/
def reduceProdLast {seqLen embedDim : Nat} (x : Tensor α (.dim seqLen (.dim embedDim .scalar))) (h
  : Shape.reducibleAlong (Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim seqLen (.dim
  embedDim .scalar))) :
  Tensor α (.dim seqLen .scalar) :=
  reduceLastDim prodSpec x h

/-- Max-reduce along the last axis. -/
def reduceMaxLast {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong (Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceMax (Shape.rank s - 1) x h

/-- Min-reduce along the last axis. -/
def reduceMinLast {s : Shape} (x : Tensor α s) (h : Shape.reducibleAlong (Shape.rank s - 1) s) :
  Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceMin (Shape.rank s - 1) x h

/-- Variance-reduce along the last axis (specialized to a leading batch dimension). -/
def reduceVarLast
  {n : Nat} {s : Shape}
  (x : Tensor α (.dim n s)) (h : Shape.reducibleAlong (Shape.rank (.dim n s) - 1) (.dim n s)) :
  Tensor α (shapeAfterSum (.dim n s) (Shape.rank (.dim n s) - 1)) :=
  reduceVar (Shape.rank (.dim n s) - 1) x h

/-- Variance-reduce along the last axis (with axis validity as a typeclass argument). -/
def reduceVarLastGeneral {n : Nat}  {s : Shape}
  (x : Tensor α (.dim n s))
  (h : Shape.valid_axis_inst (Shape.rank (.dim n s) - 1) (.dim n s))
  : Tensor α (shapeAfterSum (.dim n s) (Shape.rank (.dim n s) - 1)) :=
  reduceVarAuto (Shape.rank (.dim n s) - 1) h x

/-- Mean-reduce along the last axis (with axis validity as a typeclass argument). -/
def reduceMeanLastGeneral {s : Shape}
  (x : Tensor α s)
  (h : Shape.valid_axis_inst (Shape.rank s - 1) s)
  : Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceMeanAuto (Shape.rank s - 1) h x

/-- Mean-reduce along the last axis, specialized for proofs that assume well-formedness. -/
def reduceMeanLastGeneralWf {s : Shape}
  (x : Tensor α s)
  [_h_wf : Shape.WellFormed s]
  (_h_rank : Shape.rank s > 0)
  (h_valid : Shape.valid_axis_inst (Shape.rank s - 1) s)
  : Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceMean (Shape.rank s - 1) x (Shape.proveReducibleAlong (Shape.rank s - 1) s h_valid.proof)

/-- Sum-reduce along the last axis (with axis validity inferred via `valid_axis_inst`). -/
def reduceSumLastGeneral {s : Shape}
  (x : Tensor α s)
  [h : Shape.valid_axis_inst (Shape.rank s - 1) s]
  : Tensor α (shapeAfterSum s (Shape.rank s - 1)) :=
  reduceSumAuto (Shape.rank s - 1) x

-- Transpose operations
/-- Transpose a matrix `(m×n)` into `(n×m)`.

PyTorch analogy: `A.transpose(0, 1)` or `A.T` for 2D tensors. -/
def matrixTransposeSpec
  {α : Type} {m n : Nat}
  (t : Tensor α (.dim m (.dim n .scalar))) :
  Tensor α (.dim n (.dim m .scalar)) :=
  match t with
  | Tensor.dim rows =>
    Tensor.dim (fun j : Fin n =>
      Tensor.dim (fun i : Fin m =>
        match rows i with
        | Tensor.dim cols =>
          match cols j with
          | Tensor.scalar value => Tensor.scalar value))

-- Advanced Transpose Operations
/-- Permute a 3D tensor from `(a,b,c)` to `(b,c,a)`. -/
def transpose3DFirstToLastSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim b (.dim c (.dim a .scalar))) :=
  match t with
  | .dim f =>
    .dim fun j =>
      .dim fun k =>
        .dim fun i =>
          match f i with
          | .dim g =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Permute a 3D tensor from `(a,b,c)` to `(c,a,b)`. -/
def transpose3DLastToFirstSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim c (.dim a (.dim b .scalar))) :=
  match t with
  | .dim f =>
    .dim fun k =>
      .dim fun i =>
        .dim fun j =>
          match f i with
          | .dim g =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Swap the last two axes of a 3D tensor: `(a,b,c)` to `(a,c,b)`. -/
def transpose3DLastTwoSpec {α : Type} {a b c : Nat}
  (t : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
  Tensor α (.dim a (.dim c (.dim b .scalar))) :=
  match t with
  | .dim f =>
    .dim fun i =>
      match f i with
      | .dim g =>
        .dim fun k =>
          .dim fun j =>
            match g j with
            | .dim h => .scalar (match h k with | .scalar x => x)

/-- Swap the first two dimensions of a tensor `(m,n,...)` to `(n,m,...)`. -/
def swapFirstTwoSpec {α : Type} {m n : Nat} {s : Shape}
  (t : Tensor α (.dim m (.dim n s))) :
  Tensor α (.dim n (.dim m s)) :=
  match t with
  | .dim f =>
    .dim fun j =>
      .dim fun i =>
        match f i with
        | .dim g => g j

/-- Helper for swapping adjacent dims at a given depth (see `Shape.swapAdjacentAtDepth`). -/
def swapAtDepthHelper {β : Type} {shape : Shape} (tensor : Tensor β shape) (d : Nat) :
      Tensor β (shape.swapAdjacentAtDepth d) :=
      match d, shape, tensor with
      | 0, .dim m (.dim k rest), .dim g =>
        -- Swap dimensions 0 and 1 at this level
        .dim fun j =>
          .dim fun i =>
            match g i with
            | .dim h => h j
      | d + 1, .dim m rest, .dim g =>
        -- Recurse deeper
        .dim fun i => swapAtDepthHelper (g i) d
      | _, .scalar, .scalar x =>
        -- Scalar case - no change needed
        by simp [Shape.swapAdjacentAtDepth]; exact .scalar x
      | 0, .dim _ .scalar, .dim g =>
        -- Only one dimension at this level - no swap possible
        .dim g

/-- Swap adjacent dimensions at a given depth inside a leading batch dimension. -/
def swapAtDepthSpec {α : Type} {n : Nat} {s : Shape}
  (t : Tensor α (.dim n s)) (depth : Nat) :
  Tensor α (.dim n (s.swapAdjacentAtDepth depth)) :=
  match t with
  | .dim f =>
    .dim fun i => swapAtDepthHelper (f i) depth

-- Backward pass for matrix multiplication
/-- Backward pass for matrix multiplication: returns `(dA, dB)` given `dC`.

PyTorch analogy: if `C = A @ B`, then:
- `dA = dC @ Bᵀ`
- `dB = Aᵀ @ dC` -/
def matMulBackwardSpec
  {m n p : Nat}
  (A : Tensor α (.dim m (.dim n .scalar)))
  (B : Tensor α (.dim n (.dim p .scalar)))
  (dC : Tensor α (.dim m (.dim p .scalar))) :
  (Tensor α (.dim m (.dim n .scalar))) × (Tensor α (.dim n (.dim p .scalar))) :=
  let dA := matMulSpec dC (matrixTransposeSpec B) -- dA = dC * Bᵀ
  let dB := matMulSpec (matrixTransposeSpec A) dC -- dB = Aᵀ * dC
  (dA, dB)

/-- Batched matrix multiplication: `[batch,m,n] × [batch,n,p] → [batch,m,p]`. -/
def bmmSpec {α : Type} [Add α] [Mul α] [Zero α]
  {batch m n p : Nat}
  (A : Tensor α (.dim batch (.dim m (.dim n .scalar))))
  (B : Tensor α (.dim batch (.dim n (.dim p .scalar)))) :
  Tensor α (.dim batch (.dim m (.dim p .scalar))) :=
  match A, B with
  | .dim fA, .dim fB =>
      .dim fun i => matMulSpec (match fA i with | t => t) (match fB i with | t => t)

/-- Backward pass for batched matrix multiplication. -/
def bmmBackwardSpec {α : Type} [Add α] [Mul α] [Zero α]
  {batch m n p : Nat}
  (A : Tensor α (.dim batch (.dim m (.dim n .scalar))))
  (B : Tensor α (.dim batch (.dim n (.dim p .scalar))))
  (dC : Tensor α (.dim batch (.dim m (.dim p .scalar)))) :
  (Tensor α (.dim batch (.dim m (.dim n .scalar)))) × (Tensor α (.dim batch (.dim n (.dim p
    .scalar)))) :=
  let dA :=
    bmmSpec (α := α) (batch := batch) (m := m) (n := p) (p := n) dC
      (transpose3DLastTwoSpec (α := α) (a := batch) (b := n) (c := p) B)
  let dB :=
    bmmSpec (α := α) (batch := batch) (m := n) (n := m) (p := p)
      (transpose3DLastTwoSpec (α := α) (a := batch) (b := m) (c := n) A) dC
  (dA, dB)

/-- Runtime check that a tensor value matches a runtime `Shape`.

We use this in a few “dynamic” utilities where we have a runtime shape value and want to guard
access/casts in a total way.
-/
def matchShape {s : Shape} (t : Tensor α s) : Shape → Prop :=
  match t with
  | .scalar _ =>
      fun
      | .scalar => True
      | .dim _ _ => False
  | .dim (n := n) f =>
      fun
      | .scalar => False
      | .dim n' s' => n = n' ∧ ∀ i : Fin n, (f i).matchShape s'

/-- Concatenate a list of `(n,d)` tensors along the last axis, producing `(n, headCount*d)`.

This is mainly used by attention blocks that split/merge heads.

PyTorch analogy: `torch.cat(heads, dim=-1)` after splitting heads, followed by a reshape.
-/
def concatSpec
  {α : Type} [Inhabited α]
  {n d : Nat}
  (headCount : Nat)
  (tensors : List (Tensor α (.dim n (.dim d .scalar))))
  (_h_len : tensors.length = headCount) :
  Tensor α (.dim n (.dim (headCount * d) .scalar)) :=

  -- Helper to get a single row at index i
  let concatRow (i : Fin n) : Tensor α (.dim (headCount * d) .scalar) :=
    let rec buildRow (ts : List (Tensor α (.dim n (.dim d .scalar)))) : List α :=
      match ts with
      | [] => []
      | t :: rest =>
        match t with
        | Tensor.dim f =>
          match f i with
          | Tensor.dim g =>
            let rowElems := (List.finRange d).map (fun j =>
              match g j with
              | Tensor.scalar a => a)
            rowElems ++ buildRow rest

    let values := buildRow tensors
    Tensor.dim (fun j : Fin (headCount * d) =>
      Tensor.scalar (values.getD j.val Inhabited.default))

  Tensor.dim (fun i : Fin n => concatRow i)

/-- Concatenate two vectors by appending `v2` after `v1`. -/
def concatVectorsSpec {α : Type} {n m : Nat}
  (v1 : Tensor α (.dim n .scalar))
  (v2 : Tensor α (.dim m .scalar)) :
  Tensor α (.dim (n + m) .scalar) :=
  match v1, v2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < n then
        f1 ⟨i.val, h⟩
      else
        let j : Fin m :=
        ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-- Concatenate along axis 0 (append `t2` after `t1`). -/
def concatDim0Spec {α : Type} {n m : Nat} {s : Shape}
  (t1 : Tensor α (.dim n s))
  (t2 : Tensor α (.dim m s)) :
  Tensor α (.dim (n + m) s) :=
  match t1, t2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < n then
        f1 ⟨i.val, h⟩
      else
        let j : Fin m :=
          ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-!
## Slicing / concatenation on the leading axis

`concat_dim0_spec` is the "append on axis 0" primitive that powers many higher-level utilities
(sequence concatenation, channel skip connections, etc.).

For backprop and for "undoing" concatenations, it is convenient to have an explicit slice operation.
We keep the API compact and index-safe:

- `slice_range0_spec start len` selects `len` consecutive entries starting at `start` along axis 0.
- `concat_dim0_backward_spec` is the adjoint of `concat_dim0_spec` (splits a gradient tensor).
-/

/-- Slice `len` entries along axis 0, starting at `start`.

This is the simplest "range slice" one typically needs to express:
- taking the first `n` channels/tokens,
- extracting the skip-connection half after a concat,
- implementing `take`/`drop` without changing the inner shape.

The proof `len + start ≤ n` makes the slice total (no out-of-bounds behavior). -/
def sliceRange0Spec {α : Type} {n : Nat} {s : Shape}
  (start len : Nat) (h : len + start ≤ n)
  (t : Tensor α (.dim n s)) : Tensor α (.dim len s) :=
  match t with
  | .dim f =>
      .dim fun i =>
        let idx : Nat := start + i.val
        have h1 : idx < start + len := by
          simp [idx]
        have h2 : start + len ≤ n := by
          simpa [Nat.add_comm] using h
        f ⟨idx, lt_of_lt_of_le h1 h2⟩

/-- Backward (adjoint) of `concat_dim0_spec`.

If `y = concat_dim0_spec x1 x2`, then in reverse-mode we split the upstream gradient `δy` into:
- `δx1` = the first `n` entries of `δy`,
- `δx2` = the last  `m` entries of `δy`. -/
def concatDim0BackwardSpec {α : Type} {n m : Nat} {s : Shape}
  (δ : Tensor α (.dim (n + m) s)) :
  Tensor α (.dim n s) × Tensor α (.dim m s) :=
  let δ₁ := sliceRange0Spec (α := α) (n := n + m) (s := s) 0 n (Nat.le_add_right n m) δ
  let δ₂ :=
    sliceRange0Spec (α := α) (n := n + m) (s := s) n m
      (by simp [Nat.add_comm]) δ
  (δ₁, δ₂)

/--
Backward (adjoint) of `slice_range0_spec`.

If `y = slice_range0_spec start len x`, then `slice_range0_backward_spec start len δy` re-inserts
the gradient into the original shape and fills everything outside the slice with zeros.
-/
def sliceRange0BackwardSpec {α : Type} [Zero α] {n : Nat} {s : Shape}
  (start len : Nat) (_h : len + start ≤ n)
  (δ : Tensor α (.dim len s)) : Tensor α (.dim n s) :=
  -- This is the adjoint of `slice_range0_spec`: the gradient is re-inserted into the
  -- original shape and everything outside the slice is filled with zeros.
  Tensor.dim (fun i =>
    if h1 : i.val < start then
      fill (0 : α) s
    else if h2 : i.val < start + len then
      let j : Fin len :=
        ⟨i.val - start, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h1) h2⟩
      getAtSpec δ j
    else
      fill (0 : α) s)

/--
Concatenate two sequences along time (axis 0), producing a longer sequence.

If `seq1 : (seqLen1 x hidden)` and `seq2 : (seqLen2 x hidden)`, this returns
`(seqLen1 + seqLen2) x hidden` by appending `seq2` after `seq1`.

Do not confuse this with `Spec.concatSequenceSpec` (defined in `NN.Spec.Core.Sequence`), which
concatenates along the feature dimension for *same-length* sequences.
-/
def concatSequenceSpec {α : Type} {seqLen1 seqLen2 hiddenSize : Nat}
  (seq1 : Tensor α (.dim seqLen1 (.dim hiddenSize .scalar)))
  (seq2 : Tensor α (.dim seqLen2 (.dim hiddenSize .scalar))) :
  Tensor α (.dim (seqLen1 + seqLen2) (.dim hiddenSize .scalar)) :=
  match seq1, seq2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < seqLen1 then
        f1 ⟨i.val, h⟩
      else
        let j : Fin seqLen2 :=
          ⟨i.val - seqLen1, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-- Concatenate two sequences along the feature dimension (inner axis). -/
def concatSequenceInnerSpec {α : Type} {seqLen hiddenSize1 hiddenSize2 : Nat}
  (seq1 : Tensor α (.dim seqLen (.dim hiddenSize1 .scalar)))
  (seq2 : Tensor α (.dim seqLen (.dim hiddenSize2 .scalar))) :
  Tensor α (.dim seqLen (.dim (hiddenSize1 + hiddenSize2) .scalar)) :=
  match seq1, seq2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      match f1 i, f2 i with
      | .dim g1, .dim g2 =>
        .dim fun j =>
          if h : j.val < hiddenSize1 then
            g1 ⟨j.val, h⟩
          else
            let k : Fin hiddenSize2 :=
              ⟨j.val - hiddenSize1, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) j.is_lt⟩
            g2 k

-- Expand operations
/-- Expand a `(n, s)` tensor into `(n, 1, s)` by inserting a trailing dimension of size 1.

PyTorch analogy: `t.unsqueeze(-1)` for a rank-1 outer dimension (or `unsqueeze(dim=1)` in 2D terms).
  -/
def expandToColSpec {n s} (t : Tensor α (.dim n s)) : Tensor α (.dim n (.dim 1 s)) :=
  Tensor.dim (fun i => Tensor.dim (fun _ => getAtSpec t i))

-- Expand a vector of shape (n) into a column matrix (n × 1)
/-- Same as `expand_to_col_spec`, specialized to vectors. -/
def expandToColSpecAlt {α : Type} {n : Nat} (v : Tensor α (Shape.dim n Shape.scalar)) :
    Tensor α (Shape.dim n (Shape.dim 1 Shape.scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun _ => getAtSpec v i))

-- Squeeze operations
/-- Squeeze a `(n,1,s)` tensor back into `(n,s)` by dropping the singleton dimension. -/
def squeezeColSpec {n s} (t : Tensor α (.dim n (.dim 1 s))) : Tensor α (.dim n s) :=
  Tensor.dim (fun i => getAtSpec (getAtSpec t i) 0)

-- Squeeze a column matrix (n × 1) back into a vector of shape (n)
/-- Same as `squeeze_col_spec`, specialized to vectors. -/
def squeezeColSpecAlt {α : Type} {n : Nat} (t : Tensor α (Shape.dim n (Shape.dim 1
  Shape.scalar))) :
    Tensor α (Shape.dim n Shape.scalar) :=
  Tensor.dim (fun i => getAtSpec (getAtSpec t i) 0)

-- Unsqueeze operations
/-- Unsqueeze (insert a singleton dim). Currently implemented as `expand_to_col_spec`.

Core uses singleton insertion mainly for column vectors, so this operation is specialized to that
use case.
General axis insertion can extend this definition. -/
def unsqueezeSpec {n s} (t : Tensor α (.dim n s)) (_dim : Nat) : Tensor α (.dim n (.dim 1 s)) :=
  expandToColSpec t

-- Expand vector to batch dimension
/-- Turn a vector `(n)` into a batch of size 1: `(1,n)`. -/
def expandVecToBatchSpec {α : Type} {n : Nat} (v : Tensor α (Shape.dim n Shape.scalar)) :
    Tensor α (Shape.dim 1 (Shape.dim n Shape.scalar)) :=
  Tensor.dim (fun _ => v)

-- Batch dimension manipulation: move batch to end
/-- Move a leading batch dimension to the innermost position. -/
def batchToEndSpec {α : Type} {batch : Nat} {s : Shape}
  (t : Tensor α (.dim batch s)) :
  Tensor α (s.appendDim batch) :=
  match s, t with
  | .scalar, .dim f =>
    -- Input: [batch, scalar] -> Output: [scalar, batch] = [batch]
    -- f : Fin batch -> Tensor α .scalar
    Tensor.dim fun i => f i
  | .dim _ _, .dim f =>
    -- Input: [batch, n, rest...] -> Output: [n, rest..., batch]
    -- f : Fin batch -> Tensor α (.dim n rest)
    -- We need to build: Tensor α (.dim n (rest.appendDim batch))
    Tensor.dim fun j =>
      -- For each position j in the new first dimension n
      -- We need: Tensor α (rest.appendDim batch)
      -- This comes from collecting f[i][j] for all i and restructuring
      collectAtIndexSpec (fun i => match f i with | Tensor.dim g => Tensor.dim (fun _ => g j)) j

-- Channel-first to channel-last (common in vision): (batch, channels, height, width) -> (batch,
-- height, width, channels)
/-- Convert channel-first images `(b,c,h,w)` into channel-last `(b,h,w,c)`. -/
def channelFirstToLastSpec {α : Type} {b c h w : Nat}
  (t : Tensor α (.dim b (.dim c (.dim h (.dim w .scalar))))) :
  Tensor α (.dim b (.dim h (.dim w (.dim c .scalar)))) :=
  match t with
  | .dim f_b =>
    .dim fun i_b =>
      match f_b i_b with
      | .dim f_c =>
        .dim fun i_h =>
          .dim fun i_w =>
            .dim fun i_c =>
              match f_c i_c with
              | .dim f_h =>
                match f_h i_h with
                | .dim f_w => .scalar (match f_w i_w with | .scalar x => x)

end Tensor
end Spec
