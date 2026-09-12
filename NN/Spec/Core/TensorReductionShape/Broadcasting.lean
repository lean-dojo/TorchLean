/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorOps

@[expose] public section


open Spec _root_.TorchLean

namespace TorchLean.Tensor

/-!
# Broadcasting

Spec-level broadcasting and broadcasted binary maps.

Broadcasting is defined in terms of the shape proposition `Shape.CanBroadcastTo`: pick an explicit
target shape and provide evidence that each operand broadcasts to it. The proposition depends only
on the two shapes, so the result never depends on how the evidence was obtained.

Execution goes through the internal `Rep.broadcast`: the source is padded with leading singleton
axes by a zero-copy reshape and then pulled back along the coordinate map that fixes expanded axes
at index zero. This is one buffer fill, independent of the rank.

TorchLean standardizes on this explicit target style throughout core. There is intentionally no
second "implicit" API that infers a common output shape from two operands, because that would split
the codebase into two parallel styles. The backward pass of a broadcast is the sum-reduction over
the broadcast axes (`reduceFromBroadcastTo` in the reductions module).
-/

/-- `Shape.CanBroadcastTo` in the list form used by the internal representation: after padding the
source with leading ones to the target rank, extents agree pointwise or the source extent is one. -/
theorem _root_.Spec.Shape.CanBroadcastTo.forall₂_toList {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) :
    List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft (s₂.rank - s₁.rank) s₁).toList s₂.toList := by
  induction s₂ generalizing s₁ with
  | scalar =>
      cases s₁ with
      | scalar => exact List.Forall₂.nil
      | dim _ _ => exact absurd h Shape.not_canBroadcastTo_dim_scalar
  | dim n t ih =>
      cases s₁ with
      | scalar =>
          have hk : (Shape.dim n t).rank - Shape.rank .scalar =
              (t.rank - Shape.rank .scalar) + 1 := by
            simp [Shape.rank]
          rw [hk]
          exact List.Forall₂.cons (Or.inr rfl) (ih (Shape.canBroadcastTo_scalar_dim.mp h))
      | dim m s =>
          by_cases hRank : s.rank = t.rank
          · obtain ⟨hHead, hTail⟩ := (Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h
            have hk : (Shape.dim n t).rank - (Shape.dim m s).rank = 0 := by
              simp [Shape.rank, hRank]
            have hk' : t.rank - s.rank = 0 := by grind
            have hTailList := ih hTail
            rw [hk'] at hTailList
            rw [hk]
            exact List.Forall₂.cons hHead hTailList
          · have hTail := (Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp h
            have hle := hTail.rank_le
            have hk : (Shape.dim n t).rank - (Shape.dim m s).rank =
                (t.rank - (Shape.dim m s).rank) + 1 := by
              simp only [Shape.rank] at hle ⊢
              grind
            rw [hk]
            exact List.Forall₂.cons (Or.inr rfl) (ih hTail)

namespace Broadcasting.Internal

/-- Broadcast a source padded with `k` leading singleton axes.

Separating the padding count from the shapes lets the structural lemmas below be stated for
literal counts `0` and `k + 1`, independently of the rank arithmetic in `broadcastTo`. -/
def broadcastPadded {α : Type} [TorchLean.Storage α] {s₁ s₂ : Shape} (k : Nat)
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft k s₁).toList s₂.toList)
    (tensor : Tensor α s₁) : Tensor α s₂ :=
  TorchLean.Tensor.Internal.Rep.broadcast h (Spec.padLeft (n := k) tensor)

/-- Transport the padding count along an equality. -/
theorem broadcastPadded_congr {α : Type} [TorchLean.Storage α] {s₁ s₂ : Shape}
    {k k' : Nat} (hk : k = k')
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft k s₁).toList s₂.toList)
    (tensor : Tensor α s₁) :
    broadcastPadded k h tensor = broadcastPadded k' (hk ▸ h) tensor := by
  subst hk
  rfl

/-- Broadcasting a scalar to the scalar shape is the identity. -/
theorem broadcastPadded_zero_scalar {α : Type} [TorchLean.Storage α]
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft 0 Shape.scalar).toList (Shape.scalar).toList)
    (tensor : Tensor α .scalar) :
    broadcastPadded 0 h tensor = tensor :=
  (TorchLean.Tensor.Internal.Rep.broadcast_nil h (Spec.padLeft (n := 0) tensor)).trans
    (Spec.padLeft_zero tensor)

/-- One more padding axis stacks the broadcast of the remaining padding along the leading target
axis. -/
theorem broadcastPadded_succ {α : Type} [TorchLean.Storage α] {k n : Nat} {s t : Shape}
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft (k + 1) s).toList (Shape.dim n t).toList)
    (tensor : Tensor α s) :
    broadcastPadded (k + 1) h tensor =
      Tensor.dim fun _ => broadcastPadded k (List.forall₂_cons.mp h).2 tensor := by
  refine (TorchLean.Tensor.Internal.Rep.broadcast_cons_one (s := (Shape.padLeft k s).toList)
    (t := t.toList) h (Spec.padLeft (n := k + 1) tensor)).trans ?_
  congr 1
  funext _
  congr 1
  rw [Spec.padLeft_succ]
  exact Tensor.unstack_dim (fun _ => Spec.padLeft (n := k) tensor) 0

/-- With no padding and equal leading extents, broadcasting acts slice by slice. -/
theorem broadcastPadded_zero_dim_eq {α : Type} [TorchLean.Storage α] {n : Nat} {s t : Shape}
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft 0 (Shape.dim n s)).toList (Shape.dim n t).toList)
    (tensor : Tensor α (.dim n s)) :
    broadcastPadded 0 h tensor =
      Tensor.dim fun i =>
        broadcastPadded 0 (List.forall₂_cons.mp h).2 (Tensor.unstack tensor i) := by
  refine (TorchLean.Tensor.Internal.Rep.broadcast_cons_eq (s := s.toList) (t := t.toList) h
    (Spec.padLeft (n := 0) tensor)).trans ?_
  congr 1

/-- With no padding and a leading source extent of one, every target slice reads slice zero. -/
theorem broadcastPadded_zero_dim_one {α : Type} [TorchLean.Storage α] {n : Nat} {s t : Shape}
    (h : List.Forall₂ (fun a b => a = b ∨ a = 1)
      (Shape.padLeft 0 (Shape.dim 1 s)).toList (Shape.dim n t).toList)
    (tensor : Tensor α (.dim 1 s)) :
    broadcastPadded 0 h tensor =
      Tensor.dim fun _ =>
        broadcastPadded 0 (List.forall₂_cons.mp h).2 (Tensor.unstack tensor 0) := by
  refine (TorchLean.Tensor.Internal.Rep.broadcast_cons_one (s := s.toList) (t := t.toList) h
    (Spec.padLeft (n := 0) tensor)).trans ?_
  congr 1

end Broadcasting.Internal

/-- Broadcast a tensor to a compatible target shape (spec-level analogue of
`torch.broadcast_to`).

The source is padded with leading singleton axes to the target rank and every singleton axis is
replicated. The result depends only on the shapes, not on how `h` was proved. -/
def broadcastTo {α : Type} [TorchLean.Storage α] {s₁ s₂ : Shape}
    (h : Shape.CanBroadcastTo s₁ s₂) (tensor : Tensor α s₁) : Tensor α s₂ :=
  Broadcasting.Internal.broadcastPadded (s₂.rank - s₁.rank) h.forall₂_toList tensor

/-- Broadcasting a scalar tensor to the scalar shape is the identity. -/
@[simp] theorem broadcastTo_scalar {α : Type} [TorchLean.Storage α]
    (h : Shape.CanBroadcastTo .scalar .scalar) (tensor : Tensor α .scalar) :
    broadcastTo h tensor = tensor :=
  Broadcasting.Internal.broadcastPadded_zero_scalar _ tensor

/-- Equal leading extents broadcast slice by slice. -/
theorem broadcastTo_dim_eq {α : Type} [TorchLean.Storage α] {n : Nat} {s t : Shape}
    (hRank : s.rank = t.rank) (h : Shape.CanBroadcastTo (.dim n s) (.dim n t))
    (tensor : Tensor α (.dim n s)) :
    broadcastTo h tensor =
      Tensor.dim fun i =>
        broadcastTo ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).2
          (Tensor.unstack tensor i) := by
  have hk : (Shape.dim n t).rank - (Shape.dim n s).rank = 0 := by
    simp [Shape.rank, hRank]
  have hk' : t.rank - s.rank = 0 := by grind
  unfold broadcastTo
  rw [Broadcasting.Internal.broadcastPadded_congr hk,
    Broadcasting.Internal.broadcastPadded_zero_dim_eq]
  congr 1
  funext i
  rw [Broadcasting.Internal.broadcastPadded_congr hk']

/-- A leading source extent of one is replicated along the leading target axis. -/
theorem broadcastTo_dim_one {α : Type} [TorchLean.Storage α] {n : Nat} {s t : Shape}
    (hRank : s.rank = t.rank) (h : Shape.CanBroadcastTo (.dim 1 s) (.dim n t))
    (tensor : Tensor α (.dim 1 s)) :
    broadcastTo h tensor =
      Tensor.dim fun _ =>
        broadcastTo ((Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp h).2
          (Tensor.unstack tensor 0) := by
  have hk : (Shape.dim n t).rank - (Shape.dim 1 s).rank = 0 := by
    simp [Shape.rank, hRank]
  have hk' : t.rank - s.rank = 0 := by grind
  unfold broadcastTo
  rw [Broadcasting.Internal.broadcastPadded_congr hk,
    Broadcasting.Internal.broadcastPadded_zero_dim_one]
  congr 1
  funext _
  rw [Broadcasting.Internal.broadcastPadded_congr hk']

/-- A target axis that only raises the rank replicates the whole source. -/
theorem broadcastTo_expand {α : Type} [TorchLean.Storage α] {n : Nat} {s t : Shape}
    (hRank : s.rank ≤ t.rank) (h : Shape.CanBroadcastTo s (.dim n t)) (tensor : Tensor α s) :
    broadcastTo h tensor = Tensor.dim fun _ => broadcastTo (h.of_expand_dims hRank) tensor := by
  have hk : (Shape.dim n t).rank - s.rank = (t.rank - s.rank) + 1 := by
    simp only [Shape.rank]
    grind
  unfold broadcastTo
  rw [Broadcasting.Internal.broadcastPadded_congr hk,
    Broadcasting.Internal.broadcastPadded_succ]

/-- Broadcasting a tensor to its own shape changes nothing. -/
@[simp] theorem broadcastTo_self {α : Type} [TorchLean.Storage α] {shape : Shape}
    (h : Shape.CanBroadcastTo shape shape) (tensor : Tensor α shape) :
    broadcastTo h tensor = tensor := by
  induction shape with
  | scalar => exact broadcastTo_scalar h tensor
  | dim n rest ih =>
      rw [broadcastTo_dim_eq rfl h]
      conv_rhs => rw [← Tensor.dim_unstack tensor]
      congr 1
      funext i
      exact ih _ (Tensor.unstack tensor i)

/-- Broadcasting across one new leading target axis replicates the source. -/
@[simp] theorem broadcastTo_dim_self {α : Type} [TorchLean.Storage α] {n : Nat} {s : Shape}
    (h : Shape.CanBroadcastTo s (.dim n s)) (tensor : Tensor α s) :
    broadcastTo h tensor = Tensor.dim fun _ => tensor := by
  rw [broadcastTo_expand (Nat.le_refl _) h]
  congr 1
  funext _
  exact broadcastTo_self _ tensor

/-! ## Broadcasted maps -/

/-- Helper: map a scalar on the left over any tensor shape. -/
def mapScalarLeft {α : Type} [TorchLean.Storage α]
    (f : α → α → α) (x : α) {s : Shape} (tensor : Tensor α s) :
    Tensor α s :=
  Tensor.mapSpec (f x) tensor

/-- Helper: map a scalar on the right over any tensor shape. -/
def mapScalarRight {α : Type} [TorchLean.Storage α]
    (f : α → α → α) (y : α) {s : Shape} (tensor : Tensor α s) :
    Tensor α s :=
  Tensor.mapSpec (fun x => f x y) tensor

/--
Binary element-wise operation with broadcasting to an explicit target shape.

This is the helper you typically want in spec code:
- pick the output shape `t`,
- broadcast each operand to `t`,
- then `map2Spec` the pointwise operation.

PyTorch analogy: `f(x, y)` where `x` and/or `y` are broadcastable to a common shape.
The common shape is explicit rather than discovered at runtime, which makes the result type
predictable and fixes the intended output shape at the call site.
-/
def broadcastMapTo {α} [TorchLean.Storage α]
    (f : α → α → α)
    {s₁ s₂ t : Shape} (cbx : Shape.CanBroadcastTo s₁ t) (cby : Shape.CanBroadcastTo s₂ t) :
    Tensor α s₁ → Tensor α s₂ → Tensor α t :=
  fun x y => map2Spec f (broadcastTo cbx x) (broadcastTo cby y)

end TorchLean.Tensor
