/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Reduce.Backward
public import TorchLean.Tensor.Layers.Activation
public import TorchLean.Tensor.List

/-!
# Softmax, log-softmax and log-sum-exp along any axes

`softmax x dim mask schedule` and `logSoftmax x dim` act on every fiber of `x` along `dim`, at any
positive rank. A Boolean `mask` blocks the entries where it is false: a blocked entry has a zero
numerator, so weight zero where the fiber's allowed scores are finite, and every entry of a fiber
without an allowed entry is zero, the finite-scalar form of PyTorch's `masked_fill(~mask, -inf)`
before softmax. The `Softmax.Schedule` says how a fiber is
traversed. `twoPass` applies the stable vector kernels `TorchLean.Softmax.vec` and
`TorchLean.Softmax.hardMaskedVec`: the axis moves last, the last-axis kernel runs on every row, and
the axis moves back. It keeps the stalled-sum detection and exact replay
(`TorchLean.Softmax.normalizeWeights`). `online blocks` makes one pass over the blocks of the
allowed keys with a running maximum, a fold of the IEEE `TorchLean.Context.maximum`, and a
normalizer rescaled by `exp (old - new)` when a block raises the maximum (`Softmax.onlineStep`,
FlashAttention's schedule), then divides each shifted exponential by the normalizer. From the
running maximum `-∞`, a block of `-∞` scores would make the normalizer NaN, as `-∞ - (-∞)` is NaN
(IEEE 754-2019 invalid operation). The running maximum starts at the first allowed score that is not
`-∞` (`Softmax.onlineStart`), or at `-∞` when every allowed score is `-∞`. Over the reals the two
schedules agree (`Tensor.softmax_online_eq`), and in floating point they can round differently.
Log-softmax keeps its strict-argmax pivot and `log1p` normalizer.

`softmaxBackward x dY dim` is the softmax VJP at the logits `x`, and `maskedSoftmaxBackward x mask
dY dim` the VJP of the masked softmax: zero on a fiber without an allowed entry, and at blocked
entries where the scores and the upstream gradient are finite. `logSoftmaxJVP y dx dim` and
`logSoftmaxBackward y dY dim` are the log-softmax JVP and VJP at the output `y`. Each acts on the
fibers along `dim` in the same way, with the centered and pivoted vector formulas of `Activation`.

`logsumexp x dims keepdim` reduces over several axes. Each entry reads its fiber in row-major
order, takes the first strict maximum as pivot and adds to it `log1p` of the sum of the other
shifted exponentials. An infinite pivot cannot be shifted out, so that fiber gives the `log` of the
sum of its exponentials, as `torch.logsumexp` does.

We wrote the kernels by hand. The online schedule is the forward pass of FlashAttention-2 (Dao,
2023, arXiv:2307.08691, Algorithm 1 of Sec. 3.1.1): each block's exponentials are taken against
the raised maximum, and the weighted sum stays unscaled until one division at the end. Algorithm 1
starts the running maximum at `-∞` (its line 5), and `onlineStart` is where we part from it.
`TorchLeanTest/Tensor/OnlineSoftmax.lean` checks `onlineStep` at a NaN, signed zeros and
infinities. `logsumexpFiber` is line 7 of Algorithm 4.1 in Blanchard, Higham and Higham (2019,
arXiv:1909.03469), `f = a + log1p s` with `s` the sum of the shifted exponentials other than the
pivot's.

The `_apply` lemmas read an entry of each operator as its kernel on the fiber through it.
-/

@[expose] public section

open TorchLean.Softmax
open FloatLib.Numerics (MathFunctions)

/-! ## Schedules -/

namespace TorchLean.Softmax

/-- The keys of a fiber split into consecutive blocks of `size` keys, the last one possibly
shorter, with `size = 0` keeping all keys in one block. -/
structure BlockPartition where
  /-- The number of keys in every block but the last. -/
  size : Nat
  deriving DecidableEq, Repr, Inhabited

/-- How softmax and attention pass over the keys of a fiber: `twoPass` finds the maximum and the
normalizer before the weights, and `online blocks` makes one pass over the blocks with a running
maximum and a normalizer rescaled by `exp (old - new)` when a block raises it, as FlashAttention
does. -/
inductive Schedule where
  /-- Find the maximum and the normalizer before the weights. -/
  | twoPass
  /-- One pass over the blocks with a running maximum and a rescaled normalizer. -/
  | online (blocks : BlockPartition)
  deriving DecidableEq, Repr, Inhabited

/-- One key block of the online schedule: raise the running maximum over the block's scores with
the IEEE `Context.maximum`, rescale the normalizer and the weighted sum by `exp (old - new)`, and
add the block's weights to the normalizer and their products with the values to the weighted sum.
`round` is applied to each weight before the product and `roundSum` before the sum, as when the
weights are stored in a narrower format for the matrix product: FlashAttention sums the unrounded
weights (`roundSum = id`), and summing the stored ones (`roundSum = round`) keeps the weights of
the two sums equal. -/
def onlineStep {β : Type} [Context β]
    (exp roundSum round : β → β) (state : β × β × β) (block : List (β × β)) : β × β × β :=
  let m := block.foldl (fun m e => Context.maximum m e.1) state.1
  let scale := exp (state.1 - m)
  (m, scale * state.2.1 + block.foldl (fun l e => l + roundSum (exp (e.1 - m))) 0,
    scale * state.2.2 + block.foldl (fun o e => o + round (exp (e.1 - m)) * e.2) 0)

/-- The online softmax average of one query and one value coordinate: fold the blocks of
`(score, value)` pairs from the running maximum `start` and empty sums, then divide the weighted
sum by the normalizer. In IEEE arithmetic it is NaN when `start = -∞`, the first block's scores are
all `-∞` and `exp` maps NaN to NaN, since `-∞ - (-∞)` is NaN (IEEE 754-2019 invalid operation) and
`onlineStep` then gives NaN sums. -/
def onlineAverage {β : Type} [Context β]
    (exp roundSum round : β → β) (start : β) (blocks : List (List (β × β))) : β :=
  let state := blocks.foldl (onlineStep exp roundSum round) (start, 0, 0)
  state.2.2 / state.2.1

/-- A left fold of additions from `a` adds the sum of the terms. -/
theorem foldl_add_eq_add_sum {β : Type} (f : β → ℝ) :
    ∀ (l : List β) (a : ℝ), l.foldl (fun acc e => acc + f e) a = a + (l.map f).sum
  | [], a => by simp
  | e :: l, a => by
      rw [List.foldl_cons, foldl_add_eq_add_sum f l, List.map_cons, List.sum_cons]; ring

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- The allowed keys of a fiber in index order, each as its score and its value. -/
def onlineKeys {n : Nat} (score value : Fin n → α) (allowed : Fin n → Bool) : List (α × α) :=
  ((List.finRange n).filter allowed).map fun j => (score j, value j)

/-- The first score of the keys `first :: rest` that is not `-∞`, or `first.1` when every score is
`-∞`. `onlineWeight` and `onlineAttend` start the running maximum there, since `onlineStep` from
the running maximum `-∞` over a block of `-∞` scores gives NaN sums, from the invalid operation
`-∞ - (-∞)` of IEEE 754-2019. -/
def onlineStart {β : Type} [Context β] (first : β × β) (rest : List (β × β)) : β :=
  (((first :: rest).find? fun e => !Context.isInf e.1 || e.1 > 0).getD first).1

/-- `onlineStart first rest` is `first.1` when `Context.isInf first.1 = false`. -/
theorem onlineStart_of_isInf_eq_false {β : Type} [Context β] (first : β × β)
    (rest : List (β × β)) (h : Context.isInf first.1 = false) :
    onlineStart first rest = first.1 := by
  simp [onlineStart, h]

/-- The online softmax weight of key `j`: one pass of `onlineStep` over the blocks of the allowed
keys from `onlineStart` gives the maximum `M` and the normalizer `l`, and the weight is
`exp (score j - M) / l` at an allowed key and `0` at a blocked one or when no key is allowed. -/
def onlineWeight {n : Nat} (blocks : BlockPartition) (score : Fin n → α) (allowed : Fin n → Bool)
    (j : Fin n) : α :=
  match onlineKeys score (fun _ => 0) allowed with
  | [] => 0
  | first :: rest =>
    let state := (List.toChunks blocks.size (first :: rest)).foldl
      (onlineStep MathFunctions.exp id id) (onlineStart first rest, 0, 0)
    if allowed j then MathFunctions.exp (score j - state.1) / state.2.1 else 0

/-- The online attention output of one query and one value coordinate: `onlineAverage` over the
blocks of the allowed keys from `onlineStart`, and `0` when no key is allowed. -/
def onlineAttend {n : Nat} (blocks : BlockPartition) (score value : Fin n → α)
    (allowed : Fin n → Bool) : α :=
  match onlineKeys score value allowed with
  | [] => 0
  | first :: rest =>
    onlineAverage MathFunctions.exp id id (onlineStart first rest)
      (List.toChunks blocks.size (first :: rest))

end TorchLean.Softmax

/-! ## Hard-masked softmax kernels -/

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
## Exact hard masking

TorchLean encodes the usual "true `-∞` before softmax" behavior without requiring the tensor scalar
type to contain infinities. No finite sentinel replaces a blocked logit: with `rowMax` the greatest
allowed score in the row, the stable softmax numerators are

`numerator_j = if mask_j then exp(score_j - rowMax) else 0`.

A blocked entry gets the zero numerator that `exp(-∞)` contributes. The zero-mass conclusion used
in causal proofs requires the scalar division laws and a nonzero denominator, and a row with no
allowed positions returns zeros.
-/

/-- Maximum allowed score in one hard-masked row, or `none` when every entry is blocked. -/
def Softmax.hardMaskedMax? {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) : Option α :=
  (List.finRange n).foldl (fun best i =>
    let score := Tensor.getScalar scores i
    let allowed := Tensor.getScalar mask i
    if allowed then
      match best with
      | none => some score
      | some current => some (if score > current then score else current)
    else
      best) none

/-- The coordinate `hardMaskedMax?` reads: one left fold over the allowed coordinates that moves
to a strictly larger entry, or `none` when every entry is blocked. -/
def Softmax.hardMaskedPivot? {n : Nat} (scores : Tensor α [n]) (mask : Tensor Bool [n]) :
    Option (Fin n) :=
  (List.finRange n).foldl (fun best i =>
    if Tensor.getScalar mask i then
      match best with
      | none => some i
      | some current =>
          some (if Tensor.getScalar scores i > Tensor.getScalar scores current then i else current)
    else
      best) none

/-- The maximum allowed score is the entry at the hard-masked pivot. -/
theorem Softmax.hardMaskedMax?_eq_map {n : Nat} (scores : Tensor α [n]) (mask : Tensor Bool [n]) :
    hardMaskedMax? scores mask =
      (hardMaskedPivot? scores mask).map (Tensor.getScalar scores) := by
  unfold hardMaskedMax? hardMaskedPivot?
  refine List.foldl_hom (Option.map (Tensor.getScalar scores)) (init := none) ?_
  intro best i
  cases Tensor.getScalar mask i <;> cases best <;>
    simp only [Bool.false_eq_true, ↓reduceIte, Option.map_some, Option.map_none]
  split <;> rfl

/-- Hard-masked softmax on one vector.

`mask[j] = false` makes the `j`-th numerator exactly zero before normalization, the ordinary
finite-scalar encoding of softmax with true `-∞` masked logits. The maximum and denominator are
computed only over allowed entries, and subtracting the allowed-row maximum gives the usual
numerically stable softmax formula. If every mask entry is false, the result is the zero vector,
matching PyTorch SDPA and the LibTorch CUDA attention bridge (`TorchLean/NN/Backend/LibTorch.lean`).
-/
def Softmax.hardMaskedVec {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) :
    Tensor α [n] :=
  match hardMaskedMax? scores mask with
  | none => Tensor.full _ 0
  | some rowMax =>
      let numerators : Tensor α [n] :=
        zipWith
          (fun score allowed =>
            if allowed then MathFunctions.exp (score - rowMax) else 0)
          scores mask
      normalizeWeights numerators

/-- Hard-masked softmax along the innermost axis of an arbitrary tensor. -/
def Softmax.hardMasked : {s : Shape} → Tensor α s → Tensor Bool s → Tensor α s
  | .scalar, _scores, mask =>
      Tensor.scalar (if mask.item then 1 else 0)
  | .dim _ .scalar, scores, mask => hardMaskedVec scores mask
  | .dim n inner, scores, mask =>
      Tensor.dim (fun i : Fin n =>
        hardMasked (s := inner)
          (Tensor.unstack scores i) (Tensor.unstack mask i))

/-- The pullback of `hardMaskedVec scores mask` at `dY`: zero on a vector without an
allowed entry, otherwise the centered softmax pullback of the masked weights, anchored at the
upstream value of the masked pivot. -/
def Softmax.hardMaskedPullback {n : Nat} (scores : Tensor α [n]) (mask : Tensor Bool [n])
    (dY : Tensor α [n]) : Tensor α [n] :=
  match Softmax.hardMaskedPivot? scores mask with
  | none => Tensor.full _ 0
  | some pivot =>
      pullback (hardMaskedVec scores mask) dY
        (dY.getScalar pivot)

/-- `hardMaskedPullback` on every last-axis row: the scores, and the mask paired with the
upstream gradient. A scalar, which has no row, gives zero. -/
def Softmax.hardMaskedInnermostPullback :
    {s : Shape} → Tensor α s → Tensor (Bool × α) s → Tensor α s
  | .scalar, _scores, _maskGrad => Tensor.scalar 0
  | .dim _ .scalar, scores, maskGrad =>
      Softmax.hardMaskedPullback scores (Tensor.map Prod.fst maskGrad)
        (Tensor.map Prod.snd maskGrad)
  | .dim n inner, scores, maskGrad =>
      Tensor.dim fun i : Fin n =>
        hardMaskedInnermostPullback (s := inner) (Tensor.unstack scores i)
          (Tensor.unstack maskGrad i)

end TorchLean

namespace TorchLean.Tensor

variable {α : Type} [Storage α] [Context α] {s : Shape}

/-! ## Softmax -/

/-- Softmax of `x` along `dim` by `schedule`, with the entries where `mask` is false given a zero
numerator (weight zero where the allowed scores are finite): two-pass applies
`Softmax.vec`, or `Softmax.hardMasked` with a mask, to every fiber
along `dim`, and the online schedule `Softmax.onlineWeight`. -/
def softmax (x : Tensor α s) (dim : Fin s.rank) (mask : Option (Tensor Bool s) := none)
    (schedule : Softmax.Schedule := .twoPass) : Tensor α s :=
  match schedule, mask with
  | .twoPass, none => alongAxis dim (fun x => Softmax.innermost x) x
  | .twoPass, some mask => alongAxis₂ dim (fun x mask => Softmax.hardMasked x mask) x mask
  | .online blocks, mask =>
    Rep.ofFn fun c =>
      Softmax.onlineWeight blocks (x.fiber dim c).getScalar
        (fun j => mask.elim true fun mask => (mask.fiber dim c).getScalar j)
        (Shape.Coord.axisEquiv dim (shape := s) c).1

/-- An entry of `softmax x dim` is the vector softmax of the fiber through it, at its index along
`dim`. -/
theorem softmax_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    (softmax x dim) c = (Softmax.vec (x.fiber dim c)).getScalar
      (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis_apply (fun x => Softmax.innermost x) (fun _ => rfl) dim x c


/-- An entry of `softmax x dim (some mask)` is the hard-masked vector softmax of the fibers of the
scores and the mask through it, at its index along `dim`. -/
theorem softmax_mask_apply (x : Tensor α s) (dim : Fin s.rank) (mask : Tensor Bool s)
    (c : s.Coord) :
    (softmax x dim (some mask)) c =
      (Softmax.hardMaskedVec (x.fiber dim c) (mask.fiber dim c)).getScalar
        (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis₂_apply (fun x mask => Softmax.hardMasked x mask) (fun _ _ => rfl)
    dim x mask c

/-- Along the last axis, `softmax x dim (some mask)` is `Softmax.hardMasked`. -/
theorem softmax_mask_last (x : Tensor α s) (mask : Tensor Bool s) (h : s.rank - 1 < s.rank) :
    softmax x ⟨s.rank - 1, h⟩ (some mask) = Softmax.hardMasked x mask :=
  alongAxis₂_of_last _ (fun x mask => Softmax.hardMasked x mask) x mask
    (show s.rank - 1 + 1 = s.rank by omega)

/-- An entry of the online softmax is `Softmax.onlineWeight` of the fiber through it, at
its index along `dim`. -/
theorem softmax_online_apply (x : Tensor α s) (dim : Fin s.rank) (mask : Option (Tensor Bool s))
    (blocks : Softmax.BlockPartition) (c : s.Coord) :
    (softmax x dim mask (.online blocks)) c =
      Softmax.onlineWeight blocks (x.fiber dim c).getScalar
        (fun j => mask.elim true fun mask => (mask.fiber dim c).getScalar j)
        (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  Rep.get_ofFn _ c

/-- The softmax VJP along `dim` at the logits `x`: the centered row pullback of
`Softmax.innermostBackward` on every fiber along `dim`. -/
def softmaxBackward (x dY : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  alongAxis₂ dim Softmax.innermostBackward x dY

/-- An entry of `softmaxBackward x dY dim` is the vector pullback on the fibers through it. -/
theorem softmaxBackward_apply (x dY : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    softmaxBackward x dY dim c =
      (Softmax.innermostBackward (x.fiber dim c) (dY.fiber dim c)).getScalar
        (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis₂_apply Softmax.innermostBackward
    (fun _ _ => rfl) dim x dY c


/-- The VJP of `softmax x dim (some mask)` at `dY`: on every fiber along `dim`, the pullback of the
hard-masked vector softmax, which is zero on a fiber without an allowed entry and at blocked entries
where the scores and `dY` are finite. Each fiber's pullback is computed once, by moving `dim` last,
applying `Softmax.hardMaskedInnermostPullback` to the scores and the mask paired with
`dY`, and moving it back. -/
def maskedSoftmaxBackward (x : Tensor α s) (mask : Tensor Bool s) (dY : Tensor α s)
    (dim : Fin s.rank) : Tensor α s :=
  alongAxis₂ dim Softmax.hardMaskedInnermostPullback x (zipWith Prod.mk mask dY)

/-- An entry of `maskedSoftmaxBackward x mask dY dim` is the hard-masked vector pullback on the
fibers through it, at its index along `dim`. -/
theorem maskedSoftmaxBackward_apply (x : Tensor α s) (mask : Tensor Bool s) (dY : Tensor α s)
    (dim : Fin s.rank) (c : s.Coord) :
    maskedSoftmaxBackward x mask dY dim c =
      (Softmax.hardMaskedPullback (x.fiber dim c) (mask.fiber dim c)
        (dY.fiber dim c)).getScalar (Shape.Coord.axisEquiv dim (shape := s) c).1 := by
  refine (alongAxis₂_apply Softmax.hardMaskedInnermostPullback (fun _ _ => rfl) dim x
    (zipWith Prod.mk mask dY) c).trans ?_
  have hmask : Tensor.map Prod.fst ((zipWith Prod.mk mask dY).fiber dim c) = mask.fiber dim c := by
    apply Rep.ext
    intro j
    simp [fiber_apply]
  have hgrad : Tensor.map Prod.snd ((zipWith Prod.mk mask dY).fiber dim c) = dY.fiber dim c := by
    apply Rep.ext
    intro j
    simp [fiber_apply]
  show (Softmax.hardMaskedPullback (x.fiber dim c)
    (Tensor.map Prod.fst ((zipWith Prod.mk mask dY).fiber dim c))
    (Tensor.map Prod.snd ((zipWith Prod.mk mask dY).fiber dim c))).getScalar _ = _
  rw [hmask, hgrad]

/-! ## Log-softmax -/

/-- Log-softmax of `x` along `dim`: `LogSoftmax.vec` on every fiber along
`dim`, computed by moving `dim` last, applying the last-axis kernel and moving it back. -/
def logSoftmax (x : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  alongAxis dim (fun x => LogSoftmax.innermost x) x

/-- An entry of `logSoftmax x dim` is the vector log-softmax of the fiber through it, at its index
along `dim`. -/
theorem logSoftmax_apply (x : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    logSoftmax x dim c = (LogSoftmax.vec (x.fiber dim c)).getScalar
      (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis_apply (fun x => LogSoftmax.innermost x) (fun _ => rfl) dim x c


/-- The log-softmax JVP along `dim` at the output `y`: the pivoted row JVP of
`LogSoftmax.innermostJVP` on every fiber along `dim`. -/
def logSoftmaxJVP (y dx : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  alongAxis₂ dim LogSoftmax.innermostJVP y dx

/-- An entry of `logSoftmaxJVP y dx dim` is the vector JVP on the fibers through it. -/
theorem logSoftmaxJVP_apply (y dx : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    logSoftmaxJVP y dx dim c =
      (LogSoftmax.innermostJVP (y.fiber dim c) (dx.fiber dim c)).getScalar
        (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis₂_apply LogSoftmax.innermostJVP
    (fun _ _ => rfl) dim y dx c


/-- The log-softmax VJP along `dim` at the output `y`: the pivoted row pullback of
`LogSoftmax.vecBackward` on every fiber along `dim`. -/
def logSoftmaxBackward (y dY : Tensor α s) (dim : Fin s.rank) : Tensor α s :=
  alongAxis₂ dim LogSoftmax.innermostBackward y dY

/-- An entry of `logSoftmaxBackward y dY dim` is the vector pullback on the fibers through it. -/
theorem logSoftmaxBackward_apply (y dY : Tensor α s) (dim : Fin s.rank) (c : s.Coord) :
    logSoftmaxBackward y dY dim c =
      (LogSoftmax.vecBackward (y.fiber dim c) (dY.fiber dim c)).getScalar
        (Shape.Coord.axisEquiv dim (shape := s) c).1 :=
  alongAxis₂_apply LogSoftmax.innermostBackward
    (fun _ _ => rfl) dim y dY c


/-! ## Log-sum-exp -/

/-- The log-sum-exp of one fiber listed as a vector: its first strict maximum plus `log1p` of the
sum of the other exponentials shifted by it, and `log 0` when the fiber is empty. At an infinite
pivot, where the shift `∞ - ∞` is NaN, it is the `log` of the sum of the exponentials: `-∞` when
every entry is `-∞`, `+∞` at a `+∞` entry without a NaN, and NaN at a NaN entry. -/
def logsumexpFiber {n : Nat} (t : Tensor α [n]) : α :=
  if h : 0 < n then
    let pivot : Fin n := Softmax.pivot t h
    let largest := t.getScalar pivot
    if Context.isInf largest then
      MathFunctions.log (Tensor.sum (ofFn fun i : Fin n => MathFunctions.exp (t.getScalar i)))
    else
      let shifted := t - Tensor.full _ largest
      let tail : Tensor α [n] := ofFn fun i =>
        if i = pivot then 0 else MathFunctions.exp (shifted.getScalar i)
      largest + Context.log1p (Tensor.sum tail)
  else MathFunctions.log 0

/-- Log-sum-exp of `x` over the axes `dims`: each entry is `logsumexpFiber` of its fiber
in row-major order of `x`. `keepdim` keeps the listed axes with extent one. -/
def logsumexp (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool) :
    Tensor α (s.reduce dims keepdim) :=
  Rep.ofFn fun o =>
    logsumexpFiber (ofFn fun i : Fin (Shape.size (s.selectAxes dims)) =>
      x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i)))

/-- An entry of `logsumexp x dims keepdim` is the log-sum-exp of its fiber in row-major order. -/
theorem logsumexp_apply (x : Tensor α s) (dims : List (Fin s.rank)) (keepdim : Bool)
    (o : (s.reduce dims keepdim).Coord) :
    logsumexp x dims keepdim o =
      logsumexpFiber (ofFn fun i : Fin (Shape.size (s.selectAxes dims)) =>
        x (s.reduceSource dims keepdim o (Shape.Coord.unlinearize i))) :=
  Rep.get_ofFn _ o

end TorchLean.Tensor
