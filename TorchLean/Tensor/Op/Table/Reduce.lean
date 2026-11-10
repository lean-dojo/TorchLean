/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Table.Rule

/-!
# Equation lemmas of the reductions and the operators along an axis

Each lemma here states one arm of `Op.rule?` for `reduce`, `mean`, a cumulative operator,
`selectiveScan`, a softmax (`softmax`, `logSoftmax`, `maskedSoftmax`) or an ordering (`sort`,
`topk`, `argmax`, `argmin`, `topKMask`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_nil)

namespace TorchLean

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

namespace Op

/-- The rule of `Op.cumsum`. -/
theorem rule?_cumsum [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.cumsum dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.cumsum xs.1 dim, ())
        backward := fun _ g => (Tensor.cumsumBackward g.1 dim, ())
        jvp? := some fun _ dxs => (Tensor.cumsum dxs.1 dim, ()) } :=
  rfl

/-- The rule of `Op.softmax`. -/
theorem rule?_softmax [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.softmax dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.softmax xs.1 dim, ())
        backward := fun xs g => (Tensor.softmaxBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs => (Tensor.softmaxBackward xs.1 dxs.1 dim, ()) } :=
  rfl

/-- The rule of `Op.logSoftmax`. -/
theorem rule?_logSoftmax [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.logSoftmax dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.logSoftmax xs.1 dim, ())
        backward := fun xs g =>
          (Tensor.logSoftmaxBackward (Tensor.logSoftmax xs.1 dim) g.1 dim, ())
        prepare := fun xs flags =>
          let y := Tensor.logSoftmax xs.1 dim
          ⟨(y, ()), fun g =>
            (if flags.1 then some (Tensor.logSoftmaxBackward y g.1 dim) else none, ())⟩
        prepare_eq := by intros; rfl
        jvp? := some fun xs dxs =>
          (Tensor.logSoftmaxJVP (Tensor.logSoftmax xs.1 dim) dxs.1 dim, ()) } :=
  rfl

/-- The rule of `Op.sort`. -/
theorem rule?_sort [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (descending : Bool) :
    (Op.sort dim descending).rule? (α := α) = some
      { forward := fun xs =>
          let sorted := Tensor.sort xs.1 dim descending
          (sorted.1, sorted.2, ())
        backward := fun xs g => (Tensor.sortBackward xs.1 g.1 dim descending, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.sort xs.1 dim descending).2 c)), (), ()) } :=
  rfl

/-- The rule of `Op.topk`. -/
theorem rule?_topk [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (k : Nat) (largest : Bool)
    (h : k ≤ s.extent dim) :
    (Op.topk dim k largest h).rule? (α := α) = some
      { forward := fun xs =>
          let top := Tensor.topk xs.1 dim k largest h
          (top.1, top.2, ())
        backward := fun xs g => (Tensor.topkBackward xs.1 dim k largest h g.1, ())
        jvp? := some fun xs dxs =>
          (Tensor.slice (Tensor.Rep.ofFn fun c =>
              dxs.1 (Shape.Coord.update c dim ((Tensor.sort xs.1 dim largest).2 c)))
            [dim] [0] [k] [1]
            (by simpa using Shape.sliceValid_singleton s dim 0 k (by simpa using h)), (), ()) } :=
  rfl

/-- The rule of `Op.cumprod`. -/
theorem rule?_cumprod [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.cumprod dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.cumprod xs.1 dim, ())
        backward := fun xs g => (Tensor.cumprodBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs => (Tensor.cumprodJVP xs.1 dxs.1 dim, ()) } :=
  rfl

/-- The rule of `Op.cummax`. -/
theorem rule?_cummax [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.cummax dim).rule? (α := α) = some
      { forward := fun xs =>
          let running := Tensor.cummax xs.1 dim
          (running.1, running.2, ())
        backward := fun xs g => (Tensor.cummaxBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.cummax xs.1 dim).2 c)), (), ()) } :=
  rfl

/-- The rule of `Op.cummin`. -/
theorem rule?_cummin [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.cummin dim).rule? (α := α) = some
      { forward := fun xs =>
          let running := Tensor.cummin xs.1 dim
          (running.1, running.2, ())
        backward := fun xs g => (Tensor.cumminBackward xs.1 g.1 dim, ())
        jvp? := some fun xs dxs =>
          (Tensor.Rep.ofFn fun c =>
            dxs.1 (Shape.Coord.update c dim ((Tensor.cummin xs.1 dim).2 c)), (), ()) } :=
  rfl

/-- The rule of `Op.argmax`. -/
theorem rule?_argmax [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (keepdim : Bool)
    (h : 0 < s.extent dim) :
    (Op.argmax dim keepdim h).rule? (α := α) = some
      { forward := fun xs => (Tensor.argmax xs.1 dim keepdim h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.argmin`. -/
theorem rule?_argmin [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (keepdim : Bool)
    (h : 0 < s.extent dim) :
    (Op.argmin dim keepdim h).rule? (α := α) = some
      { forward := fun xs => (Tensor.argmin xs.1 dim keepdim h, ())
        backward := fun _ _ => ((), ()) } :=
  rfl

/-- The rule of `Op.reduce`. -/
theorem rule?_reduce [Op.Capabilities α] {s : Shape} (op : Tensor.ReduceOp)
    (dims : List (Fin s.rank)) (keepdim : Bool)
    (h : op = .max ∨ op = .min → ∀ d ∈ dims, 0 < s.extent d) :
    (Op.reduce op dims keepdim h).rule? (α := α) = some
      { forward := fun xs => (match op, h with
          | .sum, _ => Tensor.reduce .sum xs.1 dims keepdim
          | .prod, _ => Tensor.reduce .prod xs.1 dims keepdim
          | .max, h => Tensor.reduce .max xs.1 dims keepdim fun _ => h (Or.inl rfl)
          | .min, h => Tensor.reduce .min xs.1 dims keepdim fun _ => h (Or.inr rfl), ())
        backward := fun xs g => (Tensor.reduceBackward op xs.1 dims keepdim h g.1, ())
        jvp? := match op with
          | .sum => some fun _ dxs => (Tensor.reduce .sum dxs.1 dims keepdim, ())
          | _ => some fun xs dxs => (Tensor.reduceJVP _ xs.1 dims keepdim h dxs.1, ()) } :=
  rfl

/-- The rule of `Op.mean`. -/
theorem rule?_mean [Op.Capabilities α] {s : Shape} (dims : List (Fin s.rank)) (keepdim : Bool) :
    (Op.mean dims keepdim).rule? (α := α) = some
      { forward := fun xs => (Tensor.mean xs.1 dims keepdim, ())
        backward := fun _ g => (Tensor.meanBackward dims keepdim g.1, ())
        jvp? := some fun _ dxs => (Tensor.mean dxs.1 dims keepdim, ()) } :=
  rfl

/-- The rule of `Op.maskedSoftmax`. -/
theorem rule?_maskedSoftmax [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) :
    (Op.maskedSoftmax dim).rule? (α := α) = some
      { forward := fun xs => (Tensor.softmax xs.1 dim (some xs.2.1), ())
        backward := fun xs g => (Tensor.maskedSoftmaxBackward xs.1 xs.2.1 g.1 dim, (), ())
        jvp? := some fun xs dxs => (Tensor.maskedSoftmaxBackward xs.1 xs.2.1 dxs.1 dim, ()) } :=
  rfl

/-- The rule of `Op.topKMask`. -/
theorem rule?_topKMask [Op.Capabilities α] {s : Shape} (dim : Fin s.rank) (k : Nat) :
    (Op.topKMask dim k).rule? (α := α) = some
      { forward := fun xs => (MoE.topKMask k xs.1 dim, ())
        backward := fun _ _ => ((), ())
        jvp? := some fun _ _ => (Tensor.full _ 0, ()) } :=
  rfl

/-- The rule of `Op.selectiveScan`. -/
theorem rule?_selectiveScan [Op.Capabilities α] {s : Shape} (dim : Fin s.rank)
    (sharedA sharedB : Bool) :
    (Op.selectiveScan dim sharedA sharedB).rule? (α := α) = some
      { forward := fun xs =>
          (Tensor.selectiveScan dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1, ())
        backward := fun xs g =>
          let gradients :=
            Tensor.selectiveScanBackward dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1 g.1
          (gradients.1, gradients.2.1, gradients.2.2.1, gradients.2.2.2, ())
        jvp? := some fun xs dxs =>
          (Tensor.selectiveScanJVP dim sharedA sharedB xs.1 xs.2.1 xs.2.2.1 xs.2.2.2.1 dxs.1
            dxs.2.1 dxs.2.2.1 dxs.2.2.2.1, ()) } :=
  rfl

end Op

end TorchLean
