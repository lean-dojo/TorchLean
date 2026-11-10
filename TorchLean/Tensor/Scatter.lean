/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Movement

/-!
# Scattering

`scatter reduce base dims indices updates` writes the entries of `updates` into `base` at the
positions in `indices`, the transpose of `gather`. Each entry of `indices` holds one index for each
axis in `dims`, or is `none` to drop its updates, as in JAX's drop mode. Each output entry is one
left fold over the coordinates of `indices` in row-major order, seeded by the entry of `base`:
`.sum` adds every matching update, `.replace` keeps the last one, `.prod` multiplies, and `.max`
and `.min` take `Context.maximum` and `Context.minimum` (IEEE 754-2019 maximum and minimum in a
binary format, `Float` and `Float32` included: a NaN propagates and `-0` is below `+0`, whatever
the order of the participants).

`scatter` is the transpose of `gather`, written by hand. Its fold starts from the base entry, as
`include_self=True` does in `torch.Tensor.scatter_reduce_` (PyTorch 2.8 docs), and the capture
builds `include_self=False` from a replacing scatter and this fold (`python/torchlean/_emit.py`).
The `torch.Tensor.scatter_` page lets repeated indices pick a value "arbitrarily" and send the
gradient "to all locations in the source that correspond to the same index". The last-writer rule
settles both: `[1, 2, 3]` scattered into one entry gives `3`, and an upstream `8` there sends
`[0, 0, 8]` back, where torch 2.8.0 sends `[8, 8, 8]`. At a tie of `-0` and `0` torch's `amax`
and `amin` keep whichever they meet first. The capture test
`test_topk_pool_indices_and_scatter_folds_match_eager` in `scripts/checks/test_ir_capture.py`
compares the `prod`, `amax` and `amin` folds with eager PyTorch where the two tie rules agree.
Torch 2.8.0 gives the gradients that the guards of `TorchLeanTest/OpTable.lean` hold, in binary64
and binary32.

* `ScatterReduce` and `ScatterReduce.Scalar` are the combining rules and their scalar step. A sum,
  product, maximum or minimum steps as the reduction of the same name, `ReduceOp.Scalar.step`.
* `scatter_apply` reads an output entry as its fold.
* `scatterUpdatesBackward` routes the upstream gradient to the updates at total index tuples: each
  update reads it at the entry it lands on, and under `.replace` only an update that no later one
  overwrites receives it.
* `scatterProdBackward` and `scatterExtremumBackward` are the pullbacks of a product and of a
  maximum or minimum at total index tuples, to the base and to the updates.
-/

@[expose] public section

namespace TorchLean.Tensor

/-- How `scatter` combines an update with the current entry. -/
inductive ScatterReduce
  /-- The update replaces the entry. -/
  | replace
  /-- The update is added to the entry. -/
  | sum
  /-- The entry is multiplied by the update. -/
  | prod
  /-- The maximum of the entry and the update, `Context.maximum acc update`: IEEE maximum in a
  binary format, the meaning of PyTorch's `scatter_reduce` "amax", JAX's `scatter_max` and the
  IR's `max_elem`. -/
  | max
  /-- The minimum of the entry and the update, `Context.minimum acc update`: IEEE minimum in a
  binary format, the meaning of PyTorch's `scatter_reduce` "amin", JAX's `scatter_min` and the
  IR's `min_elem`. -/
  | min
  deriving DecidableEq, Repr

/-- The scalar step of a scatter rule, which `scatter` applies once for each matching update. -/
class ScatterReduce.Scalar (op : ScatterReduce) (α : Type) where
  /-- Combine the current entry with an update. -/
  combine : α → α → α

/-- Replacing keeps the update. -/
instance ScatterReduce.scalarReplace {α : Type} : ScatterReduce.Scalar .replace α :=
  ⟨fun _ update => update⟩

/-- Summing takes the step of the sum reduction: the current entry plus the update. -/
instance ScatterReduce.scalarSum {α : Type} [ReduceOp.Scalar .sum α] :
    ScatterReduce.Scalar .sum α :=
  ⟨ReduceOp.Scalar.step (op := .sum)⟩

/-- A product takes the step of the product reduction: the current entry times the update. -/
instance ScatterReduce.scalarProd {α : Type} [ReduceOp.Scalar .prod α] :
    ScatterReduce.Scalar .prod α :=
  ⟨ReduceOp.Scalar.step (op := .prod)⟩

/-- A maximum takes the step of the maximum reduction, which over a `Context` scalar is
`Context.maximum` of the current entry and the update. -/
instance ScatterReduce.scalarMax {α : Type} [ReduceOp.Scalar .max α] :
    ScatterReduce.Scalar .max α :=
  ⟨ReduceOp.Scalar.step (op := .max)⟩

/-- A minimum takes the step of the minimum reduction, which over a `Context` scalar is
`Context.minimum` of the current entry and the update. -/
instance ScatterReduce.scalarMin {α : Type} [ReduceOp.Scalar .min α] :
    ScatterReduce.Scalar .min α :=
  ⟨ReduceOp.Scalar.step (op := .min)⟩

variable {α : Type} [Storage α] {s : Shape}

/-- Scatter `updates` into `base` at the positions in `indices` along the axes `dims`. The output
entry at `c` folds, over the coordinates `u` of `indices` in row-major order and from `base c`,
the update at `u` and the other axes of `c` whenever `indices u` is the index of `c` along `dims`.
-/
def scatter (reduce : ScatterReduce) [ScatterReduce.Scalar reduce α] (base : Tensor α s)
    (dims : List (Fin s.rank)) {t : Shape}
    (indices : Tensor (Option (Einops.AxisTuple s.extent dims)) t)
    (updates : Tensor α (t ++ s.eraseAxes dims)) : Tensor α s :=
  Rep.ofFn fun c =>
    (List.finRange (Shape.size t)).foldl
      (fun acc i =>
        if indices (Shape.Coord.unlinearize i) =
            some fun k => Shape.Coord.get c (dims.get k) then
          ScatterReduce.Scalar.combine (op := reduce) acc
            (updates ((Shape.Coord.appendEquiv t (s.eraseAxes dims)).symm
              (Shape.Coord.unlinearize i,
                (Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm
                  fun k => Shape.Coord.get c ((s.otherAxes dims).get k))))
        else acc)
      (base c)

/-- An entry of `scatter` folds the matching updates into the entry of `base`, in row-major order
of the index tensor. -/
@[simp] theorem scatter_apply (reduce : ScatterReduce) [ScatterReduce.Scalar reduce α]
    (base : Tensor α s) (dims : List (Fin s.rank)) {t : Shape}
    (indices : Tensor (Option (Einops.AxisTuple s.extent dims)) t)
    (updates : Tensor α (t ++ s.eraseAxes dims)) (c : s.Coord) :
    scatter reduce base dims indices updates c =
      (List.finRange (Shape.size t)).foldl
        (fun acc i =>
          if indices (Shape.Coord.unlinearize i) =
              some fun k => Shape.Coord.get c (dims.get k) then
            ScatterReduce.Scalar.combine (op := reduce) acc
              (updates ((Shape.Coord.appendEquiv t (s.eraseAxes dims)).symm
                (Shape.Coord.unlinearize i,
                  (Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm
                    fun k => Shape.Coord.get c ((s.otherAxes dims).get k))))
          else acc)
        (base c) :=
  Rep.get_ofFn _ c

/-- The upstream gradient `g` routed to the updates of `scatter rule base dims indices updates`,
for total index tuples: each update reads `g` at the entry it lands on. A replacement routes `g`
only to an update that no later update in row-major order of the index tensor overwrites, the
others zero. It is the pullback to the updates of a sum and of a replacement, and the read of
`g` at the landing entries that the product and extremum pullbacks scale. -/
def scatterUpdatesBackward [Zero α] (rule : ScatterReduce) (dims : List (Fin s.rank))
    (h : dims.Nodup) {t : Shape} (indices : Tensor (Einops.AxisTuple s.extent dims) t)
    (g : Tensor α s) : Tensor α (t ++ s.eraseAxes dims) :=
  let read := gather g dims [] indices (t ++ s.eraseAxes dims) ⟨by simpa using h, by simp, by simp⟩
    (by simp)
  match rule with
  | .sum | .prod | .max | .min => read
  | .replace => Rep.ofFn fun u =>
      let i := (Shape.Coord.appendEquiv t (s.eraseAxes dims) u).1
      if (List.finRange (Shape.size t)).all fun j =>
          decide (j ≤ Shape.Coord.linearize i) ||
            !decide (indices (Shape.Coord.unlinearize j) = indices i) then read u
      else 0

/-- Whether an entry ties an extremum: equal by the scalar's `==`, or both NaN. -/
def extremumTie [Context α] (entry extremum : α) : Bool :=
  entry == extremum || (Context.isNaN entry && Context.isNaN extremum)

/-- The pullback of `scatter .prod base dims indices updates` at `g` for total index tuples, to the
base and to the updates. A base entry receives `g` times the updates landing there, in row-major
order of the index tensor (`scatter .prod g`). An update receives `g` where it lands times the
product of the other participants there: the base entry, then the other updates landing there in
row-major order. The update's own occurrence is skipped, so a zero needs no division. -/
def scatterProdBackward [Mul α] [Zero α] [One α] (base : Tensor α s) (dims : List (Fin s.rank))
    (h : dims.Nodup) {t : Shape} (indices : Tensor (Einops.AxisTuple s.extent dims) t)
    (updates : Tensor α (t ++ s.eraseAxes dims)) (g : Tensor α s) :
    Tensor α s × Tensor α (t ++ s.eraseAxes dims) :=
  let split := Shape.Coord.appendEquiv t (s.eraseAxes dims)
  (scatter .prod g dims (map some indices) updates,
    Rep.ofFn fun u =>
      let i := (split u).1
      scatterUpdatesBackward .sum dims h indices g u *
        (List.finRange (Shape.size t)).foldl
          (fun acc j =>
            if j ≠ Shape.Coord.linearize i ∧
                indices (Shape.Coord.unlinearize j) = indices i then
              acc * updates (split.symm (Shape.Coord.unlinearize j, (split u).2))
            else acc)
          (scatterUpdatesBackward .sum dims h indices base u))

/-- The pullback at `g`, for total index tuples, of a scatter maximum or minimum of `base dims
indices updates` whose output is `y` (`scatter .max` or `.min`), to the base and to the updates.
The participants of an output entry, its base entry and the updates landing there,
that tie it (`extremumTie`: equal by `==`, or both NaN) share `g` evenly: `g` divided
once by their count, taken in `Nat` (`Context.divNat`). The others receive zero. The caller
passes the forward output of the same base, indices and updates as `y`. Where some participant
ties each output entry (both rules on IEEE formats: the result is a participant, or a NaN, which
a NaN participant ties), the count is at least one. An unrelated `y` can make it zero. It is a
total pullback convention, an even split at ties, not a derivative there. -/
def scatterExtremumBackward [Context α] (y base : Tensor α s) (dims : List (Fin s.rank))
    (h : dims.Nodup) {t : Shape} (indices : Tensor (Einops.AxisTuple s.extent dims) t)
    (updates : Tensor α (t ++ s.eraseAxes dims)) (g : Tensor α s) :
    Tensor α s × Tensor α (t ++ s.eraseAxes dims) :=
  let split := Shape.Coord.appendEquiv t (s.eraseAxes dims)
  let share : Tensor α s := Rep.ofFn fun c =>
    let rest := (Einops.AxisTuple.coordEquiv s.extent (s.otherAxes dims)).symm
      fun k => Shape.Coord.get c ((s.otherAxes dims).get k)
    let ties := (List.finRange (Shape.size t)).countP fun j =>
      decide (indices (Shape.Coord.unlinearize j) = fun k => Shape.Coord.get c (dims.get k)) &&
        extremumTie (updates (split.symm (Shape.Coord.unlinearize j, rest))) (y c)
    Context.divNat (g c) (if extremumTie (base c) (y c) then ties + 1 else ties)
  let landed := scatterUpdatesBackward .sum dims h indices y
  let shared := scatterUpdatesBackward .sum dims h indices share
  (Rep.ofFn fun c => if extremumTie (base c) (y c) then share c else 0,
    Rep.ofFn fun u => if extremumTie (updates u) (landed u) then shared u else 0)

end TorchLean.Tensor
