/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.LinearAlgebra.Identity
public import TorchLean.Tensor.Reduce.Backward

/-!
# Contraction over paired axes

`contract x y dims batchDims out h` multiplies `x` and `y` entrywise along the paired axes `dims`
and sums the products, at any rank. Each output entry is one left fold from zero over the
coordinates of the contracted axes, in row-major order with the first pair outermost, and each
step adds `x _ * y _`. The batch pairs `batchDims` index both operands at once. The output shape
`out` is an argument, with the proof `out = Shape.contract s t dims batchDims` that `tensor_shape`
gives by default, so a caller with literal shapes gets a tensor of the shape it spells.

* `Shape.ContractValid` states that the pairs name each axis of an operand at most once and that
  paired axes have equal extents. Its proofs `.matVec`, `.vecMat` and `.matMul` select the matrix
  products, which contract the axis pairs `Shape.matVecDims`, `Shape.vecMatDims` and
  `Shape.matMulDims` with the batch list `Shape.noBatch`. The axis lists are definitions, so `simp`
  leaves them intact and the read lemmas keep matching.
* `Shape.contract` is the computed output shape: the batch axes, then the other axes of `x`, then
  those of `y`.
* `Shape.contractLeft` and `Shape.ContractValid.contractRight` are the coordinates read in `x`
  and `y`.
* `contract_apply` reads an output entry as its fold. `contract_matVec_apply`,
  `contract_vecMat_apply` and `contract_matMul_apply` read the matrix products as their loops over
  the shared axis.
* `matmul` is the matrix product, and `batchMatmul` multiplies matrices over broadcast batch
  prefixes. `Shape.matmulLeft`, `Shape.matmulRight` and `Shape.matmulOut` are the operand and
  result shapes of a product whose vector operands are one-row or one-column matrices, as in
  `torch.matmul`.
* `contractBackwardLeft` and `contractBackwardRight` are the pullbacks of `contract` to each
  operand: an entry is one left fold from zero over the other operand's other axes in row-major
  order, at the output coordinates `Shape.contractOutputLeft` and
  `Shape.ContractValid.contractOutputRight`. `Shape.Coord.restrict` reads the indices of a
  coordinate along a list of axes.

We wrote `contract` by hand from the index form of a contraction, the sum that `torch.tensordot`
(PyTorch 2.8 docs) computes, with the other axes of `x` before those of `y`. The batch axes come
first, as in the batched `torch.matmul`. Each product is rounded before its add. On our machine
(MKL, AVX-512) torch 2.8.0's CPU `matmul` rounds once per multiply-add: for 200 random pairs of
binary32 vectors of length 2 its dot equals a fused left fold every time, and ours in 150 of them.
At `[[-0]] @ [[1]]` both give `+0`, since our fold starts from `+0`. The capture test
`test_data_movement_and_contraction` in `scripts/checks/test_ir_capture.py` compares a batched
`jnp.einsum` with eager JAX on integer-valued binary32 inputs, and
`test_softmax_family_and_products` runs `torch.bmm` and `torch.matmul` without comparing the
results with eager PyTorch.
-/

@[expose] public section

open TorchLean.Shape (rank rank_eq_length size size_append)

namespace TorchLean.Shape

/-- Valid axis pairs for `contract x y dims batchDims`: the contracted and batch axes of each
operand are distinct, and paired axes have equal extents. -/
structure ContractValid (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank)) :
    Prop where
  /-- The batch and contracted axes of the left operand are distinct. -/
  nodup_left : ((batchDims ++ dims).map Prod.fst).Nodup
  /-- The batch and contracted axes of the right operand are distinct. -/
  nodup_right : ((batchDims ++ dims).map Prod.snd).Nodup
  /-- Paired axes have equal extents. -/
  extent_eq : ∀ p ∈ batchDims ++ dims, s.extent p.1 = t.extent p.2

/-- The output shape of `contract`: the batch axes, then the other axes of `s`, then the other
axes of `t`, each in their order. -/
def contract (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank)) : Shape :=
  (batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst) ++
    t.eraseAxes ((batchDims ++ dims).map Prod.snd)

/-- The index along an axis of `t` paired with a listed axis of `s` is below its extent. -/
theorem Coord.index_idxOf_snd_lt {s t : Shape} {l : List (Fin s.rank × Fin t.rank)}
    (r : Coord ((l.map Prod.fst).map s.extent)) (he : ∀ p ∈ l, s.extent p.1 = t.extent p.2)
    {b : Fin t.rank} (hb : b ∈ l.map Prod.snd) :
    Coord.index r ((l.map Prod.snd).idxOf b) < t.extent b := by
  have hk : (l.map Prod.snd).idxOf b < (l.map Prod.snd).length :=
    List.idxOf_lt_length_iff.mpr hb
  have hkl : (l.map Prod.snd).idxOf b < l.length := by simpa using hk
  have hr : (l.map Prod.snd).idxOf b < Shape.rank ((l.map Prod.fst).map s.extent) := by
    simpa [Shape.rank_eq_length] using hk
  have hlt := Coord.index_lt r hr
  have hb' : (l[(l.map Prod.snd).idxOf b]'hkl).2 = b := by
    have := List.getElem_idxOf hk
    rw [List.getElem_map] at this
    exact this
  have he' : extent ((l.map Prod.fst).map s.extent) ⟨_, hr⟩ = t.extent b := by
    have h₁ := he _ (List.getElem_mem hkl)
    rw [hb'] at h₁
    simp only [extent, List.getElem_map] at h₁ ⊢
    exact h₁
  exact lt_of_lt_of_eq hlt he'

/-- The coordinate of `s` read by `contract` at the output coordinate `o` and the contracted
coordinate `r`: the indices of `r` along the contracted axes, of the batch part of `o` along the
batch axes, and of the part of `o` for `s` along the other axes. -/
def contractLeft (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank))
    (o : (s.contract t dims batchDims).Coord) (r : Coord ((dims.map Prod.fst).map s.extent)) :
    s.Coord :=
  let parts := Coord.appendEquiv
    ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
    (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) o
  let lead := Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
    (s.eraseAxes ((batchDims ++ dims).map Prod.fst)) parts.1
  Coord.ofFun fun a =>
    if ha : a ∈ dims.map Prod.fst then
      ⟨Coord.index r ((dims.map Prod.fst).idxOf a), Coord.index_idxOf_lt r ha⟩
    else if hb : a ∈ batchDims.map Prod.fst then
      ⟨Coord.index lead.1 ((batchDims.map Prod.fst).idxOf a), Coord.index_idxOf_lt lead.1 hb⟩
    else
      ⟨Coord.index lead.2 ((s.otherAxes ((batchDims ++ dims).map Prod.fst)).idxOf a),
        Coord.index_idxOf_lt (l := s.otherAxes ((batchDims ++ dims).map Prod.fst)) lead.2
          (mem_otherAxes.mpr (by
            simp only [List.map_append, List.mem_append, not_or]
            exact ⟨hb, ha⟩))⟩

/-- The coordinate of `t` read by `contract` at the output coordinate `o` and the contracted
coordinate `r`: the indices of `r` along the paired axes, of the batch part of `o` along the batch
axes, and of the part of `o` for `t` along the other axes. -/
def ContractValid.contractRight {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : ContractValid s t dims batchDims) (o : (s.contract t dims batchDims).Coord)
    (r : Coord ((dims.map Prod.fst).map s.extent)) : t.Coord :=
  let parts := Coord.appendEquiv
    ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
    (t.eraseAxes ((batchDims ++ dims).map Prod.snd)) o
  let lead := Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
    (s.eraseAxes ((batchDims ++ dims).map Prod.fst)) parts.1
  Coord.ofFun fun b =>
    if hb : b ∈ dims.map Prod.snd then
      ⟨Coord.index r ((dims.map Prod.snd).idxOf b),
        Coord.index_idxOf_snd_lt r (fun p hp => h.extent_eq p (List.mem_append_right _ hp)) hb⟩
    else if hc : b ∈ batchDims.map Prod.snd then
      ⟨Coord.index lead.1 ((batchDims.map Prod.snd).idxOf b),
        Coord.index_idxOf_snd_lt lead.1 (fun p hp => h.extent_eq p (List.mem_append_left _ hp))
          hc⟩
    else
      ⟨Coord.index parts.2 ((t.otherAxes ((batchDims ++ dims).map Prod.snd)).idxOf b),
        Coord.index_idxOf_lt (l := t.otherAxes ((batchDims ++ dims).map Prod.snd)) parts.2
          (mem_otherAxes.mpr (by
            simp only [List.map_append, List.mem_append, not_or]
            exact ⟨hc, hb⟩))⟩

/-- The indices of `c` along the axes `l`, in the order of `l`. -/
def Coord.restrict {s : Shape} (c : s.Coord) (l : List (Fin s.rank)) : Coord (l.map s.extent) :=
  Coord.ofFun fun k =>
    ⟨(Coord.get c (l.get (k.cast (by simp [Shape.rank_eq_length])))).val, by
      have := (Coord.get c (l.get (k.cast (by simp [Shape.rank_eq_length])))).isLt
      simpa [extent] using this⟩

/-- The `k`-th index of `c.restrict l` is the index of `c` along `l[k]`. -/
@[simp] theorem Coord.index_restrict {s : Shape} (c : s.Coord) (l : List (Fin s.rank)) (k : Nat)
    (hk : k < l.length) : (c.restrict l).index k = (Coord.get c l[k]).val := by
  simp [Coord.restrict, Coord.index_ofFun, Shape.rank_eq_length, hk]

/-- The batch axes of `t` have the extents of the batch axes of `s` they are paired with. -/
theorem ContractValid.batch_extents {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : ContractValid s t dims batchDims) :
    (batchDims.map Prod.snd).map t.extent = (batchDims.map Prod.fst).map s.extent := by
  simp only [List.map_map]
  exact List.map_congr_left fun p hp => (h.extent_eq p (List.mem_append_left _ hp)).symm

/-- The contracted axes of `t` have the extents of the axes of `s` they are paired with. -/
theorem ContractValid.dims_extents {s t : Shape} {dims batchDims : List (Fin s.rank × Fin t.rank)}
    (h : ContractValid s t dims batchDims) :
    (dims.map Prod.snd).map t.extent = (dims.map Prod.fst).map s.extent := by
  simp only [List.map_map]
  exact List.map_congr_left fun p hp => (h.extent_eq p (List.mem_append_right _ hp)).symm

/-- The coordinate of `contract x y dims batchDims` that reads `x` at `c`, at the coordinate `j` of
the other axes of `y`: the batch and other indices of `c`, then `j`. -/
def contractOutputLeft (s t : Shape) (dims batchDims : List (Fin s.rank × Fin t.rank))
    (c : s.Coord) (j : Coord (t.eraseAxes ((batchDims ++ dims).map Prod.snd))) :
    (s.contract t dims batchDims).Coord :=
  (Coord.appendEquiv
      ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
      (t.eraseAxes ((batchDims ++ dims).map Prod.snd))).symm
    ((Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
      (s.eraseAxes ((batchDims ++ dims).map Prod.fst))).symm
      (Coord.restrict c (batchDims.map Prod.fst),
        Coord.restrict c (s.otherAxes ((batchDims ++ dims).map Prod.fst))), j)

/-- The coordinate of `contract x y dims batchDims` that reads `y` at `r`, at the coordinate `i` of
the other axes of `x`: the batch indices of `r`, then `i`, then the other indices of `r`. -/
def ContractValid.contractOutputRight {s t : Shape}
    {dims batchDims : List (Fin s.rank × Fin t.rank)} (h : ContractValid s t dims batchDims)
    (r : t.Coord) (i : Coord (s.eraseAxes ((batchDims ++ dims).map Prod.fst))) :
    (s.contract t dims batchDims).Coord :=
  (Coord.appendEquiv
      ((batchDims.map Prod.fst).map s.extent ++ s.eraseAxes ((batchDims ++ dims).map Prod.fst))
      (t.eraseAxes ((batchDims ++ dims).map Prod.snd))).symm
    ((Coord.appendEquiv ((batchDims.map Prod.fst).map s.extent)
      (s.eraseAxes ((batchDims ++ dims).map Prod.fst))).symm
      (Coord.reindexEq h.batch_extents (Coord.restrict r (batchDims.map Prod.snd)), i),
      Coord.restrict r (t.otherAxes ((batchDims ++ dims).map Prod.snd)))

/-- The empty list of batch axis pairs of a contraction of `s` with `t`. The matrix products use it
in place of a literal `[]`, whose axis types `simp` would rewrite. -/
def noBatch (s t : Shape) : List (Fin s.rank × Fin t.rank) := []

/-- The contracted axis pair of a matrix-vector product: the columns of the matrix with the
vector. -/
def matVecDims (m n : Nat) : List (Fin (Shape.rank [m, n]) × Fin (Shape.rank [n])) :=
  [(⟨1, by simp [Shape.rank_eq_length]⟩, ⟨0, by simp [Shape.rank_eq_length]⟩)]

/-- A matrix-vector product contracts the columns of the matrix with the vector. -/
theorem ContractValid.matVec {m n :
    Nat} : ContractValid [m, n] [n] (matVecDims m n) (noBatch [m, n] [n]) where
  nodup_left := List.nodup_singleton _
  nodup_right := List.nodup_singleton _
  extent_eq := by
    intro q hq
    simp only [noBatch, matVecDims, List.nil_append, List.mem_singleton] at hq
    subst hq
    rfl

/-- The contracted axis pair of a vector-matrix product: the vector with the rows of the matrix. -/
def vecMatDims (m n : Nat) : List (Fin (Shape.rank [m]) × Fin (Shape.rank [m, n])) :=
  [(⟨0, by simp [Shape.rank_eq_length]⟩, ⟨0, by simp [Shape.rank_eq_length]⟩)]

/-- A vector-matrix product contracts the vector with the rows of the matrix. -/
theorem ContractValid.vecMat {m n :
    Nat} : ContractValid [m] [m, n] (vecMatDims m n) (noBatch [m] [m, n]) where
  nodup_left := List.nodup_singleton _
  nodup_right := List.nodup_singleton _
  extent_eq := by
    intro q hq
    simp only [noBatch, vecMatDims, List.nil_append, List.mem_singleton] at hq
    subst hq
    rfl

/-- The contracted axis pair of a matrix product: the columns of the left matrix with the rows of
the right. -/
def matMulDims (m n p : Nat) : List (Fin (Shape.rank [m, n]) × Fin (Shape.rank [n, p])) :=
  [(⟨1, by simp [Shape.rank_eq_length]⟩, ⟨0, by simp [Shape.rank_eq_length]⟩)]

/-- Matrix multiplication contracts the columns of the left matrix with the rows of the right. -/
theorem ContractValid.matMul {m n p :
    Nat} : ContractValid [m, n] [n, p] (matMulDims m n p) (noBatch [m, n] [n, p]) where
  nodup_left := List.nodup_singleton _
  nodup_right := List.nodup_singleton _
  extent_eq := by
    intro q hq
    simp only [noBatch, matMulDims, List.nil_append, List.mem_singleton] at hq
    subst hq
    rfl

/-- The left operand of a matrix product: `batch ++ [m, n]`, or `batch ++ [n]` for a vector. -/
def matmulLeft (lv : Bool) (batch : Shape) (m n : Nat) : Shape :=
  batch ++ (bif lv then [n] else [m, n])

/-- The right operand of a matrix product: `batch ++ [n, p]`, or `batch ++ [n]` for a vector. -/
def matmulRight (rv : Bool) (batch : Shape) (n p : Nat) : Shape :=
  batch ++ (bif rv then [n] else [n, p])

/-- The result of a matrix product: `batch ++ [m, p]` without the axis of a vector operand. -/
def matmulOut (lv rv : Bool) (batch : Shape) (m p : Nat) : Shape :=
  batch ++ ((bif lv then [] else [m]) ++ (bif rv then [] else [p]))

/-- A left vector has the entries of a one-row matrix. -/
theorem size_matmulLeft {lv : Bool} {m : Nat} (hl : lv = true → m = 1) (batch : Shape)
    (n : Nat) : (matmulLeft lv batch m n).size = (batch ++ [m, n]).size := by
  cases lv with
  | false => rfl
  | true =>
      rw [hl rfl, matmulLeft, Shape.size_append, Shape.size_append]
      simp

/-- A right vector has the entries of a one-column matrix. -/
theorem size_matmulRight {rv : Bool} {p : Nat} (hr : rv = true → p = 1) (batch : Shape)
    (n : Nat) : (matmulRight rv batch n p).size = (batch ++ [n, p]).size := by
  cases rv with
  | false => rfl
  | true =>
      rw [hr rfl, matmulRight, Shape.size_append, Shape.size_append]
      simp

/-- Leaving out the axis of a vector operand keeps the number of entries. -/
theorem size_matmulOut {lv rv : Bool} {m p : Nat} (hl : lv = true → m = 1)
    (hr : rv = true → p = 1) (batch : Shape) :
    (matmulOut lv rv batch m p).size = (batch ++ [m, p]).size := by
  rw [matmulOut, Shape.size_append, Shape.size_append]
  cases lv <;> cases rv <;> simp_all

end TorchLean.Shape

namespace TorchLean.Tensor

variable {α : Type} [Storage α] {s t : Shape}

/-- Contract `x` and `y` along the paired axes `dims`, with the batch pairs `batchDims`, into the
shape `out`. Each output entry adds the products `x _ * y _` to zero over the coordinates of the
contracted axes, in row-major order with the first pair outermost. -/
def contract [Add α] [Mul α] [Zero α] (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims)
    (hout : out = s.contract t dims batchDims := by tensor_shape) : Tensor α out :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size ((dims.map Prod.fst).map s.extent))).foldl
      (fun acc i =>
        acc + x (s.contractLeft t dims batchDims (Shape.Coord.reindexEq hout o)
            (Shape.Coord.unlinearize i)) *
          y (h.contractRight (Shape.Coord.reindexEq hout o) (Shape.Coord.unlinearize i)))
      0

/-- An entry of a contraction adds the products of the entries it pairs to zero in row-major order
of the contracted axes. -/
@[simp] theorem contract_apply [Add α] [Mul α] [Zero α] (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims) (hout : out = s.contract t dims batchDims)
    (o : out.Coord) :
    contract x y dims batchDims out h hout o =
      (List.finRange (Shape.size ((dims.map Prod.fst).map s.extent))).foldl
        (fun acc i =>
          acc + x (s.contractLeft t dims batchDims (Shape.Coord.reindexEq hout o)
              (Shape.Coord.unlinearize i)) *
            y (h.contractRight (Shape.Coord.reindexEq hout o) (Shape.Coord.unlinearize i)))
        0 :=
  Rep.get_ofFn _ o

/-- The pullback of `contract x y dims batchDims` to `x` at `g`. The entry at `c` adds `g o * y r`
to zero over the coordinates `j` of the other axes of `y` in row-major order, where `o` is the
output coordinate `contractOutputLeft c j` and `r` is the coordinate of `y` that `contract` reads
at `o` with the contracted indices of `c`: the batch indices of `c`, the contracted indices of `c`
at the paired axes and `j` at the other axes. -/
def contractBackwardLeft [Add α] [Mul α] [Zero α] (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) (g : Tensor α (s.contract t dims batchDims))
    (y : Tensor α t) : Tensor α s :=
  Rep.ofFn fun c =>
    (List.finRange (Shape.size (t.eraseAxes ((batchDims ++ dims).map Prod.snd)))).foldl
      (fun acc j =>
        acc + g (Shape.contractOutputLeft s t dims batchDims c (Shape.Coord.unlinearize j)) *
          y (h.contractRight (Shape.contractOutputLeft s t dims batchDims c
            (Shape.Coord.unlinearize j)) (Shape.Coord.restrict c (dims.map Prod.fst))))
      0

/-- An entry of the pullback of `contract` to `x` is its fold over the other axes of `y`. -/
@[simp] theorem contractBackwardLeft_apply [Add α] [Mul α] [Zero α]
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.ContractValid s t dims batchDims)
    (g : Tensor α (s.contract t dims batchDims)) (y : Tensor α t) (c : s.Coord) :
    contractBackwardLeft dims batchDims h g y c =
      (List.finRange (Shape.size (t.eraseAxes ((batchDims ++ dims).map Prod.snd)))).foldl
        (fun acc j =>
          acc + g (Shape.contractOutputLeft s t dims batchDims c (Shape.Coord.unlinearize j)) *
            y (h.contractRight (Shape.contractOutputLeft s t dims batchDims c
              (Shape.Coord.unlinearize j)) (Shape.Coord.restrict c (dims.map Prod.fst))))
        0 :=
  Rep.get_ofFn _ c

/-- The pullback of `contract x y dims batchDims` to `y` at `g`. The entry at `r` adds `x l * g o`
to zero over the coordinates `i` of the other axes of `x` in row-major order, where `o` is the
output coordinate `contractOutputRight r i` and `l` is the coordinate of `x` that `contract` reads
at `o` with the contracted indices of `r`: the batch indices of `r`, its contracted indices at the
paired axes and `i` at the other axes. -/
def contractBackwardRight [Add α] [Mul α] [Zero α] (dims batchDims : List (Fin s.rank × Fin t.rank))
    (h : Shape.ContractValid s t dims batchDims) (x : Tensor α s)
    (g : Tensor α (s.contract t dims batchDims)) : Tensor α t :=
  Rep.ofFn fun r =>
    (List.finRange (Shape.size (s.eraseAxes ((batchDims ++ dims).map Prod.fst)))).foldl
      (fun acc i =>
        acc + x (s.contractLeft t dims batchDims (h.contractOutputRight r
            (Shape.Coord.unlinearize i))
            (Shape.Coord.reindexEq h.dims_extents (Shape.Coord.restrict r (dims.map Prod.snd)))) *
          g (h.contractOutputRight r (Shape.Coord.unlinearize i)))
      0

/-- An entry of the pullback of `contract` to `y` is its fold over the other axes of `x`. -/
@[simp] theorem contractBackwardRight_apply [Add α] [Mul α] [Zero α]
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (h : Shape.ContractValid s t dims batchDims)
    (x : Tensor α s) (g : Tensor α (s.contract t dims batchDims)) (r : t.Coord) :
    contractBackwardRight dims batchDims h x g r =
      (List.finRange (Shape.size (s.eraseAxes ((batchDims ++ dims).map Prod.fst)))).foldl
        (fun acc i =>
          acc + x (s.contractLeft t dims batchDims (h.contractOutputRight r
              (Shape.Coord.unlinearize i))
              (Shape.Coord.reindexEq h.dims_extents
                (Shape.Coord.restrict r (dims.map Prod.snd)))) *
            g (h.contractOutputRight r (Shape.Coord.unlinearize i)))
        0 :=
  Rep.get_ofFn _ r

/-- A fold over the indices of a vector is the fold over its coordinates. -/
theorem foldl_finRange_size_singleton {β : Type} (n : Nat) (f : β → Fin n → β) (b : β) :
    (List.finRange n).foldl f b =
      (List.finRange (Shape.size [n])).foldl
        (fun a i => f a (Shape.Coord.unlinearize i : Shape.Coord [n]).1) b := by
  rw [← Fin.foldl_eq_finRange_foldl, ← Fin.foldl_eq_finRange_foldl]
  have size : Shape.size [n] = n := by simp [Shape.size]
  rw [Fin.foldl_congr size]
  congr 1
  funext a i
  congr 1
  apply Fin.ext
  simp [Shape.Coord.unlinearizeFast_eq, Shape.Coord.unlinearizeFast]
  rfl

/-! ## Matrix products -/

/-- An entry of a matrix-vector contraction adds `A i k * v k` to zero in increasing `k`. -/
theorem contract_matVec_apply [Add α] [Mul α] [Zero α] {m n : Nat} (A : Tensor α [m, n])
    (v : Tensor α [n]) (c : Shape.Coord [m]) :
    (contract A v _ _ [m] .matVec) c =
      (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * v.getScalar k) 0 := by
  rw [contract_apply]
  simp only [get2_eq_apply, getScalar_eq_apply]
  generalize hc' : Shape.Coord.reindexEq _ c = c'
  have hc : ∀ k, Shape.Coord.index c' k = Shape.Coord.index c k := fun k => by
    rw [← hc']
    exact Shape.Coord.index_reindexEq _ c k
  refine ((foldl_finRange_size_singleton n _ 0).trans
    (foldl_finRange_congr rfl
      (fun a (r : Shape.Coord [n]) => a + A (c.1, r.1, PUnit.unit) * v (r.1, PUnit.unit))
      (fun a r => a + A (Shape.contractLeft [m, n] [n] _ _ c' r) *
        v ((Shape.ContractValid.matVec (m := m) (n := n)).contractRight c' r)) ?_ 0)).symm
  intro a r r' hrr
  have hA : ((c.1, r.1, PUnit.unit) : Shape.Coord [m, n]) =
      Shape.contractLeft [m, n] [n] _ _ c' r' := by
    refine Shape.Coord.ext_index (s := [m, n]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact (hc 0).symm
    | 1, _ => exact hrr 0
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  have hv : ((r.1, PUnit.unit) : Shape.Coord [n]) =
      (Shape.ContractValid.matVec (m := m) (n := n)).contractRight c' r' := by
    refine Shape.Coord.ext_index (s := [n]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact hrr 0
    | k + 1, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  show a + A _ * v _ = a + A _ * v _
  rw [hA, hv]

/-- An entry of a vector-matrix contraction adds `v k * A k j` to zero in increasing `k`. -/
theorem contract_vecMat_apply [Add α] [Mul α] [Zero α] {m n : Nat} (v : Tensor α [m])
    (A : Tensor α [m, n]) (c : Shape.Coord [n]) :
    (contract v A _ _ [n] .vecMat) c =
      (List.finRange m).foldl (fun sum i => sum + v.getScalar i * get2 A i c.1) 0 := by
  rw [contract_apply]
  simp only [get2_eq_apply, getScalar_eq_apply]
  generalize hc' : Shape.Coord.reindexEq _ c = c'
  have hc : ∀ k, Shape.Coord.index c' k = Shape.Coord.index c k := fun k => by
    rw [← hc']
    exact Shape.Coord.index_reindexEq _ c k
  refine ((foldl_finRange_size_singleton m _ 0).trans
    (foldl_finRange_congr rfl
      (fun a (r : Shape.Coord [m]) => a + v (r.1, PUnit.unit) * A (r.1, c.1, PUnit.unit))
      (fun a r => a + v (Shape.contractLeft [m] [m, n] _ _ c' r) *
        A ((Shape.ContractValid.vecMat (m := m) (n := n)).contractRight c' r)) ?_ 0)).symm
  intro a r r' hrr
  have hv : ((r.1, PUnit.unit) : Shape.Coord [m]) =
      Shape.contractLeft [m] [m, n] _ _ c' r' := by
    refine Shape.Coord.ext_index (s := [m]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact hrr 0
    | k + 1, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  have hA : ((r.1, c.1, PUnit.unit) : Shape.Coord [m, n]) =
      (Shape.ContractValid.vecMat (m := m) (n := n)).contractRight c' r' := by
    refine Shape.Coord.ext_index (s := [m, n]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact hrr 0
    | 1, _ => exact (hc 0).symm
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  show a + v _ * A _ = a + v _ * A _
  rw [hv, hA]

/-- An entry of a matrix product adds `A i k * B k j` to zero in increasing `k`. -/
theorem contract_matMul_apply [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) (c : Shape.Coord [m, p]) :
    (contract A B _ _ [m, p] .matMul) c =
      (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * get2 B k c.2.1) 0 := by
  rw [contract_apply]
  simp only [get2_eq_apply]
  generalize hc' : Shape.Coord.reindexEq _ c = c'
  have hc : ∀ k, Shape.Coord.index c' k = Shape.Coord.index c k := fun k => by
    rw [← hc']
    exact Shape.Coord.index_reindexEq _ c k
  refine ((foldl_finRange_size_singleton n _ 0).trans
    (foldl_finRange_congr rfl
      (fun a (r : Shape.Coord [n]) => a + A (c.1, r.1, PUnit.unit) * B (r.1, c.2.1, PUnit.unit))
      (fun a r => a + A (Shape.contractLeft [m, n] [n, p] _ _ c' r) *
        B ((Shape.ContractValid.matMul (m := m) (n := n) (p := p)).contractRight c' r)) ?_ 0)).symm
  intro a r r' hrr
  have hA : ((c.1, r.1, PUnit.unit) : Shape.Coord [m, n]) =
      Shape.contractLeft [m, n] [n, p] _ _ c' r' := by
    refine Shape.Coord.ext_index (s := [m, n]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact (hc 0).symm
    | 1, _ => exact hrr 0
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  have hB : ((r.1, c.2.1, PUnit.unit) : Shape.Coord [n, p]) =
      (Shape.ContractValid.matMul (m := m) (n := n) (p := p)).contractRight c' r' := by
    refine Shape.Coord.ext_index (s := [n, p]) ?_
    intro k hk
    match k, hk with
    | 0, _ => exact hrr 0
    | 1, _ => exact (hc 1).symm
    | k + 2, hk =>
      exact absurd hk (by
        simp only [Shape.rank_eq_length, List.length_cons, List.length_nil]
        omega)
  show a + A _ * B _ = a + A _ * B _
  rw [hA, hB]

/-- The `getScalar` form of `contract_matVec_apply`: entry `i` adds `A i k * v k` to zero in
increasing `k`. -/
theorem getScalar_contract_matVec [Add α] [Mul α] [Zero α] {m n : Nat}
    (A : Tensor α [m, n]) (v : Tensor α [n]) (i : Fin m) :
    (contract A v _ _ [m] .matVec).getScalar i =
      (List.finRange n).foldl (fun sum k => sum + get2 A i k * v.getScalar k) 0 := by
  rw [getScalar_eq_apply, contract_matVec_apply]

/-- The `getScalar` form of `contract_vecMat_apply`: entry `j` adds `v k * A k j` to zero in
increasing `k`. -/
theorem getScalar_contract_vecMat [Add α] [Mul α] [Zero α] {m n : Nat}
    (v : Tensor α [m]) (A : Tensor α [m, n]) (j : Fin n) :
    (contract v A _ _ [n] .vecMat).getScalar j =
      (List.finRange m).foldl (fun sum i => sum + v.getScalar i * get2 A i j) 0 := by
  rw [getScalar_eq_apply, contract_vecMat_apply]

/-- A matrix-vector contraction is the vector of its loops over the shared axis. -/
theorem contract_matVec_eq_ofFn [Add α] [Mul α] [Zero α] {m n : Nat} (A : Tensor α [m, n])
    (v : Tensor α [n]) :
    contract A v _ _ [m] .matVec =
      Rep.ofFn fun c : Shape.Coord [m] =>
        (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * v.getScalar k) 0 :=
  Rep.ext fun c => by rw [contract_matVec_apply, Rep.get_ofFn]

/-- A vector-matrix contraction is the vector of its loops over the shared axis. -/
theorem contract_vecMat_eq_ofFn [Add α] [Mul α] [Zero α] {m n : Nat} (v : Tensor α [m])
    (A : Tensor α [m, n]) :
    contract v A _ _ [n] .vecMat =
      Rep.ofFn fun c : Shape.Coord [n] =>
        (List.finRange m).foldl (fun sum i => sum + v.getScalar i * get2 A i c.1) 0 :=
  Rep.ext fun c => by rw [contract_vecMat_apply, Rep.get_ofFn]

/-- A matrix product is the matrix of its loops over the shared axis. -/
theorem contract_matMul_eq_ofFn [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) :
    contract A B _ _ [m, p] .matMul =
      Rep.ofFn fun c : Shape.Coord [m, p] =>
        (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * get2 B k c.2.1) 0 :=
  Rep.ext fun c => by rw [contract_matMul_apply, Rep.get_ofFn]

/-! ## Compiled matrix products -/

section

open Shape (Coord)

/-- The entries of `contract`, one fold each: the fallback of `contractFast`. -/
def contractFold {s t : Shape} [Add α] [Mul α] [Zero α] (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims)
    (hout : out = s.contract t dims batchDims) : Tensor α out :=
  Rep.ofFn fun o =>
    (List.finRange (Shape.size ((dims.map Prod.fst).map s.extent))).foldl
      (fun acc i =>
        acc + x (s.contractLeft t dims batchDims (Shape.Coord.reindexEq hout o)
            (Coord.unlinearize i)) *
          y (h.contractRight (Shape.Coord.reindexEq hout o) (Coord.unlinearize i)))
      0

/-- The matrix-vector product as its loop over the shared axis. -/
def matVecLoop [Add α] [Mul α] [Zero α] {m n : Nat} (A : Tensor α [m, n]) (v : Tensor α [n]) :
    Tensor α [m] :=
  Rep.ofFn fun c => (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * v.getScalar k) 0

/-- The vector-matrix product as its loop over the shared axis. -/
def vecMatLoop [Add α] [Mul α] [Zero α] {m n : Nat} (v : Tensor α [m]) (A : Tensor α [m, n]) :
    Tensor α [n] :=
  Rep.ofFn fun c => (List.finRange m).foldl (fun sum i => sum + v.getScalar i * get2 A i c.1) 0

/-- The matrix product as its loop over the shared axis. -/
def matMulLoop [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n]) (B : Tensor α [n, p]) :
    Tensor α [m, p] :=
  Rep.ofFn fun c =>
    (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * get2 B k c.2.1) 0

/-- `contract` with the matrix-vector, vector-matrix and matrix products computed by their loops
over the shared axis, and every other contraction by `contractFold`. -/
def contractFast {s t : Shape} [Add α] [Mul α] [Zero α] (x : Tensor α s) (y : Tensor α t)
    (dims batchDims : List (Fin s.rank × Fin t.rank)) (out : Shape)
    (h : Shape.ContractValid s t dims batchDims)
    (hout : out = s.contract t dims batchDims) : Tensor α out :=
  match s, t, x, y, dims, batchDims, h, hout with
  | [_, _], [_], x, y, [(⟨1, _⟩, ⟨0, _⟩)], [], h, hout =>
      castShape (matVecLoop x (castShape y (congrArg (fun k => [k])
        (h.extent_eq _ (List.mem_singleton_self _)).symm))) hout.symm
  | [_], [_, _], x, y, [(⟨0, _⟩, ⟨0, _⟩)], [], h, hout =>
      castShape (vecMatLoop (castShape x (congrArg (fun k => [k])
        (h.extent_eq _ (List.mem_singleton_self _)))) y) hout.symm
  | [_, _], [_, p], x, y, [(⟨1, _⟩, ⟨0, _⟩)], [], h, hout =>
      castShape (matMulLoop x (castShape y (congrArg (fun k => [k, p])
        (h.extent_eq _ (List.mem_singleton_self _)).symm))) hout.symm
  | _, _, x, y, dims, batchDims, h, hout => contractFold x y dims batchDims out h hout

end

/-- Compiled code computes `contract` with `contractFast`, which runs the matrix products
as their loops over the shared axis. -/
@[csimp] theorem contractFast_eq : @contract = @contractFast := by
  funext α _ s t _ _ _ x y dims batchDims out h hout
  unfold contractFast
  split
  · rename_i _ n n' A v _ _ hv hout
    obtain ⟨⟩ : n = n' := hv.extent_eq _ (List.mem_singleton_self _)
    subst hout
    exact contract_matVec_eq_ofFn A v
  · rename_i m m' _ v A _ _ hv hout
    obtain ⟨⟩ : m = m' := hv.extent_eq _ (List.mem_singleton_self _)
    subst hout
    exact contract_vecMat_eq_ofFn v A
  · rename_i _ n n' _ A B _ _ hv hout
    obtain ⟨⟩ : n = n' := hv.extent_eq _ (List.mem_singleton_self _)
    subst hout
    exact contract_matMul_eq_ofFn A B
  · rfl

/-! ## Matrix multiplication -/

/--
Multiply an `m x n` matrix by an `n x p` matrix: the contraction of the columns of `A` with the
rows of `B`.

Example:
```lean
-- `[2, 3] * [3, 2]` contracts the shared `3`, and mismatched inner sizes fail to typecheck.
def left : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 2] := [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def product : Tensor Float [2, 2] := Tensor.matmul left right
```
-/
def matmul [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n]) (B : Tensor α [n, p]) :
    Tensor α [m, p] :=
  contract A B _ _ [m, p] .matMul

/-- An entry of `matmul A B` adds `A i k * B k j` to zero in increasing `k`. -/
theorem matmul_apply [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) (c : Shape.Coord [m, p]) :
    matmul A B c =
      (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * get2 B k c.2.1) 0 :=
  contract_matMul_apply A B c

/-- The `Tensor.get2` form of `matmul_apply`: entry `(i, j)` of `matmul A B` adds `A i k * B k j` to
zero in increasing `k`. -/
theorem get2_matmul [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) (i : Fin m) (j : Fin p) :
    get2 (matmul A B) i j =
      (List.finRange n).foldl (fun sum k => sum + get2 A i k * get2 B k j) 0 := by
  rw [get2_eq_apply, matmul_apply]

/-- `matmul A B` is the matrix of its loops over the shared axis. -/
theorem matmul_eq_ofFn [Add α] [Mul α] [Zero α] {m n p : Nat} (A : Tensor α [m, n])
    (B : Tensor α [n, p]) :
    matmul A B =
      Rep.ofFn fun c : Shape.Coord [m, p] =>
        (List.finRange n).foldl (fun sum k => sum + get2 A c.1 k * get2 B k c.2.1) 0 :=
  contract_matMul_eq_ofFn A B

namespace LinearAlgebra

/-- Concatenated shapes add their ranks. -/
theorem _root_.TorchLean.Shape.rank_concat (left right : Shape) :
    (left ++ right).rank = left.rank + right.rank := by
  simp only [Shape.rank_eq_length, List.length_append]

/-- If `source` broadcasts to `target`, then `source ++ suffix` broadcasts to
`target ++ suffix`. -/
theorem _root_.TorchLean.Shape.CanBroadcastTo.concat_right {source target : Shape} (suffix : Shape)
    (broadcast : Shape.CanBroadcastTo source target) :
    Shape.CanBroadcastTo (source ++ suffix) (target ++ suffix) := by
  induction target generalizing source with
  | scalar =>
      cases source with
      | scalar => exact Shape.CanBroadcastTo.refl suffix
      | dim _ _ => exact absurd broadcast Shape.not_canBroadcastTo_dim_scalar
  | dim n target ih =>
      cases source with
      | scalar => exact (ih (Shape.canBroadcastTo_scalar_dim.mp broadcast)).expand_dims
      | dim m source =>
          by_cases hRank : source.rank = target.rank
          · obtain ⟨hHead, hTail⟩ := (Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp broadcast
            exact (Shape.canBroadcastTo_dim_dim_of_rank_eq
              (by simp only [List.append_eq, Shape.rank_concat, hRank])).mpr ⟨hHead, ih hTail⟩
          · exact (ih ((Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp broadcast)).expand_dims

/-- A batch prefix followed by a matrix has the size of the flattened batch matrix. -/
theorem _root_.TorchLean.Shape.size_concat_matrix (batch : Shape) (m n : Nat) :
    (batch ++ [m, n]).size = Shape.size [batch.size, m, n] := by
  simp [Shape.size_append]

/-- Multiply the matrices of `A` and `B` at each coordinate of their shared batch prefix, one outer
batch axis at a time. -/
def _root_.TorchLean.Tensor.matmulCommonBatch {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batch : Shape} {m n p : Nat}
    (A : Tensor α (batch ++ [m, n])) (B : Tensor α (batch ++ [n, p])) :
    Tensor α (batch ++ [m, p]) :=
  match batch with
  | .scalar => matmul A B
  | .dim _ _ =>
      Tensor.dim fun index =>
        matmulCommonBatch
          (Tensor.unstack A index) (Tensor.unstack B index)

/-- With no batch axes left, batched matmul is plain matmul. -/
@[simp] theorem _root_.TorchLean.Tensor.matmulCommonBatch_scalar {α : Type}
    [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) :
    matmulCommonBatch (batch := .scalar) left right = matmul left right := by
  rfl

/-- Batched matmul over an outer axis is the batched matmul of each pair of slices. -/
@[simp] theorem _root_.TorchLean.Tensor.matmulCommonBatch_dim {α : Type}
    [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {count m n p : Nat} {rest : Shape}
    (left : Fin count → Tensor α (rest ++ [m, n]))
    (right : Fin count → Tensor α (rest ++ [n, p])) :
    matmulCommonBatch (batch := .dim count rest) (.dim left) (.dim right) =
      .dim (fun index => matmulCommonBatch (left index) (right index)) := by
  simp [matmulCommonBatch]

open Shape

/-- An entry of a batched matrix product, at batch coordinate `b` and matrix coordinate `(i, j)`,
adds `left b i k * right b k j` to zero in increasing `k`. -/
theorem _root_.TorchLean.Tensor.matmulCommonBatch_apply {α : Type} [Storage α] [Add α]
    [Mul α] [Zero α]
    {batch : Shape} {m n p : Nat}
    (left : Tensor α (batch ++ [m, n]))
    (right : Tensor α (batch ++ [n, p]))
    (coordinate : (batch ++ [m, p]).Coord) :
    matmulCommonBatch left right coordinate =
      let split := Coord.appendEquiv batch [m, p]
        coordinate
      (List.finRange n).foldl
        (fun total k => total +
          left ((Coord.appendEquiv batch [m, n]).symm
              (split.1, split.2.1, k, PUnit.unit)) *
          right ((Coord.appendEquiv batch [n, p]).symm
              (split.1, k, split.2.2.1, PUnit.unit))) 0 := by
  -- A cast along `count :: source = count :: target` keeps the outer index and casts the tail.
  have cast_cons {source target : Shape} {count : Nat}
      (equal : source = target) (index : Coord (count :: source)) :
      _root_.cast (congrArg Coord (congrArg (List.cons count) equal)) index =
        (index.1, _root_.cast (congrArg Coord equal) index.2) := by
    cases equal
    rfl
  induction batch with
  | scalar =>
      change matmul left right coordinate = _
      rw [matmul_apply]
      simp only [get2_eq_apply]
      rfl
  | dim count batch ih =>
      change (Tensor.dim fun outer =>
        matmulCommonBatch (Tensor.unstack left outer)
          (Tensor.unstack right outer)) coordinate = _
      rw [Tensor.dim_apply, ih]
      simp only [Tensor.unstack_apply]
      rfl

end LinearAlgebra

/-- Batched matrix product with broadcast batch prefixes.

`A` has shape `batchA ++ [m, n]` and `B` has shape `batchB ++ [n, p]`. Both are broadcast to the
batch prefix `batch` by the proofs `broadcastA` and `broadcastB`. The result has shape
`batch ++ [m, p]` and holds the product of their matrices at each batch coordinate. -/
def batchMatmul {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA ++ [m, n])) (B : Tensor α (batchB ++ [n, p])) :
    Tensor α (batch ++ [m, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  let commonA := Tensor.broadcast A _ hA.dims hA.broadcastDims
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  let commonB := Tensor.broadcast B _ hB.dims hB.broadcastDims
  matmulCommonBatch commonA commonB

/-- The pullback of `batchMatmul` at `dC`. With `A'` and `B'` the operands broadcast to the
prefix `batch`, it returns the batched products `dC * B'ᵀ` and `A'ᵀ * dC`, each summed over the
broadcast axes back to the shape of its operand. -/
def matmulBackward {α : Type} [TorchLean.Storage α]
    [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA ++ [m, n])) (B : Tensor α (batchB ++ [n, p]))
    (dC : Tensor α (batch ++ [m, p])) :
    Tensor α (batchA ++ [m, n]) × Tensor α (batchB ++ [n, p]) :=
  have hA := Shape.CanBroadcastTo.concat_right [m, n] broadcastA
  let commonA := Tensor.broadcast A _ hA.dims hA.broadcastDims
  have hB := Shape.CanBroadcastTo.concat_right [n, p] broadcastB
  let commonB := Tensor.broadcast B _ hB.dims hB.broadcastDims
  let commonBT : Tensor α (batch ++ [p, n]) :=
    permute commonB (Shape.adjacentSwap (batch ++ [n, p]) batch.rank) (batch ++ [p, n])
  let commonAT : Tensor α (batch ++ [n, m]) :=
    permute commonA (Shape.adjacentSwap (batch ++ [m, n]) batch.rank) (batch ++ [n, m])
  let dCommonA := matmulCommonBatch dC commonBT
  let dCommonB := matmulCommonBatch commonAT dC
  let dA := reduceBroadcast hA.dims hA.broadcastDims dCommonA
  let dB := reduceBroadcast hB.dims hB.broadcastDims dCommonB
  (dA, dB)

end TorchLean.Tensor
