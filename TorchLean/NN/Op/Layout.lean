/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.PullbackTensor
public import TorchLean.NN.Op.Certificates
public import TorchLean.Tensor.AdjacentSwap.Adjoint
public import TorchLean.Tensor.Op.Cert
import Mathlib.Data.Fin.Tuple.Take

/-!
# Derivative certificates of the layout operators

We prove that the backward of `Op.reshape`, `Op.permute`, `Op.broadcast`, `Op.flip`, `Op.pad`
and `Op.cat` is the adjoint of the derivative of the forward in the value inputs, at every
argument of the operators, jointly in the input and the padding value for `Op.pad` and jointly in
all the inputs for `Op.cat`. Reshaping, permuting, broadcasting and flipping read the input
through a fixed map of coordinates, so each is a continuous linear map, and the content is the
inner product identity with the stored backward. Padding with a varying value adds the input
placed at its `Tensor.padPosition`s to the value times the padding mask, so it is affine in the
input and the value together.

A flip reads its input through `Tensor.flipSource`, an involution of the coordinates, so it is
self-adjoint (`Tensor.inner_flip`). The adjoint of a permutation permutes back
(`Tensor.permuteCLM_adjoint_apply`), and the two parts of padding are the derivatives of
`Op.pad_hasFDerivAt` and `Op.pad_value_hasFDerivAt`, whose adjoints are the stored pullbacks by
`Op.pad_backward_eq_adjoint` and `Op.pad_value_backward_eq_adjoint`. Each tensor function has a
`fun_prop` rule of `HasPullbackAt`, so `op_deriv` composes through it. Padding has one rule with
the value given as the item of a varying scalar tensor (`HasPullbackAt.pad`) and one with a
fixed value (`HasPullbackAt.pad_const`).

Concatenation along `dim` places entry `c` of input `i` at `Tensor.catCoordEquiv ⟨i, c⟩`, an
equivalence from the coordinates of the inputs, indexed by the input, to those of the output. So
it is a continuous linear map `Tensor.catCLM` on the L2 product of the inputs, whose adjoint reads
the cotangent back at those coordinates (`Tensor.inner_cat`). The certificate of `Op.cat` takes
its point in that product and reads it into the tuple of inputs by `Op.catElementValues`, the
inverse of `Op.catElementParts`, and `Tensor.catBlock_eq_catBackward` identifies the stored blocks
of the cotangent with that adjoint. The `fun_prop` rule `HasPullbackAt.cat_family` concatenates a
family of maps, and `HasPullbackAt.cat`, the rule for one map into the L2 product, gives the
certificate.
-/

@[expose] public section

open scoped RealInnerProductSpace

namespace TorchLean

namespace Tensor

variable {s : Shape}

/-! ## Flipping -/

/-- The coordinate that `flip` along `dims` reads at `c`: the reversed index along each axis of
`dims` and the same index along the others. -/
def flipSource (dims : List (Fin s.rank)) (c : s.Coord) : s.Coord :=
  Shape.Coord.ofFun fun d => if d ∈ dims then (Shape.Coord.get c d).rev else Shape.Coord.get c d

/-- `flipSource dims` is an involution: reading through it twice returns the coordinate. -/
theorem flipSource_involutive (dims : List (Fin s.rank)) : Function.Involutive (flipSource dims) :=
  fun c => by
    refine Shape.Coord.ext fun d => ?_
    simp only [flipSource, Shape.Coord.get_ofFun]
    split_ifs <;> simp

/-- Flipping a real tensor along `dims`, as a continuous linear map. -/
noncomputable def flipCLM (dims : List (Fin s.rank)) : Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  Rep.pullCLM (flipSource dims)

/-- `flipCLM dims` applied to `y` is `flip y dims`. -/
@[simp] theorem flipCLM_apply (dims : List (Fin s.rank)) (y : Tensor ℝ s) :
    flipCLM dims y = flip y dims :=
  rfl

/-- Flipping along `dims` is self-adjoint for the Euclidean inner product. -/
theorem inner_flip (dims : List (Fin s.rank)) (u δ : Tensor ℝ s) :
    ⟪flip u dims, δ⟫ = ⟪u, flip δ dims⟫ := by
  rw [inner_eq_sum, inner_eq_sum]
  refine Fintype.sum_bijective (flipSource dims) (flipSource_involutive dims).bijective _ _
    fun c => ?_
  change Rep.get (Rep.pull (flipSource dims) u) c * Rep.get δ c =
    Rep.get u (flipSource dims c) * Rep.get (Rep.pull (flipSource dims) δ) (flipSource dims c)
  rw [Rep.pull_apply, Rep.pull_apply, flipSource_involutive dims c]

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s} {bf : Tensor ℝ s → E}

/-- Permuting the axes by `perm` into `out`, pulled back by permuting back with
`s.unpermute perm hout`. -/
@[fun_prop]
theorem permute (perm : Equiv.Perm (Fin s.rank)) (out : Shape) (hout : out = s.permute perm)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.permute (f y) perm out hout)
      ((permuteCLM perm out hout).comp Df)
      (fun δ => bf (Tensor.permute δ (s.unpermute perm hout) s
        (s.permute_unpermute perm hout).symm)) x := by
  simpa only [permuteCLM_apply] using
    clm (permuteCLM perm out hout)
      (fun δ => Tensor.permute δ (s.unpermute perm hout) s (s.permute_unpermute perm hout).symm)
      (fun u δ => by rw [← ContinuousLinearMap.adjoint_inner_right, permuteCLM_adjoint_apply]) hf

/-- Flipping along `dims`, pulled back by flipping along `dims`. -/
@[fun_prop]
theorem flip (dims : List (Fin s.rank)) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.flip (f y) dims) ((flipCLM dims).comp Df)
      (fun δ => bf (Tensor.flip δ dims)) x := by
  simpa only [flipCLM_apply] using
    clm (flipCLM dims) (fun δ => Tensor.flip δ dims)
      (fun u δ => by rw [flipCLM_apply, inner_flip]) hf

/-- Padding `f y` with the item of `g y`: the derivative is the adjoint of reading back through
`padPosition` after `Df` plus `Op.padValueCLM` after `Dg`, and the pullback reads the cotangent at
the positions of the input and sums it over the padding positions. -/
@[fun_prop]
theorem pad (low high interior : Fin s.rank → Nat) {g : E → Tensor ℝ []}
    {Dg : E →L[ℝ] Tensor ℝ []} {bg : Tensor ℝ [] → E} (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.pad (f y) low high interior (g y).item)
      ((Rep.pullCLM (padPosition low high interior)).adjoint.comp Df +
        (Op.padValueCLM low high interior).comp Dg)
      (fun δ => bf (padBackward low high interior δ) +
        bg (Tensor.scalar (Tensor.sum (Tensor.select
          (Tensor.pad (Tensor.full s false) low high interior true) δ (Tensor.full _ 0))))) x := by
  have hvalue : ∀ (u : Tensor ℝ []) (δ : Tensor ℝ (s.pad low high interior)),
      ⟪Op.padValueCLM low high interior u, δ⟫ = ⟪u, Tensor.scalar (Tensor.sum (Tensor.select
        (Tensor.pad (Tensor.full s false) low high interior true) δ (Tensor.full _ 0)))⟫ := by
    intro u δ
    have h := Op.pad_value_backward_eq_adjoint low high interior 0 u δ
    rw [(Op.pad_value_hasFDerivAt low high interior 0 u).fderiv] at h
    simp only [Op.rule?_pad, Option.get_some] at h
    rw [h, ContinuousLinearMap.adjoint_inner_right]
  refine ((clm (Rep.pullCLM (padPosition low high interior)).adjoint
      (padBackward low high interior) (fun u δ => ?_) hf).add
    (clm (Op.padValueCLM low high interior) _ hvalue hg)).congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun y => ?_)
  · rw [ContinuousLinearMap.adjoint_inner_left]
    rfl
  · show Tensor.pad (f y) low high interior (g y).item =
      (Rep.pullCLM (padPosition low high interior)).adjoint (f y) +
        Op.padValueCLM low high interior (g y)
    rw [Rep.pullCLM_adjoint, ← Op.pad_zero_eq_sumFibers, Op.pad_eq_add, Op.pad_zero_eq_smul]
    rfl

/-- Padding `f y` with a fixed value `a`: the derivative is the adjoint of reading back through
`padPosition` after `Df`, and the pullback reads the cotangent at the positions of the input. -/
@[fun_prop]
theorem pad_const (low high interior : Fin s.rank → Nat) (a : ℝ) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.pad (f y) low high interior a)
      ((Rep.pullCLM (padPosition low high interior)).adjoint.comp Df)
      (fun δ => bf (padBackward low high interior δ)) x := by
  have h := HasPullbackAt.pad low high interior hf (HasPullbackAt.const (Tensor.scalar a) x)
  rw [ContinuousLinearMap.comp_zero, add_zero] at h
  refine (h.congr_deriv rfl fun δ => add_zero _).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun y => ?_)
  rw [Tensor.item_scalar]

end HasPullbackAt

/-! ## Certificates -/

/-- The forward of `Op.reshape` is differentiable in its input, and its backward, the reshape
back, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.reshape derivative]
theorem reshape_backward_eq_adjoint {s t : Shape} (h : s.size = t.size) (x : Tensor ℝ s)
    (δ : Tensor ℝ t) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.reshape h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.reshape h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.reshape h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_reshape, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.permute` is differentiable in its input, and its backward, the permutation
back by `s.unpermute perm hout`, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.permute derivative]
theorem permute_backward_eq_adjoint {s : Shape} (perm : Equiv.Perm (Fin s.rank)) (out : Shape)
    (hout : out = s.permute perm) (x : Tensor ℝ s) (δ : Tensor ℝ out) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.permute perm out hout).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.permute perm out hout).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.permute perm out hout).rule? (α := ℝ) |>.get rfl).forward (x', ())).1)
        x).adjoint δ := by
  simp only [Op.rule?_permute, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.broadcast` is differentiable in its input, and its backward,
`Tensor.reduceBroadcast`, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.broadcast derivative]
theorem broadcast_backward_eq_adjoint {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (h : Shape.BroadcastDims s t dims) (x : Tensor ℝ s) (δ : Tensor ℝ t) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.broadcast dims h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.broadcast dims h).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.broadcast dims h).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_broadcast, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.flip` is differentiable in its input, and its backward, the flip along the
same axes, is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.flip derivative]
theorem flip_backward_eq_adjoint {s : Shape} (dims : List (Fin s.rank)) (x δ : Tensor ℝ s) :
    DifferentiableAt ℝ (fun x' =>
        (((Op.flip dims).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x ∧
    (((Op.flip dims).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
      (fderiv ℝ (fun x' =>
        (((Op.flip dims).rule? (α := ℝ) |>.get rfl).forward (x', ())).1) x).adjoint δ := by
  simp only [Op.rule?_flip, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · rfl

/-- The forward of `Op.pad` is differentiable in the input and the padding value together, on
their L2 product, and its pair of pullbacks is the adjoint of that derivative. -/
@[op_cert TorchLean.Op.pad derivative 0 1]
theorem pad_prod_backward_eq_adjoint {s : Shape} (low high interior : Fin s.rank → Nat)
    (x : Tensor ℝ s) (value : Tensor ℝ []) (δ : Tensor ℝ (s.pad low high interior)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ []) =>
        (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (x, value)) ∧
    WithLp.toLp 2
        ((((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).backward
          (x, value, ()) (δ, ())).1,
        (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).backward
          (x, value, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ []) =>
        (((Op.pad (s := s) low high interior).rule? (α := ℝ) |>.get rfl).forward
          (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, value))).adjoint δ := by
  simp only [Op.rule?_pad, Option.get_some]
  apply HasPullbackAt.differentiableAt_and
  · fun_prop
  · refine withLp_prod_ext ?_ ?_ <;>
      simp only [WithLp.add_fst, WithLp.add_snd, WithLp.toLp_fst, WithLp.toLp_snd, add_zero,
        zero_add]

/-! ## Concatenation -/

namespace Op

variable {α : Type} [Storage α] {e : Op.Element} {s : Shape}

/-- The tuple of inputs of `cat` along `dim` at the element `e` whose tensors by position are
`xs`, the inverse of `Op.catElementParts`. -/
def catElementValues (dim : Fin s.rank) : (sizes : List Nat) →
    ((i : Fin sizes.length) → (e.port .value (s.set dim (sizes.get i))).Carrier α) →
    Values α (sizes.map fun n => e.port .value (s.set dim n))
  | [], _ => ()
  | _ :: rest, xs => (xs ⟨0, Nat.zero_lt_succ _⟩, catElementValues dim rest fun i => xs i.succ)

/-- The tensors by position of the tuple `catElementValues dim sizes xs` are `xs`. -/
@[simp] theorem catElementParts_catElementValues (dim : Fin s.rank) : (sizes : List Nat) →
    (xs : (i : Fin sizes.length) → (e.port .value (s.set dim (sizes.get i))).Carrier α) →
    catElementParts dim sizes (catElementValues dim sizes xs) = xs
  | [], _ => funext fun i => i.elim0
  | _ :: rest, xs => funext fun
    | ⟨0, _⟩ => rfl
    | ⟨k + 1, hk⟩ => congrFun (catElementParts_catElementValues dim rest fun i => xs i.succ)
        ⟨k, Nat.lt_of_succ_lt_succ hk⟩

/-- The tuple built from the tensors by position of the inputs `xs` of `cat` is `xs`. -/
@[simp] theorem catElementValues_catElementParts (dim : Fin s.rank) : (sizes : List Nat) →
    (xs : Values α (sizes.map fun n => e.port .value (s.set dim n))) →
    catElementValues dim sizes (catElementParts dim sizes xs) = xs
  | [], () => rfl
  | _ :: rest, (_, xs) => congrArg _ (catElementValues_catElementParts dim rest xs)

/-- Input `i` of the gradients `catGrads dim g offset sizes h` is the block of `g` along `dim` of
`sizes.get i` entries from `o`, the sum of `offset` and the sizes before `i`. -/
theorem catTangentParts_catGrads {total : Nat} (dim : Fin s.rank)
    (g : Tensor α (s.set dim total)) : (offset : Nat) → (sizes : List Nat) →
    (h : offset + sizes.sum ≤ total) → (i : Fin sizes.length) → (o : Nat) →
    o = offset + (sizes.take i).sum → (h' : o + sizes.get i ≤ total) →
    catTangentParts dim sizes (catGrads dim g offset sizes h) i = catBlock dim g o (sizes.get i) h'
  | _, [], _, i, _, _, _ => i.elim0
  | offset, n :: rest, h, ⟨0, _⟩, o, ho, h' => by
      subst ho
      rfl
  | offset, n :: rest, h, ⟨k + 1, hk⟩, o, ho, h' =>
      catTangentParts_catGrads dim g (offset + n) rest (by rw [List.sum_cons] at h; omega)
        ⟨k, Nat.lt_of_succ_lt_succ hk⟩ o
        (by simp only [ho, List.take_succ_cons, List.sum_cons]; omega) h'

end Op

namespace Tensor

variable {α : Type} [Storage α] {s : Shape}

/-- The coordinates of the parts of a concatenation along `dim`, indexed by the part, as the
coordinates of the concatenation: index `j` along `dim` in part `i` goes to `j` plus the sizes
before `i`. -/
def catCoordEquiv {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) :
    (Σ i, Shape.Coord (s.set dim (sizes i))) ≃ Shape.Coord (s.set dim (∑ i, sizes i)) :=
  (Equiv.sigmaCongrRight fun i =>
      Shape.Coord.setEquiv dim.val (shape := s) (sizes i)).trans
    ((Equiv.sigmaProdDistrib (fun i => Fin (sizes i)) _).symm.trans
      ((Equiv.prodCongr finSigmaFinEquiv (Equiv.refl _)).trans
        (Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)).symm))

/-- `catCoordEquiv` sends coordinate `c` of part `i` to the index `finSigmaFinEquiv ⟨i, j⟩` along
`dim`, for `j` the index of `c` along `dim`, and the indices of `c` along the other axes. -/
theorem catCoordEquiv_apply {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (i : Fin k)
    (c : Shape.Coord (s.set dim (sizes i))) :
    catCoordEquiv dim sizes ⟨i, c⟩ =
      (Shape.Coord.setEquiv dim.val (shape := s) (∑ i, sizes i)).symm
        (finSigmaFinEquiv ⟨i, (Shape.Coord.setEquiv dim.val (shape := s) (sizes i) c).1⟩,
          (Shape.Coord.setEquiv dim.val (shape := s) (sizes i) c).2) :=
  rfl

/-- The concatenation of `xs` along `dim` holds entry `c` of part `i` at
`catCoordEquiv dim sizes ⟨i, c⟩`. -/
theorem cat_catCoordEquiv {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    (xs : (i : Fin k) → Tensor α (s.set dim (sizes i))) (i : Fin k)
    (c : Shape.Coord (s.set dim (sizes i))) :
    cat dim sizes xs _ rfl (catCoordEquiv dim sizes ⟨i, c⟩) = xs i c := by
  rw [cat_apply]
  dsimp only
  rw [Shape.Coord.reindexEq_self, catCoordEquiv_apply, Equiv.apply_symm_apply]
  dsimp only
  rw [Equiv.symm_apply_apply]
  exact congrArg (xs i) (Equiv.symm_apply_apply _ c)

/-- The pullback of a concatenation along `dim` into `out`: part `i` of the cotangent `δ` reads
`δ` at the coordinates of part `i`. -/
def catBackward {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (δ : Tensor α out) (i : Fin k) :
    Tensor α (s.set dim (sizes i)) :=
  Rep.ofFn fun c => δ (Shape.Coord.reindexEq hout.symm (catCoordEquiv dim sizes ⟨i, c⟩))

/-- The inner product of the concatenation of `xs` with `δ` is the sum over the parts of the
inner products of `xs i` with part `i` of `catBackward`. -/
theorem inner_cat {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    (xs : (i : Fin k) → Tensor ℝ (s.set dim (sizes i))) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) (δ : Tensor ℝ out) :
    ⟪cat dim sizes xs out hout, δ⟫ = ∑ i, ⟪xs i, catBackward dim sizes out hout δ i⟫ := by
  subst hout
  simp only [inner_eq_sum, catBackward, Rep.get_ofFn, Shape.Coord.reindexEq_self]
  rw [← Fintype.sum_sigma (fun σ : (Σ i, Shape.Coord (s.set dim (sizes i))) =>
    xs σ.1 σ.2 * δ (catCoordEquiv dim sizes σ))]
  exact (Fintype.sum_equiv (catCoordEquiv dim sizes) _ _ fun σ => by
    rw [cat_catCoordEquiv]).symm

/-- Concatenating a family of real tensors along `dim` into `out`, as a continuous linear map on
their L2 product. -/
noncomputable def catCLM {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i)) :
    PiLp 2 (fun i => Tensor ℝ (s.set dim (sizes i))) →L[ℝ] Tensor ℝ out :=
  LinearMap.toContinuousLinearMap
    { toFun := fun p => cat dim sizes p.ofLp out hout
      map_add' := fun p q => Rep.ext fun c => by
        simp only [cat_apply, Tensor.add_apply, WithLp.ofLp_add, Pi.add_apply]
      map_smul' := fun a p => Rep.ext fun c => by
        simp only [cat_apply, Rep.smul_apply, WithLp.ofLp_smul, Pi.smul_apply,
          RingHom.id_apply, smul_eq_mul] }

/-- `catCLM dim sizes out hout` applied to `p` is the concatenation of the family `p`. -/
@[simp] theorem catCLM_apply {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i))
    (p : PiLp 2 (fun i => Tensor ℝ (s.set dim (sizes i)))) :
    catCLM dim sizes out hout p = cat dim sizes p.ofLp out hout :=
  rfl

/-- The block of `δ` along `dim` of `sizes i` entries from the sum of the sizes before `i` is
part `i` of `catBackward`. -/
theorem catBlock_eq_catBackward {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat)
    {total : Nat} (htotal : total = ∑ i, sizes i) (δ : Tensor α (s.set dim total))
    (i : Fin k) (offset : Nat) (hoffset : offset = ∑ j : Fin i, sizes (Fin.castLE i.isLt.le j))
    (h : offset + sizes i ≤ total) :
    Op.catBlock dim δ offset (sizes i) h =
      catBackward dim sizes _ (congrArg (s.set dim) htotal) δ i := by
  subst hoffset
  refine Rep.ext fun c => ?_
  simp only [Op.catBlock, catBackward, Rep.get_ofFn]
  congr 1
  refine Shape.Coord.ext_index fun m _ => ?_
  simp only [Shape.Coord.index_reindexEq, catCoordEquiv_apply,
    Shape.Coord.index_setEquiv_symm, finSigmaFinEquiv_apply]

end Tensor

namespace HasPullbackAt

open Tensor

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}

/-- Concatenating along `dim` the family of tensors `f y`, a point of their L2 product: the
derivative is `catCLM` after `Df`, and the pullback applies `bf` to `catBackward`. -/
theorem cat {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i))
    {f : E → PiLp 2 (fun i => Tensor ℝ (s.set dim (sizes i)))}
    {Df : E →L[ℝ] PiLp 2 (fun i => Tensor ℝ (s.set dim (sizes i)))}
    {bf : PiLp 2 (fun i => Tensor ℝ (s.set dim (sizes i))) → E}
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.cat dim sizes (f y).ofLp out hout)
      ((catCLM dim sizes out hout).comp Df)
      (fun δ => bf (WithLp.toLp 2 (catBackward dim sizes out hout δ))) x := by
  simpa only [catCLM_apply] using
    clm (catCLM dim sizes out hout) (fun δ => WithLp.toLp 2 (catBackward dim sizes out hout δ))
      (fun u δ => by rw [catCLM_apply, inner_cat, PiLp.inner_apply]) hf

/-- Concatenating along `dim` the tensors `f i y`: the derivative is `catCLM` after the
derivatives `Df i`, and the pullback sums `bf i` at the parts of `catBackward`. -/
@[fun_prop]
theorem cat_family {k : Nat} (dim : Fin s.rank) (sizes : Fin k → Nat) (out : Shape)
    (hout : out = s.set dim (∑ i, sizes i))
    {f : (i : Fin k) → E → Tensor ℝ (s.set dim (sizes i))}
    {Df : (i : Fin k) → E →L[ℝ] Tensor ℝ (s.set dim (sizes i))}
    {bf : (i : Fin k) → Tensor ℝ (s.set dim (sizes i)) → E}
    (hf : ∀ i, HasPullbackAt (f i) (Df i) (bf i) x) :
    HasPullbackAt (fun y => Tensor.cat dim sizes (fun i => f i y) out hout)
      ((catCLM dim sizes out hout).comp
        ((PiLp.continuousLinearEquiv 2 ℝ _).symm.toContinuousLinearMap.comp
          (ContinuousLinearMap.pi Df)))
      (fun δ => ∑ i, bf i (catBackward dim sizes out hout δ i)) x := by
  have hp : HasPullbackAt (fun y => WithLp.toLp 2 fun i => f i y)
      ((PiLp.continuousLinearEquiv 2 ℝ _).symm.toContinuousLinearMap.comp
        (ContinuousLinearMap.pi Df))
      (fun δ => ∑ i, bf i (δ i)) x :=
    ⟨(PiLp.continuousLinearEquiv 2 ℝ _).symm.toContinuousLinearMap.hasFDerivAt.comp x
        (hasFDerivAt_pi.2 fun i => (hf i).hasFDerivAt), fun u δ => by
      rw [inner_sum, PiLp.inner_apply]
      exact Finset.sum_congr rfl fun i _ => (hf i).inner_eq u (δ i)⟩
  exact cat dim sizes out hout hp

end HasPullbackAt

/-- The forward of `Op.cat` is differentiable in its inputs together, on their L2 product, and
its backward, the blocks of the cotangent at the offsets of the inputs, is the adjoint of that
derivative. -/
@[op_cert TorchLean.Op.cat derivative]
theorem cat_backward_eq_adjoint {s : Shape} (dim : Fin s.rank) (sizes : List Nat)
    (xs : Values ℝ (sizes.map fun n => .value (s.set dim n)))
    (δ : Tensor ℝ (s.set dim sizes.sum)) :
    DifferentiableAt ℝ (fun p : PiLp 2 (fun i : Fin sizes.length =>
          Tensor ℝ (s.set dim (sizes.get i))) =>
        (((Op.cat dim sizes).rule? (α := ℝ) |>.get rfl).forward
          (Op.catElementValues dim sizes p.ofLp)).1)
      (WithLp.toLp 2 (Op.catElementParts (e := .scalar) dim sizes xs)) ∧
    WithLp.toLp 2 (Op.catTangentParts dim sizes
        (((Op.cat dim sizes).rule? (α := ℝ) |>.get rfl).backward xs (δ, ()))) =
      (fderiv ℝ (fun p : PiLp 2 (fun i : Fin sizes.length =>
          Tensor ℝ (s.set dim (sizes.get i))) =>
        (((Op.cat dim sizes).rule? (α := ℝ) |>.get rfl).forward
          (Op.catElementValues dim sizes p.ofLp)).1)
        (WithLp.toLp 2 (Op.catElementParts (e := .scalar) dim sizes xs))).adjoint δ := by
  simp only [Op.rule?_cat, Option.get_some, Op.catElementParts_catElementValues]
  apply HasPullbackAt.differentiableAt_and
  · exact HasPullbackAt.cat dim sizes.get _ _ (HasPullbackAt.id _)
  · refine congrArg (WithLp.toLp 2) (funext fun i => ?_)
    have hi := List.sum_take_add_sum_drop sizes (i + 1)
    rw [List.sum_take_succ sizes i i.isLt] at hi
    rw [Op.catTangentParts_catGrads dim δ 0 sizes _ i (sizes.take i).sum (zero_add _).symm
        (by simp only [List.get_eq_getElem]; omega),
      Tensor.catBlock_eq_catBackward dim sizes.get (by simp) δ i (sizes.take i).sum
        (by rw [← Fin.ofFn_take_get sizes i.isLt.le, List.sum_ofFn]; rfl)]

end TorchLean
