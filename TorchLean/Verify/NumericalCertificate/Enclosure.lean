/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Softmax
public import TorchLean.NN.Backend.Profile -- shake: keep
public import TorchLean.Tensor.ConfiguredBinary -- shake: keep
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval.Outward
public import FloatLib.Numerics.Enclosure.Interval.Real
public import TorchLean.Tensor.Op.Cert

/-!
# Numerical certificate enclosures

Source-range checks, real and IEEE enclosure predicates, replay checks and pointwise error
bounds for graph numerical certificates. A real value lies in a binary32 interval when it lies
between the real values of the endpoints (`Binary.Interval.RealMem`).

The tensor enclosure lemmas of the ReLU, absolute-value, square-root, addition, subtraction,
multiplication and softmax transfers are stated on the operator's rule (`TorchLean.Op.rule?`) and
registered with `op_cert ... enclosure`.

Main declarations: `TensorEnclosed`, `tensorWithinRange_eq_true_iff`,
`TensorErrorLe.of_check`, `checkSources`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean
namespace NumericalCertificate

open Tensor
open FloatLib.Floats.Formats.BinaryInterchange
open scoped BigOperators

/-! ## Raw and checked source assumptions -/

/-- A binary32 range supplied for a source node: an input, a constant, or a random draw
(`randUniform` or `bernoulliMask`). -/
structure SourceRange where
  /-- The id of the source node. -/
  nodeId : Nat
  /-- The binary32 interval supplied for the node. -/
  enclosure : Interval (Binary 8 23)
  deriving Repr

/-- A source range after the checker has established finite, ordered endpoints. -/
structure CheckedSourceRange extends SourceRange where
  /-- The endpoints of `enclosure` are finite and ordered. -/
  valid : Binary.Interval.Valid enclosure

/-- A checked source range prints as its `SourceRange`, without the proof `valid`. -/
instance : Repr CheckedSourceRange where
  reprPrec r _ := repr r.toSourceRange

/-- Bitwise equality of the endpoints of two binary32 intervals.

Comparing bits tells `+0` from `-0` and keeps the endpoints exactly as the certificate wrote them.
`Binary.Interval.Valid` rejects NaN endpoints separately.
-/
def sameIntervalBits (a b : Interval (Binary 8 23)) : Bool :=
  ExecFloat.Binary.toBits32 a.lo == ExecFloat.Binary.toBits32 b.lo &&
    ExecFloat.Binary.toBits32 a.hi == ExecFloat.Binary.toBits32 b.hi

/-- `Binary.Interval.leB` decides the proposition-level IEEE non-strict order. -/
theorem leB_eq_true_iff (x y : Binary 8 23) :
    Binary.Interval.leB x y = true <-> LE.le x y :=
  (Model.Interval.le_iff_leB_eq_true
    (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y)).symm.trans
    FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.symm

/-! ## Real semantics of the arithmetic transfers

The checker propagates binary32 endpoints and the graph specification is read over real
scalars. `Binary.Interval.RealMem` relates the two. Runtime rounding error is a separate claim,
which `FwdGraph.eval_approx` and `RevGraph.backprop_approx` prove for a graph that the caller
builds. No theorem relates such a graph to a `TorchLean.IR.Graph`.
-/

/-- If the rational decodings of `interval`'s endpoints bound `value`, then `interval` encloses
`value`: finite endpoints have the same real value under both decodings. -/
theorem realMem_of_containsReal {interval : Interval (Binary 8 23)} {value : ℝ}
    (h : interval.ContainsReal Binary.toRat? value) : Binary.Interval.RealMem interval value := by
  rcases h with ⟨lo, hi, hlo, hhi, hxlo, hxhi⟩
  cases ha : Binary.toRat? interval.lo with
  | none => simp [ha] at hlo
  | some a =>
    cases hb : Binary.toRat? interval.hi with
    | none => simp [hb] at hhi
    | some b =>
      simp [ha] at hlo
      simp [hb] at hhi
      subst lo hi
      change Binary.toReal interval.lo ≤ value ∧ value ≤ Binary.toReal interval.hi
      rw [Binary.toReal_of_toRat?_eq_some ha, Binary.toReal_of_toRat?_eq_some hb]
      exact ⟨hxlo, hxhi⟩

/-- If valid intervals `a` and `b` enclose `x` and `y` and their outward-rounded sum is valid, that
sum encloses `x + y`. -/
theorem add_realMem {a b : Interval (Binary 8 23)} {x y : Real}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.add a b))
    (hx : Binary.Interval.RealMem a x) (hy : Binary.Interval.RealMem b y) :
    Binary.Interval.RealMem (Binary.Interval.add a b) (x + y) :=
  (Binary.Interval.eRealMem_coe_iff_of_valid hout _).mp
    (Binary.Interval.add_sound a b ha hb hx hy)

/-- If valid intervals `a` and `b` enclose `x` and `y` and their outward-rounded difference is
valid, it encloses `x - y`. -/
theorem sub_realMem {a b : Interval (Binary 8 23)} {x y : Real}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.sub a b))
    (hx : Binary.Interval.RealMem a x) (hy : Binary.Interval.RealMem b y) :
    Binary.Interval.RealMem (Binary.Interval.sub a b) (x - y) :=
  (Binary.Interval.eRealMem_coe_iff_of_valid hout _).mp
    (Binary.Interval.sub_sound a b ha hb hx hy)

/-- If valid intervals `a` and `b` enclose `x` and `y` and their outward-rounded product is valid,
it encloses `x * y`. -/
theorem mul_realMem {a b : Interval (Binary 8 23)} {x y : Real}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.mul a b))
    (hx : Binary.Interval.RealMem a x) (hy : Binary.Interval.RealMem b y) :
    Binary.Interval.RealMem (Binary.Interval.mul a b) (x * y) :=
  (Binary.Interval.eRealMem_coe_iff_of_valid hout _).mp
    (Binary.Interval.mul_sound a b ha hb hx hy)

/-- Outward-rounded enclosure of a left fold that adds `count` values from `range`, starting from
`+0` as the tensor reduction does. -/
def sumLeftRange (count : Nat) (range : Interval (Binary 8 23)) : Interval (Binary 8 23) :=
  (List.range count).foldl
    (fun acc _ => Binary.Interval.add acc range)
    (Binary.Interval.point (ExecFloat.Binary.zero false : Binary 8 23))

/-- Enclose a scaled contraction without requiring its unscaled product to fit binary32.

The rational calculation is exact, and only its final endpoints are rounded outward to binary32.
When the ordinary staged binary32 enclosure is finite, its endpoints widen the result too. Replay
checks the executed IEEE values against the returned interval, and the transfer assumes nothing
about how a native kernel evaluates the contraction.
-/
def scaledContractionRange? (count : Nat) (left right scale : Interval (Binary 8 23)) :
    Option (Interval (Binary 8 23)) := do
  let a ← left.decode? Binary.toRat?
  let b ← right.decode? Binary.toRat?
  let c ← scale.decode? Binary.toRat?
  let product ← Interval.mul? Interval.Internal.rationalOutwardRounding a b
  let sum : Interval Rat := ⟨count * product.lo, count * product.hi⟩
  let weighted ← Interval.mul? Interval.Internal.rationalOutwardRounding sum c
  let staged := Binary.Interval.mul
    (sumLeftRange count (Binary.Interval.mul left right)) scale
  let bounds := match staged.decode? Binary.toRat? with
    | none => weighted
    | some previous => ⟨min weighted.lo previous.lo, max weighted.hi previous.hi⟩
  Interval.encloseInterval? Binary.intervalRounding bounds

/-- If `scaledContractionRange?` succeeds with `output`, then `output` contains
`(∑ i, x i * y i) * coefficient` for all summands and coefficients drawn from the input intervals.

Without `h`, the successful outward rounding, there is no finite enclosure and so no certificate.
The statement concerns the real expression, and concrete IEEE values pass replay separately.
-/
theorem scaledContractionRange?_sound {count : Nat}
    {left right scale output : Interval (Binary 8 23)}
    (h : scaledContractionRange? count left right scale = some output)
    (x y : Fin count → ℝ) (coefficient : ℝ)
    (hx : ∀ i, left.ContainsReal Binary.toRat? (x i))
    (hy : ∀ i, right.ContainsReal Binary.toRat? (y i))
    (hc : scale.ContainsReal Binary.toRat? coefficient) :
    output.ContainsReal Binary.toRat? ((∑ i, x i * y i) * coefficient) := by
  simp only [scaledContractionRange?, Option.bind_eq_bind, Option.bind_eq_some_iff] at h
  obtain ⟨a, ha, b, hb, c, hc', product, hp, weighted, hw, hout⟩ := h
  have hproduct (i : Fin count) :
      (product.lo : ℝ) ≤ x i * y i ∧ x i * y i ≤ (product.hi : ℝ) := by
    apply (Interval.containsReal_some_iff product _).1
    exact Interval.containsReal_mul? Interval.Internal.rationalOutwardRounding hp
      ((Interval.containsReal_some_iff _ _).2
        ((Interval.containsReal_iff_of_decode? ha _).1 (hx i)))
      ((Interval.containsReal_some_iff _ _).2
        ((Interval.containsReal_iff_of_decode? hb _).1 (hy i)))
  have hsum : (⟨count * product.lo, count * product.hi⟩ : Interval Rat).ContainsReal some
      (∑ i, x i * y i) := by
    rw [Interval.containsReal_some_iff]
    constructor
    · simpa using Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hproduct i).1)
    · simpa using Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hproduct i).2)
  have hweighted : weighted.ContainsReal some ((∑ i, x i * y i) * coefficient) :=
    Interval.containsReal_mul? Interval.Internal.rationalOutwardRounding hw hsum
      ((Interval.containsReal_some_iff _ _).2
        ((Interval.containsReal_iff_of_decode? hc' _).1 hc))
  apply Interval.containsReal_encloseInterval? Binary.intervalRounding hout
  cases hstaged : (Binary.Interval.mul
      (sumLeftRange count (Binary.Interval.mul left right)) scale).decode? Binary.toRat? with
  | none => simpa [hstaged] using (Interval.containsReal_some_iff _ _).1 hweighted
  | some previous =>
      have hw := (Interval.containsReal_some_iff _ _).1 hweighted
      simp only [Rat.cast_min, Rat.cast_max]
      exact ⟨(min_le_left _ _).trans hw.1, hw.2.trans (le_max_left _ _)⟩

/-- Each entry of the real grouped matrix specification `Attention.scaledProductsSlice` with
one group lies in the output of `scaledContractionRange?`. The proof never equates rounded staged
multiplication with coefficient replay. -/
theorem scaledProductsSlice_contains {m n p : Nat}
    {left right scale output : Interval (Binary 8 23)}
    (h : scaledContractionRange? n left right scale = some output)
    (a : Tensor ℝ [m, n]) (b : Tensor ℝ [n, p]) (coefficient : ℝ)
    (ha : ∀ i k, left.ContainsReal Binary.toRat? (get2 a i k))
    (hb : ∀ k j, right.ContainsReal Binary.toRat? (get2 b k j))
    (hc : scale.ContainsReal Binary.toRat? coefficient) (i : Fin m) (j : Fin p) :
    output.ContainsReal Binary.toRat?
      (get2 (Attention.scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) coefficient)
        i j) := by
  simp only [Attention.scaledProductsSlice_one Numeric.QuotientArithmetic.supported_real,
    Bool.false_eq_true, ↓reduceIte]
  have hentry : get2 (Tensor.scale (Tensor.matmul a b) coefficient) i j =
      (∑ k, get2 a i k * get2 b k j) * coefficient := by
    simp [Tensor.scale, Tensor.get2_matmul_eq_sum]
  rw [hentry]
  exact scaledContractionRange?_sound h (fun k => get2 a i k) (fun k => get2 b k j)
    coefficient (ha i) (fun k => hb k j) hc

/-- If a valid interval `a` encloses `x` and its outward-rounded reciprocal is valid, that
reciprocal encloses `x⁻¹`. -/
theorem inv_realMem {a : Interval (Binary 8 23)} {x : Real}
    (ha : Binary.Interval.Valid a) (hout : Binary.Interval.Valid (Binary.Interval.inv a))
    (hx : Binary.Interval.RealMem a x) :
    Binary.Interval.RealMem (Binary.Interval.inv a) x⁻¹ := by
  have h := (Binary.Interval.eRealMem_coe_iff_of_valid hout _).mp
    (Binary.Interval.inv_sound a ha hx)
  rw [one_div] at h
  exact h

/-- Every scalar entry of a shape-indexed real tensor lies in one interval. -/
def TensorEnclosed (interval : Interval (Binary 8 23)) :
    {shape : Shape} -> Tensor Real shape -> Prop
  | .scalar, tensor => Binary.Interval.RealMem interval tensor.item
  | .dim _ _, tensor => ∀ i, TensorEnclosed interval (tensor.unstack i)

/-- The grouped real matrix result satisfies the tensor enclosure used by graph certificates. -/
theorem scaledProductsSlice_enclosed {m n p : Nat}
    {left right scale output : Interval (Binary 8 23)}
    (h : scaledContractionRange? n left right scale = some output)
    (a : Tensor ℝ [m, n]) (b : Tensor ℝ [n, p]) (coefficient : ℝ)
    (ha : ∀ i k, left.ContainsReal Binary.toRat? (get2 a i k))
    (hb : ∀ k j, right.ContainsReal Binary.toRat? (get2 b k j))
    (hc : scale.ContainsReal Binary.toRat? coefficient) :
    TensorEnclosed output
      (Attention.scaledProductsSlice (fun _ : Fin 1 => a) (fun _ => b) coefficient) := by
  intro i j
  exact realMem_of_containsReal
    (scaledProductsSlice_contains h a b coefficient ha hb hc i j)

/-- The exact executable interval `[0,1]`. -/
def unitInterval : Interval (Binary 8 23) :=
  { lo := (ExecFloat.Binary.zero false : Binary 8 23), hi := (1 : Binary 8 23) }

/-- The exact executable interval `[-1,1]`. -/
def signedUnitInterval : Interval (Binary 8 23) :=
  { lo := (-1 : Binary 8 23), hi := (1 : Binary 8 23) }

/-- Executable test for a nonnegative finite endpoint. Both signed zeros pass, as does any value
with a clear sign bit, and interval validity supplies finiteness. -/
def nonnegativeEndpoint (x : Binary 8 23) : Bool :=
  ExecFloat.Binary.isZero x || !ExecFloat.Binary.signBit x

private theorem toReal_posZero :
    (ExecFloat.Binary.toModel (ExecFloat.Binary.zero false : Binary 8 23)).toReal = 0 := by
  have hmodel : ExecFloat.Binary.toModel (ExecFloat.Binary.zero false : Binary 8 23) =
      Model.zero FloatFormat.binary32 false :=
    ExecFloat.Binary.toModel_ofModel _
  exact (congrArg Model.toReal hmodel).trans (Model.toReal_zero _ _)

/-- The real softmax of a vector (`Op.softmax` on its one axis) is enclosed by the certificate
transfer `[0,1]`. -/
@[op_cert Op.softmax enclosure]
theorem softmaxVec_tensorEnclosed {n : Nat}
    (input : Tensor Real [Nat.succ n]) :
    TensorEnclosed unitInterval
      (((Op.softmax (s := [Nat.succ n]) ⟨0, Nat.zero_lt_one⟩).rule? (α := ℝ) |>.get rfl).forward
        (input, ())).1 := by
  change TensorEnclosed _ (Tensor.softmax input ⟨0, Nat.zero_lt_one⟩)
  rw [Tensor.softmax, Tensor.alongAxis_of_last _ _ _ rfl]
  change TensorEnclosed _ (Softmax.vec input)
  intro i
  have h := Softmax.vec_mem_unitInterval input i
  have hone : (ExecFloat.Binary.toModel (1 : Binary 8 23)).toReal = 1 :=
    ExecFloat.Binary.toReal_one
  simpa [TensorEnclosed, Binary.Interval.RealMem, Binary.Interval.toModel, unitInterval,
    Tensor.getScalar, toReal_posZero, hone] using h

/-- If a valid interval `a` encloses `x`, the ReLU interval transfer encloses `max x 0`. -/
theorem relu_realMem {a : Interval (Binary 8 23)} {x : Real}
    (ha : Binary.Interval.Valid a) (hx : Binary.Interval.RealMem a x) :
    Binary.Interval.RealMem (Binary.Interval.relu a) (max x 0) := by
  change Model.Interval.RealMem (Binary.Interval.toModel (Binary.Interval.relu a)) (max x 0)
  simp only [Binary.Interval.toModel_relu]
  change
    Model.toReal (Model.maximum (Binary.Interval.toModel a).lo (Model.zero _ false)) ≤ max x 0 ∧
    max x 0 ≤ Model.toReal (Model.maximum (Binary.Interval.toModel a).hi (Model.zero _ false))
  rw [Model.toReal_maximum_eq_max_of_isFinite _ _ ha.1 (by decide),
    Model.toReal_maximum_eq_max_of_isFinite _ _ ha.2.1 (by decide), Model.toReal_zero]
  exact ⟨max_le_max hx.1 le_rfl, max_le_max hx.2 le_rfl⟩

/-- If a valid interval `a` encloses `x`, the absolute-value interval transfer encloses `|x|`. -/
theorem abs_realMem {a : Interval (Binary 8 23)} {x : Real}
    (ha : Binary.Interval.Valid a) (hx : Binary.Interval.RealMem a x) :
    Binary.Interval.RealMem (Binary.Interval.abs a) |x| := by
  change Model.Interval.RealMem (Binary.Interval.toModel (Binary.Interval.abs a)) |x|
  simp only [Binary.Interval.toModel_abs]
  change Model.Interval.RealMem (Binary.Interval.toModel a) x at hx
  by_cases hneg : Model.Interval.leB (Binary.Interval.toModel a).hi (Model.zero _ true) = true
  · have hhiNonpos : Model.toReal (Binary.Interval.toModel a).hi ≤ 0 := by
      simpa only [Model.toReal_zero] using
        (Model.Interval.leB_eq_true_iff_toReal_le_of_isFinite
          _ _ ha.2.1 (by decide)).mp hneg
    have hxNonpos : x <= 0 := hx.2.trans hhiNonpos
    rw [Model.Interval.abs, ite_eq_left hneg, abs_of_nonpos hxNonpos]
    change Model.toReal (Model.neg (Binary.Interval.toModel a).hi) ≤ -x ∧
      -x ≤ Model.toReal (Model.neg (Binary.Interval.toModel a).lo)
    rw [Model.toReal_neg _ ha.2.1, Model.toReal_neg _ ha.1]
    exact ⟨neg_le_neg hx.2, neg_le_neg hx.1⟩
  · by_cases hpos : Model.Interval.leB (Model.zero _ false) (Binary.Interval.toModel a).lo = true
    · have hloNonneg : 0 ≤ Model.toReal (Binary.Interval.toModel a).lo := by
        simpa only [Model.toReal_zero] using
          (Model.Interval.leB_eq_true_iff_toReal_le_of_isFinite
            _ _ (by decide) ha.1).mp hpos
      have hxNonneg : 0 <= x := hloNonneg.trans hx.1
      rw [Model.Interval.abs, ite_eq_right hneg, ite_eq_left hpos, abs_of_nonneg hxNonneg]
      exact hx
    · rw [Model.Interval.abs, ite_eq_right hneg, ite_eq_right hpos]
      change Model.toReal (Model.zero _ false) ≤ |x| ∧
        |x| ≤ Model.toReal (Model.maximum
          (Model.neg (Binary.Interval.toModel a).lo) (Binary.Interval.toModel a).hi)
      rw [Model.toReal_zero, Model.toReal_maximum_eq_max_of_isFinite _ _
        (by simpa only [Model.isFinite_neg] using ha.1) ha.2.1,
        Model.toReal_neg _ ha.1]
      refine ⟨abs_nonneg x, (abs_le).2 ⟨?_, hx.2.trans (le_max_right _ _)⟩⟩
      have := le_max_left
        (-Model.toReal (Binary.Interval.toModel a).lo) (Model.toReal (Binary.Interval.toModel a).hi)
      linarith [hx.1]

private theorem toReal_nonneg_of_endpoint {x : Binary 8 23}
    (hdomain : nonnegativeEndpoint x = true) :
    0 ≤ (ExecFloat.Binary.toModel x).toReal := by
  by_cases hzero : ExecFloat.Binary.isZero x = true
  · exact (Model.toReal_eq_zero_of_isZero (ExecFloat.Binary.toModel x) hzero).ge
  · have hsign : ExecFloat.Binary.signBit x = false := by
      simpa [nonnegativeEndpoint, hzero] using hdomain
    exact Model.toReal_nonneg_of_signBit_eq_false hsign

/-- For a valid interval `a` enclosing `x`, directed interval square root encloses `Real.sqrt x`
when `a.lo` passes `nonnegativeEndpoint` and the result is valid. The upper endpoint needs no sign
test, since interval order makes it at least the lower one. -/
theorem sqrt_realMem {a : Interval (Binary 8 23)} {x : Real}
    (ha : Binary.Interval.Valid a) (hlo : nonnegativeEndpoint a.lo = true)
    (hout : Binary.Interval.Valid (Binary.Interval.sqrt a))
    (hx : Binary.Interval.RealMem a x) :
    Binary.Interval.RealMem (Binary.Interval.sqrt a) (Real.sqrt x) :=
  (Binary.Interval.eRealMem_coe_iff_of_valid hout _).mp
    (Binary.Interval.sqrt_sound a ha (toReal_nonneg_of_endpoint hlo) hx)

/-- Lift a sound unary scalar transfer to tensors of arbitrary rank. -/
theorem map_tensorEnclosed
    (op : Real -> Real) (input output : Interval (Binary 8 23))
    (sound : ∀ {x}, Binary.Interval.RealMem input x -> Binary.Interval.RealMem output (op x)) :
    ∀ {shape : Shape} {x : Tensor Real shape},
      TensorEnclosed input x -> TensorEnclosed output (Tensor.map op x) := by
  intro shape
  induction shape with
  | scalar =>
      intro x hx
      exact sound hx
  | dim n shape ih =>
      intro x hx i
      rw [show (Tensor.map op x).unstack i = Tensor.map op (x.unstack i) by
        exact Tensor.unstack_map op x i]
      exact ih (x := x.unstack i) (hx i)

/-- Tensor-level soundness of the ReLU interval transfer for `Op.relu`. -/
@[op_cert Op.relu enclosure]
theorem relu_tensorEnclosed {shape : Shape} {x : Tensor Real shape}
    {a : Interval (Binary 8 23)} (ha : Binary.Interval.Valid a) (hx : TensorEnclosed a x) :
    TensorEnclosed (Binary.Interval.relu a)
      (((Op.relu (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1 := by
  change TensorEnclosed _ (Tensor.map Activation.relu x)
  rw [show (Activation.relu : ℝ → ℝ) = fun value => max value 0 from
    funext Activation.relu_eq_max]
  exact map_tensorEnclosed (fun value => max value 0) a (Binary.Interval.relu a)
    (fun hx' => relu_realMem ha hx') hx

/-- Tensor-level soundness of the absolute-value interval transfer for `Op.abs`. -/
@[op_cert Op.abs enclosure]
theorem abs_tensorEnclosed {shape : Shape} {x : Tensor Real shape}
    {a : Interval (Binary 8 23)} (ha : Binary.Interval.Valid a) (hx : TensorEnclosed a x) :
    TensorEnclosed (Binary.Interval.abs a)
      (((Op.abs (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1 := by
  change TensorEnclosed _ (Tensor.map _root_.abs x)
  exact map_tensorEnclosed _root_.abs a (Binary.Interval.abs a)
    (fun hx' => abs_realMem ha hx') hx

/-- Tensor-level soundness of directed interval square root for `Op.sqrt`. -/
@[op_cert Op.sqrt enclosure]
theorem sqrt_tensorEnclosed {shape : Shape} {x : Tensor Real shape}
    {a : Interval (Binary 8 23)} (ha : Binary.Interval.Valid a)
    (hlo : nonnegativeEndpoint a.lo = true)
    (hout : Binary.Interval.Valid (Binary.Interval.sqrt a)) (hx : TensorEnclosed a x) :
    TensorEnclosed (Binary.Interval.sqrt a)
      (((Op.sqrt (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, ())).1 := by
  change TensorEnclosed _ (Tensor.map Real.sqrt x)
  exact map_tensorEnclosed Real.sqrt a (Binary.Interval.sqrt a)
    (fun hx' => sqrt_realMem ha hlo hout hx') hx

/-- Lift a sound binary scalar transfer to tensors of arbitrary rank. -/
theorem zipWith_tensorEnclosed
    (op : Real -> Real -> Real) (a b out : Interval (Binary 8 23))
    (sound : ∀ {x y}, Binary.Interval.RealMem a x -> Binary.Interval.RealMem b y ->
      Binary.Interval.RealMem out (op x y)) :
    ∀ {shape : Shape} {x y : Tensor Real shape},
      TensorEnclosed a x -> TensorEnclosed b y ->
        TensorEnclosed out (Tensor.zipWith op x y) := by
  intro shape
  induction shape with
  | scalar =>
      intro x y hx hy
      exact sound hx hy
  | dim n shape ih =>
      intro x y hx hy i
      rw [show (Tensor.zipWith op x y).unstack i =
          Tensor.zipWith op (x.unstack i) (y.unstack i) by
        exact (Tensor.Rep.zipWith_unstack op x y i).symm]
      exact ih (x := x.unstack i) (y := y.unstack i) (hx i) (hy i)

/-- Tensor-level soundness of outward-rounded interval addition for `Op.add`. -/
@[op_cert Op.add enclosure]
theorem add_tensorEnclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : Interval (Binary 8 23)}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.add a b))
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (Binary.Interval.add a b)
      (((Op.add (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, y, ())).1 :=
  zipWith_tensorEnclosed (fun u v => u + v) a b (Binary.Interval.add a b)
    (fun hx' hy' => add_realMem ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval subtraction for `Op.sub`. -/
@[op_cert Op.sub enclosure]
theorem sub_tensorEnclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : Interval (Binary 8 23)}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.sub a b))
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (Binary.Interval.sub a b)
      (((Op.sub (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, y, ())).1 :=
  zipWith_tensorEnclosed (fun u v => u - v) a b (Binary.Interval.sub a b)
    (fun hx' hy' => sub_realMem ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval multiplication for `Op.mul`. -/
@[op_cert Op.mul enclosure]
theorem mul_tensorEnclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : Interval (Binary 8 23)}
    (ha : Binary.Interval.Valid a) (hb : Binary.Interval.Valid b)
    (hout : Binary.Interval.Valid (Binary.Interval.mul a b))
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (Binary.Interval.mul a b)
      (((Op.mul (s := shape)).rule? (α := ℝ) |>.get rfl).forward (x, y, ())).1 :=
  zipWith_tensorEnclosed (fun u v => u * v) a b (Binary.Interval.mul a b)
    (fun hx' hy' => mul_realMem ha hb hout hx' hy') hx hy

/-! ## Replay against bit-level graph execution -/

/-- Executable check that every binary32 tensor entry lies in an interval. -/
def tensorWithinRange (interval : Interval (Binary 8 23)) :
    {shape : Shape} -> Tensor (Binary 8 23) shape -> Bool
  | .scalar, tensor =>
      ExecFloat.Binary.isFinite tensor.item &&
        (Binary.Interval.leB interval.lo tensor.item &&
          Binary.Interval.leB tensor.item interval.hi)
  | .dim n _, tensor =>
      (List.finRange n).all (fun i => tensorWithinRange interval (tensor.unstack i))

/-- Proposition expressed by `tensorWithinRange`. -/
def IEEETensorEnclosed (interval : Interval (Binary 8 23)) :
    {shape : Shape} -> Tensor (Binary 8 23) shape -> Prop
  | .scalar, tensor =>
      ExecFloat.Binary.isFinite tensor.item = true ∧
        LE.le interval.lo tensor.item ∧ LE.le tensor.item interval.hi
  | .dim _ _, tensor => ∀ i, IEEETensorEnclosed interval (tensor.unstack i)

/-- `tensorWithinRange` returns `true` exactly when `IEEETensorEnclosed` holds. -/
theorem tensorWithinRange_eq_true_iff (interval : Interval (Binary 8 23))
    {shape : Shape} (tensor : Tensor (Binary 8 23) shape) :
    tensorWithinRange interval tensor = true <-> IEEETensorEnclosed interval tensor := by
  induction shape with
  | scalar =>
      simp [tensorWithinRange, IEEETensorEnclosed, leB_eq_true_iff]
  | dim n shape ih =>
      simp [tensorWithinRange, IEEETensorEnclosed, List.all_eq_true, ih]

/-! ## From checked ranges to explicit error bounds -/

/-- Every entry of an executable tensor, decoded to a real number, lies in `interval`. The
approximation layer works with these real values, while `IEEETensorEnclosed` compares entries in
the IEEE order. -/
def DecodedTensorEnclosed (interval : Interval (Binary 8 23)) :
    {shape : Shape} -> Tensor (Binary 8 23) shape -> Prop
  | .scalar, tensor =>
      Binary.Interval.RealMem interval ((ExecFloat.Binary.toModel tensor.item).toReal)
  | .dim _ _, tensor => ∀ i, DecodedTensorEnclosed interval (tensor.unstack i)

/-- For a valid interval, `IEEETensorEnclosed` implies `DecodedTensorEnclosed`. The IEEE predicate
requires every entry to be finite, so NaN and the infinities never receive a real value. -/
theorem decodedTensorEnclosed_of_ieee {interval : Interval (Binary 8 23)}
    (valid : Binary.Interval.Valid interval) :
    ∀ {shape : Shape} {tensor : Tensor (Binary 8 23) shape},
      IEEETensorEnclosed interval tensor -> DecodedTensorEnclosed interval tensor := by
  intro shape
  induction shape with
  | scalar =>
      intro tensor htensor
      exact ⟨(ExecFloat.Binary.le_iff_toReal_le valid.1 htensor.1).mp htensor.2.1,
        (ExecFloat.Binary.le_iff_toReal_le htensor.1 valid.2.1).mp htensor.2.2⟩
  | dim n shape ih =>
      intro tensor htensor i
      exact ih (htensor i)

/-- Every entry of an executable binary32 tensor is within `eps` of the matching entry of a real
specification tensor. Both tensors share one shape index, so the relation hides no runtime shape
cast. -/
def TensorErrorLe (eps : Real) :
    {shape : Shape} -> Tensor Real shape -> Tensor (Binary 8 23) shape -> Prop
  | .scalar, exact, computed =>
      |(ExecFloat.Binary.toModel computed.item).toReal - exact.item| <= eps
  | .dim _ _, exact, computed =>
      ∀ i, TensorErrorLe eps (exact.unstack i) (computed.unstack i)

/-- Width of a finite executable interval, interpreted in the reals. -/
noncomputable def intervalWidth (interval : Interval (Binary 8 23)) : Real :=
  (ExecFloat.Binary.toModel interval.hi).toReal - (ExecFloat.Binary.toModel interval.lo).toReal

/-- A valid interval has nonnegative real width. -/
theorem intervalWidth_nonneg {interval : Interval (Binary 8 23)}
    (valid : Binary.Interval.Valid interval) :
    0 <= intervalWidth interval := by
  change 0 ≤ Model.toReal (Binary.Interval.toModel interval).hi -
    Model.toReal (Binary.Interval.toModel interval).lo
  exact sub_nonneg.mpr (Model.Interval.Valid.toReal_ordered valid)

/-- Two tensors enclosed by the same interval differ entrywise by at most its width. -/
theorem TensorErrorLe.of_enclosed {interval : Interval (Binary 8 23)} :
    ∀ {shape : Shape} {exact : Tensor Real shape} {computed : Tensor (Binary 8 23) shape},
      TensorEnclosed interval exact ->
      DecodedTensorEnclosed interval computed ->
      TensorErrorLe (intervalWidth interval) exact computed := by
  intro shape
  induction shape with
  | scalar =>
      intro exact computed hexact hcomputed
      change |Model.toReal (ExecFloat.Binary.toModel computed.item) - exact.item| ≤
        Model.toReal (Binary.Interval.toModel interval).hi -
          Model.toReal (Binary.Interval.toModel interval).lo
      apply (abs_le).2
      constructor <;> linarith [hexact.1, hexact.2, hcomputed.1, hcomputed.2]
  | dim n shape ih =>
      intro exact computed hexact hcomputed i
      exact ih (hexact i) (hcomputed i)

/-- If a valid `interval` encloses the real tensor `exact` and `tensorWithinRange` accepts
`computed`, every entry of `computed` is within `intervalWidth interval` of `exact`. Graph-wide
numerical certificates apply it to each certificate row (`SomeTensorErrorLe.of_check`). -/
theorem TensorErrorLe.of_check {interval : Interval (Binary 8 23)}
    (valid : Binary.Interval.Valid interval) {shape : Shape} {exact : Tensor Real shape}
    {computed : Tensor (Binary 8 23) shape}
    (hexact : TensorEnclosed interval exact)
    (hcheck : tensorWithinRange interval computed = true) :
    TensorErrorLe (intervalWidth interval) exact computed :=
  of_enclosed hexact <|
    decodedTensorEnclosed_of_ieee valid <|
      (tensorWithinRange_eq_true_iff interval computed).mp hcheck

/-- Check source ranges once, rejecting malformed intervals and duplicate node ids. -/
def checkSources (sources : Array SourceRange) : Except String (Array CheckedSourceRange) := do
  let mut checked : Array CheckedSourceRange := #[]
  let mut seen : Array Nat := #[]
  for source in sources do
    if seen.contains source.nodeId then
      throw s!"numerical certificate: duplicate source range for node {source.nodeId}"
    if h : Binary.Interval.Valid source.enclosure then
      checked := checked.push { source with valid := h }
      seen := seen.push source.nodeId
    else
      throw (s!"numerical certificate: source range for node {source.nodeId} is not finite " ++
        "and ordered")
  pure checked

/-- Find the checked assumption for a source node. -/
def findSource (sources : Array CheckedSourceRange) (nodeId : Nat) :
    Except String CheckedSourceRange :=
  match sources.find? (fun source => source.nodeId == nodeId) with
  | some source => pure source
  | none => throw s!"numerical certificate: missing source range for node {nodeId}"

end NumericalCertificate
end TorchLean
