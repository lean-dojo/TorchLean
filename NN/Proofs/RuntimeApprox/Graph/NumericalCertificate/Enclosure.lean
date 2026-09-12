/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Proofs.Analysis.Softmax
public import NN.Floats.IEEEExec.Semantics.ERealSemantics
public import NN.Floats.Interval.IEEEExec32AddSoundness
public import NN.Floats.Interval.IEEEExec32DivSoundness
public import NN.Floats.Interval.IEEEExec32MulSoundness
public import NN.Backend.Profile -- shake: keep
public import NN.Floats.Interval.IEEEExec32Soundness -- shake: keep
public import NN.IR.Semantics -- shake: keep
public import NN.Spec.Core.FloatInstances -- shake: keep
public import NN.Spec.Core.TensorOps -- shake: keep
public import NN.Floats.IEEEExec.Bridge.FP32.Compare -- shake: keep
public import NN.Floats.IEEEExec.Bridge.FP32.Core -- shake: keep
public import NN.Floats.IEEEExec.Bridge.FP32Total.Order -- shake: keep
public import NN.Floats.IEEEExec.DirectedRoundingSoundness.SignedOps -- shake: keep
public import NN.Floats.IEEEExec.Semantics.RealSemantics -- shake: keep

/-!
# Numerical certificate enclosures

Foundational source-range validation, real and IEEE enclosure semantics, replay checks, and
pointwise error traces for graph numerical certificates. Most users should import
`NN.Proofs.RuntimeApprox.Graph.NumericalCertificate`.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace NumericalCertificate

open NN
open NN.Backend
open NN.IR
open Spec TorchLean
open TorchLean.Floats.IEEE754

/-! ## Raw and checked source assumptions -/

/-- A binary32 range supplied for an input, constant, or explicit random source node. -/
structure SourceRange where
  nodeId : Nat
  enclosure : IEEE32Exec.Interval32
  deriving Repr

/-- A source range after the checker has established finite, ordered endpoints. -/
structure CheckedSourceRange extends SourceRange where
  valid : enclosure.Valid

instance : Repr CheckedSourceRange where
  reprPrec r _ := repr r.toSourceRange

/-- Bitwise equality for executable binary32 intervals.

Bitwise equality is intentional: it distinguishes signed zero and preserves the exact endpoints
written in a certificate. NaNs are rejected separately by `Interval32.Valid`.
-/
def sameIntervalBits (a b : IEEE32Exec.Interval32) : Bool :=
  a.lo.bits == b.lo.bits && a.hi.bits == b.hi.bits

/-- Executable counterpart of `Interval32.Valid`. -/
def validInterval (interval : IEEE32Exec.Interval32) : Bool :=
  IEEE32Exec.isFinite interval.lo &&
    (IEEE32Exec.isFinite interval.hi && IEEE32Exec.Interval32.leB interval.lo interval.hi)

/-- `Interval32.leB` decides the proposition-level IEEE non-strict order. -/
theorem leB_eq_true_iff (x y : IEEE32Exec) :
    IEEE32Exec.Interval32.leB x y = true <-> IEEE32Exec.le x y := by
  unfold IEEE32Exec.Interval32.leB IEEE32Exec.le
  cases h : IEEE32Exec.compare x y with
  | none => simp
  | some order =>
      cases order <;> simp

/-- IEEE comparison between finite values implies the corresponding order on their real
interpretations. This lemma is intentionally finite: IEEE comparisons involving NaN are unordered,
and `toReal` is not the semantic interface for infinities. -/
theorem toReal_le_toReal_of_le {x y : IEEE32Exec}
    (hx : IEEE32Exec.isFinite x = true) (hy : IEEE32Exec.isFinite y = true)
    (hxy : IEEE32Exec.le x y) : IEEE32Exec.toReal x <= IEEE32Exec.toReal y := by
  unfold IEEE32Exec.le at hxy
  cases hcompare : IEEE32Exec.compare x y with
  | none => simp [hcompare] at hxy
  | some order =>
      cases order with
      | lt =>
          exact le_of_lt <|
            (IEEE32Exec.compare_eq_some_lt_iff_toReal_lt_of_isFinite x y hx hy).mp hcompare
      | eq =>
          exact le_of_eq <|
            (IEEE32Exec.compare_eq_some_eq_iff_toReal_eq_of_isFinite x y hx hy).mp hcompare
      | gt => simp [hcompare] at hxy

/-- Negation of a finite executable binary32 value decodes to real negation. -/
theorem toReal_neg_of_isFinite {x : IEEE32Exec} (hx : IEEE32Exec.isFinite x = true) :
    IEEE32Exec.toReal (IEEE32Exec.neg x) = -IEEE32Exec.toReal x := by
  obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hx
  exact IEEE32Exec.toReal_neg_eq_neg x hdx

/-- Flipping the sign bit preserves finiteness. -/
theorem isFinite_neg_of_isFinite {x : IEEE32Exec} (hx : IEEE32Exec.isFinite x = true) :
    IEEE32Exec.isFinite (IEEE32Exec.neg x) = true := by
  obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hx
  have hdxNeg := IEEE32Exec.toDyadic?_neg_of_toDyadic?_some x hdx
  have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdxNeg
  have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdxNeg
  exact IEEE32Exec.isFinite_eq_true_of_isNaN_eq_false_of_isInf_eq_false
    (IEEE32Exec.neg x) hnan hinf

/-- The executable validity test accepts exactly finite, ordered intervals. -/
theorem validInterval_eq_true_iff (interval : IEEE32Exec.Interval32) :
    validInterval interval = true <-> interval.Valid := by
  simp [validInterval, IEEE32Exec.Interval32.Valid, leB_eq_true_iff]

/-! ## Real semantics of the arithmetic transfers

The executable checker propagates binary32 endpoints, while the graph specification is normally
read over real scalars. `RealEncloses` is the bridge between those views. Runtime rounding error is
then composed separately by `FwdGraph.eval_approx` and `RevGraph.backprop_approx`; keeping these two
claims separate prevents an interval enclosure from silently standing in for a floating-point
error theorem.
-/

/-- A real scalar lies between the real interpretations of an executable interval's endpoints. -/
def RealEncloses (interval : IEEE32Exec.Interval32) (value : Real) : Prop :=
  value ∈ Set.Icc (IEEE32Exec.toReal interval.lo) (IEEE32Exec.toReal interval.hi)

/-- Convert the extended-real endpoint form used by the interval soundness library into an
ordinary real interval when the output endpoints are finite. -/
theorem realEncloses_of_eReal_bounds {interval : IEEE32Exec.Interval32} {value : Real}
    (valid : interval.Valid)
    (bounds : IEEE32Exec.toEReal interval.lo <= (value : EReal) ∧
      (value : EReal) <= IEEE32Exec.toEReal interval.hi) :
    RealEncloses interval value := by
  have hlo := IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (x := interval.lo) valid.1
  have hhi := IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (x := interval.hi) valid.2.1
  constructor
  · rw [hlo] at bounds
    exact EReal.coe_le_coe_iff.mp bounds.1
  · rw [hhi] at bounds
    exact EReal.coe_le_coe_iff.mp bounds.2

/-- Sound real enclosure for the canonical addition transfer. -/
theorem add_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.add b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.add b) (x + y) :=
  realEncloses_of_eReal_bounds hout (a.add_sound b ha hb hx hy)

/-- Sound real enclosure for the canonical subtraction transfer. -/
theorem sub_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.sub b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.sub b) (x - y) :=
  realEncloses_of_eReal_bounds hout (a.sub_sound b ha hb hx hy)

/-- Sound real enclosure for the canonical multiplication transfer. -/
theorem mul_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.mul b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.mul b) (x * y) :=
  realEncloses_of_eReal_bounds hout (a.mul_sound b ha.1 ha.2.1 hb.1 hb.2.1 hx hy)

/-- Sound real enclosure for the canonical reciprocal transfer. -/
theorem inv_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hout : a.inv.Valid) (hx : RealEncloses a x) :
    RealEncloses a.inv x⁻¹ := by
  simpa [one_div] using
    (realEncloses_of_eReal_bounds hout (a.inv_sound ha hx))

/-- Every scalar entry of a shape-indexed real tensor lies in one interval. -/
def TensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor Real shape -> Prop
  | .scalar, tensor => RealEncloses interval tensor.item
  | .dim _ _, tensor => ∀ i, TensorEnclosed interval (tensor.unstack i)

/-- The exact executable interval `[0,1]`. -/
def unitInterval : IEEE32Exec.Interval32 :=
  { lo := IEEE32Exec.posZero, hi := IEEE32Exec.posOne }

/-- The exact executable interval `[-1,1]`. -/
def signedUnitInterval : IEEE32Exec.Interval32 :=
  { lo := IEEE32Exec.negOne, hi := IEEE32Exec.posOne }

/-- Executable test for a finite endpoint's nonnegative IEEE sign. Both signed zeros are accepted;
all other accepted values have a clear sign bit. Finiteness is supplied by interval validity. -/
def nonnegativeEndpoint (x : IEEE32Exec) : Bool :=
  IEEE32Exec.isZero x || !IEEE32Exec.signBit x

/-- The stable real vector softmax is enclosed by the certificate transfer `[0,1]`. -/
theorem softmaxVec_tensor_enclosed {n : Nat}
    (input : Tensor Real [Nat.succ n]) :
    TensorEnclosed unitInterval (Activation.softmaxVecSpec input) := by
  intro i
  have h := Proofs.softmax_vec_spec_mem_unitInterval input i
  simpa [TensorEnclosed, RealEncloses, unitInterval, Spec.get, TorchLean.Tensor.getScalar,
    IEEE32Exec.Interval32.toReal_posOne] using h

/-- Sound real enclosure for the canonical ReLU interval transfer. -/
theorem relu_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.relu (max x 0) := by
  have hzero : IEEE32Exec.isFinite IEEE32Exec.posZero = true := by decide
  have hlo := IEEE32Exec.toReal_maximum_eq_max_of_isFinite a.lo IEEE32Exec.posZero ha.1 hzero
  have hhi := IEEE32Exec.toReal_maximum_eq_max_of_isFinite a.hi IEEE32Exec.posZero ha.2.1 hzero
  constructor
  · simpa [IEEE32Exec.Interval32.relu, hlo] using max_le_max hx.1 (le_refl (0 : Real))
  · simpa [IEEE32Exec.Interval32.relu, hhi] using max_le_max hx.2 (le_refl (0 : Real))

/-- Sound real enclosure for the canonical absolute-value interval transfer. -/
theorem abs_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.abs |x| := by
  by_cases hneg : IEEE32Exec.Interval32.leB a.hi IEEE32Exec.negZero = true
  · have hhiNonpos : IEEE32Exec.toReal a.hi <= 0 := by
      have hle := (leB_eq_true_iff a.hi IEEE32Exec.negZero).mp hneg
      simpa using toReal_le_toReal_of_le ha.2.1 (by decide) hle
    have hxNonpos : x <= 0 := hx.2.trans hhiNonpos
    have hnegLo := toReal_neg_of_isFinite ha.1
    have hnegHi := toReal_neg_of_isFinite ha.2.1
    rw [abs_of_nonpos hxNonpos]
    constructor <;>
      simp only [IEEE32Exec.Interval32.abs, hneg, if_pos, IEEE32Exec.Interval32.neg] <;>
      simp only [hnegLo, hnegHi] <;> linarith [hx.1, hx.2]
  · by_cases hpos : IEEE32Exec.Interval32.leB IEEE32Exec.posZero a.lo = true
    · have hloNonneg : 0 <= IEEE32Exec.toReal a.lo := by
        have hle := (leB_eq_true_iff IEEE32Exec.posZero a.lo).mp hpos
        simpa using toReal_le_toReal_of_le (by decide) ha.1 hle
      have hxNonneg : 0 <= x := hloNonneg.trans hx.1
      simpa [IEEE32Exec.Interval32.abs, hneg, hpos, abs_of_nonneg hxNonneg] using hx
    · have hzero : IEEE32Exec.isFinite IEEE32Exec.posZero = true := by decide
      have hnegLo := toReal_neg_of_isFinite ha.1
      have hmax := IEEE32Exec.toReal_maximum_eq_max_of_isFinite
        (IEEE32Exec.neg a.lo) a.hi
        (isFinite_neg_of_isFinite ha.1) ha.2.1
      have hupper : |x| <= max (-IEEE32Exec.toReal a.lo) (IEEE32Exec.toReal a.hi) := by
        apply (abs_le).2
        constructor
        · have := le_max_left (-IEEE32Exec.toReal a.lo) (IEEE32Exec.toReal a.hi)
          linarith [hx.1]
        · exact hx.2.trans (le_max_right _ _)
      constructor
      · simp [IEEE32Exec.Interval32.abs, hneg, hpos, abs_nonneg]
      · simpa [IEEE32Exec.Interval32.abs, hneg, hpos, hnegLo, hmax] using hupper

/-- A directed lower square-root endpoint lies below the exact real square root. Signed zero is
handled separately because IEEE preserves its sign, while the general directed-rounding theorem is
stated for sign-bit-false inputs. -/
theorem toReal_sqrtDown_le {x : IEEE32Exec}
    (hfin : IEEE32Exec.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : IEEE32Exec.isFinite (IEEE32Exec.sqrtDown x) = true) :
    IEEE32Exec.toReal (IEEE32Exec.sqrtDown x) <= Real.sqrt (IEEE32Exec.toReal x) := by
  by_cases hzero : IEEE32Exec.isZero x = true
  · obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hfin
    have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdx
    have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdx
    have hchoose : IEEE32Exec.chooseNaN1 x = none := by simp [IEEE32Exec.chooseNaN1, hnan]
    have hsqrt : IEEE32Exec.sqrtDown x = x := by
      simp [IEEE32Exec.sqrtDown, hchoose, hinf, hzero]
    have hreal := IEEE32Exec.toReal_eq_zero_of_isZero x hdx hzero
    simp [hsqrt, hreal]
  · have hsign : IEEE32Exec.signBit x = false := by
      simp [nonnegativeEndpoint, hzero] at hdomain
      exact hdomain
    have h := IEEE32Exec.toEReal_sqrtDown_le x hfin hsign
    rw [IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (IEEE32Exec.sqrtDown x) hout] at h
    exact EReal.coe_le_coe_iff.mp h

/-- Upper counterpart of `toReal_sqrtDown_le`. -/
theorem toReal_sqrtUp_ge {x : IEEE32Exec}
    (hfin : IEEE32Exec.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : IEEE32Exec.isFinite (IEEE32Exec.sqrtUp x) = true) :
    Real.sqrt (IEEE32Exec.toReal x) <= IEEE32Exec.toReal (IEEE32Exec.sqrtUp x) := by
  by_cases hzero : IEEE32Exec.isZero x = true
  · obtain ⟨dx, hdx⟩ := IEEE32Exec.exists_toDyadic?_of_isFinite hfin
    have hnan := IEEE32Exec.isNaN_eq_false_of_toDyadic?_some hdx
    have hinf := IEEE32Exec.isInf_eq_false_of_toDyadic?_some hdx
    have hchoose : IEEE32Exec.chooseNaN1 x = none := by simp [IEEE32Exec.chooseNaN1, hnan]
    have hsqrt : IEEE32Exec.sqrtUp x = x := by
      simp [IEEE32Exec.sqrtUp, hchoose, hinf, hzero]
    have hreal := IEEE32Exec.toReal_eq_zero_of_isZero x hdx hzero
    simp [hsqrt, hreal]
  · have hsign : IEEE32Exec.signBit x = false := by
      simp [nonnegativeEndpoint, hzero] at hdomain
      exact hdomain
    have h := IEEE32Exec.toEReal_sqrtUp_ge x hfin hsign
    rw [IEEE32Exec.toEReal_eq_coe_toReal_of_isFinite (IEEE32Exec.sqrtUp x) hout] at h
    exact EReal.coe_le_coe_iff.mp h

/-- Sound real enclosure for directed interval square root. -/
theorem sqrt_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hlo : nonnegativeEndpoint a.lo = true)
    (hhi : nonnegativeEndpoint a.hi = true) (hout : a.sqrt.Valid)
    (hx : RealEncloses a x) : RealEncloses a.sqrt (Real.sqrt x) := by
  constructor
  · exact (toReal_sqrtDown_le ha.1 hlo hout.1).trans (Real.sqrt_le_sqrt hx.1)
  · exact (Real.sqrt_le_sqrt hx.2).trans (toReal_sqrtUp_ge ha.2.1 hhi hout.2.1)

/-- Lift a sound unary scalar transfer to tensors of arbitrary rank. -/
theorem tensor_map_enclosed
    (op : Real -> Real) (input output : IEEE32Exec.Interval32)
    (sound : ∀ {x}, RealEncloses input x -> RealEncloses output (op x)) :
    ∀ {shape : Shape} {x : Tensor Real shape},
      TensorEnclosed input x -> TensorEnclosed output (Tensor.mapSpec op x) := by
  intro shape
  induction shape with
  | scalar =>
      intro x hx
      exact sound hx
  | dim n shape ih =>
      intro x hx i
      rw [show (Tensor.mapSpec op x).unstack i = Tensor.mapSpec op (x.unstack i) by
        exact (TorchLean.Tensor.Internal.Rep.map_unstack op x i).symm]
      exact ih (x := x.unstack i) (hx i)

/-- Tensor-level soundness of the ReLU interval transfer. -/
theorem tensor_relu_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.relu (Tensor.mapSpec (fun value => max value 0) x) :=
  tensor_map_enclosed (fun value => max value 0) a a.relu
    (fun hx' => relu_realEncloses ha hx') hx

/-- Tensor-level soundness of the absolute-value interval transfer. -/
theorem tensor_abs_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.abs (Tensor.mapSpec abs x) :=
  tensor_map_enclosed abs a a.abs (fun hx' => abs_realEncloses ha hx') hx

/-- Tensor-level soundness of directed interval square root. -/
theorem tensor_sqrt_enclosed {shape : Shape} {x : Tensor Real shape}
    {a : IEEE32Exec.Interval32} (ha : a.Valid)
    (hlo : nonnegativeEndpoint a.lo = true) (hhi : nonnegativeEndpoint a.hi = true)
    (hout : a.sqrt.Valid) (hx : TensorEnclosed a x) :
    TensorEnclosed a.sqrt (Tensor.mapSpec Real.sqrt x) :=
  tensor_map_enclosed Real.sqrt a a.sqrt
    (fun hx' => sqrt_realEncloses ha hlo hhi hout hx') hx

/-- Lift a sound binary scalar transfer to tensors of arbitrary rank. -/
theorem tensor_map2_enclosed
    (op : Real -> Real -> Real) (a b out : IEEE32Exec.Interval32)
    (sound : ∀ {x y}, RealEncloses a x -> RealEncloses b y -> RealEncloses out (op x y)) :
    ∀ {shape : Shape} {x y : Tensor Real shape},
      TensorEnclosed a x -> TensorEnclosed b y ->
        TensorEnclosed out (Tensor.map2Spec op x y) := by
  intro shape
  induction shape with
  | scalar =>
      intro x y hx hy
      exact sound hx hy
  | dim n shape ih =>
      intro x y hx hy i
      rw [show (Tensor.map2Spec op x y).unstack i =
          Tensor.map2Spec op (x.unstack i) (y.unstack i) by
        exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack op x y i).symm]
      exact ih (x := x.unstack i) (y := y.unstack i) (hx i) (hy i)

/-- Tensor-level soundness of outward-rounded interval addition. -/
theorem tensor_add_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.add b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.add b) (Tensor.addSpec x y) :=
  tensor_map2_enclosed (fun u v => u + v) a b (a.add b)
    (fun hx' hy' => add_realEncloses ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval subtraction. -/
theorem tensor_sub_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.sub b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.sub b) (Tensor.subSpec x y) :=
  tensor_map2_enclosed (fun u v => u - v) a b (a.sub b)
    (fun hx' hy' => sub_realEncloses ha hb hout hx' hy') hx hy

/-- Tensor-level soundness of outward-rounded interval multiplication. -/
theorem tensor_mul_enclosed {shape : Shape} {x y : Tensor Real shape}
    {a b : IEEE32Exec.Interval32}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.mul b).Valid)
    (hx : TensorEnclosed a x) (hy : TensorEnclosed b y) :
    TensorEnclosed (a.mul b) (Tensor.mulSpec x y) :=
  tensor_map2_enclosed (fun u v => u * v) a b (a.mul b)
    (fun hx' hy' => mul_realEncloses ha hb hout hx' hy') hx hy

/-! ## Replay against bit-level graph execution -/

/-- Executable check that every binary32 tensor entry lies in an interval. -/
def tensorWithinRange (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Bool
  | .scalar, tensor =>
      IEEE32Exec.isFinite tensor.item &&
        (IEEE32Exec.Interval32.leB interval.lo tensor.item &&
          IEEE32Exec.Interval32.leB tensor.item interval.hi)
  | .dim n _, tensor =>
      (List.finRange n).all (fun i => tensorWithinRange interval (tensor.unstack i))

/-- Proposition expressed by `tensorWithinRange`. -/
def IEEETensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Prop
  | .scalar, tensor =>
      IEEE32Exec.isFinite tensor.item = true ∧
        IEEE32Exec.le interval.lo tensor.item ∧ IEEE32Exec.le tensor.item interval.hi
  | .dim _ _, tensor => ∀ i, IEEETensorEnclosed interval (tensor.unstack i)

/-- The executable tensor range check is exact for the IEEE comparison semantics. -/
theorem tensorWithinRange_eq_true_iff (interval : IEEE32Exec.Interval32)
    {shape : Shape} (tensor : Tensor IEEE32Exec shape) :
    tensorWithinRange interval tensor = true <-> IEEETensorEnclosed interval tensor := by
  induction shape with
  | scalar =>
      simp [tensorWithinRange, IEEETensorEnclosed, leB_eq_true_iff]
  | dim n shape ih =>
      simp [tensorWithinRange, IEEETensorEnclosed, List.all_eq_true, ih]

/-! ## From checked ranges to explicit error bounds -/

/-- Decode an executable tensor entrywise and state that the resulting real tensor lies in an
interval. Unlike `IEEETensorEnclosed`, this predicate talks directly about the real values used by
the approximation layer. -/
def DecodedTensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor IEEE32Exec shape -> Prop
  | .scalar, tensor => RealEncloses interval (IEEE32Exec.toReal tensor.item)
  | .dim _ _, tensor => ∀ i, DecodedTensorEnclosed interval (tensor.unstack i)

/-- A successful IEEE range check decodes to an ordinary real enclosure. Finiteness is an explicit
part of `IEEETensorEnclosed`, so this theorem never assigns a real meaning to NaN or infinity. -/
theorem decodedTensorEnclosed_of_ieee {interval : IEEE32Exec.Interval32}
    (valid : interval.Valid) :
    ∀ {shape : Shape} {tensor : Tensor IEEE32Exec shape},
      IEEETensorEnclosed interval tensor -> DecodedTensorEnclosed interval tensor := by
  intro shape
  induction shape with
  | scalar =>
      intro tensor htensor
      exact ⟨toReal_le_toReal_of_le valid.1 htensor.1 htensor.2.1,
        toReal_le_toReal_of_le htensor.1 valid.2.1 htensor.2.2⟩
  | dim n shape ih =>
      intro tensor htensor i
      exact ih (htensor i)

/-- Pointwise absolute error between a real specification tensor and an executable binary32
tensor. The shape index is shared, so no runtime shape cast is hidden in the relation. -/
def TensorErrorLe (eps : Real) :
    {shape : Shape} -> Tensor Real shape -> Tensor IEEE32Exec shape -> Prop
  | .scalar, exact, computed =>
      |IEEE32Exec.toReal computed.item - exact.item| <= eps
  | .dim _ _, exact, computed =>
      ∀ i, TensorErrorLe eps (exact.unstack i) (computed.unstack i)

/-- Width of a finite executable interval, interpreted in the reals. -/
noncomputable def intervalWidth (interval : IEEE32Exec.Interval32) : Real :=
  IEEE32Exec.toReal interval.hi - IEEE32Exec.toReal interval.lo

/-- A valid interval has nonnegative real width. -/
theorem intervalWidth_nonneg {interval : IEEE32Exec.Interval32} (valid : interval.Valid) :
    0 <= intervalWidth interval := by
  have hle := toReal_le_toReal_of_le valid.1 valid.2.1 valid.2.2
  simp only [intervalWidth]
  linarith

/-- Two tensors enclosed by the same interval differ entrywise by at most its width.

This is the elementary bridge from range analysis to approximation analysis. It is deliberately
pointwise; a later norm theorem can package the same statement as an `L∞` bound without changing
the checker or its certificate format. -/
theorem tensor_error_le_width_of_enclosed {interval : IEEE32Exec.Interval32} :
    ∀ {shape : Shape} {exact : Tensor Real shape} {computed : Tensor IEEE32Exec shape},
      TensorEnclosed interval exact ->
      DecodedTensorEnclosed interval computed ->
      TensorErrorLe (intervalWidth interval) exact computed := by
  intro shape
  induction shape with
  | scalar =>
      intro exact computed hexact hcomputed
      simp only [TensorErrorLe, intervalWidth]
      apply (abs_le).2
      constructor <;> linarith [hexact.1, hexact.2, hcomputed.1, hcomputed.2]
  | dim n shape ih =>
      intro exact computed hexact hcomputed i
      exact ih (hexact i) (hcomputed i)

/-- A successful executable range check and a real enclosure proof yield a concrete pointwise
error bound. This theorem is the tensor-level core used by graph-wide numerical certificates. -/
theorem tensor_error_le_width_of_check {interval : IEEE32Exec.Interval32}
    (valid : interval.Valid) {shape : Shape} {exact : Tensor Real shape}
    {computed : Tensor IEEE32Exec shape}
    (hexact : TensorEnclosed interval exact)
    (hcheck : tensorWithinRange interval computed = true) :
    TensorErrorLe (intervalWidth interval) exact computed :=
  tensor_error_le_width_of_enclosed hexact <|
    decodedTensorEnclosed_of_ieee valid <|
      (tensorWithinRange_eq_true_iff interval computed).mp hcheck

/-- Check source ranges once, rejecting malformed intervals and duplicate node ids. -/
def checkSources (sources : Array SourceRange) : Except String (Array CheckedSourceRange) := do
  let mut checked : Array CheckedSourceRange := #[]
  let mut seen : Array Nat := #[]
  for source in sources do
    if seen.contains source.nodeId then
      throw s!"numerical certificate: duplicate source range for node {source.nodeId}"
    if h : validInterval source.enclosure then
      checked := checked.push
        { source with valid := (validInterval_eq_true_iff source.enclosure).mp h }
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

/-- Whether a graph node obtains its enclosure directly from a certificate source assumption. -/
def opUsesSourceRange : OpKind -> Bool
  | .input | .const _ | .randUniform _ | .bernoulliMask _ => true
  | _ => false

/-- Reject source assumptions that do not name a source-like node in the checked graph.

Unused assumptions do not make interval propagation unsound, but they make artifacts ambiguous:
an exporter may have attached a valid range to the wrong node id without noticing. Requiring every
row to be consumed gives source arrays one canonical interpretation and catches that error before
range propagation begins.
-/
def checkSourceOwnership (graph : Graph) (sources : Array CheckedSourceRange) :
    Except String Unit :=
  for source in sources do
    match graph.nodes[source.nodeId]? with
    | none =>
        throw s!"numerical certificate: source range names missing node {source.nodeId}"
    | some node =>
        if opUsesSourceRange node.kind then
          pure ()
        else
          throw (s!"numerical certificate: node {source.nodeId} ({node.kind.describe}) does not " ++
            "consume a source range")

end NumericalCertificate
end RuntimeApprox
end Proofs
