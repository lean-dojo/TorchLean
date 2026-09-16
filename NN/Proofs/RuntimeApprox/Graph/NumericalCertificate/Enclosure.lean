/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Proofs.Analysis.Softmax
public import NN.Floats.Interval.IEEEExec32
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Rounding.Proof
public import NN.Backend.Profile -- shake: keep
public import NN.IR.Semantics -- shake: keep
public import NN.Spec.Core.FloatInstances -- shake: keep
public import NN.Spec.Core.TensorOps -- shake: keep

/-!
# Numerical certificate enclosures

Foundational source-range validation, real and IEEE enclosure semantics, replay checks, and
pointwise error traces for graph numerical certificates. Most users should import
`NN.Proofs.RuntimeApprox.Graph.NumericalCertificate`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)


namespace Proofs
namespace RuntimeApprox
namespace NumericalCertificate

open NN
open NN.Backend
open NN.IR
open Spec TorchLean
open TorchLean.Floats.IEEE754
open FloatLib.Floats.Formats.BinaryInterchange

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
  ExecFloat.Binary.toBits32 a.lo == ExecFloat.Binary.toBits32 b.lo &&
    ExecFloat.Binary.toBits32 a.hi == ExecFloat.Binary.toBits32 b.hi

/-- Executable counterpart of `Interval32.Valid`. -/
def validInterval (interval : IEEE32Exec.Interval32) : Bool :=
  ExecFloat.Binary.isFinite interval.lo &&
    (ExecFloat.Binary.isFinite interval.hi && IEEE32Exec.Interval32.leB interval.lo interval.hi)

/-- `Interval32.leB` decides the proposition-level IEEE non-strict order. -/
theorem leB_eq_true_iff (x y : ExecFloat.Binary 8 23) :
    IEEE32Exec.Interval32.leB x y = true <-> LE.le x y := by
  exact (Model.Interval.le_iff_leB_eq_true
    (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y)).symm.trans
    FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.symm

/-- IEEE comparison between finite values implies the corresponding order on their real
interpretations. This lemma is intentionally finite: IEEE comparisons involving NaN are unordered,
and `toReal` is not the semantic interface for infinities. -/
theorem toReal_le_toReal_of_le {x y : ExecFloat.Binary 8 23}
    (hx : ExecFloat.Binary.isFinite x = true) (hy : ExecFloat.Binary.isFinite y = true)
    (hxy : LE.le x y) : (ExecFloat.Binary.toModel x).toReal <= (ExecFloat.Binary.toModel y).toReal
      := by
  exact (Model.Interval.le_iff_toReal_le_of_isFinite
    (ExecFloat.Binary.toModel x) (ExecFloat.Binary.toModel y) hx hy).mp
    (FloatLib.Floats.ExecFloat.Binary.le_iff_le_toModel.mp hxy)

private theorem toModel_neg (x : ExecFloat.Binary 8 23) :
    ExecFloat.Binary.toModel (Neg.neg x) = Model.neg (ExecFloat.Binary.toModel x) := by
  change ExecFloat.Binary.toModel (ExecFloat.Binary.ofModel (Model.neg (ExecFloat.Binary.toModel
    x))) = _
  exact ExecFloat.Binary.toModel_ofModel _

/-- Negation of a finite executable binary32 value decodes to real negation. -/
theorem toReal_neg_of_isFinite {x : ExecFloat.Binary 8 23} (hx : ExecFloat.Binary.isFinite x = true)
  :
    (ExecFloat.Binary.toModel (Neg.neg x)).toReal = -(ExecFloat.Binary.toModel x).toReal := by
  rw [toModel_neg]
  exact Model.toReal_neg (ExecFloat.Binary.toModel x) hx

/-- Flipping the sign bit preserves finiteness. -/
theorem isFinite_neg_of_isFinite {x : ExecFloat.Binary 8 23} (hx : ExecFloat.Binary.isFinite x =
  true) :
    ExecFloat.Binary.isFinite (Neg.neg x) = true := by
  change Model.isFinite (ExecFloat.Binary.toModel (Neg.neg x)) = true
  rw [toModel_neg, Model.isFinite_neg]
  exact hx

/-- The executable validity test accepts exactly finite, ordered intervals. -/
theorem validInterval_eq_true_iff (interval : IEEE32Exec.Interval32) :
    validInterval interval = true <-> interval.Valid := by
  change (Model.isFinite (ExecFloat.Binary.toModel interval.lo) &&
    (Model.isFinite (ExecFloat.Binary.toModel interval.hi) &&
      Model.Interval.leB (ExecFloat.Binary.toModel interval.lo)
        (ExecFloat.Binary.toModel interval.hi))) = true ↔
    Model.isFinite (ExecFloat.Binary.toModel interval.lo) = true ∧
      Model.isFinite (ExecFloat.Binary.toModel interval.hi) = true ∧
        Model.le (ExecFloat.Binary.toModel interval.lo) (ExecFloat.Binary.toModel interval.hi)
  simp only [Bool.and_eq_true, ← Model.Interval.le_iff_leB_eq_true]

/-! ## Real semantics of the arithmetic transfers

The executable checker propagates binary32 endpoints, while the graph specification is normally
read over real scalars. `RealEncloses` is the bridge between those views. Runtime rounding error is
then composed separately by `FwdGraph.eval_approx` and `RevGraph.backprop_approx`; keeping these two
claims separate prevents an interval enclosure from silently standing in for a floating-point
error theorem.
-/

/-- A real scalar lies between the real interpretations of an executable interval's endpoints. -/
def RealEncloses (interval : IEEE32Exec.Interval32) (value : Real) : Prop :=
  value ∈ Set.Icc ((ExecFloat.Binary.toModel interval.lo).toReal) ((ExecFloat.Binary.toModel
    interval.hi).toReal)

/-- Convert the extended-real endpoint form used by the interval soundness library into an
ordinary real interval when the output endpoints are finite. -/
theorem realEncloses_of_eReal_bounds {interval : IEEE32Exec.Interval32} {value : Real}
    (valid : interval.Valid)
    (bounds : (ExecFloat.Binary.toModel interval.lo).toEReal <= (value : EReal) ∧
      (value : EReal) <= (ExecFloat.Binary.toModel interval.hi).toEReal) :
    RealEncloses interval value := by
  exact (Model.Interval.eRealMem_coe_iff_of_valid valid value).mp bounds

/-- Sound real enclosure for the canonical addition transfer. -/
theorem add_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.add b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.add b) (x + y) := by
  apply realEncloses_of_eReal_bounds hout
  change Model.Interval.ERealMem (a.add b).toModel ((x + y : Real) : EReal)
  rw [IEEE32Exec.Interval32.toModel_add]
  exact Model.Interval.add_sound a.toModel b.toModel (by decide) ha hb hx hy

/-- Sound real enclosure for the canonical subtraction transfer. -/
theorem sub_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.sub b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.sub b) (x - y) := by
  apply realEncloses_of_eReal_bounds hout
  change Model.Interval.ERealMem (a.sub b).toModel ((x - y : Real) : EReal)
  rw [IEEE32Exec.Interval32.toModel_sub]
  exact Model.Interval.sub_sound a.toModel b.toModel (by decide) ha hb hx hy

/-- Sound real enclosure for the canonical multiplication transfer. -/
theorem mul_realEncloses {a b : IEEE32Exec.Interval32} {x y : Real}
    (ha : a.Valid) (hb : b.Valid) (hout : (a.mul b).Valid)
    (hx : RealEncloses a x) (hy : RealEncloses b y) :
    RealEncloses (a.mul b) (x * y) := by
  apply realEncloses_of_eReal_bounds hout
  change Model.Interval.ERealMem (a.mul b).toModel ((x * y : Real) : EReal)
  rw [IEEE32Exec.Interval32.toModel_mul]
  exact Model.Interval.mul_sound a.toModel b.toModel (by decide) ha hb hx hy

/-- Sound real enclosure for the canonical reciprocal transfer. -/
theorem inv_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hout : a.inv.Valid) (hx : RealEncloses a x) :
    RealEncloses a.inv x⁻¹ := by
  apply realEncloses_of_eReal_bounds hout
  change Model.Interval.ERealMem a.inv.toModel ((x⁻¹ : Real) : EReal)
  rw [IEEE32Exec.Interval32.toModel_inv]
  simpa only [one_div] using Model.Interval.inv_sound a.toModel (by decide) ha hx

/-- Every scalar entry of a shape-indexed real tensor lies in one interval. -/
def TensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor Real shape -> Prop
  | .scalar, tensor => RealEncloses interval tensor.item
  | .dim _ _, tensor => ∀ i, TensorEnclosed interval (tensor.unstack i)

/-- The exact executable interval `[0,1]`. -/
def unitInterval : IEEE32Exec.Interval32 :=
  { lo := (ExecFloat.Binary.zero false : ExecFloat.Binary 8 23), hi := (1 : ExecFloat.Binary 8 23) }

/-- The exact executable interval `[-1,1]`. -/
def signedUnitInterval : IEEE32Exec.Interval32 :=
  { lo := (-1 : ExecFloat.Binary 8 23), hi := (1 : ExecFloat.Binary 8 23) }

/-- Executable test for a finite endpoint's nonnegative IEEE sign. Both signed zeros are accepted;
all other accepted values have a clear sign bit. Finiteness is supplied by interval validity. -/
def nonnegativeEndpoint (x : ExecFloat.Binary 8 23) : Bool :=
  ExecFloat.Binary.isZero x || !ExecFloat.Binary.signBit x

private theorem toReal_posZero : (ExecFloat.Binary.toModel (ExecFloat.Binary.zero false :
  ExecFloat.Binary 8 23)).toReal = 0 := by
  have hmodel : ExecFloat.Binary.toModel (ExecFloat.Binary.zero false :
      ExecFloat.Binary 8 23) = Model.zero FloatFormat.binary32 false :=
    ExecFloat.Binary.toModel_ofModel _
  exact (congrArg Model.toReal hmodel).trans (Model.toReal_zero _ _)

private theorem toReal_posOne : (ExecFloat.Binary.toModel (1 : ExecFloat.Binary 8 23)).toReal = 1 :=
  by
  have hmodel : ExecFloat.Binary.toModel (1 : ExecFloat.Binary 8 23) =
      Model.roundRatQ FloatFormat.binary32 1 :=
    ExecFloat.Binary.toModel_ofModel _
  have hone : Model.roundRatQ FloatFormat.binary32 1 = Model.posOne FloatFormat.binary32 := by
    decide
  exact (congrArg Model.toReal (hmodel.trans hone)).trans (Model.toReal_posOne _)

/-- The stable real vector softmax is enclosed by the certificate transfer `[0,1]`. -/
theorem softmaxVec_tensor_enclosed {n : Nat}
    (input : Tensor Real [Nat.succ n]) :
    TensorEnclosed unitInterval (Activation.softmaxVecSpec input) := by
  intro i
  have h := Proofs.softmax_vec_spec_mem_unitInterval input i
  simpa [TensorEnclosed, RealEncloses, unitInterval, Spec.get, TorchLean.Tensor.getScalar,
    toReal_posZero, toReal_posOne] using h

/-- Sound real enclosure for the canonical ReLU interval transfer. -/
theorem relu_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.relu (max x 0) := by
  change Model.Interval.RealMem a.relu.toModel (max x 0)
  rw [IEEE32Exec.Interval32.toModel_relu]
  change Model.toReal (Model.maximum a.toModel.lo (Model.zero _ false)) ≤ max x 0 ∧
    max x 0 ≤ Model.toReal (Model.maximum a.toModel.hi (Model.zero _ false))
  rw [Model.toReal_maximum_eq_max_of_isFinite _ _ ha.1 (by decide),
    Model.toReal_maximum_eq_max_of_isFinite _ _ ha.2.1 (by decide), Model.toReal_zero]
  exact ⟨max_le_max hx.1 le_rfl, max_le_max hx.2 le_rfl⟩

/-- Sound real enclosure for the canonical absolute-value interval transfer. -/
theorem abs_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hx : RealEncloses a x) :
    RealEncloses a.abs |x| := by
  change Model.Interval.RealMem a.abs.toModel |x|
  rw [IEEE32Exec.Interval32.toModel_abs]
  change Model.Interval.RealMem a.toModel x at hx
  by_cases hneg : Model.Interval.leB a.toModel.hi (Model.zero _ true) = true
  · have hhiNonpos : Model.toReal a.toModel.hi ≤ 0 := by
      simpa only [Model.toReal_zero] using
        (Model.Interval.leB_eq_true_iff_toReal_le_of_isFinite
          _ _ ha.2.1 (by decide)).mp hneg
    have hxNonpos : x <= 0 := hx.2.trans hhiNonpos
    rw [Model.Interval.abs, ite_eq_left hneg, abs_of_nonpos hxNonpos]
    change Model.toReal (Model.neg a.toModel.hi) ≤ -x ∧
      -x ≤ Model.toReal (Model.neg a.toModel.lo)
    rw [Model.toReal_neg _ ha.2.1, Model.toReal_neg _ ha.1]
    exact ⟨neg_le_neg hx.2, neg_le_neg hx.1⟩
  · by_cases hpos : Model.Interval.leB (Model.zero _ false) a.toModel.lo = true
    · have hloNonneg : 0 ≤ Model.toReal a.toModel.lo := by
        simpa only [Model.toReal_zero] using
          (Model.Interval.leB_eq_true_iff_toReal_le_of_isFinite
            _ _ (by decide) ha.1).mp hpos
      have hxNonneg : 0 <= x := hloNonneg.trans hx.1
      rw [Model.Interval.abs, ite_eq_right hneg, ite_eq_left hpos, abs_of_nonneg hxNonneg]
      exact hx
    · rw [Model.Interval.abs, ite_eq_right hneg, ite_eq_right hpos]
      change Model.toReal (Model.zero _ false) ≤ |x| ∧
        |x| ≤ Model.toReal (Model.maximum (Model.neg a.toModel.lo) a.toModel.hi)
      rw [Model.toReal_zero, Model.toReal_maximum_eq_max_of_isFinite _ _
        (by simpa only [Model.isFinite_neg] using ha.1) ha.2.1,
        Model.toReal_neg _ ha.1]
      refine ⟨abs_nonneg x, (abs_le).2 ⟨?_, hx.2.trans (le_max_right _ _)⟩⟩
      have := le_max_left (-Model.toReal a.toModel.lo) (Model.toReal a.toModel.hi)
      linarith [hx.1]

private theorem nonnegativeEndpoint_toReal_nonneg {x : ExecFloat.Binary 8 23}
    (hfin : ExecFloat.Binary.isFinite x = true) (hdomain : nonnegativeEndpoint x = true) :
    0 ≤ (ExecFloat.Binary.toModel x).toReal := by
  by_cases hzero : ExecFloat.Binary.isZero x = true
  · exact (Model.toReal_eq_zero_of_isZero (ExecFloat.Binary.toModel x) hzero).ge
  · have hsign : ExecFloat.Binary.signBit x = false := by
      simpa [nonnegativeEndpoint, hzero] using hdomain
    exact Model.toReal_nonneg_of_isFinite_of_signBit_eq_false
      (ExecFloat.Binary.toModel x) hfin hsign

private theorem toModel_sqrtDown (x : ExecFloat.Binary 8 23) :
    ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity)) x) =
      Model.sqrtDown (ExecFloat.Binary.toModel x) := by
  exact FloatLib.Floats.ExecFloat.Binary.toModel_sqrt x .towardNegativeInfinity

private theorem toModel_sqrtUp (x : ExecFloat.Binary 8 23) :
    ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity)) x) =
      Model.sqrtUp (ExecFloat.Binary.toModel x) := by
  exact FloatLib.Floats.ExecFloat.Binary.toModel_sqrt x .towardPositiveInfinity

/-- A directed lower square-root endpoint lies below the exact real square root. FloatLib's
nonnegative-input theorem includes both signed zeros. -/
theorem toReal_sqrtDown_le {x : ExecFloat.Binary 8 23}
    (hfin : ExecFloat.Binary.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : ExecFloat.Binary.isFinite ((ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity))
      x) = true) :
    (ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity))
      x)).toReal <= Real.sqrt ((ExecFloat.Binary.toModel x).toReal) := by
  have h := Model.toEReal_sqrtDown_le_of_nonnegative (ExecFloat.Binary.toModel x) (by decide)
    hfin (nonnegativeEndpoint_toReal_nonneg hfin hdomain)
  rw [← toModel_sqrtDown,
    Model.toEReal_eq_coe_toReal_of_isFinite (ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt
      (rounding := .towardNegativeInfinity)) x)) hout] at h
  exact EReal.coe_le_coe_iff.mp h

/-- Upper counterpart of `toReal_sqrtDown_le`. -/
theorem toReal_sqrtUp_ge {x : ExecFloat.Binary 8 23}
    (hfin : ExecFloat.Binary.isFinite x = true) (hdomain : nonnegativeEndpoint x = true)
    (hout : ExecFloat.Binary.isFinite ((ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity))
      x) = true) :
    Real.sqrt ((ExecFloat.Binary.toModel x).toReal) <= (ExecFloat.Binary.toModel
      ((ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity)) x)).toReal := by
  have h := Model.le_toEReal_sqrtUp_of_nonnegative (ExecFloat.Binary.toModel x) (by decide)
    hfin (nonnegativeEndpoint_toReal_nonneg hfin hdomain)
  rw [← toModel_sqrtUp,
    Model.toEReal_eq_coe_toReal_of_isFinite (ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt
      (rounding := .towardPositiveInfinity)) x)) hout] at h
  exact EReal.coe_le_coe_iff.mp h

/-- Sound real enclosure for directed interval square root. -/
theorem sqrt_realEncloses {a : IEEE32Exec.Interval32} {x : Real}
    (ha : a.Valid) (hlo : nonnegativeEndpoint a.lo = true)
    (hhi : nonnegativeEndpoint a.hi = true) (hout : a.sqrt.Valid)
    (hx : RealEncloses a x) : RealEncloses a.sqrt (Real.sqrt x) := by
  have hlow : a.sqrt.lo = (ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity)) a.lo := by
    apply FloatLib.Floats.ExecFloat.Binary.toModel_inj.mp
    change ExecFloat.Binary.toModel (ExecFloat.Binary.ofModel
      (Model.sqrtDown (ExecFloat.Binary.toModel a.lo))) =
        ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt (rounding := .towardNegativeInfinity))
          a.lo)
    rw [ExecFloat.Binary.toModel_ofModel, toModel_sqrtDown]
  have hhigh : a.sqrt.hi = (ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity)) a.hi := by
    apply FloatLib.Floats.ExecFloat.Binary.toModel_inj.mp
    change ExecFloat.Binary.toModel (ExecFloat.Binary.ofModel
      (Model.sqrtUp (ExecFloat.Binary.toModel a.hi))) =
        ExecFloat.Binary.toModel ((ExecFloat.Binary.sqrt (rounding := .towardPositiveInfinity))
          a.hi)
    rw [ExecFloat.Binary.toModel_ofModel, toModel_sqrtUp]
  constructor
  · change (ExecFloat.Binary.toModel a.sqrt.lo).toReal ≤ Real.sqrt x
    rw [hlow]
    have hlowFinite : ExecFloat.Binary.isFinite a.sqrt.lo = true := hout.1
    exact (toReal_sqrtDown_le ha.1 hlo (by simpa only [hlow] using hlowFinite)).trans
      (Real.sqrt_le_sqrt hx.1)
  · change Real.sqrt x ≤ (ExecFloat.Binary.toModel a.sqrt.hi).toReal
    rw [hhigh]
    have hhighFinite : ExecFloat.Binary.isFinite a.sqrt.hi = true := hout.2.1
    exact (Real.sqrt_le_sqrt hx.2).trans
      (toReal_sqrtUp_ge ha.2.1 hhi (by simpa only [hhigh] using hhighFinite))

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
    {shape : Shape} -> Tensor (ExecFloat.Binary 8 23) shape -> Bool
  | .scalar, tensor =>
      ExecFloat.Binary.isFinite tensor.item &&
        (IEEE32Exec.Interval32.leB interval.lo tensor.item &&
          IEEE32Exec.Interval32.leB tensor.item interval.hi)
  | .dim n _, tensor =>
      (List.finRange n).all (fun i => tensorWithinRange interval (tensor.unstack i))

/-- Proposition expressed by `tensorWithinRange`. -/
def IEEETensorEnclosed (interval : IEEE32Exec.Interval32) :
    {shape : Shape} -> Tensor (ExecFloat.Binary 8 23) shape -> Prop
  | .scalar, tensor =>
      ExecFloat.Binary.isFinite tensor.item = true ∧
        LE.le interval.lo tensor.item ∧ LE.le tensor.item interval.hi
  | .dim _ _, tensor => ∀ i, IEEETensorEnclosed interval (tensor.unstack i)

/-- The executable tensor range check is exact for the IEEE comparison semantics. -/
theorem tensorWithinRange_eq_true_iff (interval : IEEE32Exec.Interval32)
    {shape : Shape} (tensor : Tensor (ExecFloat.Binary 8 23) shape) :
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
    {shape : Shape} -> Tensor (ExecFloat.Binary 8 23) shape -> Prop
  | .scalar, tensor => RealEncloses interval ((ExecFloat.Binary.toModel tensor.item).toReal)
  | .dim _ _, tensor => ∀ i, DecodedTensorEnclosed interval (tensor.unstack i)

/-- A successful IEEE range check decodes to an ordinary real enclosure. Finiteness is an explicit
part of `IEEETensorEnclosed`, so this theorem never assigns a real meaning to NaN or infinity. -/
theorem decodedTensorEnclosed_of_ieee {interval : IEEE32Exec.Interval32}
    (valid : interval.Valid) :
    ∀ {shape : Shape} {tensor : Tensor (ExecFloat.Binary 8 23) shape},
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
    {shape : Shape} -> Tensor Real shape -> Tensor (ExecFloat.Binary 8 23) shape -> Prop
  | .scalar, exact, computed =>
      |(ExecFloat.Binary.toModel computed.item).toReal - exact.item| <= eps
  | .dim _ _, exact, computed =>
      ∀ i, TensorErrorLe eps (exact.unstack i) (computed.unstack i)

/-- Width of a finite executable interval, interpreted in the reals. -/
noncomputable def intervalWidth (interval : IEEE32Exec.Interval32) : Real :=
  (ExecFloat.Binary.toModel interval.hi).toReal - (ExecFloat.Binary.toModel interval.lo).toReal

/-- A valid interval has nonnegative real width. -/
theorem intervalWidth_nonneg {interval : IEEE32Exec.Interval32} (valid : interval.Valid) :
    0 <= intervalWidth interval := by
  change 0 ≤ Model.toReal interval.toModel.hi - Model.toReal interval.toModel.lo
  exact sub_nonneg.mpr (Model.Interval.Valid.toReal_ordered valid)

/-- Two tensors enclosed by the same interval differ entrywise by at most its width.

This is the elementary bridge from range analysis to approximation analysis. It is deliberately
pointwise; a later norm theorem can package the same statement as an `L∞` bound without changing
the checker or its certificate format. -/
theorem tensor_error_le_width_of_enclosed {interval : IEEE32Exec.Interval32} :
    ∀ {shape : Shape} {exact : Tensor Real shape} {computed : Tensor (ExecFloat.Binary 8 23) shape},
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
    {computed : Tensor (ExecFloat.Binary 8 23) shape}
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
