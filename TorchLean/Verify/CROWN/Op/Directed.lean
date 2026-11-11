/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.DirectedNonlinear
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Op.Table.Linear
public import TorchLean.Tensor.Op.Table.Loss
public import TorchLean.Tensor.Op.Table.Movement
public import TorchLean.Tensor.Op.Table.Normalization
public import TorchLean.Tensor.Op.Table.Random
public import TorchLean.Tensor.Op.Table.Reduce
import TorchLean.Tensor.Op.Pointwise.Real
import TorchLean.NN.Random.Range
import TorchLean.Verify.CROWN.FlatBox.Real
import TorchLean.NN.Gradients.Activation
import TorchLean.Tensor.ShapeChange.Flatten

/-!
# Soundness of the rounded IBP pass on operator graphs

`Op.runIBP` computes an optional box per node of a graph of registered operators at any scalar with
endpoint operations. At a scalar whose endpoint operations are lawful (`LawfulBoundOps`,
`LawfulNonlinearBoundOps`), `ibp?` is a sound rule (`ibp?_encloses`, `RuleEncloses`): every box
it returns encloses the flat value of the node's first output, when the references' boxes enclose
the flat values they read and each constant the operator reads holds the real value of its rounded
constant (`ConstsReal`). It boxes only operators of `Supported`, whose kind in the table
`ruleKind` is not `none` (`supported_of_ibp?`). Under any sound rule, every box of a table of
`step?` fixpoints encloses the flat value of its node in a successful real run whose payload holds
the reals of the rounded constants (`RealConsts`) and whose graph inputs the seed boxes enclose
(`step?_encloses` at one node, `encloses_of_step` on the table), and `encloses_of_run` is this for
the boxes of `Op.runIBP`.

Both laws hold at `ℝ` and at `FP32`, the binary32 model with an unbounded exponent
(`TorchLean.Verify.CROWN.BoundOps.FP32`). The IBP node certificate checker
`Certificate.IBPNode.check` runs the pass at the executable `ExecFloat.Binary 8 23`, which holds
infinities and NaN and gets no global lawful instance, so its acceptance has no theorem over `ℝ`.
Native `Float` and `Float32` get no instance either.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape (size size_append size_cons size_nil)

namespace TorchLean.CROWN.Op.Directed

open Tensor
open Graph
open CROWN.Directed MathFunctions

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

noncomputable section

variable [NonlinearBoundOps α] [LawfulNonlinearBoundOps α]

/-- The operators whose kind in `ruleKind` is not `none`: the operators whose rule in `ibp?` is
proved sound here (`CROWN.Op.ibp?_encloses`), which are all the operators `ibp?` boxes
(`supported_of_ibp?`).

The operators that copy, move or select entries are detach, reshape, permute, slice, flip,
broadcast, indexAt, cat, pad, select, gather and embedding. The arithmetic maps are add, sub, mul,
neg, square, abs, max, min, fmax, fmin, atan2, hypot, fmod, remainder, copysign and div with
rounding `none` or `trunc`. The other elementwise maps are relu, gelu, sign, floor, ceil, round,
trunc, frac, sinh, cosh, asinh, tan, atan, asin, acos, erf, expm1, exp2, log1p, log2, log10, rsqrt,
softplus and safeLog. The maps scale, leakyRelu, elu and clamp take scalar parameters, and exp,
log, sqrt, reciprocal, tanh, sigmoid, silu, sin and cos use the backend's enclosures. The layers
are linear, matmul, scaledMatmul, contract, conv, layerNorm and batchNormRunning. Along axes there
are cumsum, softmax, logSoftmax, maskedSoftmax, pool with any `PoolOp` and mean over one axis
without `keepdim`. A reduce qualifies by max or min over any valid axes with either `keepdim`, and
by sum without `keepdim` over every axis in order or over one axis. The others are mseLoss and the
samples randUniform and bernoulliMask. An operator with an element parameter (`Op.Element`,
`Op.NumericElement`) qualifies only at the element `.scalar`. -/
def Supported (o : IR.Operator) : Prop :=
  ruleKind o ≠ .none

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Where `directed?` has no rule, a box of `ibp?` is the box of `shared?`. -/
theorem _root_.TorchLean.CROWN.Op.shared?_of_ibp? [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {boxes : Array (Option (FlatBox α))} {o : IR.Operator}
    {refs : List IR.Ref} {B : FlatBox α} (h : ibp? G P boxes o refs = some B)
    (hd : directed? G P boxes o refs = none) : shared? G P boxes o refs = some B := by
  unfold ibp? at h
  cases hs : shared? G P boxes o refs <;> simp_all

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Where `shared?` has no rule, a box of `ibp?` is the box of `directed?`. -/
theorem _root_.TorchLean.CROWN.Op.directed?_of_ibp? [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {boxes : Array (Option (FlatBox α))} {o : IR.Operator}
    {refs : List IR.Ref} {B : FlatBox α} (h : ibp? G P boxes o refs = some B)
    (hs : shared? G P boxes o refs = none) : directed? G P boxes o refs = some B := by
  unfold ibp? at h
  rw [hs] at h
  exact h

omit [LawfulBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Every operator to which `ibp?` gives a box is a `Supported` operator: the other operators
have no rule in `shared?` or `directed?`, a sum or mean that `ibpReduce?` or `ibpMean?` accepts
runs over every axis in order or over one axis, without `keepdim`, and a division it boxes rounds
by `none` or `trunc`. -/
theorem supported_of_ibp? [Op.Capabilities α] {G : IR.Graph}
    {P : IR.Payload α} {boxes : Array (Option (FlatBox α))} {o : IR.Operator}
    {refs : List IR.Ref} {B : FlatBox α} (h : ibp? G P boxes o refs = some B) :
    Supported o := by
  obtain ⟨ins, outs, op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›) <;>
    simp only [Supported, ruleKind.eq_def, ne_eq, ite_eq_right_iff, reduceCtorEq, imp_false,
      not_not, not_true_eq_false, not_false_eq_true] <;> try trivial
  case div.integer =>
    cases ‹DivRounding› <;> exact (Option.some_ne_none B (h.symm.trans rfl)).elim
  case «reduce» s op dims keepdim _ =>
    have hd := directed?_of_ibp? h rfl
    rcases refs with _ | ⟨r, _ | ⟨r', rs⟩⟩
    · cases op <;> exact (Option.some_ne_none B (hd.symm.trans rfl)).elim
    · cases op
      case max => exact Or.inr (Or.inl rfl)
      case min => exact Or.inr (Or.inr rfl)
      case prod =>
        obtain ⟨X, -, hX⟩ := Option.bind_eq_some_iff.mp hd
        simp [ibpReduce?] at hX
      case sum =>
        obtain ⟨X, -, hX⟩ := Option.bind_eq_some_iff.mp hd
        unfold ibpReduce? at hX
        by_cases hk : keepdim = false
        · subst hk
          by_cases hall : (dims.map Fin.val).toArray = Array.range s.rank
          · exact Or.inl ⟨rfl, rfl, Or.inl (by simpa using congrArg Array.toList hall)⟩
          · by_cases hone : (dims.map Fin.val).toArray.size = 1
            · exact Or.inl ⟨rfl, rfl, Or.inr (by simpa using hone)⟩
            · rw [ite_eq_left ⟨rfl, rfl⟩, ite_eq_right hall, dite_eq_right hone] at hX
              cases hX
        · rw [ite_eq_right fun h => hk h.2] at hX
          cases hX
    · cases op <;> exact (Option.some_ne_none B (hd.symm.trans rfl)).elim
  case mean s dims keepdim =>
    have hd := directed?_of_ibp? h rfl
    rcases refs with _ | ⟨r, _ | ⟨r', rs⟩⟩
    · exact (Option.some_ne_none B (hd.symm.trans rfl)).elim
    · obtain ⟨X, -, hX⟩ := Option.bind_eq_some_iff.mp hd
      unfold ibpMean? at hX
      by_cases hone : (dims.map Fin.val).toArray.size = 1 ∧ keepdim = false
      · obtain ⟨hsize, rfl⟩ := hone
        exact ⟨rfl, by simpa using hsize⟩
      · rw [dite_eq_right hone] at hX
        cases hX
    · exact (Option.some_ne_none B (hd.symm.trans rfl)).elim
  case div s rounding =>
    have hd := directed?_of_ibp? h rfl
    cases rounding
    · exact Or.inl rfl
    · exact Or.inr rfl
    · rcases refs with _ | ⟨r₁, _ | ⟨r₂, _ | ⟨r₃, rs⟩⟩⟩ <;>
        exact (Option.some_ne_none B (hd.symm.trans rfl)).elim

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- A rule for operator nodes is sound: at every operator node whose references' boxes enclose the
flat values they read and whose references to rounded constants read their real values
(`ConstsReal`), every box the rule returns encloses the flat value of the node's first output. -/
def RuleEncloses (G : IR.Graph) (P : IR.Payload α) (Pr : IR.Payload ℝ)
    (rule : Array (Option (FlatBox α)) → IR.Operator → List IR.Ref → Option (FlatBox α)) :
    Prop :=
  ∀ (boxes : Array (Option (FlatBox α))) (inputs : List (Op.Value ℝ))
    (earlier : Array (List (Op.Value ℝ))) (id : Nat) (o : IR.Operator)
    (refs : Array IR.Ref) (B : FlatBox α) (outs : List (Op.Value ℝ)),
    ConstsReal G P earlier refs.toList →
    (∀ r ∈ refs.toList, ∀ (Br : FlatBox α) (w : Op.Value ℝ) (v : FlatTensor ℝ),
      box? boxes r = some Br → r.read earlier = .ok w → flatOf w = some v → Encloses Br v) →
    rule boxes o refs.toList = some B →
    (IR.Node.op o refs).eval Pr inputs earlier id = .ok outs →
    ∀ v, (outs[0]? >>= flatOf) = some v → Encloses B v

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
/-- At `ℝ`, the integer toward zero lies within 1 of its argument. -/
private theorem truncRound_mem (x : ℝ) :
    x - 1 ≤ (FloatLib.Floats.Formats.Flocq.truncRound x : ℝ) ∧
      (FloatLib.Floats.Formats.Flocq.truncRound x : ℝ) ≤ x + 1 := by
  unfold FloatLib.Floats.Formats.Flocq.truncRound
  split_ifs with h
  · exact ⟨(by linarith : x - 1 ≤ x).trans (Int.le_ceil x), (Int.ceil_lt_add_one x).le⟩
  · exact ⟨(Int.sub_one_lt_floor x).le, (Int.floor_le x).trans (by linarith)⟩

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem floor_mem (x : ℝ) : x - 1 ≤ Rounding.floor x ∧ Rounding.floor x ≤ x + 1 := by
  rw [Context.floor_real]
  exact ⟨(Int.sub_one_lt_floor x).le, (Int.floor_le x).trans (by linarith)⟩

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem ceil_mem (x : ℝ) : x - 1 ≤ Rounding.ceil x ∧ Rounding.ceil x ≤ x + 1 := by
  rw [Context.ceil_real]
  exact ⟨(by linarith : x - 1 ≤ x).trans (Int.le_ceil x), (Int.ceil_lt_add_one x).le⟩

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem round_mem (x : ℝ) : x - 1 ≤ Rounding.round x ∧ Rounding.round x ≤ x + 1 := by
  rw [Context.round_real]
  have h1 := FloatLib.Floats.Formats.Flocq.nearestEven_is_nearest_integer x ⌊x⌋
  have h2 : |(⌊x⌋ : ℝ) - x| ≤ 1 := by
    rw [abs_le]; constructor <;> linarith [Int.floor_le x, Int.sub_one_lt_floor x]
  have h3 := abs_le.mp (h1.trans h2)
  exact ⟨by linarith [h3.1], by linarith [h3.2]⟩

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem trunc_mem (x : ℝ) : x - 1 ≤ Rounding.trunc x ∧ Rounding.trunc x ≤ x + 1 := by
  rw [Context.trunc_real]
  exact truncRound_mem x

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem frac_mem (x : ℝ) :
    -((1 : ℕ) : ℝ) ≤ Context.frac x ∧ Context.frac x ≤ ((1 : ℕ) : ℝ) := by
  rw [Context.frac_real]
  have := truncRound_mem x
  push_cast; constructor <;> linarith [this.1, this.2]

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem arctan_mem (x : ℝ) :
    -((2 : ℕ) : ℝ) ≤ Real.arctan x ∧ Real.arctan x ≤ ((2 : ℕ) : ℝ) := by
  have := Real.neg_pi_div_two_lt_arctan x; have := Real.arctan_lt_pi_div_two x
  have := Real.pi_le_four; push_cast; constructor <;> linarith

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem arcsin_mem (x : ℝ) :
    -((2 : ℕ) : ℝ) ≤ Real.arcsin x ∧ Real.arcsin x ≤ ((2 : ℕ) : ℝ) := by
  have := Real.neg_pi_div_two_le_arcsin x; have := Real.arcsin_le_pi_div_two x
  have := Real.pi_le_four; push_cast; constructor <;> linarith

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem arccos_mem (x : ℝ) :
    -((0 : ℕ) : ℝ) ≤ Real.arccos x ∧ Real.arccos x ≤ ((4 : ℕ) : ℝ) := by
  have := Real.arccos_nonneg x; have := Real.arccos_le_pi x
  have := Real.pi_le_four; push_cast; constructor <;> linarith

omit [Storage α] [Context α] [BoundOps α] [NonlinearBoundOps α] [LawfulBoundOps α]
  [LawfulNonlinearBoundOps α] in
private theorem erf_mem (x : ℝ) : -((1 : ℕ) : ℝ) ≤ erf x ∧ erf x ≤ ((1 : ℕ) : ℝ) := by
  have := neg_one_lt_erf x; have := erf_lt_one x
  push_cast; constructor <;> linarith

/-- At a lawful scalar `ibp?` is a sound rule (`RuleEncloses`). At an operator node, let each
reference's box enclose the flat value the reference reads, and each reference to a rounded
constant read that constant's real value (`ConstsReal`). If the node evaluates on the real values
its references read, every box `ibp?` returns encloses the flat value of the node's first output. -/
theorem _root_.TorchLean.CROWN.Op.ibp?_encloses {G : IR.Graph} {P : IR.Payload α}
    {Pr : IR.Payload ℝ} :
    RuleEncloses G P Pr (ibp? G P) := by
  intro boxes inputs earlier id o refs B outs hconsts hpe hc h
  have hrule := supported_of_ibp? hc
  obtain ⟨ins, outs', op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›) <;>
    simp only [Supported, ruleKind.eq_def, ne_eq, ite_eq_right_iff, reduceCtorEq, imp_false,
      not_not, not_true_eq_false, not_false_eq_true] at hrule
  case tanh s =>
    exact encloses_unary Op.rule?_tanh (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.tanhBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case sigmoid s =>
    exact encloses_unary Op.rule?_sigmoid (fun _ => rfl)
      (F := Real.sigmoid) (fun z => by rw [Context.sigmoid_eq_inv_exp, Real.sigmoid_def])
      (fun hx hb =>
        boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.sigmoidBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case sin s =>
    exact encloses_unary Op.rule?_sin (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.sinBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case cos s =>
    exact encloses_unary Op.rule?_cos (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.cosBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case exp s =>
    exact encloses_unary Op.rule?_exp (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.expBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case sqrt s policy =>
    cases policy
    · exact encloses_unary Op.rule?_sqrt (F := Real.sqrt) (fun _ => rfl) (fun _ => rfl)
        (fun hx hb =>
          boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.sqrtBounds_enclosure hx hb)
        hpe (fun r hr => by rw [hr] at hc; exact hc) h
    · exact encloses_unary Op.rule?_sqrt_clampNegative (fun _ => rfl) sqrt_max_zero
        (fun hx hb =>
          boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.sqrtBounds_enclosure hx hb)
        hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case reciprocal s =>
    exact encloses_unary Op.rule?_reciprocal (fun _ => rfl) (fun z => one_div z)
      (fun hx hb => boxUnaryEnclosure?_encloses unaryEnclosure_inv hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case silu s =>
    exact encloses_unary Op.rule?_silu (fun _ => rfl)
      (F := fun x => x * Real.sigmoid x)
      (fun z => by
        rw [Activation.silu_eq_mul_sigmoid, Context.sigmoid_eq_inv_exp, Real.sigmoid_def])
      (fun hx hb => ibpSiLU?_encloses hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact shared?_of_ibp? hc rfl) h
  case relu s =>
    exact encloses_unary Op.rule?_relu (fun _ => rfl)
      (fun z => Activation.relu_zero_branch_eq_max z)
      (T := fun B => some (boxReLU B))
      (fun hx hb => by obtain rfl := Option.some.inj hb; exact boxReLU_encloses hx)
      hpe (fun r hr => by
        rw [hr] at hc
        have hc' : (box? boxes r).map boxReLU = some B := shared?_of_ibp? hc rfl
        cases hb : box? boxes r <;> simp_all) h
  case neg s =>
    exact encloses_unary Op.rule?_neg (fun _ => rfl) (F := fun x => -x) (fun _ => rfl)
      (T := fun B => some (boxNeg B))
      (fun hx hb => by obtain rfl := Option.some.inj hb; exact boxNeg_encloses hx)
      hpe (fun r hr => by
        rw [hr] at hc
        have hc' : (box? boxes r).map boxNeg = some B := hc
        cases hb : box? boxes r <;> simp_all) h
  case square s =>
    exact encloses_unary Op.rule?_square (fun _ => rfl) (F := fun x => x ^ 2)
      (fun z => (pow_two z).symm) (T := fun B => some (boxSquare B))
      (fun hx hb => by obtain rfl := Option.some.inj hb; exact boxSquare_encloses hx)
      hpe (fun r hr => by
        rw [hr] at hc
        have hc' : (box? boxes r).map boxSquare = some B := hc
        cases hb : box? boxes r <;> simp_all) h
  case scale s =>
    exact encloses_param Op.rule?_scale (f := fun a z => z * a)
      (T := fun c => boxUnaryEnclosure? (scaleBounds c)) (fun _ _ => rfl) (fun _ _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (scaleBounds_enclosure _) hx hb) hconsts hpe
      (fun rx rc hr => by rw [hr] at hc; exact hc) h
  case leakyReLU s =>
    exact encloses_param Op.rule?_leakyReLU
      (f := fun a z => Activation.leakyReLU z a)
      (F := fun a z => if 0 < z then z else a * z)
      (T := fun c => boxUnaryEnclosure? (leakyReLUBounds c)) (fun _ _ => rfl)
      (fun a z => by unfold Activation.leakyReLU; split_ifs <;> rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (leakyReLUBounds_enclosure _) hx hb) hconsts hpe
      (fun rx rc hr => by rw [hr] at hc; exact hc) h
  case clamp s =>
    obtain ⟨rx, ra, rb, wx, wa, wb, x, a, b, hr, hreadx, hreada, hreadb, hx, ha, hb, rfl⟩ :=
      invoke_three Op.rule?_clamp h
    rw [hr] at hc hpe hconsts
    have hc' : (do
        let a ← constAt? G P ra (.frozen [])
        let b ← constAt? G P rb (.frozen [])
        box? boxes rx >>= boxUnaryEnclosure?
          (clampBounds (Tensor.item a) (Tensor.item b))) = some B := hc
    simp only [Option.bind_eq_bind] at hc'
    obtain ⟨a', ha', hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨b', hb', hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨Bx, hBx, hT⟩ := Option.bind_eq_some_iff.mp hc'
    obtain rfl := read_constAt?_carrier hconsts (by simp) ha' hreada ha
    obtain rfl := read_constAt?_carrier hconsts (by simp) hb' hreadb hb
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    simp only [realCarrier, item_map, clamp_full_real]
    exact encloses_map (fun _ => rfl) (hpe rx (by simp) Bx wx _ hBx hreadx
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      fun hg => boxUnaryEnclosure?_encloses (clampBounds_enclosure _ _) hg hT
  case elu s =>
    exact encloses_param Op.rule?_elu
      (f := fun a z => Activation.elu z a)
      (F := fun a z => if 0 < z then z else a * (Real.exp z - 1))
      (T := fun c => boxUnaryEnclosure? (eluBounds c)) (fun _ _ => rfl)
      (fun a z => by
        unfold Activation.elu
        rw [show Context.expm1 z = Real.exp z - 1 from Context.expm1_eq_exp_sub_one z])
      (fun hx hb => boxUnaryEnclosure?_encloses (eluBounds_enclosure _) hx hb) hconsts hpe
      (fun rx rc hr => by rw [hr] at hc; exact hc) h
  case gelu s approximate =>
    cases approximate
    · exact encloses_unary (Op.rule?_gelu_none (α := ℝ) (s := s))
        (F := fun z => z * ((1 / 2 : ℚ) : ℝ) *
          (1 + Erf.erf (z * MathFunctions.sqrt ((1 / 2 : ℚ) : ℝ))))
        (fun _ => rfl) (fun _ => by
          simp only [Context.geluErf, Context.isInf_real, Bool.false_eq_true, ↓reduceIte]; rfl)
        (fun hx hb => boxUnaryEnclosure?_encloses (zeroHullBounds_enclosure gelu_mem_real) hx hb)
        hpe (fun r hr => by rw [hr] at hc; exact hc) h
    · exact encloses_unary Op.rule?_gelu (fun _ => rfl) (fun _ => rfl)
        (fun hx hb => boxUnaryEnclosure?_encloses
          (zeroHullBounds_enclosure Activation.geluTanh_mem_real)
          hx hb)
        hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case sinh s =>
    exact encloses_unary Op.rule?_sinh (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses sinhBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case cosh s =>
    exact encloses_unary Op.rule?_cosh (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses coshBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case logSoftmax s dim =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_logSoftmax dim) h
    rw [hr] at hc hpe
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    have hc' : (box? boxes r >>= coordwiseBox? (s := s) (t := s)
        (logSoftmaxEntry dim)) = some B := hc
    obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
    exact coordwiseBox?_encloses (hpe r (by simp) Bx w _ hBx hread
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      (fun lo hi hlh c b hb => logSoftmaxEntry_encloses dim hlh c hb) hB
  case cumsum s dim =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_cumsum dim) h
    rw [hr] at hc hpe
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    have hc' : (box? boxes r >>= coordwiseBox? (s := s) (t := s)
        (cumsumBounds dim)) = some B := hc
    obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
    exact coordwiseBox?_encloses (hpe r (by simp) Bx w _ hBx hread
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      (fun lo hi hlh c b hb => cumsumBounds_encloses dim hlh c hb) hB
  case select s =>
    obtain ⟨rc, rx, ry, wc, wx, wy, c, x, y, hr, -, hreadx, hready, -, hx, hy, rfl⟩ :=
      invoke_three Op.rule?_select h
    rw [hr] at hc hpe
    have hc' : (do
        let Bx ← box? boxes rx
        let By ← box? boxes ry
        selectBox? Bx By) = some B := hc
    obtain ⟨Bx, hBx, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨By, hBy, hB⟩ := Option.bind_eq_some_iff.mp hc'
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    exact selectBox?_encloses c
      (hpe rx (by simp) Bx wx _ hBx hreadx
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      (hpe ry (by simp) By wy _ hBy hready
        ((flatOf_of_read (Or.inl rfl) hy).trans (flatOf_value y))) hB
  case pad s low high interior =>
    obtain ⟨rx, rv, wx, wv, x, v, hr, hreadx, hreadv, hx, hv, rfl⟩ :=
      invoke_two (Op.rule?_pad low high interior) h
    rw [hr] at hc hpe
    have hc' : (do
        let Bx ← box? boxes rx
        let Bv ← box? boxes rv
        padBox? (s := s) low high interior Bx Bv) = some B := hc
    obtain ⟨Bx, hBx, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨Bv, hBv, hB⟩ := Option.bind_eq_some_iff.mp hc'
    intro w hw
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hw
    subst hw
    exact padBox?_encloses low high interior
      (hpe rx (by simp) Bx wx _ hBx hreadx
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      (hpe rv (by simp) Bv wv _ hBv hreadv
        ((flatOf_of_read (Or.inl rfl) hv).trans (flatOf_value v))) hB
  case gather s t dims batchDims hg =>
    obtain ⟨rx, ri, wx, wi, x, idx, hr, hreadx, -, hx, -, rfl⟩ :=
      invoke_two (Op.rule?_gather dims batchDims hg) h
    rw [hr] at hc hpe
    have hc' : (box? boxes rx >>= gatherBox? hg) = some B := hc
    obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    exact gatherBox?_encloses hg _ _ _
      (hpe rx (by simp) Bx wx _ hBx hreadx
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) hB
  case embedding num dim t =>
    obtain ⟨rx, ri, wx, wi, x, idx, hr, hreadx, -, hx, -, rfl⟩ :=
      invoke_two Op.rule?_embedding h
    rw [hr] at hc hpe
    have hc' : (box? boxes rx >>= gatherBox? (s := [num, dim]) (t := t)
        (dims := Shape.selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩)
        (batchDims := Shape.noBatch [num, dim] t)
        ⟨List.nodup_singleton _, List.nodup_nil, fun _ h => nomatch h⟩) = some B := hc
    obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    exact gatherBox?_encloses _ _ _ rfl
      (hpe rx (by simp) Bx wx _ hBx hreadx
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) hB
  case sign s =>
    exact encloses_unary Op.rule?_sign (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses signBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case floor s =>
    exact encloses_unary (Op.rule?_floor (α := ℝ) (s := s))
      (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (withinOneBounds_enclosure floor_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case ceil s =>
    exact encloses_unary (Op.rule?_ceil (α := ℝ) (s := s))
      (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (withinOneBounds_enclosure ceil_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case round s =>
    exact encloses_unary (Op.rule?_round (α := ℝ) (s := s))
      (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (withinOneBounds_enclosure round_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case trunc s =>
    exact encloses_unary (Op.rule?_trunc (α := ℝ) (s := s))
      (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (withinOneBounds_enclosure trunc_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case frac s =>
    exact encloses_unary (Op.rule?_frac (α := ℝ) (s := s))
      (fun _ => rfl)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (rangeBounds_enclosure frac_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case atan s =>
    exact encloses_unary (Op.rule?_atan (α := ℝ) (s := s))
      (fun _ => rfl) (F := Real.arctan)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (rangeBounds_enclosure arctan_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case asin s =>
    exact encloses_unary (Op.rule?_asin (α := ℝ) (s := s))
      (fun _ => rfl) (F := Real.arcsin)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (unitRangeBounds_enclosure arcsin_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case acos s =>
    exact encloses_unary (Op.rule?_acos (α := ℝ) (s := s))
      (fun _ => rfl) (F := Real.arccos)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (unitRangeBounds_enclosure arccos_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case erf s =>
    exact encloses_unary (Op.rule?_erf (α := ℝ) (s := s))
      (fun _ => rfl) (F := TorchLean.erf)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses (rangeBounds_enclosure erf_mem) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case expm1 s =>
    exact encloses_unary Op.rule?_expm1 (fun _ => rfl) (F := fun z => Real.exp z - 1)
      Context.expm1_eq_exp_sub_one
      (fun hx hb => boxUnaryEnclosure?_encloses expm1Bounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case exp2 s =>
    exact encloses_unary Op.rule?_exp2 (fun _ => rfl) (F := fun z => (2 : ℝ) ^ z)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses exp2Bounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case asinh s =>
    exact encloses_unary Op.rule?_asinh (fun _ => rfl) (F := Real.arsinh)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses arsinhBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case tan s =>
    exact encloses_unary Op.rule?_tan (fun _ => rfl) (F := Real.tan)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses tanBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case log1p s =>
    exact encloses_unary Op.rule?_log1p (fun _ => rfl) (F := fun z => Real.log (1 + z))
      Context.log1p_eq_log_one_add
      (fun hx hb => boxUnaryEnclosure?_encloses log1pBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case log2 s =>
    exact encloses_unary Op.rule?_log2 (fun _ => rfl) (F := Real.logb ((2 : ℕ) : ℝ))
      (fun z => by rw [Context.log2_real, Nat.cast_ofNat])
      (fun hx hb => boxUnaryEnclosure?_encloses (logbBounds_enclosure 2) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case log10 s =>
    exact encloses_unary Op.rule?_log10 (fun _ => rfl) (F := Real.logb ((10 : ℕ) : ℝ))
      (fun z => by rw [Context.log10_real, Nat.cast_ofNat])
      (fun hx hb => boxUnaryEnclosure?_encloses (logbBounds_enclosure 10) hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case rsqrt s =>
    exact encloses_unary Op.rule?_rsqrt (fun _ => rfl) (F := fun z => 1 / Real.sqrt z)
      (fun _ => rfl)
      (fun hx hb => boxUnaryEnclosure?_encloses rsqrtBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case fmax s =>
    exact encloses_binary Op.rule?_fmax (f := Context.fmax) (fun _ _ => rfl)
      (F := fun a b => max a b) (fun a b => Context.fmax_real a b)
      (T := fun B₁ B₂ => if B₁.dim = B₂.dim then some (boxMaxElem B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxMaxElem_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case fmin s =>
    exact encloses_binary Op.rule?_fmin (f := Context.fmin) (fun _ _ => rfl)
      (F := fun a b => min a b) (fun a b => Context.fmin_real a b)
      (T := fun B₁ B₂ => if B₁.dim = B₂.dim then some (boxMinElem B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxMinElem_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case div s rounding =>
    rcases hrule with rfl | rfl
    · exact encloses_binary (Op.rule?_div (α := ℝ) (s := s)) (fun _ _ => rfl)
        (fun _ _ => rfl) (T := binaryBox? NonlinearBoundOps.divBounds)
        (fun hx hy hb => binaryBox?_encloses LawfulNonlinearBoundOps.divBounds_enclosure hx hy hb)
        hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
    · obtain ⟨r₁, r₂, w₁, w₂, x₁, x₂, hr, hread₁, hread₂, hx₁, hx₂, rfl⟩ :=
        invoke_two (Op.rule?_div_trunc (α := ℝ) (s := s)) h
      rw [hr] at hc hpe
      have hc' : (do boxUnaryEnclosure? withinOneBounds (← binaryBox?
          NonlinearBoundOps.divBounds (← box? boxes r₁) (← box? boxes r₂))) = some B := hc
      obtain ⟨B₁, hB₁, hc'⟩ := Option.bind_eq_some_iff.mp hc'
      obtain ⟨B₂, hB₂, hc'⟩ := Option.bind_eq_some_iff.mp hc'
      obtain ⟨Q, hQ, hB⟩ := Option.bind_eq_some_iff.mp hc'
      intro v hv
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      have hq : Encloses Q (Graph.flatOfSome ⟨s, Tensor.zipWith (· / ·) x₁ x₂⟩) :=
        encloses_zipWith (fun _ _ => rfl)
          (hpe r₁ (by simp) B₁ w₁ _ hB₁ hread₁
            ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁)))
          (hpe r₂ (by simp) B₂ w₂ _ hB₂ hread₂
            ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂)))
          fun hg hh => binaryBox?_encloses LawfulNonlinearBoundOps.divBounds_enclosure hg hh hQ
      -- Over the reals the forward is the truncated quotient, which the box of the truncation
      -- encloses.
      refine (congrArg (fun t : Tensor ℝ s => Encloses B (Graph.flatOfSome ⟨s, t⟩))
        (Tensor.copysign_floorDivide_abs_real x₁ x₂)).mpr ?_
      exact encloses_map (f := Rounding.trunc) (fun _ => rfl) hq
        fun hg => boxUnaryEnclosure?_encloses (withinOneBounds_enclosure trunc_mem) hg hB
  case atan2 s =>
    exact encloses_binary (Op.rule?_atan2 (α := ℝ) (s := s)) (fun _ _ => rfl)
      (fun _ _ => rfl)
      (T := binaryBox? (fun aLo aHi _ _ => rangeBounds 4 4 aLo aHi))
      (fun hx hy hb => binaryBox?_encloses atan2Range_enclosure hx hy hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case hypot s =>
    exact encloses_binary Op.rule?_hypot (fun _ _ => rfl) (fun _ _ => rfl)
      (T := binaryBox? hypotBounds)
      (fun hx hy hb => binaryBox?_encloses hypotBounds_enclosure hx hy hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case contract s t dims batchDims hct =>
    obtain ⟨rx, ry, wx, wy, x, y, hr, hreadx, hready, hx, hy, rfl⟩ :=
      invoke_two (Op.rule?_contract dims batchDims hct) h
    rw [hr] at hc hpe
    obtain ⟨Bx, By, hBx, hBy, hB⟩ := bind_box_pair (T := contractBox? hct) hc
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    exact contractBox?_encloses hct
      (hpe rx (by simp) Bx wx _ hBx hreadx
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      (hpe ry (by simp) By wy _ hBy hready
        ((flatOf_of_read (Or.inl rfl) hy).trans (flatOf_value y))) hB
  case fmod s =>
    exact encloses_binary (Op.rule?_fmod (α := ℝ) (s := s)) (fun _ _ => rfl)
      (fun _ _ => rfl)
      (T := binaryBox? (nonzeroDivisor? fun aLo aHi _ _ => zeroHullBounds aLo aHi))
      (fun hx hy hb =>
        binaryBox?_encloses (nonzeroDivisor?_enclosure fmodBounds_enclosure) hx hy hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case remainder s =>
    exact encloses_binary (Op.rule?_remainder (α := ℝ) (s := s)) (fun _ _ => rfl)
      (fun _ _ => rfl)
      (T := binaryBox? (nonzeroDivisor? fun aLo aHi bLo bHi => symBounds
        (Context.maximum (Graph.absBound aLo aHi) (Graph.absBound bLo bHi))))
      (fun hx hy hb =>
        binaryBox?_encloses (nonzeroDivisor?_enclosure remainderBounds_enclosure) hx hy hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case copysign s =>
    exact encloses_binary Op.rule?_copysign (fun _ _ => rfl) (fun _ _ => rfl)
      (T := binaryBox? (fun aLo aHi _ _ => symBounds (Graph.absBound aLo aHi)))
      (fun hx hy hb => binaryBox?_encloses copysignBounds_enclosure hx hy hb)
      hpe (fun r₁ r₂ hr => bind_box_pair (by rw [hr] at hc; exact hc)) h
  case log s =>
    exact encloses_unary Op.rule?_log (fun _ => rfl) (fun _ => rfl)
      (T := fun B => if (List.finRange B.dim).all fun i => decide (0 < B.lo.getScalar i) then
        boxUnaryEnclosure? NonlinearBoundOps.logBounds B else none)
      (fun hx hb => by
        split at hb
        · exact boxUnaryEnclosure?_encloses LawfulNonlinearBoundOps.logBounds_enclosure hx hb
        · cases hb)
      hpe (fun r hr => by rw [hr] at hc; exact hc) h
  case add s =>
    exact encloses_binary Op.rule?_add (fun _ _ => rfl) (fun _ _ => rfl)
      (T := fun B₁ B₂ => if _h : B₁.dim = B₂.dim then some (boxAdd B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxAdd_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => by
        rw [hr] at hc
        have hc' := shared?_of_ibp? hc rfl
        simp only [shared?] at hc'
        rcases h₁ : box? boxes r₁ with _ | B₁ <;> rcases h₂ : box? boxes r₂ with _ | B₂ <;>
          simp only [h₁, h₂, reduceCtorEq] at hc'
        exact ⟨B₁, B₂, rfl, rfl, hc'⟩) h
  case sub s =>
    exact encloses_binary Op.rule?_sub (fun _ _ => rfl) (fun _ _ => rfl)
      (T := fun B₁ B₂ => if _h : B₁.dim = B₂.dim then some (boxSub B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxSub_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => by
        rw [hr] at hc
        have hc' := shared?_of_ibp? hc rfl
        simp only [shared?] at hc'
        rcases h₁ : box? boxes r₁ with _ | B₁ <;> rcases h₂ : box? boxes r₂ with _ | B₂ <;>
          simp only [h₁, h₂, reduceCtorEq] at hc'
        exact ⟨B₁, B₂, rfl, rfl, hc'⟩) h
  case mul s =>
    exact encloses_binary Op.rule?_mul (fun _ _ => rfl) (fun _ _ => rfl)
      (T := boxMulElem) (fun hx hy hb => boxMulElem_encloses hx hy hb)
      hpe (fun r₁ r₂ hr => by
        rw [hr] at hc
        have hc' := shared?_of_ibp? hc rfl
        simp only [shared?] at hc'
        rcases h₁ : box? boxes r₁ with _ | B₁ <;> rcases h₂ : box? boxes r₂ with _ | B₂ <;>
          simp only [h₁, h₂, reduceCtorEq] at hc'
        exact ⟨B₁, B₂, rfl, rfl, hc'⟩) h
  case matmul bA bB b m n p lv rv hl hr hA hB =>
    obtain ⟨r₁, r₂, w₁, w₂, x₁, x₂, hrefs, hread₁, hread₂, hx₁, hx₂, rfl⟩ :=
      invoke_two (Op.rule?_matmul lv rv hl hr hA hB) h
    rw [hrefs] at hc hpe
    have hc' := shared?_of_ibp? hc rfl
    simp only [shared?] at hc'
    rcases h₁ : box? boxes r₁ with _ | L <;> rcases h₂ : box? boxes r₂ with _ | R <;>
      simp only [h₁, h₂, reduceCtorEq, Option.some.injEq] at hc'
    subst hc'
    have e₁ := hpe r₁ (by simp) L w₁ _ h₁ hread₁
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁))
    have e₂ := hpe r₂ (by simp) R w₂ _ h₂ hread₂
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂))
    intro v hv
    have hv' : v = ⟨(matmulLayout lv rv bA bB b m n p).outShape.size,
        IR.MatmulDims.matmulFlat (matmulLayout lv rv bA bB b m n p)
          (flatOfSome ⟨_, x₁⟩).v (flatOfSome ⟨_, x₂⟩).v⟩ := by
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      rw [← hv]
      exact flat_matmul hl hr hA hB x₁ x₂
    subst hv'
    have hm := Directed.binaryMatmulBox_encloses (matmulLayout lv rv bA bB b m n p) e₁ e₂
    rw [Directed.pointVector_getAtOrZero, Directed.pointVector_getAtOrZero] at hm
    exact hm
  case scaledMatmul bA bB b m n p lv rv hl hr hA hB =>
    obtain ⟨r₁, r₂, r₃, w₁, w₂, w₃, x₁, x₂, x₃, hrefs, hread₁, hread₂, hread₃, hx₁, hx₂, hx₃,
      rfl⟩ := invoke_three (Op.rule?_scaledMatmul lv rv hl hr hA hB) h
    rw [hrefs] at hc hpe
    have hc' := shared?_of_ibp? hc rfl
    simp only [shared?] at hc'
    rcases h₁ : box? boxes r₁ with _ | L <;> rcases h₂ : box? boxes r₂ with _ | R <;>
      rcases h₃ : box? boxes r₃ with _ | S <;>
      simp only [h₁, h₂, h₃, reduceCtorEq, Option.some.injEq] at hc'
    subst hc'
    have e₁ := hpe r₁ (by simp) L w₁ _ h₁ hread₁
      ((flatOf_of_read (Or.inl rfl) hx₁).trans (flatOf_value x₁))
    have e₂ := hpe r₂ (by simp) R w₂ _ h₂ hread₂
      ((flatOf_of_read (Or.inl rfl) hx₂).trans (flatOf_value x₂))
    have e₃ := hpe r₃ (by simp) S w₃ (flatOfSome ⟨[], x₃⟩) h₃ hread₃
      ((flatOf_of_read (Or.inr (Or.inl rfl)) hx₃).trans rfl)
    intro v hv
    have hv' : v = ⟨(matmulLayout lv rv bA bB b m n p).outShape.size,
        Tensor.map (fun z => z * getAtOrZero (flatOfSome ⟨[], x₃⟩).v [0])
          (IR.MatmulDims.matmulFlat (matmulLayout lv rv bA bB b m n p)
            (flatOfSome ⟨_, x₁⟩).v (flatOfSome ⟨_, x₂⟩).v)⟩ := by
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      rw [← hv]
      exact flat_scaledMatmul hl hr hA hB x₁ x₂ x₃
    subst hv'
    have hm := Directed.binaryMatmulBox_encloses (matmulLayout lv rv bA bB b m n p) e₁ e₂
    rw [Directed.pointVector_getAtOrZero, Directed.pointVector_getAtOrZero] at hm
    refine (Directed.boxScale_encloses hm e₃).congr fun i => ?_
    simp only [Tensor.getAtOrZero_eq_getScalar, Tensor.getScalar_map]
  case cat s dim sizes =>
    obtain ⟨values, xs, rule, -, hin, hx, hs, -, rfl⟩ := Op.invoke_eq_ok h
    rw [Op.rule?_cat] at hs
    cases hs
    have hc' := shared?_of_ibp? hc rfl
    simp only [shared?, catBox?, Option.bind_eq_bind] at hc'
    obtain ⟨bs, hbs, hB⟩ := Option.bind_eq_some_iff.mp hc'
    have hvals : ∀ r ∈ refs.toList, ∀ w, r.read earlier = .ok w →
        box? (runVals earlier) r = flatOf w := fun r hr w hread =>
      box?_runVals (out_eq_zero_of_mapM_box? hbs r hr) hread
    have hfs := mapM_box?_cat dim sizes hin hx hvals
    intro v hv
    simp only [Values.toErased, List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some,
      flatOf_value, Option.some.injEq] at hv
    subst hv
    refine encloses_concat hB (flat_cat dim sizes _ _ _) fun i Bp vp hBp hvp => ?_
    rw [List.getElem?_toArray] at hBp hvp
    obtain ⟨r, hr, hrB⟩ := mapM_getElem? hbs i Bp hBp
    obtain ⟨r', hr', hrv⟩ := mapM_getElem? hfs i vp hvp
    rw [hr] at hr'
    cases hr'
    have hmem := List.mem_of_getElem? hr
    obtain ⟨w, hw⟩ := read_of_mapM (f := (·.read earlier)) (fun r y hy => ⟨y, hy⟩) hin r hmem
    exact hpe r hmem Bp w vp hrB hw ((hvals r hmem w hw).symm.trans hrv)
  case conv s k r dims stride padLow padHigh dilation groups L x y hlead hx hy =>
    obtain ⟨rK, rb, rx, wK, wb, wx, Kr, br, xt, hr, hreadK, hreadb, hreadx, hxK, hxb, hxx, rfl⟩ :=
      invoke_three (Op.rule?_conv dims stride padLow padHigh dilation groups hlead hx hy) h
    rw [hr] at hc hpe hconsts
    have hc' := shared?_of_ibp? hc rfl
    simp only [shared?, Option.bind_eq_bind] at hc'
    obtain ⟨K, hK, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨b, hb, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨X, hX, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain rfl := read_constAt? hconsts (by simp) hK hreadK hxK
    obtain rfl := read_constAt? hconsts (by simp) hb hreadb hxb
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    -- Without the goal as expected type, `rule?_conv` fixes the rule before any unification.
    have hB := encloses_conv hc' xt (hpe rx (by simp) X wx _ hX hreadx
        ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value xt)))
      (Op.rule?_conv dims stride padLow padHigh dilation groups hlead hx hy)
    exact hB
  case detach s =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one Op.rule?_detach h
    rw [hr] at hc hpe
    have hc : box? boxes r = some B := shared?_of_ibp? hc rfl
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some] at hv
    exact hpe r (by simp) B w v hc hread ((flatOf_of_read (Or.inr (Or.inr rfl)) hx).trans hv)
  case reshape s t hst =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_reshape hst) h
    rw [hr] at hc hpe
    have hc : box? boxes r = some B := hc
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    have hB := hpe r (by simp) B w _ hc hread
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))
    refine ⟨hB.1.trans hst, fun i => ?_⟩
    have hi := hB.2 (Fin.cast hst.symm i)
    simp only [Graph.flatOfSome, Tensor.reshape_vector_reshape, getAtOrZero_eqRec] at hi ⊢
    exact hi
  case permute s perm out hout =>
    exact encloses_endpoints (Op.rule?_permute perm out hout)
      (Op.rule?_permute perm out hout)
      (fun x c => by
        simp only [Tensor.permute_apply, Tensor.map_apply])
      (fun x y hxy c => by simp only [Tensor.permute_apply]; exact hxy _)
      hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
  case slice s dims starts stops steps hv =>
    exact encloses_endpoints (Op.rule?_slice dims starts stops steps hv)
      (Op.rule?_slice dims starts stops steps hv)
      (fun x c => by
        simp only [Tensor.slice_apply, Tensor.map_apply])
      (fun x y hxy c => by simp only [Tensor.slice_apply]; exact hxy _)
      hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
  case flip s dims =>
    exact encloses_endpoints (Op.rule?_flip dims) (Op.rule?_flip dims)
      (fun x c => by
        simp only [Tensor.flip_apply, Tensor.map_apply])
      (fun x y hxy c => by simp only [Tensor.flip_apply]; exact hxy _)
      hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
  case broadcast s t dims hb =>
    exact encloses_endpoints (Op.rule?_broadcast dims hb)
      (Op.rule?_broadcast dims hb)
      (fun x c => by
        simp only [Tensor.broadcast_apply, Tensor.map_apply])
      (fun x y hxy c => by simp only [Tensor.broadcast_apply]; exact hxy _)
      hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
  case indexAt s dim index =>
    exact encloses_endpoints (Op.rule?_indexAt dim index)
      (Op.rule?_indexAt dim index)
      (fun x c => by
        simp only [Tensor.reshape_apply, Tensor.slice_apply,
          Tensor.map_apply])
      (fun x y hxy c => by simp only [Tensor.reshape_apply, Tensor.slice_apply]; exact hxy _)
      hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
  case linear L n m =>
    obtain ⟨rW, rb, rx, vW, vb, vx, Wr, br, x, hr, hreadW, hreadb, hreadx, hxW, hxb, hxx, rfl⟩ :=
      invoke_three (Option.some_get Op.rule?_linear_isSome).symm h
    rw [hr] at hc hpe hconsts
    have hc' := shared?_of_ibp? hc rfl
    simp only [shared?, Option.bind_eq_bind] at hc'
    obtain ⟨W, hW, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨b, hb, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨Xin, hX, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain rfl := read_constAt? hconsts (by simp) hW hreadW hxW
    obtain rfl := read_constAt? hconsts (by simp) hb hreadb hxb
    have hx := hpe rx (by simp) Xin vx _ hX hreadx
      ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value x))
    intro v hv
    rw [Op.rule?_linear_forward] at hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    have hdrop : (L ++ [n]).dropLast = L := by
      simp
    have hsize : (L ++ [m]).size = L.size * m := by simp [Shape.size_append]
    unfold ibpLinearAt at hc'
    by_cases h0 : n = 0
    · rw [ite_eq_left h0] at hc'
      obtain ⟨-, hc'⟩ := Option.dite_none_right_eq_some.mp hc'
      obtain rfl := Option.some.inj hc'
      rw [hdrop]
      refine ⟨hsize.symm, fun i => ?_⟩
      have hk : i.val < L.size * m := hsize ▸ i.isLt
      have hv := flat_linear_empty h0 W b x i.val hk
      have hlo : getAtOrZero (linearBiasRows b L.size) [i.val] =
          (linearBiasRows b L.size).getScalar ⟨i.val, hk⟩ :=
        getAtOrZero_eq_getScalar _ ⟨i.val, hk⟩
      change value (getAtOrZero (linearBiasRows b L.size) [i.val]) ≤ _ ∧
        _ ≤ value (getAtOrZero (linearBiasRows b L.size) [i.val])
      beta_reduce
      rw [hlo, hv]
      exact ⟨le_rfl, le_rfl⟩
    · rw [ite_eq_right h0] at hc'
      have hXin : Xin.dim = n := by
        unfold ibpLinearParams at hc'
        exact (Option.dite_none_right_eq_some.mp hc').1
      have hone : L.size = 1 := by
        have h1 := hx.1.symm.trans hXin
        simp only [Graph.flatOfSome, Shape.size_append, Shape.size_nil,
            Shape.size_cons] at h1
        exact (Nat.mul_eq_right (Nat.pos_of_ne_zero h0).ne').mp (by simpa using h1)
      have hm : (L ++ [m]).size = m := by rw [hsize, hone, Nat.one_mul]
      have hn : (L ++ [n]).size = n := by
        simp [Shape.size_append, hone]
      have henc := ibpLinearParams_encloses
        (p := { inDim := n, outDim := m, spec := { weights := W, bias := b } })
        (g := fun i => getAtOrZero (Graph.flatOfSome ⟨L ++ [m],
          Linear.forwardLeading L ⟨Tensor.map value W, Tensor.map value b⟩ x⟩).v [i])
        (rowEncloses_width hn hx) (fun i => ?_) hc'
      · exact rowEncloses_width hm.symm henc
      · exact flat_linear_one hone W b x i
  case pool s op dims kernel stride padLow padHigh dilation ceilMode =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ :=
      invoke_one (Op.rule?_pool op dims kernel stride padLow padHigh dilation ceilMode) h
    rw [hr] at hc hpe
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    obtain rfl | hop | ⟨c, rfl⟩ : op = .sum ∨ (op = .max ∨ op = .min) ∨ ∃ c, op = .avg c := by
      cases op <;> simp
    · have hc' : (box? boxes r >>= coordwiseBox? (s := s)
          (t := s.pool dims kernel stride padLow padHigh dilation ceilMode)
          fun lo hi o => some (poolSumBounds lo hi o)) = some B := hc
      obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
      refine coordwiseBox?_encloses (hpe r (by simp) Bx w _ hBx hread
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) ?_ hB
      intro lo hi hlh o b hb
      obtain rfl := Option.some.inj hb
      rw [Tensor.pool_sum_apply]
      exact poolSumBounds_encloses hlh o
    · obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc
      exact poolBox?_encloses hop (hpe r (by simp) Bx w _ hBx hread
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) hB
    · have hc' : (box? boxes r >>= coordwiseBox? (s := s)
          (t := s.pool dims kernel stride padLow padHigh dilation ceilMode) fun lo hi o => do
            let sum ← finiteBounds? (poolSumBounds lo hi o)
            let count ← finiteBounds? (Graph.countBounds (Tensor.poolCount c o))
            NonlinearBoundOps.divBounds sum.1 sum.2 count.1 count.2 >>= finiteBounds?) =
          some B := hc
      obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
      refine coordwiseBox?_encloses (hpe r (by simp) Bx w _ hBx hread
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))) ?_ hB
      intro lo hi hlh o b hb
      obtain ⟨sum, hsum, hb₁⟩ := Option.bind_eq_some_iff.mp hb
      obtain ⟨count, hcount, hb₂⟩ := Option.bind_eq_some_iff.mp hb₁
      obtain ⟨q, hq, hb₃⟩ := Option.bind_eq_some_iff.mp hb₂
      clear hb hb₁ hb₂
      obtain rfl := finiteBounds?_eq_of_eq_some hsum
      obtain rfl := finiteBounds?_eq_of_eq_some hcount
      obtain rfl := finiteBounds?_eq_of_eq_some hb₃
      rw [Tensor.pool_avg_apply, Context.divNat_real]
      have hs := poolSumBounds_encloses hlh o
      have hn := countBounds_encloses (α := α) (Tensor.poolCount c o)
      exact LawfulNonlinearBoundOps.divBounds_enclosure hq hs.1 hs.2 hn.1 hn.2
  case batchNormRunning s dim =>
    obtain ⟨values, xs, rule, -, hin, hxs, hs, -, rfl⟩ := Op.invoke_eq_ok h
    rw [Op.rule?_batchNormRunning] at hs
    cases hs
    obtain ⟨x, g, bt, mn, vr, ep, ⟨⟩⟩ := xs
    obtain ⟨rx, rest, vx, vs, hr, hreadx, hxx, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rg, rest, vg, vs, rfl, hreadg, hxg, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rb, rest, vb, vs, rfl, hreadb, hxb, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rm, rest, vm, vs, rfl, hreadm, hxm, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rv, rest, vv, vs, rfl, hreadv, hxv, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨re, rest, ve, vs, rfl, hreade, hxe, hin, hxs⟩ := read_cons hin hxs
    obtain rfl := read_nil hin hxs
    rw [hr] at hc hpe hconsts
    have hc' : (do
        let gamma ← constAt? G P rg (.value [s.extent dim])
        let beta ← constAt? G P rb (.value [s.extent dim])
        let mean ← constAt? G P rm (.frozen [s.extent dim])
        let var ← constAt? G P rv (.frozen [s.extent dim])
        let epsilon ← constAt? G P re (.frozen [])
        let B ← box? boxes rx
        let d ← Tensor.sequenceFinM fun k : Fin (s.extent dim) =>
          batchNormDenominator? (mean.getScalar k) (var.getScalar k)
            (gamma.getScalar k) (beta.getScalar k) (Tensor.item epsilon)
        coordwiseBox? (s := s) (t := s) (fun lo hi c =>
          let k := Shape.Coord.get c dim
          batchNormBounds (lo c) (hi c) (mean.getScalar k) (gamma.getScalar k)
            (beta.getScalar k) (d k)) B) = some B := hc
    simp only [Option.bind_eq_bind] at hc'
    obtain ⟨gamma, hgamma, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨beta, hbeta, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨mean, hmean, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨var, hvar, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨epsilon, hepsilon, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨Bx, hBx, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨d, hd, hB⟩ := Option.bind_eq_some_iff.mp hc'
    obtain rfl := read_constAt?_carrier hconsts (by simp) hgamma hreadg hxg
    obtain rfl := read_constAt?_carrier hconsts (by simp) hbeta hreadb hxb
    obtain rfl := read_constAt?_carrier hconsts (by simp) hmean hreadm hxm
    obtain rfl := read_constAt?_carrier hconsts (by simp) hvar hreadv hxv
    obtain rfl := read_constAt?_carrier hconsts (by simp) hepsilon hreade hxe
    intro v hv
    simp only [Values.toErased, List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some,
      flatOf_value, Option.some.injEq] at hv
    subst hv
    refine coordwiseBox?_encloses (hpe rx (by simp) Bx vx _ hBx hreadx
      ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value x))) ?_ hB
    intro lo hi hlh c b hb
    rw [Tensor.batchNorm_running_apply]
    simp only [realCarrier, getScalar_map, item_map, Context.maximum_real, sqrt_max_zero]
    exact batchNormBounds_encloses (hlh c)
      (batchNormDenominator?_encloses (Tensor.sequenceFinM_get_of_eq_some hd _)) hb
  case layerNorm rows w _ hw _ _ _ _ =>
    obtain ⟨values, xs, rule, -, hin, hxs, hs, -, rfl⟩ := Op.invoke_eq_ok h
    rw [Op.rule?_layerNorm] at hs
    cases hs
    obtain ⟨x, g, bt, ep, ⟨⟩⟩ := xs
    obtain ⟨rx, rest, vx, vs, hr, hreadx, hxx, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rg, rest, vg, vs, rfl, hreadg, hxg, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨rb, rest, vb, vs, rfl, hreadb, hxb, hin, hxs⟩ := read_cons hin hxs
    obtain ⟨re, rest, ve, vs, rfl, hreade, hxe, hin, hxs⟩ := read_cons hin hxs
    obtain rfl := read_nil hin hxs
    rw [hr] at hc hpe hconsts
    have hc' : (do
        let gamma ← constAt? G P rg (.value [w])
        let beta ← constAt? G P rb (.value [w])
        let epsilon ← constAt? G P re (.frozen [])
        let B ← box? boxes rx
        layerNormBox? rows w gamma beta (Tensor.item epsilon) B) = some B := hc
    simp only [Option.bind_eq_bind] at hc'
    obtain ⟨gamma, hgamma, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨beta, hbeta, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨epsilon, hepsilon, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨Bx, hBx, hB⟩ := Option.bind_eq_some_iff.mp hc'
    obtain rfl := read_constAt?_carrier hconsts (by simp) hgamma hreadg hxg
    obtain rfl := read_constAt?_carrier hconsts (by simp) hbeta hreadb hxb
    obtain rfl := read_constAt?_carrier hconsts (by simp) hepsilon hreade hxe
    have hx := hpe rx (by simp) Bx vx _ hBx hreadx
      ((flatOf_of_read (Or.inl rfl) hxx).trans (flatOf_value x))
    intro v hv
    simp only [Values.toErased, List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some,
      flatOf_value, Option.some.injEq] at hv
    subst hv
    rw [encloses_flat_reshape]
    simp only [realCarrier, item_map]
    exact layerNormBox?_encloses hw ((encloses_flat_reshape _).mpr hx) hB
  case «reduce» s op dims keepdim hop =>
    rcases hrule with ⟨rfl, rfl, hdims⟩ | rfl | rfl
    rotate_left
    · exact encloses_endpoints (Op.rule?_reduce .max dims keepdim hop)
        (Op.rule?_reduce .max dims keepdim hop)
        (fun x c => by
          simp only [Tensor.reduce_max_apply, Tensor.map_apply,
            Context.maximum_real]
          exact (List.foldl_hom value fun _ _ => (LawfulBoundOps.toReal_maximum _ _).symm).symm)
        (fun x y hxy c => by
          simp only [Tensor.reduce_max_apply, Context.maximum_real]
          exact List.foldl_rel (r := (· ≤ ·)) (hxy _) fun _ _ _ _ h => max_le_max h (hxy _))
        hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
    · exact encloses_endpoints (Op.rule?_reduce .min dims keepdim hop)
        (Op.rule?_reduce .min dims keepdim hop)
        (fun x c => by
          simp only [Tensor.reduce_min_apply, Tensor.map_apply,
            Context.minimum_real]
          exact (List.foldl_hom value fun _ _ => (LawfulBoundOps.toReal_minimum _ _).symm).symm)
        (fun x y hxy c => by
          simp only [Tensor.reduce_min_apply, Context.minimum_real]
          exact List.foldl_rel (r := (· ≤ ·)) (hxy _) fun _ _ _ _ h => min_le_min h (hxy _))
        hpe (fun r hr => by rw [hr] at hc; exact directed?_of_ibp? hc rfl) h
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_reduce .sum _ false hop) h
    rw [hr] at hc hpe
    have hc' : (box? boxes r >>= ibpReduce? .sum (dims.map Fin.val).toArray false s) = some B := hc
    obtain ⟨Bx, hBx, hred⟩ := Option.bind_eq_some_iff.mp hc'
    have henc := hpe r (by simp) Bx w _ hBx hread
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    by_cases hall : dims.map Fin.val = List.range s.rank
    · obtain rfl : dims = List.finRange s.rank :=
        List.map_injective_iff.mpr Fin.val_injective
          (hall.trans (List.map_coe_finRange_eq_range).symm)
      have hrange : ((List.finRange s.rank).map Fin.val).toArray = Array.range s.rank := by
        apply Array.ext'
        simp [List.map_coe_finRange_eq_range]
      rw [hrange] at hred
      simp only [ibpReduce?, and_self, ↓reduceIte, Option.some.injEq] at hred
      subst hred
      have hone : (s.reduce (List.finRange s.rank) false).size = 1 := by
        rw [Shape.reduce_finRange]; rfl
      refine rowEncloses_width hone.symm (boxSum_encloses (rowEncloses_width henc.1.symm henc) ?_)
      rw [flat_reduce_sum]
      exact (Fin.sum_congr' (fun i : Fin s.size =>
        getAtOrZero (Graph.flatOfSome ⟨s, x⟩).v [i.val]) henc.1).symm
    obtain ⟨d, rfl⟩ := List.length_eq_one_iff.mp (hdims.resolve_left hall)
    have hne : ([d].map Fin.val).toArray ≠ Array.range s.rank := fun e =>
      hall (by simpa using congrArg Array.toList e)
    have hred' : ibpReduceSumAxis d.val Bx s = some B := by
      rw [← hred]
      unfold ibpReduce?
      rw [ite_eq_left ⟨rfl, rfl⟩, ite_eq_right hne,
        dite_eq_left (show ([d].map Fin.val).toArray.size = 1 from rfl)]
      rfl
    refine (rowEncloses_width (congrArg Shape.size (Shape.reduce_singleton s d)).symm
      (ibpReduceSumAxis_encloses (rowEncloses_width rfl henc) hred')).congr fun i => ?_
    exact flat_reduce_sum_axis x d _ i.val
  case mean s dims keepdim =>
    obtain ⟨rfl, hlen⟩ := hrule
    obtain ⟨d, rfl⟩ := List.length_eq_one_iff.mp hlen
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_mean [d] false) h
    rw [hr] at hc hpe
    have hc' : (box? boxes r >>= ibpMean? ([d].map Fin.val).toArray false s) = some B := hc
    obtain ⟨Bx, hBx, hred⟩ := Option.bind_eq_some_iff.mp hc'
    have hred' : ibpReduceMeanAxis d.val Bx s = some B := by
      rw [← hred]
      unfold ibpMean?
      rw [dite_eq_left (show ([d].map Fin.val).toArray.size = 1 ∧ false = false from ⟨rfl, rfl⟩)]
      rfl
    have henc := hpe r (by simp) Bx w _ hBx hread
      ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x))
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    refine (rowEncloses_width (congrArg Shape.size (Shape.reduce_singleton s d)).symm
      (ibpReduceMeanAxis_encloses (rowEncloses_width rfl henc) hred')).congr fun i => ?_
    exact flat_mean_axis x d (ibpReduceMeanAxis_extent_pos hred' d.isLt) i

  case abs s =>
    exact encloses_unary Op.rule?_abs (fun _ => rfl) (F := fun x => |x|) (fun _ => rfl)
      (T := fun B => some (boxAbs B))
      (fun hx hb => by obtain rfl := Option.some.inj hb; exact boxAbs_encloses hx)
      hpe (fun r hr => by
        rw [hr] at hc
        have hc' : (box? boxes r).map boxAbs = some B := hc
        cases hb : box? boxes r <;> simp_all) h
  case max s =>
    exact encloses_binary Op.rule?_max (f := Context.maximum) (fun _ _ => rfl)
      (F := fun a b => max a b) (fun a b => Context.maximum_real a b)
      (T := fun B₁ B₂ => if B₁.dim = B₂.dim then some (boxMaxElem B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxMaxElem_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => by
        rw [hr] at hc
        have hc' := directed?_of_ibp? hc rfl
        simp only [directed?] at hc'
        rcases h₁ : box? boxes r₁ with _ | B₁ <;> rcases h₂ : box? boxes r₂ with _ | B₂ <;>
          simp only [h₁, h₂, reduceCtorEq] at hc'
        exact ⟨B₁, B₂, rfl, rfl, hc'⟩) h
  case min s =>
    exact encloses_binary Op.rule?_min (f := Context.minimum) (fun _ _ => rfl)
      (F := fun a b => min a b) (fun a b => Context.minimum_real a b)
      (T := fun B₁ B₂ => if B₁.dim = B₂.dim then some (boxMinElem B₁ B₂) else none)
      (fun hx hy hb => by
        split at hb
        · obtain rfl := Option.some.inj hb; exact boxMinElem_encloses hx hy
        · cases hb)
      hpe (fun r₁ r₂ hr => by
        rw [hr] at hc
        have hc' := directed?_of_ibp? hc rfl
        simp only [directed?] at hc'
        rcases h₁ : box? boxes r₁ with _ | B₁ <;> rcases h₂ : box? boxes r₂ with _ | B₂ <;>
          simp only [h₁, h₂, reduceCtorEq] at hc'
        exact ⟨B₁, B₂, rfl, rfl, hc'⟩) h
  case softplus s =>
    exact encloses_unary Op.rule?_softplus (fun _ => rfl)
      (F := fun x => Real.log (1 + Real.exp x)) Activation.softplus_eq_log_one_add_exp
      (T := boxSoftplus?) (fun hx hb => boxUnaryEnclosure?_encloses softplusBounds_enclosure hx hb)
      hpe (fun r hr => by rw [hr] at hc; exact shared?_of_ibp? hc rfl) h
  case safeLog s =>
    obtain ⟨r₁, r₂, w₁, w₂, x, e, hr, hread₁, hread₂, hx, he, rfl⟩ :=
      invoke_two Op.rule?_safeLog h
    rw [hr] at hc hpe
    have hc' : (match box? boxes r₁, box? boxes r₂ with
        | some B, some epsilon => boxSafeLog? B epsilon
        | _, _ => none) = some B := shared?_of_ibp? hc rfl
    rcases h₁ : box? boxes r₁ with _ | Bx <;> rcases h₂ : box? boxes r₂ with _ | Be <;>
      simp only [h₁, h₂, reduceCtorEq] at hc'
    have hBe := hpe r₂ (by simp) Be w₂ (Graph.flatOfSome ⟨[], e⟩) h₂ hread₂
      ((flatOf_of_read (Or.inr (Or.inl rfl)) he).trans rfl)
    have h0 : getAtOrZero (Graph.flatOfSome ⟨[], e⟩).v [0] = e.item :=
      Tensor.getAtOrZero_reshape_toFin e PUnit.unit
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    refine encloses_map (F := fun z => Real.log (Real.log (1 + Real.exp z) +
        getAtOrZero (Graph.flatOfSome ⟨[], e⟩).v [0])) (fun z => ?_)
      (hpe r₁ (by simp) Bx w₁ _ h₁ hread₁
        ((flatOf_of_read (Or.inl rfl) hx).trans (flatOf_value x)))
      fun hg => boxSafeLog?_encloses hg hBe hc'
    rw [h0, ← Activation.softplus_eq_log_one_add_exp]
    rfl
  case mseLoss s =>
    obtain ⟨r₁, r₂, w₁, w₂, y, t, hr, hread₁, hread₂, hy, ht, rfl⟩ :=
      invoke_two Op.rule?_mseLoss h
    rw [hr] at hc hpe
    have hc' : (match box? boxes r₁, box? boxes r₂ with
        | some y, some t => if y.dim = t.dim then boxMean? (boxSquare (boxSub y t)) else none
        | _, _ => none) = some B := hc
    rcases h₁ : box? boxes r₁ with _ | By <;> rcases h₂ : box? boxes r₂ with _ | Bt <;>
      simp only [h₁, h₂, reduceCtorEq] at hc'
    split at hc'
    · have hBy := hpe r₁ (by simp) By w₁ _ h₁ hread₁
        ((flatOf_of_read (Or.inl rfl) hy).trans (flatOf_value y))
      have hBt := hpe r₂ (by simp) Bt w₂ _ h₂ hread₂
        ((flatOf_of_read (Or.inl rfl) ht).trans (flatOf_value t))
      intro v hv
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      exact boxMean?_encloses (boxSquare_encloses (boxSub_encloses hBy hBt))
        (flat_meanSquaredError y t) hc'
    · cases hc'
  case softmax s dim =>
    obtain ⟨r, w, x, hr, hread, hx, rfl⟩ := invoke_one (Op.rule?_softmax dim) h
    rw [hr] at hc hpe
    have hc' : (do
        let B ← box? boxes r
        finiteBox? B
        if B.dim = s.size then some (ibpSoftmaxRange s dim.val B.dim) else none) = some B := hc
    simp only [Option.bind_eq_bind] at hc'
    obtain ⟨Bx, -, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨_, -, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    split at hc'
    · rename_i hd
      obtain rfl := Option.some.inj hc'
      intro v hv
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      have : Shape.AxisInBounds dim.val s := ⟨dim.isLt⟩
      rw [hd]
      exact ibpSoftmaxRange_encloses dim.val x fun c => Tensor.getAtOrZero_reshape_toFin _ c
    · cases hc'
  case maskedSoftmax s dim =>
    obtain ⟨r, rm, _, wm, x, m, hr, -, hreadm, -, hm, rfl⟩ :=
      invoke_two (Op.rule?_maskedSoftmax dim) h
    rw [hr] at hc hconsts
    have hc' : (do
        let B ← box? boxes r
        let allowed ← constAt? G P rm (.bool s)
        finiteBox? B
        if dim.val + 1 = s.rank then maskedSoftmaxBox? s allowed B else none) =
          some B := hc
    simp only [Option.bind_eq_bind] at hc'
    obtain ⟨Bx, -, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨allowed, hallowed, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    obtain ⟨_, -, hc'⟩ := Option.bind_eq_some_iff.mp hc'
    split at hc'
    · rename_i hdim
      obtain rfl : m = allowed := read_constAt?_carrier hconsts (by simp) hallowed hreadm hm
      intro v hv
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
        Option.some.injEq] at hv
      subst hv
      exact maskedSoftmaxBox?_encloses hdim x hc'
    · cases hc'
  case randUniform s key =>
    obtain ⟨values, xs, rule, -, hin, hxs, hs, -, rfl⟩ := Op.invoke_eq_ok h
    rw [Op.rule?_randUniform] at hs
    cases hs
    rw [read_nil hin hxs] at hc
    have hc' : some (unitBox (α := α) s.size) = some B := hc
    obtain rfl := Option.some.inj hc'
    intro v hv
    simp only [Values.toErased, List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some,
      flatOf_value, Option.some.injEq] at hv
    subst hv
    refine encloses_of_coord rfl fun c => ?_
    have hsupport := Random.uniform_mem_Ico key c
    simp only [unitBox, getAtOrZero_eq_getScalar, getScalar_full,
      LawfulBoundOps.toReal_zero (α := α), LawfulBoundOps.toReal_one (α := α)]
    exact ⟨hsupport.1, hsupport.2.le⟩
  case bernoulliMask s key =>
    obtain ⟨r, -, p, hr, -, -, rfl⟩ := invoke_one (Op.rule?_bernoulliMask key) h
    rw [hr] at hc
    have hc' : some (unitBox (α := α) s.size) = some B := hc
    obtain rfl := Option.some.inj hc'
    intro v hv
    simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some, flatOf_value,
      Option.some.injEq] at hv
    subst hv
    refine encloses_of_coord rfl fun c => ?_
    have hsupport := Random.mask_mem_Icc key p.item c
    simp only [unitBox, getAtOrZero_eq_getScalar, getScalar_full,
      LawfulBoundOps.toReal_zero (α := α), LawfulBoundOps.toReal_one (α := α)]
    exact Set.mem_Icc.mp hsupport

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- The flat tensor of a real value is the flat tensor of the rounded value read through
`toReal`. -/
theorem flatOf_realValue (v : Op.Value α) :
    flatOf (realValue v) = (flatOf v).map fun t => ⟨t.n, Tensor.map value t.v⟩ := by
  have hr : ∀ {t : Shape} (z : Tensor α t),
      Tensor.reshape (Tensor.map value z) [t.size] (Shape.size_vector _).symm =
        Tensor.map value (Tensor.reshape z [t.size] (Shape.size_vector _).symm) := by
    intro t z
    unfold Tensor.reshape
    exact (Tensor.Rep.map_reshape _ _ _).symm
  obtain ⟨⟨d, s⟩, x⟩ := v
  cases d <;> simp [realValue, realCarrier, flatOf, hr]

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- In a successful real run on a payload holding the reals of the rounded constants, every
reference of a node to a rounded constant reads that constant's real value. -/
theorem constsReal_of_run {G : IR.Graph} {P : IR.Payload α} {Pr : IR.Payload ℝ}
    (hP : RealConsts Pr P) {inputs : List (Op.Value ℝ)}
    {values : Array (List (Op.Value ℝ))} (hrun : G.run Pr inputs = .ok values)
    {pre : Array (List (Op.Value ℝ))} {j : Nat}
    (hagree : ∀ k, k < j → pre[k]? = values[k]?) (hsize : pre.size = j) (refs : List IR.Ref) :
    ConstsReal G P pre refs := by
  intro r _ w hw h0 v hv
  have hnode : ∃ q, G.nodes[r.node]? = some (.const q) := by
    unfold constValue? at hv
    split at hv
    · exact ⟨_, by assumption⟩
    · cases hv
  obtain ⟨q, hq⟩ := hnode
  obtain ⟨_, outs, -, -, heval, hout⟩ := IR.Graph.run_step hrun hq
  obtain ⟨w', hw', rfl⟩ := const_eval hq heval
  have hpre := (hagree r.node (hsize ▸ IR.Ref.read_lt hw)).trans hout
  rw [constValue?_real hP, hv, Option.map_some] at hw'
  obtain rfl := (Option.some.inj hw').symm
  unfold IR.Ref.read at hw
  rw [hpre, h0] at hw
  exact (Except.ok.inj hw).symm

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- On any table of boxes, the box `step?` gives a node under a sound rule (`RuleEncloses`) encloses
the node's flat value in a successful real run on a payload holding the reals of the rounded
constants, when every seed box encloses the graph input its input node reads and the table's box at
each reference of the node to an earlier node encloses the flat value there. -/
theorem step?_encloses (G : IR.Graph) (P : IR.Payload α) (Pr : IR.Payload ℝ)
    {rule : Array (Option (FlatBox α)) → IR.Operator → List IR.Ref → Option (FlatBox α)}
    (hrule : RuleEncloses G P Pr rule) (seeds : Nat → Option (FlatBox α))
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hP : RealConsts Pr P)
    (hin : ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v → Encloses B v)
    (hrun : G.run Pr inputs = .ok values) (tbl : Array (Option (FlatBox α))) (id : Nat)
    (hparents : ∀ o refs, G.nodes[id]? = some (.op o refs) → ∀ r ∈ refs.toList, r.node < id →
      ∀ (B : FlatBox α) (v : FlatTensor ℝ), box? tbl r = some B → box? (runVals values) r = some v →
        Encloses B v)
    (B : FlatBox α) (v : FlatTensor ℝ) (hstep : step? G P rule seeds tbl id = some B)
    (hv : runVal values id = some v) : Encloses B v := by
  obtain ⟨n, hn⟩ : ∃ n, G.nodes[id]? = some n := by
    cases hn : G.nodes[id]? with
    | none => simp [step?, hn] at hstep
    | some n => exact ⟨n, rfl⟩
  obtain ⟨pre, outs, hsize, hagree, heval, hout⟩ := IR.Graph.run_step hrun hn
  rw [runVal_eq hout] at hv
  unfold step? at hstep
  replace hstep := (Option.filter_eq_some_iff.mp hstep).1
  rw [hn] at hstep
  cases n with
  | input k p =>
      obtain ⟨x, y, hx, hy, rfl⟩ := input_eval heval
      exact hin id k p hn B hstep x y v hx hy hv
  | const p =>
      obtain ⟨w, hw, rfl⟩ := const_eval hn heval
      rw [constValue?_real hP] at hw
      obtain ⟨u, hu, rfl⟩ := Option.map_eq_some_iff.mp hw
      simp only [List.getElem?_cons_zero, Option.bind_eq_bind, Option.bind_some,
        flatOf_realValue] at hv
      simp only [hu, Option.bind_eq_bind, Option.bind_some] at hstep
      obtain ⟨t, ht, rfl⟩ := Option.map_eq_some_iff.mp hv
      rw [ht, Option.map_some] at hstep
      obtain rfl := Option.some.inj hstep
      refine ⟨rfl, fun i => ?_⟩
      simp only [getAtOrZero_eq_getScalar, getScalar_map]
      exact ⟨le_rfl, le_rfl⟩
  | op o refs =>
      simp only at hstep
      split at hstep
      · rename_i hall
        have h0 : ∀ r ∈ refs.toList, r.out = 0 := by
          intro r hr
          have hr' : r ∈ refs := by simpa using hr
          obtain ⟨i, hi, rfl⟩ := Array.getElem_of_mem hr'
          simpa using Array.all_eq_true.mp hall i hi
        refine hrule tbl inputs pre id o refs B outs
          (constsReal_of_run hP hrun hagree hsize refs.toList) (fun r hr Br w u hBr hread hu => ?_)
          hstep heval v hv
        have hlt : r.node < id := hsize ▸ IR.Ref.read_lt hread
        exact hparents o refs hn r hr hlt Br u hBr
          ((box?_runVals (h0 r hr) (read_of_agree hagree hsize hread)).trans hu)
      · cases hstep

omit [NonlinearBoundOps α] [LawfulNonlinearBoundOps α] in
/-- Under a sound rule (`RuleEncloses`), every box of a table whose entry at each node is the
`step?` of the rule on a table agreeing with it below the node encloses the flat value of its node
in a successful real run on a payload holding the reals of the rounded constants, when every seed
box encloses the graph input its input node reads. -/
theorem encloses_of_step (G : IR.Graph) (P : IR.Payload α) (Pr : IR.Payload ℝ)
    {rule : Array (Option (FlatBox α)) → IR.Operator → List IR.Ref → Option (FlatBox α)}
    (hrule : RuleEncloses G P Pr rule) (seeds : Nat → Option (FlatBox α))
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hP : RealConsts Pr P)
    (hin : ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v → Encloses B v)
    (hrun : G.run Pr inputs = .ok values) (tbl : Array (Option (FlatBox α)))
    (htbl : ∀ id, id < G.nodes.size → ∃ T : Array (Option (FlatBox α)),
      (∀ j, j < id → T[j]? = tbl[j]?) ∧ tbl[id]! = step? G P rule seeds T id) :
    ∀ id, id < G.nodes.size → ∀ (B : FlatBox α) (v : FlatTensor ℝ),
      tbl[id]! = some B → runVal values id = some v → Encloses B v := by
  intro id
  induction id using Nat.strong_induction_on with
  | _ k ih =>
    intro hk B v hB hv
    obtain ⟨T, hT, hstep⟩ := htbl k hk
    refine step?_encloses G P Pr hrule seeds inputs values hP hin hrun T k
      (fun o refs hn r hr hlt Br u hBr hu => ?_) B v (hstep.symm.trans hB) hv
    have hbox : box? T r = box? tbl r := by unfold box?; rw [hT r.node hlt]
    rw [hbox] at hBr
    exact ih r.node hlt (lt_trans hlt hk) Br u (getElem!_of_box? hBr)
      ((join_runVals values r.node).symm.trans (join_of_box? hu))

/-- Every box of the rounded forward pass at a lawful scalar encloses the flat value of its node in
a successful real run on a payload holding the reals of the rounded constants. The only other
premise is that the seed boxes enclose the graph inputs their input nodes read, with none on the
operators, the parents' shapes or the node order. -/
theorem encloses_of_run (G : IR.Graph) (P : IR.Payload α) (Pr : IR.Payload ℝ)
    (seeds : Nat → Option (FlatBox α)) (inputs : List (Op.Value ℝ))
    (values : Array (List (Op.Value ℝ))) (hP : RealConsts Pr P)
    (hin : ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v → Encloses B v)
    (hrun : G.run Pr inputs = .ok values) :
    ∀ id, id < G.nodes.size → ∀ (B : FlatBox α) (v : FlatTensor ℝ),
      (runIBP G P seeds)[id]! = some B → runVal values id = some v → Encloses B v :=
  encloses_of_step G P Pr ibp?_encloses seeds inputs values hP hin hrun (runIBP G P seeds)
    fun _ hid => by unfold runIBP; exact ⟨_, tabulate_getElem! hid⟩

end

end TorchLean.CROWN.Op.Directed
