/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.NumericalCertificate.Enclosure
public import TorchLean.NN.IR.Eval
public import TorchLean.NN.IR.Pretty

/-!
# Numerical certificate contracts

Canonical local range rules, the range transfer of each supported operator, read off its
constructor of `TorchLean.Op`, and graph range trace construction. Most users should import
`TorchLean.Verify.NumericalCertificate`.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Numerics (Interval)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace TorchLean
namespace NumericalCertificate

open Backend
open IR

/-! ## Canonical local transfer rules -/

/-- How a node enclosure was obtained from source assumptions or earlier nodes.

The rule is recorded to make certificate diagnostics useful. It is not accepted on faith:
`check` reconstructs the rule and endpoints from the graph.
-/
inductive RangeRule where
  /-- The source assumption of the certificate for an input, constant or random node. -/
  | source
  /-- The enclosure of node `parent`, unchanged. -/
  | preserve (parent : Nat)
  /-- The outward-rounded interval sum of the enclosures of nodes `left` and `right`. -/
  | add (left right : Nat)
  /-- The outward-rounded interval difference of the enclosures of `left` and `right`. -/
  | sub (left right : Nat)
  /-- The outward-rounded interval product of the enclosures of `left` and `right`. -/
  | mul (left right : Nat)
  /-- The interval reciprocal of the enclosure of `parent`, which excludes zero. -/
  | inv (parent : Nat)
  /-- The hull of the enclosures of the nodes `parents`. -/
  | hull (parents : Array Nat)
  /-- The hull of the enclosure of `parent` and zero. -/
  | hullZero (parent : Nat)
  /-- `sumLeftRange count` of the enclosure of `parent`: a left fold adding `count` entries. -/
  | sumLeft (parent count : Nat)
  /-- `meanLeftRange count` of the enclosure of `parent`: the left-fold mean of `count`
  entries. -/
  | meanLeft (parent count : Nat)
  /-- `sumLeftRange innerDim` of the interval product of the enclosures of `left` and
  `right`. -/
  | matmulLeft (left right innerDim : Nat)
  /-- `scaledContractionRange? innerDim` of the enclosures of `left`, `right` and `scale`. -/
  | scaledMatmul (left right scale innerDim : Nat)
  /-- `meanLeftRange windowSize` of the enclosure of `parent`, joined with zero when
  `includesPadding` holds. -/
  | averageWindowLeft (parent windowSize : Nat) (includesPadding : Bool)
  /-- The left-fold mean over `count` entries of the squared residual range of `prediction`
  against `target`, clamped below at zero, or the point zero when `count = 0`. -/
  | mseLeft (prediction target count : Nat)
  /-- The LayerNorm transfer of `parent` over a last axis of `normalizedSize` entries, with
  the enclosures of `scale`, `bias` and `epsilon`. -/
  | layerNormLeft (parent scale bias epsilon normalizedSize : Nat)
  /-- The unit interval, which encloses the softmax of `parent` along the nonempty axis
  `axis`. -/
  | softmaxUnit (parent axis : Nat)
  /-- The interval ReLU of the enclosure of `parent`. -/
  | relu (parent : Nat)
  /-- The interval absolute value of the enclosure of `parent`. -/
  | abs (parent : Nat)
  /-- The interval square root of the enclosure of `parent`, whose endpoints are nonnegative. -/
  | sqrtNonnegative (parent : Nat)
  /-- The interval `[0, 1]`, which encloses the sigmoid of `parent`. -/
  | unitBound (parent : Nat)
  /-- The interval `[-1, 1]`, which encloses the tanh, sine or cosine of `parent`. -/
  | signedUnitBound (parent : Nat)
  deriving DecidableEq, Repr

/-- Proof-free data for one graph node's numerical range. -/
structure NodeRange where
  /-- The id of the node in the graph. -/
  nodeId : Nat
  /-- The shape of the node's output. -/
  outShape : Shape
  /-- The rule that computes `enclosure`. -/
  rule : RangeRule
  /-- The binary32 interval that `rule` computes for the node. -/
  enclosure : Interval (Binary 8 23)
  deriving Repr

/-- A node range whose executable interval has passed the finite/order check. -/
structure CheckedNodeRange extends NodeRange where
  /-- The endpoints of `enclosure` are finite and ordered. -/
  valid : Binary.Interval.Valid enclosure

/-- A checked node range prints as its `NodeRange`, without the proof `valid`. -/
instance : Repr CheckedNodeRange where
  reprPrec r _ := repr r.toNodeRange

/-- Check a dynamic graph value against the declared shape and interval of one certificate row. -/
def dvalWithinRange (range : CheckedNodeRange) (value : SomeTensor (Binary 8 23)) : Bool :=
  if h : value.shape = range.outShape then
    tensorWithinRange range.enclosure (h ▸ value.tensor)
  else
    false

/-- A real dynamic graph value has the shape declared by a certificate row and is enclosed by its
interval. The shape equality is an existential witness, and the tensor is cast along it. -/
def SomeTensorEnclosed (range : CheckedNodeRange) (value : SomeTensor Real) : Prop :=
  ∃ h : value.shape = range.outShape,
    TensorEnclosed range.enclosure (h ▸ value.tensor)

/-- A real and an IEEE dynamic graph value both have the shape of a certificate row, and each
entry of the IEEE value is within the width of the row's interval of the real entry. -/
def SomeTensorErrorLe (range : CheckedNodeRange) (exact : SomeTensor Real)
    (computed : SomeTensor (Binary 8 23)) : Prop :=
  ∃ hexact : exact.shape = range.outShape,
    ∃ hcomputed : computed.shape = range.outShape,
      TensorErrorLe (intervalWidth range.enclosure)
        (hexact ▸ exact.tensor) (hcomputed ▸ computed.tensor)

/-- If a real value is enclosed by a certificate row and an IEEE value passes `dvalWithinRange` at
that row, the two satisfy `SomeTensorErrorLe` there. -/
theorem SomeTensorErrorLe.of_check {range : CheckedNodeRange}
    {exact : SomeTensor Real} {computed : SomeTensor (Binary 8 23)}
    (hexact : SomeTensorEnclosed range exact)
    (hcomputed : dvalWithinRange range computed = true) :
    SomeTensorErrorLe range exact computed := by
  rcases hexact with ⟨hexactShape, hexactRange⟩
  unfold dvalWithinRange at hcomputed
  split at hcomputed
  next hcomputedShape =>
    exact ⟨hexactShape, hcomputedShape,
      TensorErrorLe.of_check range.valid hexactRange hcomputed⟩
  next _ => simp at hcomputed

/-- Check every node value of a graph run (`nodeValues`) against the corresponding certificate
row. -/
def withinRanges (ranges : Array CheckedNodeRange)
    (values : Array (SomeTensor (Binary 8 23))) : Bool :=
  decide (ranges.size = values.size) &&
    (Array.zipWith dvalWithinRange ranges values).all id

/-- Pointwise relation between two equally sized runtime arrays. -/
def ArraysRelated {α β : Type} (relation : α → β → Prop)
    (xs : Array α) (ys : Array β) : Prop :=
  xs.size = ys.size ∧
    ∀ (i : Nat) (hxs : i < xs.size) (hys : i < ys.size), relation xs[i] ys[i]

/-- A successful replay check is exactly a size match plus a successful check at every node. -/
theorem withinRanges_eq_true_iff
    (ranges : Array CheckedNodeRange) (values : Array (SomeTensor (Binary 8 23))) :
    withinRanges ranges values = true ↔
      ArraysRelated (fun range value => dvalWithinRange range value = true) ranges values := by
  constructor
  · intro h
    simp only [withinRanges, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases h with ⟨hsize, hall⟩
    refine ⟨hsize, ?_⟩
    intro i hrange hvalue
    have hzip : i < (Array.zipWith dvalWithinRange ranges values).size := by
      simpa [Array.size_zipWith] using And.intro hrange hvalue
    have hi := (Array.all_eq_true.mp hall) i hzip
    simpa [Array.getElem_zipWith] using hi
  · rintro ⟨hsize, hat⟩
    simp only [withinRanges, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨hsize, Array.all_eq_true.mpr ?_⟩
    intro i hzip
    have hi : i < ranges.size ∧ i < values.size := by
      simpa [Array.size_zipWith] using hzip
    simpa [Array.getElem_zipWith] using hat i hi.1 hi.2

/-- Pointwise error bounds over a whole graph: one real and one IEEE value per row, each pair
related by `SomeTensorErrorLe` at its row. -/
def ErrorTrace (ranges : Array CheckedNodeRange)
    (exact : Array (SomeTensor Real))
    (computed : Array (SomeTensor (Binary 8 23))) : Prop :=
  ranges.size = exact.size ∧ ranges.size = computed.size ∧
    ∀ (i : Nat) (hrange : i < ranges.size) (hexact : i < exact.size)
      (hcomputed : i < computed.size),
      SomeTensorErrorLe ranges[i] exact[i] computed[i]

/-- Compose real enclosure proofs and successful IEEE replay checks into an error trace. -/
theorem ErrorTrace.of_check
    {ranges : Array CheckedNodeRange} {exact : Array (SomeTensor Real)}
    {computed : Array (SomeTensor (Binary 8 23))}
    (hexact : ArraysRelated SomeTensorEnclosed ranges exact)
    (hcomputed : withinRanges ranges computed = true) :
    ErrorTrace ranges exact computed := by
  rcases hexact with ⟨hexactSize, hexactAt⟩
  rcases (withinRanges_eq_true_iff ranges computed).mp hcomputed with
    ⟨hcomputedSize, hcomputedAt⟩
  refine ⟨hexactSize, hcomputedSize, ?_⟩
  intro i hrange hexactIndex hcomputedIndex
  exact SomeTensorErrorLe.of_check
    (hexactAt i hrange hexactIndex) (hcomputedAt i hrange hcomputedIndex)

/-- Compare a checked canonical row with untrusted raw certificate data. -/
def sameNodeRange (checked : CheckedNodeRange) (raw : NodeRange) : Bool :=
  decide (checked.nodeId = raw.nodeId) &&
    decide (checked.outShape = raw.outShape) &&
    decide (checked.rule = raw.rule) &&
    sameIntervalBits checked.enclosure raw.enclosure

/-- Read the checked row of the node a reference names. Rows exist only for nodes with one output,
so a reference to another output is refused. Graph checking makes successful lookups refer only
to earlier rows, and the errors guard calls made outside that check. -/
def parentNodeRange (ranges : Array CheckedNodeRange) (r : Ref) :
    Except String CheckedNodeRange := do
  unless r.out = 0 do
    throw s!"numerical certificate: output {r.out} of node {r.node} has no range row"
  match ranges[r.node]? with
  | some range => pure range
  | none => throw s!"numerical certificate: missing range for parent node {r.node}"

/-- Read the checked enclosure of a parent node. -/
def parentRange (ranges : Array CheckedNodeRange) (r : Ref) :
    Except String (Interval (Binary 8 23)) :=
  (·.enclosure) <$> parentNodeRange ranges r

/-- Left-fold mean range, using the same binary32 conversion of the divisor as the tensor
context. -/
def meanLeftRange (count : Nat) (range : Interval (Binary 8 23)) : Interval (Binary 8 23) :=
  Binary.Interval.div (sumLeftRange count range)
    (Binary.Interval.point (count : Binary 8 23))

/-- The numerical policy of the capsule the plan selected for node `nodeId`, or `none` when the
plan has no kernel for it. -/
def nodePolicy (plan : AcceptedGraphKernelPlan) (nodeId : Nat) : Option NumericalPolicy :=
  (plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == nodeId)).map
    (fun kernel => kernel.capsule.numericalPolicy)

/-- Reductions are propagated only when the selected capsule of node `id` promises the same fixed
left fold as the canonical tensor semantics. Other schedules need the order-independent reduction
bound from `TorchLean.Verify.Floats.IEEE754.Reductions`, and they are rejected here so that none is
labeled deterministic. A node without a numerical policy is rejected too. -/
def fixedLeftGuard (plan : AcceptedGraphKernelPlan) (id : Nat) (node : Node) :
    Except String Unit :=
  match nodePolicy plan id with
  | some policy =>
      if policy.reduction = .fixedLeft then
        pure ()
      else
        throw (s!"numerical certificate: node {id} ({node.tag}) uses reduction " ++
          s!"policy {repr policy.reduction}; fixedLeft is required by this transfer")
  | none =>
      throw (s!"numerical certificate: node {id} ({node.tag}) has no backend " ++
        "numerical policy")

/-- Hull of a nonempty array of parent ranges. -/
def hullParents (ranges : Array CheckedNodeRange) (parents : Array Ref) :
    Except String (Interval (Binary 8 23)) := do
  let parent <- match parents[0]? with
    | some parent => pure parent
    | none => throw "numerical certificate: an interval hull requires at least one parent"
  let first <- parentRange ranges parent
  (parents.extract 1 parents.size).foldlM (fun acc r => do
    let next <- parentRange ranges r
    pure (Binary.Interval.hull acc next)) first

/-! ## Graph range transfers

Range propagation never looks at architectures. A model is a graph of registered operators
(`IR.Graph`), and each node is handled by the range transfer of its operator, which
`rangeTransfer?` reads off the constructor of `TorchLean.Op`. This keeps MLPs, convolutional
networks, transformers and other model families on one checker path. Adding a model needs no
certificate traversal of its own, and adding a primitive needs one local transfer. Certificate
generation and checking use the same transfers.
-/

/-- Whether a node's range is a certificate assumption: a graph input, a constant and the random
draws `randUniform` and `bernoulliMask`. -/
def isRangeSource : Node → Bool
  | .input .. | .const _ => true
  | .op ⟨_, _, .randUniform _⟩ _ | .op ⟨_, _, .bernoulliMask _⟩ _ => true
  | .op _ _ => false

/-- Read-only state supplied to one local range transfer. -/
structure RangeContext where
  /-- The checked source assumptions of the certificate. -/
  sources : Array CheckedSourceRange
  /-- The accepted kernel plan of the graph. -/
  plan : AcceptedGraphKernelPlan
  /-- The checked ranges of the earlier nodes, indexed by node id. -/
  ranges : Array CheckedNodeRange

/-- Result computed by one range transfer. -/
abbrev RangeTransferResult := Prod RangeRule (Interval (Binary 8 23))

/-- Executable range transfer of one operation, applied to a node and its id. Proofs read its row
as `SomeTensorEnclosed`, and it has no proof field, so serializable certificates cannot inject
proof evidence. -/
abbrev RangeTransfer := RangeContext → Nat → Node → Except String RangeTransferResult

/-- Standard diagnostic for a contract whose node does not have the inputs of its operation. -/
def arityError (contractName : String) (id : Nat) (node : Node) (expected : String) : String :=
  let count := match node with
    | .op _ refs => refs.size
    | _ => 0
  s!"numerical contract {contractName}: node {id} ({node.tag}) " ++
    s!"expected {expected}, got {count} input(s)"

/-- Shared transfer for one-input operations whose output keeps the input enclosure. -/
def preserveTransfer (contractName : String) (context : RangeContext) (id : Nat) :
    Node → Except String RangeTransferResult
  | .op _ #[r] => do
      let enclosure <- parentRange context.ranges r
      pure (.preserve r.node, enclosure)
  | node => throw (arityError contractName id node "one input")

/-- Shared source-node transfer: the node's interval is its source assumption in the certificate,
read with `findSource`. -/
def sourceContract : RangeTransfer := fun context id _ => do
  let assumption <- findSource context.sources id
  pure (.source, assumption.enclosure)

/-- Transfer of a pointwise binary interval operation. -/
def binaryContract (op : String) (rule : Nat -> Nat -> RangeRule)
    (transfer : Interval (Binary 8 23) -> Interval (Binary 8 23) -> Interval (Binary 8 23)) :
    RangeTransfer := fun context id node =>
  match node with
  | .op _ #[left, right] => do
      let a <- parentRange context.ranges left
      let b <- parentRange context.ranges right
      pure (rule left.node right.node, transfer a b)
  | _ => throw (arityError op id node "two inputs")

/-- Transfer of an operation whose output is enclosed by the hull of its inputs. -/
def hullContract (op : String) : RangeTransfer := fun context id node =>
  match node with
  | .op _ refs => do
      let enclosure <- hullParents context.ranges refs
      pure (.hull (refs.map (·.node)), enclosure)
  | _ => throw (arityError op id node "inputs")

/-- The reduction, window size and whether any padding is set, of a pooling operator. -/
def poolParams? : Operator → Option (Tensor.PoolOp × Nat × Bool)
  | ⟨_, _, @Op.pool s op dims kernel _ padLow padHigh _ _⟩ =>
      some (op, (s.poolWindow dims kernel).size, padLow.any (· != 0) || padHigh.any (· != 0))
  | _ => none

/-- The reduction, window size and padding of a pooling node, with its one input. -/
def poolInput? : Node → Option (Tensor.PoolOp × Nat × Bool × Ref)
  | .op o refs =>
      if h : refs.size = 1 then
        (poolParams? o).map fun (op, count, padded) => (op, count, padded, refs[0])
      else none
  | _ => none

/-- Pooling (`Op.pool`): a maximum or minimum keeps the input enclosure, joined with zero when
padding is set, and an average is the fixed-left mean of its window over that enclosure, under the
fixed-left policy. A sum pool has no transfer. -/
def poolContract : RangeTransfer := fun context id node =>
  match poolInput? node with
  | some (op, count, padded, r) => do
      -- Every window's first cell lies in the padded input
      -- (`Shape.kernel_ne_zero_and_mul_stride_lt_of_lt_poolExtent`), so without padding
      -- every window reads an input entry. A maximum or minimum selects an entry, or zero for a
      -- window of padding only. An average adds its window from zero, padding adding zero, and
      -- divides by a count of at least the input cells it read (zero over the reals for a count
      -- of zero), so it is a mean of values of the joined range, within `meanLeftRange`.
      let input <- parentRange context.ranges r
      let source :=
        if padded then
          Binary.Interval.hull input
            (Binary.Interval.point (ExecFloat.Binary.zero false : Binary 8 23))
        else
          input
      match op with
      | .max | .min =>
          pure (if padded then (.hullZero r.node, source) else (.preserve r.node, input))
      | .avg _ => do
          if count = 0 then
            throw s!"numerical certificate: node {id} has an empty pooling window"
          pure (.averageWindowLeft r.node count padded, meanLeftRange count source)
      | .sum => throw s!"numerical certificate: node {id}: sum pooling has no range transfer"
  | none => throw (arityError "pooling" id node "one input")

/-- Reciprocal transfer, which rejects an input range that contains zero. -/
def inverseContract : RangeTransfer := fun context id node =>
  match node with
  | .op _ #[r] => do
      let enclosure <- parentRange context.ranges r
      if (Binary.Interval.containsZero enclosure) then
        throw s!"numerical certificate: node {id} reciprocal range contains zero"
      pure (.inv r.node, Binary.Interval.inv enclosure)
  | _ => throw (arityError "reciprocal" id node "one input")

/-- Sum (`Op.reduce .sum`) and mean (`Op.mean`) transfer over any dimensions: each output entry
folds the entries of its fiber, whose count is the product of the extents of the listed
dimensions. -/
def reductionContract (mean : Bool) : RangeTransfer := fun context id node => do
  let (r, count) <- match node, mean with
    | .op ⟨_, _, @Op.reduce s .sum dims _ _⟩ #[r], false =>
        pure (r, (s.selectAxes dims).size)
    | .op ⟨_, _, @Op.mean s dims _⟩ #[r], true => pure (r, (s.selectAxes dims).size)
    | _, _ => throw s!"numerical certificate: node {id} is not a fixed-left reduction"
  let input <- parentRange context.ranges r
  if mean then
    if count = 0 then
      throw s!"numerical certificate: node {id} cannot certify a mean over no entries"
    pure (.meanLeft r.node count, meanLeftRange count input)
  else
    pure (.sumLeft r.node count, sumLeftRange count input)

/-- Matrix multiplication transfer using the selected fixed-left accumulation schedule over the
operator's inner extent. -/
def matmulContract : RangeTransfer := fun context id node =>
  match node with
  | .op ⟨_, _, @Op.matmul _ _ _ _ n _ _ _ _ _ _ _⟩ #[left, right] => do
      let a <- parentRange context.ranges left
      let b <- parentRange context.ranges right
      let product := Binary.Interval.mul a b
      pure (.matmulLeft left.node right.node n, sumLeftRange n product)
  | _ => throw (arityError "matrix multiplication" id node "two inputs")

/-- Grouped contractions require the reference's checked fixed-left-or-exact policy. The transfer
rejects nodes with an implementation-defined native schedule or with no numerical policy. -/
def scaledMatmulContract : RangeTransfer := fun context id node => do
  match nodePolicy context.plan id with
  | some policy =>
      unless policy.reduction == .fixedLeftOrExact do
        throw s!"numerical certificate: node {id}: fixedLeftOrExact is required"
  | none => throw s!"numerical certificate: node {id}: missing numerical policy"
  match node with
  | .op ⟨_, _, @Op.scaledMatmul _ _ _ _ n _ _ _ _ _ _ _⟩ #[left, right, scale] => do
      let a ← parentRange context.ranges left
      let b ← parentRange context.ranges right
      let c ← parentRange context.ranges scale
      let some enclosure := scaledContractionRange? n a b c
        | throw s!"numerical certificate: node {id}: no finite scaled contraction enclosure"
      pure (.scaledMatmul left.node right.node scale.node n, enclosure)
  | _ => throw (arityError "scaled matrix contraction" id node "three inputs")

/-- Mean-squared-error transfer with nonnegativity restored after dependent squaring. -/
def mseContract : RangeTransfer := fun context id node =>
  match node with
  | .op ⟨_, _, @Op.mseLoss s⟩ #[prediction, target] => do
      let y <- parentRange context.ranges prediction
      let t <- parentRange context.ranges target
      let residual := Binary.Interval.sub y t
      -- The two residual occurrences are dependent. Generic interval multiplication forgets
      -- that dependency, and the proved ReLU transfer restores nonnegativity of the square.
      let squared := (Binary.Interval.relu (Binary.Interval.mul residual residual))
      let count := s.size
      let enclosure :=
        if count = 0 then Binary.Interval.point (ExecFloat.Binary.zero false :
          Binary 8 23)
        else meanLeftRange count squared
      pure (.mseLeft prediction.node target.node count, enclosure)
  | _ => throw (arityError "mean squared error" id node "two inputs")

/-- LayerNorm transfer (`Op.layerNorm`) over the last axis, with scale, bias and epsilon inputs:
the centered input divided by the square root of the variance plus epsilon, times the scale, plus
the bias. -/
def layerNormContract : RangeTransfer := fun context id node =>
  match node with
  | .op ⟨_, _, @Op.layerNorm _ normalizedSize _ _ _ _ _ _⟩
      #[parent, scale, bias, epsilon] => do
      let input <- parentRange context.ranges parent
      let gamma <- parentRange context.ranges scale
      let beta <- parentRange context.ranges bias
      let eps <- parentRange context.ranges epsilon
      let mean := meanLeftRange normalizedSize input
      let centered := Binary.Interval.sub input mean
      let squared := (Binary.Interval.relu (Binary.Interval.mul centered centered))
      let variance := meanLeftRange normalizedSize squared
      let stabilized := Binary.Interval.add variance eps
      unless nonnegativeEndpoint stabilized.lo && nonnegativeEndpoint stabilized.hi do
        throw s!"numerical certificate: node {id} layer_norm variance range became negative"
      let denominator := (Binary.Interval.sqrt stabilized)
      if (Binary.Interval.containsZero denominator) then
        throw s!"numerical certificate: node {id} layer_norm denominator may be zero"
      let normalized := Binary.Interval.div centered denominator
      pure (.layerNormLeft parent.node scale.node bias.node epsilon.node normalizedSize,
        Binary.Interval.add (Binary.Interval.mul normalized gamma) beta)
  | _ => throw (arityError "layer normalization" id node "four inputs")

/-- Softmax transfer: exact-real outputs lie in the unit interval on every nonempty axis. -/
def softmaxContract : RangeTransfer := fun context id node =>
  match node with
  | .op ⟨_, _, @Op.softmax s dim⟩ #[r] => do
      let _ <- parentNodeRange context.ranges r
      if s.extent dim = 0 then
        throw s!"numerical certificate: node {id} cannot certify softmax on an empty axis"
      pure (.softmaxUnit r.node dim.val, unitInterval)
  | _ => throw (arityError "softmax" id node "one input")

/-- ReLU interval transfer. -/
def reluContract : RangeTransfer := fun context id node =>
  match node with
  | .op _ #[r] => do
      let input <- parentRange context.ranges r
      pure (.relu r.node, (Binary.Interval.relu input))
  | _ => throw (arityError "ReLU" id node "one input")

/-- Real SiLU lies between its input and zero. IEEE replay still checks each rounded result. -/
def siluContract : RangeTransfer := fun context id node =>
  match node with
  | .op _ #[r] => do
      let input ← parentRange context.ranges r
      pure (.hullZero r.node, Binary.Interval.hull input
        (Binary.Interval.point (ExecFloat.Binary.zero false : Binary 8 23)))
  | _ => throw (arityError "SiLU" id node "one input")

/-- Absolute-value interval transfer. -/
def absContract : RangeTransfer := fun context id node =>
  match node with
  | .op _ #[r] => do
      let input <- parentRange context.ranges r
      pure (.abs r.node, (Binary.Interval.abs input))
  | _ => throw (arityError "absolute value" id node "one input")

/-- Square-root transfer with a checked nonnegative domain. -/
def sqrtContract : RangeTransfer := fun context id node =>
  match node with
  | .op _ #[r] => do
      let input <- parentRange context.ranges r
      if nonnegativeEndpoint input.lo && nonnegativeEndpoint input.hi then
        pure (.sqrtNonnegative r.node, (Binary.Interval.sqrt input))
      else
        throw s!"numerical certificate: node {id} square-root range contains negative values"
  | _ => throw (arityError "square root" id node "one input")

/-- Transfer of a bounded transcendental operation: its codomain enclosure. -/
def boundedUnaryContract (op : String) (rule : Nat -> RangeRule)
    (enclosure : Interval (Binary 8 23)) : RangeTransfer := fun _ id node =>
  match node with
  | .op _ #[r] => pure (rule r.node, enclosure)
  | _ => throw (arityError op id node "one input")

/-- The range transfer of a node, read off its constructor of `TorchLean.Op`: the source transfer
for a range source, one transfer per supported operator, grouped by operation semantics and never
by architecture, and `none` for an operator without one. -/
def rangeTransfer? : Node → Option RangeTransfer
  | .input .. | .const _ => some sourceContract
  | .op ⟨_, _, .randUniform _⟩ _ | .op ⟨_, _, .bernoulliMask _⟩ _ => some sourceContract
  | .op ⟨_, _, .detach ..⟩ _ => some (preserveTransfer "detach")
  | .op ⟨_, _, .permute ..⟩ _ => some (preserveTransfer "permute")
  | .op ⟨_, _, .broadcast ..⟩ _ => some (preserveTransfer "broadcast")
  | .op ⟨_, _, .reshape ..⟩ _ => some (preserveTransfer "reshape")
  | .op ⟨_, _, .slice ..⟩ _ => some (preserveTransfer "slice")
  | .op ⟨_, _, .pool ..⟩ _ => some poolContract
  | .op ⟨_, _, .add ..⟩ _ => some (binaryContract "add" .add Binary.Interval.add)
  | .op ⟨_, _, .sub ..⟩ _ => some (binaryContract "sub" .sub Binary.Interval.sub)
  | .op ⟨_, _, .mul ..⟩ _ => some (binaryContract "mul" .mul Binary.Interval.mul)
  | .op ⟨_, _, .reciprocal ..⟩ _ => some inverseContract
  | .op ⟨_, _, .max ..⟩ _ => some (hullContract "max")
  | .op ⟨_, _, .min ..⟩ _ => some (hullContract "min")
  | .op ⟨_, _, .cat ..⟩ _ => some (hullContract "cat")
  | .op ⟨_, _, .reduce ..⟩ _ => some (reductionContract false)
  | .op ⟨_, _, .mean ..⟩ _ => some (reductionContract true)
  | .op ⟨_, _, .matmul ..⟩ _ => some matmulContract
  | .op ⟨_, _, .scaledMatmul ..⟩ _ => some scaledMatmulContract
  | .op ⟨_, _, .mseLoss ..⟩ _ => some mseContract
  | .op ⟨_, _, .layerNorm ..⟩ _ => some layerNormContract
  | .op ⟨_, _, .softmax ..⟩ _ => some softmaxContract
  | .op ⟨_, _, .relu ..⟩ _ => some reluContract
  | .op ⟨_, _, .silu ..⟩ _ => some siluContract
  | .op ⟨_, _, .abs ..⟩ _ => some absContract
  | .op ⟨_, _, .sqrt ..⟩ _ => some sqrtContract
  | .op ⟨_, _, .sigmoid ..⟩ _ => some (boundedUnaryContract "sigmoid" .unitBound unitInterval)
  | .op ⟨_, _, .tanh ..⟩ _ =>
      some (boundedUnaryContract "tanh" .signedUnitBound signedUnitInterval)
  | .op ⟨_, _, .sin ..⟩ _ => some (boundedUnaryContract "sin" .signedUnitBound signedUnitInterval)
  | .op ⟨_, _, .cos ..⟩ _ => some (boundedUnaryContract "cos" .signedUnitBound signedUnitInterval)
  | .op _ _ => none

/-- Whether a node counts as a reduction folded from the left: matrix products, every `Op.reduce`
(only the sum has a transfer), means, average pooling, mean squared error and LayerNorm.
`deriveNodeRange` admits such a node only after `fixedLeftGuard` accepts its selected
capsule. -/
def fixedLeftNode : Node → Bool
  | .op ⟨_, _, .matmul ..⟩ _ | .op ⟨_, _, .reduce ..⟩ _ | .op ⟨_, _, .mean ..⟩ _
  | .op ⟨_, _, .pool (.avg _) ..⟩ _ | .op ⟨_, _, .mseLoss ..⟩ _
  | .op ⟨_, _, .layerNorm ..⟩ _ => true
  | _ => false

/-- One graph node whose operator has no range transfer. -/
structure MissingContract where
  /-- The id of the node in the graph. -/
  nodeId : Nat
  /-- The tag of the node's operator. -/
  operation : String
  deriving Repr

/-- Architecture-independent coverage report of a graph of registered operators. -/
structure CoverageReport where
  /-- The number of nodes in the graph. -/
  nodeCount : Nat
  /-- The number of nodes whose operator has a range transfer. -/
  coveredCount : Nat
  /-- The nodes whose operator has no range transfer. -/
  missing : Array MissingContract
  deriving Repr

/-- Inspect range-transfer coverage without attempting interval propagation. -/
def coverage (graph : Graph) : CoverageReport :=
  let missing := graph.nodes.zipIdx.filterMap fun (node, id) =>
    if (rangeTransfer? node).isSome then none
    else some { nodeId := id, operation := node.tag }
  { nodeCount := graph.nodes.size
    coveredCount := graph.nodes.size - missing.size
    missing }

/-- Reject a graph before propagation when any primitive lacks a range transfer. -/
def requireCoverage (graph : Graph) : Except String CoverageReport := do
  let report := coverage graph
  if report.missing.isEmpty then
    pure report
  else
    throw (s!"numerical certificate: no range transfer covers graph nodes " ++
      s!"{repr report.missing}")

/-- Compute the range of node `node` at id `id` with the range transfer of its operator, after
`fixedLeftGuard` for a node that folds a reduction from the left (`fixedLeftNode`). -/
def deriveNodeRange (sources : Array CheckedSourceRange) (plan : AcceptedGraphKernelPlan)
    (ranges : Array CheckedNodeRange) (id : Nat) (node : Node) :
    Except String RangeTransferResult :=
  match rangeTransfer? node with
  | none => throw s!"numerical certificate: node {id} ({node.tag}) has no range transfer"
  | some transfer =>
      if fixedLeftNode node then
        fixedLeftGuard plan id node >>= fun _ =>
          transfer { sources, plan, ranges } id node
      else
        transfer { sources, plan, ranges } id node

/-- Reject source assumptions that do not name a source node of the checked graph
(`isRangeSource`).

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
        unless isRangeSource node do
          throw (s!"numerical certificate: node {source.nodeId} ({node.tag}) does not " ++
            "consume a source range")

/-- Construct and validate the canonical range trace: the graph's structural check, range-transfer
coverage and source ownership, then one row per node, at the shape of its one tensor of scalars. -/
def buildRangeTrace (graph : Graph) (sources : Array CheckedSourceRange)
    (plan : AcceptedGraphKernelPlan) :
    Except String (Array CheckedNodeRange) := do
  graph.check
  let _ <- requireCoverage graph
  checkSourceOwnership graph sources
  let mut ranges : Array CheckedNodeRange := #[]
  for (node, id) in graph.nodes.zipIdx do
    let outShape <- match node.outs with
      | [p] =>
          if p.kind.isScalar then pure p.shape
          else throw s!"numerical certificate: node {id} ({node.tag}) has no scalar output"
      | _ => throw s!"numerical certificate: node {id} ({node.tag}) has more than one output"
    let (rule, enclosure) <- deriveNodeRange sources plan ranges id node
    if h : Binary.Interval.Valid enclosure then
      ranges := ranges.push
        { nodeId := id
          outShape
          rule
          enclosure
          valid := h }
    else
      throw (s!"numerical certificate: node {id} ({node.tag}) produced a " ++
        "non-finite or unordered enclosure")
  pure ranges

/-- Erase validity proofs from a checked trace. -/
def eraseRangeTrace (ranges : Array CheckedNodeRange) : Array NodeRange :=
  ranges.map (fun range => range.toNodeRange)

/-- Compare a canonical checked trace with untrusted raw rows. -/
def sameRangeTrace (checked : Array CheckedNodeRange) (raw : Array NodeRange) : Bool :=
  checked.size == raw.size &&
    (List.finRange checked.size).all (fun i =>
      match checked[i]?, raw[i]? with
      | some expected, some claimed => sameNodeRange expected claimed
      | _, _ => false)

end NumericalCertificate
end TorchLean
