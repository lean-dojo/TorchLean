/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate.Enclosure

/-!
# Numerical certificate contracts

Canonical local range rules, numerical operation contracts, contract registries, and graph range
trace construction. Most users should import
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

/-! ## Canonical local transfer rules -/

/-- How a node enclosure was obtained from source assumptions or earlier nodes.

The rule is recorded to make certificate diagnostics useful. It is not accepted on faith:
`check` reconstructs the rule and endpoints from the graph.
-/
inductive RangeRule where
  | source
  | preserve (parent : Nat)
  | add (left right : Nat)
  | sub (left right : Nat)
  | mul (left right : Nat)
  | inv (parent : Nat)
  | hull (parents : Array Nat)
  | hullZero (parent : Nat)
  | sumLeft (parent count : Nat)
  | meanLeft (parent count : Nat)
  | matmulLeft (left right innerDim : Nat)
  | averageWindowLeft (parent windowSize : Nat) (includesPadding : Bool)
  | mseLeft (prediction target count : Nat)
  | layerNormLeft (parent axis normalizedSize : Nat)
  | softmaxUnit (parent axis : Nat)
  | relu (parent : Nat)
  | abs (parent : Nat)
  | sqrtNonnegative (parent : Nat)
  | unitBound (parent : Nat)
  | signedUnitBound (parent : Nat)
  deriving DecidableEq, Repr

/-- Proof-free data for one graph node's numerical range. -/
structure NodeRange where
  nodeId : Nat
  outShape : Shape
  rule : RangeRule
  enclosure : IEEE32Exec.Interval32
  deriving Repr

/-- A node range whose executable interval has passed the finite/order check. -/
structure CheckedNodeRange extends NodeRange where
  valid : enclosure.Valid

instance : Repr CheckedNodeRange where
  reprPrec r _ := repr r.toNodeRange

/-- Check a dynamic graph value against the declared shape and interval of one certificate row. -/
def dvalWithinRange (range : CheckedNodeRange) (value : Spec.SomeTensor (ExecFloat.Binary 8 23)) :
  Bool :=
  if h : value.shape = range.outShape then
    tensorWithinRange range.enclosure (h ▸ value.tensor)
  else
    false

/-- A real dynamic graph value has the shape declared by a certificate row and is enclosed by its
interval. The equality witness makes the dependent tensor cast explicit. -/
def SomeTensorEnclosed (range : CheckedNodeRange) (value : Spec.SomeTensor Real) : Prop :=
  ∃ h : value.shape = range.outShape,
    TensorEnclosed range.enclosure (h ▸ value.tensor)

/-- Pointwise approximation relation for real and IEEE dynamic graph values at one certificate
row. -/
def SomeTensorErrorLe (range : CheckedNodeRange) (exact : Spec.SomeTensor Real)
    (computed : Spec.SomeTensor (ExecFloat.Binary 8 23)) : Prop :=
  ∃ hexact : exact.shape = range.outShape,
    ∃ hcomputed : computed.shape = range.outShape,
      TensorErrorLe (intervalWidth range.enclosure)
        (hexact ▸ exact.tensor) (hcomputed ▸ computed.tensor)

/-- One successful dynamic replay row yields a pointwise error bound whenever the corresponding
real graph value has the proved enclosure. -/
theorem dval_error_le_of_range_check {range : CheckedNodeRange}
    {exact : Spec.SomeTensor Real} {computed : Spec.SomeTensor (ExecFloat.Binary 8 23)}
    (hexact : SomeTensorEnclosed range exact)
    (hcomputed : dvalWithinRange range computed = true) :
    SomeTensorErrorLe range exact computed := by
  rcases hexact with ⟨hexactShape, hexactRange⟩
  unfold dvalWithinRange at hcomputed
  split at hcomputed
  next hcomputedShape =>
    exact ⟨hexactShape, hcomputedShape,
      tensor_error_le_width_of_check range.valid hexactRange hcomputed⟩
  next _ => simp at hcomputed

/-- Check every value produced by `IR.Graph.denoteAll` against the corresponding certificate row. -/
def executionWithinRanges (ranges : Array CheckedNodeRange)
    (values : Array (Spec.SomeTensor (ExecFloat.Binary 8 23))) : Bool :=
  decide (ranges.size = values.size) &&
    (Array.zipWith dvalWithinRange ranges values).all id

/-- Pointwise relation between two equally sized runtime arrays. -/
def ArraysRelated {α β : Type} (relation : α → β → Prop)
    (xs : Array α) (ys : Array β) : Prop :=
  xs.size = ys.size ∧
    ∀ (i : Nat) (hxs : i < xs.size) (hys : i < ys.size), relation xs[i] ys[i]

/-- A successful replay check is exactly a size match plus a successful check at every node. -/
theorem executionWithinRanges_eq_true_iff
    (ranges : Array CheckedNodeRange) (values : Array (Spec.SomeTensor (ExecFloat.Binary 8 23))) :
    executionWithinRanges ranges values = true ↔
      ArraysRelated (fun range value => dvalWithinRange range value = true) ranges values := by
  constructor
  · intro h
    simp only [executionWithinRanges, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases h with ⟨hsize, hall⟩
    refine ⟨hsize, ?_⟩
    intro i hrange hvalue
    have hzip : i < (Array.zipWith dvalWithinRange ranges values).size := by
      simpa [Array.size_zipWith] using And.intro hrange hvalue
    have hi := (Array.all_eq_true.mp hall) i hzip
    simpa [Array.getElem_zipWith] using hi
  · rintro ⟨hsize, hat⟩
    simp only [executionWithinRanges, Bool.and_eq_true, decide_eq_true_eq]
    refine ⟨hsize, Array.all_eq_true.mpr ?_⟩
    intro i hzip
    have hi : i < ranges.size ∧ i < values.size := by
      simpa [Array.size_zipWith] using hzip
    simpa [Array.getElem_zipWith] using hat i hi.1 hi.2

/-- Graph-wide pointwise approximation evidence, one row per intermediate value. -/
def ExecutionErrorTrace (ranges : Array CheckedNodeRange)
    (exact : Array (Spec.SomeTensor Real))
    (computed : Array (Spec.SomeTensor (ExecFloat.Binary 8 23))) : Prop :=
  ranges.size = exact.size ∧ ranges.size = computed.size ∧
    ∀ (i : Nat) (hrange : i < ranges.size) (hexact : i < exact.size)
      (hcomputed : i < computed.size),
      SomeTensorErrorLe ranges[i] exact[i] computed[i]

/-- Compose real enclosure proofs and successful IEEE replay checks into an error trace. -/
theorem execution_error_trace_of_check
    {ranges : Array CheckedNodeRange} {exact : Array (Spec.SomeTensor Real)}
    {computed : Array (Spec.SomeTensor (ExecFloat.Binary 8 23))}
    (hexact : ArraysRelated SomeTensorEnclosed ranges exact)
    (hcomputed : executionWithinRanges ranges computed = true) :
    ExecutionErrorTrace ranges exact computed := by
  rcases hexact with ⟨hexactSize, hexactAt⟩
  rcases (executionWithinRanges_eq_true_iff ranges computed).mp hcomputed with
    ⟨hcomputedSize, hcomputedAt⟩
  refine ⟨hexactSize, hcomputedSize, ?_⟩
  intro i hrange hexactIndex hcomputedIndex
  exact dval_error_le_of_range_check
    (hexactAt i hrange hexactIndex) (hcomputedAt i hrange hcomputedIndex)

/-- Compare a checked canonical row with untrusted raw certificate data. -/
def sameNodeRange (checked : CheckedNodeRange) (raw : NodeRange) : Bool :=
  decide (checked.nodeId = raw.nodeId) &&
    decide (checked.outShape = raw.outShape) &&
    decide (checked.rule = raw.rule) &&
    sameIntervalBits checked.enclosure raw.enclosure

/-- Read a previously checked parent enclosure. Graph well-formedness guarantees that successful
lookups refer only to earlier rows; the explicit error still protects this API when called alone. -/
def parentRange (ranges : Array CheckedNodeRange) (nodeId : Nat) :
    Except String IEEE32Exec.Interval32 :=
  match ranges[nodeId]? with
  | some range => pure range.enclosure
  | none => throw s!"numerical certificate: missing range for parent node {nodeId}"

/-- Read the complete checked row for a parent node. -/
def parentNodeRange (ranges : Array CheckedNodeRange) (nodeId : Nat) :
    Except String CheckedNodeRange :=
  match ranges[nodeId]? with
  | some range => pure range
  | none => throw s!"numerical certificate: missing range for parent node {nodeId}"

/-- Outward-rounded left-fold range for a sum of `count` values from one enclosure. The initial
point interval at positive zero matches `Tensor.sumSpec`. -/
def sumLeftRange (count : Nat) (range : IEEE32Exec.Interval32) : IEEE32Exec.Interval32 :=
  (List.range count).foldl
    (fun acc _ => IEEE32Exec.Interval32.add acc range)
    (IEEE32Exec.Interval32.point (ExecFloat.Binary.zero false : ExecFloat.Binary 8 23))

/-- Left-fold mean range, using the same binary32 conversion of the divisor as the tensor
context. -/
def meanLeftRange (count : Nat) (range : IEEE32Exec.Interval32) : IEEE32Exec.Interval32 :=
  IEEE32Exec.Interval32.div (sumLeftRange count range)
    (IEEE32Exec.Interval32.point (count : ExecFloat.Binary 8 23))

/-- Numerical policy selected for a runtime-relevant graph node. -/
def nodeNumericalPolicy (plan : AcceptedGraphKernelPlan) (nodeId : Nat) : Option NumericalPolicy :=
  (plan.graphPlan.kernels.find? (fun kernel => kernel.nodeId == nodeId)).map
    (fun kernel => kernel.capsule.numericalPolicy)

/-- Reductions are propagated only when the selected capsule promises the same fixed left fold as
the canonical tensor semantics. Other schedules need the order-independent reduction bound from
`NN.Proofs.RuntimeApprox.Reductions.IEEE32` and are rejected here rather than mislabeled as
deterministic. -/
def requireFixedLeftReduction (plan : AcceptedGraphKernelPlan) (node : Node) : Except String Unit :=
  match nodeNumericalPolicy plan node.id with
  | some policy =>
      if policy.reduction = .fixedLeft then
        pure ()
      else
        throw (s!"numerical certificate: node {node.id} ({node.kind.describe}) uses reduction " ++
          s!"policy {repr policy.reduction}; fixedLeft is required by this transfer")
  | none =>
      throw (s!"numerical certificate: node {node.id} ({node.kind.describe}) has no backend " ++
        "numerical policy")

/-- Inner accumulation length for the rank-2 and batched rank-3 matrix products implemented by
`IR.Graph.denoteAll`. The graph shape checker has already validated matching dimensions; retaining
the checks here gives callers of `deriveNodeRange` a precise error instead of relying on that
ambient invariant. -/
def matmulInnerDim (left right : Shape) : Except String Nat :=
  match left, right with
  | .dim _ (.dim n .scalar), .dim n' (.dim _ .scalar) =>
      if n = n' then pure n else throw s!"matmul inner dimensions differ: {n} vs {n'}"
  | .dim batch (.dim _ (.dim n .scalar)),
      .dim batch' (.dim n' (.dim _ .scalar)) =>
      if batch = batch' then
        if n = n' then pure n else throw s!"batched matmul inner dimensions differ: {n} vs {n'}"
      else
        throw s!"batched matmul batch dimensions differ: {batch} vs {batch'}"
  | _, _ => throw s!"unsupported matmul shapes: {repr left} and {repr right}"

/-- Hull of a nonempty array of parent ranges. -/
def hullParents (ranges : Array CheckedNodeRange) (parents : Array Nat) :
    Except String IEEE32Exec.Interval32 := do
  let parent <- match parents[0]? with
    | some parent => pure parent
    | none => throw "numerical certificate: an interval hull requires at least one parent"
  let first <- parentRange ranges parent
  (parents.extract 1 parents.size).foldlM (fun acc id => do
    let next <- parentRange ranges id
    pure (IEEE32Exec.Interval32.hull acc next)) first

/-! ## Graph range contracts

Architectures do not participate in range propagation directly. They lower to `NN.IR.Graph`, and
each graph node is handled by a reusable operation contract. This keeps MLPs, convolutional
networks, transformers, and future model families on one checker path: adding a model requires no
new certificate traversal, while adding a genuinely new primitive requires one local contract.

The registry is an explicit value rather than global mutable state. Certificate generation and
checking therefore use the same inspectable rule set, and downstream projects may extend it
without changing TorchLean's graph walker.
-/

/-- Stable key for a numerical range contract.

Input-like nodes share the `source` contract, `detach` uses the structural identity contract, and
runtime operations use the same `BackendOp` vocabulary as kernel capsules and execution plans.
-/
inductive NumericalOpKey where
  | source
  | structural
  | wholeSum
  | maxPool
  | maxPoolPad
  | averagePool
  | averagePoolPad
  | backend (op : BackendOp)
  | unclassified
  deriving DecidableEq, Repr

/-- Classify an IR operation for numerical-contract lookup. -/
def numericalOpKey : OpKind -> NumericalOpKey
  | .input | .const _ | .randUniform _ | .bernoulliMask _ => .source
  | .detach => .structural
  | .sum => .wholeSum
  | .maxPool config =>
      if (Tensor.to config.padding (List Nat)).any (fun padding => padding != 0) then
        .maxPoolPad
      else
        .maxPool
  | .avgPool config =>
      if (Tensor.to config.padding (List Nat)).any (fun padding => padding != 0) then
        .averagePoolPad
      else
        .averagePool
  | kind =>
      match NN.Backend.IR.op? kind with
      | some op => .backend op
      | none => .unclassified

/-- Read-only state supplied to one local range transfer. -/
structure NumericalRangeContext where
  sources : Array CheckedSourceRange
  plan : AcceptedGraphKernelPlan
  ranges : Array CheckedNodeRange

/-- Result computed by one numerical operation contract. -/
abbrev RangeTransferResult := Prod RangeRule IEEE32Exec.Interval32

/-- Executable range transfer for one operation family.

The proof-facing meaning of the resulting row remains `SomeTensorEnclosed`; local soundness lemmas
for interval arithmetic and NF approximation are kept in their mathematical modules. The contract
contains only executable dispatch and a stable key, so serializable certificates cannot inject
proof evidence.
-/
structure GraphRangeContract where
  key : NumericalOpKey
  name : String
  derive : NumericalRangeContext -> Node -> Except String RangeTransferResult

/-- Deterministic registry used by graph certificate generation and replay. -/
structure GraphRangeRegistry where
  name : String
  contracts : Array GraphRangeContract

namespace GraphRangeRegistry

/-- Empty named registry for downstream composition. -/
def empty (name : String) : GraphRangeRegistry := ⟨name, #[]⟩

/-- Find the unique contract associated with a numerical operation key. -/
def find? (registry : GraphRangeRegistry) (key : NumericalOpKey) :
    Option GraphRangeContract :=
  registry.contracts.find? (fun contract => decide (contract.key = key))

/-- Add one contract, rejecting duplicate keys so dispatch never depends on registration order. -/
def register (registry : GraphRangeRegistry) (contract : GraphRangeContract) :
    Except String GraphRangeRegistry :=
  match registry.find? contract.key with
  | some previous =>
      throw (s!"numerical contract: duplicate key {repr contract.key} " ++
        s!"({previous.name}, {contract.name})")
  | none => pure { registry with contracts := registry.contracts.push contract }

/-- Build a registry while checking key uniqueness. -/
def ofArray (name : String) (contracts : Array GraphRangeContract) :
    Except String GraphRangeRegistry :=
  contracts.foldlM register (empty name)

end GraphRangeRegistry

/-- One graph node for which a numerical registry has no local transfer. -/
structure MissingNumericalContract where
  nodeId : Nat
  operation : String
  key : NumericalOpKey
  deriving Repr

/-- Architecture-independent coverage report obtained after lowering a model to `NN.IR.Graph`. -/
structure NumericalCoverageReport where
  registryName : String
  nodeCount : Nat
  coveredCount : Nat
  missing : Array MissingNumericalContract
  deriving Repr

/-- Inspect contract coverage without attempting interval propagation. -/
def numericalCoverage (registry : GraphRangeRegistry) (graph : Graph) :
    NumericalCoverageReport :=
  let missing := graph.nodes.filterMap fun node =>
    let key := numericalOpKey node.kind
    if registry.find? key |>.isSome then none
    else some { nodeId := node.id, operation := node.kind.describe, key }
  { registryName := registry.name
    nodeCount := graph.nodes.size
    coveredCount := graph.nodes.size - missing.size
    missing }

/-- Reject a graph before propagation when any primitive lacks a numerical contract. -/
def requireNumericalCoverage (registry : GraphRangeRegistry) (graph : Graph) :
    Except String NumericalCoverageReport := do
  let report := numericalCoverage registry graph
  if report.missing.isEmpty then
    pure report
  else
    throw (s!"numerical certificate: registry {registry.name} does not cover graph nodes " ++
      s!"{repr report.missing}")

/-- Standard diagnostic for a contract whose graph arity does not match its operation. -/
def arityError (contractName : String) (node : Node) (expected : String) : String :=
  s!"numerical contract {contractName}: node {node.id} ({node.kind.describe}) " ++
    s!"expected {expected}, got {node.parents.size} parent(s)"

/-- Shared source-node contract. The source interval remains an explicit certificate assumption. -/
def sourceContract : GraphRangeContract where
  key := .source
  name := "source"
  derive := fun context node => do
    let assumption <- findSource context.sources node.id
    pure (.source, assumption.enclosure)

/-- Structural identity used by `detach`. -/
def structuralContract : GraphRangeContract where
  key := .structural
  name := "structural identity"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError "structural identity" node "one parent")

/-- Reusable contract constructor for value-preserving graph operations. -/
def preserveContract (op : BackendOp) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} value preservation"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError op.name node "one parent")

/-- Reusable contract constructor for pointwise binary interval operations. -/
def binaryContract (op : BackendOp) (rule : Nat -> Nat -> RangeRule)
    (transfer : IEEE32Exec.Interval32 -> IEEE32Exec.Interval32 -> IEEE32Exec.Interval32) :
    GraphRangeContract where
  key := .backend op
  name := s!"{op.name} binary transfer"
  derive := fun context node =>
    match node.parents with
    | #[left, right] => do
        let a <- parentRange context.ranges left
        let b <- parentRange context.ranges right
        pure (rule left right, transfer a b)
    | _ => throw (arityError op.name node "two parents")

/-- Reusable contract for operations whose output is enclosed by the hull of their parents. -/
def hullContract (op : BackendOp) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} hull"
  derive := fun context node => do
    let enclosure <- hullParents context.ranges node.parents
    pure (.hull node.parents, enclosure)


/-- Max pooling without padding selects existing values and therefore preserves the input hull. -/
def maxPoolContract : GraphRangeContract where
  key := .maxPool
  name := "max-pool value preservation"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let enclosure <- parentRange context.ranges parent
        pure (.preserve parent, enclosure)
    | _ => throw (arityError "max pool" node "one parent")

/-- Padded max pooling may additionally select the padding value zero. -/
def maxPoolPadContract : GraphRangeContract where
  key := .maxPoolPad
  name := "padded max-pool hull"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let input <- parentRange context.ranges parent
        let enclosure := IEEE32Exec.Interval32.hull input
          (IEEE32Exec.Interval32.point (ExecFloat.Binary.zero false : ExecFloat.Binary 8 23))
        pure (.hullZero parent, enclosure)
    | _ => throw (arityError "padded max pool" node "one parent")

/-- Shared average-pooling contract constructor. -/
def averagePoolContract (padded : Bool) : GraphRangeContract where
  key := if padded then .averagePoolPad else .averagePool
  name := if padded then "padded average-pool fixed-left transfer"
    else "average-pool fixed-left transfer"
  derive := fun context node =>
    match node.kind, node.parents with
    | .avgPool config, #[parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentRange context.ranges parent
        let count := (Tensor.to config.kernel (List Nat)).foldl (fun size extent => size * extent) 1
        if count = 0 then
          throw s!"numerical certificate: node {node.id} has an empty average-pooling window"
        let hasPadding := (Tensor.to config.padding (List Nat)).any (fun padding => padding != 0)
        let source :=
          if hasPadding then
            IEEE32Exec.Interval32.hull input
              (IEEE32Exec.Interval32.point (ExecFloat.Binary.zero false : ExecFloat.Binary 8 23))
          else
            input
        pure (.averageWindowLeft parent count hasPadding, meanLeftRange count source)
    | _, _ => throw (arityError "average pool" node "one parent")

/-- Reciprocal contract with an explicit nonzero-domain check. -/
def inverseContract : GraphRangeContract where
  key := .backend .inv
  name := "reciprocal"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let enclosure <- parentRange context.ranges parent
        if enclosure.containsZero then
          throw s!"numerical certificate: node {node.id} reciprocal range contains zero"
        pure (.inv parent, IEEE32Exec.Interval32.inv enclosure)
    | _ => throw (arityError "reciprocal" node "one parent")

/-- Whole-tensor fixed-left sum contract. -/
def sumContract : GraphRangeContract where
  key := .wholeSum
  name := "fixed-left sum"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count := input.outShape.size
        pure (.sumLeft parent count, sumLeftRange count input.enclosure)
    | _ => throw (arityError "sum" node "one parent")

/-- Axis reduction contract shared by sum and mean. -/
def axisReductionContract (mean : Bool) : GraphRangeContract where
  key := .backend (if mean then .reduceMean else .reduceSum)
  name := if mean then "fixed-left axis mean" else "fixed-left axis sum"
  derive := fun context node =>
    match node.kind, node.parents with
    | .reduceSum axis, #[parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none =>
              throw s!"numerical certificate: node {node.id} has invalid reduction axis {axis}"
        pure (.sumLeft parent count, sumLeftRange count input.enclosure)
    | .reduceMean axis, #[parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none =>
              throw s!"numerical certificate: node {node.id} has invalid reduction axis {axis}"
        if count = 0 then
          throw s!"numerical certificate: node {node.id} cannot certify a mean over an empty axis"
        pure (.meanLeft parent count, meanLeftRange count input.enclosure)
    | _, _ => throw (arityError "axis reduction" node "one parent")

/-- Matrix multiplication contract using the selected fixed-left accumulation schedule. -/
def matmulContract : GraphRangeContract where
  key := .backend .matmul
  name := "fixed-left matrix multiplication"
  derive := fun context node =>
    match node.parents with
    | #[left, right] => do
        requireFixedLeftReduction context.plan node
        let a <- parentNodeRange context.ranges left
        let b <- parentNodeRange context.ranges right
        let innerDim <- matmulInnerDim a.outShape b.outShape
        let product := IEEE32Exec.Interval32.mul a.enclosure b.enclosure
        pure (.matmulLeft left right innerDim, sumLeftRange innerDim product)
    | _ => throw (arityError "matrix multiplication" node "two parents")

/-- Mean-squared-error contract with nonnegativity restored after dependent squaring. -/
def mseContract : GraphRangeContract where
  key := .backend .mseLoss
  name := "mean squared error"
  derive := fun context node =>
    match node.parents with
    | #[prediction, target] => do
        requireFixedLeftReduction context.plan node
        let y <- parentNodeRange context.ranges prediction
        let t <- parentNodeRange context.ranges target
        if y.outShape != t.outShape then
          throw s!"numerical certificate: node {node.id} MSE parents have different shapes"
        let residual := IEEE32Exec.Interval32.sub y.enclosure t.enclosure
        -- The two residual occurrences are dependent. Generic interval multiplication forgets
        -- that dependency; the proved ReLU transfer restores nonnegativity of the square.
        let squared := (IEEE32Exec.Interval32.mul residual residual).relu
        let count := y.outShape.size
        let enclosure :=
          if count = 0 then IEEE32Exec.Interval32.point (ExecFloat.Binary.zero false :
            ExecFloat.Binary 8 23)
          else meanLeftRange count squared
        pure (.mseLeft prediction target count, enclosure)
    | _ => throw (arityError "mean squared error" node "two parents")

/-- Pure LayerNorm contract over an arbitrary normalized suffix. -/
def layerNormContract : GraphRangeContract where
  key := .backend .layerNorm
  name := "layer normalization"
  derive := fun context node =>
    match node.kind, node.parents with
    | .layernorm axis, #[parent] => do
        requireFixedLeftReduction context.plan node
        let input <- parentNodeRange context.ranges parent
        let (_, normalizedSize) <-
          match OpContracts.layerNormMatrixDims axis input.outShape with
          | .ok dims => pure dims
          | .error message =>
              throw s!"numerical certificate: node {node.id} layernorm: {message}"
        if normalizedSize = 0 then
          throw s!"numerical certificate: node {node.id} cannot normalize an empty suffix"
        let mean := meanLeftRange normalizedSize input.enclosure
        let centered := IEEE32Exec.Interval32.sub input.enclosure mean
        let squared := (IEEE32Exec.Interval32.mul centered centered).relu
        let variance := meanLeftRange normalizedSize squared
        let epsilon : ExecFloat.Binary 8 23 := TorchLean.normalizationEpsilon
        let stabilized := IEEE32Exec.Interval32.add variance
          (IEEE32Exec.Interval32.point epsilon)
        if nonnegativeEndpoint stabilized.lo && nonnegativeEndpoint stabilized.hi then
          let denominator := stabilized.sqrt
          if denominator.containsZero then
            throw s!"numerical certificate: node {node.id} layernorm denominator may be zero"
          pure (.layerNormLeft parent axis normalizedSize,
            IEEE32Exec.Interval32.div centered denominator)
        else
          throw s!"numerical certificate: node {node.id} layernorm variance range became negative"
    | _, _ => throw (arityError "layer normalization" node "one parent")

/-- Softmax contract: exact-real outputs lie in the unit interval on every nonempty axis. -/
def softmaxContract : GraphRangeContract where
  key := .backend .softmax
  name := "softmax unit interval"
  derive := fun context node =>
    match node.kind, node.parents with
    | .softmax axis, #[parent] => do
        let input <- parentNodeRange context.ranges parent
        let count <- match input.outShape.getDim axis with
          | some count => pure count
          | none => throw s!"numerical certificate: node {node.id} has invalid softmax axis {axis}"
        if count = 0 then
          throw s!"numerical certificate: node {node.id} cannot certify softmax on an empty axis"
        pure (.softmaxUnit parent axis, unitInterval)
    | _, _ => throw (arityError "softmax" node "one parent")

/-- ReLU interval contract. -/
def reluContract : GraphRangeContract where
  key := .backend .relu
  name := "ReLU"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let input <- parentRange context.ranges parent
        pure (.relu parent, input.relu)
    | _ => throw (arityError "ReLU" node "one parent")

/-- Absolute-value interval contract. -/
def absContract : GraphRangeContract where
  key := .backend .abs
  name := "absolute value"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let input <- parentRange context.ranges parent
        pure (.abs parent, input.abs)
    | _ => throw (arityError "absolute value" node "one parent")

/-- Square-root contract with a checked nonnegative domain. -/
def sqrtContract : GraphRangeContract where
  key := .backend .sqrt
  name := "square root"
  derive := fun context node =>
    match node.parents with
    | #[parent] => do
        let input <- parentRange context.ranges parent
        if nonnegativeEndpoint input.lo && nonnegativeEndpoint input.hi then
          pure (.sqrtNonnegative parent, input.sqrt)
        else
          throw s!"numerical certificate: node {node.id} square-root range contains negative values"
    | _ => throw (arityError "square root" node "one parent")

/-- Constructor for bounded transcendental contracts. -/
def boundedUnaryContract (op : BackendOp) (rule : Nat -> RangeRule)
    (enclosure : IEEE32Exec.Interval32) : GraphRangeContract where
  key := .backend op
  name := s!"{op.name} codomain"
  derive := fun _ node =>
    match node.parents with
    | #[parent] => pure (rule parent, enclosure)
    | _ => throw (arityError op.name node "one parent")

/-- Built-in numerical contracts. Grouping is by operation semantics, never by architecture. -/
def defaultContracts : Array GraphRangeContract :=
  #[ sourceContract
  , structuralContract
  , preserveContract .permute
  , preserveContract .broadcast
  , preserveContract .reshape
  , maxPoolContract
  , maxPoolPadContract
  , averagePoolContract false
  , averagePoolContract true
  , binaryContract .add .add IEEE32Exec.Interval32.add
  , binaryContract .sub .sub IEEE32Exec.Interval32.sub
  , binaryContract .mul .mul IEEE32Exec.Interval32.mul
  , inverseContract
  , hullContract .max
  , hullContract .min
  , hullContract .concat
  , sumContract
  , axisReductionContract false
  , axisReductionContract true
  , matmulContract
  , mseContract
  , layerNormContract
  , softmaxContract
  , reluContract
  , absContract
  , sqrtContract
  , boundedUnaryContract .sigmoid .unitBound unitInterval
  , boundedUnaryContract .tanh .signedUnitBound signedUnitInterval
  , boundedUnaryContract .sin .signedUnitBound signedUnitInterval
  , boundedUnaryContract .cos .signedUnitBound signedUnitInterval
  ]

/-- TorchLean's built-in numerical registry. Construction is checked once at use sites so a future
duplicate produces an explicit configuration failure. -/
def defaultRegistry : Except String GraphRangeRegistry :=
  GraphRangeRegistry.ofArray "torchlean.graph-numerics.v1" defaultContracts

/-- Compute one node range using an explicit numerical contract registry. -/
def deriveNodeRangeWith (registry : GraphRangeRegistry)
    (sources : Array CheckedSourceRange) (plan : AcceptedGraphKernelPlan)
    (ranges : Array CheckedNodeRange) (node : Node) : Except String RangeTransferResult := do
  let key := numericalOpKey node.kind
  let contract <- match registry.find? key with
    | some contract => pure contract
    | none =>
        throw (s!"numerical certificate: node {node.id} ({node.kind.describe}) " ++
          "has no registered numerical contract")
  contract.derive { sources, plan, ranges } node

/-- Compute one node range using TorchLean's built-in registry. -/
def deriveNodeRange (sources : Array CheckedSourceRange) (plan : AcceptedGraphKernelPlan)
    (ranges : Array CheckedNodeRange) (node : Node) : Except String RangeTransferResult := do
  let registry <- defaultRegistry
  deriveNodeRangeWith registry sources plan ranges node

/-- Construct and validate the canonical range trace using an explicit contract registry. -/
def buildRangeTraceWith (registry : GraphRangeRegistry)
    (graph : Graph) (sources : Array CheckedSourceRange)
    (plan : AcceptedGraphKernelPlan) :
    Except String (Array CheckedNodeRange) := do
  graph.checkWellFormed
  let _ <- requireNumericalCoverage registry graph
  checkSourceOwnership graph sources
  let mut ranges : Array CheckedNodeRange := #[]
  for node in graph.nodes do
    let (rule, enclosure) <- deriveNodeRangeWith registry sources plan ranges node
    if h : validInterval enclosure then
      ranges := ranges.push
        { nodeId := node.id
          outShape := node.outShape
          rule
          enclosure
          valid := (validInterval_eq_true_iff enclosure).mp h }
    else
      throw (s!"numerical certificate: node {node.id} ({node.kind.describe}) produced a " ++
        "non-finite or unordered enclosure")
  pure ranges

/-- Construct and validate the canonical range trace using TorchLean's built-in contracts. -/
def buildRangeTrace (graph : Graph) (sources : Array CheckedSourceRange)
    (plan : AcceptedGraphKernelPlan) : Except String (Array CheckedNodeRange) := do
  let registry <- defaultRegistry
  buildRangeTraceWith registry graph sources plan

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
end RuntimeApprox
end Proofs
