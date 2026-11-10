/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.OpGraph
public import TorchLean.NN.IR.OpContracts
public import TorchLean.NN.Batch
public import TorchLean.NN.Program -- shake: keep
public import TorchLean.Tensor.Op.Codec

/-!
# Recording registered operators in the verifier IR

`record op args` appends the steps of one registered operator `op : TorchLean.Op` applied to `args`
to the typed graph of operators (`IR.OpGraph`) under construction, whose input is value `0`.
The `Ops` instance of `RecordM` implements `apply` as `record` on the arguments that
`Args.ofPortArgs` converts (frozen data becomes a constant reference, a frozen reference passes
unchanged), and it refuses `applyRandom`. A step is a registered operator on positions of the
graph. Each operand reference is resolved when it is read, before later operands add nodes, and
its value must be at a port that the operator's input accepts. The operator must be admissible
(`Op.Admissible`: its attribute check passes, and its rule checks no input value and reads no
optional capability). `record` never reads the capability record of the call site's
`Op.Supported` instance, and `RecordM.input` is value `0` however often it is called.

An operand is a reference (`Ref`): a value of the graph, or a constant tensor that becomes a
constant each time it is read. A value, detached or frozen port takes a reference, so frozen data
is passed as `.const t`, and an index, Boolean or integer port takes its tensor. Most operators
are one step of themselves on their operands, in port order. The others are recorded as follows.
- `scale` is a product with a constant tensor filled with the factor.
- `clamp` is the maximum with the lower bound, then the minimum with the upper bound, each with
  the constant bound as its first operand. `Context.clamp` puts the value first, so when the value
  and a bound are both NaN the recorded graph carries the bound's NaN where `Context.clamp` carries
  the value's. A scalar context with differentiation metadata is refused, because the graph cannot
  conditionally clear its tangents at the bounds.
- `scaledMatmul` is itself where `QuotientArithmetic.supported` holds, and otherwise a matrix
  product followed by a product with a filled constant, as `scaledProductsSlice` computes it.
- `linear`, `layerNorm` and `batchNormRunning` are themselves. Their weights, scale, bias, running
  statistics and epsilon must be constants, and they become constants after the activation is
  resolved.
- `attentionFromScores` is a softmax over the last dimension, a product with the weights and a
  matrix product with the values. A mask is a Boolean constant that hard-masks the softmax: a
  blocked entry has a zero numerator, with no finite sentinel.
- `multiheadAttention` without leading axes is the projections, the head split by a reshape and
  an exchange of the token and head axes, the scaled scores, the softmax and the output
  projection. Over leading axes whose first extent is positive, each sequence is recorded alone by
  `mapBatch` along each leading axis.
- Selecting one coordinate (`indexAt`) is a length-one slice followed by a reshape that drops that
  dimension.
- `Op.conv` is itself with the leading shape `[1]` or `[]`, channels-first dimension numbers, unit
  dilation, symmetric padding and one group, and its kernel and bias must be constants.
- `Op.pool` is itself for max pooling and padded average pooling over every axis after the first,
  with symmetric padding, unit dilation and floor mode.

Random operators are refused: graph random nodes derive keys from node positions, while eager and
typed execution count random calls. A matrix product with a vector operand, GELU, training batch
normalization and any other pooling are refused with their own messages, and every other operator
with `TorchLean→IR: <tag> is outside the verifier IR fragment`, where `<tag>` is the operator's
codec tag.

Values are graph positions, so every parameter constant has its own position: an input followed by
a `linear` is input `0`, weight `1`, bias `2` and the linear map `3`.
`TorchLean.IR.OpGraph.eval_toOp` proves that the graph of registered operators of a typed graph
returns its typed values at the outputs, for every input. No theorem relates the recorded graph to
the source program.

References (informal):
- IBP: Gowal et al. (2018).
- CROWN / DeepPoly-style linear relaxations: Zhang et al. (2018).
- LiRPA unification viewpoint: Xu et al. (2020).
-/

public section

namespace TorchLean.IR

open _root_.TorchLean.Tensor

/-- A reference produced while recording a TorchLean program: a value of the graph, by its
position, or a compile-time tensor constant that becomes a constant each time it is an operand.
A position is checked against the current graph (bounds and port) when an operation reads it, and
nothing ties it to the recording that produced it. -/
inductive RecordM.Ref (α : Type) [Storage α] : Shape → Type where
  | node  {s : Shape} (id : Nat) : Ref α s
  | const {s : Shape} (t : Tensor α s) : Ref α s

/-- The typed graph of operators recorded so far over an input of shape `σ`. -/
structure RecordState (α : Type) [Storage α] (σ : Shape) where
  /-- The ports of the graph's values, the input first. -/
  ports : List Port
  /-- The graph, with each constant stored as its tensor. -/
  graph : OpGraph (Port.Carrier α) σ ports

/-- The state holding only the graph input. -/
def RecordState.input {α : Type} [Storage α] {σ : Shape} : RecordState α σ :=
  ⟨[.value σ], .input⟩

/-- The recording monad of TorchLean-to-IR lowering, for programs with an input of shape `σ`. -/
abbrev RecordM (α : Type) [Storage α] (σ : Shape) : Type → Type :=
  StateT (RecordState α σ) (Except String)

/-- The argument of an operator port: a reference for a value, detached or frozen port, the tensor
for an index, Boolean or integer port. -/
abbrev RecordM.Arg (α : Type) [Storage α] : Port → Type :=
  Port.Slot (Ref α) (Ref α) (Ref α)

/-- The arguments of a list of ports, as nested pairs ending in `Unit`. -/
abbrev RecordM.Args (α : Type) [Storage α] : List Port → Type := List.TProd (Arg α)

namespace RecordM

variable {α : Type} [Storage α] {σ : Shape}

/-- Run a shared IR shape contract and surface its error. -/
private def requireContract {β : Type} (r : Except String β) : RecordM α σ Unit := do
  match r with
  | .ok _ => pure ()
  | .error msg => throw msg

/-- The arguments of the input ports `ins` at the positions `ids`, refused unless each position
names a value at a port that its input port accepts. -/
private def args (ports : List Port) :
    (ins : List Port) → List Nat → Except String (OpGraph.Args ports ins)
  | [], [] => .ok ()
  | p :: ps, id :: ids => do
      if h : id < ports.length then
        if hp : p.accepts (ports.get ⟨id, h⟩) = true then
          pure (⟨⟨id, h⟩, hp⟩, ← args ports ps ids)
        else throw s!"TorchLean→IR: node {id} has port {repr (ports.get ⟨id, h⟩)}, which the \
          input port {repr p} does not accept"
      else throw s!"TorchLean→IR: node {id} is not in the recorded graph"
  | _, _ => throw "TorchLean→IR: internal error (operand count)"

/-- Append operator `o` on the values at `ids`, after checking the ports of its operands and that
it is admissible, and return the position of its first output, which must be a value of shape
`s`. -/
private def push {ins outs : List Port} {s : Shape} (o : Op ins outs) (spec : o.HasRuleEverywhere)
    (ids : List Nat) : RecordM α σ (Ref α s) := do
  let st ← get
  let name := (Op.encode ⟨ins, outs, o⟩).tag
  unless outs.head? = some (.value s) do
    throw s!"TorchLean→IR: {name} has outputs {repr outs}, not a value of shape {repr s} first"
  let a ← StateT.lift (args st.ports ins ids)
  match hc : o.check with
  | .error e => throw e
  | .ok () =>
      if hv : o.checksValues = false then
        if hn : o.needsCapability = false then
          set ({ ports := st.ports ++ outs, graph := st.graph.push (.op o a ⟨hc, hv, hn, spec⟩) } :
            RecordState α σ)
          pure (.node st.ports.length)
        else throw s!"TorchLean→IR: {name} reads an optional capability of the scalar"
      else throw s!"TorchLean→IR: {name} checks its input values"

/-- The position of a reference. A graph value is checked against the current graph (bounds and
port) and returned, and a constant tensor is appended to the graph as a constant. -/
def materialize {s : Shape} (r : Ref α s) : RecordM α σ Nat := do
  match r with
  | .node id =>
      let st ← get
      if h : id < st.ports.length then
        if st.ports.get ⟨id, h⟩ = .value s then pure id
        else throw s!"TorchLean→IR: node {id} has port {repr (st.ports.get ⟨id, h⟩)}, not \
          {repr (Port.value s)}"
      else throw s!"TorchLean→IR: node {id} is not in the recorded graph"
  | .const t => constant (.value s) t
where
  /-- Append a constant at port `p` and return its position. -/
  constant (p : Port) (t : p.Carrier α) : RecordM α σ Nat := do
    let st ← get
    set ({ ports := st.ports ++ [p], graph := st.graph.const p t } : RecordState α σ)
    pure st.ports.length

/-- The graph input, value `0`. -/
def input : RecordM α σ (Ref α σ) :=
  pure (.node 0)

/-- The tensor of a constant reference. A graph value is refused. -/
private def getConst {s : Shape} (r : Ref α s) : RecordM α σ (Tensor α s) :=
  match r with
  | .const t => pure t
  | .node _ =>
      throw "TorchLean IR lowering: expected a compile-time constant tensor (got a graph node)"

/-- Operator `o` on the given operands, materialized in order. -/
def op {ins outs : List Port} {s : Shape} (o : Op ins outs) (spec : o.HasRuleEverywhere)
    (operands : List ((t : Shape) × Ref α t)) : RecordM α σ (Ref α s) := do
  let ids ← operands.mapM fun ⟨_, r⟩ => materialize r
  push o spec ids

variable [Context α]

/-- The matrix product over the leading shape `batch`, with the optional `scale` inside each
contraction where the scalar supports it (`QuotientArithmetic.supported`). Otherwise the scale
must be a constant, applied as a product with a filled constant after the contraction. -/
private def product {batch : Shape} {m n p : Nat} (a : Ref α (batch ++ [m, n]))
    (b : Ref α (batch ++ [n, p])) (scale : Option (Ref α [])) :
    RecordM α σ (Ref α (batch ++ [m, p])) :=
  let matmul : Op [.value (Shape.matmulLeft false batch m n),
      .value (Shape.matmulRight false batch n p)]
      [.value (Shape.matmulOut false false batch m p)] :=
    .matmul false false (fun h => nomatch h) (fun h => nomatch h)
      (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch)
  match scale with
  | none => op matmul (fun _ _ _ _ => rfl) [⟨_, a⟩, ⟨_, b⟩]
  | some c =>
      if (inferInstance : Numeric.QuotientArithmetic α).supported then
        op (.scaledMatmul (batchA := batch) (batchB := batch) (batch := batch) (m := m) (n := n)
            (p := p) false false (fun h => nomatch h) (fun h => nomatch h)
            (Shape.CanBroadcastTo.refl batch) (Shape.CanBroadcastTo.refl batch))
          (fun _ _ _ _ => rfl) [⟨_, a⟩, ⟨_, b⟩, ⟨_, c⟩]
      else do
        let c ← getConst c
        let y : Ref α (batch ++ [m, p]) ← op matmul (fun _ _ _ _ => rfl) [⟨_, a⟩, ⟨_, b⟩]
        op (Op.mul (s := batch ++ [m, p])) (fun _ _ _ _ => rfl)
          [⟨_, y⟩, ⟨_, .const (Tensor.full (batch ++ [m, p]) c.item)⟩]

/-- The reshape of a value of shape `s` to the shape `t` of the same size. -/
private def reshape {s : Shape} (t : Shape) (x : Ref α s) : RecordM α σ (Ref α t) :=
  if h : s.size = t.size then op (Op.reshape (s := s) (t := t) h) (fun _ _ _ _ => rfl) [⟨_, x⟩]
  else throw s!"TorchLean→IR: reshape: {repr s} and {repr t} differ in size"

/-- The exchange of the axes `a` and `b` of a value of shape `s`, declared at `t`. -/
private def swap {s : Shape} (t : Shape) (a b : Nat) (x : Ref α s) : RecordM α σ (Ref α t) :=
  match s.perm? (((Array.range s.rank).map (Equiv.swapCore a b)).map Int.ofNat) with
  | some perm => op (Op.permute perm (s.permute perm) rfl) (fun _ _ _ _ => rfl) [⟨_, x⟩]
  | none => throw s!"TorchLean→IR: permute: axes {a} and {b} of {repr s}"

/-- The softmax of a value of shape `s` over its last dimension, hard-masked by `mask` when given:
the mask is a Boolean constant read by `Op.maskedSoftmax`. -/
private def softmaxLast {s : Shape} (mask : Option (Tensor Bool s)) (x : Ref α s) :
    RecordM α σ (Ref α s) :=
  if h : s.rank - 1 < s.rank then
    match mask with
    | none => op (Op.softmax ⟨s.rank - 1, h⟩) (fun _ _ _ _ => rfl) [⟨_, x⟩]
    | some m => do
        let xId ← materialize x
        let mId ← materialize.constant (.bool s) m
        push (Op.maskedSoftmax ⟨s.rank - 1, h⟩) (fun _ _ _ _ => rfl) [xId, mId]
  else throw "TorchLean→IR: softmax of a scalar"

/-- Softmax over the last dimension of the scores, hard-masked by `mask` repeated over the batch
axes when given, then the product with `weights` and the matrix product with `value`. -/
private def attentionFromScores {batch : Shape} {rows keys width : Nat}
    (mask : Option (Tensor Bool [rows, keys])) (scores : Ref α (batch ++ [rows, keys]))
    (value : Ref α (batch ++ [keys, width])) (weights : Ref α (batch ++ [rows, keys])) :
    RecordM α σ (Ref α (batch ++ [rows, width])) := do
  let rec repeatMask (mask : Tensor Bool [rows, keys]) :
      (batch : Shape) → Tensor Bool (batch ++ [rows, keys])
    | .scalar => mask
    | .dim _ inner => Tensor.dim (fun _ => repeatMask mask inner)
  let probabilities ← softmaxLast (mask.map (repeatMask · batch)) scores
  let weighted : Ref α (batch ++ [rows, keys]) ←
    op (Op.mul (s := batch ++ [rows, keys])) (fun _ _ _ _ => rfl)
      [⟨_, probabilities⟩, ⟨_, weights⟩]
  product weighted value none

/-- Multi-head attention on one sample: the three projections, each reshaped to
`[n, numHeads, headDim]` and with its token and head axes exchanged, the scores scaled by
`1 / Attention.scaleDenom headDim`, the softmax over the keys (hard-masked when given), the value
product, the exchange back, the reshape to `[n, numHeads * headDim]` and the output projection. -/
private def multiheadAttention {n numHeads dModel headDim : Nat}
    (wq wk wv : Ref α [dModel, numHeads * headDim]) (wo : Ref α [numHeads * headDim, dModel])
    (x : Ref α [n, dModel]) (mask : Option (Tensor Bool [n, n])) :
    RecordM α σ (Ref α [n, dModel]) := do
  let q : Ref α [n, numHeads * headDim] ← product (batch := .scalar) x wq none
  let k : Ref α [n, numHeads * headDim] ← product (batch := .scalar) x wk none
  let v : Ref α [n, numHeads * headDim] ← product (batch := .scalar) x wv none
  -- Each projected coordinate is `(head, coordinate-within-head)` in row-major order. Reshaping
  -- each token first and then moving the head axis outward keeps that reading. A single reshape
  -- of `(n, numHeads * headDim)` to `(numHeads, n, headDim)` would give each head the wrong
  -- features.
  let qProjected : Ref α [n, numHeads, headDim] ← reshape _ q
  let kProjected : Ref α [n, numHeads, headDim] ← reshape _ k
  let vProjected : Ref α [n, numHeads, headDim] ← reshape _ v
  let qh : Ref α [numHeads, n, headDim] ← swap _ 0 1 qProjected
  let kh : Ref α [numHeads, n, headDim] ← swap _ 0 1 kProjected
  let vh : Ref α [numHeads, n, headDim] ← swap _ 0 1 vProjected
  let kt : Ref α [numHeads, headDim, n] ← swap _ 1 2 kh
  let invScale : α := 1 / Attention.scaleDenom (α := α) headDim
  let scores : Ref α [numHeads, n, n] ←
    product (batch := [numHeads]) qh kt (some (.const (Tensor.scalar invScale)))
  let attn ← softmaxLast
    (mask.map fun m => (Tensor.dim (fun _ => m) : Tensor Bool [numHeads, n, n])) scores
  let outHeads : Ref α [numHeads, n, headDim] ← product (batch := [numHeads]) attn vh none
  -- Moving the head axis back inward returns to the token-major projected layout.
  let swapped : Ref α [n, numHeads, headDim] ← swap _ 0 1 outHeads
  let concat : Ref α [n, numHeads * headDim] ← reshape _ swapped
  product (batch := .scalar) concat wo none

/-- The slice `[start, start + len)` of a value of shape `s` along its first axis, declared at
`t`. -/
private def sliceFirst {s : Shape} (t : Shape) (start len : Nat) (x : Ref α s) :
    RecordM α σ (Ref α t) :=
  match sliceDims s #[0] #[start] #[start + len] #[1] with
  | .ok ds => op (Op.slice ds.1 [start] [start + len] [1] ds.2) (fun _ _ _ _ => rfl) [⟨_, x⟩]
  | .error e => throw e

/-- The concatenation of two values along their first axis, declared at `t`. -/
private def catFirst {s₁ s₂ : Shape} (t : Shape) (x : Ref α s₁) (y : Ref α s₂) :
    RecordM α σ (Ref α t) :=
  if h : 0 < s₁.rank then
    op (Op.cat (s := s₁) ⟨0, h⟩ [s₁.extent ⟨0, h⟩, s₂.headD 0]) (fun _ _ _ _ => rfl)
      [⟨_, x⟩, ⟨_, y⟩]
  else throw "TorchLean→IR: cat of a scalar"

/-- Multi-head attention of every sequence along the leading axes `batch`: each sequence alone, by
`mapBatch` along each leading axis, with the slices, reshapes and concatenations it makes. -/
private def multiheadAttentionOver {n numHeads dModel headDim : Nat}
    (wq wk wv : Ref α [dModel, numHeads * headDim]) (wo : Ref α [numHeads * headDim, dModel])
    (mask : Option (Tensor Bool [n, n])) :
    (batch : Shape) → Ref α (batch ++ [n, dModel]) →
      RecordM α σ (Ref α (batch ++ [n, dModel]))
  | .scalar, x => multiheadAttention wq wk wv wo x mask
  | .dim _ inner, x =>
      mapBatch
        (pure (.const (Tensor.dim (fun i : Fin 0 => Fin.elim0 i))))
        (fun x start len _ => sliceFirst _ start len x)
        (fun {_ s₂} x _ => reshape s₂ x)
        (fun x y => catFirst _ x y)
        (multiheadAttentionOver wq wk wv wo mask inner) x

/-- The window of a pooling over every axis after the first with symmetric padding, unit dilation
and floor mode, and `none` for any other pooling. -/
private def window? {r : Nat} (dims : List (Fin r))
    (kernel stride padLow padHigh dilation : List Nat) (ceilMode : Bool) : Option WindowConfig :=
  let d := r - 1
  if dims.map (·.val) = (List.range r).tail ∧ padLow = padHigh ∧ dilation = List.replicate d 1 ∧
      ceilMode = false ∧ kernel.length = d ∧ stride.length = d ∧ padLow.length = d then
    some { spatialRank := d
           kernel := Tensor.ofFn fun i => kernel.getD i.val 0
           stride := Tensor.ofFn fun i => stride.getD i.val 0
           padding := Tensor.ofFn fun i => padLow.getD i.val 0 }
  else none

end RecordM

variable {α : Type} [Storage α] [Context α] {σ : Shape}

/-- The operands of a list of ports with a reference, in port order. -/
private def RecordM.Args.refs : (ps : List Port) → Args α ps → List ((t : Shape) × Ref α t)
  | [], () => []
  | ⟨.value, s⟩ :: ps, (x, xs) => ⟨s, x⟩ :: refs ps xs
  | ⟨.frozen, s⟩ :: ps, (x, xs) => ⟨s, x⟩ :: refs ps xs
  | ⟨.detached, s⟩ :: ps, (x, xs) => ⟨s, x⟩ :: refs ps xs
  | ⟨.index _, _⟩ :: ps, (_, xs) => refs ps xs
  | ⟨.bool, _⟩ :: ps, (_, xs) => refs ps xs
  | ⟨.integer _ _, _⟩ :: ps, (_, xs) => refs ps xs

/-- The arguments of a model's call (`TorchLean.PortArgs`) in the form `IR.record` takes. Frozen
data becomes a constant reference, and every other argument, a frozen reference included, passes
unchanged. -/
def RecordM.Args.ofPortArgs (ps : List Port)
    (args : PortArgs α (Ref α) (fun β _ s => Tensor β s) ps) :
    Args α ps :=
  Id.run <| PortArgs.mapFrozen (m := Id) (F := Ref α) ps args
    fun | .data t => pure (.const t) | .ref r => pure r

/-- A proof that the operator has a rule at every scalar, for each operator recorded as one step
of itself on its operands in port order, and `none` for any other operator. -/
private def direct? {ins outs : List Port} : (o : Op ins outs) → Option (PLift o.HasRuleEverywhere)
  | .add | .sub | .mul | .max | .min | .abs | .sqrt .raw | .sqrt .clampNegative | .relu | .sigmoid
  | .silu | .tanh | .softplus | .exp | .sin | .cos | .log | .reciprocal | .detach | .safeLog
  | .mseLoss
  | .matmul false false _ _ _ _ | .broadcast _ _ | .reshape _ | .permute _ _ _ | .reduce _ _ _ _
  | .mean _ _ | .slice _ _ _ _ _ | .cat _ _ | .softmax _ | .logSoftmax _ | .topKMask _ _ =>
      some ⟨fun _ _ _ _ => rfl⟩
  | _ => none

/-- The refusal of a random operator: graph random nodes derive their keys from node positions,
while eager and typed execution count random calls. -/
def randomRefusal : String :=
  "TorchLean→IR: random invocation keys are not represented by node-indexed IR randomness"

/-- The refusal of a matrix product with a vector operand. The recorder rebuilds the IR product
from the operand shapes, and with matrix operands the output shape identifies the product. With a
vector operand, equal shapes need not identify it. -/
def matmulVectorRefusal : String :=
  "TorchLean→IR: a matrix product with a vector operand is outside the generic recorder, " ++
    "which records matrix operands only"

/-- The refusal of an operator outside the verifier fragment, named by its codec tag. -/
private def refusal {ins outs : List Port} (op : Op ins outs) : String :=
  s!"TorchLean→IR: {(Op.encode ⟨ins, outs, op⟩).tag} is outside the verifier IR fragment"

/-- `Op.conv` (`o`) on the kernel `w`, the bias `b` and the input `xr`, where the leading shape
`lead` is `[1]` or `[]`, the dimension numbers are channels first (the batch `[1, inC, spatial..]`,
the kernel `[outC, inC, window..]`), the dilation is one, the padding is symmetric and there is one
group. It is one step of `o`, after checks of nonzero kernel extents and strides and of the IR's
convolution contract, and its kernel and bias must be constants that become constants after the
input is resolved. Any other convolution is refused with the message `refused`. -/
private def recordConv {s k x y : Shape} {r : Nat} (dims : Shape.ConvDims s k r)
    (o : Op [.value k, .value [k.extent (dims.kernel ⟨0, by omega⟩)], .value x] [.value y])
    (spec : o.HasRuleEverywhere) (stride padLow padHigh dilation : Fin r → Nat) (groups : Nat)
    (lead : Shape) (refused : String) (w : RecordM.Ref α k)
    (b : RecordM.Ref α [k.extent (dims.kernel ⟨0, by omega⟩)]) (xr : RecordM.Ref α x) :
    RecordM α σ (RecordM.Ref α y) :=
  if (lead = [1] ∨ lead = []) ∧ (∀ p, (dims.input p).val = p.val) ∧
      (∀ p, (dims.kernel p).val = p.val) ∧ (∀ i, dilation i = 1) ∧ (∀ i, padLow i = padHigh i) ∧
      groups = 1 then do
    let kernel : Tensor Nat [r] := Tensor.ofFn fun i => k.extent (dims.kernel ⟨i.val + 2, by omega⟩)
    let strides : Tensor Nat [r] := Tensor.ofFn stride
    let padding : Tensor Nat [r] := Tensor.ofFn padLow
    let inC := k.extent (dims.kernel ⟨1, by omega⟩)
    let outC := k.extent (dims.kernel ⟨0, by omega⟩)
    if ∀ i : Fin r, kernel.getScalar i ≠ 0 then
      if ∀ i : Fin r, strides.getScalar i ≠ 0 then
        RecordM.requireContract <|
          IR.inferConvConfigOutShape "conv"
            { spatialRank := r, kernel := kernel, stride := strides, padding := padding,
              dilation := Tensor.full [r] 1, paddingAfter := padding, groups := 1,
              channelAxis := lead.length, inChannels := inC, outChannels := outC } x
        let _ ← RecordM.getConst w
        let _ ← RecordM.getConst b
        let xId ← RecordM.materialize xr
        let wId ← RecordM.materialize w
        let bId ← RecordM.materialize b
        RecordM.push o spec [wId, bId, xId]
      else
        throw "TorchLean→IR: conv: every stride must be nonzero"
    else
      throw "TorchLean→IR: conv: every kernel extent must be nonzero"
  else throw refused

/-- Record the registered operator `op` on `args`: append its steps to the graph and return its
value output. -/
def record {ins rest : List Port} {s : Shape} (op : Op ins (.value s :: rest))
    (args : RecordM.Args α ins) :
    RecordM α σ (RecordM.Ref α s) :=
  match op, args with
  | .scale (s := s), (x, c, ()) => do
      let c ← RecordM.getConst c
      RecordM.op (Op.mul (s := s)) (fun _ _ _ _ => rfl)
        [⟨_, x⟩, ⟨_, .const (Tensor.full s c.item)⟩]
  | .clamp (s := s), (x, lo, hi, ()) => do
      match Context.stopGradient? (α := α) with
      | some _ =>
          throw
            ("TorchLean→IR: clamp with scalar differentiation metadata is outside the " ++
              "verifier IR fragment; use eager execution for conditional " ++
              "tangent clearing.")
      | none =>
          let lo ← RecordM.getConst lo
          let hi ← RecordM.getConst hi
          let y : RecordM.Ref α s ← RecordM.op (Op.max (s := s)) (fun _ _ _ _ => rfl)
            [⟨_, .const (Tensor.full s lo.item)⟩, ⟨_, x⟩]
          RecordM.op (Op.min (s := s)) (fun _ _ _ _ => rfl)
            [⟨_, .const (Tensor.full s hi.item)⟩, ⟨_, y⟩]
  | .scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch) (m := m) (n := n)
      (p := p) false false hl hr broadcastA broadcastB, (a, b, c, ()) =>
      if (inferInstance : Numeric.QuotientArithmetic α).supported then
        RecordM.op (Op.scaledMatmul (batchA := batchA) (batchB := batchB) (batch := batch)
            (m := m) (n := n) (p := p) false false hl hr broadcastA broadcastB)
          (fun _ _ _ _ => rfl) [⟨_, a⟩, ⟨_, b⟩, ⟨_, c⟩]
      else do
        let c ← RecordM.getConst c
        let y : RecordM.Ref α s ← RecordM.op (Op.matmul (batchA := batchA) (batchB := batchB)
            (batch := batch) (m := m) (n := n) (p := p) false false hl hr broadcastA broadcastB)
          (fun _ _ _ _ => rfl) [⟨_, a⟩, ⟨_, b⟩]
        RecordM.op (Op.mul (s := s)) (fun _ _ _ _ => rfl)
          [⟨_, y⟩, ⟨_, .const (Tensor.full s c.item)⟩]
  | .matmul true _ _ _ _ _, _ | .matmul false true _ _ _ _, _
  | .scaledMatmul _ _ _ _ _ _, _ => throw matmulVectorRefusal
  | .attentionFromScores mask, (scores, value, weights, ()) =>
      RecordM.attentionFromScores mask scores value weights
  | .multiheadAttention (batch := .scalar) _ mask, (wq, wk, wv, wo, x, ()) =>
      RecordM.multiheadAttention wq wk wv wo x mask
  | .multiheadAttention (batch := .dim (b + 1) inner) _ mask, (wq, wk, wv, wo, x, ()) =>
      RecordM.multiheadAttentionOver wq wk wv wo mask (.dim (b + 1) inner) x
  | .indexAt (s := s) dim index, (x, ()) => do
      let sliced : RecordM.Ref α (s.set dim 1) ←
        match sliceDims s #[dim.val] #[index.val] #[index.val + 1] #[1] with
        | .ok ds =>
            RecordM.op (Op.slice ds.1 [index.val] [index.val + 1] [1] ds.2) (fun _ _ _ _ => rfl)
              [⟨_, x⟩]
        | .error e => throw e
      RecordM.reshape (s.eraseIdx dim) sliced
  | .conv dims stride padLow padHigh dilation groups (lead := lead) hlead hx hy, (w, b, x, ()) =>
      recordConv dims (.conv dims stride padLow padHigh dilation groups hlead hx hy)
        (fun _ _ _ _ => rfl) stride padLow padHigh dilation groups lead
        (refusal (Op.conv dims stride padLow padHigh dilation groups hlead hx hy)) w b x
  | .linear (lead := lead) (inDim := inDim) (outDim := outDim), (w, b, x, ()) => do
      let _ ← RecordM.getConst w
      let _ ← RecordM.getConst b
      let xId ← RecordM.materialize x
      let wId ← RecordM.materialize w
      let bId ← RecordM.materialize b
      RecordM.push (Op.linear (lead := lead) (inDim := inDim) (outDim := outDim))
        (fun _ _ _ _ => by cases lead <;> rfl) [wId, bId, xId]
  | .layerNorm hSeq hEmbed hlead hx, (x, gamma, beta, epsilon, ()) => do
      let _ ← RecordM.getConst gamma
      let _ ← RecordM.getConst beta
      let _ ← RecordM.getConst epsilon
      let xId ← RecordM.materialize x
      let gId ← RecordM.materialize gamma
      let bId ← RecordM.materialize beta
      let eId ← RecordM.materialize epsilon
      RecordM.push (Op.layerNorm hSeq hEmbed hlead hx) (fun _ _ _ _ => rfl) [xId, gId, bId, eId]
  | .batchNormRunning dim, (x, gamma, beta, mean, var, epsilon, ()) => do
      let _ ← RecordM.getConst gamma
      let _ ← RecordM.getConst beta
      let _ ← RecordM.getConst mean
      let _ ← RecordM.getConst var
      let _ ← RecordM.getConst epsilon
      let xId ← RecordM.materialize x
      let gId ← RecordM.materialize gamma
      let bId ← RecordM.materialize beta
      let mId ← RecordM.materialize mean
      let vId ← RecordM.materialize var
      let eId ← RecordM.materialize epsilon
      RecordM.push (Op.batchNormRunning dim) (fun _ _ _ _ => rfl) [xId, gId, bId, mId, vId, eId]
  | .pool (s := s₀) pop dims kernel stride padLow padHigh dilation ceilMode, (x, ()) =>
      let name? : Option String := match pop with
        | .max => some "max_pool"
        | .avg true => some "avg_pool"
        | _ => none
      match name?, RecordM.window? dims kernel stride padLow padHigh dilation ceilMode with
      | some name, some config => do
          RecordM.requireContract <| IR.inferWindowOutShape name config s₀
          RecordM.op (Op.pool pop dims kernel stride padLow padHigh dilation ceilMode)
            (fun _ _ _ _ => rfl) [⟨_, x⟩]
      | _, _ =>
          throw ("TorchLean→IR: pooling other than max or padded average over every " ++
            "axis after the first, with symmetric padding, unit dilation and floor mode, is " ++
            "outside the verifier IR fragment")
  | .randUniform _, () => throw randomRefusal
  | .randn _, () => throw randomRefusal
  | .bernoulliMask _, _ => throw randomRefusal
  | .gelu, _ =>
      throw
        ("TorchLean→IR: GELU is outside the verifier IR fragment; " ++
          "use eager execution. The verifier fragment cannot lower its gate division.")
  | .batchNorm _, _ =>
      throw
        ("TorchLean→IR: batch_norm (training-style BN: stats from " ++
          "x) is outside the verifier IR fragment. For inference-time BN, " ++
          "use the eval-mode batch-normalization operation with explicit running statistics.")
  | op, args =>
      match direct? op with
      | some ⟨h⟩ => RecordM.op op h (RecordM.Args.refs ins args)
      | none => throw (refusal op)

instance : Ops (m := RecordM α σ) (α := α) where
  Ref := RecordM.Ref α
  DataRef := fun β _ s => Tensor β s
  const := fun {_s} t => pure (.const t)
  dataConst := fun t => t
  mapData := fun f t => f t
  observe := fun _ =>
    throw "TorchLean→IR: eager scalar control is outside the verifier IR fragment"
  apply := fun {ins _ _} op [_supported : Op.Supported α op] args =>
    record op (RecordM.Args.ofPortArgs ins args)
  applyRandom := fun {_ _ _} op [_supported : Op.Supported α op] _ _ => throw randomRefusal
  atomic := id

/-- Record `model` on the graph input, its last argument, with the tensors `params` as constants:
the recorded state and the position of the output, a constant when the output is a constant. -/
def recordForward {paramShapes : List Shape} {inShape outShape : Shape}
    (model : Program α (paramShapes ++ [inShape]) outShape)
    (params : TensorPack α paramShapes) :
    Except String ((st : RecordState α inShape) ×
      {out : Fin st.ports.length // st.ports.get out = .value outShape}) :=
  let rec constRefs : {shapes : List Shape} → TensorPack α shapes →
      Pack (RecordM.Ref α) shapes
    | [], .nil => .nil
    | _ :: shapes, .cons tensor tensors =>
        .cons (.const tensor) (constRefs (shapes := shapes) tensors)
  let build : RecordM α inShape Nat := do
    let x : RecordM.Ref α inShape ← RecordM.input
    let allRefs : Pack (RecordM.Ref α) (paramShapes ++ [inShape]) :=
      Pack.append (ss₁ := paramShapes) (ss₂ := [inShape])
        (constRefs params) (.cons x .nil)
    let outRef ← CurriedRef.uncurry
      (Ref := fun s => RecordM.Ref α s) (ss := paramShapes ++ [inShape])
        (model (m := RecordM α inShape))
      allRefs
    RecordM.materialize outRef
  match build.run .input with
  | .error e => .error e
  | .ok (outId, st) =>
      if h : outId < st.ports.length then
        if hp : st.ports.get ⟨outId, h⟩ = .value outShape then .ok ⟨st, ⟨outId, h⟩, hp⟩
        else .error s!"TorchLean→IR: node {outId} has port {repr (st.ports.get ⟨outId, h⟩)}, \
          not {repr (Port.value outShape)}"
      else .error s!"TorchLean→IR: node {outId} is not in the recorded graph"

end TorchLean.IR
