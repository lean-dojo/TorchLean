/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Export.Core
public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# IRPyTorch

IR → PyTorch code generation.

This module takes an op-tagged `NN.IR.Graph` plus a `NN.MLTheory.CROWN.Graph.ParamStore` payload and
emits a standalone PyTorch `nn.Module` implementation as a Python source string.

What this is (and isn't):

* This is an extraction/convenience layer used in round-trip examples: run/train in Python, then
  optionally import weights back to Lean.
* This is **not** a formal proof of semantic equivalence between PyTorch execution and the Lean IR
  denotation. The reference semantics remain the Lean definitions (`NN.IR.Graph.denote*`) and the
  proved `IRExec.ForwardGraph` bridge.

Assumptions:

* Node ids index `g.nodes` consistently; `getNode` checks that invariant.
* The ParamStore contains parameters for the node kinds that require them (linear/convolution/etc.).
* Only a supported subset of IR node kinds is lowered.

Failure modes (reported as `Except String`):

* missing/malformed ParamStore entries,
* shape mismatches between graph nodes and parameters,
* unsupported node kinds.

PyTorch context (comments only):

PyTorch’s ONNX exporter and `torch.export` can capture graphs for execution in other runtimes.
TorchLean’s emitter here prints readable Python that mirrors the IR, rather than producing an
execution-focused serialized graph artifact.
-/

public section


namespace Export
namespace IRPyTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.IR
open NN.MLTheory.CROWN.Graph
open Std
open Export.PyTorch

/-- Flatten a tensor and render it as a Python list literal (used for
  `torch.tensor([...]).reshape(...)`). -/
def tensorToPyFlat {s : Shape} (t : Tensor Float s) : String :=
  tensorToPyString (Tensor.flattenSpec (α := Float) t)

/-- Return `true` iff every array entry equals the first one, vacuously for an empty array. -/
def allEq (xs : Array Float) : Bool :=
  match xs[0]? with
  | none => true
  | some x => xs.all (fun y => y == x)

/--
Options controlling IR-to-PyTorch emission.

Most knobs here configure example emission, dtype handling, and how to
materialize constant nodes as buffers or parameters.
-/
structure Options where
  /-- Class name to use in the emitted Python source. -/
  className : String := "ExportedIRModel"
  /-- Python expression used for the tensor dtype (e.g. `"torch.float32"`). -/
  dtypeExpr : String := "torch.float32"
  /-- If true, include a small training skeleton and runtime-check in the emitted script. -/
  includeTrainingSkeleton : Bool := true
  /-- If true, emit IR `const` nodes as `nn.Parameter` when appropriate (learnable). -/
  learnableConsts : Bool := true
  deriving Repr

/--
How an IR `const` node is represented in the emitted PyTorch module.

- `bufferFull`: a non-learnable `register_buffer(...)` tensor
- `paramFull`: a learnable `nn.Parameter` with the full tensor shape.
-/
inductive ConstBinding where
  | bufferFull (attr : String)
  | paramFull (attr : String)
  deriving Repr

/-- Map from IR node id to how its constant should be referenced in the PyTorch module. -/
abbrev ConstBindings := HashMap Nat ConstBinding

/-- The Python attribute name used to reference a bound constant (`self.<attr>`). -/
def ConstBinding.attr : ConstBinding → String
  | .bufferFull a => a
  | .paramFull a => a

/-- Default attribute name for a const node: `self.const_<id>`. -/
def constAttr (id : Nat) : String := s!"const_{id}"

/-- Attribute name for a linear layer weight tensor in the ParamStore (`self.linear_<id>_W`). -/
def linearWAttr (id : Nat) : String := s!"linear_{id}_W"
/-- Attribute name for a linear layer bias tensor in the ParamStore (`self.linear_<id>_b`). -/
def linearBAttr (id : Nat) : String := s!"linear_{id}_b"

/-- Attribute name for a convolution kernel tensor in the parameter store. -/
def convKernelAttr (id : Nat) : String := s!"conv_{id}_kernel"
/-- Attribute name for a convolution bias tensor in the parameter store. -/
def convBiasAttr (id : Nat) : String := s!"conv_{id}_bias"

/-- Render a fixed-length natural-number vector as a Python tuple. -/
def natTensorToPyTuple {n : Nat} (v : TorchLean.Tensor Nat [n]) : String :=
  "(" ++ ", ".intercalate ((Tensor.to v (List Nat)).map toString) ++
    (if n = 1 then "," else "") ++ ")"

/-- Attribute name for a BatchNorm scale tensor (`self.batchnorm_<id>_gamma`). -/
def batchNormGammaAttr (id : Nat) : String := s!"batchnorm_{id}_gamma"
/-- Attribute name for a BatchNorm bias tensor (`self.batchnorm_<id>_beta`). -/
def batchNormBetaAttr (id : Nat) : String := s!"batchnorm_{id}_beta"
/-- Attribute name for a BatchNorm running mean tensor (`self.batchnorm_<id>_mean`). -/
def batchNormMeanAttr (id : Nat) : String := s!"batchnorm_{id}_mean"
/-- Attribute name for a BatchNorm running variance tensor (`self.batchnorm_<id>_var`). -/
def batchNormVarAttr (id : Nat) : String := s!"batchnorm_{id}_var"

/-- Emit a `torch.tensor([...]).reshape(...)` expression from a flat Python list literal and a
  `Shape`. -/
def pyTensorFromFlat (flatList : String) (shape : Shape) (dtypeVar : String := "dtype") : String :=
  s!"torch.tensor({flatList}, dtype={dtypeVar}).reshape({shapeToPyTupleString shape})"

/--
Retrieve a node from the graph and validate its id invariant.

This yields more actionable error messages than directly indexing into `g.nodes`.
-/
def getNode (g : NN.IR.Graph) (id : Nat) : Except String NN.IR.Node := do
  match g.nodes[id]? with
  | none => throw s!"IR→PyTorch: node id out of bounds: {id}"
  | some n =>
      if n.id != id then
        throw s!"IR→PyTorch: internal error: nodes[{id}].id = {n.id} (expected {id})"
      pure n

/-- Expect a node to have exactly one parent, returning that parent id. -/
private def expectUnary (id : Nat) (parents : Array Nat) : Except String Nat := do
  match parents with
  | #[p] => pure p
  | _ => throw s!"IR→PyTorch: node {id}: expected 1 parent, got {parents.size}"

/-- Expect a node to have exactly two parents, returning the pair. -/
private def expectBinary (id : Nat) (parents : Array Nat) : Except String (Nat × Nat) := do
  match parents with
  | #[a, b] => pure (a, b)
  | _ => throw s!"IR→PyTorch: node {id}: expected 2 parents, got {parents.size}"

/-- Return `true` iff `id` refers to a `.const` node in the graph. -/
private def isConstId (g : NN.IR.Graph) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some n =>
      match n.kind with
      | .const _ => true
      | _ => false

/--
Detect constant node ids that correspond to **LayerNorm affine parameters** (gamma/beta) emitted
by TorchLean-to-IR lowering.

In the IR backend, `layer_norm(x, gamma, beta)` is lowered into:
1) `layernorm(x)` (pure normalization)
2) `mul_elem(layernorm(x), gammaB)` where `gammaB` is a broadcasted const
3) `add(..., betaB)` where `betaB` is a broadcasted const

Those broadcasted consts should be emitted as `nn.Parameter` in PyTorch, even if they are
uniform (gamma initialized to ones, beta to zeros).
-/
private def detectLayerNormAffineConstIds (g : NN.IR.Graph) : Std.HashSet Nat :=
  Id.run do
    let mut s : Std.HashSet Nat := {}
    for ln in g.nodes do
      match ln.kind with
      | .layernorm _ =>
          for mulN in g.nodes do
            match mulN.kind with
            | .mulElem =>
                match mulN.parents with
                | #[a, b] =>
                    let gammaId? : Option Nat :=
                      if a == ln.id && isConstId g b then some b
                      else if b == ln.id && isConstId g a then some a
                      else none
                    if let some gammaId := gammaId? then
                      if (g.nodes[gammaId]?.map (fun n => n.outShape) == some ln.outShape) then
                        s := s.insert gammaId
                      for addN in g.nodes do
                        match addN.kind with
                        | .add =>
                            match addN.parents with
                            | #[p, q] =>
                                let betaId? : Option Nat :=
                                  if p == mulN.id && isConstId g q then some q
                                  else if q == mulN.id && isConstId g p then some p
                                  else none
                                if let some betaId := betaId? then
                                  if (g.nodes[betaId]?.map (fun n => n.outShape) == some
                                    ln.outShape) then
                                    s := s.insert betaId
                            | _ => ()
                        | _ => ()
                | _ => ()
            | _ => ()
      | _ => ()
    pure s

/--
Collect Python attribute bindings for all learnable parameters and constants.

This returns a mapping from constant node identifiers to Python references and the generated
`__init__` lines that materialize parameters and buffers.
-/
private def collectBindings (g : NN.IR.Graph) (ps : ParamStore Float) (options : Options) :
    Except String (ConstBindings × Array String) := do
  let mut bindings : ConstBindings := HashMap.emptyWithCapacity
  let mut initLines : Array String := #[]
  let forcedLearnableConsts := detectLayerNormAffineConstIds g

  -- Pass 1: linear and convolution parameters (stored in ParamStore keyed by node id).
  for n in g.nodes do
    match n.kind with
    | .linear =>
        match ps.linearWB.get? n.id with
        | none => throw s!"IR→PyTorch: missing linear params for node {n.id}"
        | some lp =>
            let wShape : Shape := .dim lp.m (.dim lp.n .scalar)
            let bShape : Shape := .dim lp.m .scalar
            let wFlat := tensorToPyFlat (s := wShape) lp.w
            let bFlat := tensorToPyFlat (s := bShape) lp.b
            initLines := initLines ++
              #[ indentFour
                s!"self.{linearWAttr n.id} = nn.Parameter({pyTensorFromFlat wFlat wShape})"
              , indentFour
                s!"self.{linearBAttr n.id} = nn.Parameter({pyTensorFromFlat bFlat bShape})"
              ]
    | .conv .. =>
        match ps.convCfg.get? n.id with
        | none => throw s!"IR→PyTorch: missing convolution params for node {n.id}"
        | some config =>
            let kShape : Shape :=
              .dim config.outChannels
                (.dim config.inChannels (Shape.ofList (Tensor.to config.kernel (List Nat))))
            let bShape : Shape := .dim config.outChannels .scalar
            let kFlat := tensorToPyFlat (s := kShape) config.spec.kernel
            let bFlat := tensorToPyFlat (s := bShape) config.spec.bias
            initLines := initLines ++
              #[ indentFour
                s!"self.{convKernelAttr n.id} = nn.Parameter({pyTensorFromFlat kFlat kShape})"
              , indentFour
                s!"self.{convBiasAttr n.id} = nn.Parameter({pyTensorFromFlat bFlat bShape})"
              ]
    | .batchNormEval .. =>
        match ps.batchNormEval.get? n.id with
        | none => throw s!"IR→PyTorch: missing BatchNorm parameters for node {n.id}"
        | some p =>
            let sC : Shape := .dim p.c .scalar
            let gamma := pyTensorFromFlat (tensorToPyFlat (s := sC) p.gamma) sC
            let beta := pyTensorFromFlat (tensorToPyFlat (s := sC) p.beta) sC
            let mean := pyTensorFromFlat (tensorToPyFlat (s := sC) p.mean) sC
            let var := pyTensorFromFlat (tensorToPyFlat (s := sC) p.var) sC
            initLines := initLines ++
              #[ indentFour s!"self.{batchNormGammaAttr n.id} = nn.Parameter({gamma})"
              , indentFour s!"self.{batchNormBetaAttr n.id} = nn.Parameter({beta})"
              , indentFour s!"self.register_buffer(\"{batchNormMeanAttr n.id}\", {mean})"
              , indentFour s!"self.register_buffer(\"{batchNormVarAttr n.id}\", {var})"
              ]
    | _ => pure ()

  -- Pass 2: const nodes (stored as flattened values in ParamStore.constVals).
  for n in g.nodes do
    match n.kind with
    | .const s =>
        match ps.constVals.get? n.id with
        | none => throw s!"IR→PyTorch: missing const value for node {n.id}"
        | some fv =>
            let flatListStr := tensorToPyString fv.v
            let flatVals := Tensor.to fv.v (Array Float)
            let uniform := allEq flatVals
            let forceLearn :=
              options.learnableConsts && forcedLearnableConsts.contains n.id
            let shouldLearn := options.learnableConsts && (forceLearn || !uniform)
            if shouldLearn then
              let attr := constAttr n.id
              initLines := initLines ++
                #[ indentFour s!"self.{attr} = nn.Parameter({pyTensorFromFlat flatListStr s})" ]
              bindings := bindings.insert n.id (.paramFull attr)
            else
              let attr := constAttr n.id
              initLines := initLines ++
                #[ indentFour
                  s!"self.register_buffer(\"{attr}\", {pyTensorFromFlat flatListStr s})" ]
              bindings := bindings.insert n.id (.bufferFull attr)
    | _ => pure ()

  pure (bindings, initLines)

/-- Emit the Python expression used to reference a bound constant (`self.<attr>` plus expansions).
  -/
private def constExpr (bindings : ConstBindings) (id : Nat) : Except String String := do
  match bindings.get? id with
  | none => throw s!"IR→PyTorch: missing const binding for node {id}"
  | some b =>
      match b with
      | .bufferFull a => pure s!"self.{a}"
      | .paramFull a => pure s!"self.{a}"

/--
Emit the body of a Python `forward(self, x)` function for a given IR graph.

Each IR node `id` becomes a Python local `v{id}`. We emit nodes in graph order and end by returning
`v{outputId}`.
-/
private def emitForwardBody (g : NN.IR.Graph) (ps : ParamStore Float) (bindings : ConstBindings)
    (inputId outputId : Nat) : Except String (Array String) := do
  -- Validate that input and output nodes exist.
  let _ ← getNode g inputId
  let outNode ← getNode g outputId
  let _ := outNode

  let mut lines : Array String := #[]

  for n in g.nodes do
    let id := n.id
    match n.kind with
    | .input =>
        lines := lines ++ #[indentFour s!"v{id} = x"]
    | .const _ =>
        let e ← constExpr bindings id
        lines := lines ++ #[indentFour s!"v{id} = {e}"]
    | .randUniform seed =>
        let shp := shapeToPyTupleString n.outShape
        lines := lines ++
          #[ indentFour s!"_gen{id} = torch.Generator(device=x.device)"
          , indentFour s!"_gen{id}.manual_seed({seed} + {id})"
          , indentFour
            s!"v{id} = torch.rand({shp}, generator=_gen{id}, device=x.device, dtype=x.dtype)"
          ]
    | .bernoulliMask seed =>
        let p ← expectUnary id n.parents
        let shp := shapeToPyTupleString n.outShape
        let rand :=
          s!"torch.rand({shp}, generator=_gen{id}, device=v{p}.device, dtype=v{p}.dtype)"
        lines := lines ++
          #[ indentFour s!"_gen{id} = torch.Generator(device=v{p}.device)"
          , indentFour s!"_gen{id}.manual_seed({seed} + {id})"
          , indentFour s!"v{id} = ({rand} < v{p}).to(v{p}.dtype)"
          ]
    | .detach =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{p}.detach()"]
    | .add =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{a} + v{b}"]
    | .sub =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{a} - v{b}"]
    | .mulElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{a} * v{b}"]
    | .minElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.minimum(v{a}, v{b})"]
    | .maxElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.maximum(v{a}, v{b})"]
    | .sum =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.sum(v{p})"]
    | .reduceSum axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.sum(v{p}, dim={axis})"]
    | .reduceMean axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.mean(v{p}, dim={axis})"]
    | .broadcastTo _inShape outShape =>
        let p ← expectUnary id n.parents
        let shp := shapeToPyTupleString outShape
        lines := lines ++ #[indentFour s!"v{id} = torch.broadcast_to(v{p}, {shp})"]
    | .matmul =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.matmul(v{a}, v{b})"]
    | .linear =>
        let xId ← expectUnary id n.parents
        match ps.linearWB.get? id with
        | none => throw s!"IR→PyTorch: missing linear params for node {id}"
        | some _ =>
            lines := lines ++
              #[ indentFour
                s!"v{id} = F.linear(v{xId}, self.{linearWAttr id}, self.{linearBAttr id})" ]
    | .conv config =>
        let xId ← expectUnary id n.parents
        match ps.convCfg.get? id with
        | none => throw s!"IR→PyTorch: missing convolution params for node {id}"
        | some _ =>
            let fn ← match config.spatialRank with
              | 1 => pure "conv1d"
              | 2 => pure "conv2d"
              | 3 => pure "conv3d"
              | rank =>
                throw (s!"IR→PyTorch: convolution rank {rank} has no direct "
                  ++ "PyTorch functional operator")
            let wName := convKernelAttr id
            let bName := convBiasAttr id
            let stride := natTensorToPyTuple config.stride
            let dilation := natTensorToPyTuple config.dilation
            let paddingPairs :=
              ((Tensor.to config.padding (List Nat)).zip
                (Tensor.to config.paddingAfter (List Nat))).reverse
            let explicitPadding :=
              "(" ++ ", ".intercalate (paddingPairs.flatMap fun pair =>
                [toString pair.1, toString pair.2]) ++ ")"
            let inChannelsPerGroup := config.inChannels / config.groups
            let outChannelsPerGroup := config.outChannels / config.groups
            let convLine :=
              s!"_y = F.{fn}(_x, _grouped_w, self.{bName}, " ++
                s!"stride={stride}, padding=0, dilation={dilation}, groups={config.groups})"
            lines := lines ++
              #[ indentFour s!"_x = v{xId}"
              , indentFour s!"_prefix = list(_x.shape[:{config.channelAxis}])"
              , indentFour s!"_spatial = list(_x.shape[{config.channelAxis + 1}:])"
              , indentFour s!"_x = _x.reshape((-1, {config.inChannels}, *_spatial))"
              , indentFour s!"_x = F.pad(_x, {explicitPadding})"
              , indentFour <|
                  s!"_grouped_w = torch.cat([self.{wName}[g*{outChannelsPerGroup}:" ++
                    s!"(g+1)*{outChannelsPerGroup}, g*{inChannelsPerGroup}:" ++
                    s!"(g+1)*{inChannelsPerGroup}] for g in range({config.groups})], dim=0)"
              , indentFour convLine
              , indentFour s!"v{id} = _y.reshape((*_prefix, {config.outChannels}, *_y.shape[2:]))"
              ]
    | .batchNormEval channelAxis channels =>
        let xId ← expectUnary id n.parents
        match ps.batchNormEval.get? id with
        | none => throw s!"IR→PyTorch: missing BatchNorm parameters for node {id}"
        | some p =>
            let rank := Shape.toList n.outShape |>.length
            lines := lines ++
              #[ indentFour s!"_x = v{xId}"
              , indentFour s!"_view = [1] * {rank}"
              , indentFour s!"_view[{channelAxis}] = {channels}"
              , indentFour s!"_mean = self.{batchNormMeanAttr id}.reshape(_view)"
              , indentFour s!"_var = self.{batchNormVarAttr id}.reshape(_view)"
              , indentFour s!"_gamma = self.{batchNormGammaAttr id}.reshape(_view)"
              , indentFour s!"_beta = self.{batchNormBetaAttr id}.reshape(_view)"
              , indentFour s!"_eps = {floatToPyString p.eps}"
              , indentFour s!"v{id} = (_x - _mean) * torch.rsqrt(_var + _eps) * _gamma + _beta"
              ]
    | .maxPool config =>
        let xId ← expectUnary id n.parents
        let fn ← match config.spatialRank with
          | 1 => pure "max_pool1d"
          | 2 => pure "max_pool2d"
          | 3 => pure "max_pool3d"
          | rank =>
            throw s!"IR→PyTorch: max-pool rank {rank} has no direct PyTorch functional operator"
        let kernel := natTensorToPyTuple config.kernel
        let stride := natTensorToPyTuple config.stride
        let padding := natTensorToPyTuple config.padding
        lines := lines ++
          #[ indentFour s!"_x = v{xId}"
          , indentFour s!"_prefix = list(_x.shape[:-{config.spatialRank}])"
          , indentFour s!"_spatial = list(_x.shape[-{config.spatialRank}:])"
          , indentFour "_x = _x.reshape((-1, 1, *_spatial))"
          , indentFour s!"_y = F.{fn}(_x, kernel_size={kernel}, stride={stride}, padding={padding})"
          , indentFour s!"v{id} = _y.reshape((*_prefix, *_y.shape[2:]))"
          ]
    | .avgPool config =>
        let xId ← expectUnary id n.parents
        let fn ← match config.spatialRank with
          | 1 => pure "avg_pool1d"
          | 2 => pure "avg_pool2d"
          | 3 => pure "avg_pool3d"
          | rank =>
            throw s!"IR→PyTorch: average-pool rank {rank} has no direct PyTorch functional operator"
        let kernel := natTensorToPyTuple config.kernel
        let stride := natTensorToPyTuple config.stride
        let padding := natTensorToPyTuple config.padding
        lines := lines ++
          #[ indentFour s!"_x = v{xId}"
          , indentFour s!"_prefix = list(_x.shape[:-{config.spatialRank}])"
          , indentFour s!"_spatial = list(_x.shape[-{config.spatialRank}:])"
          , indentFour "_x = _x.reshape((-1, 1, *_spatial))"
          , indentFour (s!"_y = F.{fn}(_x, kernel_size={kernel}, stride={stride}, "
              ++ s!"padding={padding}, count_include_pad=True)")
          , indentFour s!"v{id} = _y.reshape((*_prefix, *_y.shape[2:]))"
          ]
    | .relu =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.relu(v{p})"]
    | .tanh =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.tanh(v{p})"]
    | .sin =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.sin(v{p})"]
    | .cos =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.cos(v{p})"]
    | .sigmoid =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.sigmoid(v{p})"]
    | .softplus =>
        let p ← expectUnary id n.parents
        -- Select the exponent's argument before calling exp. Both tensor branches may be
        -- evaluated by PyTorch, so selecting between two exponentials would still overflow.
        lines := lines ++ #[
          indentFour s!"_positive = v{p} > 0",
          indentFour s!"_tail = torch.log(1 + torch.exp(torch.where(_positive, -v{p}, v{p})))",
          indentFour s!"v{id} = torch.where(_positive, v{p} + _tail, _tail)"
        ]
    | .safeLog =>
        let (p, epsilon) ← expectBinary id n.parents
        lines := lines ++ #[
          indentFour s!"_positive = v{p} > 0",
          indentFour s!"_tail = torch.log(1 + torch.exp(torch.where(_positive, -v{p}, v{p})))",
          indentFour s!"_softplus = torch.where(_positive, v{p} + _tail, _tail)",
          indentFour s!"v{id} = torch.log(_softplus + v{epsilon})"
        ]
    | .exp =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.exp(v{p})"]
    | .log =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.log(v{p})"]
    | .inv =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.reciprocal(v{p})"]
    | .abs =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.abs(v{p})"]
    | .sqrt =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.sqrt(v{p})"]
    | .softmax axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = torch.softmax(v{p}, dim={axis})"]
    | .hardMaskedSoftmax mask =>
        let p ← expectUnary id n.parents
        match NN.IR.HardMask.validateAs mask n.outShape with
        | .ok _ => pure ()
        | .error message => throw s!"IR→PyTorch: node {id}: {message}"
        let values := ", ".intercalate <| mask.allowed.toList.map fun allowed =>
          if allowed then "True" else "False"
        let shape := shapeToPyTupleString n.outShape
        let maskLine :=
          s!"mask{id} = torch.tensor([{values}], dtype=torch.bool, " ++
            s!"device=v{p}.device).reshape({shape})"
        lines := lines ++
          #[ indentFour maskLine
          , indentFour s!"masked{id} = v{p}.masked_fill(~mask{id}, float('-inf'))"
          , indentFour s!"probs{id} = torch.softmax(masked{id}, dim=-1)"
          , indentFour <|
              s!"v{id} = torch.where(mask{id}.any(dim=-1, keepdim=True), probs{id}, " ++
                s!"torch.zeros_like(probs{id}))"
          ]
    | .layernorm axis =>
        let p ← expectUnary id n.parents
        let dims := Shape.toList n.outShape
        let normalized := dims.drop axis
        let normalizedShape :=
          match normalized with
          | [] => "()"
          | [d] => s!"({d},)"
          | _ => "(" ++ ", ".intercalate (normalized.map toString) ++ ")"
        lines := lines ++ #[indentFour
          s!"v{id} = F.layer_norm(v{p}, normalized_shape={normalizedShape})"]
    | .reshape _inShape outShape =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{p}.reshape({shapeToPyTupleString outShape})"]
    | .flatten _ =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{p}.reshape({shapeToPyTupleString n.outShape})"]
    | .permute perm =>
        let p ← expectUnary id n.parents
        if perm.isEmpty then
          lines := lines ++ #[indentFour s!"v{id} = v{p}"]
        else
          let permStr := ", ".intercalate (perm.map toString).toList
          lines := lines ++ #[indentFour s!"v{id} = v{p}.permute({permStr})"]
    | .concat axis =>
        if n.parents.isEmpty then
          throw s!"IR→PyTorch: node {id}: concat expects ≥1 parent"
        else
          let args := ", ".intercalate (n.parents.map (fun p => s!"v{p}")).toList
          lines := lines ++ #[indentFour s!"v{id} = torch.cat([{args}], dim={axis})"]
    | .transpose axis₁ axis₂ =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = v{p}.transpose({axis₁}, {axis₂})"]
    | .mseLoss =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indentFour s!"v{id} = F.mse_loss(v{a}, v{b}, reduction='mean')"]

  lines := lines ++ #[indentFour s!"return v{outputId}"]
  pure lines

/--
Emit a standalone PyTorch `nn.Module` class for an IR graph.

This is the main entrypoint for IR exporters: it bundles:
- imports,
- a class definition with parameters/buffers materialized from the `ParamStore`,
- a `forward` method implementing the IR, and
- (optionally) a compact training file for export checks.
-/
def emit
    (g : NN.IR.Graph) (ps : ParamStore Float) (inputId outputId : Nat)
    (options : Options := {}) :
    Except String String := do
  let inNode ← getNode g inputId
  let outNode ← getNode g outputId
  let inputShape := inNode.outShape
  let outputShape := outNode.outShape

  let (bindings, initParamLines) ← collectBindings g ps options
  let forwardBody ← emitForwardBody g ps bindings inputId outputId

  let imports : Array String :=
    #[ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    ]

  let classHeader : Array String :=
    #[ s!"class {options.className}(nn.Module):"
    , indentTwo "def __init__(self):"
    , indentFour "super().__init__()"
    , indentFour s!"dtype = {options.dtypeExpr}"
    ]

  let classForwardHeader : Array String :=
    #[ ""
    , indentTwo "def forward(self, x):"
    ]

  let classLines : Array String :=
    classHeader ++ initParamLines ++ classForwardHeader ++ forwardBody

  let helpers : Array String :=
    if !options.includeTrainingSkeleton then
      #[]
    else
      let outIsLoss : Bool :=
        match outNode.kind with
        | .mseLoss => true
        | _ => false
      let xTuple := shapeToPyTupleString inputShape
      let yTuple := shapeToPyTupleString outputShape
      let (trainStepDef, mainLines) :=
        if outIsLoss then
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, opt=None):"
            , indentTwo "model.train()"
            , indentTwo "if opt is not None:"
            , indentFour "opt.zero_grad(set_to_none=True)"
            , indentTwo "loss = model(x)"
            , indentTwo "if opt is not None:"
            , indentFour "loss.backward()"
            , indentFour "opt.step()"
            , indentTwo "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indentTwo "torch.manual_seed(0)"
            , indentTwo "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indentTwo s!"model = {options.className}().to(device)"
            , indentTwo "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indentTwo s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indentTwo "loss = train_step(model, x, opt)"
            , indentTwo "print('loss', loss)"
            ] )
        else
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, opt=None):"
            , indentTwo "model.train()"
            , indentTwo "if opt is not None:"
            , indentFour "opt.zero_grad(set_to_none=True)"
            , indentTwo "out = model(x)"
            , indentTwo "loss = F.mse_loss(out, y, reduction='mean')"
            , indentTwo "if opt is not None:"
            , indentFour "loss.backward()"
            , indentFour "opt.step()"
            , indentTwo "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indentTwo "torch.manual_seed(0)"
            , indentTwo "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indentTwo s!"model = {options.className}().to(device)"
            , indentTwo "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indentTwo s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indentTwo s!"y = torch.randn({yTuple}, dtype={options.dtypeExpr}, device=device)"
            , indentTwo "loss = train_step(model, x, y, opt)"
            , indentTwo "print('loss', loss)"
            ] )

      #[ ""
      , "def make_optimizer(model: nn.Module, kind: str = 'adam', lr: float = 1e-3):"
      , indentTwo "params = [p for p in model.parameters() if p.requires_grad]"
      , indentTwo "if len(params) == 0:"
      , indentFour "return None"
      , indentTwo "if kind == 'sgd':"
      , indentFour "return torch.optim.SGD(params, lr=lr)"
      , indentTwo "if kind == 'adam':"
      , indentFour "return torch.optim.Adam(params, lr=lr)"
      , indentTwo "raise ValueError(f'unknown optimizer kind: {kind}')"
      ] ++ trainStepDef ++ mainLines

  pure (joinLines (imports ++ classLines ++ helpers))

end IRPyTorch
end Export
