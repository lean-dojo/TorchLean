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

* Node ids index `g.nodes` consistently; `node` checks that invariant.
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
def flat {s : Shape} (t : Tensor Float s) : String :=
  tensorLiteral (Tensor.flattenSpec (α := Float) t)

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

/-- Map from IR constant node id to its Python attribute name, for both buffers and parameters. -/
abbrev Bindings := HashMap Nat String

/-- Python attribute name for a node and optional parameter field.
The spelling is shared by initialization and forward evaluation, including saved state keys. -/
def attr (kind : String) (id : Nat) (field : String := "") : String :=
  s!"{kind}_{id}" ++ if field.isEmpty then "" else "_" ++ field

/-- Emit a `torch.tensor([...]).reshape(...)` expression from a flat Python list literal and a
  `Shape`. -/
def tensor (flatList : String) (shape : Shape) (dtypeVar : String := "dtype") : String :=
  s!"torch.tensor({flatList}, dtype={dtypeVar}).reshape({shapeLiteral shape})"

/--
Retrieve a node from the graph and validate its id invariant.

This yields more actionable error messages than directly indexing into `g.nodes`.
-/
def node (g : NN.IR.Graph) (id : Nat) : Except String NN.IR.Node := do
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
private def affineParameters (g : NN.IR.Graph) : Std.HashSet Nat :=
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
private def bindings (g : NN.IR.Graph) (ps : ParamStore Float) (options : Options) :
    Except String (Bindings × Array String) := do
  let mut bindings : Bindings := HashMap.emptyWithCapacity
  let mut initLines : Array String := #[]
  let forcedLearnableConsts := affineParameters g

  -- Pass 1: linear and convolution parameters (stored in ParamStore keyed by node id).
  for n in g.nodes do
    match n.kind with
    | .linear =>
        match ps.linearWB.get? n.id with
        | none => throw s!"IR→PyTorch: missing linear params for node {n.id}"
        | some lp =>
            let wShape : Shape := .dim lp.m (.dim lp.n .scalar)
            let bShape : Shape := .dim lp.m .scalar
            let wFlat := flat (s := wShape) lp.w
            let bFlat := flat (s := bShape) lp.b
            initLines := initLines ++
              #[ indent 4
                s!"self.{attr "linear" n.id "W"} = nn.Parameter({tensor wFlat wShape})"
              , indent 4
                s!"self.{attr "linear" n.id "b"} = nn.Parameter({tensor bFlat bShape})"
              ]
    | .conv .. =>
        match ps.convCfg.get? n.id with
        | none => throw s!"IR→PyTorch: missing convolution params for node {n.id}"
        | some config =>
            let kShape : Shape :=
              .dim config.outChannels
                (.dim config.inChannels (Shape.ofList (Tensor.to config.kernel (List Nat))))
            let bShape : Shape := .dim config.outChannels .scalar
            let kFlat := flat (s := kShape) config.spec.kernel
            let bFlat := flat (s := bShape) config.spec.bias
            initLines := initLines ++
              #[ indent 4
                s!"self.{attr "conv" n.id "kernel"} = nn.Parameter({tensor kFlat kShape})"
              , indent 4
                s!"self.{attr "conv" n.id "bias"} = nn.Parameter({tensor bFlat bShape})"
              ]
    | .batchNormEval .. =>
        match ps.batchNormEval.get? n.id with
        | none => throw s!"IR→PyTorch: missing BatchNorm parameters for node {n.id}"
        | some p =>
            let sC : Shape := .dim p.c .scalar
            let gamma := tensor (flat (s := sC) p.gamma) sC
            let beta := tensor (flat (s := sC) p.beta) sC
            let mean := tensor (flat (s := sC) p.mean) sC
            let var := tensor (flat (s := sC) p.var) sC
            initLines := initLines ++
              #[ indent 4 s!"self.{attr "batchnorm" n.id "gamma"} = nn.Parameter({gamma})"
              , indent 4 s!"self.{attr "batchnorm" n.id "beta"} = nn.Parameter({beta})"
              , indent 4 s!"self.register_buffer(\"{attr "batchnorm" n.id "mean"}\", {mean})"
              , indent 4 s!"self.register_buffer(\"{attr "batchnorm" n.id "var"}\", {var})"
              ]
    | _ => pure ()

  -- Pass 2: const nodes (stored as flattened values in ParamStore.constVals).
  for n in g.nodes do
    match n.kind with
    | .const s =>
        match ps.constVals.get? n.id with
        | none => throw s!"IR→PyTorch: missing const value for node {n.id}"
        | some fv =>
            let flatListStr := tensorLiteral fv.v
            let flatVals := Tensor.to fv.v (Array Float)
            let uniform := allEq flatVals
            let forceLearn :=
              options.learnableConsts && forcedLearnableConsts.contains n.id
            let shouldLearn := options.learnableConsts && (forceLearn || !uniform)
            if shouldLearn then
              let attr := attr "const" n.id
              initLines := initLines ++
                #[ indent 4 s!"self.{attr} = nn.Parameter({tensor flatListStr s})" ]
              bindings := bindings.insert n.id attr
            else
              let attr := attr "const" n.id
              initLines := initLines ++
                #[ indent 4
                  s!"self.register_buffer(\"{attr}\", {tensor flatListStr s})" ]
              bindings := bindings.insert n.id attr
    | _ => pure ()

  pure (bindings, initLines)

/-- Emit the Python expression used to reference a bound constant (`self.<attr>`). -/
private def constant (bindings : Bindings) (id : Nat) : Except String String := do
  match bindings.get? id with
  | none => throw s!"IR→PyTorch: missing const binding for node {id}"
  | some attr => pure s!"self.{attr}"

/--
Emit the body of a Python `forward(self, x)` function for a given IR graph.

Each IR node `id` becomes a Python local `v{id}`. We emit nodes in graph order and end by returning
`v{outputId}`.
-/
private def forward (g : NN.IR.Graph) (ps : ParamStore Float) (bindings : Bindings)
    (outputId : Nat) : Except String (Array String) := do
  let mut lines : Array String := #[]

  for n in g.nodes do
    let id := n.id
    match n.kind with
    | .input =>
        lines := lines ++ #[indent 4 s!"v{id} = x"]
    | .custom name .. =>
        throw s!"PyTorch export: custom {name} at node {id} has no Python body translation"
    | .const _ =>
        let e ← constant bindings id
        lines := lines ++ #[indent 4 s!"v{id} = {e}"]
    | .randUniform seed =>
        let shp := shapeLiteral n.outShape
        lines := lines ++
          #[ indent 4 s!"_gen{id} = torch.Generator(device=x.device)"
          , indent 4 s!"_gen{id}.manual_seed({seed} + {id})"
          , indent 4
            s!"v{id} = torch.rand({shp}, generator=_gen{id}, device=x.device, dtype=x.dtype)"
          ]
    | .bernoulliMask seed =>
        let p ← expectUnary id n.parents
        let shp := shapeLiteral n.outShape
        let rand :=
          s!"torch.rand({shp}, generator=_gen{id}, device=v{p}.device, dtype=v{p}.dtype)"
        lines := lines ++
          #[ indent 4 s!"_gen{id} = torch.Generator(device=v{p}.device)"
          , indent 4 s!"_gen{id}.manual_seed({seed} + {id})"
          , indent 4 s!"v{id} = ({rand} < v{p}).to(v{p}.dtype)"
          ]
    | .detach =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{p}.detach()"]
    | .add =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{a} + v{b}"]
    | .sub =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{a} - v{b}"]
    | .mulElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{a} * v{b}"]
    | .minElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.minimum(v{a}, v{b})"]
    | .maxElem =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.maximum(v{a}, v{b})"]
    | .sum =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.sum(v{p})"]
    | .reduceSum axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.sum(v{p}, dim={axis})"]
    | .reduceMean axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.mean(v{p}, dim={axis})"]
    | .broadcastTo _inShape outShape =>
        let p ← expectUnary id n.parents
        let shp := shapeLiteral outShape
        lines := lines ++ #[indent 4 s!"v{id} = torch.broadcast_to(v{p}, {shp})"]
    | .matmul =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.matmul(v{a}, v{b})"]
    | .linear =>
        let xId ← expectUnary id n.parents
        match ps.linearWB.get? id with
        | none => throw s!"IR→PyTorch: missing linear params for node {id}"
        | some _ =>
            let weight := attr "linear" id "W"
            let bias := attr "linear" id "b"
            lines := lines ++
              #[ indent 4
                s!"v{id} = F.linear(v{xId}, self.{weight}, self.{bias})" ]
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
            let wName := attr "conv" id "kernel"
            let bName := attr "conv" id "bias"
            let stride := tupleLiteral config.stride
            let dilation := tupleLiteral config.dilation
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
              #[ indent 4 s!"_x = v{xId}"
              , indent 4 s!"_prefix = list(_x.shape[:{config.channelAxis}])"
              , indent 4 s!"_spatial = list(_x.shape[{config.channelAxis + 1}:])"
              , indent 4 s!"_x = _x.reshape((-1, {config.inChannels}, *_spatial))"
              , indent 4 s!"_x = F.pad(_x, {explicitPadding})"
              , indent 4 <|
                  s!"_grouped_w = torch.cat([self.{wName}[g*{outChannelsPerGroup}:" ++
                    s!"(g+1)*{outChannelsPerGroup}, g*{inChannelsPerGroup}:" ++
                    s!"(g+1)*{inChannelsPerGroup}] for g in range({config.groups})], dim=0)"
              , indent 4 convLine
              , indent 4 s!"v{id} = _y.reshape((*_prefix, {config.outChannels}, *_y.shape[2:]))"
              ]
    | .batchNormEval channelAxis channels =>
        let xId ← expectUnary id n.parents
        match ps.batchNormEval.get? id with
        | none => throw s!"IR→PyTorch: missing BatchNorm parameters for node {id}"
        | some p =>
            let rank := Shape.toList n.outShape |>.length
            lines := lines ++
              #[ indent 4 s!"_x = v{xId}"
              , indent 4 s!"_view = [1] * {rank}"
              , indent 4 s!"_view[{channelAxis}] = {channels}"
              , indent 4 s!"_mean = self.{attr "batchnorm" id "mean"}.reshape(_view)"
              , indent 4 s!"_var = self.{attr "batchnorm" id "var"}.reshape(_view)"
              , indent 4 s!"_gamma = self.{attr "batchnorm" id "gamma"}.reshape(_view)"
              , indent 4 s!"_beta = self.{attr "batchnorm" id "beta"}.reshape(_view)"
              , indent 4 s!"_eps = {floatLiteral p.eps}"
              , indent 4 s!"v{id} = (_x - _mean) * torch.rsqrt(_var + _eps) * _gamma + _beta"
              ]
    | .maxPool config =>
        let xId ← expectUnary id n.parents
        let fn ← match config.spatialRank with
          | 1 => pure "max_pool1d"
          | 2 => pure "max_pool2d"
          | 3 => pure "max_pool3d"
          | rank =>
            throw s!"IR→PyTorch: max-pool rank {rank} has no direct PyTorch functional operator"
        let kernel := tupleLiteral config.kernel
        let stride := tupleLiteral config.stride
        let padding := tupleLiteral config.padding
        lines := lines ++
          #[ indent 4 s!"_x = v{xId}"
          , indent 4 s!"_prefix = list(_x.shape[:-{config.spatialRank}])"
          , indent 4 s!"_spatial = list(_x.shape[-{config.spatialRank}:])"
          , indent 4 "_x = _x.reshape((-1, 1, *_spatial))"
          , indent 4 s!"_y = F.{fn}(_x, kernel_size={kernel}, stride={stride}, padding={padding})"
          , indent 4 s!"v{id} = _y.reshape((*_prefix, *_y.shape[2:]))"
          ]
    | .avgPool config =>
        let xId ← expectUnary id n.parents
        let fn ← match config.spatialRank with
          | 1 => pure "avg_pool1d"
          | 2 => pure "avg_pool2d"
          | 3 => pure "avg_pool3d"
          | rank =>
            throw s!"IR→PyTorch: average-pool rank {rank} has no direct PyTorch functional operator"
        let kernel := tupleLiteral config.kernel
        let stride := tupleLiteral config.stride
        let padding := tupleLiteral config.padding
        lines := lines ++
          #[ indent 4 s!"_x = v{xId}"
          , indent 4 s!"_prefix = list(_x.shape[:-{config.spatialRank}])"
          , indent 4 s!"_spatial = list(_x.shape[-{config.spatialRank}:])"
          , indent 4 "_x = _x.reshape((-1, 1, *_spatial))"
          , indent 4 (s!"_y = F.{fn}(_x, kernel_size={kernel}, stride={stride}, "
              ++ s!"padding={padding}, count_include_pad=True)")
          , indent 4 s!"v{id} = _y.reshape((*_prefix, *_y.shape[2:]))"
          ]
    | .relu =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.relu(v{p})"]
    | .tanh =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.tanh(v{p})"]
    | .sin =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.sin(v{p})"]
    | .cos =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.cos(v{p})"]
    | .sigmoid =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.sigmoid(v{p})"]
    | .softplus =>
        let p ← expectUnary id n.parents
        -- Select the exponent's argument before calling exp. Both tensor branches may be
        -- evaluated by PyTorch, so selecting between two exponentials would still overflow.
        lines := lines ++ #[
          indent 4 s!"_positive = v{p} > 0",
          indent 4 s!"_tail = torch.log(1 + torch.exp(torch.where(_positive, -v{p}, v{p})))",
          indent 4 s!"v{id} = torch.where(_positive, v{p} + _tail, _tail)"
        ]
    | .safeLog =>
        let (p, epsilon) ← expectBinary id n.parents
        lines := lines ++ #[
          indent 4 s!"_positive = v{p} > 0",
          indent 4 s!"_tail = torch.log(1 + torch.exp(torch.where(_positive, -v{p}, v{p})))",
          indent 4 s!"_softplus = torch.where(_positive, v{p} + _tail, _tail)",
          indent 4 s!"v{id} = torch.log(_softplus + v{epsilon})"
        ]
    | .exp =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.exp(v{p})"]
    | .log =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.log(v{p})"]
    | .inv =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.reciprocal(v{p})"]
    | .abs =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.abs(v{p})"]
    | .sqrt =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.sqrt(v{p})"]
    | .softmax axis =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = torch.softmax(v{p}, dim={axis})"]
    | .hardMaskedSoftmax mask =>
        let p ← expectUnary id n.parents
        match NN.IR.HardMask.validateAs mask n.outShape with
        | .ok _ => pure ()
        | .error message => throw s!"IR→PyTorch: node {id}: {message}"
        let values := ", ".intercalate <| mask.allowed.toList.map boolLiteral
        let shape := shapeLiteral n.outShape
        let maskLine :=
          s!"mask{id} = torch.tensor([{values}], dtype=torch.bool, " ++
            s!"device=v{p}.device).reshape({shape})"
        lines := lines ++
          #[ indent 4 maskLine
          , indent 4 s!"masked{id} = v{p}.masked_fill(~mask{id}, float('-inf'))"
          , indent 4 s!"probs{id} = torch.softmax(masked{id}, dim=-1)"
          , indent 4 <|
              s!"v{id} = torch.where(mask{id}.any(dim=-1, keepdim=True), probs{id}, " ++
                s!"torch.zeros_like(probs{id}))"
          ]
    | .layernorm axis =>
        let p ← expectUnary id n.parents
        let dims := Shape.toList n.outShape
        let normalized := dims.drop axis
        let normalizedShape := shapeLiteral (Shape.ofList normalized)
        lines := lines ++ #[indent 4
          s!"v{id} = F.layer_norm(v{p}, normalized_shape={normalizedShape})"]
    | .reshape _inShape outShape =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{p}.reshape({shapeLiteral outShape})"]
    | .flatten _ =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{p}.reshape({shapeLiteral n.outShape})"]
    | .permute perm =>
        let p ← expectUnary id n.parents
        if perm.isEmpty then
          lines := lines ++ #[indent 4 s!"v{id} = v{p}"]
        else
          let permStr := ", ".intercalate (perm.map toString).toList
          lines := lines ++ #[indent 4 s!"v{id} = v{p}.permute({permStr})"]
    | .concat axis =>
        if n.parents.isEmpty then
          throw s!"IR→PyTorch: node {id}: concat expects ≥1 parent"
        else
          let args := ", ".intercalate (n.parents.map (fun p => s!"v{p}")).toList
          lines := lines ++ #[indent 4 s!"v{id} = torch.cat([{args}], dim={axis})"]
    | .transpose axis₁ axis₂ =>
        let p ← expectUnary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = v{p}.transpose({axis₁}, {axis₂})"]
    | .mseLoss =>
        let (a, b) ← expectBinary id n.parents
        lines := lines ++ #[indent 4 s!"v{id} = F.mse_loss(v{a}, v{b}, reduction='mean')"]

  lines := lines ++ #[indent 4 s!"return v{outputId}"]
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
  let inNode ← node g inputId
  let outNode ← node g outputId
  let inputShape := inNode.outShape
  let outputShape := outNode.outShape

  let (bindings, initParamLines) ← bindings g ps options
  let forwardBody ← forward g ps bindings outputId

  let imports : Array String :=
    #[ "import torch"
    , "import torch.nn as nn"
    , "import torch.nn.functional as F"
    , ""
    ]

  let classHeader : Array String :=
    #[ s!"class {options.className}(nn.Module):"
    , indent 2 "def __init__(self):"
    , indent 4 "super().__init__()"
    , indent 4 s!"dtype = {options.dtypeExpr}"
    ]

  let classForwardHeader : Array String :=
    #[ ""
    , indent 2 "def forward(self, x):"
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
      let xTuple := shapeLiteral inputShape
      let yTuple := shapeLiteral outputShape
      let (trainStepDef, mainLines) :=
        if outIsLoss then
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, opt=None):"
            , indent 2 "model.train()"
            , indent 2 "if opt is not None:"
            , indent 4 "opt.zero_grad(set_to_none=True)"
            , indent 2 "loss = model(x)"
            , indent 2 "if opt is not None:"
            , indent 4 "loss.backward()"
            , indent 4 "opt.step()"
            , indent 2 "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indent 2 "torch.manual_seed(0)"
            , indent 2 "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indent 2 s!"model = {options.className}().to(device)"
            , indent 2 "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indent 2 s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 "loss = train_step(model, x, opt)"
            , indent 2 "print('loss', loss)"
            ] )
        else
          ( #[ ""
            , "def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor, opt=None):"
            , indent 2 "model.train()"
            , indent 2 "if opt is not None:"
            , indent 4 "opt.zero_grad(set_to_none=True)"
            , indent 2 "out = model(x)"
            , indent 2 "loss = F.mse_loss(out, y, reduction='mean')"
            , indent 2 "if opt is not None:"
            , indent 4 "loss.backward()"
            , indent 4 "opt.step()"
            , indent 2 "return float(loss.detach().cpu())"
            ]
          , #[ ""
            , "if __name__ == '__main__':"
            , indent 2 "torch.manual_seed(0)"
            , indent 2 "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')"
            , indent 2 s!"model = {options.className}().to(device)"
            , indent 2 "opt = make_optimizer(model, kind='adam', lr=1e-3)"
            , indent 2 s!"x = torch.randn({xTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 s!"y = torch.randn({yTuple}, dtype={options.dtypeExpr}, device=device)"
            , indent 2 "loss = train_step(model, x, y, opt)"
            , indent 2 "print('loss', loss)"
            ] )

      #[ ""
      , "def make_optimizer(model: nn.Module, kind: str = 'adam', lr: float = 1e-3):"
      , indent 2 "params = [p for p in model.parameters() if p.requires_grad]"
      , indent 2 "if len(params) == 0:"
      , indent 4 "return None"
      , indent 2 "if kind == 'sgd':"
      , indent 4 "return torch.optim.SGD(params, lr=lr)"
      , indent 2 "if kind == 'adam':"
      , indent 4 "return torch.optim.Adam(params, lr=lr)"
      , indent 2 "raise ValueError(f'unknown optimizer kind: {kind}')"
      ] ++ trainStepDef ++ mainLines

  pure (joinLines (imports ++ classLines ++ helpers))

end IRPyTorch
end Export
