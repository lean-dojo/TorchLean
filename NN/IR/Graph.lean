/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape

/-!
# IR Graph

`NN.IR.Graph` is TorchLean’s canonical *op-tagged* DAG IR.

Today it is used as the shared target for:
- TorchLean → verifier compilation (`NN/Verification/TorchLean/Compile.lean`),
- bound-propagation / verification tooling (CROWN/LiRPA) (`NN/MLTheory/CROWN/Graph.lean`),
- IR → PyTorch emission (`NN/Runtime/PyTorch/Export/IRPyTorch.lean`),
- small tutorial graphs (e.g. `NN/Examples/Advanced/GraphSpec/Tutorial.lean`).

Longer-term, the intent is to use the same IR as a bridge target for:
- spec-level graphs (compile a model spec to an IR graph),
- runtime autograd traces (reify a runtime tape/graph into the same IR),
- verifiers (IBP/CROWN/affine passes) and export tooling.

What this file commits to is the graph *structure* (ops, dependencies, and shapes). Parameter
payloads (weights/bias/const values) live in backend-specific stores keyed by node id. This split
keeps one graph format usable across:

- verification (where parameters often carry additional metadata like bounds or perturbation sets),
- export (where parameters may be emitted as PyTorch `nn.Parameter`s or ONNX initializers),
- and runtime execution/tracing (where parameters may already live in a separate module state).

If you are coming from PyTorch: the mental model is similar to a PyTorch FX graph or TorchScript IR:
nodes are ops, edges are “data dependencies”, and execution is in topological order. The difference
is that TorchLean attaches explicit *shape* metadata at every node, since our verification and
proof tooling needs shape information to be first-class.

References / related systems:
- PyTorch FX docs: https://pytorch.org/docs/stable/fx.html
- TorchScript overview: https://pytorch.org/docs/stable/jit.html
- ONNX (graph + initializers as separate parameter store): https://onnx.ai/

## Conventions (important)

- **Topo order**: a node only references parents with smaller ids.
- **Id discipline**: in most builders, `node.id` is expected to equal its index in `Graph.nodes`.
  (E.g. the TorchLean compiler uses `freshId := nodes.size` and then appends.)
- **External parameters**:
  - `OpKind.const` stores its `valueShape` here, but the constant value is stored externally
    (e.g. in a verifier `ParamStore` keyed by node id).
  - Some ops (notably `OpKind.linear` and `OpKind.conv2d`) typically use *external* parameter stores
    keyed by node id; in those cases the node’s `parents` list only contains the *runtime inputs*
    (e.g. the activation input `x`), not the weights/bias tensors.

This file does **not** implement evaluation or shape inference. Those live in:
- `NN/IR/Semantics.lean` (evaluation semantics for a chosen scalar backend),
- `NN/IR/Infer.lean` / `NN/IR/Check.lean` (shape inference/checking utilities),
- and backend-specific passes (verification/export) that interpret `OpKind` in their own setting.
-/

@[expose] public section


namespace NN.IR

open Spec

/-- Operation kinds in an op-tagged computation graph. -/
inductive OpKind where
  | input
      -- Designated graph input (analogous to a PyTorch FX graph input).
  | const (valueShape : Shape)
      -- Constant tensor. We record the shape here, but keep the *value* in an external store
      -- (e.g. verifier parameters, exporter initializers).
  | permute (perm : List Nat)
      -- Permute axes (0-based). Similar to `torch.permute`.
  | detach
      -- Identity in the forward pass; marks a gradient stop at runtime (analogous to
      -- `Tensor.detach()`).
  | randUniform (seed : Nat)
      -- Deterministic U[0,1) tensor (seeded). We keep RNG explicit because verification needs a
      -- stable, replayable source of “randomness”.
  | bernoulliMask (seed : Nat)
      -- Deterministic {0,1} mask (seeded); parent is keepProb : scalar.
      -- This is the IR-level representation we use for dropout-style masks.
  | add
      -- Elementwise addition (broadcasting is explicit via `broadcastTo`).
  | sub
      -- Elementwise subtraction.
  | mul_elem
      -- Elementwise multiplication.
  | abs
      -- Elementwise absolute value.
  | sqrt
      -- Elementwise square root (ties to the scalar backend semantics).
  | inv
      -- Elementwise reciprocal (1/x).
  | maxElem
      -- Elementwise max.
  | minElem
      -- Elementwise min.
  | maxPool2d (kH kW stride : Nat)
      -- Max pool over CHW (no padding). Similar to `torch.nn.functional.max_pool2d`.
  | maxPool2dPad (kH kW stride padding : Nat)
      -- Max pool over CHW (symmetric zero padding).
  | avgPool2d (kH kW stride : Nat)
      -- Average pool over CHW (no padding).
  | avgPool2dPad (kH kW stride padding : Nat)
      -- Average pool over CHW (symmetric zero padding).
  | broadcastTo (s₁ s₂ : Shape)
      -- Broadcast parent from s₁→s₂ (analogous to `torch.broadcast_to` / `Tensor.expand`).
  | reduceSum (axis : Nat)
      -- Sum along an axis (axis must be valid).
  | reduceMean (axis : Nat)
      -- Mean along an axis (axis must be valid).
  | sum
      -- Sum reduction to scalar (convenience op used by some loss/verification code paths).
  | matmul
      -- 2D matrix multiply (similar to `torch.matmul` in the rank-2 case).
  | linear
      -- Affine layer `y = W x + b`. Parameters live in an external store keyed by node id;
      -- the sole parent is the activation input `x`.
  | conv2d (inC outC kH kW stride padding : Nat)
      -- 2D convolution (NCHW-style). Parameters live in an external store keyed by node id.
  | relu | tanh | sigmoid | exp | log | sin | cos
      -- Common elementwise activations / nonlinearities.
  | softmax (axis : Nat)
      -- Softmax along an axis.
  | layernorm (axis : Nat)
      -- LayerNorm over the suffix of dimensions starting at `axis`.
      -- PyTorch analogue: `F.layer_norm(x, normalized_shape=x.shape[axis:])`.
      -- Note: this IR node is the *pure* normalization (gamma=1, beta=0); the common affine form
      -- is typically represented by surrounding `mul_elem`/`add` nodes with broadcasted constants.
  | reshape (inShape outShape : Shape)
      -- Pure reshape (no data movement).
  | flatten (s : Shape)
      -- Flatten to a vector of length `Shape.size s`.
  | concat (axis : Nat)
      -- Concatenate along an axis (verifier/export may allow an arbitrary number of parents ≥ 2).
  | swap_first_two
      -- Swap the outermost two axes (rank ≥ 2).
  | transpose3dLastTwo
      -- Swap the last two axes (rank = 3).
  | mseLoss
      -- Scalar mean squared error (used in some training/verification examples).
  deriving Repr

namespace OpKind

/--
The minimum number of parent nodes expected by an `OpKind`.

This is a *structural* convention only.

For example, `linear` has arity 1 here because weights/biases are typically stored externally and
keyed by the node id; the only data dependency in the graph itself is the activation input.
-/
def minParents : OpKind → Nat
  | .input => 0
  | .const _ => 0
  | .permute .. => 1
  | .detach => 1
  | .randUniform .. => 0
  | .bernoulliMask .. => 1
  | .add => 2
  | .sub => 2
  | .mul_elem => 2
  | .abs => 1
  | .sqrt => 1
  | .inv => 1
  | .maxElem => 2
  | .minElem => 2
  | .maxPool2d .. => 1
  | .maxPool2dPad .. => 1
  | .avgPool2d .. => 1
  | .avgPool2dPad .. => 1
  | .broadcastTo .. => 1
  | .reduceSum .. => 1
  | .reduceMean .. => 1
  | .sum => 1
  | .matmul => 2
  | .linear => 1
  | .conv2d .. => 1
  | .relu => 1
  | .tanh => 1
  | .sigmoid => 1
  | .exp => 1
  | .log => 1
  | .sin => 1
  | .cos => 1
  | .softmax .. => 1
  | .layernorm .. => 1
  | .reshape .. => 1
  | .flatten .. => 1
  | .concat .. => 2
  | .swap_first_two => 1
  | .transpose3dLastTwo => 1
  | .mseLoss => 2

/--
An optional maximum number of parent nodes expected by an `OpKind`.

For `concat`, the verifier permits an arbitrary number of inputs (at least 2), so this returns
`none`.
-/
def maxParents? : OpKind → Option Nat
  | .concat .. => none
  | k => some (minParents k)

/-- A short human-readable tag for error messages and debugging output. -/
def tag : OpKind → String
  | .input => "input"
  | .const .. => "const"
  | .permute .. => "permute"
  | .detach => "detach"
  | .randUniform .. => "rand_uniform"
  | .bernoulliMask .. => "bernoulli_mask"
  | .add => "add"
  | .sub => "sub"
  | .mul_elem => "mul_elem"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .inv => "inv"
  | .maxElem => "max_elem"
  | .minElem => "min_elem"
  | .maxPool2d .. => "max_pool2d"
  | .maxPool2dPad .. => "max_pool2d_pad"
  | .avgPool2d .. => "avg_pool2d"
  | .avgPool2dPad .. => "avg_pool2d_pad"
  | .broadcastTo .. => "broadcastTo"
  | .reduceSum .. => "reduce_sum"
  | .reduceMean .. => "reduce_mean"
  | .sum => "sum"
  | .matmul => "matmul"
  | .linear => "linear"
  | .conv2d .. => "conv2d"
  | .relu => "relu"
  | .tanh => "tanh"
  | .sigmoid => "sigmoid"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .softmax .. => "softmax"
  | .layernorm .. => "layernorm"
  | .reshape .. => "reshape"
  | .flatten .. => "flatten"
  | .concat .. => "concat"
  | .swap_first_two => "swap_first_two"
  | .transpose3dLastTwo => "transpose3d_last_two"
  | .mseLoss => "mse_loss"

/--
Human-facing operation description including operation-local parameters.

`tag` is intentionally short and stable for grouping/log filtering. `describe` is for diagnostics:
it prints axes, shapes, seeds, and convolution/pooling metadata so malformed graph dumps are useful
without cross-referencing the original builder.
-/
def describe : OpKind → String
  | .input => "input"
  | .const valueShape => s!"const(shape={repr valueShape})"
  | .permute perm => s!"permute(perm={repr perm})"
  | .detach => "detach"
  | .randUniform seed => s!"rand_uniform(seed={seed})"
  | .bernoulliMask seed => s!"bernoulli_mask(seed={seed})"
  | .add => "add"
  | .sub => "sub"
  | .mul_elem => "mul_elem"
  | .abs => "abs"
  | .sqrt => "sqrt"
  | .inv => "inv"
  | .maxElem => "max_elem"
  | .minElem => "min_elem"
  | .maxPool2d kH kW stride => s!"max_pool2d(kH={kH}, kW={kW}, stride={stride})"
  | .maxPool2dPad kH kW stride padding =>
      s!"max_pool2d_pad(kH={kH}, kW={kW}, stride={stride}, padding={padding})"
  | .avgPool2d kH kW stride => s!"avg_pool2d(kH={kH}, kW={kW}, stride={stride})"
  | .avgPool2dPad kH kW stride padding =>
      s!"avg_pool2d_pad(kH={kH}, kW={kW}, stride={stride}, padding={padding})"
  | .broadcastTo s₁ s₂ => s!"broadcastTo(from={repr s₁}, to={repr s₂})"
  | .reduceSum axis => s!"reduce_sum(axis={axis})"
  | .reduceMean axis => s!"reduce_mean(axis={axis})"
  | .sum => "sum"
  | .matmul => "matmul"
  | .linear => "linear(payload=node_id)"
  | .conv2d inC outC kH kW stride padding =>
      s!"conv2d(inC={inC}, outC={outC}, kH={kH}, kW={kW}, stride={stride}, padding={padding})"
  | .relu => "relu"
  | .tanh => "tanh"
  | .sigmoid => "sigmoid"
  | .exp => "exp"
  | .log => "log"
  | .sin => "sin"
  | .cos => "cos"
  | .softmax axis => s!"softmax(axis={axis})"
  | .layernorm axis => s!"layernorm(axis={axis})"
  | .reshape inShape outShape => s!"reshape(from={repr inShape}, to={repr outShape})"
  | .flatten s => s!"flatten(shape={repr s})"
  | .concat axis => s!"concat(axis={axis})"
  | .swap_first_two => "swap_first_two"
  | .transpose3dLastTwo => "transpose3d_last_two"
  | .mseLoss => "mse_loss"

end OpKind

/-- Node in the graph. Edges are implicit via parent indices. -/
structure Node where
  /-- Node id. By convention this is also the node's index in `Graph.nodes`. -/
  id       : Nat
  /-- Parent node ids, i.e. data dependencies. Each parent must be smaller than `id`. -/
  parents  : List Nat
  /-- Operation tag and any operation-local metadata. -/
  kind     : OpKind
  /-- Declared output shape. `NN.IR.Infer` can recompute/check this from parents. -/
  outShape : Shape
  deriving Repr

namespace Node

/-- Check the basic parent-count convention for this node kind. -/
def hasValidArity (n : Node) : Bool :=
  let p := n.parents.length
  match n.kind.maxParents? with
  | some hi => (n.kind.minParents ≤ p) && (p ≤ hi)
  | none => (n.kind.minParents ≤ p)

/--
Check that every parent id is strictly smaller than this node id (topological order).

This is the single most important invariant for the IR:
- it guarantees acyclicity,
- it makes evaluation/inference a simple left-to-right pass,
- and it makes backends predictable (no hidden recursion or “graph rewriting during execution”).
-/
def parentsBelow (n : Node) : Bool :=
  n.parents.all (fun pid => pid < n.id)

/-- Render a compact, user-facing summary (useful in error messages). -/
def summary (n : Node) : String :=
  s!"Node(id={n.id}, kind={n.kind.describe}, parents={n.parents}, outShape={repr n.outShape})"

end Node

/-- Entire graph as an array of nodes. Parents must have smaller ids (topo order). -/
structure Graph where
  /-- nodes. -/
  nodes : Array Node
  deriving Repr

namespace Graph

/-- Number of nodes in the graph. -/
def size (g : Graph) : Nat :=
  g.nodes.size

/-- Safe node lookup by id (treating ids as array indices). -/
def getNode? (g : Graph) (id : Nat) : Option Node :=
  g.nodes[id]?

/--
Total node lookup that enforces the common "id discipline" invariant (`nodes[id].id = id`).

This is convenient for backends that treat node ids as array indices (verifiers, exporters, pretty
printers). The error message is meant to point to a builder bug rather than a user error.
-/
def getNode (g : Graph) (id : Nat) : Except String Node := do
  match g.getNode? id with
  | none => throw s!"IR graph: node id out of bounds: {id}"
  | some n =>
      if n.id != id then
        throw s!"IR graph: internal error: nodes[{id}].id = {n.id} (expected {id})"
      pure n

/-- Safe outShape lookup by id. -/
def outShape? (g : Graph) (id : Nat) : Option Shape :=
  (getNode? g id).map (·.outShape)

/--
Explain why `Node.hasValidArity` failed.

This is intentionally *stringly-typed*; it is primarily meant for human-facing error messages.
-/
def arityError (n : Node) : String :=
  let got := n.parents.length
  match n.kind.maxParents? with
  | some hi =>
      s!"bad parent count for {n.kind.tag}: expected {n.kind.minParents}..{hi}, got {got}"
  | none =>
      s!"bad parent count for {n.kind.tag}: expected at least {n.kind.minParents}, got {got}"

/--
Basic well-formedness check used by verifier code paths.

This checks:
- node ids match array indices (common construction invariant),
- each node respects its op arity convention, and
- all parent ids are strictly smaller than the node id (topological order).

We keep this as a boolean predicate because some passes want a fast “yes/no” filter. If you need a
human-facing error, use `checkWellFormed`.
-/
def wellFormed (g : Graph) : Bool :=
  (List.finRange g.nodes.size).all (fun i =>
    match g.nodes[i]? with
    | none => false
    | some n => (n.id = i) && n.hasValidArity && n.parentsBelow)

/--
Like `wellFormed`, but returns a helpful error message on failure.

This is useful when you want a *clean* user error rather than a silent `false`.
-/
def checkWellFormed (g : Graph) : Except String Unit := do
  for i in [0:g.nodes.size] do
    match g.nodes[i]? with
    | none =>
        throw s!"IR graph: internal error: missing node at index {i}"
    | some n =>
        if n.id != i then
          throw s!"IR graph: id discipline violated at index {i}: nodes[{i}].id = {n.id}"
        if !n.hasValidArity then
          throw s!"IR graph: node {i}: {arityError n} ({n.summary})"
        -- Because we have `n.id = i`, checking `pid < n.id` also implies `pid` is in-bounds.
        for pid in n.parents do
          if pid ≥ n.id then
            throw s!"IR graph: node {i}: parent id {pid} is not < {n.id} ({n.summary})"

end Graph

/-- Default node used only to satisfy generic container APIs; real graphs should not rely on it. -/
instance : Inhabited Node where
  default := { id := 0, parents := [], kind := OpKind.input, outShape := Shape.scalar }

end NN.IR
