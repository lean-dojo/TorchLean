/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.CROWNQuery
public import TorchLean.Verify.Certificate.NodeReplay
public import TorchLean.Verify.CROWN.FlatBox.Affine
public import TorchLean.NN.IR.Eval
public import TorchLean.Tensor.Rep.Traversal.Sequence

/-!
# Finite binary32 affine artifacts

Each binary32 coefficient of a CROWN certificate is read here as its exact rational value, and an
infinity or a NaN is refused. Agreement with the rounded affine replay gives no enclosure of the
decoded program's real computation on its own. Local dominance checks in rational arithmetic close
that gap: over the input box, each supplied bound must dominate the exact transfer of its parent's
bound (`Chain.check`).

The supported fragment is a graph of registered operators (`IR.Graph`) with one nonempty
vector input followed by linear layers and ReLU nodes, whose last node is the graph's one output. A
linear layer is its weight and bias constants and then `Op.linear` on them and the previous node.
The decoder checks the graph's structure and its payload constants. `decodeGraph_denote` proves
that `Program.eval` agrees with `IR.Graph.eval` under the payload decoded from the same constants,
and from that agreement `accepts_graph_sound` establishes the requested margins for the original
graph. Both are in `TorchLean.Verify.Certificate.FiniteArtifact.Semantics`.
-/

@[expose] public section

namespace TorchLean.Certificate.FiniteArtifact

open TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open CROWN
open Graph
open CROWNQuery
open scoped RationalAlgebraic

/-- Decode every scalar exactly, rejecting both infinities and NaNs. -/
def decodeTensor : {s : Shape} → Tensor (ExecFloat.Binary 8 23) s →
    Option (Tensor ℚ s)
  | .scalar, value => Tensor.scalar <$> ExecFloat.Binary.toRat? value.item
  | .dim _ _, value => do
      let rows ← Tensor.sequenceFinM fun i => decodeTensor (value.unstack i)
      pure (Tensor.dim rows)

/-- Decode the coefficients of one supplied affine form. -/
def decodeAffine {n m : Nat} (a : Linear (ExecFloat.Binary 8 23) n m) :
    Option (Linear ℚ n m) := do
  pure ⟨← decodeTensor a.weights, ← decodeTensor a.bias⟩

/-- The exact rational bounds of an artifact entry with input dimension `n` and output dimension
`m`, or `none` when the dimensions differ or a coefficient is not finite. -/
def decodeBounds (n m : Nat) (b : FlatAffineBounds (ExecFloat.Binary 8 23)) :
    Option (Bounds n m) := do
  if hn : b.inDim = n then
    if hm : b.outDim = m then
      let lo ← decodeAffine b.loAff
      let hi ← decodeAffine b.hiAff
      pure (hn ▸ hm ▸ Bounds.mk lo hi)
    else none
  else none

/-- The affine difference whose nonpositivity expresses pointwise domination. -/
def affineDifference {n m : Nat} (left right : Linear ℚ n m) : Linear ℚ n m :=
  ⟨left.weights - right.weights, left.bias - right.bias⟩

/-- Whether `supplied` dominates `exact` over the box `input`: at every point of the box its lower
form is at most the exact lower form and its upper form at least the exact upper form. -/
def dominates {n m : Nat} (supplied exact : Bounds n m) (input : Box ℚ [n]) : Bool :=
  checkUpper false (affineDifference supplied.lo exact.lo) input &&
    checkUpper false (affineDifference exact.hi supplied.hi) input

/-- A decoded vector graph, including the affine entry supplied for every node of the chain. -/
inductive Chain (n : Nat) : Nat → Type where
  /-- The input node, with its supplied entry. -/
  | input (bounds : Bounds n n) : Chain n n
  /-- A dense layer after `parent`, with its supplied entry. -/
  | linear {m k : Nat} (parent : Chain n m) (layer : Linear ℚ m k)
      (bounds : Bounds n k) : Chain n k
  /-- A ReLU after `parent`, with its slopes `alpha` and its supplied entry. -/
  | relu {m : Nat} (parent : Chain n m) (alpha : Tensor ℚ [m])
      (bounds : Bounds n m) : Chain n m

/-- The artifact entry at the last node of the chain. -/
def Chain.bounds {n m : Nat} : Chain n m → Bounds n m
  | .input b => b
  | .linear _ _ b => b
  | .relu _ _ b => b

/-- Whether every entry of the chain dominates over `input` the exact transfer of its parent's
entry, and the entry at the input node the identity. The input and the input of every linear layer
must be nonempty, and every ReLU slope must lie in `[0, 1]`. -/
def Chain.check {n m : Nat} (chain : Chain n m) (input : Box ℚ [n]) : Bool :=
  match chain with
  | .input b => decide (0 < n) && dominates b (Bounds.identity n) input
  | .linear parent layer b =>
      (parent.check input && decide (0 < m)) &&
        dominates b (parent.bounds.linear layer.weights layer.bias) input
  | .relu parent alpha b =>
      (parent.check input && checkAlpha alpha) &&
        dominates b (parent.bounds.relu input alpha) input

/-- The graph computation after finite parameter decoding, independent of any certificate. -/
inductive Program (n : Nat) : Nat → Type where
  /-- The input node. -/
  | input : Program n n
  /-- A dense layer after `parent`. -/
  | linear {m k : Nat} (parent : Program n m) (layer : Linear ℚ m k) : Program n k
  /-- A ReLU after `parent`. -/
  | relu {m : Nat} (parent : Program n m) : Program n m

/-- Count every node of the decoded graph, including its input and the weight and bias constants of
each linear layer. -/
def Program.length {n m : Nat} : Program n m → Nat
  | .input => 1
  | .linear parent _ => parent.length + 3
  | .relu parent => parent.length + 1

/-- Erase certificate entries and relaxation choices without changing the graph computation. -/
def Chain.program {n m : Nat} : Chain n m → Program n m
  | .input _ => .input
  | .linear parent layer _ => .linear parent.program layer
  | .relu parent _ _ => .relu parent.program

/-- Finite parameters and input domain decoded from the graph, without consulting an artifact. -/
structure DecodedGraph where
  /-- The input dimension. -/
  inputDim : Nat
  /-- The output dimension. -/
  outputDim : Nat
  /-- The input box. -/
  input : Box ℚ [inputDim]
  /-- The decoded computation. -/
  program : Program inputDim outputDim

/-- Payload constant `id` as an exact rational tensor of shape `s`, and `none` unless the constant
is at the value port of that shape and every word is finite. -/
def constTensor? (P : IR.Payload (ExecFloat.Binary 8 23)) (id : Nat) (s : Shape) :
    Option (Tensor ℚ s) :=
  match P.const? id with
  | some ⟨⟨.value, t⟩, x⟩ => if h : t = s then decodeTensor (h ▸ x) else none
  | _ => none

/-- Decode the nodes after the input. A linear layer is its weight and bias constants and then
`Op.linear` on them and the previous node, with a nonempty output, and a ReLU is `Op.relu` on the
previous node. Any other node, or a constant that `constTensor?` refuses, gives `none`. -/
def decodeTail (P : IR.Payload (ExecFloat.Binary 8 23)) {n : Nat} :
    List IR.Node → {m : Nat} → Program n m → Option (Σ k, Program n k)
  | [], _, program => some ⟨_, program⟩
  | .op o refs :: rest, m, program =>
      if o = ⟨_, _, Op.relu (s := [m])⟩ ∧ refs = #[⟨program.length - 1, 0⟩] then
        decodeTail P rest (.relu program)
      else none
  | .const pw :: .const pb :: .op o refs :: rest, m, program =>
      match pb with
      | ⟨.value, [k]⟩ =>
          if pw = .value [k, m] ∧ 0 < k ∧
              o = ⟨_, _, Op.linear (lead := []) (inDim := m) (outDim := k)⟩ ∧
              refs = #[⟨program.length, 0⟩, ⟨program.length + 1, 0⟩, ⟨program.length - 1, 0⟩] then
            match constTensor? P program.length [k, m],
                constTensor? P (program.length + 1) [k] with
            | some w, some b => decodeTail P rest (.linear program ⟨w, b⟩)
            | _, _ => none
          else none
      | _ => none
  | _, _, _ => none

/-- Decode a graph that passes `Graph.check`: exactly one nonempty vector input, with the dimension
of the box `input` and finite box endpoints, then a linear/ReLU chain whose last node is the graph's
one output. -/
def decodeGraph (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) : Option DecodedGraph :=
  match G.check with
  | .error _ => none
  | .ok () => do
    let .input 0 ⟨.value, [n]⟩ :: rest := G.nodes.toList | none
    if n = input.dim ∧ 0 < n ∧ G.outputs = #[⟨G.nodes.size - 1, 0⟩] then
      let lo ← decodeTensor input.lo
      let hi ← decodeTensor input.hi
      let ⟨m, program⟩ ← decodeTail P rest (Program.input (n := input.dim))
      pure ⟨input.dim, m, ⟨lo, hi⟩, program⟩
    else none

/-- A chain of artifact entries whose erased program is the independently decoded `program`. -/
structure Attached {n m : Nat} (program : Program n m) where
  /-- The chain of artifact entries. -/
  chain : Chain n m
  /-- Erasing the entries of `chain` gives `program`. -/
  chain_program : chain.program = program

/-- The exact bounds of the entry of node `id`, which must be present, finite and of dimensions `n`
and `m`. -/
def nodeBounds (cert : CROWNNode) (id n m : Nat) :
    Option (Bounds n m) := do
  let entry ← (cert.crown[id]?).join
  decodeBounds n m entry

/-- Attach the entries of `cert` to the nodes of the program, and `none` when an entry is missing or
does not decode. A ReLU without an α vector takes the default slopes of the interval that its
parent's entry gives over `input`. -/
def attach (cert : CROWNNode) {n : Nat} (input : Box ℚ [n]) :
    {m : Nat} → (program : Program n m) → Option (Attached program)
  | _, .input => do
      let bounds ← nodeBounds cert 0 n n
      pure ⟨.input bounds, rfl⟩
  | m, .linear parent layer => do
      let previous ← attach cert input parent
      let bounds ← nodeBounds cert (parent.length + 2) n m
      pure ⟨.linear previous.chain layer bounds, by
        simp only [Chain.program, previous.chain_program]⟩
  | m, .relu parent => do
      let previous ← attach cert input parent
      let bounds ← nodeBounds cert parent.length n m
      let alpha ← match (cert.alpha[parent.length]?).join with
        | none =>
            let pre := previous.chain.bounds.interval input
            pure (CROWN.Cert.defaultAlphaVec pre.lo pre.hi)
        | some vector =>
            if h : vector.n = m then
              pure (h ▸ (← decodeTensor vector.v))
            else none
      pure ⟨.relu previous.chain alpha bounds, by
        simp only [Chain.program, previous.chain_program]⟩

/-- Requested inequalities with binary32 coefficients at the output node `outputId`, which `decode`
requires to be the last node. -/
structure Query where
  /-- The output node. -/
  outputId : Nat
  /-- The requested inequalities, with binary32 coefficients. -/
  inequalities : SomeLinear (ExecFloat.Binary 8 23)
  /-- Whether each inequality is strict. -/
  strict : Bool

/-- The graph, its attached artifact, and the exact interpretation of the requested margins. -/
structure Decoded where
  /-- The decoded graph. -/
  graph : DecodedGraph
  /-- The artifact entries attached to the program of `graph`. -/
  artifact : Attached graph.program
  /-- The number of requested inequalities. -/
  numConstraints : Nat
  /-- The requested inequalities, with exact rational coefficients. -/
  inequalities : Linear ℚ graph.outputDim numConstraints
  /-- Whether each inequality is strict. -/
  strict : Bool

/-- Decode the graph with its constants and input box, the artifact and the output query that
`accepts` checks. The certificate's context must name input `0` with the graph's input dimension,
and its `crown` array needs one slot per node. The decoded program must cover every node, and the
query must be at the last node with the graph's output dimension. -/
def decode (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (cert : CROWNNode) (query : Query) :
    Option Decoded := do
  let graph ← decodeGraph G P input
  if cert.ctx.inputId = 0 ∧ cert.ctx.inputDim = graph.inputDim ∧
      cert.crown.size = G.nodes.size ∧ graph.program.length = G.nodes.size ∧
      query.outputId + 1 = G.nodes.size then
    let artifact ← attach cert graph.input graph.program
    let p := query.inequalities
    if h : p.inDim = graph.outputDim then
      let w ← decodeTensor p.spec.weights
      let b ← decodeTensor p.spec.bias
      let inequalities : Linear ℚ graph.outputDim p.outDim := h ▸ ⟨w, b⟩
      pure ⟨graph, artifact, p.outDim, inequalities, query.strict⟩
    else none
  else none

/-- Whether the input, output and margin dimensions are nonzero, the input box is ordered, the chain
passes `Chain.check`, and every output margin passes `checkUpper` over the box. -/
def Decoded.check (decoded : Decoded) : Bool :=
  ((decide (0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
    0 < decoded.numConstraints) &&
  Tensor.boundsOrdered decoded.graph.input.lo decoded.graph.input.hi) &&
  decoded.artifact.chain.check decoded.graph.input) &&
  checkUpper decoded.strict
    (decoded.artifact.chain.bounds.linear
      decoded.inequalities.weights decoded.inequalities.bias).hi decoded.graph.input

/-- Decode the covered program and run the exact dominance and margin checks of `Decoded.check`.
Bounds from any producer pass when they dominate the exact transfers, whether or not they match the
binary32 replay. -/
def accepts (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (cert : CROWNNode) (query : Query) :
    Bool :=
  match decode G P input cert query with
  | none => false
  | some decoded => decoded.check

end TorchLean.Certificate.FiniteArtifact
