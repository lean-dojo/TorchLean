/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.CROWNQuery
public import NN.Verification.Cert.NodeReplay
public import NN.Tensor.Internal.Laws.Sequence

/-!
# Finite binary32 affine artifacts

The additional checks in this module interpret the supplied binary32 coefficients exactly.
Local rational dominance checks repair the gap between agreement with rounded affine replay
and enclosure of the decoded program's real computation. The supported graph fragment is a
nonempty vector input followed by unary linear and ReLU nodes. The decoder checks the graph's
structure and parameters. `FiniteArtifactSemantics` proves agreement between `Program.eval`
and `NN.IR.Graph.denote` under the payload decoded from the same parameters, then establishes
the requested margins for the original graph.
-/

@[expose] public section

namespace NN.Verification.Cert.FiniteArtifact

open _root_.Spec TorchLean TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.CROWNQuery
open NN.Verification.Cert.NodeReplay
open scoped Spec.RationalAlgebraic

/-- Decode every scalar exactly, rejecting both infinities and NaNs. -/
def decodeTensor : {s : Shape} → Tensor (ExecFloat.Binary 8 23) s →
    Option (Tensor ℚ s)
  | .scalar, value => Tensor.scalar <$> ExecFloat.Binary.toRat? value.item
  | .dim _ _, value => do
      let rows ← Tensor.Internal.sequenceFinM fun i => decodeTensor (value.unstack i)
      pure (Tensor.dim rows)

/-- Decode the coefficients of one supplied affine form. -/
def decodeAffine {n m : Nat} (a : AffineVec (ExecFloat.Binary 8 23) n m) :
    Option (AffineVec ℚ n m) := do
  pure ⟨← decodeTensor a.A, ← decodeTensor a.c⟩

/-- Require the artifact's input and output dimensions to match the decoded node. -/
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
def affineDifference {n m : Nat} (left right : AffineVec ℚ n m) : AffineVec ℚ n m :=
  ⟨Tensor.subSpec left.A right.A, Tensor.subSpec left.c right.c⟩

/-- The supplied lower form is no larger, and its upper form no smaller, than the exact transfer. -/
def dominates {n m : Nat} (supplied exact : Bounds n m) (input : Box ℚ [n]) : Bool :=
  checkUpper false (affineDifference supplied.lo exact.lo) input &&
    checkUpper false (affineDifference exact.hi supplied.hi) input

/-- A decoded vector graph, including the affine entry supplied for every node. -/
inductive Chain (n : Nat) : Nat → Type where
  | input (bounds : Bounds n n) : Chain n n
  | linear {m k : Nat} (parent : Chain n m) (layer : LinearSpec ℚ m k)
      (bounds : Bounds n k) : Chain n k
  | relu {m : Nat} (parent : Chain n m) (alpha : Tensor ℚ [m])
      (bounds : Bounds n m) : Chain n m

/-- The actual artifact entry at the final node. -/
def Chain.bounds {n m : Nat} : Chain n m → Bounds n m
  | .input b => b
  | .linear _ _ b => b
  | .relu _ _ b => b

/-- Number of covered nodes, including the input. -/
def Chain.length {n m : Nat} : Chain n m → Nat
  | .input _ => 1
  | .linear parent _ _ => parent.length + 1
  | .relu parent _ _ => parent.length + 1

/-- Check every supplied node entry against a transfer from its already checked parent. -/
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
  | input : Program n n
  | linear {m k : Nat} (parent : Program n m) (layer : LinearSpec ℚ m k) : Program n k
  | relu {m : Nat} (parent : Program n m) : Program n m

/-- Count every node of the decoded graph, including its input. -/
def Program.length {n m : Nat} : Program n m → Nat
  | .input => 1
  | .linear parent _ => parent.length + 1
  | .relu parent => parent.length + 1

/-- Erase certificate entries and relaxation choices without changing the graph computation. -/
def Chain.program {n m : Nat} : Chain n m → Program n m
  | .input _ => .input
  | .linear parent layer _ => .linear parent.program layer
  | .relu parent _ _ => .relu parent.program

/-- Finite parameters and input domain decoded from the graph, without consulting an artifact. -/
structure DecodedGraph where
  inputDim : Nat
  outputDim : Nat
  input : Box ℚ [inputDim]
  program : Program inputDim outputDim

/-- Decode the remaining graph nodes, requiring the immediately preceding node as sole parent. -/
def decodeTail (ps : ParamStore (ExecFloat.Binary 8 23)) {n : Nat} :
    List Node → {m : Nat} → Program n m → Option (Σ k, Program n k)
  | [], _, program => some ⟨_, program⟩
  | node :: rest, m, program => do
      if node.id != program.length || node.parents != #[program.length - 1] then none
      else
        match node.kind with
        | .linear =>
            let p ← ps.linearWB[node.id]?
            if h : p.n = m then
              if p.m == 0 || node.outShape != [p.m] then none
              else
                let w ← decodeTensor p.w
                let b ← decodeTensor p.b
                let layer : LinearSpec ℚ m p.m := h ▸ ⟨w, b⟩
                decodeTail ps rest (.linear program layer)
            else none
        | .relu =>
            if node.outShape == [m] then decodeTail ps rest (.relu program) else none
        | _ => none

/-- Decode exactly one nonempty vector input and a unary linear/ReLU chain. -/
def decodeGraph (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23)) :
    Option DecodedGraph :=
  if g.wellFormed then do
    let first :: rest := g.nodes.toList | none
    let .input := first.kind | none
    let box ← ps.inputBoxes[0]?
    if first.id != 0 || !first.parents.isEmpty || box.dim == 0 ||
        first.outShape != [box.dim] then none
    else
      let lo ← decodeTensor box.lo
      let hi ← decodeTensor box.hi
      let ⟨m, program⟩ ← decodeTail ps rest (Program.input (n := box.dim))
      pure ⟨box.dim, m, ⟨lo, hi⟩, program⟩
  else none

/-- An artifact attached to a particular independently decoded graph. -/
structure Attached {n m : Nat} (program : Program n m) where
  chain : Chain n m
  sameProgram : chain.program = program

/-- Read a mandatory finite affine entry for a covered graph node. -/
def nodeBounds (cert : CROWNNodeCoreCertificate) (id n m : Nat) :
    Option (Bounds n m) := do
  let entry ← (cert.crown[id]?).join
  decodeBounds n m entry

/-- Attach the actual per-node coefficients; a missing entry causes rejection. -/
def attach (cert : CROWNNodeCoreCertificate) {n : Nat} (input : Box ℚ [n]) :
    {m : Nat} → (program : Program n m) → Option (Attached program)
  | _, .input => do
      let bounds ← nodeBounds cert 0 n n
      pure ⟨.input bounds, rfl⟩
  | m, .linear parent layer => do
      let previous ← attach cert input parent
      let bounds ← nodeBounds cert parent.length n m
      pure ⟨.linear previous.chain layer bounds, by
        simp only [Chain.program, previous.sameProgram]⟩
  | m, .relu parent => do
      let previous ← attach cert input parent
      let bounds ← nodeBounds cert parent.length n m
      let alpha ← match (cert.alpha[parent.length]?).join with
        | none =>
            let pre := previous.chain.bounds.interval input
            pure (NN.MLTheory.CROWN.Cert.defaultAlphaVec pre.lo pre.hi)
        | some vector =>
            if h : vector.n = m then
              pure (h ▸ (← decodeTensor vector.v))
            else none
      pure ⟨.relu previous.chain alpha bounds, by
        simp only [Chain.program, previous.sameProgram]⟩

/-- Requested inequalities at an explicitly designated output, with binary32 coefficients. -/
structure OutputQuery where
  outputId : Nat
  inequalities : LinParams (ExecFloat.Binary 8 23)
  strict : Bool

/-- The graph, its attached artifact, and the exact interpretation of the requested margins. -/
structure Decoded where
  graph : DecodedGraph
  artifact : Attached graph.program
  numConstraints : Nat
  inequalities : LinearSpec ℚ graph.outputDim numConstraints
  strict : Bool

/-- Decode the same graph, parameters, domain, artifact, and output query that acceptance checks. -/
def decode (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : CROWNNodeCoreCertificate) (query : OutputQuery) : Option Decoded := do
  let graph ← decodeGraph g ps
  if cert.ctx.inputId = 0 ∧ cert.ctx.inputDim = graph.inputDim ∧
      cert.crown.size = g.nodes.size ∧ graph.program.length = g.nodes.size ∧
      query.outputId + 1 = g.nodes.size then
    let artifact ← attach cert graph.input graph.program
    let p := query.inequalities
    if h : p.n = graph.outputDim then
      let w ← decodeTensor p.w
      let b ← decodeTensor p.b
      let inequalities : LinearSpec ℚ graph.outputDim p.m := h ▸ ⟨w, b⟩
      pure ⟨graph, artifact, p.m, inequalities, query.strict⟩
    else none
  else none

/-- Check nonempty dimensions, the input box, local artifact dominance, and all output margins. -/
def Decoded.check (decoded : Decoded) : Bool :=
  ((decide (0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
    0 < decoded.numConstraints) &&
  (List.finRange decoded.graph.inputDim).all
    (fun i => decide (decoded.graph.input.lo.getScalar i ≤
      decoded.graph.input.hi.getScalar i))) &&
  decoded.artifact.chain.check decoded.graph.input) &&
  checkUpper decoded.strict
    (decoded.artifact.chain.bounds.linear
      decoded.inequalities.weights decoded.inequalities.bias).hi decoded.graph.input

/-- Keep exact binary32 local replay as a consistency check on the very same artifact. -/
def replayAccepts (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : CROWNNodeCoreCertificate) : Bool :=
  let ibp := runIBP g ps
  crownLocalReplayAccepts g
    (fun replay id => NN.MLTheory.CROWN.Cert.alphaCrownStepNode?
      g.nodes ps ibp cert.alpha replay cert.ctx id) cert.crown

/-- Require replay and exact-real dominance/margin checks for the covered decoded program. -/
def accepts (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : CROWNNodeCoreCertificate) (query : OutputQuery) : Bool :=
  match decode g ps cert query with
  | none => false
  | some decoded => decoded.check && replayAccepts g ps cert

end NN.Verification.Cert.FiniteArtifact
