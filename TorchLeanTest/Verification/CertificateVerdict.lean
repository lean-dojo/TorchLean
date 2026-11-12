/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.IR.Builder
public import TorchLean.Verify.Certificate.CROWNNode
public meta import TorchLean.NN.IR.Builder
public meta import TorchLean.Verify.Certificate.CROWNNode

/-!
# The CROWN node verdict and the replay diagnostic

`CROWNNode.check` decides a certificate by `FiniteArtifact.accepts`, and agreement with the
binary32 α-CROWN replay is the separate diagnostic `CROWNNode.replayDiagnostic`. The checks:

- An `example` derives from an accepting `CROWNNode.verdict` the requested inequalities for every
  real input in the box (`CROWNNode.verdict_ok`, then `FiniteArtifact.accepts_graph_sound`), for
  every graph, certificate and query.
- One artifact on which the replay and the verdict disagree. The graph is `y = a relu (a x)` with
  `a = 1 + 2 ^ -23`, the binary32 successor of `1`, on the box `1 ≤ x ≤ 2`, where `a x > 0`. The
  replay rounds `a a = 1 + 2 ^ -22 + 2 ^ -46` to nearest, to `1 + 2 ^ -22`, so its upper form
  `(1 + 2 ^ -22) x` lies below `y` on the whole box. The replay's own table agrees with the
  replay, and its upper form passes the margin of the claim `y ≤ 2 + 2 ^ -21`, which `x = 2`
  violates. The verdict rejects that table. The same table with the upper weight `1 + 3 * 2 ^ -23`
  differs from the replay, and the verdict accepts it with the true claim `y ≤ 3` and rejects it
  with the false claim.

`run` repeats the last checks on certificate files, through `CROWNNode.replayDiagnostic` and
`CROWNNode.check`.
-/

public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open TorchLean.Tensor
open TorchLean.CROWN
open Graph
open TorchLean.Certificate
open TorchLean.Certificate.Rational (realTensor)
open DecimalJson (Value)

namespace TorchLean.Test.CertificateVerdict

/-! ## The verdict is the sound path -/

/-- An accepting verdict gives, for every real input in the box, an output of the graph evaluated
over `ℝ` that meets every requested inequality. -/
example (G : IR.Graph) (P : IR.Payload (Binary 8 23)) (input : FlatBox (Binary 8 23))
    (cert : CROWNNode) (query : FiniteArtifact.Query)
    (h : (CROWNNode.verdict G P input cert query).ok = true) :
    ∃ decoded : FiniteArtifact.Decoded, FiniteArtifact.decode G P input cert query = some decoded ∧
      ∀ x : Tensor ℝ [decoded.graph.inputDim],
        Directed.RowEncloses
          ⟨decoded.graph.inputDim, realTensor decoded.graph.input.lo,
            realTensor decoded.graph.input.hi⟩ decoded.graph.inputDim
          (fun i => getAtOrZero x [i]) →
        ∃ y : Tensor ℝ [decoded.graph.outputDim],
          G.eval (FiniteArtifact.exactPayload P) [⟨.value [decoded.graph.inputDim], x⟩] =
            .ok [⟨.value [decoded.graph.outputDim], y⟩] ∧
          ∀ i : Fin decoded.numConstraints,
            let margin := (Tensor.contract (realTensor decoded.inequalities.weights) y _ _
                  [decoded.numConstraints] .matVec) +
              realTensor decoded.inequalities.bias
            if decoded.strict then margin.getScalar i < 0 else margin.getScalar i ≤ 0 := by
  obtain ⟨decoded, hd, -, -, -, -, -, hs⟩ := FiniteArtifact.accepts_graph_sound G P input cert
    query ((CROWNNode.verdict_ok G P input cert query).1 h)
  exact ⟨decoded, hd, hs⟩

/-! ## An artifact that agrees with the replay and fails the verdict -/

/-- The binary32 word with bits `n`. -/
def word (n : Nat) : Binary 8 23 :=
  Binary.ofNatBits n

/-- The weight `a = 1 + 2 ^ -23`, the binary32 successor of `1`. -/
def a : Binary 8 23 :=
  word 0x3F800001

/-- The layer `x ↦ a x` from one feature to one. -/
def layer : SomeLinear (Binary 8 23) :=
  ⟨1, 1, ⟨Tensor.matrix fun _ _ => a, Tensor.ofFn fun _ => word 0⟩⟩

/-- The graph `y = a relu (a x)` and its payload. -/
def net : Except String (IR.Graph × IR.Payload (Binary 8 23)) :=
  IR.Graph.mlp (fun n => ⟨_, _, Op.relu (s := [n])⟩) #[layer, layer]

#guard net.toBool

/-- The graph of `net`. -/
def G : IR.Graph :=
  match net with
  | .ok (G, _) => G
  | .error _ => ⟨#[], #[]⟩

/-- The payload of `net`. -/
def P : IR.Payload (Binary 8 23) :=
  match net with
  | .ok (_, P) => P
  | .error _ => {}

/-- The input box `1 ≤ x ≤ 2`. -/
def box : FlatBox (Binary 8 23) :=
  ⟨1, Tensor.ofFn fun _ => word 0x3F800000, Tensor.ofFn fun _ => word 0x40000000⟩

/-- Lean's IBP trace of `G` on `box`. -/
def ibp : Array (Option (FlatBox (Binary 8 23))) :=
  Op.runIBP G P (Op.inputSeeds G [box])

/-- The certificate with Lean's boxes and no affine entries, α vectors or phases. -/
def bare : CROWNNode :=
  { ctx := ⟨0, 1⟩, ibp := ibp, crown := #[], alpha := Array.replicate G.nodes.size none }

/-- The replay's table: each entry is the replay of the entries before it. -/
def replayed : CROWNNode :=
  { bare with crown := Op.tabulate (CROWNNode.replayStep G P ibp bare) G.nodes.size }

/-- The output node. -/
def output : Nat := 7

/-- The table of `replayed` with the upper weight `1 + 3 * 2 ^ -23` at the output, which is at least
`a a = 1 + 2 ^ -22 + 2 ^ -46`. -/
def raised : CROWNNode :=
  { replayed with crown := replayed.crown.modify output fun entry => entry.map fun e =>
      { e with hiAff := { e.hiAff with weights := Tensor.matrix fun _ _ => word 0x3F800003 } } }

/-- The claim `y + c ≤ 0` at the output, with `c` the binary32 word with bits `bits`. -/
def claim (bits : Nat) : FiniteArtifact.Query :=
  { outputId := output, strict := false,
    inequalities :=
      ⟨1, 1, ⟨Tensor.matrix fun _ _ => word 0x3F800000, Tensor.ofFn fun _ => word bits⟩⟩ }

/-- The true claim `y ≤ 3`. -/
def claimTrue : FiniteArtifact.Query := claim 0xC0400000

/-- The false claim `y ≤ 2 + 2 ^ -21`: at `x = 2`, `y = 2 a a = 2 + 2 ^ -21 + 2 ^ -45`. -/
def claimFalse : FiniteArtifact.Query := claim 0xC0000002

-- The output is the last of the eight nodes, and the ReLU is stable on the box.
#guard G.nodes.size = 8 && G.outputs == #[⟨output, 0⟩]
#guard (ibp[3]?).join.any fun B => B.dim == 1 &&
  (List.finRange B.dim).all fun i => (Binary.toRat? (B.lo.getScalar i)).any (0 < ·)

-- The claim `y ≤ 2 + 2 ^ -21` is false at `x = 2`, in exact arithmetic.
#guard (Binary.toRat? a).any fun q => decide (2 + 1 / 2 ^ 21 < 2 * (q * q))

-- The replay's upper form at the output is `(1 + 2 ^ -22) x + 0`.
#guard (replayed.crown[output]?).join.any fun e =>
  (List.finRange e.outDim).all fun i => (List.finRange e.inDim).all fun j =>
    (Tensor.get2 e.hiAff.weights i j).toNatBits == 0x3F800002 &&
      (e.hiAff.bias.getScalar i).toNatBits == 0

/-- The node diagnostics of `cert` against the replay. -/
def found (cert : CROWNNode) : Array String :=
  CROWNNode.mismatches G ibp cert (CROWNNode.replayStep G P ibp cert)

-- The replay's table agrees with the replay at every node.
#guard (found replayed).isEmpty && CROWNNode.replayAgrees replayed G P ibp true

-- Its upper form passes the margin of the false claim, its output entry does not dominate the
-- exact transfer, and the verdict rejects it with either claim.
#guard match FiniteArtifact.decode G P box replayed claimFalse with
  | none => false
  | some d =>
      CROWNQuery.checkUpper d.strict
          (d.artifact.chain.bounds.linear d.inequalities.weights d.inequalities.bias).hi
          d.graph.input &&
        !d.artifact.chain.check d.graph.input
#guard !(CROWNNode.verdict G P box replayed claimFalse).ok
#guard !(CROWNNode.verdict G P box replayed claimTrue).ok

-- The raised table differs from the replay at the output only, and the verdict accepts it with the
-- true claim and rejects it with the false one.
#guard (found raised).size == 1 && !CROWNNode.replayAgrees raised G P ibp (found raised).isEmpty
#guard (CROWNNode.verdict G P box raised claimTrue).ok
#guard !(CROWNNode.verdict G P box raised claimFalse).ok

/-- The JSON text of `cert` without α vectors or phases, `none` when an affine entry has no JSON
form. -/
def certText (cert : CROWNNode) : Option String := do
  let crown ← cert.crown.mapM fun
    | none => some Value.null
    | some e => NodeReplay.affineBoundsValue? e
  let nat (n : Nat) : Value := .num { negative := false, significand := n, exponent := 0 }
  pure <| Value.render <| .obj #[
    ("ctx", .obj #[("inputId", nat cert.ctx.inputId), ("inputDim", nat cert.ctx.inputDim)]),
    ("ibp", .arr (cert.ibp.map fun B => (B.map NodeReplay.flatBoxValue).getD .null)),
    ("crown", .arr crown)]

/-- The replay diagnostic and the verdict with `query` of `cert` written to a file. -/
def onFile (cert : CROWNNode) (query : FiniteArtifact.Query) : IO (Array String × Report) := do
  let some text := certText cert
    | throw <| IO.userError "certificate verdict: an affine entry has no JSON form"
  IO.FS.withTempFile fun handle path => do
    handle.putStr text
    handle.flush
    let diagnostic ← CROWNNode.replayDiagnostic G P [box] path.toString
    let report ← CROWNNode.check G P box query path.toString
    pure (diagnostic, report)

/-- On certificate files: the replay's table agrees with the replay and the verdict rejects it,
and the raised table differs from the replay and the verdict accepts it with the true claim. -/
def run : IO Unit := do
  let (agreeing, rejected) ← onFile replayed claimTrue
  let (differing, accepted) ← onFile raised claimTrue
  for (ok, msg) in [(agreeing.isEmpty, "the replay diagnostic reported the replay's table"),
      (!rejected.ok, "the verdict accepted the replay's table"),
      (differing.size == 1, "the replay diagnostic did not report the raised table once"),
      (accepted.ok, "the verdict rejected the raised table")] do
    unless ok do
      throw <| IO.userError s!"certificate verdict: {msg}"
  IO.println "  CROWN node verdict: replay diagnostic and verdict tests passed"

end TorchLean.Test.CertificateVerdict
