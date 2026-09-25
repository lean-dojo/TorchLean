/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.FiniteArtifactSoundness
public import NN.Verification.Cert.FiniteArtifactNodes

/-!
# Exact-real IR semantics for the finite artifact fragment

The payload below reads the same binary32 parameter store as the fragment decoder. Successful
tensor decoding retains each finite word's exact real value. The node equations use the ordinary
IR evaluator, including its shape and parent checks.
-/

public section

namespace NN.Verification.Cert.FiniteArtifact

open _root_.Spec TorchLean TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.Verification.Cert.RationalReflection

/-- Decode finite linear parameters into the payload consumed by the real IR evaluator. -/
noncomputable def exactPayload (ps : ParamStore (ExecFloat.Binary 8 23)) : NN.IR.Payload ℝ :=
  { linear? := fun id => do
      let p ← ps.linearWB[id]?
      let weights ← decodeTensor p.w
      let bias ← decodeTensor p.b
      pure ⟨p.m, p.n, realTensor weights, realTensor bias⟩ }

/-- A successful parameter read gives precisely the weights used by the real IR evaluator. -/
theorem exactPayload_linear (ps : ParamStore (ExecFloat.Binary 8 23)) (id : Nat)
    (p : LinParams (ExecFloat.Binary 8 23)) (weights : Tensor ℚ [p.m, p.n])
    (bias : Tensor ℚ [p.m]) (hp : ps.linearWB[id]? = some p)
    (hw : decodeTensor p.w = some weights) (hb : decodeTensor p.b = some bias) :
    (exactPayload ps).linear? id =
      some ⟨p.m, p.n, realTensor weights, realTensor bias⟩ := by
  simp [exactPayload, hp, hw, hb]

/-- Every decoded program contains its input node. -/
theorem Program.length_pos {n m : Nat} (program : Program n m) : 0 < program.length := by
  cases program <;> simp [Program.length]

/-- Successful decoding supplies the ordinary IR evaluator's structural guard. -/
theorem decodeGraph_wellFormed (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (decoded : DecodedGraph) (h : decodeGraph g ps = some decoded) :
    g.wellFormed = true := by
  unfold decodeGraph at h
  split at h
  next hw => exact hw
  next => contradiction

private theorem getNode_of_split (g : Graph) (before rest : List Node) (node : Node)
    (h : g.nodes.toList = before ++ node :: rest) (hid : node.id = before.length) :
    g.getNode before.length = .ok node := by
  have hget : g.nodes[before.length]? = some node := by
    have hh := congrArg (fun nodes : List Node => nodes[before.length]?) h
    simpa using hh
  simp [NN.IR.Graph.getNode, NN.IR.Graph.getNode?, hget, hid, Pure.pure, Except.pure]

private theorem decodeTail_denote (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    {n : Nat} (x : Tensor ℝ [n]) (before rest : List Node) {m : Nat}
    (program : Program n m) (output : Σ k, Program n k) (values : Array (SomeTensor ℝ))
    (hgraph : g.nodes.toList = before ++ rest) (hbefore : before.length = program.length)
    (hsize : values.size = program.length)
    (hlast : values[program.length - 1]? = some ⟨[m], program.eval x⟩)
    (hdecode : decodeTail ps rest program = some output) :
    ∃ result, g.denoteAllFrom (exactPayload ps) ⟨[n], x⟩ program.length values = .ok result ∧
      result[g.nodes.size - 1]? = some ⟨[output.1], output.2.eval x⟩ := by
  cases rest with
  | nil =>
      cases Option.some.inj hdecode
      have hlength : g.nodes.size = program.length := by
        have hh := congrArg List.length hgraph
        simpa [hbefore] using hh
      refine ⟨values, ?_, ?_⟩
      · unfold NN.IR.Graph.denoteAllFrom
        simp [hlength, Pure.pure, Except.pure]
      · simpa [hlength] using hlast
  | cons node rest =>
      simp only [decodeTail] at hdecode
      split at hdecode
      next => contradiction
      next hvalid =>
        have hid : node.id = program.length := by
          by_contra hne
          simp [hne] at hvalid
        have hparents : node.parents = #[program.length - 1] := by
          by_contra hne
          simp [hne] at hvalid
        have hget : g.getNode program.length = .ok node := by
          simpa [hbefore] using getNode_of_split g before rest node hgraph
            (hid.trans hbefore.symm)
        have hlt : program.length < g.nodes.size := by
          have hh := congrArg List.length hgraph
          simp only [Array.length_toList, List.length_append, List.length_cons] at hh
          omega
        have hgraph' : g.nodes.toList = (before ++ [node]) ++ rest := by
          simpa [List.append_assoc] using hgraph
        cases hk : node.kind <;> simp only [hk] at hdecode <;> try contradiction
        case linear =>
          obtain ⟨p, hp, hdecode⟩ := Option.bind_eq_some_iff.mp hdecode
          split at hdecode
          next hdim =>
            subst m
            split at hdecode
            next => contradiction
            next hshape =>
              have hout : node.outShape = [p.m] := by
                by_contra hne
                simp [hne] at hshape
              obtain ⟨weights, hw, hdecode⟩ := Option.bind_eq_some_iff.mp hdecode
              obtain ⟨bias, hb, hdecode⟩ := Option.bind_eq_some_iff.mp hdecode
              let next : Program n p.m := .linear program ⟨weights, bias⟩
              let value : SomeTensor ℝ := ⟨[p.m], next.eval x⟩
              have hn : node =
                  ⟨program.length, #[program.length - 1], .linear, [p.m]⟩ := by
                cases node
                simp_all
              have hstep : g.evalAt (exactPayload ps) ⟨[n], x⟩ values program.length =
                  .ok value := by
                simpa [NN.IR.Graph.evalAt, hget, hn, next, value, Program.eval,
                  Bind.bind, Except.bind] using
                  evalNode_linear (exactPayload ps) ⟨[n], x⟩ values program.length
                    (program.length - 1) (program.eval x)
                    (realTensor weights) (realTensor bias) hlast
                    (by simpa [hid] using exactPayload_linear ps node.id p weights bias hp hw hb)
              obtain ⟨result, hr, ho⟩ := decodeTail_denote g ps x (before ++ [node]) rest next
                output (values.push value) hgraph'
                (by simp [next, Program.length, hbefore])
                (by simp [next, Program.length, hsize])
                (by simp [next, Program.length, ← hsize, value]) hdecode
              refine ⟨result, ?_, ho⟩
              unfold NN.IR.Graph.denoteAllFrom
              rw [dite_eq_left hlt]
              simpa [hstep, next, Program.length, Bind.bind, Except.bind] using hr
          next => contradiction
        case relu =>
          split at hdecode
          next hshape =>
            have hout : node.outShape = [m] := by simpa using hshape
            let next : Program n m := .relu program
            let value : SomeTensor ℝ := ⟨[m], next.eval x⟩
            have hn : node =
                ⟨program.length, #[program.length - 1], .relu, [m]⟩ := by
              cases node
              simp_all
            have hstep : g.evalAt (exactPayload ps) ⟨[n], x⟩ values program.length =
                .ok value := by
              simpa [NN.IR.Graph.evalAt, hget, hn, next, value, Program.eval,
                Bind.bind, Except.bind] using
                evalNode_relu (exactPayload ps) ⟨[n], x⟩ values program.length
                  (program.length - 1) (program.eval x) hlast
            obtain ⟨result, hr, ho⟩ := decodeTail_denote g ps x (before ++ [node]) rest next
              output (values.push value) hgraph'
              (by simp [next, Program.length, hbefore])
              (by simp [next, Program.length, hsize])
              (by simp [next, Program.length, ← hsize, value]) hdecode
            refine ⟨result, ?_, ho⟩
            unfold NN.IR.Graph.denoteAllFrom
            rw [dite_eq_left hlt]
            simpa [hstep, next, Program.length, Bind.bind, Except.bind] using hr
          next => contradiction
termination_by rest.length

/-- Successful decoding determines the ordinary graph evaluator's exact-real output. -/
theorem decodeGraph_denote (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (decoded : DecodedGraph) (h : decodeGraph g ps = some decoded)
    (x : Tensor ℝ [decoded.inputDim]) :
    g.denote (exactPayload ps) ⟨[decoded.inputDim], x⟩ (g.nodes.size - 1) =
      .ok ⟨[decoded.outputDim], decoded.program.eval x⟩ := by
  have hw := decodeGraph_wellFormed g ps decoded h
  unfold decodeGraph at h
  simp only [hw, ↓reduceIte] at h
  cases hnodes : g.nodes.toList with
  | nil => simp [hnodes] at h
  | cons first rest =>
      simp only [hnodes] at h
      cases hk : first.kind <;> simp only [hk] at h <;> try contradiction
      case input =>
        obtain ⟨box, _, h⟩ := Option.bind_eq_some_iff.mp h
        split at h
        next => contradiction
        next hvalid =>
          have hid : first.id = 0 := by
            by_contra hne
            simp [hne] at hvalid
          have hparents : first.parents = #[] := by
            cases hp : first.parents with
            | mk parents => cases parents <;> simp_all
          have hshape : first.outShape = [box.dim] := by
            by_contra hne
            simp [hne] at hvalid
          obtain ⟨lo, _, h⟩ := Option.bind_eq_some_iff.mp h
          obtain ⟨hi, _, h⟩ := Option.bind_eq_some_iff.mp h
          obtain ⟨output, ht, h⟩ := Option.bind_eq_some_iff.mp h
          cases output with
          | mk m program =>
            cases Option.some.inj h
            have hn : first = ⟨0, #[], .input, [box.dim]⟩ := by
              cases first
              simp_all
            have hget : g.getNode 0 = .ok first :=
              getNode_of_split g [] rest first (by simpa using hnodes) hid
            have hstep : g.evalAt (exactPayload ps) ⟨[box.dim], x⟩ #[] 0 =
                .ok ⟨[box.dim], x⟩ := by
              simpa [NN.IR.Graph.evalAt, hget, hn, Bind.bind, Except.bind] using
                evalNode_input (exactPayload ps) x #[]
            obtain ⟨result, hr, ho⟩ :=
              decodeTail_denote g ps x [first] rest .input ⟨m, program⟩
                #[⟨[box.dim], x⟩] (by simpa using hnodes) rfl rfl
                (by simp [Program.length, Program.eval]) ht
            have hlt : 0 < g.nodes.size := by
              have hh := congrArg List.length hnodes
              simp at hh
              omega
            have hall : g.denoteAll (exactPayload ps) ⟨[box.dim], x⟩ = .ok result := by
              simp only [NN.IR.Graph.denoteAll, hw, ↓reduceIte]
              unfold NN.IR.Graph.denoteAllFrom
              rw [dite_eq_left hlt]
              simpa [hstep, Program.length, Bind.bind, Except.bind] using hr
            simp [NN.IR.Graph.denote, hall, ho, Pure.pure, Except.pure, Bind.bind, Except.bind]

/-- The requested margins hold for the original graph under its exactly decoded payload. -/
theorem accepts_graph_sound (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : NodeReplay.CROWNNodeCoreCertificate) (query : OutputQuery)
    (h : accepts g ps cert query = true) :
    ∃ decoded, decode g ps cert query = some decoded ∧
      nodeBounds cert query.outputId decoded.graph.inputDim decoded.graph.outputDim =
        some decoded.artifact.chain.bounds ∧
      0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
      0 < decoded.numConstraints ∧
      ∀ x : Tensor ℝ [decoded.graph.inputDim],
        Theorems.Semantics.encloses
          ⟨decoded.graph.inputDim, realTensor decoded.graph.input.lo,
            realTensor decoded.graph.input.hi⟩ x →
        ∃ y : Tensor ℝ [decoded.graph.outputDim],
          g.denote (exactPayload ps) ⟨[decoded.graph.inputDim], x⟩ query.outputId =
            .ok ⟨[decoded.graph.outputDim], y⟩ ∧
          ∀ i : Fin decoded.numConstraints,
            let margin := Tensor.addSpec
              (matVecMulSpec (realTensor decoded.inequalities.weights) y)
              (realTensor decoded.inequalities.bias)
            if decoded.strict then margin.getScalar i < 0 else margin.getScalar i ≤ 0 := by
  obtain ⟨decoded, hd, hg, ho, hb, hn, hm, hk, hs⟩ :=
    accepts_sound g ps cert query h
  refine ⟨decoded, hd, hb, hn, hm, hk, ?_⟩
  intro x hx
  refine ⟨decoded.graph.program.eval x, ?_, hs x hx⟩
  have hid : query.outputId = g.nodes.size - 1 := by omega
  rw [hid]
  exact decodeGraph_denote g ps decoded.graph hg x

end NN.Verification.Cert.FiniteArtifact
