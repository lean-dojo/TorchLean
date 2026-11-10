/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.Certificate.FiniteArtifact.Soundness
public import TorchLean.Verify.Certificate.FiniteArtifact.Nodes

/-!
# Exact-real operator-graph semantics for the finite artifact fragment

The real payload `exactPayload` reads the same binary32 constants as the fragment decoder, and a
decoded tensor keeps the exact real value of each finite word. The node equations use the evaluator
of graphs of registered operators, port checks included. With them `decodeGraph_denote` shows that
a decoded graph evaluates to its decoded program, and `accepts_graph_sound` carries the accepted
margins over to the original graph.
-/

public section

namespace TorchLean.Certificate.FiniteArtifact

open Tensor
open FloatLib.Floats (ExecFloat)
open CROWN
open Rational

/-- The real evaluator's payload holding the exact values of the value constants of `P`. A constant
at another port, or with a word that is not finite, is absent. -/
noncomputable def exactPayload (P : IR.Payload (ExecFloat.Binary 8 23)) : IR.Payload ℝ :=
  { const? := fun id =>
      match P.const? id with
      | some ⟨⟨.value, s⟩, x⟩ => (decodeTensor x).map fun q => ⟨.value s, realTensor q⟩
      | _ => none }

/-- A value constant that decodes to `q` is the constant `realTensor q` of the real payload. -/
theorem exactPayload_const (P : IR.Payload (ExecFloat.Binary 8 23)) (id : Nat) {s : Shape}
    (t : Tensor (ExecFloat.Binary 8 23) s) (q : Tensor ℚ s)
    (hp : P.const? id = some ⟨.value s, t⟩) (hq : decodeTensor t = some q) :
    (exactPayload P).const? id = some ⟨.value s, realTensor q⟩ := by
  simp [exactPayload, hp, hq]

/-- A graph that decodes passes `Graph.check`. -/
theorem decodeGraph_check (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (decoded : DecodedGraph)
    (h : decodeGraph G P input = some decoded) :
    G.check = .ok () := by
  unfold decodeGraph at h
  split at h
  next => contradiction
  next hc => exact hc

/-- A constant the decoder reads is the real evaluator's constant at its node. -/
private theorem exactPayload_const' {P : IR.Payload (ExecFloat.Binary 8 23)} {id : Nat}
    {s : Shape} {q : Tensor ℚ s} (h : constTensor? P id s = some q) :
    (exactPayload P).const? id = some ⟨.value s, realTensor q⟩ := by
  unfold constTensor? at h
  split at h
  next t x hp =>
    split at h
    next ht =>
      subst ht
      exact exactPayload_const P id x q hp h
    next => contradiction
  next => contradiction

/-- A constant node at the value port of `s` returns the real evaluator's constant at its id. -/
private theorem eval_const {P : IR.Payload ℝ} {inputs : List (Op.Value ℝ)}
    {earlier : Array (List (Op.Value ℝ))} {id : Nat} {s : Shape} {t : Tensor ℝ s}
    (h : P.const? id = some ⟨.value s, t⟩) :
    (IR.Node.const (.value s)).eval P inputs earlier id = .ok [⟨.value s, t⟩] := by
  simp [IR.Node.eval, h, Port.read_self, pure, Except.pure, bind, Except.bind]

/-- The newest output of a run is read at its node. -/
private theorem read_push {earlier : Array (List (Op.Value ℝ))}
    {v : Op.Value ℝ} :
    (⟨earlier.size, 0⟩ : IR.Ref).read (earlier.push [v]) = .ok v := by
  simp [IR.Ref.read, pure, Except.pure]

private theorem decodeTail_run (P : IR.Payload (ExecFloat.Binary 8 23)) {n : Nat}
    (x : Tensor ℝ [n]) (inputs : List (Op.Value ℝ)) (rest : List IR.Node) {m : Nat}
    (program : Program n m) (output : Σ k, Program n k)
    (values : Array (List (Op.Value ℝ))) (hsize : values.size = program.length)
    (hlast : (⟨program.length - 1, 0⟩ : IR.Ref).read values =
      .ok ⟨.value [m], program.eval x⟩)
    (hdecode : decodeTail P rest program = some output) :
    ∃ result, IR.runNodes (exactPayload P) inputs values rest = .ok result ∧
      result.size = output.2.length ∧
      (⟨output.2.length - 1, 0⟩ : IR.Ref).read result =
        .ok ⟨.value [output.1], output.2.eval x⟩ := by
  have hpos : 0 < program.length := by cases program <;> simp [Program.length]
  have hprev : ∀ (more : Array (List (Op.Value ℝ))),
      (⟨program.length - 1, 0⟩ : IR.Ref).read (values ++ more) =
        .ok ⟨.value [m], program.eval x⟩ := fun more => IR.Ref.read_append hlast more
  match rest, hdecode with
  | [], hdecode =>
      cases Option.some.inj hdecode
      exact ⟨values, rfl, hsize, hlast⟩
  | .op o refs :: rest, hdecode =>
      simp only [decodeTail] at hdecode
      split at hdecode
      next hvalid =>
        obtain ⟨ho, hr⟩ := hvalid
        subst ho hr
        let value : Op.Value ℝ := ⟨.value [m], (Program.relu program).eval x⟩
        have hstep : (IR.Node.op ⟨_, _, Op.relu (s := [m])⟩
            #[⟨program.length - 1, 0⟩]).eval (exactPayload P) inputs values values.size =
            .ok [value] :=
          evalNode_relu _ inputs values _ _ _ hlast
        have hread : (⟨(Program.relu program).length - 1, 0⟩ : IR.Ref).read
            (values.push [value]) = .ok value := by
          have h := read_push (earlier := values) (v := value)
          rw [hsize] at h
          simpa [Program.length] using h
        obtain ⟨result, hr, hs, hl⟩ := decodeTail_run P x inputs rest (.relu program) output
          (values.push [value]) (by simp [hsize, Program.length]) hread hdecode
        refine ⟨result, ?_, hs, hl⟩
        rw [IR.runNodes_cons, hstep]
        exact hr
      next => contradiction
  | .const pw :: .const pb :: .op o refs :: rest, hdecode =>
      simp only [decodeTail] at hdecode
      split at hdecode
      next k =>
        split at hdecode
        next hvalid =>
          obtain ⟨hw, _, ho, hr⟩ := hvalid
          subst hw ho hr
          split at hdecode
          next w b hw hb =>
            let vw : Op.Value ℝ := ⟨.value [k, m], realTensor w⟩
            let vb : Op.Value ℝ := ⟨.value [k], realTensor b⟩
            let value : Op.Value ℝ :=
              ⟨.value [k], (Program.linear program ⟨w, b⟩).eval x⟩
            have hw' := exactPayload_const' hw
            have hb' := exactPayload_const' hb
            have h1 : (IR.Node.const (.value [k, m])).eval (exactPayload P) inputs values
                values.size = .ok [vw] := eval_const (by rw [hsize]; exact hw')
            have h2 : (IR.Node.const (.value [k])).eval (exactPayload P) inputs
                (values.push [vw]) (values.push [vw]).size = .ok [vb] :=
              eval_const (by rw [Array.size_push, hsize]; exact hb')
            have hreadW : (⟨program.length, 0⟩ : IR.Ref).read
                ((values.push [vw]).push [vb]) = .ok vw := by
              rw [IR.Ref.read_push (by simp [hsize])]
              have h := read_push (earlier := values) (v := vw)
              rw [hsize] at h
              exact h
            have hreadB : (⟨program.length + 1, 0⟩ : IR.Ref).read
                ((values.push [vw]).push [vb]) = .ok vb := by
              have h := read_push (earlier := values.push [vw]) (v := vb)
              rw [Array.size_push, hsize] at h
              exact h
            have hreadX : (⟨program.length - 1, 0⟩ : IR.Ref).read
                ((values.push [vw]).push [vb]) = .ok ⟨.value [m], program.eval x⟩ := by
              rw [IR.Ref.read_push (by simp [hsize]),
                IR.Ref.read_push (by simp [hsize]; omega)]
              exact hlast
            have h3 : (IR.Node.op
                ⟨_, _, Op.linear (lead := []) (inDim := m) (outDim := k)⟩
                #[⟨program.length, 0⟩, ⟨program.length + 1, 0⟩, ⟨program.length - 1, 0⟩]).eval
                  (exactPayload P) inputs ((values.push [vw]).push [vb])
                  ((values.push [vw]).push [vb]).size = .ok [value] :=
              evalNode_linear _ inputs _ _ _ _ _ _ _ _ hreadW hreadB hreadX
            have hread : (⟨(Program.linear program ⟨w, b⟩).length - 1, 0⟩ : IR.Ref).read
                (((values.push [vw]).push [vb]).push [value]) = .ok value := by
              have h := read_push (earlier := (values.push [vw]).push [vb]) (v := value)
              rw [Array.size_push, Array.size_push, hsize] at h
              simpa [Program.length] using h
            obtain ⟨result, hr, hs, hl⟩ := decodeTail_run P x inputs rest
              (.linear program ⟨w, b⟩) output (((values.push [vw]).push [vb]).push [value])
              (by simp [hsize, Program.length]) hread hdecode
            refine ⟨result, ?_, hs, hl⟩
            rw [IR.runNodes_cons, h1]
            simp only [bind, Except.bind]
            rw [IR.runNodes_cons, h2]
            simp only [bind, Except.bind]
            rw [IR.runNodes_cons, h3]
            exact hr
          next => contradiction
        next => contradiction
      next => contradiction
  | .input _ _ :: _, hdecode => simp [decodeTail] at hdecode
  | .const _ :: [], hdecode => simp [decodeTail] at hdecode
  | .const _ :: .input _ _ :: _, hdecode => simp [decodeTail] at hdecode
  | .const _ :: .op _ _ :: _, hdecode => simp [decodeTail] at hdecode
  | .const _ :: .const _ :: [], hdecode => simp [decodeTail] at hdecode
  | .const _ :: .const _ :: .input _ _ :: _, hdecode => simp [decodeTail] at hdecode
  | .const _ :: .const _ :: .const _ :: _, hdecode => simp [decodeTail] at hdecode
termination_by rest.length

/-- A successfully decoded graph's one output is its last node. -/
private theorem decodeGraph_outputs {G : IR.Graph} {P : IR.Payload (ExecFloat.Binary 8 23)}
    {input : FlatBox (ExecFloat.Binary 8 23)} {decoded : DecodedGraph}
    (h : decodeGraph G P input = some decoded) :
    G.outputs = #[⟨G.nodes.size - 1, 0⟩] := by
  unfold decodeGraph at h
  split at h
  next => contradiction
  next =>
    split at h
    next =>
      split at h
      next hvalid => exact hvalid.2.2
      next => contradiction
    next => contradiction

/-- If `G` decodes, its real evaluation under `exactPayload P` at an input `x` returns the decoded
program's output `decoded.program.eval x`. -/
theorem decodeGraph_denote (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23)) (decoded : DecodedGraph)
    (h : decodeGraph G P input = some decoded) (x : Tensor ℝ [decoded.inputDim]) :
    G.eval (exactPayload P) [⟨.value [decoded.inputDim], x⟩] =
      .ok [⟨.value [decoded.outputDim], decoded.program.eval x⟩] := by
  have houts := decodeGraph_outputs h
  unfold decodeGraph at h
  split at h
  next => contradiction
  next =>
    split at h
    next n rest hnodes =>
      split at h
      next hvalid =>
        obtain ⟨hn, -, -⟩ := hvalid
        subst hn
        obtain ⟨lo, -, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨hi, -, h⟩ := Option.bind_eq_some_iff.mp h
        obtain ⟨⟨m, program⟩, ht, h⟩ := Option.bind_eq_some_iff.mp h
        cases Option.some.inj h
        let value : Op.Value ℝ := ⟨.value [input.dim], x⟩
        have hfirst : (IR.Node.input 0 (.value [input.dim])).eval (exactPayload P) [value]
            #[] (#[] : Array (List (Op.Value ℝ))).size = .ok [value] :=
          evalNode_input _ x _ _
        have hread : (⟨(Program.input (n := input.dim)).length - 1, 0⟩ : IR.Ref).read
            ((#[] : Array (List (Op.Value ℝ))).push [value]) =
              .ok ⟨.value [input.dim], (Program.input (n := input.dim)).eval x⟩ := by
          simpa [Program.length, Program.eval] using
            read_push (earlier := (#[] : Array (List (Op.Value ℝ)))) (v := value)
        obtain ⟨result, hr, hs, hl⟩ := decodeTail_run P x [value] rest .input ⟨m, program⟩
          (#[].push [value]) (by simp [Program.length]) hread ht
        have hrun : G.run (exactPayload P) [value] = .ok result := by
          unfold IR.Graph.run
          rw [hnodes, IR.runNodes_cons, hfirst]
          exact hr
        have hsize := IR.Graph.run_size hrun
        unfold IR.Graph.eval
        rw [hrun, houts]
        simp only [bind, Except.bind, List.mapM_cons, List.mapM_nil]
        rw [← hsize, hs, hl]
        rfl
      next => contradiction
    next => contradiction

/-- The requested margins hold for the original graph under its exactly decoded payload. -/
theorem accepts_graph_sound (G : IR.Graph) (P : IR.Payload (ExecFloat.Binary 8 23))
    (input : FlatBox (ExecFloat.Binary 8 23))
    (cert : CROWNNode) (query : Query)
    (h : accepts G P input cert query = true) :
    ∃ decoded, decode G P input cert query = some decoded ∧
      nodeBounds cert query.outputId decoded.graph.inputDim decoded.graph.outputDim =
        some decoded.artifact.chain.bounds ∧
      G.outputs = #[⟨query.outputId, 0⟩] ∧
      0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
      0 < decoded.numConstraints ∧
      ∀ x : Tensor ℝ [decoded.graph.inputDim],
        Directed.RowEncloses
          ⟨decoded.graph.inputDim, realTensor decoded.graph.input.lo,
            realTensor decoded.graph.input.hi⟩ decoded.graph.inputDim
          (fun i => getAtOrZero x [i]) →
        ∃ y : Tensor ℝ [decoded.graph.outputDim],
          G.eval (exactPayload P) [⟨.value [decoded.graph.inputDim], x⟩] =
            .ok [⟨.value [decoded.graph.outputDim], y⟩] ∧
          ∀ i : Fin decoded.numConstraints,
            let margin := (Tensor.contract (realTensor decoded.inequalities.weights) y _ _
                  [decoded.numConstraints] .matVec) +
              realTensor decoded.inequalities.bias
            if decoded.strict then margin.getScalar i < 0 else margin.getScalar i ≤ 0 := by
  obtain ⟨decoded, hd, hg, ho, hb, hn, hm, hk, hs⟩ :=
    accepts_sound G P input cert query h
  have hid : query.outputId = G.nodes.size - 1 := by omega
  refine ⟨decoded, hd, hb, by rw [decodeGraph_outputs hg, hid], hn, hm, hk, ?_⟩
  intro x hx
  exact ⟨decoded.graph.program.eval x, decodeGraph_denote G P input decoded.graph hg x, hs x hx⟩

end TorchLean.Certificate.FiniteArtifact
