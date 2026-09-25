/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Cert.FiniteArtifact

/-!
# Real soundness of finite binary32 artifacts

Exact decoding, local dominance, and structural induction connect the supplied affine entries
to `Program.eval` on real inputs. No transfer theorem or semantic evaluation trace is an input
to the checker soundness theorem. `decode_coverage` proves node-count and final-entry coverage.
`FiniteArtifactSemantics` proves agreement with `NN.IR.Graph.denote` in `decodeGraph_denote`
and lifts acceptance to the original graph's exact-real semantics in `accepts_graph_sound`.
-/

public section

namespace NN.Verification.Cert.FiniteArtifact

open _root_.Spec TorchLean TorchLean.Tensor
open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model)
open NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN.Graph.CrownCertSoundness
open NN.Verification.CROWNQuery
open NN.Verification.Cert.RationalReflection
open scoped Spec.RationalAlgebraic BigOperators

/-- Successful scalar decoding retains the exact real denotation of the binary32 word. -/
theorem decodeScalar_sound (value : ExecFloat.Binary 8 23) (q : ℚ)
    (h : ExecFloat.Binary.toRat? value = some q) :
    Model.isFinite (ExecFloat.Binary.toModel value) = true ∧
      Model.toReal (ExecFloat.Binary.toModel value) = (q : ℝ) := by
  change Model.toRat? (ExecFloat.Binary.toModel value) = some q at h
  cases hd : Model.toDyadic? (ExecFloat.Binary.toModel value) with
  | none => simp [Model.toRat?, hd] at h
  | some d =>
      have hq : d.toRat = q := by simpa [Model.toRat?, hd] using h
      refine ⟨Model.isFinite_eq_true_of_toDyadic?_some hd, ?_⟩
      rw [Model.toReal_eq, hd, ← hq, FloatLib.Numerics.Dyadic.cast_toRat]

/-- Tensor decoding is the pointwise exact-real interpretation of finite binary32 words. -/
theorem decodeTensor_sound {s : Shape} (value : Tensor (ExecFloat.Binary 8 23) s)
    (q : Tensor ℚ s) (h : decodeTensor value = some q) :
    Tensor.Forall (fun v => Model.isFinite (ExecFloat.Binary.toModel v) = true) value ∧
      realTensor q = Tensor.map (fun v => Model.toReal (ExecFloat.Binary.toModel v)) value := by
  induction s with
  | scalar =>
      obtain ⟨r, hr, hq⟩ := Option.map_eq_some_iff.mp h
      subst q
      have hs := decodeScalar_sound value.item r hr
      refine ⟨hs.1, ?_⟩
      apply Tensor.ext_scalar
      simpa [realTensor] using hs.2.symm
  | dim n s ih =>
      obtain ⟨rows, hr, hq⟩ := Option.bind_eq_some_iff.mp h
      have hq : Tensor.dim rows = q := Option.some.inj hq
      subst q
      have hi (i : Fin n) := ih (value.unstack i) (rows i)
        (Tensor.Internal.sequenceFinM_get_of_eq_some hr i)
      refine ⟨fun i => (hi i).1, ?_⟩
      apply (Tensor.dimEquiv n s).injective
      funext i
      simpa [Tensor.dimEquiv, realTensor] using (hi i).2

/-- Evaluation of a difference of decoded affine forms is their exact real difference. -/
theorem affineDifference_eval {n m : Nat} (left right : AffineVec ℚ n m)
    (x : Tensor ℝ [n]) (i : Fin m) :
    (affineEvalAt (realAffine (affineDifference left right)) x).getScalar i =
      (affineEvalAt (realAffine left) x).getScalar i -
        (affineEvalAt (realAffine right) x).getScalar i := by
  simp only [affineEvalAt, getScalar_add_spec,
    Proofs.TensorAlgebra.getScalar_mat_vec_mul_spec, realAffine,
    realTensor_get2, realTensor_getScalar, affineDifference]
  simp [get2_eq_apply, getScalar_eq_apply, Tensor.subSpec, Tensor.map2Spec,
    sub_mul, Finset.sum_sub_distrib]
  ring

/-- Accepted dominance transfers an enclosure to the supplied artifact coefficients. -/
theorem dominates_sound {n m : Nat} (supplied exact : Bounds n m)
    (input : Box ℚ [n]) (h : dominates supplied exact input = true)
    (x : Tensor ℝ [n]) (y : Tensor ℝ [m])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x)
    (hy : exact.Encloses x y) : supplied.Encloses x y := by
  simp only [dominates, Bool.and_eq_true] at h
  obtain ⟨hl, hu⟩ := h
  intro i
  have hl' := checkUpper_sound false _ input hl x hx i
  have hu' := checkUpper_sound false _ input hu x hx i
  simp only [Bool.false_eq_true, ↓reduceIte, affineDifference_eval] at hl' hu'
  constructor
  · exact (sub_nonpos.mp hl').trans (hy i).1
  · exact (hy i).2.trans (sub_nonpos.mp hu')

/-- Exact-real evaluation of the decoded chain; artifact bounds and slopes do not alter values. -/
@[expose] noncomputable def Chain.eval {n m : Nat} (chain : Chain n m)
    (x : Tensor ℝ [n]) : Tensor ℝ [m] :=
  match chain with
  | .input _ => x
  | .linear parent layer _ =>
      Tensor.addSpec (matVecMulSpec (realTensor layer.weights) (parent.eval x))
        (realTensor layer.bias)
  | .relu parent _ _ => Activation.reluSpec (parent.eval x)

/-- Local same-artifact checks enclose the exact-real computation at the covered output. -/
theorem Chain.check_sound {n m : Nat} (chain : Chain n m) (input : Box ℚ [n])
    (h : chain.check input = true) (x : Tensor ℝ [n])
    (hx : Theorems.Semantics.encloses
      ⟨n, realTensor input.lo, realTensor input.hi⟩ x) :
    chain.bounds.Encloses x (chain.eval x) := by
  induction chain with
  | input bounds =>
      simp only [Chain.check, Bool.and_eq_true] at h
      exact dominates_sound bounds (Bounds.identity n) input
        h.2 x x hx (Bounds.identity_encloses x)
  | linear parent layer bounds ih =>
      simp only [Chain.check, Bool.and_eq_true] at h
      obtain ⟨⟨hp, _⟩, hd⟩ := h
      exact dominates_sound bounds _ input hd x _ hx
        (parent.bounds.linear_encloses layer.weights layer.bias x _ (ih hp))
  | relu parent alpha bounds ih =>
      simp only [Chain.check, Bool.and_eq_true] at h
      obtain ⟨⟨hp, ha⟩, hd⟩ := h
      exact dominates_sound bounds _ input hd x _ hx
        (parent.bounds.relu_encloses input alpha ha x _ hx (ih hp))

/-- Exact-real semantics of the program decoded independently from the graph and parameters. -/
@[expose] noncomputable def Program.eval {n m : Nat} (program : Program n m)
    (x : Tensor ℝ [n]) : Tensor ℝ [m] :=
  match program with
  | .input => x
  | .linear parent layer =>
      Tensor.addSpec (matVecMulSpec (realTensor layer.weights) (parent.eval x))
        (realTensor layer.bias)
  | .relu parent => Activation.reluSpec (parent.eval x)

/-- Removing certificate entries leaves the computed real value unchanged. -/
theorem Chain.eval_program {n m : Nat} (chain : Chain n m) (x : Tensor ℝ [n]) :
    chain.eval x = chain.program.eval x := by
  induction chain with
  | input _ => rfl
  | linear _ _ _ ih => simp only [Chain.eval, Chain.program, Program.eval, ih]
  | relu _ _ _ ih => simp only [Chain.eval, Chain.program, Program.eval, ih]

/-- Every requested inequality holds at the exact-real output of the decoded program. -/
@[expose] def Decoded.Safe (decoded : Decoded) : Prop :=
  ∀ x : Tensor ℝ [decoded.graph.inputDim],
    Theorems.Semantics.encloses
      ⟨decoded.graph.inputDim, realTensor decoded.graph.input.lo,
        realTensor decoded.graph.input.hi⟩ x →
    ∀ i : Fin decoded.numConstraints,
      let output := decoded.graph.program.eval x
      let margin := Tensor.addSpec
        (matVecMulSpec (realTensor decoded.inequalities.weights) output)
        (realTensor decoded.inequalities.bias)
      if decoded.strict then margin.getScalar i < 0 else margin.getScalar i ≤ 0

/-- The local checks prove every output margin, including strict inequalities. -/
theorem Decoded.check_sound (decoded : Decoded) (h : decoded.check = true) :
    decoded.Safe := by
  simp only [Decoded.check, Bool.and_eq_true] at h
  obtain ⟨⟨⟨_, _⟩, hc⟩, hm⟩ := h
  intro x hx i
  have he := decoded.artifact.chain.check_sound decoded.graph.input hc x hx
  rw [Chain.eval_program, decoded.artifact.sameProgram] at he
  have hq := decoded.artifact.chain.bounds.linear_encloses
    decoded.inequalities.weights decoded.inequalities.bias x
    (decoded.graph.program.eval x) he
  have hu := checkUpper_sound decoded.strict _ decoded.graph.input hm x hx i
  have hi := (hq i).2
  cases hs : decoded.strict <;> simp only [hs, Bool.false_eq_true, ↓reduceIte] at hu ⊢
  · exact hi.trans hu
  · exact hi.trans_lt hu

/-- Accepted checks cannot discharge an empty family of input, output, or margin coordinates. -/
theorem Decoded.check_nonempty (decoded : Decoded) (h : decoded.check = true) :
    0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
      0 < decoded.numConstraints := by
  simp only [Decoded.check, Bool.and_eq_true] at h
  exact of_decide_eq_true h.1.1.1

/-- Attaching an artifact retains its final supplied entry. -/
theorem attach_bounds (cert : NodeReplay.CROWNNodeCoreCertificate) {n m : Nat}
    (input : Box ℚ [n]) (program : Program n m) (attached : Attached program)
    (h : attach cert input program = some attached) :
    nodeBounds cert (program.length - 1) n m = some attached.chain.bounds := by
  cases program with
  | input =>
      obtain ⟨bounds, hb, he⟩ := Option.bind_eq_some_iff.mp h
      cases Option.some.inj he
      simpa [Program.length, Chain.bounds] using hb
  | linear parent layer =>
      obtain ⟨previous, _, ht⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨bounds, hb, he⟩ := Option.bind_eq_some_iff.mp ht
      cases Option.some.inj he
      simpa [Program.length, Chain.bounds] using hb
  | relu parent =>
      obtain ⟨previous, _, ht⟩ := Option.bind_eq_some_iff.mp h
      obtain ⟨bounds, hb, ht⟩ := Option.bind_eq_some_iff.mp ht
      dsimp only at ht
      split at ht
      · cases Option.some.inj ht
        simpa [Program.length, Chain.bounds] using hb
      · split at ht
        · obtain ⟨alpha, _, he⟩ := Option.bind_eq_some_iff.mp ht
          cases Option.some.inj he
          simpa [Program.length, Chain.bounds] using hb
        · contradiction

/-- Node-count and final-entry coverage, without an assertion about `Graph.denote`. -/
theorem decode_coverage (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : NodeReplay.CROWNNodeCoreCertificate) (query : OutputQuery) (decoded : Decoded)
    (h : decode g ps cert query = some decoded) :
    decodeGraph g ps = some decoded.graph ∧
      query.outputId + 1 = g.nodes.size ∧
      decoded.graph.program.length = g.nodes.size ∧
      nodeBounds cert query.outputId decoded.graph.inputDim decoded.graph.outputDim =
        some decoded.artifact.chain.bounds := by
  obtain ⟨graph, hg, ht⟩ := Option.bind_eq_some_iff.mp h
  split at ht
  next hvalid =>
    obtain ⟨artifact, ha, ht⟩ := Option.bind_eq_some_iff.mp ht
    dsimp only at ht
    split at ht
    next hdim =>
      obtain ⟨weights, _, ht⟩ := Option.bind_eq_some_iff.mp ht
      obtain ⟨bias, _, he⟩ := Option.bind_eq_some_iff.mp ht
      cases Option.some.inj he
      have hl := attach_bounds cert graph.input graph.program artifact ha
      have hid : query.outputId = graph.program.length - 1 := by omega
      exact ⟨hg, hvalid.2.2.2.2, hvalid.2.2.2.1, by simpa [hid] using hl⟩
    next => contradiction
  next => contradiction

/-- Acceptance proves the decoded program's margins without caller-supplied semantic premises. -/
theorem accepts_sound (g : Graph) (ps : ParamStore (ExecFloat.Binary 8 23))
    (cert : NodeReplay.CROWNNodeCoreCertificate) (query : OutputQuery)
    (h : accepts g ps cert query = true) :
    ∃ decoded, decode g ps cert query = some decoded ∧
      decodeGraph g ps = some decoded.graph ∧
      query.outputId + 1 = g.nodes.size ∧
      nodeBounds cert query.outputId decoded.graph.inputDim decoded.graph.outputDim =
        some decoded.artifact.chain.bounds ∧
      0 < decoded.graph.inputDim ∧ 0 < decoded.graph.outputDim ∧
      0 < decoded.numConstraints ∧ decoded.Safe := by
  unfold accepts at h
  cases hd : decode g ps cert query with
  | none => simp [hd] at h
  | some decoded =>
      have hc : decoded.check = true :=
        (show decoded.check = true ∧ replayAccepts g ps cert = true by
          simpa only [hd, Bool.and_eq_true] using h).1
      obtain ⟨hn, hm, hk⟩ := decoded.check_nonempty hc
      obtain ⟨hg, ho, _, hb⟩ := decode_coverage g ps cert query decoded hd
      exact ⟨decoded, rfl, hg, ho, hb, hn, hm, hk, decoded.check_sound hc⟩

end NN.Verification.Cert.FiniteArtifact
