/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness

/-!
# End-to-end CROWN certificate-checking framework (graph dialect)

This file provides the reusable end-to-end CROWN certificate-checking framework for graph dialects:

* an "end-to-end" enclosure theorem that composes:
  - a value semantics (`evalNode?` / `SemLocalOK`)
  - a per-node CROWN certificate (`FlatAffineBounds`)
  - a checker predicate (local consistency + per-op transfer soundness)

The main theorem is phrased in the standard "if the checker accepts, the bound holds" style.

## Relation to `auto_LiRPA` / `alpha-beta-CROWN`

This file is written in a producer/checker style that matches common workflows:

- an external engine (e.g. the LiRPA engine in `auto_LiRPA`, or the verifier in
  `alpha-beta-CROWN`) can act as an **untrusted producer** of per-node affine bounds;
- Lean hosts the checker theorem: given a concrete step function and proved `CrownTransferSound`
  rules, accepted per-node affine bounds imply an end-to-end enclosure against the Lean graph
  denotation.

We deliberately state the main theorem schematically in terms of an abstract `step` function:
plain CROWN, α-CROWN, and α/β-CROWN can all share the same end-to-end checker theorem once their
transfer rules are proved sound.

## Trust boundary for transcendental ops

IEEE-754 does not standardize libm transcendental functions. For ops like `exp`, `log`, `tanh`,
and `sigmoid`, soundness is therefore expressed as an explicit assumption in the checker predicate.
This matches the repo's existing design: rigorous transcendental enclosures are handled via an
oracle boundary (e.g. Arb-backed interval enclosures), not by assuming a particular `libm`.

## References (code)

- `auto_LiRPA`: https://github.com/Verified-Intelligence/auto_LiRPA
- `alpha-beta-CROWN`: https://github.com/Verified-Intelligence/alpha-beta-CROWN
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CrownCertSoundness

noncomputable section

/-!
## Pointwise interpretation of affine bounds

A `FlatAffineBounds α` represents (componentwise) affine lower/upper functions of a fixed
flattened input vector. To compare it to a concrete semantic value, we evaluate the affine maps at
the chosen input point.
-/

def affineEvalAt {α : Type} [Context α] {inDim outDim : Nat}
    (aff : AffineVec α inDim outDim) (x : Tensor α (.dim inDim .scalar)) : Tensor α (.dim outDim
      .scalar) :=
  Tensor.addSpec (Spec.matVecMulSpec (α := α) aff.A x) aff.c

/-- Evaluate affine lower/upper bounds at a concrete input point, yielding a `FlatBox`. -/
def boundsEvalAt {α : Type} [Context α]
    (b : FlatAffineBounds α) (x : Tensor α (.dim b.inDim .scalar)) : FlatBox α :=
  { dim := b.outDim
    lo := affineEvalAt (α := α) (inDim := b.inDim) (outDim := b.outDim) b.loAff x
    hi := affineEvalAt (α := α) (inDim := b.inDim) (outDim := b.outDim) b.hiAff x }

/-!
## Enclosure predicate

We reuse `Semantics.encloses` from `NN.MLTheory.CROWN.Graph` for componentwise enclosure.
-/

def EnclosesVec {α : Type} [Context α]
    (B : FlatBox α) (v : FlatVec α) : Prop :=
  ∃ h : B.dim = v.n,
    Theorems.Semantics.encloses (α := α) B (castDimScalar (α := α) (n := v.n) (n' := B.dim) h.symm
      v.v)

/-- Enclosure of a node value `v` under an affine bound `b`, evaluated at the designated input `x`.

This is a *well-typed* variant of `EnclosesVec (boundsEvalAt b x) v` that guards the dependent
dimension `b.inDim`.

In a well-formed CROWN certificate, every bound satisfies `b.inDim = ctx.inputDim`, so the guard
branch is the one that matters.
-/
def EnclosesAtInput {α : Type} [Context α]
    (ctx : AffineCtx) (x : Tensor α (.dim ctx.inputDim .scalar))
    (b : FlatAffineBounds α) (v : FlatVec α) : Prop :=
  ∃ h : b.inDim = ctx.inputDim,
    let x' : Tensor α (.dim b.inDim .scalar) :=
      castDimScalar (α := α) (n := ctx.inputDim) (n' := b.inDim) h.symm x
    EnclosesVec (α := α) (boundsEvalAt (α := α) b x') v

/-!
## Local semantic consistency

We reuse the real-valued graph dialect evaluator from `graph_cert_soundness.lean`, but we keep the
definition abstract: the CROWN enclosure theorem below holds for **any** locally-consistent
semantic interpretation `vals` (provided the graph is topologically sorted).

For IEEE32Exec, a separate evaluator can be plugged in later; the theorem below is stated for any
`vals` satisfying `SemLocalOK`.
-/

/-- Alias for the semantic value record used by `CertSoundness`. -/
abbrev Val := CertSoundness.Val
/-- Alias for the partial node evaluator used by `CertSoundness`. -/
abbrev evalNode? := CertSoundness.evalNode?
/-- Alias for the local semantic consistency predicate used by `CertSoundness`. -/
abbrev SemLocalOK := CertSoundness.SemLocalOK
/-- Alias for the topological-sorting predicate used by `CertSoundness`. -/
abbrev TopoSorted := CertSoundness.TopoSorted

/-!
## A CROWN certificate "step function" (checker interface)

The runtime `runCROWN` produces `FlatAffineBounds` by a forward pass. For a certificate/checker
architecture, we treat the producer as untrusted and phrase correctness as a local step condition:

* `CrownCertLocalOK`: the certificate is locally consistent with a (trusted) step function.
* `CrownTransferSound`: each node kind's step function is sound w.r.t. the semantics.

This file separates the generic checker theorem from per-operator transfer proofs. In particular,
transcendental operators are represented by explicit transfer-soundness assumptions supplied by the
backend or checker workflow.
-/

def getAff? {α : Type} [Context α] (cert : Array (Option (FlatAffineBounds α))) (pid : Nat) :
    Option (FlatAffineBounds α) :=
  if _h : pid < cert.size then cert[pid]! else none

/-!
`crownStepNode?` is a *parameter* to the checker theorem:
different certificate formats (plain CROWN, α/β-CROWN, split certificates) can share the same
end-to-end theorem as long as they provide a step function and discharge transfer soundness.
-/

def CrownCertLocalOK {α : Type} [Context α]
    (g : Graph) (step : Array (Option (FlatAffineBounds α)) → Nat → Option (FlatAffineBounds α))
    (cert : Array (Option (FlatAffineBounds α))) : Prop :=
  cert.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → cert[id]! = step cert id

/-!
## Transfer-rule soundness assumptions

`CrownTransferSound` is the *kernel* of the checker theorem: it states that the certificate's local
step rule is sound for each supported node kind.

For transcendental ops, this assumption is where you connect to an oracle model (e.g. Arb).
-/

def CrownTransferSound
    (g : Graph) (_ps : ParamStore ℝ) (_inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (step : Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ))
    (cert : Array (Option (FlatAffineBounds ℝ))) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    (∀ p : Nat, p ∈ (g.nodes[id]!).parents →
      match cert[p]!, vals[p]! with
      | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
      | _, _ => True) →
    match step cert id, vals[id]! with
    | some b, some v =>
        -- Per-node transfer soundness: assuming the parents satisfy their pointwise affine
        -- enclosures at `x`, the current node is enclosed as well.
        EnclosesAtInput (α := ℝ) ctx x b v
    | _, _ => True

/-!
## Checker implies enclosure

If a certificate is locally consistent and the transfer rules are sound, then every certified node
encloses the graph value computed at that node.

This theorem does not pick a certificate producer or a nonlinear backend. Those details come in
through `step` and `CrownTransferSound`.
-/

theorem crown_checker_encloses_semantics_match
    (g : Graph) (ps : ParamStore ℝ)
    (step : Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (htopo : TopoSorted g)
    (_hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hcert : CrownCertLocalOK (g := g) (step := step) cert)
    (hsound :
        CrownTransferSound (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
          (ctx := ctx) (x := x) (step := step) (cert := cert)) :
    ∀ id : Nat, id < g.nodes.size →
      match cert[id]!, vals[id]! with
      | some b, some v => EnclosesAtInput (α := ℝ) ctx x b v
      | _, _ => True := by
  classical
  intro id hid
  refine Nat.strong_induction_on id
      (p := fun k =>
        k < g.nodes.size →
          match cert[k]!, vals[k]! with
          | some b, some v => EnclosesAtInput (α := ℝ) ctx x b v
          | _, _ => True) ?_ hid
  intro k ih hk
  cases hcert with
  | intro hsz hstep =>
    have hck : cert[k]! = step cert k := hstep k hk
    have hparents :
        (∀ p : Nat, p ∈ (g.nodes[k]!).parents →
          match cert[p]!, vals[p]! with
          | some bp, some vp => EnclosesAtInput (α := ℝ) ctx x bp vp
          | _, _ => True) := by
      intro p hp
      have hpLt : p < k := htopo k hk p hp
      have hIH := ih p hpLt (lt_trans hpLt hk)
      simpa using hIH
    cases hcertk : cert[k]! with
    | none =>
        cases hvalk : vals[k]! <;> simp []
    | some b =>
        cases hvalk : vals[k]! with
        | none =>
            simp []
        | some v =>
            have hstepk : step cert k = some b := by
              have : some b = step cert k := by
                simpa [hcertk] using hck
              simpa using this.symm
            have h := hsound k hk hparents
            simpa [hcertk, hvalk, hstepk] using h

theorem crown_checker_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (step : Array (Option (FlatAffineBounds ℝ)) → Nat → Option (FlatAffineBounds ℝ))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ (.dim ctx.inputDim .scalar))
    (htopo : TopoSorted g)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hcert : CrownCertLocalOK (g := g) (step := step) cert)
    (hsound :
        CrownTransferSound (g := g) (_ps := ps) (_inputs := inputs) (vals := vals)
          (ctx := ctx) (x := x) (step := step) (cert := cert)) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v := by
  intro id hid b v hcertId hvalId
  have hmatch :=
    crown_checker_encloses_semantics_match
      (g := g) (ps := ps) (step := step) (cert := cert) (inputs := inputs) (vals := vals)
      (ctx := ctx) (x := x) htopo hsem hcert hsound id hid
  simpa [hcertId, hvalId] using hmatch

/-!
## IEEE32Exec specialization

For IEEE32 we still leave the node evaluator to the caller. The caller supplies the evaluator,
proves that `vals` is its trace, and proves that evaluating node `id` does not read `vals[id]!`.
The floating-point refinement theorem itself lives outside this checker lemma.
-/

abbrev IEEE32Val := FlatVec TorchLean.Floats.IEEE754.IEEE32Exec

abbrev IEEE32EvalNode? :=
  Array Node →
    ParamStore TorchLean.Floats.IEEE754.IEEE32Exec →
    Std.HashMap Nat IEEE32Val →
    Array (Option IEEE32Val) →
    Nat →
    Option IEEE32Val

/--
The IEEE32 node evaluator may inspect already-computed values, but not the slot it is supposed to
compute.
-/
def IEEE32EvalNoSelfDependency (evalNode? : IEEE32EvalNode?) : Prop :=
  ∀ (nodes : Array Node) (ps : ParamStore TorchLean.Floats.IEEE754.IEEE32Exec)
    (inputs : Std.HashMap Nat IEEE32Val)
    (vals vals' : Array (Option IEEE32Val)) (id : Nat),
      vals.size = vals'.size →
      (∀ j : Nat, j ≠ id → vals[j]! = vals'[j]!) →
      evalNode? nodes ps inputs vals id = evalNode? nodes ps inputs vals' id

def IEEE32SemLocalOK
    (evalNode? : IEEE32EvalNode?)
    (g : Graph) (ps : ParamStore TorchLean.Floats.IEEE754.IEEE32Exec)
    (inputs : Std.HashMap Nat IEEE32Val)
    (vals : Array (Option IEEE32Val)) : Prop :=
  IEEE32EvalNoSelfDependency evalNode? ∧
    vals.size = g.nodes.size ∧
    ∀ id : Nat, id < g.nodes.size →
      vals[id]! = evalNode? g.nodes ps inputs vals id

theorem crown_checker_encloses_semantics_ieee32exec_match
    (g : Graph) (_ps : ParamStore TorchLean.Floats.IEEE754.IEEE32Exec)
    (step : Array (Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec)) → Nat →
        Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec)))
    (evalNode? : IEEE32EvalNode?)
    (inputs : Std.HashMap Nat IEEE32Val)
    (vals : Array (Option IEEE32Val))
    (ctx : AffineCtx)
    (x : Tensor TorchLean.Floats.IEEE754.IEEE32Exec (.dim ctx.inputDim .scalar))
    [Preorder TorchLean.Floats.IEEE754.IEEE32Exec]
    (htopo : TopoSorted g)
    (_hsem : IEEE32SemLocalOK (evalNode? := evalNode?) (g := g) (ps := _ps) (inputs := inputs)
      (vals := vals))
    (hcert : CrownCertLocalOK (g := g) (step := step) cert)
    (hsound :
      ∀ id : Nat, id < g.nodes.size →
        (∀ p : Nat, p ∈ (g.nodes[id]!).parents →
          match cert[p]!, vals[p]! with
          | some bp, some vp =>
              EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x bp vp
          | _, _ => True) →
        match step cert id, vals[id]! with
        | some b, some v =>
            EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x b v
        | _, _ => True) :
    ∀ id : Nat, id < g.nodes.size →
      match cert[id]!, vals[id]! with
      | some b, some v =>
          EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x b v
      | _, _ => True := by
  classical
  intro id hid
  refine Nat.strong_induction_on id
      (p := fun k =>
        k < g.nodes.size →
          match cert[k]!, vals[k]! with
          | some b, some v =>
              EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x b v
          | _, _ => True) ?_ hid
  intro k ih hk
  cases hcert with
  | intro hsz hstep =>
    have hck : cert[k]! = step cert k := hstep k hk
    have hparents :
        (∀ p : Nat, p ∈ (g.nodes[k]!).parents →
          match cert[p]!, vals[p]! with
          | some bp, some vp =>
              EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x bp vp
          | _, _ => True) := by
      intro p hp
      have hpLt : p < k := htopo k hk p hp
      have hIH := ih p hpLt (lt_trans hpLt hk)
      simpa using hIH
    cases hcertk : cert[k]! with
    | none =>
        cases hvalk : vals[k]! <;> simp []
    | some b =>
        cases hvalk : vals[k]! with
        | none => simp []
        | some v =>
            have hstepk : step cert k = some b := by
              have : some b = step cert k := by
                simpa [hcertk] using hck
              simpa using this.symm
            have h := hsound k hk hparents
            simpa [hcertk, hvalk, hstepk] using h

theorem crown_checker_encloses_semantics_ieee32exec
    (g : Graph) (ps : ParamStore TorchLean.Floats.IEEE754.IEEE32Exec)
    (step : Array (Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec)) → Nat →
        Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec)))
    (evalNode? : IEEE32EvalNode?)
    (inputs : Std.HashMap Nat IEEE32Val)
    (vals : Array (Option IEEE32Val))
    (ctx : AffineCtx)
    (x : Tensor TorchLean.Floats.IEEE754.IEEE32Exec (.dim ctx.inputDim .scalar))
    [Preorder TorchLean.Floats.IEEE754.IEEE32Exec]
    (htopo : TopoSorted g)
    (hsem : IEEE32SemLocalOK (evalNode? := evalNode?) (g := g) (ps := ps) (inputs := inputs)
      (vals := vals))
    (hcert : CrownCertLocalOK (g := g) (step := step) cert)
    (hsound :
      ∀ id : Nat, id < g.nodes.size →
        (∀ p : Nat, p ∈ (g.nodes[id]!).parents →
          match cert[p]!, vals[p]! with
          | some bp, some vp =>
              EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x bp vp
          | _, _ => True) →
        match step cert id, vals[id]! with
        | some b, some v =>
            EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x b v
        | _, _ => True) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds TorchLean.Floats.IEEE754.IEEE32Exec) (v : IEEE32Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := TorchLean.Floats.IEEE754.IEEE32Exec) ctx x b v := by
  intro id hid b v hcertId hvalId
  have hmatch :=
    crown_checker_encloses_semantics_ieee32exec_match
      (g := g) (_ps := ps) (step := step) (cert := cert) (evalNode? := evalNode?)
      (inputs := inputs) (vals := vals) (ctx := ctx) (x := x) htopo hsem hcert hsound id hid
  simpa [hcertId, hvalId] using hmatch

end

end CrownCertSoundness

end NN.MLTheory.CROWN.Graph
