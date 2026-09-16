/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.AlphaBeta
public import NN.MLTheory.CROWN.Proofs.GraphRunibpEndToEnd

/-!
# α-CROWN and α/β-CROWN End-to-End Enclosure

The transfer theorems `alphaCrown_transfer_sound` and `alphaBetaCrown_transfer_sound` take the
IBP boxes as an assumption (`IBPEnclosesVals`): every IBP box present at a node must enclose the
semantic value at that node. This file discharges that assumption from the IBP soundness theorem
`CertSoundness.cert_encloses_semantics` and states the fully composed corollaries.

The composed statements are quantified over the node ids at which both a certificate entry and a
semantic value are present, or, given `CrownCertCovers`, over every node. The `match` form of the
underlying checker theorems is trivially true at nodes where either side is missing, so these are
the forms a caller should use.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

namespace AlphaCrownTransferSoundness

noncomputable section

open CrownCertSoundness
open CertSoundness

/-! ## Discharging `IBPEnclosesVals` -/

/--
A locally consistent IBP certificate encloses every locally consistent semantic interpretation.

This is `CertSoundness.cert_encloses_semantics` repackaged in the shape the α-CROWN transfer
theorems expect. The hypotheses are exactly those of the IBP theorem: the graph is topologically
sorted and uses only supported ops, the IBP boxes replay the checker step at every node, the
values replay the evaluator at every node, and the input values lie inside their seed boxes.
-/
theorem ibp_encloses_vals_of_cert_local_ok
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hibp : CertLocalOK (g := g) (ps := ps) ibp)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsEnclosed g ps inputs) :
    IBPEnclosesVals (ibp := ibp) (vals := vals) := by
  intro id hid
  have hid' : id < g.nodes.size := by
    rw [← hsem.1]
    exact hid
  exact cert_encloses_semantics g ps ibp inputs vals htopo hsupp hibp hsem hinputs id hid'

/--
The IBP boxes computed by the total checker pass `runIBP?` enclose the values computed by the total
evaluator `evalGraphRec`.
-/
theorem ibp_encloses_vals_runIBP?
    (g : Graph) (ps : ParamStore ℝ)
    (inputs : Std.HashMap Nat Val)
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hinputs : InputsEnclosed g ps inputs) :
    IBPEnclosesVals (ibp := runIBP? g ps) (vals := evalGraphRec g ps inputs) :=
  ibp_encloses_vals_of_cert_local_ok g ps (runIBP? g ps) inputs (evalGraphRec g ps inputs)
    htopo hsupp (runIBP?_CertLocalOK g ps htopo) (evalGraphRec_SemLocalOK g ps inputs htopo hsupp)
    hinputs

/-! ## α-CROWN -/

/--
A locally replayed α-CROWN certificate encloses every corresponding graph value, with the IBP
boxes justified rather than assumed.

Compared with `alphaCrown_transfer_sound`, the hypothesis `IBPEnclosesVals` is replaced by the
IBP-side hypotheses `Supported g`, `CertLocalOK g ps ibp`, and `InputsEnclosed g ps inputs`.

The conclusion is stated for every node id at which the certificate has an entry `b` and the
semantics has a value `v`; a node missing either one carries no claim.
-/
theorem alphaCrown_cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hibp : CertLocalOK (g := g) (ps := ps) ibp)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlpha g ps ibp alpha ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v := by
  apply crown_checker_encloses_semantics
    (g := g) (ps := ps) (step := stepAlpha g ps ibp alpha ctx)
    (cert := cert) (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hcert
  exact alphaCrown_transfer_sound
    (g := g) (ps := ps) (ibp := ibp) (alpha := alpha) (cert := cert)
    (inputs := inputs) (vals := vals) (ctx := ctx) (x := x)
    htopo hsem hinputs
    (ibp_encloses_vals_of_cert_local_ok g ps ibp inputs vals htopo hsupp hibp hsem hinputsEnc)
    halpha

/--
Every node of a fully covered α-CROWN certificate is enclosed.

`CrownCertCovers g cert vals` says that every node has both a certificate entry and a semantic
value. It is needed because the checker theorems say nothing about nodes at which the certificate
producer or the evaluator returned `none`; with coverage, the conclusion cannot hold through a
missing entry.
-/
theorem alphaCrown_cert_encloses_all_nodes
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hibp : CertLocalOK (g := g) (ps := ps) ibp)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlpha g ps ibp alpha ctx) cert)
    (hcoverage : CrownCertCovers g cert vals) :
    ∀ id : Nat, id < g.nodes.size →
      ∃ b v, cert[id]! = some b ∧ vals[id]! = some v ∧
        EnclosesAtInput (α := ℝ) ctx x b v := by
  intro id hid
  obtain ⟨b, v, hb, hv⟩ := hcoverage.2.2 id hid
  exact ⟨b, v, hb, hv,
    alphaCrown_cert_encloses_semantics g ps ibp alpha cert inputs vals ctx x
      htopo hsupp hibp hsem hinputsEnc hinputs halpha hcert id hid b v hb hv⟩

/--
α-CROWN enclosure against the total IBP pass and the total evaluator.

Here the IBP boxes are `runIBP? g ps` and the values are `evalGraphRec g ps inputs`, so the only
remaining hypotheses about the graph are `TopoSorted`, `Supported`, and the input conditions.
-/
theorem alphaCrown_cert_encloses_evalGraphRec
    (g : Graph) (ps : ParamStore ℝ)
    (alpha : Array (Option (FlatTensor ℝ)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlpha g ps (runIBP? g ps) alpha ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        (evalGraphRec g ps inputs)[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v :=
  alphaCrown_cert_encloses_semantics g ps (runIBP? g ps) alpha cert inputs
    (evalGraphRec g ps inputs) ctx x htopo hsupp (runIBP?_CertLocalOK g ps htopo)
    (evalGraphRec_SemLocalOK g ps inputs htopo hsupp) hinputsEnc hinputs halpha hcert

/-! ## α/β-CROWN -/

/--
A locally replayed α/β-CROWN certificate encloses every corresponding graph value, with the IBP
boxes justified rather than assumed.

This is `alphaBetaCrown_cert_encloses_semantics` with `IBPEnclosesVals` discharged from the IBP
soundness theorem; see `alphaCrown_cert_encloses_semantics` for the hypothesis trade.
-/
theorem alphaBetaCrown_cert_encloses_semantics'
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hibp : CertLocalOK (g := g) (ps := ps) ibp)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlphaBeta g ps ibp alpha beta ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        vals[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v :=
  alphaBetaCrown_cert_encloses_semantics g ps ibp alpha beta cert inputs vals ctx x
    htopo hsem hinputs
    (ibp_encloses_vals_of_cert_local_ok g ps ibp inputs vals htopo hsupp hibp hsem hinputsEnc)
    halpha hcert

/--
Every node of a fully covered α/β-CROWN certificate is enclosed. See
`alphaCrown_cert_encloses_all_nodes` for the role of `CrownCertCovers`.
-/
theorem alphaBetaCrown_cert_encloses_all_nodes
    (g : Graph) (ps : ParamStore ℝ)
    (ibp : Array (Option (FlatBox ℝ)))
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hibp : CertLocalOK (g := g) (ps := ps) ibp)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert : CrownCertLocalOK (g := g) (step := stepAlphaBeta g ps ibp alpha beta ctx) cert)
    (hcoverage : CrownCertCovers g cert vals) :
    ∀ id : Nat, id < g.nodes.size →
      ∃ b v, cert[id]! = some b ∧ vals[id]! = some v ∧
        EnclosesAtInput (α := ℝ) ctx x b v := by
  intro id hid
  obtain ⟨b, v, hb, hv⟩ := hcoverage.2.2 id hid
  exact ⟨b, v, hb, hv,
    alphaBetaCrown_cert_encloses_semantics' g ps ibp alpha beta cert inputs vals ctx x
      htopo hsupp hibp hsem hinputsEnc hinputs halpha hcert id hid b v hb hv⟩

/--
α/β-CROWN enclosure against the total IBP pass and the total evaluator; see
`alphaCrown_cert_encloses_evalGraphRec`.
-/
theorem alphaBetaCrown_cert_encloses_evalGraphRec
    (g : Graph) (ps : ParamStore ℝ)
    (alpha : Array (Option (FlatTensor ℝ)))
    (beta : Array (Option (Array Int)))
    (cert : Array (Option (FlatAffineBounds ℝ)))
    (inputs : Std.HashMap Nat Val)
    (ctx : AffineCtx) (x : Tensor ℝ [ctx.inputDim])
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hinputsEnc : InputsEnclosed g ps inputs)
    (hinputs : InputsMatch (inputs := inputs) (ctx := ctx) x)
    (halpha : AlphaOK (alpha := alpha))
    (hcert :
      CrownCertLocalOK (g := g) (step := stepAlphaBeta g ps (runIBP? g ps) alpha beta ctx) cert) :
    ∀ id : Nat, id < g.nodes.size →
      ∀ (b : FlatAffineBounds ℝ) (v : Val),
        cert[id]! = some b →
        (evalGraphRec g ps inputs)[id]! = some v →
        EnclosesAtInput (α := ℝ) ctx x b v :=
  alphaBetaCrown_cert_encloses_semantics' g ps (runIBP? g ps) alpha beta cert inputs
    (evalGraphRec g ps inputs) ctx x htopo hsupp (runIBP?_CertLocalOK g ps htopo)
    (evalGraphRec_SemLocalOK g ps inputs htopo hsupp) hinputsEnc hinputs halpha hcert

end

end AlphaCrownTransferSoundness

end NN.MLTheory.CROWN.Graph
