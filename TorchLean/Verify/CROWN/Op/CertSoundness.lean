/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Directed
public import TorchLean.Verify.CROWN.Activation.IBP

/-!
# Soundness of the IBP certificate on operator graphs

`cert?_encloses` proves the certificate rule `cert?` sound in the sense of
`Directed.RuleEncloses`. Its endpoint min-max boxes for tanh and the sigmoid are sound by
`IBP.mapMinmax_sound_real`, and its clipped midpoint boxes for sine and cosine by
`sinMidpoint_sound_real` and `cosMidpoint_sound_real`. Every other box it gives is the shared
rule's (`cert?_eq_shared?`), sound by `ibp?_encloses` at `ℝ`. `certLocalOK_encloses` proves that
every box of a certificate that `CertLocalOK` accepts encloses the flat value of its node in every
table of node outputs that `SemLocalOK` accepts, when the seed boxes enclose the graph inputs their
input nodes read. Those tables are exactly the outputs of the successful runs
(`semLocalOK_iff_run`).
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.CROWN.Op

open Graph
open IBP Activation Context
open TorchLean.Tensor

noncomputable section

/-- The certificate rule `cert?` is a sound rule (`Directed.RuleEncloses`) for every
real payload `Pr`: at every operator node whose references' boxes enclose the flat values they read
and whose references to constants read their values in `Pr`, each box it returns encloses the flat
value of the node's first output. -/
theorem cert?_encloses {G : IR.Graph} {P Pr : IR.Payload ℝ} :
    Directed.RuleEncloses G P Pr (cert? G P) := by
  intro boxes inputs earlier id o refs B outs hconsts hpe hc h
  by_cases ho : BackendEnclosure o
  case neg =>
    rw [cert?_eq_shared? ho] at hc
    exact ibp?_encloses boxes inputs earlier id o refs B outs hconsts hpe
      (by simp only [ibp?, hc]) h
  obtain ⟨ins, outs', op⟩ := o
  cases op <;> (try cases ‹Op.NumericElement›) <;> (try cases ‹Op.Element›) <;>
    simp only [BackendEnclosure, ruleKind.eq_def, ite_eq_iff, reduceCtorEq, and_false,
      or_self] at ho
  case tanh s =>
    exact Directed.encloses_unary (F := MathFunctions.tanh)
      (T := fun Bx => some (toFlatBox (α := ℝ) Bx.dim
        (IBP.tanhMinmax (α := ℝ) (ofFlatBox (α := ℝ) Bx))))
      Op.rule?_tanh (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => by
        obtain rfl := Option.some.inj hb
        exact rowEncloses_of_contains_map
          (fun x hx => mapMinmax_sound_real _ CROWN.tanh_monotone _ x hx) hx)
      hpe (fun r hr => by
        rw [hr] at hc
        simp only [cert?] at hc
        rw [← hc]
        cases box? boxes r <;> rfl) h
  case sigmoid s =>
    exact Directed.encloses_unary (F := Context.sigmoid)
      (T := fun Bx => some (toFlatBox (α := ℝ) Bx.dim
        (IBP.sigmoidMinmax (α := ℝ) (ofFlatBox (α := ℝ) Bx))))
      Op.rule?_sigmoid (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => by
        obtain rfl := Option.some.inj hb
        exact rowEncloses_of_contains_map
          (fun x hx => mapMinmax_sound_real _ sigmoid_monotone_real _ x hx) hx)
      hpe (fun r hr => by
        rw [hr] at hc
        simp only [cert?] at hc
        rw [← hc]
        cases box? boxes r <;> rfl) h
  case sin s =>
    exact Directed.encloses_unary (F := Real.sin)
      (T := fun Bx => some (toFlatBox (α := ℝ) Bx.dim
        (IBP.sinMidpoint (α := ℝ) (ofFlatBox (α := ℝ) Bx))))
      Op.rule?_sin (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => by
        obtain rfl := Option.some.inj hb
        exact rowEncloses_of_contains_map (fun x hx => sinMidpoint_sound_real _ x hx) hx)
      hpe (fun r hr => by
        rw [hr] at hc
        simp only [cert?] at hc
        rw [← hc]
        cases box? boxes r <;> rfl) h
  case cos s =>
    exact Directed.encloses_unary (F := Real.cos)
      (T := fun Bx => some (toFlatBox (α := ℝ) Bx.dim
        (IBP.cosMidpoint (α := ℝ) (ofFlatBox (α := ℝ) Bx))))
      Op.rule?_cos (fun _ => rfl) (fun _ => rfl)
      (fun hx hb => by
        obtain rfl := Option.some.inj hb
        exact rowEncloses_of_contains_map (fun x hx => cosMidpoint_sound_real _ x hx) hx)
      hpe (fun r hr => by
        rw [hr] at hc
        simp only [cert?] at hc
        rw [← hc]
        cases box? boxes r <;> rfl) h

/-- Every box of a certificate (`CertLocalOK`) encloses the flat value of its node in a table of
node outputs that is the fixpoint of `evalNode?` (`SemLocalOK`), when every seed box encloses the
graph input its input node reads. No operator, parent shape or ordering premise. -/
theorem certLocalOK_encloses (G : IR.Graph) (P : IR.Payload ℝ)
    (seeds : Nat → Option (FlatBox ℝ)) (cert : Array (Option (FlatBox ℝ)))
    (inputs : List (Op.Value ℝ)) (values : Array (List (Op.Value ℝ)))
    (hcert : CertLocalOK G P seeds cert) (hsem : SemLocalOK G P inputs values)
    (hin : ∀ id k p, G.nodes[id]? = some (.input k p) → ∀ B, seeds id = some B →
      ∀ x y v, inputs[k]? = some x → p.read x = .ok y → flatOf ⟨p, y⟩ = some v →
        Directed.Encloses B v) :
    ∀ id, id < G.nodes.size → ∀ (B : FlatBox ℝ) (v : FlatTensor ℝ),
      cert[id]! = some B → runVal values id = some v → Directed.Encloses B v :=
  Directed.encloses_of_step G P P cert?_encloses seeds inputs values (Directed.realConsts_self P)
    hin (semLocalOK_iff_run.mp hsem) cert fun id hid => ⟨cert, fun _ _ => rfl, hcert.2 id hid⟩

end

end TorchLean.CROWN.Op
