/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.Op.Cert

/-!
# The certificate computed node by node

`certify G P seeds` tabulates the certificate step (`tabulate`, in `TorchLean.Verify.CROWN.Op`).
On a graph whose references name earlier nodes (`TopoSorted`) it is a certificate
(`certify_localOK`), because the step at a node reads only the boxes of the nodes its references
name (`cert?_congr`, in `TorchLean.Verify.CROWN.Op.Cert`).
-/

@[expose] public section

namespace TorchLean.CROWN.Op


noncomputable section

/-- The certificate the step computes, node by node in order. -/
def certify (G : IR.Graph) (P : IR.Payload ℝ) (seeds : Nat → Option (FlatBox ℝ)) :
    Array (Option (FlatBox ℝ)) :=
  tabulate (step? G P (cert? G P) seeds) G.nodes.size

variable {G : IR.Graph} {P : IR.Payload ℝ}

/-- On a sorted graph, the entries an operator node reads are those of earlier nodes. -/
theorem box?_congr_of_topo (htopo : TopoSorted G) {β : Type} {c c' : Array (Option β)} {id : Nat}
    (hid : id < G.nodes.size) {o : IR.Operator} {refs : Array IR.Ref}
    (hn : G.nodes[id]? = some (.op o refs)) (h : ∀ j, j < id → c[j]? = c'[j]?) :
    ∀ r ∈ refs.toList, box? c r = box? c' r := by
  intro r hr
  have hlt := htopo id hid o refs hn r (by simpa using hr)
  simp [box?, h r.node hlt]

/-- The certificate step at a node of a sorted graph depends only on the entries of earlier
nodes. -/
theorem step?_cert?_congr (htopo : TopoSorted G) {seeds : Nat → Option (FlatBox ℝ)}
    {c c' : Array (Option (FlatBox ℝ))} {id : Nat} (h : ∀ j, j < id → c[j]? = c'[j]?) :
    step? G P (cert? G P) seeds c id = step? G P (cert? G P) seeds c' id := by
  unfold step?
  refine congrArg (Option.filter _) ?_
  cases hn : G.nodes[id]? with
  | none => rfl
  | some n =>
    cases n with
    | input _ _ => rfl
    | const _ => rfl
    | op o refs =>
      simp only
      split
      · exact cert?_congr (box?_congr_of_topo htopo (Array.getElem?_eq_some_iff.mp hn).1 hn h)
      · rfl

/-- On a sorted graph the computed certificate is a certificate. -/
theorem certify_localOK {seeds : Nat → Option (FlatBox ℝ)} (htopo : TopoSorted G) :
    CertLocalOK G P seeds (certify G P seeds) :=
  ⟨size_tabulate _ _, tabulate_fix fun _ _ _ _ h => step?_cert?_congr htopo h⟩

end

end TorchLean.CROWN.Op
