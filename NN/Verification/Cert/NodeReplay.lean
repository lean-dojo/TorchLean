/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCrownCertSoundness
public import NN.Runtime.PyTorch.Import.Core
public import NN.Verification.Util.Json
public import NN.MLTheory.CROWN.Graph -- shake: keep
public import NN.MLTheory.CROWN.Extras.BoundOpsIEEE32Exec -- shake: keep
public import NN.Spec.Core.Tensor -- shake: keep
public import NN.Verification.Util.FloatApprox -- shake: keep
public import Lean.Data.Json -- shake: keep

/-!
# Node-Certificate Replay

Parsing, exact comparison, and local replay for node-wise verification certificates.

The IBP, α-CROWN, and α/β-CROWN checkers all consume the same basic artifact shapes:
flat interval boxes, affine lower/upper bounds, and optional per-node vectors.  We keep those
format-level operations here so the individual checkers can focus on their propagation rule:

- `IBPNodeCert` checks interval propagation;
- `CROWNNodeCert` checks affine CROWN propagation;
- `CROWNNodeCertAlphaBeta` checks affine CROWN propagation with β phase information.

The JSON artifact is always untrusted. These helpers only parse and compare data; acceptance still
requires each checker to recompute the corresponding bound inside Lean. Interval claims are checked
by outward containment, while affine replay transcripts must match the executable binary32 result
exactly.
-/

@[expose] public section

namespace NN.Verification.Cert.NodeReplay

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Util
open NN.Verification.Json
open Import.PyTorch
open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open Lean Data Json
open TorchLean.Floats.IEEE754

/-- Whether every entry of a fixed-size vector is finite. -/
def finiteVec (n : Nat) (v : Fin n → Float) : Bool :=
  (List.finRange n).all fun i => (v i).isFinite

/-- Whether every entry of a fixed-size matrix is finite. -/
def finiteMatrix (rows cols : Nat) (A : Fin rows → Fin cols → Float) : Bool :=
  (List.finRange rows).all fun i => (List.finRange cols).all fun j => (A i j).isFinite

/-- Bitwise equality for shape-indexed binary32 tensors. -/
def exactEqTensor : {s : Shape} → Tensor IEEE32Exec s → Tensor IEEE32Exec s → Bool
  | .scalar, left, right => decide (left.item = right.item)
  | .dim n _, left, right =>
      (List.finRange n).all fun i => exactEqTensor (left.unstack i) (right.unstack i)

/-- A successful bitwise tensor comparison proves equality at every rank. -/
theorem exactEqTensor_eq_true {s : Shape} {t u : Tensor IEEE32Exec s}
    (h : exactEqTensor t u = true) : t = u := by
  induction s with
  | scalar =>
      apply Tensor.ext_scalar
      simpa [exactEqTensor] using of_decide_eq_true h
  | dim n s ih =>
      apply (Tensor.dimEquiv n s).injective
      funext i
      apply ih
      exact (List.all_eq_true.mp (by simpa [exactEqTensor] using h)) i (List.mem_finRange i)

/--
Whether `outer` contains `inner` componentwise.

This deliberately has no tolerance. A lower endpoint may be rounded farther down and an upper
endpoint farther up, but a serialized certificate may never move either endpoint inward. This is
the relation used for interval claims.
-/
def flatBoxContains (outer inner : FlatBox IEEE32Exec) : Bool :=
  if h : outer.dim = inner.dim then
    let innerLo := castDimScalar (α := IEEE32Exec) h.symm inner.lo
    let innerHi := castDimScalar (α := IEEE32Exec) h.symm inner.hi
    (List.finRange outer.dim).all fun i =>
      decide (outer.lo.getScalar i <= innerLo.getScalar i) &&
        decide (innerHi.getScalar i <= outer.hi.getScalar i)
  else
    false

/-- Bitwise equality for affine vectors, componentwise on matrix `A` and offset `c`. -/
def exactEqAffineVec {n m : Nat} (a b : AffineVec IEEE32Exec n m) : Bool :=
  exactEqTensor a.A b.A && exactEqTensor a.c b.c

/-- A successful bitwise affine-vector comparison proves equality of both affine components. -/
theorem exactEqAffineVec_eq_true {n m : Nat} {a b : AffineVec IEEE32Exec n m}
    (h : exactEqAffineVec a b = true) : a = b := by
  simp only [exactEqAffineVec, Bool.and_eq_true] at h
  cases a with
  | mk aA ac =>
      cases b with
      | mk bA bc =>
          have hA : aA = bA := exactEqTensor_eq_true h.1
          have hc : ac = bc := exactEqTensor_eq_true h.2
          cases hA
          cases hc
          rfl

/-- Bitwise equality for flattened affine lower/upper bounds. -/
def exactEqFlatAffineBounds (B1 B2 : FlatAffineBounds IEEE32Exec) : Bool :=
  if hin : B1.inDim = B2.inDim then
    if hout : B1.outDim = B2.outDim then
      match B1, B2 with
      | ⟨n1, m1, lo1, hi1⟩, ⟨_n2, _m2, lo2, hi2⟩ =>
          by
            cases hin
            cases hout
            exact exactEqAffineVec (n := n1) (m := m1) lo1 lo2 &&
              exactEqAffineVec (n := n1) (m := m1) hi1 hi2
    else false
  else false

/--
A successful bitwise affine-bound comparison proves equality of the dependent records, including
their input and output dimensions.
-/
theorem exactEqFlatAffineBounds_eq_true
    {B1 B2 : FlatAffineBounds IEEE32Exec}
    (h : exactEqFlatAffineBounds B1 B2 = true) : B1 = B2 := by
  unfold exactEqFlatAffineBounds at h
  split at h
  next hin =>
    split at h
    next hout =>
      cases B1 with
      | mk n1 m1 lo1 hi1 =>
          cases B2 with
          | mk n2 m2 lo2 hi2 =>
              simp only at hin hout
              subst n2
              subst m2
              simp only [Bool.and_eq_true] at h
              have hlo : lo1 = lo2 := exactEqAffineVec_eq_true h.1
              have hhi : hi1 = hi2 := exactEqAffineVec_eq_true h.2
              cases hlo
              cases hhi
              rfl
    next => simp at h
  next => simp at h

/-- Bitwise equality for optional affine bounds, used by the pure replay checker. -/
def exactEqOptionalAffineBounds
    (a b : Option (FlatAffineBounds IEEE32Exec)) : Bool :=
  match a, b with
  | none, none => true
  | some a, some b => exactEqFlatAffineBounds a b
  | _, _ => false

/-- Successful optional comparison proves equality of the optional affine bounds. -/
theorem exactEqOptionalAffineBounds_eq_true
    {a b : Option (FlatAffineBounds IEEE32Exec)}
    (h : exactEqOptionalAffineBounds a b = true) : a = b := by
  cases a with
  | none => cases b <;> simp_all [exactEqOptionalAffineBounds]
  | some a =>
      cases b with
      | none => simp_all [exactEqOptionalAffineBounds]
      | some b =>
          apply congrArg some
          exact exactEqFlatAffineBounds_eq_true h

/-!
## Proved local replay

The diagnostic node checkers below replay bounds incrementally so that they can report the first
bad node. The final acceptance decision also runs this pure check against the complete imported
certificate. Its soundness theorem supplies exactly the `CrownCertLocalOK` hypothesis expected by
the graph-level CROWN theorem.
-/

/--
Check that every affine entry is exactly the result of applying `step` to the imported certificate.

This is intentionally narrower than the complete certificate checker: parsing, IBP containment,
shape checks, and domain checks remain separate executable obligations. This predicate is the
local-replay component whose proposition-level meaning is `CrownCertLocalOK`.
-/
def crownLocalReplayAccepts
    (g : Graph)
    (step : Array (Option (FlatAffineBounds IEEE32Exec)) → Nat →
      Option (FlatAffineBounds IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds IEEE32Exec))) : Bool :=
  if hsize : cert.size = g.nodes.size then
    (List.finRange g.nodes.size).all fun i =>
      exactEqOptionalAffineBounds
        (cert[i.val]'(by rw [hsize]; exact i.isLt))
        (step cert i.val)
  else
    false

/-- Successful pure replay constructs the local-consistency hypothesis used by CROWN soundness. -/
theorem crownLocalReplayAccepts_eq_true
    (g : Graph)
    (step : Array (Option (FlatAffineBounds IEEE32Exec)) → Nat →
      Option (FlatAffineBounds IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds IEEE32Exec)))
    (haccept : crownLocalReplayAccepts g step cert = true) :
    NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownCertLocalOK
      (g := g) (step := step) cert := by
  unfold crownLocalReplayAccepts at haccept
  split at haccept
  next hsize =>
    refine ⟨hsize, ?_⟩
    intro id hid
    simp only [List.all_eq_true] at haccept
    have hentry := haccept ⟨id, hid⟩ (List.mem_finRange ⟨id, hid⟩)
    have heq := exactEqOptionalAffineBounds_eq_true hentry
    have hidCert : id < cert.size := by rw [hsize]; exact hid
    rw [getElem!_pos cert id hidCert]
    exact heq
  next => simp at haccept

/--
Combine the diagnostic checks with pure local replay. Acceptance of the combined decision always
supplies the proposition-level local CROWN consistency condition.
-/
def crownCertificateAccepts
    (g : Graph)
    (step : Array (Option (FlatAffineBounds IEEE32Exec)) → Nat →
      Option (FlatAffineBounds IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds IEEE32Exec)))
    (diagnosticsOk : Bool) : Bool :=
  diagnosticsOk && crownLocalReplayAccepts g step cert

/-- The complete Boolean acceptance decision implies local CROWN consistency. -/
theorem crownCertificateAccepts_eq_true
    (g : Graph)
    (step : Array (Option (FlatAffineBounds IEEE32Exec)) → Nat →
      Option (FlatAffineBounds IEEE32Exec))
    (cert : Array (Option (FlatAffineBounds IEEE32Exec)))
    (diagnosticsOk : Bool)
    (haccept : crownCertificateAccepts g step cert diagnosticsOk = true) :
    NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownCertLocalOK
      (g := g) (step := step) cert := by
  simp only [crownCertificateAccepts, Bool.and_eq_true] at haccept
  exact crownLocalReplayAccepts_eq_true g step cert haccept.2

/-- Parse a flat interval box (two arrays of floats) from JSON. -/
def parseFlatBox? (dim : Nat) (j : Json) : IO (Option (FlatBox IEEE32Exec)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObject j "ibp[i]"
      let loJ ← expectField o "lo" "ibp[i]"
      let hiJ ← expectField o "hi" "ibp[i]"
      let some loVec := parseFloatVec dim loJ
        | throw <| IO.userError s!"Invalid ibp[i].lo: expected float array length {dim}"
      let some hiVec := parseFloatVec dim hiJ
        | throw <| IO.userError s!"Invalid ibp[i].hi: expected float array length {dim}"
      unless finiteVec dim loVec && finiteVec dim hiVec do
        throw <| IO.userError "Invalid ibp[i]: interval bounds must be finite"
      unless (List.finRange dim).all (fun i => decide (loVec i <= hiVec i)) do
        throw <| IO.userError "Invalid ibp[i]: every lower bound must be <= its upper bound"
      let loT : Tensor IEEE32Exec [dim] :=
        TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn loVec)
      let hiT : Tensor IEEE32Exec [dim] :=
        TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn hiVec)
      pure (some { dim := dim, lo := loT, hi := hiT })

/--
Parse an optional α vector for α-CROWN ReLU relaxations.

The soundness theorem for the lower ReLU relaxation assumes every α component is in `[0, 1]`.
We enforce that contract at the JSON boundary, so a malformed external certificate cannot be
accepted by executable checking while relying on proof hypotheses that are false.
-/
def parseAlphaVec? (dim : Nat) (j : Json) (ctx : String := "alpha[i]") :
    IO (Option (FlatTensor IEEE32Exec)) := do
  match j with
  | .null => pure none
  | _ =>
      let some v := parseFloatVec dim j
        | throw <| IO.userError s!"Invalid {ctx}: expected float array length {dim}"
      for k in List.finRange dim do
        let a := v k
        if !a.isFinite || a < 0.0 || a > 1.0 then
          throw <| IO.userError
            s!"Invalid {ctx}[{k.val}]: α-CROWN requires 0 ≤ alpha ≤ 1, got {a}"
      let t : Tensor IEEE32Exec [dim] :=
        TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn v)
      pure (some { n := dim, v := t })

/-- Parse flattened affine bounds (lower/upper) from JSON. -/
def parseAffineBounds? (inDim outDim : Nat) (j : Json) :
    IO (Option (FlatAffineBounds IEEE32Exec)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObject j "crown[i]"
      let loAJ ← expectField o "loA" "crown[i]"
      let loCJ ← expectField o "loC" "crown[i]"
      let hiAJ ← expectField o "hiA" "crown[i]"
      let hiCJ ← expectField o "hiC" "crown[i]"
      let some loA := parseFloatMatrix outDim inDim loAJ
        | throw <| IO.userError s!"Invalid crown[i].loA: expected matrix {outDim}x{inDim}"
      let some hiA := parseFloatMatrix outDim inDim hiAJ
        | throw <| IO.userError s!"Invalid crown[i].hiA: expected matrix {outDim}x{inDim}"
      let some loC := parseFloatVec outDim loCJ
        | throw <| IO.userError s!"Invalid crown[i].loC: expected float array length {outDim}"
      let some hiC := parseFloatVec outDim hiCJ
        | throw <| IO.userError s!"Invalid crown[i].hiC: expected float array length {outDim}"
      unless finiteMatrix outDim inDim loA && finiteMatrix outDim inDim hiA &&
          finiteVec outDim loC && finiteVec outDim hiC do
        throw <| IO.userError "Invalid crown[i]: affine bounds must be finite"
      let loAff : AffineVec IEEE32Exec inDim outDim :=
        { A := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.matrix loA)
          c := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn loC) }
      let hiAff : AffineVec IEEE32Exec inDim outDim :=
        { A := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.matrix hiA)
          c := TorchLean.Tensor.map IEEE32Exec.ofFloat (TorchLean.Tensor.ofFn hiC) }
      pure (some { inDim := inDim, outDim := outDim, loAff := loAff, hiAff := hiAff })

/--
Shared in-memory representation for node-wise CROWN-style certificates.

Plain α-CROWN uses these fields directly. α/β-CROWN extends the same core artifact with a β phase
array, so parsing the common fields here keeps the two checkers from drifting apart.
-/
structure CROWNNodeCoreCertificate where
  /-- Affine-propagation context, including the chosen input node and flattened input dimension. -/
  ctx : AffineCtx
  /-- Optional per-node interval bounds used by nonlinear CROWN steps. -/
  ibp : Array (Option (FlatBox IEEE32Exec))
  /-- Optional per-node affine lower/upper bounds. -/
  crown : Array (Option (FlatAffineBounds IEEE32Exec))
  /-- Optional per-node α values for ReLU lower relaxations. -/
  alpha : Array (Option (FlatTensor IEEE32Exec))

/--
Parse the fields shared by α-CROWN and α/β-CROWN node certificates.

The producer may omit `"alpha"`; in that case we treat every node as having no custom α vector.
Whenever α is present, `parseAlphaVec?` enforces the `[0,1]` side condition required by the ReLU
relaxation proof.
-/
def parseCROWNNodeCoreCertificate (g : Graph) (topObj : Json) :
    IO CROWNNodeCoreCertificate := do
  let ctxObj ← expectFieldObject topObj "ctx" "top-level"
  let inputId ← expectFieldNat ctxObj "inputId" "ctx"
  let inputDim ← expectFieldNat ctxObj "inputDim" "ctx"
  let ctx : AffineCtx := { inputId := inputId, inputDim := inputDim }

  let ibpArr ← expectFieldArray topObj "ibp" "top-level"
  let crownArr ← expectFieldArray topObj "crown" "top-level"
  let alphaArr ←
    match ← optionalField? topObj "alpha" "top-level" with
    | none => pure (Array.replicate g.nodes.size Json.null)
    | some alphaJ => expectArray alphaJ "top-level.alpha"

  if hIbpSize : ibpArr.size = g.nodes.size then
    if hCrownSize : crownArr.size = g.nodes.size then
      if hAlphaSize : alphaArr.size = g.nodes.size then
        let mut ibp : Array (Option (FlatBox IEEE32Exec)) := Array.mkEmpty g.nodes.size
        let mut crown : Array (Option (FlatAffineBounds IEEE32Exec)) := Array.mkEmpty g.nodes.size
        let mut alpha : Array (Option (FlatTensor IEEE32Exec)) := Array.mkEmpty g.nodes.size

        for i in List.finRange g.nodes.size do
          let node := g.nodes[i.val]'i.isLt
          let outDim := node.outShape.size
          let hIbp : i.val < ibpArr.size := by
            rw [hIbpSize]
            exact i.isLt
          let hCrown : i.val < crownArr.size := by
            rw [hCrownSize]
            exact i.isLt
          let hAlpha : i.val < alphaArr.size := by
            rw [hAlphaSize]
            exact i.isLt
          let ibpJson := ibpArr[i.val]'hIbp
          let ibpEntry ← parseFlatBox? outDim ibpJson
          ibp := ibp.push ibpEntry
          let crownJson := crownArr[i.val]'hCrown
          let crownEntry ← parseAffineBounds? ctx.inputDim outDim crownJson
          crown := crown.push crownEntry
          let alphaJson := alphaArr[i.val]'hAlpha
          let alphaEntry ← parseAlphaVec? outDim alphaJson
          alpha := alpha.push alphaEntry

        pure { ctx := ctx, ibp := ibp, crown := crown, alpha := alpha }
      else
        throw <| IO.userError s!"alpha length {alphaArr.size} ≠ g.nodes.size {g.nodes.size}"
    else
      throw <| IO.userError s!"crown length {crownArr.size} ≠ g.nodes.size {g.nodes.size}"
  else
    throw <| IO.userError s!"ibp length {ibpArr.size} ≠ g.nodes.size {g.nodes.size}"

/-- Check that an optional per-node certificate array contains all parents of node `id`. -/
def parentsOk {β : Type} (g : Graph) (cert : Array (Option β)) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      node.parents.all (fun p =>
        if p < id then
          match cert[p]? with
          | some (some _) => true
          | _ => false
        else
          false)

/-- Safe lookup for optional flat boxes used by certificate-side shape checks. -/
def getFlatBox? (cert : Array (Option (FlatBox IEEE32Exec))) (id : Nat) :
    Option (FlatBox IEEE32Exec) :=
  match cert[id]? with
  | some box? => box?
  | none => none

/--
Check that binary elementwise parent boxes have the same flattened size as each other and as the
node output. This closes the hole where a malformed certificate could make the runtime helper use
the left box on a dimension mismatch.
-/
def binaryElementwiseBoxesMatchOutput
    (g : Graph) (cert : Array (Option (FlatBox IEEE32Exec))) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      match NN.IR.binaryParents? node.parents with
      | some (p1, p2) =>
          match getFlatBox? cert p1, getFlatBox? cert p2 with
          | some B1, some B2 =>
              B1.dim == B2.dim && B1.dim == node.outShape.size
          | _, _ => false
      | _ => false

/-- Check whether a flat box is entirely inside the positive domain needed by true `log`. -/
def flatBoxStrictlyAbove (B : FlatBox IEEE32Exec) (eps : IEEE32Exec) : Bool :=
  match B with
  | ⟨n, lo, hi⟩ =>
      let flo := getDimScalarFn (α := IEEE32Exec) lo
      let fhi := getDimScalarFn (α := IEEE32Exec) hi
      (List.finRange n).all (fun i =>
        match flo i, fhi i with
        | .scalar l, .scalar u => l > eps && u > eps)

/-- Check that every coordinate interval lies strictly on one side of zero. -/
def flatBoxExcludesZero (B : FlatBox IEEE32Exec) : Bool :=
  match B with
  | ⟨n, lo, hi⟩ =>
      let flo := getDimScalarFn (α := IEEE32Exec) lo
      let fhi := getDimScalarFn (α := IEEE32Exec) hi
      (List.finRange n).all (fun i =>
        match flo i, fhi i with
        | .scalar l, .scalar u => l > IEEE32Exec.posZero || u < IEEE32Exec.posZero)

/--
Domain and shape preconditions that must hold before a node-wise certificate checker replays a
bound step. These executable checks mirror the side conditions that the mathematical rules need.
-/
def ibpNodePreconditionsOk
    (g : Graph) (cert : Array (Option (FlatBox IEEE32Exec))) (id : Nat) : Bool :=
  match g.nodes[id]? with
  | none => false
  | some node =>
      match node.kind with
      | .add | .sub | .mulElem | .maxElem | .minElem =>
          binaryElementwiseBoxesMatchOutput g cert id
      | .log =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              match getFlatBox? cert p1 with
              | some B => flatBoxStrictlyAbove B Context.defaultEpsilon
              | none => false
          | _ => false
      | .inv =>
          match NN.IR.unaryParent? node.parents with
          | some p1 =>
              match getFlatBox? cert p1 with
              | some B => flatBoxExcludesZero B
              | none => false
          | _ => false
      | _ => true

/-- Pretty-printer for a flat box, used in certificate mismatch messages. -/
def prettyFlatBox (B : FlatBox IEEE32Exec) : String :=
  s!"dim={B.dim}, lo={Spec.pretty B.lo}, hi={Spec.pretty B.hi}"

/-- Pretty-printer for affine bounds, used in certificate mismatch messages. -/
def prettyAffineBounds (B : FlatAffineBounds IEEE32Exec) : String :=
  s!"inDim={B.inDim}, outDim={B.outDim}, loA={Spec.pretty B.loAff.A}, loC={Spec.pretty B.loAff.c}"

/--
Common node-level checker for CROWN-style affine certificates.

The only difference between α-CROWN and α/β-CROWN is how the candidate affine bound is recomputed.
Everything after that point is the same: parent availability, IBP side conditions, dimensions,
exact binary32 transcript equality, and diagnostic messages.
-/
def checkCROWNLikeNode
    (label : String) (g : Graph)
    (authoritativeIbp : Array (Option (FlatBox IEEE32Exec)))
    (authoritativeCrown : Array (Option (FlatAffineBounds IEEE32Exec)))
    (certCrown : Array (Option (FlatAffineBounds IEEE32Exec)))
    (ctx : AffineCtx)
    (id : Nat)
    (computed? : Option (FlatAffineBounds IEEE32Exec)) : IO Bool := do
  let some node := g.nodes[id]?
    | IO.eprintln s!"[{label}] node {id}: out of bounds for graph with {g.nodes.size} nodes"
      pure false
  let needsParents :=
    match node.kind with
    | .input | .const _ => false
    | _ => true
  if needsParents && !(parentsOk g authoritativeCrown id) then
    IO.eprintln s!"[{label}] node {id}: parent affine bounds missing or not topo"
    return false
  if !(ibpNodePreconditionsOk g authoritativeIbp id) then
    IO.eprintln
      (s!"[{label}] node {id}: authoritative trace violates shape/domain preconditions " ++
        s!"for {repr node.kind}")
    return false

  let certCrown? :=
    match certCrown[id]? with
    | some entry => entry
    | none => none
  match certCrown?, computed? with
  | none, _ =>
      IO.eprintln s!"[{label}] node {id}: certificate missing (null)"
      pure false
  | _, none =>
      IO.eprintln s!"[{label}] node {id}: Lean propagation produced no affine bound"
      pure false
  | some certB, some leanB =>
      if certB.inDim ≠ ctx.inputDim then
        IO.eprintln s!"[{label}] node {id}: cert inDim {certB.inDim} ≠ ctx.inputDim {ctx.inputDim}"
        pure false
      else if certB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[{label}] node {id}: cert outDim {certB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if leanB.outDim ≠ node.outShape.size then
        IO.eprintln
          (s!"[{label}] node {id}: Lean outDim {leanB.outDim} ≠ " ++
            s!"outShape.size {node.outShape.size}")
        pure false
      else if exactEqFlatAffineBounds certB leanB then
        pure true
      else
        IO.eprintln s!"[{label}] mismatch at node {id} ({repr node.kind})"
        IO.eprintln s!"  cert: {prettyAffineBounds certB}"
        IO.eprintln s!"  lean: {prettyAffineBounds leanB}"
        pure false

end NN.Verification.Cert.NodeReplay
