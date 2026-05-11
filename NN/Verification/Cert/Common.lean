/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Core.Utils
public import NN.Verification.Util.FloatApprox
public import NN.Verification.Util.Json
public import Lean.Data.Json

/-!
# Common Certificate Helpers

Shared JSON/parsing and approximate-comparison utilities for node-wise verification certificates.

The IBP, α-CROWN, and α/β-CROWN checkers all consume the same basic artifact shapes:
flat interval boxes, affine lower/upper bounds, and optional per-node vectors.  We keep those
format-level helpers here so the individual checkers can focus on their propagation rule:

- `IBPNodeCert` checks interval propagation;
- `CROWNNodeCert` checks affine CROWN propagation;
- `CROWNNodeCertAlphaBeta` checks affine CROWN propagation with β phase information.

The JSON artifact is always untrusted. These helpers only parse and compare data; acceptance still
requires each checker to recompute the corresponding bound inside Lean. The float tolerances here
exist because JSON stores decimal strings/numbers, not because the external producer is trusted.
-/

@[expose] public section

namespace NN.Verification.Cert.Common

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open NN.Verification.Util
open NN.Verification.Json
open Import.PyTorch
open _root_.Spec
open _root_.Spec.Tensor
open Lean Data Json

/--
Approximate equality for flat scalar tensors (length-`n` vectors), up to an absolute tolerance.

This is used when comparing Lean-recomputed bounds to decimal-serialized JSON certificate values.
-/
def approxEqTensor {n : Nat} (t u : Tensor Float (.dim n .scalar)) (tol : Float) : Bool :=
  match t, u with
  | .dim ft, .dim fu =>
      (List.finRange n).all (fun i =>
        match ft i, fu i with
        | .scalar a, .scalar b => approxEq a b (tol := tol))

/-- Approximate equality for flat matrices (shape `m × n`), up to an absolute tolerance. -/
def approxEqMatrix {m n : Nat}
    (A B : Tensor Float (.dim m (.dim n .scalar))) (tol : Float) : Bool :=
  match A, B with
  | .dim rA, .dim rB =>
      (List.finRange m).all (fun i =>
        match rA i, rB i with
        | .dim cA, .dim cB =>
            (List.finRange n).all (fun j =>
              match cA j, cB j with
              | .scalar a, .scalar b => approxEq a b (tol := tol)))

/-- Approximate equality for `FlatBox` bounds, componentwise on `lo` and `hi`. -/
def approxEqFlatBox (B1 B2 : FlatBox Float) (tol : Float) : Bool :=
  if h : B1.dim = B2.dim then
    match B1, B2 with
    | ⟨n, lo1, hi1⟩, ⟨_m, lo2, hi2⟩ =>
        by
          cases h
          exact approxEqTensor (n := n) lo1 lo2 tol && approxEqTensor (n := n) hi1 hi2 tol
  else false

/-- Approximate equality for affine vectors, componentwise on matrix `A` and offset `c`. -/
def approxEqAffineVec {n m : Nat} (a b : AffineVec Float n m) (tol : Float) : Bool :=
  approxEqMatrix (m := m) (n := n) a.A b.A tol && approxEqTensor (n := m) a.c b.c tol

/-- Approximate equality for flattened affine lower/upper bounds. -/
def approxEqFlatAffineBounds (B1 B2 : FlatAffineBounds Float) (tol : Float) : Bool :=
  if hin : B1.inDim = B2.inDim then
    if hout : B1.outDim = B2.outDim then
      match B1, B2 with
      | ⟨n1, m1, lo1, hi1⟩, ⟨_n2, _m2, lo2, hi2⟩ =>
          by
            cases hin
            cases hout
            exact approxEqAffineVec (n := n1) (m := m1) lo1 lo2 tol &&
              approxEqAffineVec (n := n1) (m := m1) hi1 hi2 tol
    else false
  else false

/-- Parse a flat interval box (two arrays of floats) from JSON. -/
def parseFlatBox? (dim : Nat) (j : Json) : IO (Option (FlatBox Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObj j "ibp[i]"
      let loJ ← expectField o "lo" "ibp[i]"
      let hiJ ← expectField o "hi" "ibp[i]"
      let some loVec := parseFloatVec dim loJ
        | throw <| IO.userError s!"Invalid ibp[i].lo: expected float array length {dim}"
      let some hiVec := parseFloatVec dim hiJ
        | throw <| IO.userError s!"Invalid ibp[i].hi: expected float array length {dim}"
      let loT : Tensor Float (.dim dim .scalar) := Spec.vectorTensor loVec
      let hiT : Tensor Float (.dim dim .scalar) := Spec.vectorTensor hiVec
      pure (some { dim := dim, lo := loT, hi := hiT })

/-- Parse a flat vector from JSON, accepting `null` as an absent optional vector. -/
def parseFlatVec? (dim : Nat) (j : Json) (ctx : String := "alpha[i]") :
    IO (Option (FlatVec Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let some v := parseFloatVec dim j
        | throw <| IO.userError s!"Invalid {ctx}: expected float array length {dim}"
      let t : Tensor Float (.dim dim .scalar) := Spec.vectorTensor v
      pure (some { n := dim, v := t })

/-- Parse flattened affine bounds (lower/upper) from JSON. -/
def parseAffineBounds? (inDim outDim : Nat) (j : Json) : IO (Option (FlatAffineBounds Float)) := do
  match j with
  | .null => pure none
  | _ =>
      let o ← expectObj j "crown[i]"
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
      let loAff : AffineVec Float inDim outDim :=
        { A := Spec.matrixTensor loA, c := Spec.vectorTensor loC }
      let hiAff : AffineVec Float inDim outDim :=
        { A := Spec.matrixTensor hiA, c := Spec.vectorTensor hiC }
      pure (some { inDim := inDim, outDim := outDim, loAff := loAff, hiAff := hiAff })

/-- Check that an optional per-node certificate array contains all parents of node `id`. -/
def parentsOk {β : Type} (g : Graph) (cert : Array (Option β)) (id : Nat) : Bool :=
  let node := g.nodes[id]!
  node.parents.all (fun p =>
    if p < id then
      match cert[p]! with
      | some _ => true
      | none => false
    else
      false)

/-- Pretty-printer for a flat box, used in certificate mismatch messages. -/
def prettyFlatBox (B : FlatBox Float) : String :=
  s!"dim={B.dim}, lo={Spec.pretty B.lo}, hi={Spec.pretty B.hi}"

/-- Pretty-printer for affine bounds, used in certificate mismatch messages. -/
def prettyAffineBounds (B : FlatAffineBounds Float) : String :=
  s!"inDim={B.inDim}, outDim={B.outDim}, loA={Spec.pretty B.loAff.A}, loC={Spec.pretty B.loAff.c}"

end NN.Verification.Cert.Common
