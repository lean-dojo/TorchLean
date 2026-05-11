/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Models.Mlp

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Proof-level IBP certificate soundness for the graph dialect

This file is the “strong” version of IBP certificate checking:

* `NN.Verification.IBPNodeCert` checks (at runtime, with floats) that a Python certificate
  matches Lean’s computed per-node bounds.
* Here we prove (at theorems level, over `ℝ`) that a *per-node certificate* implies enclosure
  of the *graph semantics*.

## Verifier Graph Semantics

The current verifier graph dialect uses a slightly non-standard convention for `.matmul`:
instead of a 2-parent matrix multiply node, the weight matrix is supplied externally via
`ParamStore.matmulW`, and the node behaves like a bias-free linear layer `y = W x`.

So in this file we define a semantics that matches that convention:

* `.linear` uses `ParamStore.linearWB` and means `y = W x + b`.
* `.matmul` uses `ParamStore.matmulW` and means `y = W x` (bias = 0).

This semantics is stated on **flattened vectors** (a `FlatVec`), matching what `runIBP` computes.

## Scope

To keep the proof manageable while matching the LiRPA graph patterns used by the verifier, we prove soundness
for the following node kinds:

* `.input`, `.const`, `.detach`
* `.add`, `.sub`, `.mul_elem`, `.relu`
* `.linear`, `.matmul` (the ParamStore-based dialect)
* `.tanh`, `.sigmoid`, `.sin`, `.cos`

All other ops are considered unsupported in this theorem file.

## What is actually proven?

We prove:

> If a certificate `cert` satisfies the *local IBP step equation* at every node
> (i.e. every node’s box is exactly what IBP would compute from its parents’ boxes and params),
> and if the concrete inputs used by the semantics are enclosed by the input boxes,
> then every node’s concrete semantic value is enclosed by its certified interval box.

This is the standard “local soundness + induction over a topological order” proof structure
used in certified verifiers.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Basic types and predicates

We work over `ℝ` because it has the right order structure for “true” soundness theorems.
The runtime checkers operate over `Float` (fast, executable), and can be used to connect a
Python-produced floating certificate to the *same* computations in Lean.
-/

abbrev Val := FlatVec ℝ

/-- Componentwise enclosure predicate for a tensor point inside a `FlatBox`. -/
abbrev encloses (B : FlatBox ℝ) (x : Tensor ℝ (.dim B.dim .scalar)) : Prop :=
  NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) B x

/-- `EnclosesBox B v` means the value vector `v` lies inside the interval box `B`.

We phrase enclosure using the existing `Sem.encloses` predicate, but our semantic values are
`FlatVec`s (carrying their dimension as a `Nat`), so we also carry a dimension equality witness.
-/
def EnclosesBox (B : FlatBox ℝ) (v : Val) : Prop :=
  ∃ h : B.dim = v.n, encloses B (castDimScalar (α := ℝ) h.symm v.v)

/-!
## Denotational (value) semantics for the verifier graph dialect

The semantics is defined as a *safe* `Option` evaluator:

* If required parameters are missing, it returns `none`.
* If parents are missing (not yet evaluated) or dimensions mismatch, it returns `none`.

This keeps the semantic definition total, and avoids the partial `get!` used in the runtime
propagation code.
-/

/-- Safe lookup of a previously computed parent value. -/
def getVal? (vals : Array (Option Val)) (pid : Nat) : Option Val :=
  if _h : pid < vals.size then vals[pid]! else none

/-- Value semantics for a single node in the supported dialect (over `ℝ`). -/
def evalNode? (nodes : Array Node) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) (id : Nat) : Option Val :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      inputs[id]?
  | .const _ =>
      ps.constVals[id]?
  | .detach =>
      match node.parents with
      | p1 :: _ => getVal? vals p1
      | _ => none
    | .add =>
        match node.parents with
        | p1 :: p2 :: _ =>
            match getVal? vals p1, getVal? vals p2 with
            | some x, some y =>
                if h : x.n = y.n then
                  -- Use an explicit cast rather than `by simpa [h]` to keep later proofs stable.
                  let yv : Tensor ℝ (.dim x.n .scalar) :=
                    castDimScalar (α := ℝ) (Eq.symm h) y.v
                  some { n := x.n, v := Tensor.addSpec (α := ℝ) x.v yv }
                else
                  none
            | _, _ => none
        | _ => none
  | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ (.dim x.n .scalar) :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.subSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              if h : x.n = y.n then
                let yv : Tensor ℝ (.dim x.n .scalar) :=
                  castDimScalar (α := ℝ) (Eq.symm h) y.v
                some { n := x.n, v := Tensor.mulSpec (α := ℝ) x.v yv }
              else
                none
          | _, _ => none
      | _ => none
  | .maxPool2d kH kW stride =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                      let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.maxPool2dMultiSpec (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                          (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .maxPool2dPad kH kW stride padding =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW
                        stride padding
                      let layer : Spec.MaxPool2DSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.maxPool2dMultiSpecPad (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                            padding)
                          (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .avgPool2d kH kW stride =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShape inC inH inW kH kW stride
                      let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.avgPool2dMultiSpec (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride)
                          (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .avgPool2dPad kH kW stride padding =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              match nodes[p1]!.outShape with
              | .dim inC (.dim inH (.dim inW .scalar)) =>
                  let expectedInDim := inC * inH * inW
                  if hIn : x.n = expectedInDim then
                    if hkH : kH = 0 then
                      none
                    else if hkW : kW = 0 then
                      none
                    else if hStride : stride = 0 then
                      none
                    else
                      let sIn : Shape := .dim inC (.dim inH (.dim inW .scalar))
                      let sFlat : Shape := .dim x.n .scalar
                      have hsize : sFlat.size = sIn.size := by
                        simp [Shape.size, sFlat, sIn, hIn, expectedInDim, Nat.mul_assoc]
                      let xCHW : Tensor ℝ sIn := Tensor.reshapeSpec (α := ℝ) (s₁ := sFlat) (s₂ :=
                        sIn) x.v hsize
                      let outShape : Shape := Spec.pool2dMultiOutShapePad inC inH inW kH kW
                        stride padding
                      let layer : Spec.AvgPool2DSpec kH kW stride hkH hkW hStride := {}
                      let y : Tensor ℝ outShape :=
                        Spec.avgPool2dMultiSpecPad (α := ℝ) (kH := kH) (kW := kW)
                          (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
                            padding)
                          (h1 := hkH) (h2 := hkW) (layer := layer) (input := xCHW)
                      let flat := Tensor.flattenSpec (α := ℝ) y
                      some { n := outShape.size, v := flat }
                  else none
              | _ => none
          | none => none
      | _ => none
  | .relu =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.reluSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .tanh =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.tanhSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x => some { n := x.n, v := Activation.sigmoidSpec (α := ℝ) x.v }
          | none => none
      | _ => none
  | .sin =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.sin z) x.v }
          | none => none
      | _ => none
  | .cos =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              some
                { n := x.n
                  v := Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.cos z) x.v }
          | none => none
      | _ => none
  | .linear =>
        match node.parents with
        | p1 :: _ =>
            match getVal? vals p1, ps.linearWB[id]? with
            | some x, some p =>
                if h : x.n = p.n then
                  let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) h x.v
                  let yv : Tensor ℝ (.dim p.m .scalar) :=
                    Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv
                  some { n := p.m, v := yv }
                else
                  none
            | _, _ => none
        | _ => none
  | .matmul =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1, ps.matmulW[id]? with
          | some x, some p =>
              if h : x.n = p.n then
                let xv : Tensor ℝ (.dim p.n .scalar) := castDimScalar (α := ℝ) h x.v
                let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m .scalar)
                let yv : Tensor ℝ (.dim p.m .scalar) :=
                  Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv
                some { n := p.m, v := yv }
              else
                none
          | _, _ => none
      | _ => none
  | .sum =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              let onesRow : Tensor ℝ (.dim 1 (.dim x.n .scalar)) :=
                Spec.fill (α := ℝ) 1 (.dim 1 (.dim x.n .scalar))
              let y : Tensor ℝ (.dim 1 .scalar) := Spec.matVecMulSpec (α := ℝ) onesRow x.v
              some { n := 1, v := y }
          | none => none
      | _ => none
  | .reshape _ _ | .flatten _ =>
      match node.parents with
      | p1 :: _ =>
          match getVal? vals p1 with
          | some x =>
              if h : x.n = node.outShape.size then
                let xv : Tensor ℝ (.dim node.outShape.size .scalar) :=
                  castDimScalar (α := ℝ) h x.v
                some { n := node.outShape.size, v := xv }
              else
                none
          | none => none
      | _ => none
  | .concat _ =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getVal? vals p1, getVal? vals p2 with
          | some x, some y =>
              match x.v, y.v with
              | .dim fx, .dim fy =>
                  let outDim := x.n + y.n
                  let z : Tensor ℝ (.dim outDim .scalar) :=
                    Tensor.dim (fun i =>
                      Fin.addCases (fun i1 => fx i1) (fun i2 => fy i2) i)
                  some { n := outDim, v := z }
          | _, _ => none
      | _ => none
  | _ =>
      none

/-- Evaluate an entire graph in node-id order using `evalNode?`. -/
def evalGraph? (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) :
    Array (Option Val) :=
  let init := Array.replicate g.nodes.size none
  (List.finRange g.nodes.size).foldl
    (fun acc i => acc.set! i (evalNode? g.nodes ps inputs acc i))
    init

/-!
Even though we provided an executable `evalGraph?`, the **main soundness theorem** below does not
depend on it.

Reason: proving properties about the `foldl` evaluator would introduce a lot of “bookkeeping”
lemmas about `Array.set!` and list folds.

Instead, we state soundness for *any* array `vals` that is a **local model** of the semantics step:
each node’s value must equal `evalNode?` computed from its parents’ values.

This is a standard technique in proof engineering: separate “semantic consistency” from
“the particular implementation of the evaluator”.
-/

/-!
## Local semantic consistency (`SemLocalOK`)

`SemLocalOK g ps inputs vals` means:

* `vals` has the correct length, and
* each entry `vals[id]` equals `evalNode?` computed from the full array `vals`.

For a DAG (and only for a DAG), this is exactly the property that `vals` is a valid interpretation
of the graph semantics.

Existence and uniqueness of `vals` are evaluator-correctness facts. This file proves the certificate
theorem in the reusable form: for any semantic interpretation `vals`, a locally-correct certificate
encloses it.
-/

def SemLocalOK (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val)) : Prop :=
  vals.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → vals[id]! = evalNode? g.nodes ps inputs vals id

/-!
## The IBP “certificate step” (safe, total)

This is a safe version of `propagateIBPNode` tailored to the subset of ops we prove soundness for.
It defines what it means for a per-node certificate to be “locally well-formed”.

Important: the runtime implementation `propagateIBPNode` uses `get!` on parent boxes; it assumes
topological order and that earlier boxes exist. Here we avoid partiality by returning `none`
whenever parents are missing.
-/

def getBox? (cert : Array (Option (FlatBox ℝ))) (pid : Nat) : Option (FlatBox ℝ) :=
  if _h : pid < cert.size then cert[pid]! else none

/--
Safe per-node IBP step for the checker semantics.

This is the total (option-returning) analogue of the runtime `propagateIBPNode`, restricted to the
ops handled in the soundness development.
-/
def certStepNode? (nodes : Array Node) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) (id :
  Nat) :
    Option (FlatBox ℝ) :=
  let node := nodes[id]!
  match node.kind with
  | .input =>
      ps.inputBoxes[id]?
  | .const _ =>
      match ps.constVals[id]? with
      | none => none
      | some v => some { dim := v.n, lo := v.v, hi := v.v }
  | .detach =>
      match node.parents with
      | p1 :: _ => getBox? cert p1
      | _ => none
  | .add =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 => some (box_add (α := ℝ) B1 B2)
          | _, _ => none
      | _ => none
  | .sub =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 => some (box_sub (α := ℝ) B1 B2)
          | _, _ => none
      | _ => none
  | .mul_elem =>
      match node.parents with
      | p1 :: p2 :: _ =>
          match getBox? cert p1, getBox? cert p2 with
          | some B1, some B2 => box_mul_elem (α := ℝ) B1 B2
          | _, _ => none
      | _ => none
  | .relu =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some B => some (box_relu (α := ℝ) B)
          | none => none
      | _ => none
  | .tanh =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .sigmoid =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := Xin.dim) (ofFlatBox
                (α := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .sin =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .cos =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin =>
              let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := Xin.dim) (ofFlatBox (α
                := ℝ) Xin)
              some (toFlatBox (α := ℝ) Xin.dim yB)
          | none => none
      | _ => none
  | .linear =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin => ibp_linear (α := ℝ) id ps Xin
          | none => none
      | _ => none
  | .matmul =>
      match node.parents with
      | p1 :: _ =>
          match getBox? cert p1 with
          | some Xin => ibp_matmul (α := ℝ) id ps Xin
          | none => none
      | _ => none
  | _ =>
      none

/-- A certificate is *locally consistent* if every node equals `certStepNode?` at that node. -/
def CertLocalOK (g : Graph) (ps : ParamStore ℝ) (cert : Array (Option (FlatBox ℝ))) : Prop :=
  cert.size = g.nodes.size ∧
  ∀ id : Nat, id < g.nodes.size → cert[id]! = certStepNode? g.nodes ps cert id

/-!
## Op-level soundness lemmas (enclosure for each supported step)

These lemmas are the building blocks for the final “certificate ⇒ semantics enclosure” theorem.

This proof reuses the following existing components:

* Linear IBP soundness over `ℝ` is already proved in `NN.MLTheory.CROWN.mlp` as
  `NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real`.
* For add/sub/relu on `FlatBox`, the graph file already contains enclosure lemmas in
  `NN.MLTheory.CROWN.Graph.Theorems.Semantics`.
-/

private theorem relu_mono_real : ∀ {a b : ℝ}, a ≤ b →
    Activation.Math.reluSpec (α := ℝ) a ≤ Activation.Math.reluSpec (α := ℝ) b := by
  intro a b hab
  -- `relu_spec x = max x 0`
  simpa [Activation.Math.reluSpec] using max_le_max hab (le_rfl : (0:ℝ) ≤ 0)

private theorem add_mono_real : ∀ {a b c d : ℝ}, a ≤ b → c ≤ d → a + c ≤ b + d := by
  intro a b c d hab hcd
  exact add_le_add hab hcd

private theorem sub_mono_real : ∀ {a b c d : ℝ}, a ≤ b → d ≤ c → a - c ≤ b - d := by
  intro a b c d hab hdc
  have hneg : -c ≤ -d := neg_le_neg hdc
  have : a + (-c) ≤ b + (-d) := add_le_add hab hneg
  simpa [sub_eq_add_neg] using this

private lemma if_lt_eq_min (a b : ℝ) :
    (if a < b then a else b) = min a b := by
  by_cases h : a < b
  · simp [h, min_eq_left (le_of_lt h)]
  · have h' : b ≤ a := le_of_not_gt h
    simp [h, min_eq_right h']

private lemma if_gt_eq_max (a b : ℝ) :
    (if a > b then a else b) = max a b := by
  by_cases h : a > b
  · simp [h, max_eq_left (le_of_lt h)]
  · have h' : a ≤ b := le_of_not_gt h
    simp [h, max_eq_right h']

private lemma mul_const_bounds {a ly uy y : ℝ} (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (a * ly) (a * uy) ≤ a * y ∧ a * y ≤ max (a * ly) (a * uy) := by
  by_cases ha : 0 ≤ a
  · have hlo : a * ly ≤ a * y := mul_le_mul_of_nonneg_left hy ha
    have hhi : a * y ≤ a * uy := mul_le_mul_of_nonneg_left hy' ha
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have ha' : a ≤ 0 := le_of_not_ge ha
    have hlo : a * uy ≤ a * y := mul_le_mul_of_nonpos_left hy' ha'
    have hhi : a * y ≤ a * ly := mul_le_mul_of_nonpos_left hy ha'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

private lemma mul_var_bounds {lx ux x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) :
    min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) := by
  by_cases hy : 0 ≤ y
  · have hlo : lx * y ≤ x * y := mul_le_mul_of_nonneg_right hx hy
    have hhi : x * y ≤ ux * y := mul_le_mul_of_nonneg_right hx' hy
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have hy' : y ≤ 0 := le_of_not_ge hy
    have hlo : ux * y ≤ x * y := mul_le_mul_of_nonpos_right hx' hy'
    have hhi : x * y ≤ lx * y := mul_le_mul_of_nonpos_right hx hy'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

private lemma interval_mul_bounds
    {lx ux ly uy x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y ∧
      x * y ≤ max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy)) := by
  have h_lx : min (lx * ly) (lx * uy) ≤ lx * y ∧ lx * y ≤ max (lx * ly) (lx * uy) :=
    mul_const_bounds (a := lx) hy hy'
  have h_ux : min (ux * ly) (ux * uy) ≤ ux * y ∧ ux * y ≤ max (ux * ly) (ux * uy) :=
    mul_const_bounds (a := ux) hy hy'
  have h_x : min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) :=
    mul_var_bounds (lx := lx) (ux := ux) (x := x) (y := y) hx hx'
  -- Lower bound: corners ≤ each endpoint product, hence ≤ min endpoint product, hence ≤ x*y.
  have hC_lx : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ lx * y := by
    exact le_trans (min_le_left _ _) h_lx.1
  have hC_ux : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ ux * y := by
    exact le_trans (min_le_right _ _) h_ux.1
  have hC_to_min : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ min (lx * y) (ux * y)
    :=
    le_min hC_lx hC_ux
  have hlo : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y :=
    le_trans hC_to_min h_x.1
  -- Upper bound: x*y ≤ max endpoint product ≤ max corner maxes.
  let C : ℝ := max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy))
  have hmax_lx : lx * y ≤ C := le_trans h_lx.2 (le_max_left _ _)
  have hmax_ux : ux * y ≤ C := le_trans h_ux.2 (le_max_right _ _)
  have hmax_to_C : max (lx * y) (ux * y) ≤ C := max_le hmax_lx hmax_ux
  have hhi : x * y ≤ C := le_trans h_x.2 hmax_to_C
  simpa [C] using And.intro hlo hhi

/-! Helpers: our bound propagation uses `BoundOps.min2/max2`, which are defined via `decide (a >
  b)`.
For `ℝ` these coincide with `min/max`. -/

private lemma min2_eq_min (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.min2 a b = min a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_right hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_left hab]

private lemma max2_eq_max (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.max2 a b = max a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_left hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_right hab]

private theorem box_mul_elem_sound_real (n : Nat)
    (lo1 hi1 lo2 hi2 x y : Tensor ℝ (.dim n .scalar))
    (hx : encloses { dim := n, lo := lo1, hi := hi1 } x)
    (hy : encloses { dim := n, lo := lo2, hi := hi2 } y) :
    ∀ {B : FlatBox ℝ},
      box_mul_elem (α := ℝ)
          { dim := n, lo := lo1, hi := hi1 }
          { dim := n, lo := lo2, hi := hi2 } = some B →
        EnclosesBox B ⟨n, Tensor.mulSpec (α := ℝ) x y⟩ := by
  classical
  cases lo1 with
  | dim l1 =>
    cases hi1 with
    | dim u1 =>
      cases lo2 with
      | dim l2 =>
        cases hi2 with
        | dim u2 =>
          cases x with
          | dim fx =>
            cases y with
            | dim fy =>
              intro B hB
              unfold box_mul_elem at hB
              simp at hB
              symm at hB
              rw [hB]
              refine ⟨rfl, ?_⟩
              dsimp [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses, getDimScalarFn,
                castDimScalar]
              intro i
              have hx_i := hx i
              have hy_i := hy i
              cases hLx : l1 i with
              | scalar lx =>
                cases hUx : u1 i with
                | scalar ux =>
                  cases hLy : l2 i with
                  | scalar ly =>
                    cases hUy : u2 i with
                    | scalar uy =>
                      cases hX : fx i with
                      | scalar xv =>
                        cases hY : fy i with
                        | scalar yv =>
                          have hx' : lx ≤ xv ∧ xv ≤ ux := by
                            simpa [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses,
                              getDimScalarFn,
                              hLx, hUx, hX] using hx_i
                          have hy' : ly ≤ yv ∧ yv ≤ uy := by
                            simpa [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses,
                              getDimScalarFn,
                              hLy, hUy, hY] using hy_i
                          have hMul :=
                            interval_mul_bounds (lx := lx) (ux := ux) (ly := ly) (uy := uy)
                              (x := xv) (y := yv) (hx := hx'.1) (hx' := hx'.2) (hy := hy'.1) (hy' :=
                                hy'.2)
                          simpa [Tensor.mulSpec, Tensor.map2Spec, min2_eq_min, max2_eq_max, hLx,
                            hUx, hLy, hUy, hX, hY] using hMul

/-!
### Casting lemmas (avoid `cases` on `B.dim = v.n`)

`FlatBox` and `FlatVec` carry their dimensions in dependent types, so it is tempting to
`cases` equalities like `h : B.dim = v.n` to “align” types. In Lean this can easily trigger
dependent elimination failures when the equality mentions fields of dependent records.

Instead, we keep such equalities as *data* and move tensors/boxes across them using
`castDimScalar` / `castBoxDim`. The following small lemmas are proved once (by `cases` on
*fresh* Nat equalities) and then used throughout the main proof without ever `cases`-ing on
`B.dim = v.n` directly.
 -/

private lemma castDimScalar_trans {n n' n'' : Nat}
    (h₁ : n = n') (h₂ : n' = n'') (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) (Eq.trans h₁ h₂) t
      = castDimScalar (α := ℝ) h₂ (castDimScalar (α := ℝ) h₁ t) := by
  cases h₁
  cases h₂
  rfl

private lemma castDimScalar_map_spec {n n' : Nat}
    (h : n = n') (f : ℝ → ℝ) (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.mapSpec (α := ℝ) f t)
      = Tensor.mapSpec (α := ℝ) f (castDimScalar (α := ℝ) h t) := by
  cases h
  rfl

private lemma castDimScalar_add_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.addSpec (α := ℝ) x y)
      = Tensor.addSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

private lemma castDimScalar_sub_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.subSpec (α := ℝ) x y)
      = Tensor.subSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

private lemma castDimScalar_mul_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.mulSpec (α := ℝ) x y)
      = Tensor.mulSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

private lemma contains_castBoxDim_iff {n n' : Nat}
    (h : n = n') (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar)) :
    Box.contains (α := ℝ) (castBoxDim (α := ℝ) h B) (castDimScalar (α := ℝ) h x)
      ↔ Box.contains (α := ℝ) B x := by
  cases h
  simp [castBoxDim, castDimScalar]

private lemma encloses_castDim {B : FlatBox ℝ} {n' : Nat}
    (h : B.dim = n') (x : Tensor ℝ (.dim B.dim .scalar)) :
    encloses B x →
      encloses { dim := n'
                 lo := castDimScalar (α := ℝ) h B.lo
                 hi := castDimScalar (α := ℝ) h B.hi }
        (castDimScalar (α := ℝ) h x) := by
  intro hx
  cases B with
  | mk n lo hi =>
      -- Now `h : n = n'`; rewrite indices and finish by `simp`.
      cases h
      simpa [encloses, castDimScalar] using hx

private theorem encloses_of_contains {n : Nat}
    (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar)) :
    Box.contains (α := ℝ) B x → encloses (toFlatBox (α := ℝ) n B) x := by
  intro hx
  cases B with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          rw [encloses]
          rw [NN.MLTheory.CROWN.Box.contains.eq_def] at hx
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                simpa [toFlatBox, getDimScalarFn, hL, hU, hX] using hx_i

private theorem contains_of_encloses
    (B : FlatBox ℝ) (x : Tensor ℝ (.dim B.dim .scalar)) :
    encloses B x → Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B) x := by
  intro hx
  cases B with
  | mk n' lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          rw [encloses] at hx
          rw [NN.MLTheory.CROWN.Box.contains.eq_def]
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                simpa [ofFlatBox, getDimScalarFn, hL, hU, hX] using hx_i

/-!
### Point Boxes Always Enclose Their Point

This is used in the `.const` case, where a constant node certifies a point box
`[v,v]` and the semantics returns exactly the same `v`.
-/

private theorem encloses_point_self_real {n : Nat} (x : Tensor ℝ (.dim n .scalar)) :
    NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := x, hi := x } x :=
      by
  cases x with
  | dim fx =>
      rw [NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses]
      intro i
      cases h : fx i with
      | scalar v =>
          simp [NN.MLTheory.CROWN.Graph.getDimScalarFn, h]

/-!
### Sigmoid monotonicity (via Mathlib’s `Real.sigmoid`)

Our `Activation.Math.sigmoid_spec` definition over `ℝ` is exactly the Mathlib sigmoid function.
So we can reuse its monotonicity lemma.
-/

private theorem sigmoid_mono_real : Monotone (Activation.Math.sigmoidSpec (α := ℝ)) := by
  intro a b hab
  -- rewrite to `Real.sigmoid` and apply `Real.sigmoid_monotone`
  -- `Real.sigmoid x = (1 + exp (-x))⁻¹` and our definition is `1 / (1 + exp (-x))`.
  simpa [Activation.Math.sigmoidSpec, Real.sigmoid, div_eq_mul_inv] using Real.sigmoid_monotone hab

/-!
### Tanh monotonicity (proved from calculus in Mathlib)

Mathlib does not expose a `Real.tanh_monotone` lemma under that name. We prove it here:

1. Use the identity `tanh x = sinh x / cosh x`.
2. Differentiate the quotient, using `d/dx sinh = cosh` and `d/dx cosh = sinh`.
3. Simplify the derivative using `cosh^2 - sinh^2 = 1`.
4. Conclude strict monotonicity from `deriv > 0`, hence monotonicity.
-/

private theorem hasDerivAt_tanh_real (x : ℝ) :
    HasDerivAt Real.tanh (1 / (Real.cosh x) ^ 2) x := by
  -- Start from `sinh / cosh` and use the quotient rule.
  have hdiv :
      HasDerivAt (fun y : ℝ => Real.sinh y / Real.cosh y)
        ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / (Real.cosh x) ^ 2) x := by
    simpa [div_eq_mul_inv] using
      (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) (by exact (Real.cosh_pos x).ne')
  -- Transfer the derivative to `Real.tanh` using `tanh = sinh/cosh`.
  have ht : (fun y : ℝ => Real.tanh y) = fun y : ℝ => Real.sinh y / Real.cosh y := by
    funext y
    simp [Real.tanh_eq_sinh_div_cosh]
  have ht' :
      HasDerivAt (fun y : ℝ => Real.tanh y)
        ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) / (Real.cosh x) ^ 2) x := by
    simpa [ht] using hdiv
  -- Simplify `(cosh*cosh - sinh*sinh)` to `1`, yielding the stated derivative.
  have hId : Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x = 1 := by
    -- `cosh x ^ 2 - sinh x ^ 2 = 1` is in Mathlib.
    -- Rewrite products as squares.
    simpa [pow_two, mul_assoc, mul_left_comm, mul_comm] using (Real.cosh_sq_sub_sinh_sq x)
  -- Finish.
  simpa [hId, div_eq_mul_inv, one_div, pow_two, mul_assoc, mul_left_comm, mul_comm] using ht'

private theorem tanh_strictMono_real : StrictMono Real.tanh := by
  -- Use the standard calculus lemma: `deriv > 0` everywhere implies strict monotonicity.
  refine strictMono_of_deriv_pos ?_
  intro x
  have hderiv : deriv Real.tanh x = 1 / (Real.cosh x) ^ 2 :=
    (hasDerivAt_tanh_real x).deriv
  -- `cosh x > 0`, so `1/(cosh x)^2 > 0`.
  have hpos : 0 < (Real.cosh x) ^ 2 := by
    have : 0 < Real.cosh x := Real.cosh_pos x
    nlinarith
  -- Conclude.
  simpa [hderiv] using (one_div_pos.mpr hpos)

private theorem tanh_mono_real : Monotone Real.tanh :=
  tanh_strictMono_real.monotone

/-!
### Soundness of `Runtime.Ops.IBP.map_minmax` for monotone scalar functions

`Runtime.Ops.IBP.sigmoid` and `Runtime.Ops.IBP.tanh` are defined using `map_minmax`.
If the activation is monotone, then the min/max of the endpoints is a correct enclosure.
-/

private theorem map_minmax_sound_real {n : Nat} (f : ℝ → ℝ) (hf : Monotone f)
    (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax (α := ℝ) (n := n) f xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) f x) := by
  -- This is a pointwise proof over coordinates of the vector.
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                have hflfu : f l ≤ f u := hf hlu
                -- `map_minmax` chooses endpoint min/max by comparing `f l` and `f u`.
                -- With monotonicity we know `f l ≤ f u`, so lower is `f l` and upper is `f u`.
                have hlo : f l ≤ f v := hf hv.1
                have hhi : f v ≤ f u := hf hv.2
                have hnot : ¬ f u < f l := not_lt_of_ge hflfu
                -- Unfold `map_minmax` and reduce to the scalar goal:
                -- `lo ≤ f v ∧ f v ≤ hi`.
                simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax, Tensor.mapSpec,
                  hL, hU, hX, NN.MLTheory.CROWN.Box.contains, hnot] using And.intro hlo hhi

/-!
### Soundness of the 1-Lipschitz `sin`/`cos` enclosures

`Runtime.Ops.IBP.sin` / `Runtime.Ops.IBP.cos` use a midpoint enclosure with radius `r=(u-l)/2`,
clamped to `[-1,1]`. This avoids periodic case splits while remaining sound.
-/

private lemma sin_lipschitz_real (x y : ℝ) : |Real.sin x - Real.sin y| ≤ |x - y| := by
  have h := Real.sin_sub_sin x y
  calc
    |Real.sin x - Real.sin y|
        = |2 * Real.sin ((x - y) / 2) * Real.cos ((x + y) / 2)| := by
            simp [h, mul_left_comm, mul_comm]
    _ = 2 * |Real.sin ((x - y) / 2)| * |Real.cos ((x + y) / 2)| := by
          simp [abs_mul, mul_left_comm, mul_comm]
    _ ≤ 2 * |(x - y) / 2| * 1 := by
          have hsin : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have hcos : |Real.cos ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_cos_le_one ((x + y) / 2)
          -- Multiply the two bounds, keeping track of nonnegativity.
          have h2 : (2 : ℝ) * |Real.sin ((x - y) / 2)| ≤ 2 * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x - y) / 2)|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * 1 :=
            mul_le_mul_of_nonneg_left hcos (mul_nonneg (by norm_num) (abs_nonneg _))
          -- Reassociate back into `2 * |sin| * |cos|`.
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          -- `2 * |(x-y)/2| = |x-y|`.
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * |(x - y) / 2| * 1 = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm]
            _ = |x - y| := by nlinarith

private lemma cos_lipschitz_real (x y : ℝ) : |Real.cos x - Real.cos y| ≤ |x - y| := by
  have h := Real.cos_sub_cos x y
  calc
    |Real.cos x - Real.cos y|
        = |(-2) * Real.sin ((x + y) / 2) * Real.sin ((x - y) / 2)| := by
            simp [h, mul_assoc]
    _ = 2 * |Real.sin ((x + y) / 2)| * |Real.sin ((x - y) / 2)| := by
          simp [abs_mul, mul_assoc]
    _ ≤ 2 * 1 * |(x - y) / 2| := by
          have hsin1 : |Real.sin ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_sin_le_one ((x + y) / 2)
          have hsin2 : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have h2 : (2 : ℝ) * |Real.sin ((x + y) / 2)| ≤ 2 * 1 :=
            mul_le_mul_of_nonneg_left hsin1 (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x + y) / 2)|) * |Real.sin ((x - y) / 2)|
                ≤ (2 * 1) * |Real.sin ((x - y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * 1) * |Real.sin ((x - y) / 2)| ≤ (2 * 1) * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin2 (by norm_num)
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * 1 * |(x - y) / 2| = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm, mul_comm]
            _ = |x - y| := by nlinarith

private lemma ibp_sin_sound_real {n : Nat} (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n
  .scalar))
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := n) xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) Real.sin x) := by
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                let m : ℝ := (l + u) / 2
                let r : ℝ := (u - l) / 2
                have hxm : |v - m| ≤ r := by
                  have hlo : -r ≤ v - m := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  have hhi : v - m ≤ r := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  exact abs_le.2 ⟨hlo, hhi⟩
                have hLip : |Real.sin v - Real.sin m| ≤ r := by
                  exact le_trans (sin_lipschitz_real v m) hxm
                have hdiff : -r ≤ Real.sin v - Real.sin m ∧ Real.sin v - Real.sin m ≤ r :=
                  abs_le.1 hLip
                have hmidLo : Real.sin m - r ≤ Real.sin v := by linarith [hdiff.1]
                have hmidHi : Real.sin v ≤ Real.sin m + r := by linarith [hdiff.2]
                have hsinRange : (-1 : ℝ) ≤ Real.sin v ∧ Real.sin v ≤ (1 : ℝ) := by
                  have habs : |Real.sin v| ≤ (1 : ℝ) := by simpa using Real.abs_sin_le_one v
                  exact abs_le.1 habs
                have hlo : max (-1 : ℝ) (Real.sin m - r) ≤ Real.sin v :=
                  max_le_iff.2 ⟨hsinRange.1, hmidLo⟩
                have hhi : Real.sin v ≤ min (1 : ℝ) (Real.sin m + r) :=
                  le_min_iff.2 ⟨hsinRange.2, hmidHi⟩
                simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.sin, Tensor.mapSpec,
                  NN.MLTheory.CROWN.Box.contains,
                  hL, hU, hX, m, r] using And.intro hlo hhi

private lemma ibp_cos_sound_real {n : Nat} (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n
  .scalar))
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := n) xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) Real.cos x) := by
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                let m : ℝ := (l + u) / 2
                let r : ℝ := (u - l) / 2
                have hxm : |v - m| ≤ r := by
                  have hlo : -r ≤ v - m := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  have hhi : v - m ≤ r := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  exact abs_le.2 ⟨hlo, hhi⟩
                have hLip : |Real.cos v - Real.cos m| ≤ r := by
                  exact le_trans (cos_lipschitz_real v m) hxm
                have hdiff : -r ≤ Real.cos v - Real.cos m ∧ Real.cos v - Real.cos m ≤ r :=
                  abs_le.1 hLip
                have hmidLo : Real.cos m - r ≤ Real.cos v := by linarith [hdiff.1]
                have hmidHi : Real.cos v ≤ Real.cos m + r := by linarith [hdiff.2]
                have hcosRange : (-1 : ℝ) ≤ Real.cos v ∧ Real.cos v ≤ (1 : ℝ) := by
                  have habs : |Real.cos v| ≤ (1 : ℝ) := by simpa using Real.abs_cos_le_one v
                  exact abs_le.1 habs
                have hlo : max (-1 : ℝ) (Real.cos m - r) ≤ Real.cos v :=
                  max_le_iff.2 ⟨hcosRange.1, hmidLo⟩
                have hhi : Real.cos v ≤ min (1 : ℝ) (Real.cos m + r) :=
                  le_min_iff.2 ⟨hcosRange.2, hmidHi⟩
                simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.cos, Tensor.mapSpec,
                  NN.MLTheory.CROWN.Box.contains,
                  hL, hU, hX, m, r] using And.intro hlo hhi

/-!
## Main theorem: local IBP certificate implies semantic enclosure (supported subset)

We use strong induction on node id, assuming a topological order:
every parent id is strictly smaller than the node id.
-/

/-- Topological order assumption: all parent ids are strictly smaller than the node id. -/
def TopoSorted (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    ∀ p : Nat, p ∈ (g.nodes[id]!).parents → p < id

/-- A graph is supported by this soundness theorem if every node kind is in our supported subset. -/
def Supported (g : Graph) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    match (g.nodes[id]!).kind with
    | .input | .const _ | .detach
    | .add | .sub | .mul_elem | .relu
    | .linear | .matmul
    | .tanh | .sigmoid | .sin | .cos => True
    | _ => False

/-- Inputs are well-formed if every `.input` node has a value, and that value is enclosed by
its input box from `ParamStore.inputBoxes`. -/
def InputsEnclosed (g : Graph) (ps : ParamStore ℝ) (inputs : Std.HashMap Nat Val) : Prop :=
  ∀ id : Nat, id < g.nodes.size →
    (g.nodes[id]!).kind = .input →
      ∃ B v, ps.inputBoxes[id]? = some B ∧ inputs[id]? = some v ∧ EnclosesBox B v

/-!
### The enclosure theorem

Assumptions:
* `TopoSorted g`: induction works (parents are earlier).
* `Supported g`: every node kind is handled by the proof.
* `CertLocalOK g ps cert`: the certificate is locally consistent with the IBP step.
* `InputsEnclosed g ps inputs`: semantic inputs are inside the certified input boxes.
* `SemLocalOK g ps inputs vals`: `vals` is a locally-consistent semantic interpretation.

Conclusion:
* For every node `id`, if the semantics produces a value `v` and the certificate has a box `B`,
  then `B` encloses `v`.
-/
theorem cert_encloses_semantics
    (g : Graph) (ps : ParamStore ℝ)
    (cert : Array (Option (FlatBox ℝ)))
    (inputs : Std.HashMap Nat Val)
    (vals : Array (Option Val))
    (htopo : TopoSorted g)
    (hsupp : Supported g)
    (hcert : CertLocalOK (g := g) (ps := ps) cert)
    (hsem : SemLocalOK (g := g) (ps := ps) (inputs := inputs) vals)
    (hinputs : InputsEnclosed g ps inputs) :
    ∀ id : Nat, id < g.nodes.size →
      match cert[id]!, vals[id]! with
      | some B, some v => EnclosesBox B v
      | _, _ => True := by
  classical
  intro id hid
  -- Strong induction on `id` (parents are strictly smaller by `TopoSorted`).
  refine Nat.strong_induction_on id
      (p := fun k =>
        k < g.nodes.size →
          match cert[k]!, vals[k]! with
          | some B, some v => EnclosesBox B v
          | _, _ => True) ?_ hid
  intro k ih hk
  -- If certificate or value is missing, the goal is trivial.
  cases hcert with
  | intro hcertSz hnode =>
    cases hsem with
    | intro hvalsSz hsemNode =>
      have hcertk : cert[k]! = certStepNode? g.nodes ps cert k := hnode k hk
      have hvalk  : vals[k]! = evalNode? g.nodes ps inputs vals k := hsemNode k hk
      cases hck : cert[k]! <;> cases hvk : vals[k]! <;> simp
      case some.some B v =>
        -- From now on, we know both a certificate box and a semantic value exist.
        have hsupk := hsupp k hk
        have hcertStep : certStepNode? g.nodes ps cert k = some B := by
          have : (some B) = certStepNode? g.nodes ps cert k := by
            simpa [hck] using hcertk
          exact this.symm
        have hvalStep : evalNode? g.nodes ps inputs vals k = some v := by
          have : (some v) = evalNode? g.nodes ps inputs vals k := by
            simpa [hvk] using hvalk
          exact this.symm
        -- We'll use IH for parents.
        have parentIH :
            ∀ p : Nat, p ∈ (g.nodes[k]!).parents →
              match cert[p]!, vals[p]! with
              | some Bp, some vp => EnclosesBox Bp vp
              | _, _ => True := by
          intro p hp
          have hpk : p < k := htopo k hk p hp
          have hps : p < g.nodes.size := lt_trans hpk hk
          exact ih p hpk hps
        -- Now do a case split on the op kind.
        -- `Supported` lets us immediately discharge *unsupported* cases.
        cases hkKind : (g.nodes[k]!).kind <;>
          (simp [hkKind] at hsupk <;> try cases hsupk)
        case input =>
          rcases hinputs k hk hkKind with ⟨Bin, vin, hB, hv, hEnc⟩
          -- Use the *step equalities* to identify `B`/`v` with the input box/value.
          have hcertStepIn : certStepNode? g.nodes ps cert k = some Bin := by
            simp [certStepNode?, hkKind, hB]
          have hvalStepIn : evalNode? g.nodes ps inputs vals k = some vin := by
            simp [evalNode?, hkKind, hv]
          have hB_eq : B = Bin := by
            have : some B = some Bin := by simpa [hcertStep] using hcertStepIn
            cases this
            rfl
          have hv_eq : v = vin := by
            have : some v = some vin := by simpa [hvalStep] using hvalStepIn
            cases this
            rfl
          subst hB_eq
          subst hv_eq
          simpa using hEnc
        case const valueShape =>
          -- Both semantics and certificate read the same constant from `ps.constVals`.
          cases hconst : ps.constVals[k]? with
          | none =>
              -- If the parameter store has no constant here, the node cannot evaluate/certify.
              simp [certStepNode?, hkKind, hconst] at hcertStep
          | some val =>
              have hcertStep' := hcertStep
              have hvalStep' := hvalStep
              simp [certStepNode?, hkKind, hconst] at hcertStep'
              simp [evalNode?, hkKind, hconst] at hvalStep'
              cases hcertStep'
              cases hvalStep'
              -- Point box encloses its point (unfold enclosure and finish by simp).
              refine ⟨rfl, ?_⟩
              -- `castDimScalar rfl.symm` is definitional; use the point-box lemma.
              simpa [encloses, castDimScalar] using encloses_point_self_real (n := v.n) (x := v.v)
        case detach =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              -- With no parents, `certStepNode?` is `none`, contradicting `some B`.
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' : some B = getBox? cert p1 := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' : some v = getVal? vals p1 := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              -- Parent boxes/values must exist, otherwise the step would be `none`.
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      have hB_eq : B = B1 := by
                        have : some B = some B1 := by simpa [hgb] using hcertStep'
                        cases this
                        rfl
                      have hv_eq : v = v1 := by
                        have : some v = some v1 := by simpa [hgv] using hvalStep'
                        cases this
                        rfl
                      subst hB_eq
                      subst hv_eq
                      simpa using hpar'
        case add =>
          -- Extract parents.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              -- `certStepNode?` would be `none`, contradicting `cert[k] = some`.
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  -- From the certificate step: B = box_add Bp1 Bp2.
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 => some (box_add (α := ℝ) B1 B2)
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  -- From the semantics step: v = x + y (and dims match).
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.addSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  -- Extract concrete parent boxes/values (they must exist since this node produced
                  -- `some`).
                  cases hgb1 : getBox? cert p1 with
                  | none =>
                      simp [hgb1] at hcertStep'
                  | some B1 =>
                      cases hgb2 : getBox? cert p2 with
                      | none =>
                          simp [hgb1, hgb2] at hcertStep'
                      | some B2 =>
                          cases hgv1 : getVal? vals p1 with
                          | none =>
                              simp [hgv1] at hvalStep'
                          | some v1 =>
                              cases hgv2 : getVal? vals p2 with
                              | none =>
                                  simp [hgv1, hgv2] at hvalStep'
                              | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1' : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2' : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1' with ⟨hDim1, hxEnc⟩
                                  rcases hpar2' with ⟨hDim2, hyEnc⟩
                                  by_cases hxy : v1.n = v2.n
                                  ·
                                  -- Dimensions agree; prove enclosure using `box_add_sound` plus
                                  -- casts.
                                    have hBB : B1.dim = B2.dim :=
                                      Eq.trans hDim1 (Eq.trans hxy (Eq.symm hDim2))
                                    have hcertStep'' := hcertStep'
                                    simp [hgb1, hgb2] at hcertStep''
                                    cases hcertStep''
                                    have hvalStep'' := hvalStep'
                                    simp [hgv1, hgv2, hxy] at hvalStep''
                                    cases hvalStep''
                                    -- Make `box_add` reducible by destructing the boxes and using
                                    -- the equal-dimension branch.
                                    cases B1 with
                                    | mk n1 lo1 hi1 =>
                                      cases B2 with
                                      | mk n2 lo2 hi2 =>
                                        have hBB' : n1 = n2 := by
                                          simpa using hBB
                                        cases hBB'
                                        let x : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim1.symm v1.v
                                        let y : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim2.symm v2.v
                                        have hEncl :
                                            encloses
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              (Tensor.addSpec (α := ℝ) x y) :=
                                          NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_add_sound
                                            (α := ℝ) (n := n1)
                                            (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                            (add_mono := add_mono_real)
                                            (x := x) (y := y)
                                            (hx := by simpa [x] using hxEnc)
                                            (hy := by simpa [y] using hyEnc)
                                        have hBoxEq :
                                            box_add (α := ℝ)
                                                { dim := n1, lo := lo1, hi := hi1 }
                                                { dim := n1, lo := lo2, hi := hi2 }
                                              =
                                                { dim := n1
                                                  lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                  hi := Tensor.addSpec (α := ℝ) hi1 hi2 } := by
                                          simpa using
                                            (NN.MLTheory.CROWN.Graph.Theorems.box_add_on_eq (α := ℝ)
                                              n1 lo1 hi1 lo2 hi2)
                                        have hProof :
                                            Eq.trans hxy.symm hDim1.symm = hDim2.symm := by
                                          apply Subsingleton.elim
                                        have hValEq :
                                            castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.addSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                              = Tensor.addSpec (α := ℝ) x y := by
                                          have h1 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.addSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                                =
                                                  Tensor.addSpec (α := ℝ)
                                                    (castDimScalar (α := ℝ) hDim1.symm v1.v)
                                                    (castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) hxy.symm v2.v)) := by
                                            simpa using
                                              castDimScalar_add_spec (h := hDim1.symm) (x := v1.v)
                                                (y := castDimScalar (α := ℝ) hxy.symm v2.v)
                                          have h2 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (castDimScalar (α := ℝ) hxy.symm v2.v)
                                                = y := by
                                            have := (castDimScalar_trans (h₁ := hxy.symm) (h₂ :=
                                              hDim1.symm) (t := v2.v)).symm
                                            simpa [y, hProof] using this
                                          simpa [x, y, h2] using h1
                                        -- Prove enclosure for the canonical “equal-dimension”
                                        -- result box,
                                        -- then rewrite the goal box (`box_add …`) using `hBoxEq`.
                                        have hCanon :
                                            EnclosesBox
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              { n := v1.n
                                                v :=
                                                  Tensor.addSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) hxy.symm v2.v) } := by
                                          refine ⟨hDim1, ?_⟩
                                          have : encloses
                                              { dim := n1
                                                lo := Tensor.addSpec (α := ℝ) lo1 lo2
                                                hi := Tensor.addSpec (α := ℝ) hi1 hi2 }
                                              (Tensor.addSpec (α := ℝ) x y) := hEncl
                                          simpa [hValEq] using this
                                        simpa [hBoxEq] using hCanon
                                  ·
                                    simp [hgv1, hgv2, hxy] at hvalStep'

        case mul_elem =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 => box_mul_elem (α := ℝ) B1 B2
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.mulSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  cases hgb1 : getBox? cert p1 with
                  | none =>
                      simp [hgb1] at hcertStep'
                  | some B1 =>
                      cases hgb2 : getBox? cert p2 with
                      | none =>
                          simp [hgb1, hgb2] at hcertStep'
                      | some B2 =>
                          cases hgv1 : getVal? vals p1 with
                          | none =>
                              simp [hgv1] at hvalStep'
                          | some v1 =>
                              cases hgv2 : getVal? vals p2 with
                              | none =>
                                  simp [hgv1, hgv2] at hvalStep'
                              | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1 : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2 : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1 with ⟨hDim1, hx1⟩
                                  rcases hpar2 with ⟨hDim2, hx2⟩
                                  have hcertStep'' : some B = box_mul_elem (α := ℝ) B1 B2 := by
                                    simpa [hgb1, hgb2] using hcertStep'
                                  cases hmul : box_mul_elem (α := ℝ) B1 B2 with
                                  | none =>
                                      simp [hmul] at hcertStep''
                                  | some Bmul =>
                                      have hB_eq : B = Bmul := by
                                        have h := hcertStep''
                                        simp [hmul] at h
                                        cases h
                                        rfl
                                      subst B
                                      by_cases hxy : v1.n = v2.n
                                      · -- Unfold the value step under the `dims match` branch.
                                        have hvEq :
                                            v =
                                              ⟨v1.n,
                                                Tensor.mulSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)⟩ := by
                                          have : (some v : Option Val) =
                                              some
                                                ⟨v1.n,
                                                  Tensor.mulSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)⟩ :=
                                                      by
                                            simpa [hgv1, hgv2, hxy] using hvalStep'
                                          cases this
                                          rfl
                                        subst hvEq
                                        -- Reduce to tensors at dimension `Bmul.dim` and apply
                                        -- `box_mul_elem_sound_real`.
                                        cases B1 with
                                        | mk n1 lo1 hi1 =>
                                          cases B2 with
                                          | mk n2 lo2 hi2 =>
                                            -- `Bmul` coming from `box_mul_elem` forces equal
                                            -- dimensions.
                                            have hn12 : n1 = n2 := by
                                              by_contra hne
                                              have : box_mul_elem (α := ℝ)
                                                  { dim := n1, lo := lo1, hi := hi1 }
                                                  { dim := n2, lo := lo2, hi := hi2 } = none := by
                                                unfold box_mul_elem
                                                simp [hne]
                                              have : False := by
                                                simp [this] at hmul
                                              exact this.elim
                                            cases hn12
                                            let x : Tensor ℝ (.dim n1 .scalar) :=
                                              castDimScalar (α := ℝ) hDim1.symm v1.v
                                            let y : Tensor ℝ (.dim n1 .scalar) :=
                                              castDimScalar (α := ℝ) hDim2.symm v2.v
                                            have hx : encloses { dim := n1, lo := lo1, hi := hi1 } x
                                              := by
                                              simpa [x] using hx1
                                            have hy : encloses { dim := n1, lo := lo2, hi := hi2 } y
                                              := by
                                              simpa [y] using hx2
                                            have hMulEncl :
                                                EnclosesBox Bmul ⟨n1, Tensor.mulSpec (α := ℝ) x y⟩
                                                  :=
                                              box_mul_elem_sound_real (n := n1)
                                                (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                                  (x := x) (y := y) hx hy
                                                (B := Bmul) (by simpa using hmul)
                                            rcases hMulEncl with ⟨hDimMul, hEncMul⟩
                                            -- We need enclosure for the semantic value `v`, whose
                                            -- dimension is `v1.n`.
                                            -- Use `hDimMul : Bmul.dim = n1` and `hDim1 : n1 =
                                            -- v1.n`.
                                            let hW : Bmul.dim = v1.n := Eq.trans hDimMul hDim1
                                            refine ⟨hW, ?_⟩
                                            -- Rewrite the value cast into the `x*y` cast used by
                                            -- `hEncMul`.
                                            have hvCast0 :
                                                castDimScalar (α := ℝ) hDim1.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                  =
                                                  Tensor.mulSpec (α := ℝ) x y := by
                                              -- Commute the cast with multiplication and rewrite
                                              -- the second input cast into `y`.
                                              have hyCast :
                                                  castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)
                                                    = y := by
                                                -- Both sides are casts of `v2.v` to dimension `n1`.
                                                have hEq :
                                                    Eq.trans (Eq.symm hxy) hDim1.symm = hDim2.symm
                                                      := by
                                                  apply Subsingleton.elim
                                                -- Reassociate casts and rewrite.
                                                have hnest :
                                                    castDimScalar (α := ℝ) hDim1.symm
                                                        (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v)
                                                      =
                                                      castDimScalar (α := ℝ) (Eq.trans (Eq.symm hxy)
                                                        hDim1.symm) v2.v := by
                                                  simpa using
                                                    (castDimScalar_trans (h₁ := (Eq.symm hxy)) (h₂
                                                      := hDim1.symm) (t := v2.v)).symm
                                                simpa [y, hEq] using hnest
                                              -- Now simplify using `x` and `hyCast`.
                                              simp [x, castDimScalar_mul_spec, hyCast]
                                            have hvCast :
                                                castDimScalar (α := ℝ) hW.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                  =
                                                  castDimScalar (α := ℝ) hDimMul.symm
                                                    (Tensor.mulSpec (α := ℝ) x y) := by
                                              -- `hW.symm` and `hDim1.symm.trans hDimMul.symm` are
                                              -- both proofs of `v1.n = Bmul.dim`.
                                              have hts : hW.symm = Eq.trans hDim1.symm hDimMul.symm
                                                := by
                                                apply Subsingleton.elim
                                              -- Expand `hW.symm` into a composite cast, then use
                                              -- `hvCast0`.
                                              calc
                                                castDimScalar (α := ℝ) hW.symm
                                                    (Tensor.mulSpec (α := ℝ) v1.v
                                                      (castDimScalar (α := ℝ) (Eq.symm hxy) v2.v))
                                                    =
                                                    castDimScalar (α := ℝ) hDimMul.symm
                                                      (castDimScalar (α := ℝ) hDim1.symm
                                                        (Tensor.mulSpec (α := ℝ) v1.v
                                                          (castDimScalar (α := ℝ) (Eq.symm hxy)
                                                            v2.v))) := by
                                                      -- Reassociate casts along `hDim1.symm` then
                                                      -- `hDimMul.symm`.
                                                      simpa [hts] using
                                                        (castDimScalar_trans (h₁ := hDim1.symm) (h₂
                                                          := hDimMul.symm)
                                                          (t := Tensor.mulSpec (α := ℝ) v1.v
                                                            (castDimScalar (α := ℝ) (Eq.symm hxy)
                                                              v2.v)))
                                                _ = castDimScalar (α := ℝ) hDimMul.symm
                                                  (Tensor.mulSpec (α := ℝ) x y) := by
                                                      simpa using
                                                        congrArg (fun t => castDimScalar (α := ℝ)
                                                          hDimMul.symm t) hvCast0
                                            -- `hEncMul` is already phrased using `hDimMul`.
                                            simpa [hvCast] using hEncMul
                                      ·
                                        simp [hgv1, hgv2, hxy] at hvalStep'

        case sub =>
          -- Similar to `.add`, using `Theorems.Semantics.box_sub_sound`.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 rest =>
              cases rest with
              | nil =>
                  simp [certStepNode?, hkKind, hparents] at hcertStep
              | cons p2 _ =>
                  have hp1c : p1 < cert.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp2c : p2 < cert.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hcertSz] using this
                  have hp1v : p1 < vals.size := by
                    have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hp2v : p2 < vals.size := by
                    have : p2 < g.nodes.size := lt_trans (htopo k hk p2 (by simp [hparents])) hk
                    simpa [hvalsSz] using this
                  have hcertStep' :
                      some B =
                        match getBox? cert p1, getBox? cert p2 with
                        | some B1, some B2 => some (box_sub (α := ℝ) B1 B2)
                        | _, _ => none := by
                    simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
                  have hvalStep' :
                      some v =
                        match getVal? vals p1, getVal? vals p2 with
                        | some x, some y =>
                            if h : x.n = y.n then
                              let yv : Tensor ℝ (.dim x.n .scalar) :=
                                castDimScalar (α := ℝ) (Eq.symm h) y.v
                              some { n := x.n, v := Tensor.subSpec (α := ℝ) x.v yv }
                            else none
                        | _, _ => none := by
                    simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
                  cases hgb1 : getBox? cert p1 with
                    | none =>
                        simp [hgb1] at hcertStep'
                    | some B1 =>
                        cases hgb2 : getBox? cert p2 with
                        | none =>
                            simp [hgb1, hgb2] at hcertStep'
                        | some B2 =>
                            cases hgv1 : getVal? vals p1 with
                            | none =>
                                simp [hgv1] at hvalStep'
                            | some v1 =>
                                cases hgv2 : getVal? vals p2 with
                                | none =>
                                    simp [hgv1, hgv2] at hvalStep'
                                | some v2 =>
                                  have hB1 : cert[p1]! = some B1 := by
                                    simpa [getBox?, hp1c] using hgb1
                                  have hB2 : cert[p2]! = some B2 := by
                                    simpa [getBox?, hp2c] using hgb2
                                  have hv1 : vals[p1]! = some v1 := by
                                    simpa [getVal?, hp1v] using hgv1
                                  have hv2 : vals[p2]! = some v2 := by
                                    simpa [getVal?, hp2v] using hgv2
                                  have hpar1' : EnclosesBox B1 v1 := by
                                    have h := parentIH p1 (by simp [hparents])
                                    simpa [hB1, hv1] using h
                                  have hpar2' : EnclosesBox B2 v2 := by
                                    have h := parentIH p2 (by simp [hparents])
                                    simpa [hB2, hv2] using h
                                  rcases hpar1' with ⟨hDim1, hxEnc⟩
                                  rcases hpar2' with ⟨hDim2, hyEnc⟩
                                  by_cases hxy : v1.n = v2.n
                                  ·
                                    have hBB : B1.dim = B2.dim :=
                                      Eq.trans hDim1 (Eq.trans hxy (Eq.symm hDim2))
                                    have hcertStep'' := hcertStep'
                                    simp [hgb1, hgb2] at hcertStep''
                                    cases hcertStep''
                                    have hvalStep'' := hvalStep'
                                    simp [hgv1, hgv2, hxy] at hvalStep''
                                    cases hvalStep''
                                    cases B1 with
                                    | mk n1 lo1 hi1 =>
                                      cases B2 with
                                      | mk n2 lo2 hi2 =>
                                        have hBB' : n1 = n2 := by
                                          simpa using hBB
                                        cases hBB'
                                        let x : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim1.symm v1.v
                                        let y : Tensor ℝ (.dim n1 .scalar) :=
                                          castDimScalar (α := ℝ) hDim2.symm v2.v
                                        have hEncl :
                                            encloses
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              (Tensor.subSpec (α := ℝ) x y) :=
                                          NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_sub_sound
                                            (α := ℝ) (n := n1)
                                            (lo1 := lo1) (hi1 := hi1) (lo2 := lo2) (hi2 := hi2)
                                            (sub_mono := sub_mono_real)
                                            (x := x) (y := y)
                                            (hx := by simpa [x] using hxEnc)
                                            (hy := by simpa [y] using hyEnc)
                                        have hBoxEq :
                                            box_sub (α := ℝ)
                                                { dim := n1, lo := lo1, hi := hi1 }
                                                { dim := n1, lo := lo2, hi := hi2 }
                                              =
                                                { dim := n1
                                                  lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                  hi := Tensor.subSpec (α := ℝ) hi1 lo2 } := by
                                          simpa using
                                            (NN.MLTheory.CROWN.Graph.Theorems.box_sub_on_eq (α := ℝ)
                                              n1 lo1 hi1 lo2 hi2)
                                        have hProof :
                                            Eq.trans hxy.symm hDim1.symm = hDim2.symm := by
                                          apply Subsingleton.elim
                                        have hValEq :
                                            castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.subSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                              = Tensor.subSpec (α := ℝ) x y := by
                                          have h1 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (Tensor.subSpec (α := ℝ) v1.v
                                                  (castDimScalar (α := ℝ) hxy.symm v2.v))
                                                =
                                                  Tensor.subSpec (α := ℝ)
                                                    (castDimScalar (α := ℝ) hDim1.symm v1.v)
                                                    (castDimScalar (α := ℝ) hDim1.symm
                                                      (castDimScalar (α := ℝ) hxy.symm v2.v)) := by
                                            simpa using
                                              castDimScalar_sub_spec (h := hDim1.symm) (x := v1.v)
                                                (y := castDimScalar (α := ℝ) hxy.symm v2.v)
                                          have h2 :
                                              castDimScalar (α := ℝ) hDim1.symm
                                                (castDimScalar (α := ℝ) hxy.symm v2.v)
                                                = y := by
                                            have := (castDimScalar_trans (h₁ := hxy.symm) (h₂ :=
                                              hDim1.symm) (t := v2.v)).symm
                                            simpa [y, hProof] using this
                                          simpa [x, y, h2] using h1
                                        have hCanon :
                                            EnclosesBox
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              { n := v1.n
                                                v :=
                                                  Tensor.subSpec (α := ℝ) v1.v
                                                    (castDimScalar (α := ℝ) hxy.symm v2.v) } := by
                                          refine ⟨hDim1, ?_⟩
                                          have : encloses
                                              { dim := n1
                                                lo := Tensor.subSpec (α := ℝ) lo1 hi2
                                                hi := Tensor.subSpec (α := ℝ) hi1 lo2 }
                                              (Tensor.subSpec (α := ℝ) x y) := hEncl
                                          simpa [hValEq] using this
                                        simpa [hBoxEq] using hCanon
                                  ·
                                    simp [hgv1, hgv2, hxy] at hvalStep'

        case relu =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some B1 => some (box_relu (α := ℝ) B1)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.reluSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      have hB_eq : B = box_relu (α := ℝ) B1 := by
                        have : some B = some (box_relu (α := ℝ) B1) := by
                          simpa [hgb] using hcertStep'
                        cases this
                        rfl
                      have hv_eq : v = { n := v1.n, v := Activation.reluSpec (α := ℝ) v1.v } := by
                        have : some v = some { n := v1.n, v := Activation.reluSpec (α := ℝ) v1.v }
                          := by
                          simpa [hgv] using hvalStep'
                        cases this
                        rfl
                      subst hB_eq
                      subst hv_eq
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hDimOut : (box_relu (α := ℝ) B1).dim = v1.n := by
                        exact Eq.trans
                          (NN.MLTheory.CROWN.Graph.Theorems.box_relu_dim (α := ℝ) B1)
                          hDim
                      refine ⟨hDimOut, ?_⟩
                      -- Use the existing `box_relu_sound` lemma at dimension `B1.dim`,
                      -- and then rewrite the semantic value (a cast of `relu_spec v1.v`)
                      -- into `relu_spec` of the casted input.
                      have hrelu :=
                        (NN.MLTheory.CROWN.Graph.Theorems.Semantics.box_relu_sound (α := ℝ)
                          (n := B1.dim) (lo := B1.lo) (hi := B1.hi)
                          (relu_mono := relu_mono_real)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                          (hx := by
                            simpa [castDimScalar] using hxEnc))
                      have hcastProof : hDimOut.symm = hDim.symm := by
                        -- Both sides are proofs of `v1.n = B1.dim`.
                        apply Subsingleton.elim
                      have hvCast :
                          castDimScalar (α := ℝ) hDimOut.symm (Activation.reluSpec (α := ℝ) v1.v)
                            = Activation.reluSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v)
                              := by
                        -- Commute the cast with `map_spec` (ReLU is elementwise).
                        simpa [hcastProof, Activation.reluSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.reluSpec (α := ℝ)) (t := v1.v))
                      -- `box_relu_sound` already inserts a cast to match `(box_relu ...).dim`;
                      -- after rewriting the value cast, it is exactly the enclosure we need.
                      simpa [EnclosesBox, hvCast] using hrelu
        case tanh =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.tanhSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := B1.dim)
                              (ofFlatBox (α := ℝ) B1))
                            (Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v))
                              := by
                        simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh, Activation.tanhSpec,
                          Activation.Math.tanhSpec,
                          MathFunctions.tanh]
                          using map_minmax_sound_real (n := B1.dim) (f := Real.tanh) tanh_mono_real
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm (Activation.tanhSpec (α := ℝ) v1.v)
                            = Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm v1.v)
                              := by
                        simpa [Activation.tanhSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.tanhSpec (α := ℝ)) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.tanh (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Activation.tanhSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                            v1.v))
                          houtContains
        case sigmoid =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x => some { n := x.n, v := Activation.sigmoidSpec (α := ℝ) x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := B1.dim)
                              (ofFlatBox (α := ℝ) B1))
                            (Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                              v1.v)) := by
                        have hs : Monotone (Activation.Math.sigmoidSpec (α := ℝ)) :=
                          sigmoid_mono_real
                        simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid, Activation.sigmoidSpec,
                          Tensor.mapSpec]
                          using map_minmax_sound_real (n := B1.dim) (f :=
                            Activation.Math.sigmoidSpec (α := ℝ)) hs
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm (Activation.sigmoidSpec (α := ℝ) v1.v)
                            = Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                              v1.v) := by
                        -- Commute cast with `map_spec` (sigmoid is elementwise).
                        simpa [Activation.sigmoidSpec] using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := Activation.Math.sigmoidSpec (α := ℝ)) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.sigmoid (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Activation.sigmoidSpec (α := ℝ) (castDimScalar (α := ℝ) hDim.symm
                            v1.v))
                          houtContains
        case sin =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x =>
                        some
                          { n := x.n
                            v :=
                              Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.sin z)
                                x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := B1.dim) (ofFlatBox
                              (α := ℝ) B1))
                            (Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.sin
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                        have h' :=
                          ibp_sin_sound_real (n := B1.dim)
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                        simpa using h'
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm
                              (Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.sin
                                z) v1.v)
                            =
                            Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.sin
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                        -- Commute cast with `map_spec` (sin is elementwise).
                        simpa using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := fun z : ℝ => Real.sin z) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) Real.sin
                            (castDimScalar (α := ℝ) hDim.symm v1.v))
                          (by simpa using houtContains)
        case cos =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin =>
                        let yB := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := Xin.dim)
                          (ofFlatBox (α := ℝ) Xin)
                        some (toFlatBox (α := ℝ) Xin.dim yB)
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1 with
                    | some x =>
                        some
                          { n := x.n
                            v :=
                              Tensor.mapSpec (α := ℝ) (s := .dim x.n .scalar) (fun z => Real.cos z)
                                x.v }
                    | none => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      have hcertStep'' := hcertStep'
                      have hvalStep'' := hvalStep'
                      simp [hgb] at hcertStep''
                      simp [hgv] at hvalStep''
                      cases hcertStep''
                      cases hvalStep''
                      have hxContains :
                          Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                            (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                        contains_of_encloses (B := B1)
                          (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using hxEnc)
                      have houtContains :
                          Box.contains (α := ℝ)
                            (NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := B1.dim) (ofFlatBox
                              (α := ℝ) B1))
                            (Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.cos
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                        have h' :=
                          ibp_cos_sound_real (n := B1.dim)
                            (xB := ofFlatBox (α := ℝ) B1)
                            (x := castDimScalar (α := ℝ) hDim.symm v1.v)
                            hxContains
                        simpa using h'
                      have hvCast :
                          castDimScalar (α := ℝ) hDim.symm
                              (Tensor.mapSpec (α := ℝ) (s := .dim v1.n .scalar) (fun z => Real.cos
                                z) v1.v)
                            =
                            Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) (fun z => Real.cos
                              z)
                              (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                        simpa using
                          (castDimScalar_map_spec (h := hDim.symm)
                            (f := fun z : ℝ => Real.cos z) (t := v1.v))
                      refine ⟨hDim, ?_⟩
                      simpa [hvCast.symm] using
                        encloses_of_contains (n := B1.dim)
                          (B := NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := B1.dim)
                            (ofFlatBox (α := ℝ) B1))
                          (x := Tensor.mapSpec (α := ℝ) (s := .dim B1.dim .scalar) Real.cos
                            (castDimScalar (α := ℝ) hDim.symm v1.v))
                          (by simpa using houtContains)
        case linear =>
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              -- Certificate step delegates to `ibp_linear`.
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin => ibp_linear (α := ℝ) k ps Xin
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              -- Semantics step uses the linear spec from ParamStore.
              have hvalStep' :
                  some v =
                    match getVal? vals p1, ps.linearWB[k]? with
                    | some x, some p =>
                        if h : x.n = p.n then
                          let xv : Tensor ℝ (.dim p.n .scalar) := by simpa [h] using x.v
                          let yv : Tensor ℝ (.dim p.m .scalar) :=
                            Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b } xv
                          some { n := p.m, v := yv }
                        else none
                    | _, _ => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      -- Unfold `ibp_linear` and use the already-proved `IBP.linear` soundness
                      -- theorem.
                      unfold ibp_linear at hcertStep'
                      cases hlin : ps.linearWB[k]? with
                      | none =>
                          have : (none : Option (FlatBox ℝ)) = some B := by
                            simpa [hlin, hgb] using (Eq.symm hcertStep')
                          cases this
                      | some p =>
                          simp [hlin, hgb] at hcertStep'
                          -- Substitute the concrete parent value into the semantic step.
                          simp [hlin, hgv] at hvalStep'
                          by_cases hXin : B1.dim = p.n
                          · simp [hXin] at hcertStep'
                            by_cases hxDim' : v1.n = p.n
                            · simp [hxDim'] at hvalStep'
                              cases hcertStep'
                              cases hvalStep'
                              -- Convert parent enclosure to `Box.contains` for the casted input
                              -- box.
                              have hxContains0 :
                                  Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                                    (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                                contains_of_encloses (B := B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using
                                    hxEnc)
                              have hxContains1 :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                                -- Transport `contains` across the dimension cast on the box.
                                exact (contains_castBoxDim_iff (h := hXin) (B := ofFlatBox (α := ℝ)
                                  B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v)).2 hxContains0
                              have hProof : Eq.trans hDim.symm hXin = hxDim' := by
                                apply Subsingleton.elim
                              have hxCastEq :
                                  castDimScalar (α := ℝ) hxDim' v1.v
                                    = castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                                -- `hxDim'` and `Eq.trans hDim.symm hXin` are the same equality, up
                                -- to proof irrelevance.
                                simpa [hProof] using
                                  (castDimScalar_trans (h₁ := hDim.symm) (h₂ := hXin) (t := v1.v))
                              have hxContains :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hxDim' v1.v) := by
                                simpa [hxCastEq] using hxContains1
                              have hb : Box.contains (α := ℝ) (Box.point (α := ℝ) p.b) p.b := by
                                cases p.b with
                                | dim fb =>
                                  intro i
                                  cases fb i with
                                  | scalar b =>
                                    simp [Box.contains]
                              have houtContains :
                                  Box.contains (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
                                      (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                      (Box.point (α := ℝ) p.b))
                                    (Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                      (castDimScalar (α := ℝ) hxDim' v1.v)) :=
                                NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                  (Box.point (α := ℝ) p.b)
                                  (castDimScalar (α := ℝ) hxDim' v1.v)
                                  p.b
                                  hxContains hb
                              refine ⟨rfl, ?_⟩
                              exact encloses_of_contains (n := p.m)
                                (B := NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n)
                                  p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1)) (Box.point (α
                                    := ℝ) p.b))
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := p.b }
                                  (castDimScalar (α := ℝ) hxDim' v1.v))
                                houtContains
                            ·
                              simp [hxDim'] at hvalStep'
                          ·
                            simp [hXin] at hcertStep'
        case matmul =>
          -- Same as `.linear`, but bias is zero and params come from `ParamStore.matmulW`.
          cases hparents : (g.nodes[k]!).parents with
          | nil =>
              simp [certStepNode?, hkKind, hparents] at hcertStep
          | cons p1 _ =>
              have hp1c : p1 < cert.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hcertSz] using this
              have hp1v : p1 < vals.size := by
                have : p1 < g.nodes.size := lt_trans (htopo k hk p1 (by simp [hparents])) hk
                simpa [hvalsSz] using this
              have hcertStep' :
                  some B =
                    match getBox? cert p1 with
                    | some Xin => ibp_matmul (α := ℝ) k ps Xin
                    | none => none := by
                simpa [certStepNode?, hkKind, hparents] using (Eq.symm hcertStep)
              have hvalStep' :
                  some v =
                    match getVal? vals p1, ps.matmulW[k]? with
                    | some x, some p =>
                        if h : x.n = p.n then
                          let xv : Tensor ℝ (.dim p.n .scalar) := by simpa [h] using x.v
                          let z : Tensor ℝ (.dim p.m .scalar) := Spec.fill (α := ℝ) 0 (.dim p.m
                            .scalar)
                          let yv : Tensor ℝ (.dim p.m .scalar) :=
                            Spec.linearSpec (α := ℝ) { weights := p.w, bias := z } xv
                          some { n := p.m, v := yv }
                        else none
                    | _, _ => none := by
                simpa [evalNode?, hkKind, hparents] using (Eq.symm hvalStep)
              cases hgb : getBox? cert p1 with
              | none =>
                  simp [hgb] at hcertStep'
              | some B1 =>
                  cases hgv : getVal? vals p1 with
                  | none =>
                      simp [hgv] at hvalStep'
                  | some v1 =>
                      have hB1 : cert[p1]! = some B1 := by
                        simpa [getBox?, hp1c] using hgb
                      have hv1 : vals[p1]! = some v1 := by
                        simpa [getVal?, hp1v] using hgv
                      have hpar' : EnclosesBox B1 v1 := by
                        have h := parentIH p1 (by simp [hparents])
                        simpa [hB1, hv1] using h
                      rcases hpar' with ⟨hDim, hxEnc⟩
                      unfold ibp_matmul at hcertStep'
                      cases hmat : ps.matmulW[k]? with
                      | none =>
                          have : (none : Option (FlatBox ℝ)) = some B := by
                            simpa [hmat, hgb] using (Eq.symm hcertStep')
                          cases this
                      | some p =>
                          simp [hmat, hgb] at hcertStep'
                          simp [hmat, hgv] at hvalStep'
                          by_cases hXin : B1.dim = p.n
                          · simp [hXin] at hcertStep'
                            by_cases hxDim' : v1.n = p.n
                            · simp [hxDim'] at hvalStep'
                              cases hcertStep'
                              cases hvalStep'
                              have hxContains0 :
                                  Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B1)
                                    (castDimScalar (α := ℝ) hDim.symm v1.v) :=
                                contains_of_encloses (B := B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v) (by simpa using
                                    hxEnc)
                              have hxContains1 :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v)) := by
                                exact (contains_castBoxDim_iff (h := hXin) (B := ofFlatBox (α := ℝ)
                                  B1)
                                  (x := castDimScalar (α := ℝ) hDim.symm v1.v)).2 hxContains0
                              have hProof : Eq.trans hDim.symm hXin = hxDim' := by
                                apply Subsingleton.elim
                              have hxCastEq :
                                  castDimScalar (α := ℝ) hxDim' v1.v
                                    = castDimScalar (α := ℝ) hXin
                                      (castDimScalar (α := ℝ) hDim.symm v1.v) := by
                                simpa [hProof] using
                                  (castDimScalar_trans (h₁ := hDim.symm) (h₂ := hXin) (t := v1.v))
                              have hxContains :
                                  Box.contains (α := ℝ)
                                    (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                    (castDimScalar (α := ℝ) hxDim' v1.v) := by
                                simpa [hxCastEq] using hxContains1
                              let z : Tensor ℝ (.dim p.m .scalar) :=
                                Spec.fill (α := ℝ) 0 (.dim p.m .scalar)
                              have hz : Box.contains (α := ℝ) (Box.point (α := ℝ) z) z := by
                                cases z with
                                | dim fb =>
                                  intro i
                                  cases fb i with
                                  | scalar b =>
                                    simp [Box.contains]
                              have houtContains :
                                  Box.contains (α := ℝ)
                                    (NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n) p.w
                                      (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                      (Box.point (α := ℝ) z))
                                    (Spec.linearSpec (α := ℝ) { weights := p.w, bias := z }
                                      (castDimScalar (α := ℝ) hxDim' v1.v)) :=
                                NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1))
                                  (Box.point (α := ℝ) z)
                                  (castDimScalar (α := ℝ) hxDim' v1.v)
                                  z
                                  hxContains hz
                              refine ⟨rfl, encloses_of_contains (n := p.m)
                                (B := NN.MLTheory.CROWN.IBP.linear (α := ℝ) (m := p.m) (n := p.n)
                                  p.w
                                  (castBoxDim (α := ℝ) hXin (ofFlatBox (α := ℝ) B1)) (Box.point (α
                                    := ℝ) z))
                                (x := Spec.linearSpec (α := ℝ) { weights := p.w, bias := z }
                                  (castDimScalar (α := ℝ) hxDim' v1.v))
                                houtContains⟩
                            ·
                              simp [hxDim'] at hvalStep'
                          ·
                            simp [hXin] at hcertStep'

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
