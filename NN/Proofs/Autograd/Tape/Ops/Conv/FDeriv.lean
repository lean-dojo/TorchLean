/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Nodes.GraphComposition
public import NN.Proofs.Autograd.Tape.Ops.Conv.BackwardDot

public import Mathlib.Analysis.Calculus.FDeriv.Bilinear
public import Mathlib.Data.Fintype.BigOperators

/-!
# FDeriv

Analytic (`HasFDerivAt`/`fderiv`) correctness for a **Conv2D**-shaped bilinear map.

This file adds a proof-only node constructor that treats convolution as a bilinear map
in `(kernel, input)` plus a linear bias broadcast, and defines its VJP as the adjoint of
the Fréchet derivative. This is enough for the general DAG theorem
`Graph.backpropVec_eq_adjoint_fderiv` to cover graphs using this node.

Note: the runtime engine uses `Spec.conv2d_backward_spec`; connecting that
handwritten backward to this analytic VJP is a separate theorem.
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace TapeNodes

open Spec
open Tensor

open scoped BigOperators

noncomputable section

namespace Conv2D

  set_option maxHeartbeats 5000000

-- We use the standard flattening order induced by `finProdFinEquiv`, with casts for reassociation.

/-- Flattened index for a 3D `(C,H,W)` tensor in row-major order. -/
private def idx3 {C H W : Nat} (c : Fin C) (i : Fin H) (j : Fin W) : Fin (C * (H * W)) :=
  finProdFinEquiv (c, finProdFinEquiv (i, j))

/-- Flattened index for a 4D `(OC,IC,KH,KW)` tensor in row-major order. -/
private def idx4 {OC IC KH KW : Nat} (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    Fin (OC * (IC * (KH * KW))) :=
  finProdFinEquiv (oc, finProdFinEquiv (ic, finProdFinEquiv (di, dj)))

@[simp] private lemma castCLM_apply {n m : Nat} (h : n = m) (v : Vec n) :
    (Graph.castCLM (h := h) : Vec n →L[ℝ] Vec m) v = castVec h v := by
  classical
  -- Unfold once: the underlying linear map is exactly `castVec h`.
  simp [Graph.castCLM]

/-- `Shape.size (.dim n .scalar)` (kept as a name for casts). -/
private abbrev vecSize (n : Nat) : Nat :=
  Shape.size (.dim n .scalar)

-- Indices for `toVecT`/`ofVecT` on 3D/4D shapes.
/-- Flattened index for `toVecT`/`ofVecT` on shape `.dim C (.dim H (.dim W .scalar))`. -/
private def idx3S {C H W : Nat} (c : Fin C) (i : Fin H) (j : Fin W) :
    Fin (Shape.size (.dim C (.dim H (.dim W .scalar)))) :=
  Fin.cast (by simp [Shape.size]) (idx3 (C := C) (H := H) (W := W) c i j)

/-- Flattened index for `toVecT`/`ofVecT` on shape `.dim OC (.dim IC (.dim KH (.dim KW .scalar)))`.
  -/
private def idx4S {OC IC KH KW : Nat} (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    Fin (Shape.size (.dim OC (.dim IC (.dim KH (.dim KW .scalar))))) :=
  Fin.cast (by simp [Shape.size]) (idx4 (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj)

/-- `toVecT` on scalar tensors always returns the scalar value (the only coordinate is `0`). -/
private lemma toVecT_scalar_apply (x : ℝ) (i : Fin (Shape.size Shape.scalar)) :
    toVecT (t := (Tensor.scalar x : Tensor ℝ Shape.scalar)) i = x := by
  simpa [toVecT, toVecE, Spec.Tensor.flattenSpec, Shape.size, Spec.toVec] using
    (euclideanEquiv_symm_ofLp
      (n := Shape.size Shape.scalar)
      (f := fun _ : Fin (Shape.size Shape.scalar) => x)
      (i := i))

/-- Relate 1D tensor `toVecT` coordinates to `get_at_or_zero` via the scalar last-axis encoding. -/
private lemma toVecT_get1
    {C : Nat} (A : Tensor ℝ (.dim C .scalar)) (c : Fin C) :
    toVecT (t := A) (Fin.cast (by simp [Shape.size]) c) =
      getAtOrZero A [c.val] := by
  classical
  cases C with
  | zero => exact (Fin.elim0 c)
  | succ C =>
      cases A with
      | dim f =>
          let k0 : Fin 1 := 0
          have hmposSc : 0 < Shape.size Shape.scalar := by
            simp [Shape.size]
          have hinner :
              toVecT (t := (Tensor.dim f : Tensor ℝ (.dim (Nat.succ C) .scalar)))
                  (finProdFinEquiv (c, k0))
                =
              toVecT (t := f c) k0 := by
            simpa using
              (toVecT_dim_apply (n := Nat.succ C) (s := Shape.scalar)
                (hmpos := hmposSc) (f := f) (p := (c, k0)))
          have hidx :
              Fin.cast (by simp) c = finProdFinEquiv (c, k0) := by
            apply Fin.ext
            simp [finProdFinEquiv, k0]
          cases hcell : f c with
          | scalar x =>
              have hsc : toVecT (t := (Tensor.scalar x : Tensor ℝ Shape.scalar)) k0 = x :=
                toVecT_scalar_apply (x := x) (i := k0)
              have hget :
                  getAtOrZero (Tensor.dim f) [c.val] = x := by
                simp [hcell, c.isLt]
              calc
                toVecT (t := (Tensor.dim f : Tensor ℝ (.dim (Nat.succ C) .scalar)))
                    (Fin.cast (by simp [Shape.size]) c)
                    =
                  toVecT (t := (Tensor.dim f : Tensor ℝ (.dim (Nat.succ C) .scalar)))
                    (finProdFinEquiv (c, k0)) := by
                      exact congrArg
                        (fun z => toVecT (t := (Tensor.dim f : Tensor ℝ (.dim (Nat.succ C)
                          .scalar))) z)
                        hidx
                _ = toVecT (t := f c) k0 := hinner
                _ = x := by simpa [hcell] using hsc
                _ = getAtOrZero (Tensor.dim f) [c.val] := by exact hget.symm

/-- `get_at_or_zero` on an `ofVecT`-constructed 1D tensor reads back the corresponding flattened
  entry. -/
private lemma get1_ofVecT
    {C : Nat} (v : Vec (Shape.size (.dim C .scalar))) (c : Fin C) :
    getAtOrZero (ofVecT (s := .dim C .scalar) v) [c.val] =
      v (Fin.cast (by simp [Shape.size]) c) := by
  have htv :=
    congrArg (fun w => w (Fin.cast (by simp [Shape.size]) c))
      (toVecT_ofVecT (s := .dim C .scalar) v)
  exact (toVecT_get1 (A := ofVecT (s := .dim C .scalar) v) c).symm.trans htv

private lemma idx3S_eq_nested
    {C H W : Nat} (c : Fin C) (i : Fin H) (j : Fin W) :
    idx3S (C := C) (H := H) (W := W) c i j =
      finProdFinEquiv (c, finProdFinEquiv (i, finProdFinEquiv (j, (0 : Fin 1)))) := by
  apply Fin.ext
  simp [idx3S, idx3, Shape.size, finProdFinEquiv, Fin.cast]

private lemma idx4S_eq_nested
    {OC IC KH KW : Nat} (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj =
      finProdFinEquiv
        (oc, finProdFinEquiv (ic, finProdFinEquiv (di, finProdFinEquiv (dj, (0 : Fin 1))))) := by
  apply Fin.ext
  simp [idx4S, idx4, Shape.size, finProdFinEquiv, Fin.cast]

/-- Relate 3D tensor `toVecT` coordinates to `get_at_or_zero` via the `idx3S` index encoding. -/
private lemma toVecT_get3
    {C H W : Nat} (A : Tensor ℝ (.dim C (.dim H (.dim W .scalar))))
    (c : Fin C) (i : Fin H) (j : Fin W) :
    toVecT (t := A) (idx3S (C := C) (H := H) (W := W) c i j) =
      getAtOrZero A [c.val, i.val, j.val] := by
  classical
  cases W with
  | zero => exact (Fin.elim0 j)
  | succ W =>
    cases H with
    | zero => exact (Fin.elim0 i)
    | succ H =>
      cases A with
      | dim fC =>
        let hW : vecSize (Nat.succ W) = Nat.succ W := by simp [vecSize, Shape.size]
        let j' : Fin (vecSize (Nat.succ W)) := Fin.cast hW.symm j
        have hmposHW : 0 < Shape.size (.dim (Nat.succ H) (.dim (Nat.succ W) .scalar)) := by
          simp [Shape.size]
        have hmposW : 0 < Shape.size (.dim (Nat.succ W) .scalar) := by
          simp [Shape.size]
        let k0 : Fin 1 := 0
        have hmposSc : 0 < Shape.size Shape.scalar := by simp [Shape.size]
        -- peel all three dims using `toVecT_dim_apply`
        have houter :
            toVecT (t := (Tensor.dim fC : Tensor ℝ (.dim C (.dim (Nat.succ H) (.dim (Nat.succ W)
              .scalar)))))
                (finProdFinEquiv (c, finProdFinEquiv (i, j')))
              =
            toVecT (t := fC c) (finProdFinEquiv (i, j')) := by
          simpa using
            (toVecT_dim_apply (n := C) (s := .dim (Nat.succ H) (.dim (Nat.succ W) .scalar))
              (hmpos := hmposHW) (f := fC) (p := (c, finProdFinEquiv (i, j'))))
        cases hrow : fC c with
        | dim fH =>
          have hmid :
              toVecT (t := (Tensor.dim fH : Tensor ℝ (.dim (Nat.succ H) (.dim (Nat.succ W)
                .scalar))))
                  (finProdFinEquiv (i, j'))
                =
              toVecT (t := fH i) j' := by
            simpa using
              (toVecT_dim_apply (n := Nat.succ H) (s := .dim (Nat.succ W) .scalar)
                (hmpos := hmposW) (f := fH) (p := (i, j')))
          cases hcol : fH i with
          | dim fW =>
            have hinner :
                toVecT (t := (Tensor.dim fW : Tensor ℝ (.dim (Nat.succ W) Shape.scalar)))
                    (finProdFinEquiv (j, k0))
                  =
                toVecT (t := fW j) k0 := by
              simpa using
                (toVecT_dim_apply (n := Nat.succ W) (s := Shape.scalar)
                  (hmpos := hmposSc) (f := fW) (p := (j, k0)))
            have hjidx : finProdFinEquiv (j, k0) = j' := by
              apply Fin.ext
              simp [j', k0, Shape.size, finProdFinEquiv]
            -- rewrite the last axis using `hinner` and `hjidx`
            have hlast :
                toVecT (t := (Tensor.dim fW : Tensor ℝ (.dim (Nat.succ W) Shape.scalar))) j' =
                  toVecT (t := fW j) k0 := by
              -- rewrite `j'` to `finProdFinEquiv (j,k0)` and apply `hinner`
              have hcast :=
                congrArg (fun z => toVecT (t := (Tensor.dim fW : Tensor ℝ (.dim (Nat.succ W)
                  Shape.scalar))) z) hjidx.symm
              exact hcast.trans hinner
            -- compute the `toVecT` entry by chaining the peels
            have htv :
                toVecT (t := (Tensor.dim fC : Tensor ℝ (.dim C (.dim (Nat.succ H) (.dim (Nat.succ W)
                  .scalar)))))
                    (idx3S (C := C) (H := Nat.succ H) (W := Nat.succ W) c i j)
                  =
                toVecT (t := fW j) k0 := by
              have hidx :
                  idx3S (C := C) (H := Nat.succ H) (W := Nat.succ W) c i j =
                    finProdFinEquiv (c, finProdFinEquiv (i, j')) := by
                have hinnerIdx :
                    finProdFinEquiv (i, finProdFinEquiv (j, k0)) =
                      finProdFinEquiv (i, j') := by
                  exact congrArg (fun z => finProdFinEquiv (i, z)) hjidx
                exact (idx3S_eq_nested (c := c) (i := i) (j := j)).trans <|
                  congrArg (fun z => finProdFinEquiv (c, z)) hinnerIdx
              calc
                toVecT
                    (t := (Tensor.dim fC : Tensor ℝ (.dim C (.dim (Nat.succ H) (.dim (Nat.succ W)
                      .scalar)))))
                    (idx3S (C := C) (H := Nat.succ H) (W := Nat.succ W) c i j)
                    =
                  toVecT
                    (t := (Tensor.dim fC : Tensor ℝ (.dim C (.dim (Nat.succ H) (.dim (Nat.succ W)
                      .scalar)))))
                    (finProdFinEquiv (c, finProdFinEquiv (i, j'))) := by
                      exact congrArg
                        (fun z =>
                          toVecT
                            (t := (Tensor.dim fC :
                              Tensor ℝ (.dim C (.dim (Nat.succ H) (.dim (Nat.succ W) .scalar)))))
                            z)
                        hidx
                _ = toVecT (t := fC c) (finProdFinEquiv (i, j')) := houter
                _ = toVecT (t := (Tensor.dim fH : Tensor ℝ (.dim (Nat.succ H) (.dim (Nat.succ W)
                  .scalar))))
                      (finProdFinEquiv (i, j')) := by simp [hrow]
                _ = toVecT (t := fH i) j' := hmid
                _ = toVecT (t := (Tensor.dim fW : Tensor ℝ (.dim (Nat.succ W) Shape.scalar))) j' :=
                  by simp [hcol]
                _ = toVecT (t := fW j) k0 := hlast
            -- compute the RHS `get_at_or_zero` by unfolding the tensor structure
            cases hcell : fW j with
            | scalar x =>
              have hget :
                  getAtOrZero (Tensor.dim fC) [c.val, i.val, j.val] = x := by
                simp [hrow, hcol, hcell, c.isLt, i.isLt, j.isLt]
              -- `toVecT (Tensor.scalar x) k0 = x`
              have hsc : toVecT (t := (Tensor.scalar x : Tensor ℝ Shape.scalar)) k0 = x :=
                toVecT_scalar_apply (x := x) (i := k0)
              simpa [htv, hget, hcell] using hsc

/-- `get_at_or_zero` on an `ofVecT`-constructed 3D tensor reads back the corresponding flattened
  entry. -/
private lemma get3_ofVecT
    {C H W : Nat} (v : Vec (Shape.size (.dim C (.dim H (.dim W .scalar)))))
    (c : Fin C) (i : Fin H) (j : Fin W) :
    getAtOrZero (ofVecT (s := .dim C (.dim H (.dim W .scalar))) v) [c.val, i.val, j.val] =
      v (idx3S (C := C) (H := H) (W := W) c i j) := by
  have htv :=
    congrArg (fun w => w (idx3S (C := C) (H := H) (W := W) c i j))
      (toVecT_ofVecT (s := .dim C (.dim H (.dim W .scalar))) v)
  -- Use `toVecT_get3` to rewrite the LHS into a `toVecT` entry, then substitute using
  -- `toVecT_ofVecT`.
  have hget :=
    (toVecT_get3 (A := ofVecT (s := .dim C (.dim H (.dim W .scalar))) v) c i j).symm
  simpa using hget.trans htv

/-- Relate 4D kernel `toVecT` coordinates to `get_at_or_zero` via the `idx4S` index encoding. -/
private lemma toVecT_get4
    {OC IC KH KW : Nat} (K : Tensor ℝ (.dim OC (.dim IC (.dim KH (.dim KW .scalar)))))
    (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    toVecT (t := K) (idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) =
      getAtOrZero K [oc.val, ic.val, di.val, dj.val] := by
  classical
  cases KW with
  | zero => exact (Fin.elim0 dj)
  | succ KW =>
    cases KH with
    | zero => exact (Fin.elim0 di)
    | succ KH =>
      cases IC with
      | zero => exact (Fin.elim0 ic)
      | succ IC =>
        cases K with
        | dim fOC =>
          let hKW : vecSize (Nat.succ KW) = Nat.succ KW := by simp [vecSize, Shape.size]
          let dj' : Fin (vecSize (Nat.succ KW)) := Fin.cast hKW.symm dj
          have hmposInner : 0 < Shape.size (.dim (Nat.succ IC) (.dim (Nat.succ KH) (.dim (Nat.succ
            KW) .scalar))) := by
            simp [Shape.size]
          have hmposKH : 0 < Shape.size (.dim (Nat.succ KH) (.dim (Nat.succ KW) .scalar)) := by
            simp [Shape.size]
          have hmposKW : 0 < Shape.size (.dim (Nat.succ KW) .scalar) := by
            simp [Shape.size]
          let k0 : Fin 1 := 0
          have hmposSc : 0 < Shape.size Shape.scalar := by simp [Shape.size]
          have houter :
              toVecT (t := (Tensor.dim fOC : Tensor ℝ (.dim OC (.dim (Nat.succ IC) (.dim (Nat.succ
                KH) (.dim (Nat.succ KW) .scalar))))))
                  (finProdFinEquiv (oc, finProdFinEquiv (ic, finProdFinEquiv (di, dj'))))
                =
              toVecT (t := fOC oc) (finProdFinEquiv (ic, finProdFinEquiv (di, dj'))) := by
            simpa using
              (toVecT_dim_apply (n := OC) (s := .dim (Nat.succ IC) (.dim (Nat.succ KH) (.dim
                (Nat.succ KW) .scalar)))
                (hmpos := hmposInner) (f := fOC) (p := (oc, finProdFinEquiv (ic, finProdFinEquiv
                  (di, dj')))))
          cases hIC : fOC oc with
          | dim fIC =>
            have hIC' :
                toVecT (t := (Tensor.dim fIC : Tensor ℝ (.dim (Nat.succ IC) (.dim (Nat.succ KH)
                  (.dim (Nat.succ KW) .scalar)))))
                    (finProdFinEquiv (ic, finProdFinEquiv (di, dj')))
                  =
                toVecT (t := fIC ic) (finProdFinEquiv (di, dj')) := by
              simpa using
                (toVecT_dim_apply (n := Nat.succ IC) (s := .dim (Nat.succ KH) (.dim (Nat.succ KW)
                  .scalar))
                  (hmpos := hmposKH) (f := fIC) (p := (ic, finProdFinEquiv (di, dj'))))
            cases hKH : fIC ic with
            | dim fKH =>
              have hKH' :
                  toVecT (t := (Tensor.dim fKH : Tensor ℝ (.dim (Nat.succ KH) (.dim (Nat.succ KW)
                    .scalar))))
                      (finProdFinEquiv (di, dj'))
                    =
                  toVecT (t := fKH di) dj' := by
                simpa using
                  (toVecT_dim_apply (n := Nat.succ KH) (s := .dim (Nat.succ KW) .scalar)
                    (hmpos := hmposKW) (f := fKH) (p := (di, dj')))
              cases hKW' : fKH di with
              | dim fKW =>
                have hinner :
                    toVecT (t := (Tensor.dim fKW : Tensor ℝ (.dim (Nat.succ KW) Shape.scalar)))
                        (finProdFinEquiv (dj, k0))
                      =
                    toVecT (t := fKW dj) k0 := by
                  simpa using
                    (toVecT_dim_apply (n := Nat.succ KW) (s := Shape.scalar)
                      (hmpos := hmposSc) (f := fKW) (p := (dj, k0)))
                have hjidx : finProdFinEquiv (dj, k0) = dj' := by
                  apply Fin.ext
                  simp [dj', k0, Shape.size, finProdFinEquiv]
                have hlast :
                    toVecT (t := (Tensor.dim fKW : Tensor ℝ (.dim (Nat.succ KW) Shape.scalar))) dj'
                      =
                      toVecT (t := fKW dj) k0 := by
                  have hcast :=
                    congrArg (fun z => toVecT (t := (Tensor.dim fKW : Tensor ℝ (.dim (Nat.succ KW)
                      Shape.scalar))) z) hjidx.symm
                  exact hcast.trans hinner
                have hidx :
                    idx4S (OC := OC) (IC := Nat.succ IC) (KH := Nat.succ KH) (KW := Nat.succ KW) oc
                      ic di dj
                      =
                    finProdFinEquiv (oc, finProdFinEquiv (ic, finProdFinEquiv (di, dj'))) := by
                  have hdi :
                      finProdFinEquiv (di, finProdFinEquiv (dj, k0)) =
                        finProdFinEquiv (di, dj') := by
                    exact congrArg (fun z => finProdFinEquiv (di, z)) hjidx
                  have hic :
                      finProdFinEquiv (ic, finProdFinEquiv (di, finProdFinEquiv (dj, k0))) =
                        finProdFinEquiv (ic, finProdFinEquiv (di, dj')) := by
                    exact congrArg (fun z => finProdFinEquiv (ic, z)) hdi
                  exact (idx4S_eq_nested (oc := oc) (ic := ic) (di := di) (dj := dj)).trans <|
                    congrArg (fun z => finProdFinEquiv (oc, z)) hic
                have htv :
                    toVecT (t := (Tensor.dim fOC : Tensor ℝ (.dim OC (.dim (Nat.succ IC) (.dim
                      (Nat.succ KH) (.dim (Nat.succ KW) .scalar))))))
                        (idx4S (OC := OC) (IC := Nat.succ IC) (KH := Nat.succ KH) (KW := Nat.succ
                          KW) oc ic di dj)
                      =
                    toVecT (t := fKW dj) k0 := by
                  calc
                    toVecT
                          (t :=
                            (Tensor.dim fOC :
                              Tensor ℝ
                                (.dim OC
                                  (.dim (Nat.succ IC) (.dim (Nat.succ KH) (.dim (Nat.succ KW)
                                    .scalar))))))
                          (idx4S (OC := OC) (IC := Nat.succ IC) (KH := Nat.succ KH) (KW := Nat.succ
                            KW) oc ic di dj)
                        =
                      toVecT (t := (Tensor.dim fOC : Tensor ℝ (.dim OC (.dim (Nat.succ IC) (.dim
                        (Nat.succ KH) (.dim (Nat.succ KW) .scalar))))))
                        (finProdFinEquiv (oc, finProdFinEquiv (ic, finProdFinEquiv (di, dj')))) :=
                          by
                          exact congrArg
                            (fun z =>
                              toVecT
                                (t := (Tensor.dim fOC :
                                  Tensor ℝ (.dim OC (.dim (Nat.succ IC) (.dim (Nat.succ KH) (.dim
                                    (Nat.succ KW) .scalar))))))
                                z)
                            hidx
                    _ = toVecT (t := fOC oc) (finProdFinEquiv (ic, finProdFinEquiv (di, dj'))) :=
                      houter
                    _ = toVecT (t := (Tensor.dim fIC : Tensor ℝ (.dim (Nat.succ IC) (.dim (Nat.succ
                      KH) (.dim (Nat.succ KW) .scalar)))))
                          (finProdFinEquiv (ic, finProdFinEquiv (di, dj'))) := by
                          simp [hIC]
                    _ = toVecT (t := fIC ic) (finProdFinEquiv (di, dj')) := hIC'
                    _ = toVecT (t := (Tensor.dim fKH : Tensor ℝ (.dim (Nat.succ KH) (.dim (Nat.succ
                      KW) .scalar))))
                          (finProdFinEquiv (di, dj')) := by
                          simp [hKH]
                    _ = toVecT (t := fKH di) dj' := hKH'
                    _ = toVecT (t := (Tensor.dim fKW : Tensor ℝ (.dim (Nat.succ KW) Shape.scalar)))
                      dj' := by
                          simp [hKW']
                    _ = toVecT (t := fKW dj) k0 := hlast
                cases hcell : fKW dj with
                | scalar x =>
                  have hget :
                      getAtOrZero (Tensor.dim fOC) [oc.val, ic.val, di.val, dj.val] = x := by
                    simp [hIC, hKH, hKW', hcell, oc.isLt, ic.isLt, di.isLt, dj.isLt]
                  have hsc : toVecT (t := (Tensor.scalar x : Tensor ℝ Shape.scalar)) k0 = x :=
                    toVecT_scalar_apply (x := x) (i := k0)
                  simpa [htv, hget, hcell] using hsc

/-- `get_at_or_zero` on an `ofVecT`-constructed 4D tensor reads back the corresponding flattened
  entry. -/
private lemma get4_ofVecT
    {OC IC KH KW : Nat} (v : Vec (Shape.size (.dim OC (.dim IC (.dim KH (.dim KW .scalar))))))
    (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
    getAtOrZero (ofVecT (s := .dim OC (.dim IC (.dim KH (.dim KW .scalar)))) v)
        [oc.val, ic.val, di.val, dj.val]
      =
    v (idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) := by
  have htv :=
    congrArg (fun w => w (idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj))
      (toVecT_ofVecT (s := .dim OC (.dim IC (.dim KH (.dim KW .scalar)))) v)
  have hget :=
    (toVecT_get4 (K := ofVecT (s := .dim OC (.dim IC (.dim KH (.dim KW .scalar)))) v) oc ic di
      dj).symm
  simpa using hget.trans htv

/-- Decode an output index `OC*(OH*OW)` into `(oc, oh, ow)`. -/
private def splitOut {OC OH OW : Nat} (ip : Fin (OC * (OH * OW))) : Fin OC × Fin OH × Fin OW :=
  let p1 : Fin OC × Fin (OH * OW) :=
    (finProdFinEquiv : Fin OC × Fin (OH * OW) ≃ Fin (OC * (OH * OW))).symm ip
  let p2 : Fin OH × Fin OW :=
    (finProdFinEquiv : Fin OH × Fin OW ≃ Fin (OH * OW)).symm p1.2
  (p1.1, p2.1, p2.2)

/-- `splitOut` is inverse to `idx3` for output-shaped indices. -/
private theorem idx3_splitOut {OC OH OW : Nat} (ip : Fin (OC * (OH * OW))) :
    idx3 (C := OC) (H := OH) (W := OW) (splitOut (OC := OC) (OH := OH) (OW := OW) ip).1
      (splitOut (OC := OC) (OH := OH) (OW := OW) ip).2.1
      (splitOut (OC := OC) (OH := OH) (OW := OW) ip).2.2
      =
    ip := by
  classical
  -- Avoid unfolding `finProdFinEquiv.symm` (which introduces `divNat`/`modNat`), and instead use
  -- `apply_symm_apply` at the pair level.
  let p1 : Fin OC × Fin (OH * OW) :=
    (finProdFinEquiv : Fin OC × Fin (OH * OW) ≃ Fin (OC * (OH * OW))).symm ip
  let p2 : Fin OH × Fin OW :=
    (finProdFinEquiv : Fin OH × Fin OW ≃ Fin (OH * OW)).symm p1.2
  have hp2 :
      (finProdFinEquiv : Fin OH × Fin OW ≃ Fin (OH * OW)) p2 = p1.2 := by
    simpa [p2] using
      (Equiv.apply_symm_apply (finProdFinEquiv : Fin OH × Fin OW ≃ Fin (OH * OW)) p1.2)
  have hp1 :
      (finProdFinEquiv : Fin OC × Fin (OH * OW) ≃ Fin (OC * (OH * OW))) p1 = ip := by
    simpa [p1] using
      (Equiv.apply_symm_apply (finProdFinEquiv : Fin OC × Fin (OH * OW) ≃ Fin (OC * (OH * OW))) ip)
  have hsplit : splitOut (OC := OC) (OH := OH) (OW := OW) ip = (p1.1, p2.1, p2.2) := by
    rfl
  -- rewrite the output index as `(p1,p2)` and chain the two `apply_symm_apply` facts
  calc
    idx3 (C := OC) (H := OH) (W := OW)
        (splitOut (OC := OC) (OH := OH) (OW := OW) ip).1
        (splitOut (OC := OC) (OH := OH) (OW := OW) ip).2.1
        (splitOut (OC := OC) (OH := OH) (OW := OW) ip).2.2
        =
      idx3 (C := OC) (H := OH) (W := OW) p1.1 p2.1 p2.2 := by
        simp [hsplit]
    _ = finProdFinEquiv (p1.1, finProdFinEquiv (p2.1, p2.2)) := rfl
    _ = finProdFinEquiv (p1.1, p1.2) := by
        simpa using congrArg (fun t => finProdFinEquiv (p1.1, t)) hp2
    _ = ip := by
        simpa using hp1

/--
Read an input coordinate from a flattened `(IC,IH,IW)` input tensor, returning `0` out of bounds.

This models the padding/out-of-range behavior in the convolution sum (at the vector level).
-/
private def getInput {IC IH IW : Nat} (x : Vec (IC * (IH * IW))) (ic : Fin IC) (i j : Nat) : ℝ :=
  if hi : i < IH then
    if hj : j < IW then
      x (idx3 (C := IC) (H := IH) (W := IW) ic ⟨i, hi⟩ ⟨j, hj⟩)
    else
      0
  else
    0

/-- `getInput` respects addition of the underlying flattened vectors. -/
private lemma getInput_add {IC IH IW : Nat} (x1 x2 : Vec (IC * (IH * IW))) (ic : Fin IC) (i j : Nat)
  :
    getInput (IC := IC) (IH := IH) (IW := IW) (x1 + x2) ic i j =
      getInput (IC := IC) (IH := IH) (IW := IW) x1 ic i j +
      getInput (IC := IC) (IH := IH) (IW := IW) x2 ic i j := by
  classical
  by_cases hi : i < IH <;> by_cases hj : j < IW <;> simp [getInput, hi, hj]

/-- `getInput` respects scalar multiplication of the underlying flattened vectors. -/
private lemma getInput_smul {IC IH IW : Nat} (r : ℝ) (x : Vec (IC * (IH * IW))) (ic : Fin IC) (i j :
  Nat) :
    getInput (IC := IC) (IH := IH) (IW := IW) (r • x) ic i j =
      r * getInput (IC := IC) (IH := IH) (IW := IW) x ic i j := by
  classical
  by_cases hi : i < IH <;> by_cases hj : j < IW <;> simp [getInput, hi, hj, smul_eq_mul]

/-- Convenience lemma: add two nested-`if` (“ite2”) expressions pointwise. -/
private lemma ite2_add {P Q : Prop} [Decidable P] [Decidable Q] (a b : ℝ) :
    (if P then 0 else if Q then 0 else a + b) =
      (if P then 0 else if Q then 0 else a) + (if P then 0 else if Q then 0 else b) := by
  by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

/-- Rewrite a disjunction test into the corresponding nested `if` form. -/
private lemma ite_or_eq_ite2 {P Q : Prop} [Decidable P] [Decidable Q] (a : ℝ) :
    (if P ∨ Q then (0 : ℝ) else a) = (if P then 0 else if Q then 0 else a) := by
  by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

/-- Bias broadcast over the last two axes: `Vec OC → Vec (OC*(OH*OW))`. -/
private def biasBroadcastVec {OC OH OW : Nat} (b : Vec OC) : Vec (OC * (OH * OW)) :=
  vecOfFun (n := OC * (OH * OW)) fun ip =>
    let oc := (splitOut (OC := OC) (OH := OH) (OW := OW) ip).1
    b oc

/-- Continuous linear map version of `biasBroadcastVec`. -/
private def biasBroadcastCLM {OC OH OW : Nat} : Vec OC →L[ℝ] Vec (OC * (OH * OW)) := by
  classical
  let fLin : Vec OC →ₗ[ℝ] Vec (OC * (OH * OW)) :=
    { toFun := biasBroadcastVec (OC := OC) (OH := OH) (OW := OW)
      map_add' := by
        intro b1 b2
        ext ip
        simp [biasBroadcastVec]
      map_smul' := by
        intro r b
        ext ip
        simp [biasBroadcastVec, smul_eq_mul] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] private lemma biasBroadcastCLM_apply {OC OH OW : Nat} (b : Vec OC) :
    biasBroadcastCLM (OC := OC) (OH := OH) (OW := OW) b = biasBroadcastVec (OC := OC) (OH := OH) (OW
      := OW) b := by
  rfl

/-- Convolution without bias, as an explicit bilinear map on flattened vectors. -/
private def convNoBiasVec
    {IC OC KH KW stride padding IH IW : Nat}
    (kernel : Vec (OC * (IC * (KH * KW))))
    (x : Vec (IC * (IH * IW)))
    (outH outW : Nat) :
    Vec (OC * (outH * outW)) :=
  vecOfFun (n := OC * (outH * outW)) fun ip =>
    let p := splitOut (OC := OC) (OH := outH) (OW := outW) ip
    let oc : Fin OC := p.1
    let oi : Fin outH := p.2.1
    let oj : Fin outW := p.2.2
    ∑ ic : Fin IC,
      ∑ di : Fin KH,
        ∑ dj : Fin KW,
          let raw_i := oi.1 * stride + di.1
          let raw_j := oj.1 * stride + dj.1
          if _hraw_i : raw_i < padding then
            0
          else if _hraw_j : raw_j < padding then
            0
          else
            let in_i := raw_i - padding
            let in_j := raw_j - padding
            kernel (idx4 (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) *
              getInput (IC := IC) (IH := IH) (IW := IW) x ic in_i in_j

/-!
The remaining definitions package the convolution computation into linear/bilinear maps on
flattened vectors. This is the standard analytic path: show the forward map is bilinear in
`(kernel, input)`, then take Fréchet derivatives and adjoints.
-/

private def convCLMRight
    {IC OC KH KW stride padding IH IW : Nat}
    (kernel : Vec (OC * (IC * (KH * KW))))
    (outH outW : Nat) :
    Vec (IC * (IH * IW)) →L[ℝ] Vec (OC * (outH * outW)) := by
  classical
  let fLin : Vec (IC * (IH * IW)) →ₗ[ℝ] Vec (OC * (outH * outW)) :=
    { toFun := fun x => convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW)
        (stride := stride) (padding := padding) (IH := IH) (IW := IW) kernel x outH outW
      map_add' := by
        intro x1 x2
        ext ip
        -- Expand, distribute `getInput` over addition, then split the finite sums.
        -- The index guards are expressed as nested `if`s; split addition inside those `if`s.
        simp [convNoBiasVec, getInput_add, ite2_add, mul_add, Finset.sum_add_distrib]
      map_smul' := by
        intro r x
        ext ip
        -- Expand, pull out scalar multiplication through the finite sums.
        simp [convNoBiasVec, getInput_smul, smul_eq_mul, Finset.mul_sum, mul_assoc,
          mul_comm] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] private lemma convCLMRight_apply
    {IC OC KH KW stride padding IH IW : Nat}
    (kernel : Vec (OC * (IC * (KH * KW)))) (outH outW : Nat) (x : Vec (IC * (IH * IW))) :
    convCLMRight (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding)
        (IH := IH) (IW := IW) kernel outH outW x
      =
    convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
      padding)
        (IH := IH) (IW := IW) kernel x outH outW := by
  rfl

/-- Convolution without bias as a continuous bilinear map on flattened vectors. -/
private def convBilin
    {IC OC KH KW stride padding IH IW : Nat}
    (outH outW : Nat) :
    Vec (OC * (IC * (KH * KW))) →L[ℝ] Vec (IC * (IH * IW)) →L[ℝ] Vec (OC * (outH * outW)) := by
  classical
  let fLin : Vec (OC * (IC * (KH * KW))) →ₗ[ℝ] Vec (IC * (IH * IW)) →L[ℝ] Vec (OC * (outH * outW))
    :=
    { toFun := fun k => convCLMRight (IC := IC) (OC := OC) (KH := KH) (KW := KW)
        (stride := stride) (padding := padding) (IH := IH) (IW := IW) k outH outW
      map_add' := by
        intro k1 k2
        ext x ip
        simp [convCLMRight, convNoBiasVec, ite2_add, add_mul, Finset.sum_add_distrib]
      map_smul' := by
        intro r k
        ext x ip
        simp [convCLMRight, convNoBiasVec, Pi.smul_apply, smul_eq_mul, Finset.mul_sum, mul_assoc] }
  refine ⟨fLin, ?_⟩
  exact LinearMap.continuous_of_finiteDimensional (f := fLin)

@[simp] private lemma convBilin_apply
    {IC OC KH KW stride padding IH IW : Nat}
    (outH outW : Nat) (k : Vec (OC * (IC * (KH * KW)))) (x : Vec (IC * (IH * IW))) :
    (convBilin (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding)
        (IH := IH) (IW := IW) outH outW k) x
      =
    convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
      padding)
        (IH := IH) (IW := IW) k x outH outW := by
  simp [convBilin]

  /-- Output height of `conv2d`, computed from input height, kernel height, stride, and padding. -/
  abbrev outH {IH KH stride padding : Nat} : Nat :=
    (IH + 2 * padding - KH) / stride + 1
  /-- Output width of `conv2d`, computed from input width, kernel width, stride, and padding. -/
  abbrev outW {IW KW stride padding : Nat} : Nat :=
    (IW + 2 * padding - KW) / stride + 1

  /-- Shape of the Conv2D kernel parameter `K` (a `OC × IC × KH × KW` tensor). -/
  abbrev sK (OC IC KH KW : Nat) : Shape := .dim OC (.dim IC (.dim KH (.dim KW .scalar)))
  /-- Shape of the Conv2D bias parameter `b` (a length-`OC` vector). -/
  abbrev sB (OC : Nat) : Shape := .dim OC .scalar
  /-- Shape of the Conv2D input `X` (a `IC × IH × IW` tensor). -/
  abbrev sX (IC IH IW : Nat) : Shape := .dim IC (.dim IH (.dim IW .scalar))
  /-- Shape of the Conv2D output `Y` (a `OC × OH × OW` tensor). -/
  abbrev sY (OC OH OW : Nat) : Shape := .dim OC (.dim OH (.dim OW .scalar))

  private lemma size_sK (OC IC KH KW : Nat) :
      Shape.size (sK OC IC KH KW) = OC * (IC * (KH * KW)) := by
    simp [Shape.size]

  private lemma size_sB (OC : Nat) : Shape.size (sB OC) = OC := by
    simp [Shape.size]

  private lemma size_sX (IC IH IW : Nat) : Shape.size (sX IC IH IW) = IC * (IH * IW) := by
    simp [Shape.size]

  private lemma size_sY (OC OH OW : Nat) : Shape.size (sY OC OH OW) = OC * (OH * OW) := by
    simp [Shape.size]

  private lemma idx3S_cast_size_sX {IC IH IW : Nat} (ic : Fin IC) (i : Fin IH) (j : Fin IW) :
      Fin.cast (size_sX IC IH IW) (idx3S (C := IC) (H := IH) (W := IW) ic i j)
        =
      idx3 (C := IC) (H := IH) (W := IW) ic i j := by
    simp [idx3S, idx3, Shape.size]

  private lemma idx3S_cast_size_sY {OC OH OW : Nat} (oc : Fin OC) (i : Fin OH) (j : Fin OW) :
      Fin.cast (size_sY OC OH OW) (idx3S (C := OC) (H := OH) (W := OW) oc i j)
        =
      idx3 (C := OC) (H := OH) (W := OW) oc i j := by
    simp [idx3S, idx3, Shape.size]

  private lemma idx4S_cast_size_sK {OC IC KH KW : Nat}
      (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
      Fin.cast (size_sK OC IC KH KW) (idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj)
        =
      idx4 (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj := by
    simp [idx4S, idx4, Shape.size]

  private lemma get_at_or_zero_ofVecT_sX_eq_getInput
      {IC IH IW : Nat} (xRaw : Vec (IC * (IH * IW))) (ic : Fin IC) (p q : Nat) :
      let xShape : Vec (Shape.size (sX IC IH IW)) := castVec (size_sX IC IH IW).symm xRaw
      getAtOrZero (ofVecT (s := sX IC IH IW) xShape) [ic.val, p, q]
        =
      getInput (IC := IC) (IH := IH) (IW := IW) xRaw ic p q := by
    intro xShape
    classical
    by_cases hp : p < IH
    · by_cases hq : q < IW
      · -- In-bounds: reduce to `get3_ofVecT` then map the index through the `castVec`.
        let iFin : Fin IH := ⟨p, hp⟩
        let jFin : Fin IW := ⟨q, hq⟩
        have hget :=
          get3_ofVecT (v := xShape) (c := ic) (i := iFin) (j := jFin)
        -- Rewrite the RHS `getInput` to the raw index form.
        have hInput : getInput (IC := IC) (IH := IH) (IW := IW) xRaw ic p q =
            xRaw (idx3 (C := IC) (H := IH) (W := IW) ic iFin jFin) := by
          simp [getInput, hp, hq, iFin, jFin]
        -- Rewrite the `xShape` entry using the `castVec` definition and the index-cast lemma.
        have hxShape :
            xShape (idx3S (C := IC) (H := IH) (W := IW) ic iFin jFin)
              =
            xRaw (idx3 (C := IC) (H := IH) (W := IW) ic iFin jFin) := by
          -- `xShape` is a cast of `xRaw`; evaluate at this index and rewrite the casted index.
          -- Keep `castVec` opaque; use `castVec_apply` and then rewrite the casted index.
          simpa [xShape] using
            congrArg xRaw (idx3S_cast_size_sX (IC := IC) (IH := IH) (IW := IW) ic iFin jFin)
        simpa [hInput, hxShape] using hget
      · -- `q` out of bounds: both sides are `0`.
        simp [getInput, hp, hq, ofVecT, Spec.Tensor.unflattenSpec, ofVecE, Spec.ofVec,
          xShape, ic.isLt, sX]
    · -- `p` out of bounds: both sides are `0`.
      simp [getInput, hp, ofVecT, Spec.Tensor.unflattenSpec, ofVecE, Spec.ofVec,
        xShape, ic.isLt, sX]

  private lemma get_at_or_zero_ofVecT_sK_eq_idx4
      {OC IC KH KW : Nat} (kRaw : Vec (OC * (IC * (KH * KW))))
      (oc : Fin OC) (ic : Fin IC) (di : Fin KH) (dj : Fin KW) :
      let kShape : Vec (Shape.size (sK OC IC KH KW)) := castVec (size_sK OC IC KH KW).symm kRaw
      getAtOrZero (ofVecT (s := sK OC IC KH KW) kShape) [oc.val, ic.val, di.val, dj.val]
        =
      kRaw (idx4 (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) := by
    intro kShape
    classical
    have hget :=
      get4_ofVecT (v := kShape) (oc := oc) (ic := ic) (di := di) (dj := dj)
    have hk :
        kShape (idx4S (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) =
          kRaw (idx4 (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj) := by
      simpa [kShape] using
        congrArg kRaw (idx4S_cast_size_sK (OC := OC) (IC := IC) (KH := KH) (KW := KW) oc ic di dj)
    exact hget.trans hk

  private lemma mul_ite2_zero_right {P Q : Prop} [Decidable P] [Decidable Q] (a b : ℝ) :
      a * (if P then 0 else if Q then 0 else b) = (if P then 0 else if Q then 0 else a * b) := by
    by_cases hP : P <;> by_cases hQ : Q <;> simp [hP, hQ]

  private lemma cast_toVecT_conv2d_spec_noBias_eq_convNoBiasVec
      {IC OC KH KW stride padding IH IW : Nat}
      {h1 : IC ≠ 0} {h2 : KH ≠ 0} {h3 : KW ≠ 0}
      (kRaw : Vec (OC * (IC * (KH * KW))))
      (xRaw : Vec (IC * (IH * IW))) :
      let OH := outH (IH := IH) (KH := KH) (stride := stride) (padding := padding)
      let OW := outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)
      let kShape : Vec (Shape.size (sK OC IC KH KW)) := castVec (size_sK OC IC KH KW).symm kRaw
      let xShape : Vec (Shape.size (sX IC IH IW)) := castVec (size_sX IC IH IW).symm xRaw
      let dKernel : Tensor ℝ (sK OC IC KH KW) := ofVecT (s := sK OC IC KH KW) kShape
      let input : Tensor ℝ (sX IC IH IW) := ofVecT (s := sX IC IH IW) xShape
      let layerK : Spec.Conv2DSpec IC OC KH KW stride padding ℝ h1 h2 h3 :=
        { kernel := dKernel, bias := fill (0 : ℝ) (sB OC) }
      castVec (size_sY OC OH OW) (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) input))
        =
      convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
        padding) (IH := IH)
        (IW := IW) kRaw xRaw OH OW := by
    intro OH OW kShape xShape dKernel input layerK
    classical
    ext ip
    -- decode the output index
    let p := splitOut (OC := OC) (OH := OH) (OW := OW) ip
    let oc : Fin OC := p.1
    let oi : Fin OH := p.2.1
    let oj : Fin OW := p.2.2
    have hip : idx3 (C := OC) (H := OH) (W := OW) oc oi oj = ip := by
      simpa [p, oc, oi, oj] using (idx3_splitOut (OC := OC) (OH := OH) (OW := OW) ip)
    have hcast :
        Fin.cast (size_sY OC OH OW).symm ip = idx3S (C := OC) (H := OH) (W := OW) oc oi oj := by
      have h := idx3S_cast_size_sY (OC := OC) (OH := OH) (OW := OW) oc oi oj
      have h' := congrArg (Fin.cast (size_sY OC OH OW).symm) h
      have : Fin.cast (size_sY OC OH OW).symm (idx3 (C := OC) (H := OH) (W := OW) oc oi oj)
            =
          idx3S (C := OC) (H := OH) (W := OW) oc oi oj := by
        -- simplify the cast-cast on the LHS of `h'`
        simpa using h'.symm
      simpa [hip] using this

    -- padding read in terms of `getInput` (using the already-proved bridge for `ofVecT`)
    have hpad :
        ∀ ic : Fin IC, ∀ p q : Nat,
          getAtOrZero
              (Proofs.Autograd.Conv2D.paddedInput (inC := IC) (inH := IH) (inW := IW) (padding :=
                padding) input)
              [ic.val, p, q]
            =
          (if p < padding ∨ q < padding then (0 : ℝ)
           else getInput (IC := IC) (IH := IH) (IW := IW) xRaw ic (p - padding) (q - padding)) := by
      intro ic p q
      have h :=
        Proofs.Autograd.Conv2D.get_at_or_zero_paddedInput (inC := IC) (inH := IH) (inW := IW)
          (padding := padding)
          (img := input) (c := ic) (p := p) (q := q)
      by_cases hPQ : p < padding ∨ q < padding
      · simpa [hPQ] using h
      · have hx' :
            getAtOrZero input [ic.val, p - padding, q - padding]
              =
            getInput (IC := IC) (IH := IH) (IW := IW) xRaw ic (p - padding) (q - padding) := by
          simpa [input, xShape] using
            (get_at_or_zero_ofVecT_sX_eq_getInput (xRaw := xRaw) (ic := ic) (p := p - padding) (q :=
              q - padding))
        simp [hPQ, h, hx']

    -- one output entry via `conv2d_spec_noBias_get`
    have hget :
        getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, oi.val, oj.val]
          =
        ∑ ic : Fin IC,
          ∑ di : Fin KH,
            ∑ dj : Fin KW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero
                  (Proofs.Autograd.Conv2D.paddedInput (inC := IC) (inH := IH) (inW := IW) (padding
                    := padding) input)
                  [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := by
      simpa [Proofs.Autograd.Conv2D.outH, Proofs.Autograd.Conv2D.outW, outH, outW, layerK] using
        (Proofs.Autograd.Conv2D.conv2d_spec_noBias_get (dKernel := dKernel) (input := input) (oc :=
          oc) (i := oi)
          (j := oj))

    -- now rewrite `toVecT` to `get_at_or_zero`, then simplify to the explicit `convNoBiasVec`
    -- formula
    have htv :
        (castVec (size_sY OC OH OW) (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK)
          input))) ip
          =
        getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, oi.val, oj.val]
          := by
      -- rewrite the casted index into the 3D coordinate and use `toVecT_get3`
      have h :=
        (toVecT_get3
          (A := Spec.conv2dSpec (α := ℝ) (layer := layerK) input) (c := oc) (i := oi) (j := oj))
      have h' :
          toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) input) (Fin.cast (size_sY OC OH
            OW).symm ip)
            =
          getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, oi.val,
            oj.val] := by
        rw [hcast]
        simpa using h
      simpa [castVec] using h'

    -- finish: expand `hget` and normalize each summand
    calc
      (castVec (size_sY OC OH OW) (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) input)))
        ip
          =
        getAtOrZero (Spec.conv2dSpec (α := ℝ) (layer := layerK) input) [oc.val, oi.val, oj.val]
          := htv
      _ =
        ∑ ic : Fin IC,
          ∑ di : Fin KH,
            ∑ dj : Fin KW,
              getAtOrZero dKernel [oc.val, ic.val, di.val, dj.val] *
                getAtOrZero
                  (Proofs.Autograd.Conv2D.paddedInput (inC := IC) (inH := IH) (inW := IW) (padding
                    := padding) input)
                  [ic.val, oi.val * stride + di.val, oj.val * stride + dj.val] := hget
      _ = _ := by
        -- unfold `convNoBiasVec` and rewrite kernel/padding reads
        simp [convNoBiasVec, p, oc, oi, oj, hpad, ite_or_eq_ite2, dKernel, kShape,
          get_at_or_zero_ofVecT_sK_eq_idx4]

  private lemma cast_toVecT_biasBroadcast_eq_biasBroadcastVec
      {OC OH OW : Nat} (bRaw : Vec OC) :
      let bShape : Vec (Shape.size (sB OC)) := castVec (size_sB OC).symm bRaw
      let db : Tensor ℝ (sB OC) := ofVecT (s := sB OC) bShape
      castVec (size_sY OC OH OW)
          (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
            db))
        =
      biasBroadcastVec (OC := OC) (OH := OH) (OW := OW) bRaw := by
    intro bShape db
    classical
    ext ip
    let p := splitOut (OC := OC) (OH := OH) (OW := OW) ip
    let oc : Fin OC := p.1
    let oi : Fin OH := p.2.1
    let oj : Fin OW := p.2.2
    have hip : idx3 (C := OC) (H := OH) (W := OW) oc oi oj = ip := by
      simpa [p, oc, oi, oj] using (idx3_splitOut (OC := OC) (OH := OH) (OW := OW) ip)
    have hcast :
        Fin.cast (size_sY OC OH OW).symm ip = idx3S (C := OC) (H := OH) (W := OW) oc oi oj := by
      have h := idx3S_cast_size_sY (OC := OC) (OH := OH) (OW := OW) oc oi oj
      have h' := congrArg (Fin.cast (size_sY OC OH OW).symm) h
      have : Fin.cast (size_sY OC OH OW).symm (idx3 (C := OC) (H := OH) (W := OW) oc oi oj)
            =
          idx3S (C := OC) (H := OH) (W := OW) oc oi oj := by
        simpa using h'.symm
      simpa [hip] using this
    -- compute the broadcast entry and reduce it to the underlying bias vector
    have hdb : getAtOrZero db [oc.val] = bRaw oc := by
      have hget1 :=
        get1_ofVecT (v := bShape) (c := oc)
      have hbShape :
          bShape (Fin.cast (by simp [Shape.size]) oc) = bRaw oc := by
        simp [bShape, Shape.size]
      simpa [db] using hget1.trans hbShape

    -- finish by rewriting the casted index to a 3D coordinate and evaluating one broadcast entry
    have htv :
        (castVec (size_sY OC OH OW)
            (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW :=
              OW) db))) ip
          =
        getAtOrZero (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
          db)
          [oc.val, oi.val, oj.val] := by
      have h :=
        (toVecT_get3
          (A := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW) db)
          (c := oc) (i := oi) (j := oj))
      have h' :
          toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
            db)
              (Fin.cast (size_sY OC OH OW).symm ip)
            =
          getAtOrZero (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW :=
            OW) db)
            [oc.val, oi.val, oj.val] := by
        rw [hcast]
        simpa using h
      simpa using h'

    calc
      (castVec (size_sY OC OH OW)
          (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
            db))) ip
          =
        getAtOrZero (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
          db)
          [oc.val, oi.val, oj.val] := htv
      _ = getAtOrZero db [oc.val] := by
          simp [Proofs.Autograd.Conv2D.biasBroadcast, oc.isLt, oi.isLt, oj.isLt]
      _ = bRaw oc := hdb
      _ = biasBroadcastVec (OC := OC) (OH := OH) (OW := OW) bRaw ip := by
          simp [biasBroadcastVec, p, oc]

  /-- Project the kernel parameter from a `CtxVec`, casting to the flattened `Vec` layout. -/
  private def projK
      {Γ : List Shape} {OC IC KH KW : Nat}
      (kernelIdx : Idx Γ (sK OC IC KH KW)) :
      CtxVec Γ → Vec (OC * (IC * (KH * KW))) :=
    fun x => castVec (size_sK OC IC KH KW) (CtxVec.get (Γ := Γ) (s := sK OC IC KH KW) kernelIdx x)

  /-- Project the bias parameter from a `CtxVec`, casting to the flattened `Vec` layout. -/
  private def projB {Γ : List Shape} {OC : Nat} (biasIdx : Idx Γ (sB OC)) : CtxVec Γ → Vec OC :=
    fun x => castVec (size_sB OC) (CtxVec.get (Γ := Γ) (s := sB OC) biasIdx x)

  /-- Project the input from a `CtxVec`, casting to the flattened `Vec` layout. -/
  private def projX
      {Γ : List Shape} {IC IH IW : Nat}
      (inputIdx : Idx Γ (sX IC IH IW)) :
      CtxVec Γ → Vec (IC * (IH * IW)) :=
    fun x => castVec (size_sX IC IH IW) (CtxVec.get (Γ := Γ) (s := sX IC IH IW) inputIdx x)

  /-- `projK` as a continuous linear map. -/
  private def projKCLM
      {Γ : List Shape} {OC IC KH KW : Nat}
      (kernelIdx : Idx Γ (sK OC IC KH KW)) :
      CtxVec Γ →L[ℝ] Vec (OC * (IC * (KH * KW))) :=
    (Graph.castCLM (h := size_sK OC IC KH KW)).comp (CtxVec.getCLM (Γ := Γ) (s := sK OC IC KH KW)
      kernelIdx)

  /-- `projB` as a continuous linear map. -/
  private def projBCLM {Γ : List Shape} {OC : Nat} (biasIdx : Idx Γ (sB OC)) :
      CtxVec Γ →L[ℝ] Vec OC :=
    (Graph.castCLM (h := size_sB OC)).comp (CtxVec.getCLM (Γ := Γ) (s := sB OC) biasIdx)

  /-- `projX` as a continuous linear map. -/
  private def projXCLM
      {Γ : List Shape} {IC IH IW : Nat}
      (inputIdx : Idx Γ (sX IC IH IW)) :
      CtxVec Γ →L[ℝ] Vec (IC * (IH * IW)) :=
    (Graph.castCLM (h := size_sX IC IH IW)).comp (CtxVec.getCLM (Γ := Γ) (s := sX IC IH IW)
      inputIdx)

  /-- Conv2D’s bilinear map, packaged as `kernel →L (input →L output)`. -/
  private def Bmul
      {IC OC KH KW stride padding IH IW : Nat}
      (OH OW : Nat) :
      Vec (OC * (IC * (KH * KW))) →L[ℝ] Vec (IC * (IH * IW)) →L[ℝ] Vec (OC * (OH * OW)) :=
    convBilin (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding)
      (IH := IH) (IW := IW)
      OH OW

  /-- Bias broadcast map, packaged as a continuous linear map. -/
  private def Bbias {OC OH OW : Nat} : Vec OC →L[ℝ] Vec (OC * (OH * OW)) :=
    biasBroadcastCLM (OC := OC) (OH := OH) (OW := OW)

  /-- Flattened Conv2D forward map, reading kernel/bias/input out of the `CtxVec` by index. -/
  private def forwardVec
      {Γ : List Shape} {IC OC KH KW stride padding IH IW : Nat}
      (kernelIdx : Idx Γ (sK OC IC KH KW))
      (biasIdx : Idx Γ (sB OC))
      (inputIdx : Idx Γ (sX IC IH IW)) :
      CtxVec Γ → Vec (Shape.size (sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding :=
        padding))
        (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)))) :=
    fun x =>
      let OH := outH (IH := IH) (KH := KH) (stride := stride) (padding := padding)
      let OW := outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)
      castVec (size_sY OC OH OW).symm <|
        (Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding)
          (IH := IH) (IW := IW)
            OH OW (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx x))
          (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x) +
        (Bbias (OC := OC) (OH := OH) (OW := OW) (projB (Γ := Γ) (OC := OC) biasIdx x))

  /-- Derivative of `forwardVec`, as a continuous linear map valued in `CtxVec Γ →L[ℝ] _`. -/
  private def derivCLM
      {Γ : List Shape} {IC OC KH KW stride padding IH IW : Nat}
      (kernelIdx : Idx Γ (sK OC IC KH KW))
      (biasIdx : Idx Γ (sB OC))
      (inputIdx : Idx Γ (sX IC IH IW)) :
      CtxVec Γ → (CtxVec Γ →L[ℝ] Vec (Shape.size (sY OC (outH (IH := IH) (KH := KH) (stride :=
        stride) (padding := padding))
        (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding))))) :=
    fun x =>
      let OH := outH (IH := IH) (KH := KH) (stride := stride) (padding := padding)
      let OW := outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)
      let Bmul0 :=
        Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding) (IH
          := IH) (IW := IW) OH OW
      let D0 : CtxVec Γ →L[ℝ] Vec (OC * (OH * OW)) :=
        (Bmul0.precompR (CtxVec Γ) (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW)
          kernelIdx x)
            (projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx)) +
          (Bmul0.precompL (CtxVec Γ)
            (projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx)
            (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x)) +
          (Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC := OC) biasIdx)
      (Graph.castCLM (h := (size_sY OC OH OW).symm)).comp D0

  /-- Proof-only Conv2D node whose VJP is `(fderiv forward)†`. -/
  private def node
      {Γ : List Shape}
      {IC OC KH KW stride padding IH IW : Nat}
      (kernelIdx : Idx Γ (sK OC IC KH KW))
      (biasIdx   : Idx Γ (sB OC))
      (inputIdx  : Idx Γ (sX IC IH IW)) :
      Node Γ (sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding := padding))
        (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding))) :=
  by
    classical
    let deriv0 := derivCLM (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
      (padding := padding)
      (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx
    refine
      Node.ofVec (Γ := Γ)
        (τ := sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding := padding))
          (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)))
        (f := forwardVec (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
          (padding := padding)
          (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx)
        (jvp := fun x dx => (deriv0 x) dx)
        (vjp := fun x δ => (deriv0 x).adjoint δ)
        (correct_inner := by
          intro x dx δ
          simpa [deriv0] using
            (ContinuousLinearMap.adjoint_inner_right (A := deriv0 x) (x := dx) (y := δ)).symm)

  /-- Proof-only Conv2D node whose VJP is `Spec.conv2d_backward_spec` (runtime-style). -/
  private def nodeSpecBackward
      {Γ : List Shape}
      {IC OC KH KW stride padding IH IW : Nat}
      {h1 : IC ≠ 0} {h2 : KH ≠ 0} {h3 : KW ≠ 0}
      (kernelIdx : Idx Γ (sK OC IC KH KW))
      (biasIdx   : Idx Γ (sB OC))
      (inputIdx  : Idx Γ (sX IC IH IW)) :
      Node Γ (sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding := padding))
        (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding))) :=
  by
    classical
    let deriv0 := derivCLM (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
      (padding := padding)
      (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx
    refine
      Node.ofVec (Γ := Γ)
        (τ := sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding := padding))
          (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)))
        (f := forwardVec (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
          (padding := padding)
          (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx)
        (jvp := fun x dx => (deriv0 x) dx)
        (vjp := fun x δ =>
          let kRaw := projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx x
          let bRaw := projB (Γ := Γ) (OC := OC) biasIdx x
          let xRaw := projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x
          let kShape : Vec (Shape.size (sK OC IC KH KW)) := castVec (size_sK OC IC KH KW).symm kRaw
          let bShape : Vec (Shape.size (sB OC)) := castVec (size_sB OC).symm bRaw
          let xShape : Vec (Shape.size (sX IC IH IW)) := castVec (size_sX IC IH IW).symm xRaw
          let kernelT : Tensor ℝ (sK OC IC KH KW) := ofVecT (s := sK OC IC KH KW) kShape
          let biasT : Tensor ℝ (sB OC) := ofVecT (s := sB OC) bShape
          let inputT : Tensor ℝ (sX IC IH IW) := ofVecT (s := sX IC IH IW) xShape
          let layer : Spec.Conv2DSpec IC OC KH KW stride padding ℝ h1 h2 h3 :=
            { kernel := kernelT, bias := biasT }
          let δT : Tensor ℝ (sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding :=
            padding))
            (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding))) :=
              ofVecT (s := sY OC (outH (IH := IH) (KH := KH) (stride := stride) (padding :=
                padding))
                (outW (IW := IW) (KW := KW) (stride := stride) (padding := padding))) δ
          let grads := Spec.conv2dBackwardSpec (α := ℝ) (layer := layer) (input := inputT)
            (grad_output := δT)
          CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t := grads.1)) +
            (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)) +
              CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t := grads.2.2))))
        (correct_inner := by
          intro x dx δ
          -- unpack raw vectors and build the corresponding spec tensors
          let kRaw := projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx x
          let bRaw := projB (Γ := Γ) (OC := OC) biasIdx x
          let xRaw := projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x
          let dkRaw := projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx dx
          let dbRaw := projB (Γ := Γ) (OC := OC) biasIdx dx
          let dxRaw := projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx dx
          let kShape : Vec (Shape.size (sK OC IC KH KW)) := castVec (size_sK OC IC KH KW).symm kRaw
          let bShape : Vec (Shape.size (sB OC)) := castVec (size_sB OC).symm bRaw
          let xShape : Vec (Shape.size (sX IC IH IW)) := castVec (size_sX IC IH IW).symm xRaw
          let dkShape : Vec (Shape.size (sK OC IC KH KW)) := castVec (size_sK OC IC KH KW).symm
            dkRaw
          let dbShape : Vec (Shape.size (sB OC)) := castVec (size_sB OC).symm dbRaw
          let dxShape : Vec (Shape.size (sX IC IH IW)) := castVec (size_sX IC IH IW).symm dxRaw
          let kernelT : Tensor ℝ (sK OC IC KH KW) := ofVecT (s := sK OC IC KH KW) kShape
          let biasT : Tensor ℝ (sB OC) := ofVecT (s := sB OC) bShape
          let inputT : Tensor ℝ (sX IC IH IW) := ofVecT (s := sX IC IH IW) xShape
          let dKernelT : Tensor ℝ (sK OC IC KH KW) := ofVecT (s := sK OC IC KH KW) dkShape
          let dBiasT : Tensor ℝ (sB OC) := ofVecT (s := sB OC) dbShape
          let dInputT : Tensor ℝ (sX IC IH IW) := ofVecT (s := sX IC IH IW) dxShape
          let layer : Spec.Conv2DSpec IC OC KH KW stride padding ℝ h1 h2 h3 :=
            { kernel := kernelT, bias := biasT }
          let layerK : Spec.Conv2DSpec IC OC KH KW stride padding ℝ h1 h2 h3 :=
            { kernel := dKernelT, bias := fill (0 : ℝ) (sB OC) }
          let layer0 : Spec.Conv2DSpec IC OC KH KW stride padding ℝ h1 h2 h3 :=
            { kernel := kernelT, bias := fill (0 : ℝ) (sB OC) }
          let OH := outH (IH := IH) (KH := KH) (stride := stride) (padding := padding)
          let OW := outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)
          let δT : Tensor ℝ (sY OC OH OW) := ofVecT (s := sY OC OH OW) δ
          let grads := Spec.conv2dBackwardSpec (α := ℝ) (layer := layer) (input := inputT)
            (grad_output := δT)

          -- Rewrite the node JVP into the sum of the three spec-level JVP components.
          have hjvp :
                    (deriv0 x) dx
                      =
                    (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec (sY OC
                      OH OW).size) +
                      ((toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH)
                        (outW := OW) dBiasT) :
                          Vec (sY OC OH OW).size) +
                        (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) : Vec (sY
                          OC OH OW).size)) := by
              -- `deriv0` expands to the bilinear JVP in `(dKernel, dInput)` plus bias.
              have hbilin :
                    (deriv0 x) dx
                      =
                    castVec (size_sY OC OH OW).symm
                      (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                        (padding := padding)
                            (IH := IH) (IW := IW) OH OW dkRaw) xRaw) +
                        (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                          (padding := padding)
                              (IH := IH) (IW := IW) OH OW kRaw) dxRaw) +
                          (Bbias (OC := OC) (OH := OH) (OW := OW) dbRaw))) := by
                        -- Avoid a large `simp` on `derivCLM` by first simplifying the three context
                        -- projections.
                        have hKCLM :
                            projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx
                              dx = dkRaw := by
                          simp [projKCLM, dkRaw, projK, ContinuousLinearMap.comp_apply]
                        have hXCLM :
                            projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx dx = dxRaw
                              := by
                          simp [projXCLM, dxRaw, projX, ContinuousLinearMap.comp_apply]
                        have hBCLM :
                            projBCLM (Γ := Γ) (OC := OC) biasIdx dx = dbRaw := by
                          simp [projBCLM, dbRaw, projB, ContinuousLinearMap.comp_apply]
                        -- Now unfold `deriv0`/`derivCLM` and evaluate at `dx`.
                        simp [deriv0, derivCLM, OH, OW, ContinuousLinearMap.comp_apply,
                          ContinuousLinearMap.add_apply,
                          ContinuousLinearMap.precompL_apply, hKCLM, hXCLM, hBCLM, add_assoc]
                        -- Normalize the remaining `castVec`/associativity/commutativity and rewrite
                        -- the forward projections.
                        have hkRaw :
                            projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx x =
                              kRaw := rfl
                        have hxRaw :
                            projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x = xRaw := rfl
                        have hcast_add {n m : Nat} (h : n = m) (u v : Vec n) :
                            castVec h (u + v) = castVec h u + castVec h v := by
                          simp
                        -- Rewrite the remaining rearranged cast/sum into the target form.
                        rw [hkRaw, hxRaw]
                        let hOut : OC * (OH * OW) = (sY OC OH OW).size := (size_sY OC OH OW).symm
                        let u :=
                          (Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                            (padding := padding)
                                (IH := IH) (IW := IW) OH OW dkRaw) xRaw
                        let v :=
                          (Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                            (padding := padding)
                                (IH := IH) (IW := IW) OH OW kRaw) dxRaw
                        let w := (Bbias (OC := OC) (OH := OH) (OW := OW) dbRaw)
                        -- The remaining goal is just `a + (b + c) = b + (a + c)`.
                        simpa using
                          (add_left_comm (a := castVec hOut v) (b := castVec hOut u) (c := castVec
                            hOut w))

              -- Convert the three vector terms to `toVecT` of the corresponding spec tensors using
              -- the bridge lemmas.
              have hK :
                  castVec (size_sY OC OH OW).symm
                    (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding
                      := padding)
                        (IH := IH) (IW := IW) OH OW dkRaw) xRaw))
                    =
                  toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) := by
                have hcast :=
                  cast_toVecT_conv2d_spec_noBias_eq_convNoBiasVec
                    (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
                      padding) (IH := IH)
                    (IW := IW) (h1 := h1) (h2 := h2) (h3 := h3) dkRaw xRaw
                have hto :
                    toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) =
                      castVec (size_sY OC OH OW).symm
                        (convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW)
                          (stride := stride) (padding := padding) (IH := IH) (IW := IW) dkRaw xRaw
                            OH OW) := by
                  have h' := congrArg (castVec (size_sY OC OH OW).symm) hcast
                  simpa [castVec_castVec, castVec_rfl] using h'
                calc
                  castVec (size_sY OC OH OW).symm
                      (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                        (padding := padding)
                            (IH := IH) (IW := IW) OH OW dkRaw) xRaw))
                      =
                    castVec (size_sY OC OH OW).symm
                      (convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW)
                        (stride := stride) (padding := padding) (IH := IH) (IW := IW) dkRaw xRaw OH
                          OW) := by
                    simp [Bmul]
                  _ = toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) := by
                    simpa using hto.symm

              have hX :
                  castVec (size_sY OC OH OW).symm
                    (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding
                      := padding)
                        (IH := IH) (IW := IW) OH OW kRaw) dxRaw))
                    =
                  toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) := by
                have hcast :=
                  cast_toVecT_conv2d_spec_noBias_eq_convNoBiasVec
                    (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
                      padding) (IH := IH)
                    (IW := IW) (h1 := h1) (h2 := h2) (h3 := h3) kRaw dxRaw
                have hto :
                    toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) =
                      castVec (size_sY OC OH OW).symm
                        (convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW)
                          (stride := stride) (padding := padding) (IH := IH) (IW := IW) kRaw dxRaw
                            OH OW) := by
                  have h' := congrArg (castVec (size_sY OC OH OW).symm) hcast
                  simpa [castVec_castVec, castVec_rfl] using h'
                calc
                  castVec (size_sY OC OH OW).symm
                      (((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                        (padding := padding)
                            (IH := IH) (IW := IW) OH OW kRaw) dxRaw))
                      =
                    castVec (size_sY OC OH OW).symm
                      (convNoBiasVec (IC := IC) (OC := OC) (KH := KH) (KW := KW)
                        (stride := stride) (padding := padding) (IH := IH) (IW := IW) kRaw dxRaw OH
                          OW) := by
                    simp [Bmul]
                  _ = toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) := by
                    simpa using hto.symm

              have hB :
                  castVec (size_sY OC OH OW).symm (Bbias (OC := OC) (OH := OH) (OW := OW) dbRaw)
                    =
                  toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW
                    := OW) dBiasT) := by
                have hcast := cast_toVecT_biasBroadcast_eq_biasBroadcastVec (OC := OC) (OH := OH)
                  (OW := OW) dbRaw
                have hto :
                    toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH)
                      (outW := OW) dBiasT) =
                      castVec (size_sY OC OH OW).symm (biasBroadcastVec (OC := OC) (OH := OH) (OW :=
                        OW) dbRaw) := by
                  have h' := congrArg (castVec (size_sY OC OH OW).symm) hcast
                  simpa [castVec_castVec, castVec_rfl] using h'
                calc
                  castVec (size_sY OC OH OW).symm (Bbias (OC := OC) (OH := OH) (OW := OW) dbRaw)
                      =
                    castVec (size_sY OC OH OW).symm (biasBroadcastVec (OC := OC) (OH := OH) (OW :=
                      OW) dbRaw) := by
                    simp [Bbias]
                  _ =
                    toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH)
                      (outW := OW) dBiasT) := by
                    simpa using hto.symm

              -- assemble
              let a :=
                ((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
                  padding)
                    (IH := IH) (IW := IW) OH OW dkRaw) xRaw)
              let b :=
                ((Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
                  padding)
                    (IH := IH) (IW := IW) OH OW kRaw) dxRaw)
              let c := (Bbias (OC := OC) (OH := OH) (OW := OW) dbRaw)
              have hadd :
                  castVec (size_sY OC OH OW).symm (a + (b + c)) =
                    castVec (size_sY OC OH OW).symm a +
                      (castVec (size_sY OC OH OW).symm b + castVec (size_sY OC OH OW).symm c) := by
                ext i
                simp [a, b, c]
              calc
                (deriv0 x) dx
                    = castVec (size_sY OC OH OW).symm (a + (b + c)) := by
                      simpa [a, b, c, add_assoc] using hbilin
                _ = castVec (size_sY OC OH OW).symm a +
                      (castVec (size_sY OC OH OW).symm b + castVec (size_sY OC OH OW).symm c) :=
                        hadd
                _ =
                      (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec (sY OC
                        OH OW).size) +
                        ((toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH :=
                          OH) (outW := OW) dBiasT) :
                            Vec (sY OC OH OW).size) +
                          (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) : Vec
                            (sY OC OH OW).size)) := by
                      have hKa : castVec (size_sY OC OH OW).symm a =
                          (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec
                            (sY OC OH OW).size) := by
                        simpa [a] using hK
                      have hXb : castVec (size_sY OC OH OW).symm b =
                          (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) : Vec
                            (sY OC OH OW).size) := by
                        simpa [b] using hX
                      have hBc : castVec (size_sY OC OH OW).symm c =
                          toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH :=
                            OH) (outW := OW) dBiasT) := by
                        simpa [c] using hB
                      -- Rewrite each summand, then reorder the inner two terms.
                      calc
                        castVec (size_sY OC OH OW).symm a +
                              (castVec (size_sY OC OH OW).symm b + castVec (size_sY OC OH OW).symm
                                c)
                            =
                          (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec
                            (sY OC OH OW).size) +
                              ((toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) :
                                Vec (sY OC OH OW).size) +
                                toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH
                                  := OH) (outW := OW) dBiasT)) := by
                            simp [hKa, hXb, hBc]
                        _ =
                          (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec
                            (sY OC OH OW).size) +
                              (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH
                                := OH) (outW := OW) dBiasT) +
                                (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) :
                                  Vec (sY OC OH OW).size)) := by
                            simp [add_left_comm, add_comm]

          -- Convert inner products to tensor dots and apply the dot-bridge lemma.
          have hleft :
                inner ℝ ((deriv0 x) dx) δ
                  =
                dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) δT
                  +
                dot (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC)
                  (outH := OH) (outW := OW) dBiasT) δT
                  +
                dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) δT := by
            have hKdot :
                inner ℝ (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)) δ
                  =
                dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) δT := by
              simpa [δT, toVecT_ofVecT] using
                (dot_eq_inner_toVecT (a := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) (b :=
                  δT)).symm
            have hBdot :
                inner ℝ (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH)
                  (outW := OW) dBiasT)) δ
                  =
                dot (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                  dBiasT) δT := by
              simpa [δT, toVecT_ofVecT] using
                (dot_eq_inner_toVecT
                  (a := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                    dBiasT)
                  (b := δT)).symm
            have hXdot :
                inner ℝ (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)) δ
                  =
                dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) δT := by
              simpa [δT, toVecT_ofVecT] using
                (dot_eq_inner_toVecT (a := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) (b
                  := δT)).symm
            -- Rewrite `((deriv0 x) dx)` using `hjvp`, split the inner product across additions,
            -- then convert each term.
            calc
              inner ℝ ((deriv0 x) dx) δ
                  =
                inner ℝ
                  ((toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) : Vec (sY OC OH
                    OW).size) +
                    ((toVecT
                        (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW
                          := OW) dBiasT) :
                        Vec (sY OC OH OW).size) +
                      (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) : Vec (sY
                        OC OH OW).size)))
                  δ := by
                    simp [hjvp]
              _ =
                inner ℝ (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)) δ +
                  (inner ℝ (toVecT (t := Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH :=
                    OH) (outW := OW) dBiasT)) δ +
                    inner ℝ (toVecT (t := Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)) δ)
                      := by
                    simp [inner_add_left]
              _ =
                dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) δT +
                  dot (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                    dBiasT) δT +
                  dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) δT := by
                    simp [hKdot, hBdot, hXdot, add_assoc]

          -- `conv2d_backward_spec_dot` rewrites the dot of the full JVP sum.
          have hdot :
              dot (addSpec
                    (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)
                    (addSpec
                      (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                        dBiasT)
                      (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)))
                  δT
                =
              dot dKernelT grads.1 + dot dBiasT grads.2.1 + dot dInputT grads.2.2 := by
            -- apply the established dot-bridge theorem
            simpa [Proofs.Autograd.Conv2D.outH, Proofs.Autograd.Conv2D.outW, outH, outW, layerK,
              layer0, grads] using
              (Proofs.Autograd.Conv2D.conv2d_backward_spec_dot
                (layer := layer) (input := inputT) (δ := δT) (dKernel := dKernelT) (dBias := dBiasT)
                  (dInput := dInputT))

          have hleft' :
              inner ℝ ((deriv0 x) dx) δ
                =
              dot (addSpec
                    (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)
                    (addSpec
                      (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                        dBiasT)
                      (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)))
                  δT := by
            -- turn the sum of three dots back into a dot of the nested sum (avoids a large `simp`
            -- call)
            calc
              inner ℝ ((deriv0 x) dx) δ
                  =
                dot (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT) δT +
                  dot (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                    dBiasT) δT +
                  dot (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT) δT := hleft
              _ =
                dot (addSpec
                      (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)
                      (addSpec
                        (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                          dBiasT)
                        (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)))
                    δT := by
                  simp [dot_add_left, add_assoc]

          -- Right side: context dot with the assembled VJP equals the sum of three dots.
          have hright :
              inner ℝ dx
                  (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t := grads.1)) +
                    (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)) +
                      CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t := grads.2.2))))
                =
              dot dKernelT grads.1 + dot dBiasT grads.2.1 + dot dInputT grads.2.2 := by
            -- expand the inner across the context sum, and use `CtxVec.inner_get_single` +
            -- `dot_eq_inner_toVecT`
            have hk :
                inner ℝ dx (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t :=
                  grads.1)))
                  =
                dot dKernelT grads.1 := by
              have h1' := (CtxVec.inner_get_single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx dx
                (toVecT (t := grads.1)))
              -- rewrite `CtxVec.get` in terms of `dkShape`
              have hget : CtxVec.get (Γ := Γ) (s := sK OC IC KH KW) kernelIdx dx = dkShape := by
                simp [dkShape, dkRaw, projK, castVec_castVec]
              -- dot = inner after vectorization
              simp [h1', hget, dKernelT, dkShape, dot_eq_inner_toVecT]
            have hb :
                inner ℝ dx (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)))
                  =
                dot dBiasT grads.2.1 := by
              have h1' := (CtxVec.inner_get_single (Γ := Γ) (s := sB OC) biasIdx dx (toVecT (t :=
                grads.2.1)))
              have hget : CtxVec.get (Γ := Γ) (s := sB OC) biasIdx dx = dbShape := by
                simp [dbShape, dbRaw, projB, castVec_castVec]
              simp [h1', hget, dBiasT, dbShape, dot_eq_inner_toVecT]
            have hx :
                inner ℝ dx (CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t :=
                  grads.2.2)))
                  =
                dot dInputT grads.2.2 := by
              have h1' := (CtxVec.inner_get_single (Γ := Γ) (s := sX IC IH IW) inputIdx dx (toVecT
                (t := grads.2.2)))
              have hget : CtxVec.get (Γ := Γ) (s := sX IC IH IW) inputIdx dx = dxShape := by
                simp [dxShape, dxRaw, projX, castVec_castVec]
              simp [h1', hget, dInputT, dxShape, dot_eq_inner_toVecT]
            -- combine the pieces (avoid a heavy `simp` with commutativity on large terms)
            calc
              inner ℝ dx
                  (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t := grads.1)) +
                    (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)) +
                      CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t := grads.2.2))))
                  =
                inner ℝ dx (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t :=
                  grads.1))) +
                  inner ℝ dx
                    (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)) +
                      CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t := grads.2.2)))
                        := by
                  simp [inner_add_right]
              _ =
                inner ℝ dx (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t :=
                  grads.1))) +
                  (inner ℝ dx (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t :=
                    grads.2.1))) +
                    inner ℝ dx (CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t :=
                      grads.2.2)))) := by
                  simp [inner_add_right]
              _ =
                dot dKernelT grads.1 + (dot dBiasT grads.2.1 + dot dInputT grads.2.2) := by
                  simp [hk, hb, hx]
              _ = dot dKernelT grads.1 + dot dBiasT grads.2.1 + dot dInputT grads.2.2 := by
                  simp [add_assoc]

          -- close the goal
          calc
            inner ℝ ((deriv0 x) dx) δ
                =
              dot (addSpec
                    (Spec.conv2dSpec (α := ℝ) (layer := layerK) inputT)
                    (addSpec
                      (Proofs.Autograd.Conv2D.biasBroadcast (outC := OC) (outH := OH) (outW := OW)
                        dBiasT)
                      (Spec.conv2dSpec (α := ℝ) (layer := layer0) dInputT)))
                  δT := hleft'
            _ = dot dKernelT grads.1 + dot dBiasT grads.2.1 + dot dInputT grads.2.2 := hdot
            _ = inner ℝ dx
                  (CtxVec.single (Γ := Γ) (s := sK OC IC KH KW) kernelIdx (toVecT (t := grads.1)) +
                    (CtxVec.single (Γ := Γ) (s := sB OC) biasIdx (toVecT (t := grads.2.1)) +
                      CtxVec.single (Γ := Γ) (s := sX IC IH IW) inputIdx (toVecT (t := grads.2.2))))
                        := by
                    simpa using hright.symm)

    /-- Proof that `node` satisfies `NodeFDerivCorrect`, using the explicit `derivCLM` derivative.
      -/
    private def nodeFderiv
        {Γ : List Shape}
        {IC OC KH KW stride padding IH IW : Nat}
        (kernelIdx : Idx Γ (sK OC IC KH KW))
        (biasIdx   : Idx Γ (sB OC))
        (inputIdx  : Idx Γ (sX IC IH IW)) :
        NodeFDerivCorrect (node (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW)
          (stride := stride) (padding := padding) (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx)
            :=
    by
      classical
      let OH := outH (IH := IH) (KH := KH) (stride := stride) (padding := padding)
      let OW := outW (IW := IW) (KW := KW) (stride := stride) (padding := padding)
      let Bmul0 :=
        Bmul (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding := padding)
          (IH := IH) (IW := IW) OH OW
      let deriv0 :=
        derivCLM (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
          (padding := padding) (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx

      refine
        { deriv := deriv0
          hasFDerivAt := ?_
          jvp_eq := ?_ }
      · intro xV
        -- Projection maps are linear.
        have hKproj :
            HasFDerivAt (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx)
              (projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx) xV := by
          have h :=
            (projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx).hasFDerivAt (x
              := xV)
          have hfun :
              (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx) =
                fun x => (projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx) x
                  := by
            funext x
            simp [projK, projKCLM, CtxVec.getCLM_apply]
          exact h.congr_of_eventuallyEq hfun.eventuallyEq
        have hXproj :
            HasFDerivAt (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx)
              (projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx) xV := by
          have h :=
            (projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx).hasFDerivAt (x := xV)
          have hfun :
              (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx) =
                fun x => (projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx) x := by
            funext x
            simp [projX, projXCLM, CtxVec.getCLM_apply]
          exact h.congr_of_eventuallyEq hfun.eventuallyEq
        have hBproj :
            HasFDerivAt (projB (Γ := Γ) (OC := OC) biasIdx) (projBCLM (Γ := Γ) (OC := OC) biasIdx)
              xV := by
          have h := (projBCLM (Γ := Γ) (OC := OC) biasIdx).hasFDerivAt (x := xV)
          have hfun :
              (projB (Γ := Γ) (OC := OC) biasIdx) = fun x => (projBCLM (Γ := Γ) (OC := OC) biasIdx)
                x := by
            funext x
            simp [projB, projBCLM, CtxVec.getCLM_apply]
          exact h.congr_of_eventuallyEq hfun.eventuallyEq

        -- Bilinear part.
        have hbilin :
            HasFDerivAt
              (fun x =>
                (Bmul0 (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx x))
                  (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx x))
              ((Bmul0.precompR (CtxVec Γ)
                  (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx xV)
                  (projXCLM (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx)) +
                (Bmul0.precompL (CtxVec Γ)
                  (projKCLM (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx)
                  (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx xV))) xV :=
          ContinuousLinearMap.hasFDerivAt_of_bilinear (B := Bmul0) (hf := hKproj) (hg := hXproj)

        -- Bias term is linear.
        have hbias :
            HasFDerivAt (fun x => (Bbias (OC := OC) (OH := OH) (OW := OW)) (projB (Γ := Γ) (OC :=
              OC) biasIdx x))
              ((Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC := OC) biasIdx))
                xV := by
          -- `Bbias` is a CLM, so `Bbias ∘ projBCLM` is a CLM.
          have hCLM :
              HasFDerivAt
                (fun x =>
                  ((Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC := OC)
                    biasIdx)) x)
                ((Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC := OC)
                  biasIdx)) xV :=
            ((Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC := OC)
              biasIdx)).hasFDerivAt (x := xV)
          have hfun :
              (fun x => (Bbias (OC := OC) (OH := OH) (OW := OW)) (projB (Γ := Γ) (OC := OC) biasIdx
                x)) =
                fun x => ((Bbias (OC := OC) (OH := OH) (OW := OW)).comp (projBCLM (Γ := Γ) (OC :=
                  OC) biasIdx)) x := by
            funext x
            simp [projB, projBCLM, ContinuousLinearMap.comp_apply, Graph.castCLM]
          exact hCLM.congr_of_eventuallyEq hfun.eventuallyEq

        -- Combine and cast output size.
        have hadd := hbilin.add hbias
        have hcast :
            HasFDerivAt (fun y : Vec (OC * (OH * OW)) => castVec (size_sY OC OH OW).symm y)
              (Graph.castCLM (h := (size_sY OC OH OW).symm))
              ((Bmul0 (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx xV))
                  (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx xV) +
                (Bbias (OC := OC) (OH := OH) (OW := OW)) (projB (Γ := Γ) (OC := OC) biasIdx xV)) :=
                  by
          simpa [Graph.castCLM] using
            ((Graph.castCLM (h := (size_sY OC OH OW).symm)).hasFDerivAt
              (x := (Bmul0 (projK (Γ := Γ) (OC := OC) (IC := IC) (KH := KH) (KW := KW) kernelIdx
                xV))
                  (projX (Γ := Γ) (IC := IC) (IH := IH) (IW := IW) inputIdx xV) +
                (Bbias (OC := OC) (OH := OH) (OW := OW)) (projB (Γ := Γ) (OC := OC) biasIdx xV)))
        have hcomp := hcast.comp xV hadd

        -- Rewrite in terms of the node's `forwardVec`.
        simpa [node, Node.forwardVec_ofVec, forwardVec, deriv0, derivCLM, projK, projB, projX, Bmul,
          Bbias] using hcomp
      · intro xV dxV
        -- `jvpVec` is definitional to applying `deriv0 xV`.
        simp [node, Node.jvpVec_ofVec, deriv0]

    /-- `NodeFDerivCorrect` for the runtime-style Conv2D node (`vjp = conv2d_backward_spec`).

    This is the *analytic* assumption needed by the global theorem
    `Graph.backpropVec_eq_adjoint_fderiv`: it only talks about `forwardVec`/`jvpVec`, so it
    is identical to `node_fderiv` (the `vjp` choice is handled separately by `correct_inner`).
    -/
    private def nodeSpecBackwardFderiv
        {Γ : List Shape}
        {IC OC KH KW stride padding IH IW : Nat}
        {h1 : IC ≠ 0} {h2 : KH ≠ 0} {h3 : KW ≠ 0}
        (kernelIdx : Idx Γ (sK OC IC KH KW))
        (biasIdx   : Idx Γ (sB OC))
        (inputIdx  : Idx Γ (sX IC IH IW)) :
        NodeFDerivCorrect (nodeSpecBackward (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW)
          (stride := stride) (padding := padding) (IH := IH) (IW := IW) (h1 := h1) (h2 := h2) (h3 :=
            h3)
          kernelIdx biasIdx inputIdx) :=
    by
      classical
      let deriv0 :=
        derivCLM (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride) (padding :=
          padding) (IH := IH)
          (IW := IW) kernelIdx biasIdx inputIdx
      refine
        { deriv := deriv0
          hasFDerivAt := ?_
          jvp_eq := ?_ }
      · intro xV
        have h :=
          (nodeFderiv (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
            (padding := padding)
            (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx).hasFDerivAt xV
        have h' :
            HasFDerivAt
              (forwardVec (Γ := Γ) (IC := IC) (OC := OC) (KH := KH) (KW := KW) (stride := stride)
                (padding := padding)
                (IH := IH) (IW := IW) kernelIdx biasIdx inputIdx)
              (deriv0 xV) xV := by
          simpa [node, Node.forwardVec_ofVec, deriv0] using h
        simpa [nodeSpecBackward, Node.forwardVec_ofVec, deriv0] using h'
      · intro xV dxV
        simp [nodeSpecBackward, Node.jvpVec_ofVec, deriv0]

    end Conv2D

    end

    end TapeNodes
  end Autograd
  end Proofs
