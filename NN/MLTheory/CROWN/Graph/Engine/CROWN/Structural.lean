/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Activations

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Structural Operators

Affine propagation through permutations, matrix products, and elementwise products.
-/

/-- Permute the output coordinates of an affine bound when the output shape permutation is valid. -/
def permuteAffineOut {inDim outDim : Nat}
  (perm : Fin outDim → Fin outDim) (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  { A := Tensor.matrix fun i j => Spec.get2 aff.A (perm i) j
    c := Tensor.ofFn fun i => Tensor.getScalar aff.c (perm i) }

/-- Exactly transport affine bounds through a valid axis permutation. -/
def permuteFlatAffineBounds? (sourceShape : Shape) (perm : Array Nat)
    (bounds : FlatAffineBounds α) : Option (FlatAffineBounds α) := do
  let flatPerm ← flatAxisPermutation? sourceShape perm bounds.outDim
  pure
    { inDim := bounds.inDim
      outDim := bounds.outDim
      loAff := permuteAffineOut (α := α) flatPerm bounds.loAff
      hiAff := permuteAffineOut (α := α) flatPerm bounds.hiAff }

/-- Transfer affine rows through a concat without changing coefficients or constants. -/
def concatFlatAffineBounds? (layout : ConcatLayout) (bounds : Array (FlatAffineBounds α)) :
    Option (FlatAffineBounds α) := do
  unless bounds.size == layout.lengths.length do failure
  let first : FlatAffineBounds α ← bounds[0]?
  let parents : Fin layout.lengths.length → FlatAffineBounds α ←
    Tensor.Internal.sequenceFinM fun parent => bounds[parent.val]?
  if h : ∀ parent, (parents parent).inDim = first.inDim ∧
      (parents parent).outDim = (layout.parentShape parent).size then
    let lower : (parent : Fin layout.lengths.length) →
        AffineVec α first.inDim (layout.parentShape parent).size :=
      fun parent => castAffineOut (h parent).2 (castAffineIn (h parent).1 (parents parent).loAff)
    let upper : (parent : Fin layout.lengths.length) →
        AffineVec α first.inDim (layout.parentShape parent).size :=
      fun parent => castAffineOut (h parent).2 (castAffineIn (h parent).1 (parents parent).hiAff)
    let coefficientLo : Tensor α [layout.outputShape.size, first.inDim] :=
      Tensor.matrix fun i j =>
        let source := layout.flatEquiv.symm i
        Spec.get2 (lower source.1).A source.2 j
    let coefficientHi : Tensor α [layout.outputShape.size, first.inDim] :=
      Tensor.matrix fun i j =>
        let source := layout.flatEquiv.symm i
        Spec.get2 (upper source.1).A source.2 j
    pure
      { inDim := first.inDim
        outDim := layout.outputShape.size
        loAff := { A := coefficientLo, c := layout.concat fun parent => (lower parent).c }
        hiAff := { A := coefficientHi, c := layout.concat fun parent => (upper parent).c } }
  else
    none

namespace Internal

/-- Matmul affine propagation using McCormick-style product planes. -/
def propagateMatmulBounds
  (sA sB : Shape) (Bx By : FlatBox α)
  (aB bB : FlatAffineBounds α) :
  Option (FlatAffineBounds α) :=
  if hin : aB.inDim = bB.inDim then
    let inDim := aB.inDim
    let bLo : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.loAff
    let bHi : AffineVec α inDim bB.outDim :=
      castAffineIn (α:=α) (n:=bB.inDim) (n':=inDim) (m:=bB.outDim) hin.symm bB.hiAff
    let split (a : α) : α × α :=
      if a > 0 then (a, 0) else (0, a)
    match (OpContracts.matmulDims sA sB).toOption with
    | none => none
    | some dims =>
      let dimA := sA.size
      let dimB := sB.size
      let outDim := dims.outShape.size
      if Bx.dim = dimA ∧ aB.outDim = dimA then
        if By.dim = dimB ∧ bB.outDim = dimB then
          let termUpperCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL

          let termUpperConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            aXpos * xU + aXneg * xL + aYpos * yU + aYneg * yL + off

          let termLowerCoeff (aIdx bIdx inJ : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.A [aIdx, inJ]
            let xU := getAtOrZero aB.hiAff.A [aIdx, inJ]
            let yL := getAtOrZero bLo.A [bIdx, inJ]
            let yU := getAtOrZero bHi.A [bIdx, inJ]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU

          let termLowerConst (aIdx bIdx : Nat) : α :=
            let lx := getAtOrZero Bx.lo [aIdx]
            let ux := getAtOrZero Bx.hi [aIdx]
            let ly := getAtOrZero By.lo [bIdx]
            let uy := getAtOrZero By.hi [bIdx]
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let l1 := lx * cy + ly * cx - lx * ly
            let l2 := ux * cy + uy * cx - ux * uy
            let aX := if l1 > l2 then ly else uy
            let aY := if l1 > l2 then lx else ux
            let off := if l1 > l2 then (-(lx * ly)) else (-(ux * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            let xL := getAtOrZero aB.loAff.c [aIdx]
            let xU := getAtOrZero aB.hiAff.c [aIdx]
            let yL := getAtOrZero bLo.c [bIdx]
            let yU := getAtOrZero bHi.c [bIdx]
            aXpos * xL + aXneg * xU + aYpos * yL + aYneg * yU + off

          let A_hi : Tensor α [outDim, inDim] :=
            Tensor.dim (fun outI =>
              let t := outI.val
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range dims.inner).foldl (fun acc kk =>
                    acc + termUpperCoeff (dims.leftIndex t kk) (dims.rightIndex t kk) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_hi : Tensor α [outDim] :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let coeff :=
                (List.range dims.inner).foldl (fun acc kk =>
                  acc + termUpperConst (dims.leftIndex t kk) (dims.rightIndex t kk)
                ) 0
              Tensor.scalar coeff)

          let A_lo : Tensor α [outDim, inDim] :=
            Tensor.dim (fun outI =>
              let t := outI.val
              Tensor.dim (fun inJ =>
                let coeff :=
                  (List.range dims.inner).foldl (fun acc kk =>
                    acc + termLowerCoeff (dims.leftIndex t kk) (dims.rightIndex t kk) inJ.val
                  ) 0
                Tensor.scalar coeff))
          let c_lo : Tensor α [outDim] :=
            Tensor.dim (fun outI =>
              let t := outI.val
              let coeff :=
                (List.range dims.inner).foldl (fun acc kk =>
                  acc + termLowerConst (dims.leftIndex t kk) (dims.rightIndex t kk)
                ) 0
              Tensor.scalar coeff)

          some
            { inDim := inDim
              outDim := outDim
              loAff := { A := A_lo, c := c_lo }
              hiAff := { A := A_hi, c := c_hi } }
        else
          none
      else
        none
  else
    none

end Internal

/-- Propagate affine bounds through componentwise multiplication using per-coordinate product
planes. -/
def propagateMulElemBounds
  (Bx By : FlatBox α)
  (xB yB : FlatAffineBounds α)
  (houtX : xB.outDim = Bx.dim) (houtY : yB.outDim = By.dim) :
  Option (FlatAffineBounds α) :=
  -- Require equal vector lengths and equal input widths.
  if hdim : Bx.dim = By.dim then
    if hin : xB.inDim = yB.inDim then
      let n := Bx.dim
      let hyo : yB.outDim = n := Eq.trans houtY (Eq.symm hdim)
      let hBy : By.dim = n := by simpa [n] using (Eq.symm hdim)
      let ByLo : Tensor α [n] := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.lo
      let ByHi : Tensor α [n] := castDimScalar (α:=α) (n:=By.dim) (n':=n) hBy By.hi

      let xLo : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.loAff
      let xHi : AffineVec α xB.inDim n :=
        castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) (by simpa [n] using houtX)
          xB.hiAff
      let yLo0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.loAff
      let yHi0 : AffineVec α yB.inDim n :=
        castAffineOut (α:=α) (n:=yB.inDim) (m:=yB.outDim) (m':=n) hyo yB.hiAff
        let yLo : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yLo0
        let yHi : AffineVec α xB.inDim n :=
          castAffineIn (α:=α) (n:=yB.inDim) (n':=xB.inDim) (m:=n) hin.symm yHi0

        -- Helper to split a scalar coefficient into (pos, neg).
        let split (a : α) : α × α := if a > 0 then (a, 0) else (0,
          a)

        -- Build row-wise A/c for upper and lower using a single selected McCormick plane per
        -- component.
        let A_hi : Tensor α [n, xB.inDim] :=
          Tensor.matrix fun i j =>
            let lx := Tensor.getScalar Bx.lo i
            let ux := Tensor.getScalar Bx.hi i
            let ly := Tensor.getScalar ByLo i
            let uy := Tensor.getScalar ByHi i
            -- Choose the tighter upper plane at the interval center.
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            aXpos * Spec.get2 xHi.A i j + aXneg * Spec.get2 xLo.A i j +
              aYpos * Spec.get2 yHi.A i j + aYneg * Spec.get2 yLo.A i j
        let c_hi : Tensor α [n] :=
          Tensor.ofFn fun i =>
            let lx := Tensor.getScalar Bx.lo i
            let ux := Tensor.getScalar Bx.hi i
            let ly := Tensor.getScalar ByLo i
            let uy := Tensor.getScalar ByHi i
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let u1 := ux * cy + ly * cx - ux * ly
            let u2 := lx * cy + uy * cx - lx * uy
            let aX := if u1 < u2 then ly else uy
            let aY := if u1 < u2 then ux else lx
            let off := if u1 < u2 then (-(ux * ly)) else (-(lx * uy))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            aXpos * Tensor.getScalar xHi.c i + aXneg * Tensor.getScalar xLo.c i +
              aYpos * Tensor.getScalar yHi.c i + aYneg * Tensor.getScalar yLo.c i + off
        let A_lo : Tensor α [n, xB.inDim] :=
          Tensor.matrix fun i j =>
            let lx := Tensor.getScalar Bx.lo i
            let ux := Tensor.getScalar Bx.hi i
            let ly := Tensor.getScalar ByLo i
            let uy := Tensor.getScalar ByHi i
            -- Choose the tighter lower plane at the interval center.
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let l1 := ux * cy + uy * cx - ux * uy
            let l2 := lx * cy + ly * cx - lx * ly
            let aX := if l1 > l2 then uy else ly
            let aY := if l1 > l2 then ux else lx
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            -- Negative coefficients select the upper affine input bound.
            aXpos * Spec.get2 xLo.A i j + aXneg * Spec.get2 xHi.A i j +
              aYpos * Spec.get2 yLo.A i j + aYneg * Spec.get2 yHi.A i j
        let c_lo : Tensor α [n] :=
          Tensor.ofFn fun i =>
            let lx := Tensor.getScalar Bx.lo i
            let ux := Tensor.getScalar Bx.hi i
            let ly := Tensor.getScalar ByLo i
            let uy := Tensor.getScalar ByHi i
            let cx := (lx + ux) * (1 / 2)
            let cy := (ly + uy) * (1 / 2)
            let l1 := ux * cy + uy * cx - ux * uy
            let l2 := lx * cy + ly * cx - lx * ly
            let aX := if l1 > l2 then uy else ly
            let aY := if l1 > l2 then ux else lx
            let off := if l1 > l2 then (-(ux * uy)) else (-(lx * ly))
            let (aXpos, aXneg) := split aX
            let (aYpos, aYneg) := split aY
            aXpos * Tensor.getScalar xLo.c i + aXneg * Tensor.getScalar xHi.c i +
              aYpos * Tensor.getScalar yLo.c i + aYneg * Tensor.getScalar yHi.c i + off

        some
          { inDim := xB.inDim
            outDim := n
            loAff := { A := A_lo, c := c_lo }
            hiAff := { A := A_hi, c := c_hi } }
    else
      none
  else
    none


end NN.MLTheory.CROWN.Graph
