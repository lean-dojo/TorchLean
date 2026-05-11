/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Layers.Conv

/-!
# Conv2D

Conv2D CROWN-IBP bounds in TorchLean.

We provide:
- Interval Bound Propagation (IBP) for Conv2D pre-activations
- A CROWN-IBP affine bound for Conv2D+ReLU, evaluated on a flattened input box

Design notes:
- For simplicity and generality, we flatten the 3D image input and the 3D conv output
  to 1D vectors when evaluating affine forms. This reuses AffineVec and its safe
  evaluation on input boxes.
- The conv linear operator is explicitly materialized as a matrix Wconv whose rows
  correspond to output positions and columns to input positions. This keeps the verifier simple and
  deterministic for the tensor sizes targeted by the CROWN operator layer.
-/

@[expose] public section

namespace NN.MLTheory.CROWN

open _root_.Spec
open _root_.Spec.Tensor

variable {α : Type} [Context α]

/-- Flatten a `Box` to a 1D box by flattening both endpoints. -/
def flattenBox {s : Shape} (B : Box α s) [Inhabited α] : Box α (.dim (Shape.size s) .scalar) :=
  { lo := Tensor.flattenSpec B.lo, hi := Tensor.flattenSpec B.hi }

/--
Interval Bound Propagation (IBP) for Conv2D pre-activations `y = conv(x, K) + b`.

This computes per-output-position min/max bounds by taking min/max of each product term.
-/
def ibpConv2d
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3)
  (xB : Box α (.dim inC (.dim inH (.dim inW .scalar)))) :
  Box α (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
    stride + 1) .scalar))) :=
  -- Compute lo/hi per output position independently
  let loT := Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Accumulate lower bound using min over each product term
        let total_lo : α :=
          (List.range inC).foldl (fun acc in_ch =>
            (List.range kH).foldl (fun acc di =>
              (List.range kW).foldl (fun acc dj =>
                -- Map kernel position to input coordinate with padding/stride
                let pi := i * stride + di
                let pj := j * stride + dj
                -- Convert padded coords to input coords if valid
                let valid_i := pi ≥ padding
                let valid_j := pj ≥ padding
                let ii := pi - padding
                let jj := pj - padding
                -- Only contribute if within input bounds
                if valid_i ∧ valid_j ∧ ii < inH ∧ jj < inW then
                  let xlo := getAtOrZero xB.lo [in_ch, ii, jj]
                  let xhi := getAtOrZero xB.hi [in_ch, ii, jj]
                  let a := getAtOrZero layer.kernel [out_ch, in_ch, di, dj]
                  let p1 := a * xlo
                  let p2 := a * xhi
                  let mn := if p1 > p2 then p2 else p1
                  acc + mn
                else acc
              ) acc
            ) acc
          ) 0
        addSpec (Tensor.scalar total_lo) (getAtSpec layer.bias out_ch)
      )
    )
  )
  let hiT := Tensor.dim (fun out_ch =>
    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        -- Accumulate upper bound using max over each product term
        let total_hi : α :=
          (List.range inC).foldl (fun acc in_ch =>
            (List.range kH).foldl (fun acc di =>
              (List.range kW).foldl (fun acc dj =>
                let pi := i * stride + di
                let pj := j * stride + dj
                let valid_i := pi ≥ padding
                let valid_j := pj ≥ padding
                let ii := pi - padding
                let jj := pj - padding
                if valid_i ∧ valid_j ∧ ii < inH ∧ jj < inW then
                  let xlo := getAtOrZero xB.lo [in_ch, ii, jj]
                  let xhi := getAtOrZero xB.hi [in_ch, ii, jj]
                  let a := getAtOrZero layer.kernel [out_ch, in_ch, di, dj]
                  let p1 := a * xlo
                  let p2 := a * xhi
                  let mx := if p1 > p2 then p1 else p2
                  acc + mx
                else acc
              ) acc
            ) acc
          ) 0
        addSpec (Tensor.scalar total_hi) (getAtSpec layer.bias out_ch)
      )
    )
  )
  { lo := loT, hi := hiT }

/--
Build the explicit Conv2D linear operator matrix `Wconv` mapping flat input to flat output.

Shapes:
- Input: `inC × inH × inW` (flat size `inC*inH*inW`)
- Output (pre-activation, without bias): `outC × outH × outW` (flat size `outC*outH*outW`)

Entry `Wconv[r,c]` is the contribution of input coordinate `c` to output coordinate `r`.
-/
def conv2dLinearMatrix
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3) :
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let inShape  := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  let nIn  := Shape.size inShape
  let nOut := Shape.size outShape
  Tensor α (.dim nOut (.dim nIn .scalar)) :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  -- Helpers to map flat indices to 3D and vice versa
  let decodeIn := fun (c : Nat) =>
    let c0 := c / (inH * inW)
    let r0 := c % (inH * inW)
    let i0 := r0 / inW
    let j0 := r0 % inW
    (c0, i0, j0)
  Tensor.dim (fun r =>
    -- Decode r to (out_ch, i, j)
    let oc := r.val / (outH * outW)
    let rs := r.val % (outH * outW)
    let i := rs / outW
    let j := rs % outW
    Tensor.dim (fun c =>
      let (in_ch, ii, jj) := decodeIn c.val
      -- Sum all kernel positions that map this input (in_ch,ii,jj) to output (oc,i,j)
      let coeff : α :=
        (List.range kH).foldl (fun acc di =>
          (List.range kW).foldl (fun acc dj =>
            let pi := i * stride + di
            let pj := j * stride + dj
            let valid_i := pi ≥ padding
            let valid_j := pj ≥ padding
            let iii := pi - padding
            let jjj := pj - padding
            -- Check alignment and bounds
            if valid_i ∧ valid_j ∧ iii = ii ∧ jjj = jj then
              let a := getAtOrZero layer.kernel [oc, in_ch, di, dj]
              acc + a
            else acc
          ) acc
        ) 0
      Tensor.scalar coeff))

/-- Row-wise scaling of a matrix by a vector: scale each row `i` by `v[i]`. -/
def matRowScaleSpec {m n : Nat}
  (A : Tensor α (.dim m (.dim n .scalar))) (v : Tensor α (.dim m .scalar)) :
  Tensor α (.dim m (.dim n .scalar)) :=
  match A, v with
  | Tensor.dim rows, Tensor.dim vec =>
    Tensor.dim (fun i =>
      match rows i, vec i with
      | Tensor.dim cols, Tensor.scalar vi =>
        Tensor.dim (fun j =>
          match cols j with
          | Tensor.scalar aij => Tensor.scalar (vi * aij)))

/-- Broadcast a per-channel bias vector across spatial positions, as a flattened output vector. -/
def conv2dBiasBroadcast [Zero α]
  {outC inH inW kH kW stride padding : ℕ}
  (bias : Tensor α (.dim outC .scalar)) :
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  Tensor α (.dim (Shape.size outShape) .scalar) :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  Tensor.dim (fun r =>
    let oc := r.val / (outH * outW)
    Tensor.scalar (getAtOrZero bias [oc]))

/--
Construct the CROWN-IBP affine form for `Conv2D` followed by `ReLU`, with respect to the flattened
  input.

This returns an `AffineVec (A,c)` so it can be composed with subsequent linear layers. Evaluation
on an input box can then be performed with `AffineVec.eval_on_box`.
-/
def crownConv2dAffineForm
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  [Inhabited α]
  (layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3)
  (xB : Box α (.dim inC (.dim inH (.dim inW .scalar)))) :
  AffineVec α (Shape.size (Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))))
    (Shape.size (Shape.dim outC (Shape.dim ((inH + 2 * padding - kH) / stride + 1) (Shape.dim ((inW
      + 2 * padding - kW) / stride + 1) Shape.scalar)))) :=
  let zB := ibpConv2d (α:=α) layer xB
  -- Build ReLU relax parameters per output position
  let relax :=
    match zB.lo, zB.hi with
    | Tensor.dim lo, Tensor.dim hi =>
      Tensor.dim (fun oc =>
        match lo oc, hi oc with
        | Tensor.dim vlo, Tensor.dim vhi =>
          Tensor.dim (fun i =>
            match vlo i, vhi i with
            | Tensor.dim wlo, Tensor.dim whi =>
              Tensor.dim (fun j =>
                match wlo j, whi j with
                | Tensor.scalar l, Tensor.scalar u =>
                    Tensor.scalar (Runtime.Ops.ReLU.relaxScalar (α:=α) l u))))
  -- Flatten relax to vector of slopes and biases in the same order as flatten_spec
  let slopeVec :=
    match relax with
    | Tensor.dim f1 =>
      -- shape: outC × outH × outW of ReLURelax → flatten to vector
      let flat := Tensor.flattenSpec (Tensor.dim (fun oc => Tensor.dim (fun i => Tensor.dim (fun j
        =>
        match f1 oc with
        | Tensor.dim g1 => match g1 i with
          | Tensor.dim g2 => match g2 j with
            | Tensor.scalar rp => Tensor.scalar rp.slope))))
      flat
  let biasVec :=
    match relax with
    | Tensor.dim f1 =>
      let flat := Tensor.flattenSpec (Tensor.dim (fun oc => Tensor.dim (fun i => Tensor.dim (fun j
        =>
        match f1 oc with
        | Tensor.dim g1 => match g1 i with
          | Tensor.dim g2 => match g2 j with
            | Tensor.scalar rp => Tensor.scalar rp.bias))))
      flat
  -- Build conv linear operator (pre-activation) and scale rows by slope
  let Wconv := conv2dLinearMatrix (α:=α) (inC:=inC) (outC:=outC) (kH:=kH) (kW:=kW)
    (stride:=stride) (padding:=padding) (inH:=inH) (inW:=inW) layer
  let A := matRowScaleSpec (α:=α) Wconv slopeVec
  -- Broadcast bias per output position and combine with ReLU bias
  let bconv := conv2dBiasBroadcast (α:=α) (outC:=outC) (inH:=inH) (inW:=inW) (kH:=kH) (kW:=kW)
    (stride:=stride) (padding:=padding) layer.bias
  let c := Tensor.addSpec (Tensor.mulSpec slopeVec bconv) biasVec
  -- Return affine (with flattened in/out dims)
  AffineVec.ofLinear (α:=α) A c

/--
Evaluate the Conv2D+ReLU CROWN-IBP affine bound on a flattened input box.

Returns a `Box` over the flattened conv output dimension.
-/
def crownConv2dAffineFlat
  {inC outC kH kW stride padding inH inW : ℕ}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  [Inhabited α]
  (layer : Spec.Conv2DSpec inC outC kH kW stride padding α h1 h2 h3)
  (xB : Box α (.dim inC (.dim inH (.dim inW .scalar)))) :
  Box α (.dim (outC * ((inH + 2 * padding - kH) / stride + 1) * ((inW + 2 * padding - kW) / stride +
    1)) .scalar) :=
  let outH := (inH + 2 * padding - kH) / stride + 1
  let outW := (inW + 2 * padding - kW) / stride + 1
  let inShape  := Shape.dim inC (Shape.dim inH (Shape.dim inW Shape.scalar))
  let outShape := Shape.dim outC (Shape.dim outH (Shape.dim outW Shape.scalar))
  -- Get affine form and evaluate on the flattened input box
  let aff := crownConv2dAffineForm (α:=α) (inC:=inC) (outC:=outC) (kH:=kH) (kW:=kW)
    (stride:=stride) (padding:=padding) (inH:=inH) (inW:=inW) layer xB
  let xBflat := flattenBox (α:=α) (s:=inShape) xB
  let yBflat := AffineVec.evalOnBox (α:=α) aff xBflat
  -- Reshape final flat output to the product-based shape promised in the return type
  let hOutSize : (Shape.dim (Shape.size outShape) Shape.scalar).size = (Shape.dim (outC * outH *
    outW) Shape.scalar).size := by
    simp [Shape.size, outShape, Nat.mul_assoc]
  { lo := Tensor.reshapeSpec (α:=α) (s₁:=.dim (Shape.size outShape) .scalar) (s₂:=.dim (outC * outH
    * outW) .scalar) yBflat.lo hOutSize
  , hi := Tensor.reshapeSpec (α:=α) (s₁:=.dim (Shape.size outShape) .scalar) (s₂:=.dim (outC * outH
    * outW) .scalar) yBflat.hi hOutSize }

end NN.MLTheory.CROWN
