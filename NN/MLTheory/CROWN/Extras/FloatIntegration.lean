/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Rounding
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Models.Mlp
public import NN.Spec.Layers.Linear

/-!
# Floats integration for CROWN (optional)

This module bridges the CROWN bound-propagation development with the Floats infrastructure.
It provides rounded-arithmetic variants of a few bound-propagation helpers, intended for
experiments where you want the CROWN computations themselves to follow an explicit rounding model.

This is not required for the core CROWN proof layer and is grouped under
  `NN/MLTheory/CROWN/Extras/`.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Float

open _root_.Spec
open _root_.Spec.Tensor
open TorchLean.Floats
open NN.MLTheory.CROWN

/- Rounding choice (can be made configurable per phase) -/
variable {β : NeuralRadix} {fexp : ℤ → ℤ}
variable (rnd : ℝ → ℤ) [NeuralValidExp fexp] [NeuralValidRnd rnd]

/-- Rounded multiply-add helper: round(a*b) then round(acc + ·). -/
noncomputable def rFMA (a b acc : ℝ) : ℝ :=
  let p := neuralRound (β := β) (fexp := fexp) rnd (a * b)
  neuralRound (β := β) (fexp := fexp) rnd (acc + p)

/-- Rounded add -/
noncomputable def rAdd (x y : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (x + y)

/-- Rounded mul -/
noncomputable def rMul (x y : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (x * y)

/-- Rounded max of two reals (stable under rounding by evaluating in ℝ, then rounding). -/
noncomputable def rMax (x y : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (if x > y then x else y)

/-- Rounded min of two reals (stable under rounding by evaluating in ℝ, then rounding). -/
noncomputable def rMin (x y : ℝ) : ℝ :=
  neuralRound (β := β) (fexp := fexp) rnd (if x > y then y else x)

/-- Interval (IBP) linear layer with rounded arithmetic. -/
noncomputable def ibpLinearFloat {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (xB : Box ℝ (.dim n .scalar))
  (bB : Box ℝ (.dim m .scalar)) : Box ℝ (.dim m .scalar) :=
match W, xB.lo, xB.hi, bB.lo, bB.hi with
| .dim rows, .dim lo, .dim hi, .dim blo, .dim bhi =>
  let loOut := Tensor.dim (fun i =>
    match rows i, blo i with
    | .dim cols, .scalar bi =>
      -- sum_j min(aij*lo, aij*hi) then add bias
      let s :=
        (List.finRange n).foldl
          (fun (acc : Tensor ℝ .scalar) (j : Fin n) =>
            match acc, cols j, lo j, hi j with
            | .scalar accv, .scalar aij, .scalar xlo, .scalar xhi =>
              let p1 := neuralRound (β := β) (fexp := fexp) rnd (aij * xlo)
              let p2 := neuralRound (β := β) (fexp := fexp) rnd (aij * xhi)
              let mn := rMin (β := β) (fexp := fexp) rnd p1 p2
              Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd (accv + mn))) (Tensor.scalar
                0)
      match s with
      | .scalar sv => Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd (sv + bi)))
  let hiOut := Tensor.dim (fun i =>
    match rows i, bhi i with
    | .dim cols, .scalar bi =>
      let s :=
        (List.finRange n).foldl
          (fun (acc : Tensor ℝ .scalar) (j : Fin n) =>
            match acc, cols j, lo j, hi j with
            | .scalar accv, .scalar aij, .scalar xlo, .scalar xhi =>
              let p1 := neuralRound (β := β) (fexp := fexp) rnd (aij * xlo)
              let p2 := neuralRound (β := β) (fexp := fexp) rnd (aij * xhi)
              let mx := rMax (β := β) (fexp := fexp) rnd p1 p2
              Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd (accv + mx))) (Tensor.scalar
                0)
      match s with
      | .scalar sv => Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd (sv + bi)))
  { lo := loOut, hi := hiOut }

/-- Interval ReLU with float rounding: computes min/max of relu over bounds then rounds. -/
noncomputable def ibpReluFloat {n : Nat}
  (xB : Box ℝ (.dim n .scalar)) : Box ℝ (.dim n .scalar) :=
match xB.lo, xB.hi with
| .dim lo, .dim hi =>
  let outLo := Tensor.dim (fun i =>
    match lo i with
    | .scalar l =>
      let v := if l > 0 then l else 0
      Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd v))
  let outHi := Tensor.dim (fun i =>
    match hi i with
    | .scalar u =>
      let v := if u > 0 then u else 0
      Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd v))
  { lo := outLo, hi := outHi }

/-- Rounded matrix-vector multiplication (m×n by n) -/
noncomputable def matVecMulFloat {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (v : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim m .scalar) :=
match W, v with
| .dim rows, .dim vec =>
  Tensor.dim (fun i =>
    match rows i with
    | .dim cols =>
      let s := (List.finRange n).foldl
        (fun (acc : ℝ) (j : Fin n) =>
          let aij := match cols j with | .scalar a => a
          let vj  := match vec j with | .scalar x => x
          rFMA (β := β) (fexp := fexp) rnd aij vj acc) 0
      Tensor.scalar s)

/-- Rounded matrix-matrix multiplication (m×n by n×p) -/
noncomputable def matMulFloat {m n p : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (B : Tensor ℝ (.dim n (.dim p .scalar))) : Tensor ℝ (.dim m (.dim p .scalar)) :=
match A, B with
| .dim rowsA, .dim rowsB =>
  Tensor.dim (fun i =>
    match rowsA i with
    | .dim colsA =>
      Tensor.dim (fun j =>
        let s := (List.finRange n).foldl
          (fun (acc : ℝ) (k : Fin n) =>
            let ak := match colsA k with | .scalar a => a
            let bkj := match rowsB k with
              | .dim colsB => match colsB j with | .scalar b => b
            rFMA (β := β) (fexp := fexp) rnd ak bkj acc) 0
        Tensor.scalar s))

/-- Rounded column scaling: multiply each column j by v[j]. -/
noncomputable def matColScaleFloat {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar)))
  (v : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim m (.dim n .scalar)) :=
match A, v with
| .dim rows, .dim vec =>
  Tensor.dim (fun i =>
    match rows i with
    | .dim cols =>
      Tensor.dim (fun j =>
        let aij := match cols j with | .scalar a => a
        let vj  := match vec j with | .scalar x => x
        Tensor.scalar (rMul (β := β) (fexp := fexp) rnd aij vj)))

-- Build slope and bias vectors from ReLU relaxations -/
noncomputable def reluRelaxSlopeVec {n : Nat}
  (relax : Tensor (ReLURelax ℝ) (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match relax with
| .dim r => Tensor.dim (fun i => match r i with | .scalar rp => Tensor.scalar rp.slope)

noncomputable def reluRelaxBiasVec {n : Nat}
  (relax : Tensor (ReLURelax ℝ) (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match relax with
| .dim r => Tensor.dim (fun i => match r i with | .scalar rp => Tensor.scalar rp.bias)

/-- Rounded elementwise add and mul on vectors -/
noncomputable def vecAddFloat {n : Nat}
  (x y : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match x, y with
| .dim fx, .dim fy =>
  Tensor.dim (fun i =>
    let xi := match fx i with | .scalar v => v
    let yi := match fy i with | .scalar v => v
    Tensor.scalar (rAdd (β := β) (fexp := fexp) rnd xi yi))

noncomputable def vecMulFloat {n : Nat}
  (x y : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match x, y with
| .dim fx, .dim fy =>
  Tensor.dim (fun i =>
    let xi := match fx i with | .scalar v => v
    let yi := match fy i with | .scalar v => v
    Tensor.scalar (rMul (β := β) (fexp := fexp) rnd xi yi))

/-- Evaluate affine on a box with rounding (vector case). -/
noncomputable def affineEvalOnBoxFloat {inDim outDim : Nat}
  (A : Tensor ℝ (.dim outDim (.dim inDim .scalar)))
  (c : Tensor ℝ (.dim outDim .scalar))
  (B : Box ℝ (.dim inDim .scalar)) : Box ℝ (.dim outDim .scalar) :=
match A, c, B.lo, B.hi with
| .dim rows, .dim cvec, .dim loVec, .dim hiVec =>
  let outLo :=
    Tensor.dim (fun i =>
      match rows i, cvec i with
      | .dim cols, .scalar ci =>
        let s := (List.finRange inDim).foldl
          (fun (acc : ℝ) (j : Fin inDim) =>
            let aij := match cols j with | .scalar a => a
            let lo  := match loVec j with | .scalar v => v
            let hi  := match hiVec j with | .scalar v => v
            let p1 := rMul (β := β) (fexp := fexp) rnd aij lo
            let p2 := rMul (β := β) (fexp := fexp) rnd aij hi
            let mn := rMin (β := β) (fexp := fexp) rnd p1 p2
            rAdd (β := β) (fexp := fexp) rnd acc mn) 0
        Tensor.scalar (rAdd (β := β) (fexp := fexp) rnd s ci))
  let outHi :=
    Tensor.dim (fun i =>
      match rows i, cvec i with
      | .dim cols, .scalar ci =>
        let s := (List.finRange inDim).foldl
          (fun (acc : ℝ) (j : Fin inDim) =>
            let aij := match cols j with | .scalar a => a
            let lo  := match loVec j with | .scalar v => v
            let hi  := match hiVec j with | .scalar v => v
            let p1 := rMul (β := β) (fexp := fexp) rnd aij lo
            let p2 := rMul (β := β) (fexp := fexp) rnd aij hi
            let mx := rMax (β := β) (fexp := fexp) rnd p1 p2
            rAdd (β := β) (fexp := fexp) rnd acc mx) 0
        Tensor.scalar (rAdd (β := β) (fexp := fexp) rnd s ci))
  { lo := outLo, hi := outHi }

/-- Affine (CROWN) bounds with rounding for a 2-layer MLP -/
noncomputable def boundAffineFloat {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar)) : Box ℝ (.dim outDim .scalar) :=
  -- 1) Pre-activation bounds via rounded IBP
  let b1B : Box ℝ (.dim hidDim .scalar) := { lo := net.b1, hi := net.b1 }
  let z1B := ibpLinearFloat (β := β) (fexp := fexp) rnd net.W1 xB b1B
  -- 2) ReLU relaxation
  let relax := ReLU.relaxVector (α:=ℝ) (n:=hidDim) z1B.lo z1B.hi
  let slopeVec := reluRelaxSlopeVec (n:=hidDim) relax
  let biasVec  := reluRelaxBiasVec  (n:=hidDim) relax
  -- 3) Compose affine with rounding
  let W2scaled := matColScaleFloat (β := β) (fexp := fexp) rnd (m:=outDim) (n:=hidDim) net.W2
    slopeVec
  let A := matMulFloat (β := β) (fexp := fexp) rnd (m:=outDim) (n:=hidDim) (p:=inDim) W2scaled
    net.W1
  let s_b1 := vecMulFloat (β := β) (fexp := fexp) rnd slopeVec net.b1
  let inner := vecAddFloat (β := β) (fexp := fexp) rnd s_b1 biasVec
  let W2inner := matVecMulFloat (β := β) (fexp := fexp) rnd (m:=outDim) (n:=hidDim) net.W2 inner
  let c := match W2inner, net.b2 with
    | .dim w2i, .dim b2v =>
      Tensor.dim (fun i =>
        let wi := match w2i i with | .scalar v => v
        let bi := match b2v i with | .scalar v => v
        Tensor.scalar (rAdd (β := β) (fexp := fexp) rnd wi bi))
  -- 4) Evaluate on the input box with rounding
  affineEvalOnBoxFloat (β := β) (fexp := fexp) rnd (inDim:=inDim) (outDim:=outDim) A c xB

/-- End-to-end rounded IBP for 2-layer MLP (mirrors CROWN.bound_ibp). -/
noncomputable def boundIbpFloat {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim)
  (xB : Box ℝ (.dim inDim .scalar)) : Box ℝ (.dim outDim .scalar) :=
  let b1B : Box ℝ (.dim hidDim .scalar) := { lo := net.b1, hi := net.b1 }
  let z1B := ibpLinearFloat (β := β) (fexp := fexp) rnd net.W1 xB b1B
  let a1B := ibpReluFloat (β := β) (fexp := fexp) rnd (n:=hidDim) z1B
  let b2B : Box ℝ (.dim outDim .scalar) := { lo := net.b2, hi := net.b2 }
  ibpLinearFloat (β := β) (fexp := fexp) rnd net.W2 a1B b2B

/-- Public API: float-rounded CROWN IBP bounds wrapper with default config. -/
noncomputable def crownMlp2BoundsFloat
    {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x_center : Tensor ℝ (.dim inDim .scalar)) (eps : ℝ)
    (β : NeuralRadix := binaryRadix) (fexp : ℤ → ℤ := fun k => k)
    (rnd : ℝ → ℤ := neuralNearestEven)
    [NeuralValidExp fexp] [NeuralValidRnd rnd] :
    Box ℝ (.dim outDim .scalar) :=
  let net : MLP2 ℝ inDim hidDim outDim := ofLinearSpecs (α:=ℝ) l1 l2
  let rad := Tensor.scaleSpec (Spec.fill (α:=ℝ) eps (.dim inDim .scalar)) 1
  let xB : Box ℝ (.dim inDim .scalar) := { lo := Tensor.subSpec x_center rad,
                                           hi := Tensor.addSpec x_center rad }
  boundIbpFloat (β := β) (fexp := fexp) rnd net xB

/-- Public API: float-rounded (IBP, Affine) bounds wrapper, mirroring CROWN.Examples. -/
noncomputable def crownMlp2BoundsFloatFull
    {inDim hidDim outDim : Nat}
    (l1 : Spec.LinearSpec ℝ inDim hidDim)
    (l2 : Spec.LinearSpec ℝ hidDim outDim)
    (x_center : Tensor ℝ (.dim inDim .scalar)) (eps : ℝ)
    (β : NeuralRadix := binaryRadix) (fexp : ℤ → ℤ := fun k => k)
    (rnd : ℝ → ℤ := neuralNearestEven)
    [NeuralValidExp fexp] [NeuralValidRnd rnd] :
    Box ℝ (.dim outDim .scalar) × Box ℝ (.dim outDim .scalar) :=
  let net : MLP2 ℝ inDim hidDim outDim := ofLinearSpecs (α:=ℝ) l1 l2
  let rad := Tensor.scaleSpec (Spec.fill (α:=ℝ) eps (.dim inDim .scalar)) 1
  let xB : Box ℝ (.dim inDim .scalar) := { lo := Tensor.subSpec x_center rad,
                                           hi := Tensor.addSpec x_center rad }
  (boundIbpFloat (β := β) (fexp := fexp) rnd net xB,
   boundAffineFloat (β := β) (fexp := fexp) rnd net xB)

/-- Rounded vector ReLU. -/
noncomputable def reluVecFloat {n : Nat}
  (x : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match x with
| .dim fx =>
  Tensor.dim (fun i =>
    let v := match fx i with | .scalar a => a
    let r := neuralRound (β := β) (fexp := fexp) rnd (if v > 0 then v else 0)
    Tensor.scalar r)

/-- Rounded forward pass for a 2-layer MLP (linear → ReLU → linear). -/
noncomputable def forwardFloat {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim)
  (x : Tensor ℝ (.dim inDim .scalar)) : Tensor ℝ (.dim outDim .scalar) :=
  let z1 := matVecMulFloat (β := β) (fexp := fexp) rnd (m:=hidDim) (n:=inDim) net.W1 x
  let z1b := vecAddFloat (β := β) (fexp := fexp) rnd (n:=hidDim) z1 net.b1
  let a1 := reluVecFloat (β := β) (fexp := fexp) rnd (n:=hidDim) z1b
  let z2 := matVecMulFloat (β := β) (fexp := fexp) rnd (m:=outDim) (n:=hidDim) net.W2 a1
  vecAddFloat (β := β) (fexp := fexp) rnd (n:=outDim) z2 net.b2

/-- Round every scalar in a vector tensor to the target float grid. -/
noncomputable def roundVecFloat {n : Nat}
  (v : Tensor ℝ (.dim n .scalar)) : Tensor ℝ (.dim n .scalar) :=
match v with
| .dim fv =>
  Tensor.dim (fun i =>
    let xi := match fv i with | .scalar a => a
    Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd xi))

/-- Round every scalar in a matrix tensor to the target float grid. -/
noncomputable def roundMatFloat {m n : Nat}
  (A : Tensor ℝ (.dim m (.dim n .scalar))) : Tensor ℝ (.dim m (.dim n .scalar)) :=
match A with
| .dim rows =>
  Tensor.dim (fun i =>
    match rows i with
    | .dim cols =>
      Tensor.dim (fun j =>
        let aij := match cols j with | .scalar a => a
        Tensor.scalar (neuralRound (β := β) (fexp := fexp) rnd aij)))

/-- Pre-quantize an MLP2's parameters (W1, b1, W2, b2) onto the float grid. -/
noncomputable def quantizeParamsFloat {inDim hidDim outDim : Nat}
  (net : MLP2 ℝ inDim hidDim outDim) : MLP2 ℝ inDim hidDim outDim :=
  { W1 := roundMatFloat (β := β) (fexp := fexp) rnd (m:=hidDim) (n:=inDim) net.W1
  , b1 := roundVecFloat (β := β) (fexp := fexp) rnd (n:=hidDim) net.b1
  , W2 := roundMatFloat (β := β) (fexp := fexp) rnd (m:=outDim) (n:=hidDim) net.W2
  , b2 := roundVecFloat (β := β) (fexp := fexp) rnd (n:=outDim) net.b2 }

end NN.MLTheory.CROWN.Float
