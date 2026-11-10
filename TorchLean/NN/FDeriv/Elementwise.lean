/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.FDeriv.Op

/-!
# Elementwise

On Euclidean coordinates, an elementwise (`map`) tensor of vector shape is `Vec.elemwise` of the
same function.

The generic coordinatewise calculus (`Vec.elemwise`, `elemwiseDerivCLM`,
`Vec.hasFDerivAt_elemwise`) lives in `TorchLean.NN.FDeriv`, and
`TorchLean.NN.Op.Pointwise` holds the derivative certificates of the pointwise operators at
every shape.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open scoped BigOperators

noncomputable section

/-- Coordinate `i` of `Tensor.map f (Tensor.ofFnE xV)` is `f (xV i)`. -/
@[simp] theorem Tensor.getScalar_map_ofFnE {n : Nat} (f : ℝ → ℝ) (xV : Vec n) (i : Fin n) :
    Tensor.getScalar (Tensor.map f (ofFnE xV)) i = f (xV i) := by
  simp [ofFnE]

/-- Vectorizing `Tensor.map f (Tensor.ofFnE xV)` gives `Vec.elemwise f xV`. An elementwise
operation's derivative can therefore be proved once on `EuclideanSpace` and transported to every
tensor of vector shape. -/
@[simp] theorem Tensor.getScalarE_map_ofFnE {n : Nat} (f : ℝ → ℝ) (xV : Vec n) :
    getScalarE (Tensor.map (shape := .dim n .scalar) f (ofFnE xV)) =
      Vec.elemwise (n := n) f xV := by
  ext i
  simp [Vec.elemwise]

end
end TorchLean
