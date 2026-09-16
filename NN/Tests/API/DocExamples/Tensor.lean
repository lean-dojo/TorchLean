/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Operations
public import NN.Tensor.Internal.Elab.TensorLiteral

/-!
# Compiled Docstring Examples: Tensors

Every `Example:` block in a `TorchLean.Tensor` docstring appears here verbatim, one namespace per
entry point. Nothing in this file runs: the guarantee we want is that the snippet a reader copies
out of the docstring still elaborates, and typechecking gives us exactly that. For tensors that is
a strong guarantee, because the shapes in the snippet are checked, not asserted.

`repo_lint.py` compares the fenced `lean` block inside each `Example:` against the namespace body
carrying the matching `doc-example:` marker, so the two cannot drift apart.
-/

@[expose] public section

namespace NN.Tests.API.DocExamples.Tensors

open TorchLean

-- doc-example: NN/Spec/Core/Tensor/Constructors.lean :: def ofFn
namespace OfFn

-- `torch.tensor([0.0, 1.0, 2.0, 3.0])`, except the entries arrive from a function on `Fin n` and
-- the length is part of the type.
def ramp : Tensor Float [4] := Tensor.ofFn fun index => index.val.toFloat

end OfFn

-- doc-example: NN/Tensor/Operations.lean :: def matmul
namespace Matmul

-- `[2, 3] * [3, 2]` contracts the shared `3`. A mismatch is a type error, not a runtime message.
def left : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def right : Tensor Float [3, 2] := [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

def product : Tensor Float [2, 2] := Tensor.matmul left right

end Matmul

-- doc-example: NN/Tensor/Operations.lean :: def matvec
namespace Matvec

-- Matrix times column vector. It stays a separate name instead of an overload of `matmul` so the
-- shapes a reader should expect are visible at the call site, the way BLAS separates gemv.
def matrix : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def rowSums : Tensor Float [2] := Tensor.matvec matrix [1.0, 1.0, 1.0]

end Matvec

-- doc-example: NN/Tensor/Operations.lean :: def vecmat
namespace Vecmat

-- Row vector times matrix: the same product read from the other side, with no transpose.
def matrix : Tensor Float [2, 3] := [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

def columnSums : Tensor Float [3] := Tensor.vecmat [1.0, 1.0] matrix

end Vecmat

-- doc-example: NN/Tensor/Operations.lean :: def linear
namespace Linear

-- The weight layout is `[outputWidth, inputWidth]`, as in PyTorch, and leading axes pass straight
-- through: a `[4, 3]` batch of rows comes back as `[4, 2]`.
def weight : Tensor Float [2, 3] := [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

def bias : Tensor Float [2] := [0.5, -0.5]

def project (rows : Tensor Float [4, 3]) : Tensor Float [4, 2] :=
  Tensor.linear rows weight bias

end Linear

-- doc-example: NN/Tensor/Operations.lean :: def stackLeading
namespace StackLeading

-- Build a leading axis from an index function: three rows of width two become one `[3, 2]`.
def rows : Tensor Float [3, 2] :=
  Tensor.stackLeading fun (row : Fin 3) =>
    Tensor.ofFn fun (column : Fin 2) => (row.val + column.val).toFloat

end StackLeading

-- doc-example: NN/Tensor/Operations.lean :: def concat
namespace Concat

-- Join on the leading axis: `2 + 3` rows that share the row shape `[2]`.
def top : Tensor Float [2, 2] := [[1.0, 2.0], [3.0, 4.0]]

def bottom : Tensor Float [3, 2] := [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]

def stacked : Tensor Float [5, 2] := Tensor.concat top bottom

end Concat

-- doc-example: NN/Tensor/Operations.lean :: def concatAfter
namespace ConcatAfter

-- The same join one axis further in, which is how a batch axis is kept out of the way:
-- `[4, 2, 3]` and `[4, 5, 3]` become `[4, 7, 3]`.
def joined (left : Tensor Float [4, 2, 3]) (right : Tensor Float [4, 5, 3]) :
    Tensor Float [4, 7, 3] :=
  Tensor.concatAfter [4] left right

end ConcatAfter

-- doc-example: NN/Tensor/Operations.lean :: def take
namespace Take

-- Keep the first two rows. The bound `2 <= 4` is discharged at the call site, so an out-of-range
-- count fails to compile instead of failing at runtime.
def firstRows (tensor : Tensor Float [4, 3]) : Tensor Float [2, 3] :=
  Tensor.take tensor 0 2

end Take

-- doc-example: NN/Tensor/Operations.lean :: def flattenAfter
namespace FlattenAfter

-- Keep the batch axis, flatten the rest: a batch of small images becomes a batch of vectors,
-- `[8, 3, 4, 4]` to `[8, 48]`.
def vectors (images : Tensor Float [8, 3, 4, 4]) : Tensor Float [8, 48] :=
  Tensor.flattenAfter [8] images

end FlattenAfter

-- doc-example: NN/Tensor/Conversion.lean :: def reshape
namespace Reshape

-- Same buffer, new shape. The size equality is the entire safety condition, and it is checked
-- here rather than trusted.
def matrix (flat : Tensor Float [12]) : Tensor Float [3, 4] :=
  Tensor.reshape flat [3, 4]

end Reshape

-- doc-example: NN/Spec/Core/Sequence.lean :: def mapLeading
namespace MapLeading

-- Write the per-sample function and let the leading axis take care of itself: `[5, 3]` in,
-- `[5, 1]` out, with no loop over the batch.
def firstFeature (batch : Tensor Float [5, 3]) : Tensor Float [5, 1] :=
  Tensor.mapLeading [5] (fun row => Tensor.take row 0 1) batch

end MapLeading

end NN.Tests.API.DocExamples.Tensors
