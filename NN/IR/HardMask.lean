/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Graph
public import NN.Spec.Core.TensorReductionShape.ShapeChange

/-!
# Hard-Mask Payloads

Conversions between typed Boolean tensors and the row-major masks stored in `NN.IR.OpKind`.
Keeping this bridge in one module gives graph builders, evaluators, and verifier passes the same
validation rule for mask payloads.
-/

@[expose] public section

namespace NN.IR.HardMask

open _root_.Spec _root_.TorchLean
open _root_.TorchLean.Tensor

/-- Flatten a typed Boolean mask into an IR payload. -/
def ofTensor {s : Shape} (mask : Tensor Bool s) : NN.IR.HardMask :=
  let flat := Tensor.flattenSpec mask
  { shape := s
    allowed := Array.ofFn fun i => Tensor.getScalar flat i }

/-- Check a hard-mask payload and return the equality needed to recover its typed shape. -/
def validateAs (mask : NN.IR.HardMask) (expected : Shape) :
    Except String (PLift (mask.shape = expected)) := do
  if hShape : mask.shape = expected then
    if _hSize : mask.allowed.size = Shape.size expected then
      pure ⟨hShape⟩
    else
      throw <|
        s!"hard mask: shape {repr expected} requires {Shape.size expected} entries, " ++
          s!"but the payload contains {mask.allowed.size}"
  else
    throw <|
      s!"hard mask: expected shape {repr expected}, but the payload has shape {repr mask.shape}"

/-- Reconstruct a typed mask, rejecting a payload whose flat length disagrees with its shape. -/
def toTensor? (mask : NN.IR.HardMask) : Except String (Tensor Bool mask.shape) := do
  if hSize : mask.allowed.size = Shape.size mask.shape then
    let flat : Tensor Bool [Shape.size mask.shape] :=
      Tensor.dim fun i => Tensor.scalar <| mask.allowed[i.val]'(by simp [hSize, i.isLt])
    pure (Tensor.unflattenSpec mask.shape flat)
  else
    throw <|
      s!"hard mask: shape {repr mask.shape} requires {Shape.size mask.shape} entries, " ++
        s!"but the payload contains {mask.allowed.size}"

/--
Decode a hard-mask payload at an expected shape.

This is the common validation boundary for graph evaluation and executable lowering. It checks
both the shape tag and the row-major payload length before transporting the decoded tensor to the
requested type.
-/
def toTensorAs? (mask : NN.IR.HardMask) (expected : Shape) :
    Except String (Tensor Bool expected) := do
  if h : mask.shape = expected then
    pure (h ▸ (← toTensor? mask))
  else
    throw <|
      s!"hard mask: expected shape {repr expected}, but the payload has shape {repr mask.shape}"

/-- Encoding and then decoding a typed mask preserves every Boolean entry. -/
@[simp] theorem toTensor?_ofTensor {s : Shape} (mask : Tensor Bool s) :
    toTensor? (ofTensor mask) = .ok mask := by
  simp [toTensor?, ofTensor]
  rw [← Tensor.unflattenSpec_flattenSpec (tensor := mask)]
  congr 2
  apply Tensor.ext_vector
  intro i
  simp

/-- Encoding a typed mask and decoding it at the same expected shape is a round trip. -/
@[simp] theorem toTensorAs?_ofTensor {s : Shape} (mask : Tensor Bool s) :
    toTensorAs? (ofTensor mask) s = .ok mask := by
  unfold toTensorAs?
  rw [dite_eq_left (by rfl)]
  rw [toTensor?_ofTensor]
  rfl

end NN.IR.HardMask
