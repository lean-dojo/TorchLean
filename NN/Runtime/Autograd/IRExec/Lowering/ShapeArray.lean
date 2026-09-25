/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Primitives

/-!
# Lowering Shape Context

An array of previously lowered shapes supports constant-time parent selection without retaining
chronological list prefixes. Its erased invariant gives each successful lookup the same typed
index as `mkIdx`, including identical errors for invalid positions and mismatched shapes.
-/

@[expose] public section

namespace Runtime.Autograd.IRExec.Internal

open Spec TorchLean
open Proofs (Idx)

/-- Previously lowered shapes together with their chronological type index. -/
@[ext] structure ShapeArray (shapes : List Shape) where
  /-- Shapes in IR node-id order. -/
  values : Array Shape
  /-- The array and the logical context describe the same positions and shapes. -/
  valid : values.toList = shapes

namespace ShapeArray

variable {shapes : List Shape}

/-- Initialize a shape context from an existing chronological prefix. -/
def ofList (shapes : List Shape) : ShapeArray shapes :=
  ⟨shapes.toArray, by simp⟩

/-- Append a shape without materializing the extended logical list. -/
def push (ctx : ShapeArray shapes) (shape : Shape) : ShapeArray (shapes ++ [shape]) :=
  ⟨ctx.values.push shape, by simp [ctx.valid]⟩

/-- Transport the erased context index without changing the stored array. -/
def cast {other : List Shape} (h : shapes = other) (ctx : ShapeArray shapes) : ShapeArray other :=
  h ▸ ctx

/-- The stored array is uniquely determined by the logical context. -/
theorem eq_ofList (ctx : ShapeArray shapes) : ctx = ofList shapes := by
  apply ShapeArray.ext
  simpa [ofList] using congrArg List.toArray ctx.valid

/-- Check a parent position and shape using the lowering context's array. -/
@[inline] def mkIdx (ctx : ShapeArray shapes) (id : Nat) (shape : Shape) :
    Except String (Idx shapes shape) :=
  if h : id < ctx.values.size then
    let got := ctx.values[id]
    if hg : got = shape then
      .ok ⟨⟨id, by simpa [← ctx.valid] using h⟩, by
        simpa [← ctx.valid, List.get_eq_getElem] using hg⟩
    else
      .error s!"IRExec: shape mismatch at id={id}: expected {Shape.pretty shape}, \
        got {Shape.pretty got}"
  else
    .error s!"IRExec: invalid id={id} for ctxLen={ctx.values.size}"

/-- Array parent checks return exactly the index or error produced by the logical list check. -/
theorem mkIdx_eq (inShape : Shape) (ss : List Shape)
    (ctx : ShapeArray ([inShape] ++ ss)) (id : Nat) (shape : Shape) :
    ctx.mkIdx id shape = Internal.mkIdx inShape ss id shape := by
  rw [ctx.eq_ofList]
  simp only [mkIdx, ofList, Internal.mkIdx, List.size_toArray, List.getElem_toArray,
    List.get_eq_getElem]
  rfl

end ShapeArray
end Runtime.Autograd.IRExec.Internal
