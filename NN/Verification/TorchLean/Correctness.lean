/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Verification.TorchLean.Compile

/-!
# Correctness

TorchLean→IR correctness helpers.

This file does **not** (yet) contain a full compiler-correctness theorem for arbitrary
`TorchLean.Program`s (the current embedding is higher-order). It provides the small, reusable
bridges needed by concrete model-correctness theorems:

- convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`;
- evaluate a `CompiledIR` graph on a concrete input.
-/

@[expose] public section


namespace NN.Verification.TorchLean

open Spec
open Tensor
open NN.IR

/--
Convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`.

This is the bridge between the CROWN/LiRPA parameter representation used by the verification
pipeline and the executable IR semantics.
-/
def payloadOfParamStore {α : Type} [Context α] (ps : NN.MLTheory.CROWN.Graph.ParamStore α) : Payload
  α :=
  { const? := fun id =>
      (ps.constVals.get? id).map (fun c =>
        { n := c.n, v := c.v })
    linear? := fun id =>
      (ps.linearWB.get? id).map (fun p =>
        { outDim := p.m, inDim := p.n, W := p.w, b := p.b })
    conv2d? := fun id =>
      (ps.conv2dCfg.get? id).map (fun cfg =>
        { inC := cfg.inC, outC := cfg.outC, kH := cfg.kH, kW := cfg.kW
          stride := cfg.stride, padding := cfg.padding, inH := cfg.inH, inW := cfg.inW
          hIn := cfg.hIn, hKH := cfg.hKH, hKW := cfg.hKW, hStride := cfg.hStride,
          spec := cfg.spec })
    batchNorm2dNchwEval? := fun id =>
      (ps.batchNorm2dNchwEval.get? id).map (fun p =>
        { c := p.c, gamma := p.gamma, beta := p.beta, mean := p.mean, var := p.var, eps := p.eps }) }

/-- Cast a tensor across a proved shape equality. -/
def castTensor {α : Type} [Context α] {s s' : Shape} (h : s = s') (t : Tensor α s) : Tensor α s' :=
  cast (congrArg (fun s : Shape => Tensor α s) h) t

/-- Evaluate a `CompiledIR` forward graph on an input tensor, returning a shape-checked tensor. -/
def evalCompiledForward1
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape outShape : Shape}
    (c : CompiledIR α) (x : Tensor α inShape) : Except String (Tensor α outShape) := do
  let input : DVal α := DVal.mk (α := α) inShape x
  let out ←
    Graph.denote (α := α) (g := c.graph) (payload := payloadOfParamStore (α := α) c.ps)
      (input := input) (outputId := c.outputId)
  if h : out.shape = outShape then
    pure (h ▸ out.tensor)
  else
    throw <|
      s!"TorchLeanCorrectness: output shape mismatch: " ++
        s!"produced={repr out.shape}, expected={repr outShape}"

end NN.Verification.TorchLean
