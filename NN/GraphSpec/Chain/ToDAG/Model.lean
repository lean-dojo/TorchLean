/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Core
public import NN.GraphSpec.DAG.Model

/-!
# DAG models from sequential GraphSpec chains

This module initializes a chain parameter ABI and packages the structurally lowered term as a
single-input DAG model. Zero initialization is total; deterministic initialization reuses each
primitive layer conversion and can therefore fail.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open Spec TorchLean

namespace LowerToDAG

/--
Initialize a parameter list by filling every tensor with zeros, for proofs and shape-only examples.
-/
def zeroInitParams : (ps : List Shape) → TorchLean.TensorPack Float ps
  | .nil => .nil
  | .cons s ss => .cons (Tensor.zeros (α := Float) s) (zeroInitParams ss)

/-!
### Deterministic initialization for chains

`Chain.toDAGModelZeroInit` is total, but its parameters are all-zero tensors, which is convenient
for proofs and shape-only examples but not representative of training setups.

For graphs whose primitives provide `Primitive.toLayerM?`, we can reuse TorchLean’s deterministic
initializers (e.g. Xavier init for linear weights) in a way that matches `ToSequential.toSeq`:

- we thread an occurrence index `i : Nat`,
- primitives with `countsAsLayer = true` increment it,
- and each primitive’s `Layer.initState` uses seeds derived from `i`.

We expose this as `Chain.toDAGModelDetInit? : Except String (DAG.Model ...)`:
it fails if any primitive lacks a `toLayerM?` lowering.
-/

/--
Compute deterministic initialization tensors for a sequential `Chain`, threading a “layer
occurrence index”.

This matches `ToSequential.toSeq`’s notion of “occurrence”: only primitives with
`countsAsLayer = true` advance the counter.
-/
def Chain.detInitParamsFrom
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ) (i : Nat) :
    Except String (TorchLean.TensorPack Float ps × Nat) :=
  match g with
  | .id _ => .ok (.nil, i)
  | .prim p =>
      match p.toLayerM? with
      | none =>
          .error <|
            (s!"graphspec.detInit: primitive `{p.name}` has no deterministic init " ++
              s!"(missing toLayerM?)")
      | some mk =>
          let ⟨l, hps⟩ := mk i
          let i' := if p.countsAsLayer then i + 1 else i
          match hps with
          | rfl => .ok (l.initState, i')
  | .seq (ps₁ := ps₁) (ps₂ := ps₂) (σ := σ) g₁ g₂ =>
      match Chain.detInitParamsFrom (ps := ps₁) (σ := σ) g₁ i with
      | .error e => .error e
      | .ok (xs, i') =>
          match Chain.detInitParamsFrom (ps := ps₂) (σ := _) g₂ i' with
          | .error e => .error e
          | .ok (ys, i'') =>
              .ok
                ( TorchLean.TensorPack.append
                    (α := Float) (ss₁ := ps₁) (ss₂ := ps₂) xs ys
                , i'')

/-- Deterministically initialize all graph parameters, starting the occurrence index at `0`. -/
def Chain.detInitParams?
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ) :
    Except String (TorchLean.TensorPack Float ps) :=
  match Chain.detInitParamsFrom (ps := ps) (σ := σ) (τ := τ) g 0 with
  | .error e => .error e
  | .ok (xs, _i) => .ok xs
/--
Lower a sequential `Chain` to a DAG `Model` with a simple default init (all zeros).

This is mainly a convenience for GraphSpec example organization; for training-oriented init,
see `NN.GraphSpec.ToSequential` (sequential-model conversion) and/or provide your own initializer.
 -/
def Chain.toDAGModelZeroInit {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    DAG.Model ps [σ] τ :=
  { initParams := zeroInitParams ps
    body := Chain.toDAGTerm (ps := ps) (σ := σ) (τ := τ) g
  }

/--
Lower a sequential `Chain` to a DAG `Model`, using deterministic initialization.

This is the DAG analogue of `ToSequential.toSeq`’s initialization semantics: it uses each
primitive’s `toLayerM?` to obtain a TorchLean `Layer`, then reuses the `Layer.initState`.

This returns `Except String` because not every primitive necessarily admits a `Layer` lowering.
-/
def Chain.toDAGModelDetInit?
    {ps : List Shape} {σ τ : Shape} (g : Chain ps σ τ) :
    Except String (DAG.Model ps [σ] τ) :=
  match Chain.detInitParams? (ps := ps) (σ := σ) (τ := τ) g with
  | .error e => .error e
  | .ok params =>
      .ok { initParams := params
            body := Chain.toDAGTerm (ps := ps) (σ := σ) (τ := τ) g }


end LowerToDAG

end GraphSpec
end NN
