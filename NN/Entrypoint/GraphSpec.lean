/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core
public import NN.GraphSpec.DAG
public import NN.GraphSpec.Models
public import NN.GraphSpec.Models.MlpDeterministicInit
public import NN.GraphSpec.Models.MlpSpecEquivalence
public import NN.GraphSpec.Primitives
public import NN.GraphSpec.Primitives.Embedding
public import NN.GraphSpec.ToTorchLean

/-!
# GraphSpec entrypoint

Curated umbrella import for GraphSpec.

This is the public single-import surface for GraphSpec users. For the full model-spec interface,
use:

```lean
import NN.Entrypoint.GraphSpec
```

It gives you:

- the canonical DAG model surface (`NN.GraphSpec.DAG.Term`, `NN.GraphSpec.DAG.Model`),
- the sequential authoring sugar (`NN.GraphSpec.Graph` + `>>>`) for chain models and its lowering
  into DAG,
- the Spec semantics (`NN.GraphSpec.Interp.spec`) and TorchLean compiler
  (`NN.GraphSpec.Compile.torchProgram`),
- sequential and DAG primitive packs,
- the GraphSpec example architectures (`NN.GraphSpec.Models`),
- the optional lowering to `TorchLean.NN.Seq` when primitives provide `toLayerDefM?`,
- and the model/primitive bridge theorems that connect GraphSpec syntax to Spec references.

We keep this file as an umbrella re-export (no real logic here) so the import path stays stable
even as the internals evolve.
-/

@[expose] public section


namespace NN
namespace GraphSpec

/-!
## Unified model type

GraphSpec's canonical “runnable + spec” representation is `DAG.Model`.

Sequential `Graph` pipelines can be lowered to DAG via `Core.LowerToDAG.Graph.toDAGTerm` and
`Core.LowerToDAG.Graph.toDAGModelZeroInit`, so users can author simple pipelines and still end up
in the same general model representation.
-/

@[inherit_doc DAG.Model]
abbrev Model := DAG.Model

namespace Model

@[inherit_doc DAG.Model.specFwd]
abbrev specFwd {ps ins : List Spec.Shape} {τ : Spec.Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α] :
    Runtime.Autograd.Torch.TList α ps → Runtime.Autograd.Torch.TList α ins → Spec.Tensor α τ :=
  DAG.Model.specFwd (ps := ps) (ins := ins) (τ := τ) m

@[inherit_doc DAG.Model.torchProgram]
abbrev torchProgram {ps ins : List Spec.Shape} {τ : Spec.Shape} (m : Model ps ins τ)
    {α : Type 0} [Context α] [DecidableEq Spec.Shape] :
    Runtime.Autograd.TorchLean.Program α (ps ++ ins) τ :=
  DAG.Model.torchProgram (ps := ps) (ins := ins) (τ := τ) m

end Model

end GraphSpec
end NN
