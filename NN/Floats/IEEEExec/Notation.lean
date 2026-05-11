/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.BridgeFP32
public import NN.Floats.IEEEExec.ERealSemantics
public import NN.Floats.IEEEExec.Exec32

/-!
# Notation for executable float32 semantics (`IEEE32Exec`)

Mathlib tends to keep non-trivial notation **scoped** (see e.g. `scoped[FinsetFamily] notation …`),
so downstream code can opt in with `open scoped …` rather than inheriting new syntax globally.

This file follows that pattern for TorchLean's executable IEEE-754 binary32 model:

```lean
open scoped IEEE754

-- constants:
∞₃₂        -- `IEEE32Exec.posInf`
-∞₃₂       -- `IEEE32Exec.negInf`
NaN₃₂      -- `IEEE32Exec.canonicalNaN`

-- decoding semantics:
⟦x⟧₃₂      -- `IEEE32Exec.toReal x`
⟦x⟧₃₂ᴱ     -- `IEEE32Exec.toEReal x`
```

The notation here is kept small: it targets readability in docs/proofs, not a full DSL.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

namespace IEEE32Exec

/-! ## Scoped constants -/

@[inherit_doc]
scoped[IEEE754] notation "∞₃₂" => _root_.TorchLean.Floats.IEEE754.IEEE32Exec.posInf

@[inherit_doc]
scoped[IEEE754] notation "-∞₃₂" => _root_.TorchLean.Floats.IEEE754.IEEE32Exec.negInf

@[inherit_doc]
scoped[IEEE754] notation "NaN₃₂" => _root_.TorchLean.Floats.IEEE754.IEEE32Exec.canonicalNaN

/-! ## Scoped decoding notation -/

/-- Scoped notation for decoding an executable float32 to `ℝ` (via `IEEE32Exec.toReal`). -/
scoped[IEEE754] notation "⟦" x "⟧₃₂" => _root_.TorchLean.Floats.IEEE754.IEEE32Exec.toReal x

/-- Scoped notation for decoding an executable float32 to `EReal` (via `IEEE32Exec.toEReal`). -/
scoped[IEEE754] notation "⟦" x "⟧₃₂ᴱ" => _root_.TorchLean.Floats.IEEE754.IEEE32Exec.toEReal x

end IEEE32Exec

end TorchLean.Floats.IEEE754
