/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

-- Downstream modules reach the marked imports through this module. Import shaking cannot see
-- those downstream uses, so keep the marked imports.
module -- shake: keep-downstream

public import TorchLean.NN.Optimizer.Config -- shake: keep
public import TorchLean.NN.Optimizer -- shake: keep
public import TorchLean.Tensor.FromFloat

/-!
# Optimizers

The `optim` namespace holds optimizer configurations for the trainer and constructors for runtime
optimizers. The configuration records come from `TorchLean.NN.Optimizer.Config`, which this module
passes on to its importers. They cover SGD, momentum SGD, AdaGrad, RMSProp, Adam, AdamW and
Adadelta, each with a self-contained update rule (PyTorch: `torch.optim.SGD`, `torch.optim.Adam` and
the rest), and `Optimizer.validateFor` checks one against the runtime scalar.

Two runtime-only constructors take a backend from the caller:

- `optim.muon` takes an orthogonalization backend and defaults to the identity, which
  serves proofs and runs without a matrix backend.
- `optim.galore.sgd` wraps SGD in GaLore-style gradient projection. Its name says which update
  rule owns the optimizer state.
-/

@[expose] public section

namespace TorchLean

namespace optim

/--
Check an optimizer configuration, then check it again after rounding its scalars into the runtime
scalar `α`.

For `Float32` and software binary32 the second check rejects overflow, positive stabilizers that
round to zero and averaging coefficients that round to one. `Float` keeps the binary64 domains.
A custom scalar can supply `FromFloat.roundForValidation`, and without it only the
binary64 checks apply. `scalarName` labels the conversion errors.
-/
def Optimizer.validateFor {α : Type} [FromFloat α]
    (optimizer : Optimizer) (scalarName : String := "the runtime scalar") : Except String Unit := do
  optimizer.validate
  let converted := Optimizer.mapScalars
    (FromFloat.roundForValidation (α := α)) optimizer
  match converted.validate with
  | .ok () => pure ()
  | .error message => throw s!"{message} after conversion to {scalarName}"

/--
Muon-style momentum for module-level training: each step updates the momentum buffer, passes it
through `orthogonalizer` and subtracts the result times `lr`, so the default identity backend gives
momentum SGD. Full Muon needs a matrix-shaped orthogonalizer, such as a Newton-Schulz iteration,
which is why this constructor builds a runtime optimizer and has no trainer configuration.
-/
def muon {α : Type} [TorchLean.Storage α] [Context α]
    (lr momentum : α)
    (orthogonalizer : {s : Shape} → Muon.Orthogonalizer α s :=
      fun {s} => Muon.identityOrthogonalizer (α := α) (s := s))
    {paramShapes : List Shape} :
    TorchLean.Optimizer α paramShapes :=
  { State := Pack (Muon.State α) paramShapes
    init := fun ps =>
      TorchLean.Optimizer.initStateList (α := α) (State := Muon.State)
        (initOne := fun {s} t =>
          Muon.init
            (α := α) (s := s) lr momentum
            (orthogonalizer (s := s)) t) ps
    step := fun st ps grads =>
      TorchLean.Optimizer.stepStateList (α := α) (State := Muon.State)
        (updateOne := fun {s} stOne params g =>
          Muon.update (α := α) (s := s) stOne params g)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

namespace galore

/--
Projected SGD in the style of GaLore: each parameter steps to `p - lift (project g) * lr`, with
one same-shape `projector` per parameter shape, fixed at construction and never refreshed. A
rank-`r` projection `project g = U (Uᵀ g)` with `lift = id` fits that type, and the identity
default gives plain SGD, which `GaLore.update_params_eq_sgd` proves for each tensor by `rfl`.
Nothing else about this update is proved.
-/
def sgd {α : Type} [TorchLean.Storage α] [Context α]
    (lr : α)
    (projector : {s : Shape} → GaLore.Projector α s s :=
      fun {s} => GaLore.identityProjector (α := α) (s := s))
    {paramShapes : List Shape} :
    TorchLean.Optimizer α paramShapes :=
  { State := Pack (fun s => GaLore.SGDState α s s) paramShapes
    init := fun ps =>
      TorchLean.Optimizer.initStateList (α := α)
        (State := fun α _ s => GaLore.SGDState α s s)
        (initOne := fun {s} _t =>
          { lr := lr, projector := projector (s := s) }) ps
    step := fun st ps grads =>
      TorchLean.Optimizer.stepStateList (α := α)
        (State := fun α _ s => GaLore.SGDState α s s)
        (updateOne := fun {s} stOne parameters gradients =>
          GaLore.update
            (α := α) (full := s) (low := s) stOne parameters gradients)
        ps st grads
    setLr := fun rate st =>
      st.map fun stOne => { stOne with lr := rate }
  }

end galore

end optim


end TorchLean
