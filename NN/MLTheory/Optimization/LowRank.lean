/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.Optimization.FirstOrder

/-!
# Low-rank and Orthogonalized Optimizer Facts

This file records the sanity checks for the new optimizer extension points:

- a GaLore-style projected gradient update reduces to SGD when the projector is identity;
- a Muon-style orthogonalized momentum update reduces to momentum SGD when the orthogonalizer is
  identity.

That gives us a clean proof boundary: projector construction and matrix orthogonalization can be
optimized later, while the surrounding update semantics already have a checked fallback case.
-/

@[expose] public section


namespace Optim
open Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

namespace GaLore

/-- With the identity projector, projected SGD is exactly ordinary SGD. -/
theorem projectedSGD_identity_eq_sgd {s : Shape} (lr : α)
    (params grads : Tensor α s) :
    projectedSGDUpdate
        ({ lr := lr, projector := identityProjector (α := α) (s := s) } : SGDState α s s)
        params grads =
      SGD.update ({ lr := lr } : SGD.State α s) params grads := by
  rfl

end GaLore

namespace Muon

/-- With the identity orthogonalizer, Muon and momentum SGD use the same parameter update. -/
theorem update_identity_params_eq_momentumSGD {s : Shape}
    (lr momentum : α) (buf params grads : Tensor α s) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := identityOrthogonalizer (α := α) (s := s) } : State α s)
        params grads).2 =
      (MomentumSGD.update ({ lr := lr, momentum := momentum, buf := buf } :
        MomentumSGD.State α s) params grads).2 := by
  rfl

/-- With the identity orthogonalizer, Muon and momentum SGD store the same next buffer. -/
theorem update_identity_buffer_eq_momentumSGD {s : Shape}
    (lr momentum : α) (buf params grads : Tensor α s) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := identityOrthogonalizer (α := α) (s := s) } : State α s)
        params grads).1.buf =
      (MomentumSGD.update ({ lr := lr, momentum := momentum, buf := buf } :
        MomentumSGD.State α s) params grads).1.buf := by
  rfl

/-- Expanded form of the Muon buffer update. -/
theorem update_buffer_eq {s : Shape} (state : State α s) (params grads : Tensor α s) :
    (update state params grads).1.buf =
      addSpec (scaleSpec state.buf state.momentum) grads := by
  rfl

/-- Expanded form of the identity-backend Muon parameter update. -/
theorem update_identity_params_eq {s : Shape}
    (lr momentum : α) (buf params grads : Tensor α s) :
    (update
        ({ lr := lr, momentum := momentum, buf := buf,
           orthogonalizer := identityOrthogonalizer (α := α) (s := s) } : State α s)
        params grads).2 =
      subSpec params
        (scaleSpec (addSpec (scaleSpec buf momentum) grads) lr) := by
  rfl

end Muon

end Optim
