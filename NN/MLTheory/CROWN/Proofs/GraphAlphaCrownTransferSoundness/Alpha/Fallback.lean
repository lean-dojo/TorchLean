/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphAlphaCrownTransferSoundness.Alpha.Basic

/-!
# α-CROWN Transfer: IBP Fallback

The cases of `alphaCrown_transfer_sound` for operators outside the affine-transfer subset. The
step can only succeed through the constant enclosure derived from the IBP box at the same node,
which is sound because `IBPEnclosesVals` says that box encloses the semantic value.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert
open CrownCertSoundness
open CertSoundness

/-- The IBP fallback: if the step returned the constant enclosure of the IBP box `B0` stored at
`id`, that enclosure is sound because `IBPEnclosesVals` says `B0` encloses the semantic value at
`id`. The dispatch in `alphaCrown_transfer_sound` obtains `hib` and `hb` by splitting the unfolded
step equation, with or without the `crownNodeSemanticsSupported` guard. -/
theorem fallback_sound {ibp : Array (Option (FlatBox ℝ))} {vals : Array (Option Val)}
    {ctx : AffineCtx} {x : Tensor ℝ [ctx.inputDim]} {id : Nat} {B0 : FlatBox ℝ}
    {b : FlatAffineBounds ℝ} {v : Val}
    (hib : ibp[id]! = some B0) (hb : Cert.boundsConst (α := ℝ) ctx.inputDim B0.dim B0.lo B0.hi = b)
    (hv : vals[id]! = some v) (hlt : id < vals.size)
    (hibp : IBPEnclosesVals (ibp := ibp) (vals := vals)) :
    EnclosesAtInput (α := ℝ) ctx x b v := by
  subst hb
  have hEnc : EnclosesBox B0 v := by
    have h := hibp id hlt
    simpa only [hib, hv] using h
  exact enclosesAtInput_boundsConst_of_enclosesBox hEnc

end NN.MLTheory.CROWN.Graph.AlphaCrownTransferSoundness.Alpha

end
