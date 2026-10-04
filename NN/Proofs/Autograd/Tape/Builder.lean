/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.TapeM

/-!
# Reasoning about the tape-builder monad

`Runtime.Autograd.TapeM` is the `StateT (Tape α) Result` wrapper users write eager programs in.
The pure engine underneath is well covered by proofs; the monadic surface was not, so a `do`-block
had no route back to a statement about the tape it built. This file supplies that route.

Every op wrapper that threads the tape is `TapeM.Internal.record` applied to its pure counterpart,
so the whole surface reduces through one lemma:

- `record_run_ok` / `record_run_error` turn a pure op's outcome into the monadic `run`'s outcome,
  and `record_run_inv` goes back. Note the pair swaps: a pure op returns tape-then-id, `run`
  returns value-then-state.
- `run_bind_inv` splits a successful `run` of `m >>= f` into its two successful stages, which is
  what peels a `do`-block one statement at a time.
- The `run_<op>_ok` family below is `record_run_ok` instantiated at each op's own pure
  counterpart. Each is a definitional instantiation (the proof term is `record_run_ok` with `op`
  supplied, with no unfolding lemma in between), so the family stays correct by construction as
  ops are added: if one of them ever stops typechecking, the wrapper it names has stopped being
  `Internal.record` at its pure op.

`TapeM.leaf` is not in the family: it is total, so its pure counterpart returns a bare pair rather
than a `Result`, and `run_leaf` states its (unconditional) run directly. `TapeM.backwardScalar` is
also outside it: it reads the tape without writing one back.

## PyTorch correspondence
`TapeM` mirrors the imperative style of building a graph by calling ops in sequence and then
calling `backward`; these lemmas are what lets a proof follow such a program.
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Builder

open Spec TorchLean
open TorchLean TorchLean.Tensor
open Runtime.Autograd (Tape TapeM Result)

variable {α : Type} [TorchLean.Storage α]

/-! ## The shared reshuffle, reduced once -/

/-- A successful pure op gives a successful monadic run. The pair swaps: the pure op returns
tape-then-value, `run` returns value-then-state. -/
theorem record_run_ok {op : Tape α → Result (Tape α × Nat)} {t t' : Tape α} {v : Nat}
    (h : op t = .ok (t', v)) :
    (TapeM.Internal.record op).run t = .ok (v, t') := by
  simp only [TapeM.Internal.record, TapeM.run, StateT.run, bind, Except.bind, pure, Except.pure, h]

/-- A failing pure op gives a failing monadic run. -/
theorem record_run_error {op : Tape α → Result (Tape α × Nat)} {t : Tape α} {e : String}
    (h : op t = .error e) :
    (TapeM.Internal.record op).run t = .error e := by
  simp only [TapeM.Internal.record, TapeM.run, StateT.run, bind, Except.bind, h]

/-- Inversion: a successful monadic run means the pure op succeeded. -/
theorem record_run_inv {op : Tape α → Result (Tape α × Nat)} {t t' : Tape α} {v : Nat}
    (h : (TapeM.Internal.record op).run t = .ok (v, t')) :
    op t = .ok (t', v) := by
  cases hg : op t with
  | ok p =>
      obtain ⟨t1, v1⟩ := p
      rw [record_run_ok (op := op) (t := t) (t' := t1) (v := v1) hg] at h
      injection h with h'
      have h1 : v1 = v := congrArg Prod.fst h'
      have h2 : t1 = t' := congrArg Prod.snd h'
      subst h1
      subst h2
      rfl
  | error e =>
      rw [record_run_error (op := op) hg] at h
      cases h

/-! ## Peeling a `do`-block -/

/-- A successful run of `m >>= f` decomposes into its two successful stages. This is what takes a
`do`-block apart one statement at a time. -/
theorem run_bind_inv {β γ : Type} {m : TapeM α β} {f : β → TapeM α γ}
    {t : Tape α} {r : γ × Tape α}
    (h : (m >>= f).run t = .ok r) :
    ∃ b t1, m.run t = .ok (b, t1) ∧ (f b).run t1 = .ok r := by
  have hb : (m >>= f).run t = (m.run t).bind fun bt => (f bt.1).run bt.2 := rfl
  rw [hb] at h
  cases hm : m.run t with
  | error e => rw [hm] at h; cases h
  | ok bt =>
      obtain ⟨b1, bt2⟩ := bt
      rw [hm] at h
      exact ⟨b1, bt2, rfl, h⟩

/-! ## `leaf`, which is total -/

/-- `TapeM.leaf` cannot fail, so its run is unconditional. -/
theorem run_leaf {s : Shape} (value : Tensor α s) (name : Option String := none)
    (requiresGrad : Bool := true) {t : Tape α} :
    (TapeM.leaf (α := α) (s := s) value name requiresGrad).run t
      = .ok ((Tape.leaf (t := t) value (name := name) (requiresGrad := requiresGrad)).2,
             (Tape.leaf (t := t) value (name := name) (requiresGrad := requiresGrad)).1) :=
  rfl

/-! ## The per-op family: `record_run_ok` at each op's own pure counterpart

Each proof below is `record_run_ok` with `op` supplied and nothing else. The binders mirror the
wrapper's own, so a lemma states exactly what its wrapper can be called with.
-/

theorem run_add_ok [Add α] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.add (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.add (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.add (t := tt) (s := s) aId bId) h

theorem run_sub_ok [Sub α] [Zero α] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sub (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.sub (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.sub (t := tt) (s := s) aId bId) h

theorem run_mul_ok [Mul α] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.mul (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.mul (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.mul (t := tt) (s := s) aId bId) h

theorem run_div_ok [Context α] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.div (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.div (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.div (t := tt) (s := s) aId bId) h

theorem run_scale_ok [Mul α] {s : Shape} (xId : Nat) (c : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.scale (t := t) (s := s) xId c = .ok (t', id)) :
    (TapeM.scale (s := s) xId c).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.scale (t := tt) (s := s) xId c) h

theorem run_abs_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.abs (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.abs (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.abs (t := tt) (s := s) xId) h

theorem run_sqrt_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sqrt (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sqrt (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.sqrt (t := tt) (s := s) xId) h

theorem run_clamp_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (xId : Nat) (minVal maxVal : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.clamp (t := t) (s := s) xId minVal maxVal = .ok (t', id)) :
    (TapeM.clamp (s := s) xId minVal maxVal).run t = .ok (id, t') :=
  record_run_ok
    (op := fun tt => Runtime.Autograd.Tape.clamp (t := tt) (s := s) xId minVal maxVal) h

theorem run_max_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.max (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.max (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.max (t := tt) (s := s) aId bId) h

theorem run_min_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape}
    (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.min (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.min (s := s) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.min (t := tt) (s := s) aId bId) h

theorem run_relu_ok [Mul α] [Zero α] [Max α] [BEq α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.relu (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.relu (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.relu (t := tt) (s := s) xId) h

theorem run_linear_ok [Add α] [Mul α] [Zero α] {inDim outDim : Nat} (wId bId xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.linear (t := t) (inDim := inDim) (outDim := outDim) wId bId
      xId = .ok (t', id)) :
    (TapeM.linear (inDim := inDim) (outDim := outDim) wId bId xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.linear (t := tt) (inDim := inDim)
    (outDim := outDim) wId bId xId) h

theorem run_matmul_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {m n p : Nat}
    (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.matmul (t := t) (m := m) (n := n) (p := p) aId bId = .ok (t', id)) :
    (TapeM.matmul (m := m) (n := n) (p := p) aId bId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.matmul (t := tt) (m := m) (n := n) (p := p)
    aId bId) h

theorem run_conv_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)] {d inC outC : Nat}
    {kernel stride padding inSpatial : TorchLean.Tensor Nat [d]}
    (kernelId biasId inputId : Nat) (name : String) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.conv (t := t) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      kernelId biasId inputId (name := name) = .ok (t', id)) :
    (TapeM.conv (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) kernelId biasId inputId name).run t
      = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.conv (t := tt) (d := d) (inC := inC)
    (outC := outC) (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name)) h

theorem run_convTranspose_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inC outC : Nat} {kernel stride padding : TorchLean.Tensor Nat [d]}
    {inSpatial : TorchLean.Tensor Nat [d]} (kernelId biasId inputId : Nat) (name : String)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.convTranspose (t := t) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial) kernelId
      biasId inputId (name := name) = .ok (t', id)) :
    (TapeM.convTranspose (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) kernelId biasId inputId name).run t
      = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.convTranspose (t := tt) (d := d)
    (inC := inC) (outC := outC) (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name)) h

theorem run_maxPool_ok [Context α] {d C : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]} (xId : Nat) {t t' : Tape α}
    {id : Nat}
    (h : Runtime.Autograd.Tape.maxPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) xId = .ok (t', id)) :
    (TapeM.maxPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
      (padding := padding) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.maxPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId) h

theorem run_smoothMaxPool_ok [Context α] [DecidableEq α] {d C : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]} (xId : Nat) (beta : α)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.smoothMaxPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) xId beta = .ok (t', id)) :
    (TapeM.smoothMaxPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) xId beta).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.smoothMaxPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId beta) h

theorem run_avgPool_ok [Context α] {d C : Nat}
    {inSpatial kernel stride padding : TorchLean.Tensor Nat [d]} (xId : Nat) {t t' : Tape α}
    {id : Nat}
    (h : Runtime.Autograd.Tape.avgPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) xId = .ok (t', id)) :
    (TapeM.avgPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
      (padding := padding) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.avgPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) xId) h

theorem run_layerNorm_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
    (xId gammaId betaId : Nat) (epsilon : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.layerNorm (t := t) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos) xId gammaId betaId
      (epsilon := epsilon) = .ok (t', id)) :
    (TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) h_seq_pos h_embed_pos xId gammaId
      betaId epsilon).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.layerNorm (t := tt) (seqLen := seqLen)
    (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos) xId gammaId
    betaId (epsilon := epsilon)) h

theorem run_batchNorm_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed) (xId gammaId betaId : Nat)
    (epsilon : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.batchNorm (t := t) (channels := channels) (sSpatial := sSpatial)
      hWellFormed xId gammaId betaId (epsilon := epsilon) = .ok (t', id)) :
    (TapeM.batchNorm (channels := channels) (sSpatial := sSpatial) hWellFormed xId gammaId
      betaId epsilon).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.batchNorm (t := tt) (channels := channels)
    (sSpatial := sSpatial) hWellFormed xId gammaId betaId (epsilon := epsilon)) h

theorem run_attention_ok [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) (wqId wkId wvId woId xId : Nat)
    (mask : Option (Tensor Bool [n, n])) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.attention (t := t) (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) (h1 := h1) wqId wkId wvId woId xId
      mask = .ok (t', id)) :
    (TapeM.attention (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) h1
      wqId wkId wvId woId xId mask).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.attention (t := tt) (n := n)
    (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1) wqId wkId wvId woId
    xId mask) h

theorem run_mseLoss_ok [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [NatCast α] {s : Shape}
    (yhatId targetId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.mseLoss (t := t) (s := s) yhatId targetId = .ok (t', id)) :
    (TapeM.mseLoss (s := s) yhatId targetId).run t = .ok (id, t') :=
  record_run_ok
    (op := fun tt => Runtime.Autograd.Tape.mseLoss (t := tt) (s := s) yhatId targetId) h

theorem run_sigmoid_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sigmoid (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sigmoid (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.sigmoid (t := tt) (s := s) xId) h

theorem run_tanh_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.tanh (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.tanh (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.tanh (t := tt) (s := s) xId) h

theorem run_softmaxLast_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.softmaxLast (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.softmaxLast (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.softmaxLast (t := tt) (s := s) xId) h

theorem run_softplus_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.softplus (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.softplus (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.softplus (t := tt) (s := s) xId) h

theorem run_exp_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.exp (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.exp (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.exp (t := tt) (s := s) xId) h

theorem run_sin_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sin (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sin (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.sin (t := tt) (s := s) xId) h

theorem run_cos_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.cos (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.cos (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.cos (t := tt) (s := s) xId) h

theorem run_log_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.log (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.log (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.log (t := tt) (s := s) xId) h

theorem run_inv_ok [Context α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.inv (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.inv (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.inv (t := tt) (s := s) xId) h

theorem run_safeLog_ok [Context α] {s : Shape} (xId : Nat) (ε : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.safeLog (t := t) (s := s) xId ε = .ok (t', id)) :
    (TapeM.safeLog (s := s) xId ε).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.safeLog (t := tt) (s := s) xId ε) h

theorem run_sum_ok [Add α] [Zero α] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sum (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sum (s := s) xId).run t = .ok (id, t') :=
  record_run_ok (op := fun tt => Runtime.Autograd.Tape.sum (t := tt) (s := s) xId) h

end Builder
end Autograd
end Proofs
