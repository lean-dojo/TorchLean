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

Every op wrapper that threads the tape is definitionally `TapeM.opM` applied to its pure
counterpart, so the whole surface reduces through one lemma:

- `opM_run_ok` / `opM_run_error` turn a pure op's outcome into the monadic `run`'s outcome, and
  `opM_run_inv` goes back. Note the pair swaps: a pure op returns tape-then-id, `run` returns
  value-then-state.
- `run_bind_inv` splits a successful `run` of `m >>= f` into its two successful stages, which is
  what peels a `do`-block one statement at a time, and `exec_inv` turns a successful `exec` into
  such a `run`.
- The `run_<op>_ok` family below is `opM_run_ok` instantiated at each op's own pure counterpart.
  Each is a definitional instantiation — the proof term is `opM_run_ok` with `g` supplied, with no
  unfolding lemma in between — so the family stays correct by construction as ops are added.

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

open Spec
open Tensor
open Runtime.Autograd (Tape TapeM Result)

variable {α : Type}

/-! ## The shared reshuffle, reduced once -/

/-- A successful pure op gives a successful monadic run. The pair swaps: the pure op returns
tape-then-value, `run` returns value-then-state. -/
theorem opM_run_ok {γ : Type} {g : Tape α → Result (Tape α × γ)} {t t' : Tape α} {v : γ}
    (h : g t = .ok (t', v)) :
    (TapeM.opM g).run t = .ok (v, t') := by
  simp only [TapeM.opM, TapeM.run, StateT.run, bind, StateT.bind, StateT.lift, liftM, monadLift,
    MonadLift.monadLift, MonadState.get, MonadStateOf.get, getThe, StateT.get,
    set, MonadStateOf.set, StateT.set, pure, StateT.pure, Except.bind, Except.pure, h]

/-- A failing pure op gives a failing monadic run. -/
theorem opM_run_error {γ : Type} {g : Tape α → Result (Tape α × γ)} {t : Tape α} {e : String}
    (h : g t = .error e) :
    (TapeM.opM g).run t = .error e := by
  simp only [TapeM.opM, TapeM.run, StateT.run, bind, StateT.bind, StateT.lift, liftM, monadLift,
    MonadLift.monadLift, MonadState.get, MonadStateOf.get, getThe, StateT.get,
    pure, Except.bind, Except.pure, h]

/-- Inversion: a successful monadic run means the pure op succeeded. -/
theorem opM_run_inv {γ : Type} {g : Tape α → Result (Tape α × γ)} {t t' : Tape α} {v : γ}
    (h : (TapeM.opM g).run t = .ok (v, t')) :
    g t = .ok (t', v) := by
  cases hg : g t with
  | ok p =>
      obtain ⟨t1, v1⟩ := p
      rw [opM_run_ok (g := g) (t := t) (t' := t1) (v := v1) hg] at h
      injection h with h'
      have h1 : v1 = v := congrArg Prod.fst h'
      have h2 : t1 = t' := congrArg Prod.snd h'
      subst h1
      subst h2
      rfl
  | error e =>
      rw [opM_run_error (g := g) hg] at h
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

/-- A successful `exec` is a successful `run` that returned some value. -/
theorem exec_inv {β : Type} {m : TapeM α β} {t t' : Tape α}
    (h : TapeM.exec t m = .ok t') :
    ∃ b, m.run t = .ok (b, t') := by
  cases hm : StateT.run m t with
  | error e =>
      simp only [TapeM.exec, TapeM.run, bind, Except.bind, hm] at h
      cases h
  | ok bt =>
      obtain ⟨b1, b2⟩ := bt
      simp only [TapeM.exec, TapeM.run, bind, Except.bind, pure, Except.pure, hm] at h
      injection h with h'
      exact ⟨b1, by show StateT.run m t = _; rw [hm, h']⟩

/-! ## `leaf`, which is total -/

/-- `TapeM.leaf` cannot fail, so its run is unconditional. -/
theorem run_leaf {s : Shape} (value : Tensor α s) (name : Option String := none)
    (requiresGrad : Bool := true) {t : Tape α} :
    (TapeM.leaf (α := α) (s := s) value name requiresGrad).run t
      = .ok ((Tape.leaf (t := t) value (name := name) (requiresGrad := requiresGrad)).2,
             (Tape.leaf (t := t) value (name := name) (requiresGrad := requiresGrad)).1) := by
  simp only [TapeM.leaf, TapeM.run, StateT.run, bind, StateT.bind, MonadState.get,
    MonadStateOf.get, getThe, StateT.get, set, MonadStateOf.set, StateT.set, pure, StateT.pure,
    Except.bind, Except.pure]

/-! ## The per-op family: `opM_run_ok` at each op's own pure counterpart

Each proof below is `opM_run_ok` with `g` supplied and nothing else. If one of them ever stops
typechecking, the wrapper it names has stopped being `opM` at its pure op — which is the fact
worth learning.
-/

theorem run_add_ok {α : Type} [Add α] [DecidableEq Shape] {s : Shape} (aId bId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.add (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.add (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.add (t := tt) (s := s) aId bId) h

theorem run_sub_ok {α : Type} [Sub α] [Zero α] [DecidableEq Shape] {s : Shape} (aId bId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sub (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.sub (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.sub (t := tt) (s := s) aId bId) h

theorem run_mul_ok {α : Type} [Mul α] [DecidableEq Shape] {s : Shape} (aId bId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.mul (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.mul (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.mul (t := tt) (s := s) aId bId) h

theorem run_div_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (aId bId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.div (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.div (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.div (t := tt) (s := s) aId bId) h

theorem run_scale_ok {α : Type} [Mul α] [DecidableEq Shape] {s : Shape} (xId : Nat) (c : α)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.scale (t := t) (s := s) xId c = .ok (t', id)) :
    (TapeM.scale (s := s) xId c).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.scale (t := tt) (s := s) xId c) h

theorem run_abs_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.abs (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.abs (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.abs (t := tt) (s := s) xId) h

theorem run_sqrt_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {s : Shape} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sqrt (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sqrt (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.sqrt (t := tt) (s := s) xId) h

theorem run_clamp_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {s : Shape} (xId : Nat) (minVal maxVal : α) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.clamp (t := t) (s := s) xId minVal maxVal = .ok (t', id)) :
    (TapeM.clamp (s := s) xId minVal maxVal).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.clamp (t := tt) (s := s) xId minVal maxVal) h

theorem run_max_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.max (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.max (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.max (t := tt) (s := s) aId bId) h

theorem run_min_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {s : Shape} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.min (t := t) (s := s) aId bId = .ok (t', id)) :
    (TapeM.min (s := s) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.min (t := tt) (s := s) aId bId) h

theorem run_relu_ok {α : Type} [Mul α] [Zero α] [Max α] [One α] [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.relu (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.relu (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.relu (t := tt) (s := s) xId) h

theorem run_linear_ok {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape] {inDim outDim : Nat}
    (wId bId xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.linear (t := t) (inDim := inDim) (outDim := outDim) wId bId
      xId = .ok (t', id)) :
    (TapeM.linear (inDim := inDim) (outDim := outDim) wId bId xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.linear (t := tt) (inDim := inDim)
    (outDim := outDim) wId bId xId) h

theorem run_matmul_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {m n p : Nat} (aId bId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.matmul (t := t) (m := m) (n := n) (p := p) aId bId = .ok (t', id)) :
    (TapeM.matmul (m := m) (n := n) (p := p) aId bId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.matmul (t := tt) (m := m) (n := n) (p := p) aId
    bId) h

theorem run_conv_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {d inC outC : Nat} {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (kernelId biasId inputId : Nat) (name : String) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.conv (t := t) (d := d) (inC := inC) (outC := outC) (kernel := kernel)
      (stride := stride) (padding := padding) (inSpatial := inSpatial) kernelId biasId inputId
      (name := name) = .ok (t', id)) :
    (TapeM.conv (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) kernelId biasId inputId
      name).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.conv (t := tt) (d := d) (inC := inC)
    (outC := outC) (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name)) h

theorem run_convTranspose_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {d inC outC : Nat} {kernel stride padding : Spec.Tensor Nat [d]}
    {inSpatial : Spec.Tensor Nat [d]} (kernelId biasId inputId : Nat) (name : String)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.convTranspose (t := t) (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial) kernelId
      biasId inputId (name := name) = .ok (t', id)) :
    (TapeM.convTranspose (d := d) (inC := inC) (outC := outC) (kernel := kernel) (stride := stride)
      (padding := padding) (inSpatial := inSpatial) kernelId biasId inputId
      name).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.convTranspose (t := tt) (d := d) (inC := inC)
    (outC := outC) (kernel := kernel) (stride := stride) (padding := padding)
    (inSpatial := inSpatial) kernelId biasId inputId (name := name)) h

theorem run_maxPool_ok {α : Type} [Context α] [DecidableEq Shape] {d C : Nat}
    {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.maxPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) (hKernel := hKernel)
      xId = .ok (t', id)) :
    (TapeM.maxPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
      (padding := padding) (hKernel := hKernel) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.maxPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel) xId) h

theorem run_smoothMaxPool_ok {α : Type} [Context α] [DecidableEq α] [DecidableEq Shape] {d C : Nat}
    {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} (xId : Nat) (beta : α) {t t' : Tape α}
    {id : Nat}
    (h : Runtime.Autograd.Tape.smoothMaxPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) (hKernel := hKernel) xId
      beta = .ok (t', id)) :
    (TapeM.smoothMaxPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel)
      (stride := stride) (padding := padding) (hKernel := hKernel) xId beta).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.smoothMaxPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel) xId beta) h

theorem run_avgPool_ok {α : Type} [Context α] [DecidableEq Shape] {d C : Nat}
    {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0) (xId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.avgPool (t := t) (d := d) (C := C) (inSpatial := inSpatial)
      (kernel := kernel) (stride := stride) (padding := padding) hKernel xId = .ok (t', id)) :
    (TapeM.avgPool (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
      (padding := padding) hKernel xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.avgPool (t := tt) (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding) hKernel
    xId) h

theorem run_layerNorm_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0)
    (h_embed_pos : embedDim > 0) (xId gammaId betaId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.layerNorm (t := t) (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos) xId gammaId betaId = .ok (t', id)) :
    (TapeM.layerNorm (seqLen := seqLen) (embedDim := embedDim) h_seq_pos h_embed_pos xId gammaId
      betaId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.layerNorm (t := tt) (seqLen := seqLen)
    (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos) xId gammaId
    betaId) h

theorem run_batchNorm_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed) (xId gammaId betaId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.batchNorm (t := t) (channels := channels) (sSpatial := sSpatial)
      hWellFormed xId gammaId betaId = .ok (t', id)) :
    (TapeM.batchNorm (channels := channels) (sSpatial := sSpatial) hWellFormed xId gammaId
      betaId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.batchNorm (t := tt) (channels := channels)
    (sSpatial := sSpatial) hWellFormed xId gammaId betaId) h

theorem run_multiHeadAttention_ok {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    [DecidableEq Shape] {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wqId wkId wvId woId xId : Nat) (mask : Option (Tensor Bool [n, n])) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.multiHeadAttention (t := t) (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) (h1 := h1) wqId wkId wvId woId xId
      mask = .ok (t', id)) :
    (TapeM.multiHeadAttention (n := n) (numHeads := numHeads) (dModel := dModel)
      (headDim := headDim) h1 wqId wkId wvId woId xId mask).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.multiHeadAttention (t := tt) (n := n)
    (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1) wqId wkId wvId woId
    xId mask) h

theorem run_mseLoss_ok {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α]
    [DecidableEq Shape] {s : Shape} (yhatId targetId : Nat) {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.mseLoss (t := t) (s := s) yhatId targetId = .ok (t', id)) :
    (TapeM.mseLoss (s := s) yhatId targetId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.mseLoss (t := tt) (s := s) yhatId targetId) h

theorem run_sigmoid_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sigmoid (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sigmoid (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.sigmoid (t := tt) (s := s) xId) h

theorem run_tanh_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.tanh (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.tanh (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.tanh (t := tt) (s := s) xId) h

theorem run_softmaxLast_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.softmaxLast (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.softmaxLast (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.softmaxLast (t := tt) (s := s) xId) h

theorem run_softplus_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.softplus (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.softplus (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.softplus (t := tt) (s := s) xId) h

theorem run_exp_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.exp (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.exp (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.exp (t := tt) (s := s) xId) h

theorem run_log_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.log (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.log (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.log (t := tt) (s := s) xId) h

theorem run_inv_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.inv (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.inv (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.inv (t := tt) (s := s) xId) h

theorem run_safeLog_ok {α : Type} [Context α] [DecidableEq Shape] {s : Shape} (xId : Nat) (ε : α)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.safeLog (t := t) (s := s) xId ε = .ok (t', id)) :
    (TapeM.safeLog (s := s) xId ε).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.safeLog (t := tt) (s := s) xId ε) h

theorem run_sum_ok {α : Type} [Add α] [Zero α] [DecidableEq Shape] {s : Shape} (xId : Nat)
    {t t' : Tape α} {id : Nat}
    (h : Runtime.Autograd.Tape.sum (t := t) (s := s) xId = .ok (t', id)) :
    (TapeM.sum (s := s) xId).run t = .ok (id, t') :=
  opM_run_ok (g := fun tt => Runtime.Autograd.Tape.sum (t := tt) (s := s) xId) h

end Builder
end Autograd
end Proofs
