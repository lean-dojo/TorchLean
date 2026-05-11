/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Tensor constructors (spec layer)

These are small, **total** constructors for building `Spec.Tensor` values directly.

They are used heavily inside the spec layer (models/layers) and in proofs, where we want:

- straightforward definitional unfolding, and
- no dependence on `IO` or dynamic shape checks.

If you want a PyTorch-style user experience for examples (dynamic dims, runtime errors, and
Float-literal casting), prefer `NN/Tensor/API.lean` instead.

Design choice (why these are "total"):

- In the spec layer we would rather make edge cases explicit than throw runtime exceptions.
- If something is shape-invalid, we want Lean to reject it at elaboration time.
- If something is index-invalid at runtime (e.g. array-backed constructor with wrong size), we
  choose a predictable fallback (usually `Inhabited.default`) and let *verification* code decide
  whether that situation is allowed.
-/

@[expose] public section


namespace Spec

/-- Fill a tensor of shape `s` with a constant value.

PyTorch analogy: `torch.full(shape, value)`.
-/
def fill {α : Type} (value : α): (s : Shape) → Tensor α s
  | Shape.scalar => Tensor.scalar value
  | Shape.dim _ s' => Tensor.dim (fun _ => fill value s')

/-- Scalar constructor (explicit name).

PyTorch analogy: a `0`-dim tensor holding one value.
-/
def scalarTensor {α : Type} (value : α) : Tensor α .scalar :=
  Tensor.scalar value

/-- Vector constructor from a `Fin n → α` function.

PyTorch analogy: `torch.tensor([...])` with shape `(n,)`, but our input is a function, not a list.
-/
def vectorTensor {α : Type} {n : Nat} (values : Fin n → α) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (values i))

/-- Matrix constructor from an `Fin m → Fin n → α` function.

PyTorch analogy: `torch.tensor([...]).reshape(m, n)` (again, function input rather than a list).
-/
def matrixTensor {α : Type} {m n : Nat} (values : Fin m → Fin n → α) : Tensor α (.dim m (.dim n
  .scalar)) :=
  Tensor.dim (fun i => vectorTensor (fun j => values i j))

/-- Generic `dim` constructor (kept for clarity at call sites). -/
def nDArrayTensor {α : Type} : ∀ {n : Nat} {s : Shape}, (Fin n → Tensor α s) → Tensor α (.dim n
  s)
  | _, _, values => Tensor.dim values

/--
Generic vector creation that works with any type (handles `n = 0`).

Why this exists:
- For `n = 0`, a function `Fin 0 → α` is fine (it has no inputs), but it can be awkward at call
  sites. This helper makes the intent explicit and keeps patterns uniform.
- The `0` case is definitionally a tensor with an empty outer dimension (so the function body is
  never evaluated).
-/
def vectorN {α : Type} [Zero α] (n : Nat) (f : Fin n → α) : Tensor α (.dim n .scalar) :=
  match n with
  | 0   => Tensor.dim (fun _ => Tensor.scalar (Zero.zero : α))
  | _   => Tensor.dim (fun i => Tensor.scalar (f i))

/-- Generic matrix creation that works with any type. -/
def matrixMN {α : Type} [Zero α] (m n : Nat) (f : Fin m → Fin n → α) :
  Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (f i j)))

/-- Build a length-`n` vector from a `Fin n → scalar` function.

This is a small wrapper that reads nicely at call sites where we already have scalar tensors.
-/
def generate {α : Type} (n : ℕ) (f : Fin n → Tensor α .scalar) : Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i => f i)

/-- A singleton vector.

PyTorch analogy: `x.unsqueeze(0)` for a scalar `x`.
-/
def singleton {α : Type} (x : α) : Tensor α (.dim 1 .scalar) :=
  Tensor.dim (fun _ => Tensor.scalar x)

/--
Pad a tensor with `n` leading dimensions of size 1.

This is the tensor-level companion of `Shape.padLeft`. It is useful for broadcasting-style
normalization: if you need a tensor to have extra leading batch dimensions of size `1`, this does
so without changing any underlying values.

PyTorch analogy: repeated `unsqueeze(0)` (or viewing a tensor as having extra leading singleton
  dims).
-/
def padLeft {α : Type} [Context α]
  {n : Nat} {s : Shape} (x : Tensor α s)
  : Tensor α (Shape.padLeft n s) :=
  match n with
  | 0 => x
  | Nat.succ _ =>
    let inner := padLeft x
    .dim (fun _ => inner)  -- Only 1 element along new dim

/--
Build a vector tensor from an array.

We require an explicit proof that the target length matches the array size. This keeps the spec
construction honest and makes mismatches obvious at call sites.
-/
def Tensor.ofArray1D {α : Type} {n : Nat} (xs : Array α) (h : n = xs.size) :
    Tensor α (.dim n .scalar) :=
  Tensor.dim (fun i : Fin n =>
    Tensor.scalar (xs[i.val]'(by simpa [h] using i.2)))

/--
Build a matrix tensor from a flat array (row-major).

PyTorch analogy: `xss.reshape(m, n)` assuming `xss` is laid out row-major.
-/
def Tensor.ofArray2D {α : Type} [Inhabited α] {m n : Nat} (xss : Array α) (_ : m * n = xss.size):
  Tensor α (.dim m (.dim n .scalar)) :=
  Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (xss.getD (i.val * n + j.val)
    (Inhabited.default))))

/--
Build a tensor with one leading dimension from an array of inner tensors.

This is the "tensor-of-tensors" analogue of `Tensor.ofArray1D`:

- input: `Array (Tensor α s)` of length `n`
- output: `Tensor α (.dim n s)`

We require an explicit proof that the array size matches `n` so callers do not silently
drop/pad data.
-/
def Tensor.ofArrayDim {α : Type} {n : Nat} {s : Shape}
    (xs : Array (Tensor α s)) (_h : n = xs.size) : Tensor α (.dim n s) :=
  Tensor.dim (fun i : Fin n =>
    xs[i.val]'(by simpa [_h] using i.2))

/--
View a length-`n` vector as an `n x 1` "column" matrix.

PyTorch analogy: `v.unsqueeze(-1)` for a 1D tensor `v` (or `v.reshape(n, 1)`).
-/
def Tensor.vecToCol {α : Type} {n : Nat}
    (v : Tensor α (.dim n .scalar)) : Tensor α (.dim n (.dim 1 .scalar)) :=
  Tensor.dim (fun i : Fin n =>
    Tensor.dim (fun _ : Fin 1 => get v i))

end Spec
