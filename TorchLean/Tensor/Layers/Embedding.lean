/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.AdjacentSwap
public import TorchLean.Tensor.Contract
public import TorchLean.Tensor.Scatter

/-!
# Embeddings

An embedding is a table whose rows are indexed by a finite vocabulary.
`Tensor.embedding weight indices` gathers the rows of `weight : Tensor α [num, dim]` that an index
tensor of any shape `t` names, giving `Tensor α (t ++ [dim])`, as PyTorch's `functional.embedding`
does. The indices have type `Fin num`, so every token id is in range by construction.
`Tensor.embeddingBackward indices dY` is its gradient for the table: a scatter-add of the rows of
`dY` into a zero table, in row-major order of `indices`, as ATen's `embedding_dense_backward`.

Over a semiring, `Tensor.matmul` of one-hot rows with the table gives the rows that
`Tensor.embedding` looks up. In a binary format it does not: each entry of the product starts from
`+0` and adds `0 * w` for the entry `w` of every other row, so a `±∞` or NaN elsewhere in the
column gives NaN, and a selected `-0` comes back as `+0`.

In torch terms the lookup is `torch.nn.functional.embedding` with every option at its default: no
`padding_idx`, no `max_norm` renormalization, no `scale_grad_by_freq`, and a dense gradient
(`sparse=False`). We wrote both functions by hand.

References / analogies:
- In most ML frameworks, an embedding table is a matrix `weight : (vocab x embedDim)` and an
  index-based lookup returns `weight[token_id]`.
- Bengio, Ducharme, Vincent and Jauvin, "A Neural Probabilistic Language Model", JMLR 3:1137-1155
  (2003), for the classic embedding-table framing in neural language models.
- Mikolov, Chen, Corrado and Dean, "Efficient Estimation of Word Representations in Vector Space"
  (2013), arXiv:1301.3781, for the modern word-embedding perspective.
- PyTorch API reference (PyTorch 2.8 docs):
  - `torch.nn.Embedding`: https://docs.pytorch.org/docs/2.8/generated/torch.nn.Embedding.html
  - `torch.nn.functional.embedding`:
    https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.embedding.html
  - `torch.nn.functional.one_hot`:
    https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.one_hot.html
-/

@[expose] public section


namespace TorchLean.Tensor

variable {α : Type} [Storage α]

/-- The rows of `weight` that `indices` name: the gather of `weight` along its first axis, with the
row width appended to the shape of `indices`. -/
def embedding {num dim : Nat} (weight : Tensor α [num, dim]) {t : Shape}
    (indices : Tensor (Fin num) t) : Tensor α (t ++ [dim]) :=
  gather weight (Shape.selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩) (Shape.noBatch [num, dim] t)
    (map (fun i k => Fin.cast (congrArg (Shape.extent _)
      (Shape.get_selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩ k).symm) i) indices)
    (t ++ [dim]) ⟨List.nodup_singleton _, List.nodup_nil, fun _ h => nomatch h⟩ rfl

/-- The gradient of `embedding` for the table: each row is the left fold from zero of the rows of
`dY` whose index names it, in row-major order of `indices`. -/
def embeddingBackward [Add α] [Zero α] {num dim : Nat} {t : Shape}
    (indices : Tensor (Fin num) t) (dY : Tensor α (t ++ [dim])) : Tensor α [num, dim] :=
  scatter .sum (full [num, dim] 0) (Shape.selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩)
    (map (fun i => some fun k => Fin.cast (congrArg (Shape.extent _)
      (Shape.get_selectDims [num, dim] ⟨0, Nat.zero_lt_two⟩ k).symm) i) indices) dY

end TorchLean.Tensor
