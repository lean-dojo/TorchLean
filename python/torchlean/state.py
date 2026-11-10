"""The TorchLean state document of a PyTorch `state_dict`.

`write(path, model)` writes `model.state_dict()` as

    {"format": "torchlean.state",
     "entries": {"fc.bias": {"dtype": "float32", "shape": [2], "words": [1065353216, 2147483648]}}}

which Lean's `PyTorch.state` reads into `TensorPack α`, keyed by the same names, when every
entry has the dtype `PyTorch.dtype α`. Each entry holds the dtype's name, the shape and the
unsigned word of every element in the dtype's own width, in row-major order, so `1065353216` is the
binary32 word of `1.0` and `2147483648` that of `-0.0`. Negative zeros, subnormals and NaN payloads
are written as they are stored, and a `bool` element as 0 or 1. The dtypes are the 13 names of
`DTYPES`. `read(path)` returns the tensors of a document with the same words, ready for
`module.load_state_dict`.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path

import torch

from ._table import DTYPES

FORMAT = "torchlean.state"

# The signed integer dtype of each element size in bytes. Viewing a tensor as the signed dtype of
# its width and masking gives its words, for every dtype of `DTYPES` other than `bool`.
_SIGNED = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _dtype_name(name: str, dtype: torch.dtype) -> str:
    text = str(dtype).removeprefix("torch.")
    if text not in DTYPES:
        raise ValueError(f"torchlean.state: {name} has the dtype {text}, not one of {DTYPES}")
    return text


def _words(tensor: torch.Tensor) -> list[int]:
    """Return the unsigned word of every element of `tensor` in its dtype's width, in row-major
    order, where a `bool` element gives 0 or 1 even when its byte holds another nonzero value."""
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    if flat.dtype == torch.bool:
        return [int(item) for item in flat.tolist()]
    mask = (1 << (8 * flat.element_size())) - 1
    return [word & mask for word in flat.view(_SIGNED[flat.element_size()]).tolist()]


def _entry(name: str, tensor: object) -> dict:
    if not isinstance(tensor, torch.Tensor):
        kind = type(tensor).__name__
        raise TypeError(f"torchlean.state: {name} is not a tensor but a {kind} value")
    dtype = _dtype_name(name, tensor.dtype)
    return {"dtype": dtype, "shape": list(tensor.shape), "words": _words(tensor)}


def document(source: torch.nn.Module | Mapping[str, torch.Tensor]) -> dict:
    """Return the state document of a module's `state_dict()`, or of a mapping from names to
    tensors, with the entries in the order of the mapping."""
    tensors = source.state_dict() if isinstance(source, torch.nn.Module) else source
    entries = {name: _entry(name, tensor) for name, tensor in tensors.items()}
    return {"format": FORMAT, "entries": entries}


def write(path: str | Path, source: torch.nn.Module | Mapping[str, torch.Tensor]) -> None:
    """Write `document(source)` to `path`, one entry per line."""
    entries = document(source)["entries"]
    lines = [f"  {json.dumps(name)}: {json.dumps(entry)}" for name, entry in entries.items()]
    text = f'{{"format": "{FORMAT}", "entries": {{\n' + ",\n".join(lines) + "\n}}\n"
    Path(path).write_text(text, encoding="utf-8")


def _keys(context: str, value: object, keys: tuple[str, ...]) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"torchlean.state: {context} is not an object")
    if set(value) != set(keys):
        raise ValueError(f"torchlean.state: {context} has the keys {sorted(value)}, "
                         f"expected {list(keys)}")
    return value


def _tensor(name: str, entry: object) -> torch.Tensor:
    entry = _keys(f"the entry {name}", entry, ("dtype", "shape", "words"))
    dtype = entry["dtype"]
    if dtype not in DTYPES:
        raise ValueError(f"torchlean.state: {name} has the dtype {dtype}, not one of {DTYPES}")
    shape, words = entry["shape"], entry["words"]
    if not (isinstance(shape, list) and all(type(extent) is int and extent >= 0
                                            for extent in shape)):
        raise ValueError(f"torchlean.state: the shape of {name} is not a list of naturals")
    if not isinstance(words, list) or len(words) != math.prod(shape):
        raise ValueError(f"torchlean.state: {name} of shape {shape} needs {math.prod(shape)} "
                         "words")
    torch_dtype = getattr(torch, dtype)
    size = torch.empty((), dtype=torch_dtype).element_size()
    width = 8 * size
    top = 1 if torch_dtype == torch.bool else (1 << width) - 1
    if not all(type(word) is int and 0 <= word <= top for word in words):
        raise ValueError(f"torchlean.state: the words of {name} are not naturals of at most {top}")
    signed = [word - (1 << width) if word >> (width - 1) else word for word in words]
    return torch.tensor(signed, dtype=_SIGNED[size]).view(torch_dtype).reshape(shape)


def tensors(state: object) -> dict[str, torch.Tensor]:
    """Return the tensors of a state document by name, each with the dtype, shape and words of its
    entry. Raise `ValueError` unless the document has exactly the keys `format` and `entries`, the
    format `torchlean.state`, and entries with exactly the keys `dtype`, `shape` and `words`, a
    dtype of `DTYPES`, one word per element and every word within the dtype's width. A shape must
    be a list of naturals and a `bool` word 0 or 1."""
    state = _keys("the document", state, ("format", "entries"))
    if state["format"] != FORMAT:
        raise ValueError(f"torchlean.state: the format is {state['format']!r}, not {FORMAT!r}")
    entries = state["entries"]
    if not isinstance(entries, Mapping):
        raise ValueError("torchlean.state: the entries are not an object")
    return {name: _tensor(name, entry) for name, entry in entries.items()}


def read(path: str | Path) -> dict[str, torch.Tensor]:
    """Return `tensors` of the state document at `path`."""
    return tensors(json.loads(Path(path).read_text(encoding="utf-8")))
