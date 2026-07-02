"""Shared bits for the small PINN trainers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - fail fast when torch missing
    raise SystemExit("PyTorch is required: pip install torch") from exc

from safe_expr import eval_expr


class PinnDataset:
    """Keeps each JSON section as a tensor on the chosen torch device."""

    def __init__(self, device: torch.device):
        self.device = device
        self.sections: Dict[str, Optional[torch.Tensor]] = {}

    @staticmethod
    def _read_entries(entries, keys: Sequence[str], device: torch.device) -> Optional[torch.Tensor]:
        if entries is None:
            return None
        if not isinstance(entries, list):
            raise ValueError("Dataset sections must be lists of objects.")
        rows: list[list[float]] = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"Dataset entry {idx} is not an object.")
            try:
                row = [float(entry[k]) for k in keys]
            except KeyError as exc:
                raise ValueError(f"Dataset entry {idx} missing key '{exc.args[0]}'") from exc
            rows.append(row)
        if not rows:
            return None
        return torch.tensor(rows, dtype=torch.float32, device=device)

    @classmethod
    def load(
        cls,
        path: str,
        schema: Mapping[str, Sequence[str]],
        device: torch.device,
    ) -> "PinnDataset":
        payload = json.loads(Path(path).read_text())
        data = cls(device)
        for section, keys in schema.items():
            data.sections[section] = cls._read_entries(payload.get(section), keys, device)
        return data

    def sample(self, section: str, count: int) -> Optional[torch.Tensor]:
        mat = self.sections.get(section)
        if mat is None:
            return None
        if mat.shape[0] == 0:
            raise ValueError(f"Dataset section '{section}' is empty; cannot sample.")
        idx = torch.randint(0, mat.shape[0], (count,), device=self.device, dtype=torch.long)
        return mat.index_select(0, idx)

    def sample_columns(self, section: str, count: int, columns: int) -> Optional[tuple[torch.Tensor, ...]]:
        samples = self.sample(section, count)
        if samples is None:
            return None
        return tuple(samples[:, i : i + 1] for i in range(columns))


def _activation_factory(name: str) -> nn.Module:
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation '{name}'")


def parse_hidden_widths(raw: str) -> List[int]:
    tokens = [tok.strip() for tok in raw.split(",")]
    widths: List[int] = []
    for tok in tokens:
        if not tok:
            continue
        try:
            width = int(tok)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid hidden width '{tok}'") from exc
        if width <= 0:
            raise argparse.ArgumentTypeError(f"Hidden width must be positive, got {width}")
        widths.append(width)
    if not widths:
        raise argparse.ArgumentTypeError("Provide at least one hidden layer width (e.g., '16,16').")
    return widths


def build_model(in_dim: int, hidden_widths: Iterable[int], activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    act = activation.lower()
    for width in hidden_widths:
        lin = nn.Linear(prev, width)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers.append(lin)
        layers.append(_activation_factory(act))
        prev = width
    out = nn.Linear(prev, 1)
    nn.init.xavier_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


def to_json_dict(model: nn.Sequential, *, meta: Dict[str, Any]) -> Dict[str, Any]:
    exported: Dict[str, Any] = {}
    for name, tensor in model.state_dict().items():
        if name.endswith(".weight") or name.endswith(".bias"):
            exported[f"layers.{name}"] = tensor.detach().cpu().numpy().tolist()
    exported["meta"] = meta
    return exported


def gradients(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        retain_graph=True,
        create_graph=True,
    )[0]


def eval_pinn_expr(expr: str, **tensors):
    try:
        value = eval_expr(expr, tensors)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ValueError(f"Failed to evaluate expression '{expr}': {exc}") from exc
    return value


def ensure_tensor(val, like: torch.Tensor) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val.to(like)
    arr = torch.as_tensor(val, dtype=like.dtype, device=like.device)
    if arr.numel() == 1:
        return torch.full_like(like, arr.item())
    return arr.reshape_as(like)


def parse_const_flags(items) -> Dict[str, float]:
    constants: Dict[str, float] = {}
    for raw in items:
        if "=" not in raw:
            raise ValueError(f"--const expects name=value, got '{raw}'")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid constant name in '{raw}'")
        try:
            constants[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid constant value in '{raw}'") from exc
    return constants


def export_model(model: nn.Sequential, *, out_ckpt: str, out_json: str, hidden_widths, activation: str) -> None:
    ckpt_path = Path(out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(ckpt_path))
    print(f"Saved checkpoint: {ckpt_path}")

    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": list(hidden_widths),
        "activation": activation,
    }
    json_path.write_text(json.dumps(to_json_dict(model, meta=meta)))
    print(f"Exported weights JSON: {json_path}")
