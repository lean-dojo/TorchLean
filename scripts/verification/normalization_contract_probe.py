#!/usr/bin/env python3
"""Measure PyTorch normalization residuals against the BugZoo constant-slice contracts.

The real-valued reference has output equal to bias and zero input/scale gradients for the
sum-of-outputs loss used here. This diagnostic prints residuals, not a conformance certificate.
Install PyTorch separately; the Lean build does not require it.
"""

import argparse
import json

import torch
import torch.nn.functional as functional


def probe(case: str, magnitude: float, dtype: torch.dtype, device: str) -> dict:
    """Run one constant input through forward and backward without masking residuals."""
    shape = (1, 1) if case == "one_feature_layer_norm" else (2, 2, 8)
    parameter_size = shape[-1] if case.endswith("layer_norm") else shape[1]
    x = torch.full(shape, magnitude, dtype=dtype, device=device, requires_grad=True)
    weight = torch.full((parameter_size,), 2.0, dtype=dtype, device=device, requires_grad=True)
    bias = torch.full((parameter_size,), 3.0, dtype=dtype, device=device, requires_grad=True)
    if case.endswith("layer_norm"):
        output = functional.layer_norm(x, (shape[-1],), weight, bias, eps=1e-5)
    elif case == "group_norm":
        output = functional.group_norm(x, 2, weight, bias, eps=1e-5)
    elif case == "instance_norm":
        output = functional.instance_norm(x, weight=weight, bias=bias,
                                          use_input_stats=True, eps=1e-5)
    else:
        output = functional.batch_norm(x, None, None, weight, bias, training=True, eps=1e-5)
    output.sum().backward()
    residuals = {
        "output_minus_bias_max_abs": (output.detach() - 3).abs().max().item(),
        "scale_gradient_max_abs": weight.grad.abs().max().item(),
        "input_gradient_max_abs": x.grad.abs().max().item(),
    }
    return {"case": case, "dtype": str(dtype), "magnitude": magnitude, **residuals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requires an available CUDA device")
    print(json.dumps({"torch": torch.__version__, "device": args.device,
                      "loss": "sum of outputs", "epsilon": 1e-5}))
    for dtype in (torch.float32, torch.float64):
        for magnitude in (1.0, 1e3, 1e6, 1e7):
            for case in ("one_feature_layer_norm", "constant_layer_norm", "group_norm",
                         "instance_norm", "batch_norm"):
                print(json.dumps(probe(case, magnitude, dtype, args.device)))


if __name__ == "__main__":
    main()
