#!/usr/bin/env python3
"""
Unified CROWN/IBP Verifier for Lean proofs.

This consolidates crown_compute.py, crown_lyapunov.py, crown_lirpa.py, 
crown_closedloop.py, and convert_twostage.py into a single tool.

Usage:
  python crown_verifier.py verify --model path.pth --region "[-1,1]x[-1,1]" --dynamics van_der_pol
  python crown_verifier.py export --model path.pth --output network.json
  python crown_verifier.py lean-cert --model path.pth --region "[-1,1]x[-1,1]" --dynamics van_der_pol

The tool outputs:
  - JSON certificate with V bounds, gradient bounds, Vdot bounds
  - Lean-formatted output (--format lean) for direct import

IMPORTANT: All bounds are computed via IBP (sound/conservative) or CROWN (when auto_LiRPA available).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

# ============================================================================
# Optional Dependencies
# ============================================================================

HAS_TORCH = False
HAS_LIRPA = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    pass

if HAS_TORCH:
    try:
        from auto_LiRPA import BoundedModule, BoundedTensor
        from auto_LiRPA.perturbations import PerturbationLpNorm
        HAS_LIRPA = True
    except ImportError:
        pass


# ============================================================================
# IBP Core (No Dependencies)
# ============================================================================

def ibp_linear(W: List[List[float]], b: List[float], 
               lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Interval bound propagation for linear layer: y = Wx + b"""
    m, n = len(W), len(W[0])
    out_lo, out_hi = [], []
    for i in range(m):
        lo_i = b[i] if b else 0.0
        hi_i = b[i] if b else 0.0
        for j in range(n):
            a = W[i][j]
            p1 = a * lo[j]
            p2 = a * hi[j]
            lo_i += min(p1, p2)
            hi_i += max(p1, p2)
        out_lo.append(lo_i)
        out_hi.append(hi_i)
    return out_lo, out_hi


def ibp_relu(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """IBP for ReLU"""
    return [max(0.0, x) for x in lo], [max(0.0, x) for x in hi]


def ibp_tanh(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """IBP for tanh (monotone)"""
    return [math.tanh(x) for x in lo], [math.tanh(x) for x in hi]


def ibp_sigmoid(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """IBP for sigmoid (monotone)"""
    def sig(x): 
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    return [sig(x) for x in lo], [sig(x) for x in hi]


# ============================================================================
# Network Class (IBP-based, no dependencies)
# ============================================================================

class IBPNetwork:
    """Neural network with IBP bound propagation"""
    
    def __init__(self, weights: List[List[List[float]]], biases: List[List[float]], 
                 activations: List[str]):
        self.weights = weights
        self.biases = biases
        self.activations = activations
    
    @classmethod
    def from_json(cls, path: str) -> 'IBPNetwork':
        """Load from JSON file"""
        with open(path) as f:
            data = json.load(f)
        return cls(
            weights=data["weights"],
            biases=data["biases"],
            activations=data.get("activations", ["relu"] * len(data["weights"]))
        )
    
    @classmethod
    def from_pytorch(cls, path: str, prefix: str = "lyapunov.") -> 'IBPNetwork':
        """Load from PyTorch checkpoint"""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        state_dict = ckpt.get('state_dict', ckpt)
        
        weights, biases, activations = [], [], []
        i = 0
        while True:
            w_key = f'{prefix}layers.{i}.weight'
            b_key = f'{prefix}layers.{i}.bias'
            
            if w_key not in state_dict:
                break
            
            W = state_dict[w_key].numpy().tolist()
            b = state_dict[b_key].numpy().tolist()
            weights.append(W)
            biases.append(b)
            
            # Determine activation (Two-Stage uses tanh hidden + sigmoid output)
            next_w_key = f'{prefix}layers.{i+2}.weight'
            if next_w_key in state_dict:
                activations.append("tanh")
            else:
                activations.append("sigmoid")
            
            i += 2  # Skip activation layer index
        
        return cls(weights, biases, activations)
    
    def forward(self, x: List[float]) -> List[float]:
        """Point-wise forward pass"""
        z = x[:]
        for W, b, act in zip(self.weights, self.biases, self.activations):
            new_z = []
            for i in range(len(W)):
                s = b[i]
                for j in range(len(z)):
                    s += W[i][j] * z[j]
                new_z.append(s)
            z = new_z
            
            if act == "relu":
                z = [max(0, v) for v in z]
            elif act == "tanh":
                z = [math.tanh(v) for v in z]
            elif act == "sigmoid":
                z = [1/(1+math.exp(-max(-500, min(500, v)))) for v in z]
        return z
    
    def ibp_forward(self, lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
        """IBP forward pass returning output bounds"""
        z_lo, z_hi = lo[:], hi[:]
        
        for W, b, act in zip(self.weights, self.biases, self.activations):
            z_lo, z_hi = ibp_linear(W, b, z_lo, z_hi)
            
            if act == "relu":
                z_lo, z_hi = ibp_relu(z_lo, z_hi)
            elif act == "tanh":
                z_lo, z_hi = ibp_tanh(z_lo, z_hi)
            elif act == "sigmoid":
                z_lo, z_hi = ibp_sigmoid(z_lo, z_hi)
        
        return z_lo, z_hi
    
    def ibp_gradient(self, lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
        """IBP bounds on output gradient ∂y/∂x (for scalar output)"""
        n_in = len(lo)
        
        # Jacobian bounds: J[i][k] = ∂z_i/∂x_k
        J_lo = [[1.0 if i == j else 0.0 for j in range(n_in)] for i in range(n_in)]
        J_hi = [[1.0 if i == j else 0.0 for j in range(n_in)] for i in range(n_in)]
        
        z_lo, z_hi = lo[:], hi[:]
        
        for W, b, act in zip(self.weights, self.biases, self.activations):
            m, n = len(W), len(W[0])
            
            # Pre-activation bounds
            new_z_lo, new_z_hi = ibp_linear(W, b, z_lo, z_hi)
            
            # Update Jacobian: J_new = W @ J
            new_J_lo = [[0.0] * n_in for _ in range(m)]
            new_J_hi = [[0.0] * n_in for _ in range(m)]
            
            for i in range(m):
                for k in range(n_in):
                    val_lo, val_hi = 0.0, 0.0
                    for j in range(n):
                        w = W[i][j]
                        if w >= 0:
                            val_lo += w * J_lo[j][k]
                            val_hi += w * J_hi[j][k]
                        else:
                            val_lo += w * J_hi[j][k]
                            val_hi += w * J_lo[j][k]
                    new_J_lo[i][k] = val_lo
                    new_J_hi[i][k] = val_hi
            
            J_lo, J_hi = new_J_lo, new_J_hi
            
            # Apply activation derivative to Jacobian
            if act == "relu":
                for i in range(m):
                    if new_z_hi[i] <= 0:
                        for k in range(n_in):
                            J_lo[i][k] = 0.0
                            J_hi[i][k] = 0.0
                    elif new_z_lo[i] < 0:
                        for k in range(n_in):
                            J_lo[i][k] = min(0.0, J_lo[i][k])
                            J_hi[i][k] = max(0.0, J_hi[i][k])
                z_lo, z_hi = ibp_relu(new_z_lo, new_z_hi)
                
            elif act == "tanh":
                for i in range(m):
                    tl, th = math.tanh(new_z_lo[i]), math.tanh(new_z_hi[i])
                    d_lo = 1 - max(tl*tl, th*th)
                    d_hi = 1.0 if (new_z_lo[i] < 0 < new_z_hi[i]) else 1 - min(tl*tl, th*th)
                    for k in range(n_in):
                        jl, jh = J_lo[i][k], J_hi[i][k]
                        prods = [d_lo*jl, d_lo*jh, d_hi*jl, d_hi*jh]
                        J_lo[i][k] = min(prods)
                        J_hi[i][k] = max(prods)
                z_lo, z_hi = ibp_tanh(new_z_lo, new_z_hi)
                
            elif act == "sigmoid":
                for i in range(m):
                    def sig(v): return 1/(1+math.exp(-max(-500, min(500, v))))
                    sl, sh = sig(new_z_lo[i]), sig(new_z_hi[i])
                    d_lo = min(sl*(1-sl), sh*(1-sh))
                    d_hi = 0.25  # max at x=0
                    for k in range(n_in):
                        jl, jh = J_lo[i][k], J_hi[i][k]
                        prods = [d_lo*jl, d_lo*jh, d_hi*jl, d_hi*jh]
                        J_lo[i][k] = min(prods)
                        J_hi[i][k] = max(prods)
                z_lo, z_hi = ibp_sigmoid(new_z_lo, new_z_hi)
                
            else:  # linear/none
                z_lo, z_hi = new_z_lo, new_z_hi
        
        # Return gradient for scalar output
        grad_lo = [J_lo[0][k] for k in range(n_in)]
        grad_hi = [J_hi[0][k] for k in range(n_in)]
        
        return grad_lo, grad_hi
    
    def to_json(self, path: str):
        """Export to JSON"""
        data = {
            "weights": self.weights,
            "biases": self.biases,
            "activations": self.activations
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @property
    def num_params(self) -> int:
        return sum(len(W)*len(W[0]) + len(b) for W, b in zip(self.weights, self.biases))


# ============================================================================
# Dynamics
# ============================================================================

def van_der_pol_ibp(lo: List[float], hi: List[float], mu: float = 1.0) -> Tuple[List[float], List[float]]:
    """Van der Pol: ẋ₁ = x₂, ẋ₂ = μ(1-x₁²)x₂ - x₁"""
    x1_lo, x2_lo = lo
    x1_hi, x2_hi = hi
    
    f1_lo, f1_hi = x2_lo, x2_hi
    
    x1_sq_lo = min(x1_lo*x1_lo, x1_hi*x1_hi)
    x1_sq_hi = max(x1_lo*x1_lo, x1_hi*x1_hi)
    if x1_lo < 0 < x1_hi:
        x1_sq_lo = 0.0
    
    one_minus_x1sq_lo = 1 - x1_sq_hi
    one_minus_x1sq_hi = 1 - x1_sq_lo
    
    prods = [one_minus_x1sq_lo * x2_lo, one_minus_x1sq_lo * x2_hi,
             one_minus_x1sq_hi * x2_lo, one_minus_x1sq_hi * x2_hi]
    term1_lo = mu * min(prods)
    term1_hi = mu * max(prods)
    
    f2_lo = term1_lo - x1_hi
    f2_hi = term1_hi - x1_lo
    
    return [f1_lo, f2_lo], [f1_hi, f2_hi]


def double_integrator_ibp(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Double integrator: ẋ₁ = x₂, ẋ₂ = -x₁ - x₂"""
    x1_lo, x2_lo = lo
    x1_hi, x2_hi = hi
    f1_lo, f1_hi = x2_lo, x2_hi
    f2_lo = -x1_hi - x2_hi
    f2_hi = -x1_lo - x2_lo
    return [f1_lo, f2_lo], [f1_hi, f2_hi]


def linear_stable_ibp(lo: List[float], hi: List[float]) -> Tuple[List[float], List[float]]:
    """Linear stable: ẋ = -x"""
    return [-h for h in hi], [-l for l in lo]


def pendulum_ibp(lo: List[float], hi: List[float], g: float = 9.81, L: float = 1.0, 
                 b: float = 0.1) -> Tuple[List[float], List[float]]:
    """Pendulum: θ̈ = -(g/L)sin(θ) - (b/L)θ̇"""
    theta_lo, omega_lo = lo
    theta_hi, omega_hi = hi
    
    f1_lo, f1_hi = omega_lo, omega_hi
    
    # sin is monotone on [-π/2, π/2], need to handle other cases
    sin_lo = math.sin(theta_lo)
    sin_hi = math.sin(theta_hi)
    if theta_lo < 0 < theta_hi:
        sin_min = min(sin_lo, sin_hi, 0)
        sin_max = max(sin_lo, sin_hi, 0)
    else:
        sin_min = min(sin_lo, sin_hi)
        sin_max = max(sin_lo, sin_hi)
    
    term1_lo = -(g/L) * sin_max
    term1_hi = -(g/L) * sin_min
    term2_lo = -(b/L) * omega_hi
    term2_hi = -(b/L) * omega_lo
    
    f2_lo = term1_lo + term2_lo
    f2_hi = term1_hi + term2_hi
    
    return [f1_lo, f2_lo], [f1_hi, f2_hi]


DYNAMICS = {
    "van_der_pol": van_der_pol_ibp,
    "vdp": van_der_pol_ibp,
    "double_integrator": double_integrator_ibp,
    "di": double_integrator_ibp,
    "linear": linear_stable_ibp,
    "pendulum": pendulum_ibp,
}


# ============================================================================
# Verification
# ============================================================================

def parse_region(spec: str) -> Tuple[List[float], List[float]]:
    """Parse region like '[-1,1]x[-1,1]' or '[0.5,1.5]x[0.5,1.5]'"""
    intervals = spec.split("x")
    lo, hi = [], []
    for interval in intervals:
        interval = interval.strip().strip("[]")
        parts = interval.split(",")
        lo.append(float(parts[0]))
        hi.append(float(parts[1]))
    return lo, hi


def verify_lyapunov(net: IBPNetwork, lo: List[float], hi: List[float], 
                    dynamics: str, use_crown: bool = True) -> Dict[str, Any]:
    """
    Verify Lyapunov conditions using IBP (or CROWN if available).
    
    Returns certificate with:
      - V_bounds: [V_lo, V_hi] bounds on V(x) in region
      - grad_bounds: [grad_lo, grad_hi] bounds on ∇V
      - Vdot_bounds: [Vdot_lo, Vdot_hi] bounds on V̇ = ∇V·f
      - verification_result: whether V > 0 and V̇ < 0
    """
    method = "IBP"
    
    # Try CROWN if available and requested
    if use_crown and HAS_LIRPA and HAS_TORCH:
        try:
            return _verify_with_crown(net, lo, hi, dynamics)
        except Exception as e:
            print(f"CROWN failed ({e}), falling back to IBP", file=sys.stderr)
    
    # IBP verification
    V_lo_list, V_hi_list = net.ibp_forward(lo, hi)
    V_lo, V_hi = V_lo_list[0], V_hi_list[0]
    
    grad_lo, grad_hi = net.ibp_gradient(lo, hi)
    
    # Get dynamics bounds
    if dynamics not in DYNAMICS:
        raise ValueError(f"Unknown dynamics: {dynamics}. Available: {list(DYNAMICS.keys())}")
    f_lo, f_hi = DYNAMICS[dynamics](lo, hi)
    
    # Compute V̇ = ∇V · f
    Vdot_lo, Vdot_hi = 0.0, 0.0
    for gl, gh, fl, fh in zip(grad_lo, grad_hi, f_lo, f_hi):
        prods = [gl * fl, gl * fh, gh * fl, gh * fh]
        Vdot_lo += min(prods)
        Vdot_hi += max(prods)
    
    return {
        "method": method,
        "dynamics": dynamics,
        "region": {"dim": len(lo), "lo": lo, "hi": hi},
        "V_bounds": {"lo": V_lo, "hi": V_hi},
        "grad_bounds": {"lo": grad_lo, "hi": grad_hi},
        "Vdot_bounds": {"lo": Vdot_lo, "hi": Vdot_hi},
        "verification": {
            "V_positive": V_lo > 0,
            "Vdot_negative": Vdot_hi < 0,
            "lyapunov_verified": V_lo > 0 and Vdot_hi < 0
        }
    }


def _verify_with_crown(net: IBPNetwork, lo: List[float], hi: List[float], 
                       dynamics: str) -> Dict[str, Any]:
    """CROWN verification using auto_LiRPA"""
    # Build PyTorch model from weights
    layers = []
    for W, b, act in zip(net.weights, net.biases, net.activations):
        linear = nn.Linear(len(W[0]), len(W))
        linear.weight.data = torch.tensor(W, dtype=torch.float32)
        linear.bias.data = torch.tensor(b, dtype=torch.float32)
        layers.append(linear)
        
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "tanh":
            layers.append(nn.Tanh())
        elif act == "sigmoid":
            layers.append(nn.Sigmoid())
    
    model = nn.Sequential(*layers)
    model.eval()
    
    # Setup bounds
    center = torch.tensor([[(l + h) / 2 for l, h in zip(lo, hi)]], dtype=torch.float32)
    eps = max((h - l) / 2 for l, h in zip(lo, hi))
    
    x_L = torch.tensor([lo], dtype=torch.float32)
    x_U = torch.tensor([hi], dtype=torch.float32)
    
    # Bound V
    bounded_model = BoundedModule(model, center)
    ptb = PerturbationLpNorm(norm=float('inf'), x_L=x_L, x_U=x_U)
    x_bounded = BoundedTensor(center, ptb)
    
    V_lb, V_ub = bounded_model.compute_bounds(x=(x_bounded,), method='CROWN')
    V_lo, V_hi = V_lb.item(), V_ub.item()
    
    # IBP for gradient (CROWN Jacobian is complex)
    grad_lo, grad_hi = net.ibp_gradient(lo, hi)
    
    # Dynamics bounds
    f_lo, f_hi = DYNAMICS[dynamics](lo, hi)
    
    # V̇ bounds
    Vdot_lo, Vdot_hi = 0.0, 0.0
    for gl, gh, fl, fh in zip(grad_lo, grad_hi, f_lo, f_hi):
        prods = [gl * fl, gl * fh, gh * fl, gh * fh]
        Vdot_lo += min(prods)
        Vdot_hi += max(prods)
    
    return {
        "method": "CROWN",
        "dynamics": dynamics,
        "region": {"dim": len(lo), "lo": lo, "hi": hi},
        "V_bounds": {"lo": V_lo, "hi": V_hi},
        "grad_bounds": {"lo": grad_lo, "hi": grad_hi},
        "Vdot_bounds": {"lo": Vdot_lo, "hi": Vdot_hi},
        "verification": {
            "V_positive": V_lo > 0,
            "Vdot_negative": Vdot_hi < 0,
            "lyapunov_verified": V_lo > 0 and Vdot_hi < 0
        }
    }


# ============================================================================
# Output Formatters
# ============================================================================

def format_lean(cert: Dict[str, Any]) -> str:
    """Format certificate as simple Lean definitions"""
    V_lo = cert['V_bounds']['lo']
    V_hi = cert['V_bounds']['hi']
    Vdot_lo = cert['Vdot_bounds']['lo']
    Vdot_hi = cert['Vdot_bounds']['hi']
    region = cert['region']
    
    lines = [
        f"-- Auto-generated Lyapunov certificate",
        f"-- Method: {cert['method']}, Dynamics: {cert['dynamics']}",
        f"",
        f"def cert_region_lo : List Float := {region['lo']}",
        f"def cert_region_hi : List Float := {region['hi']}",
        f"def cert_V_lo : Float := {V_lo}",
        f"def cert_V_hi : Float := {V_hi}",
        f"def cert_Vdot_lo : Float := {Vdot_lo}",
        f"def cert_Vdot_hi : Float := {Vdot_hi}",
        f"",
        f"-- Verification status: {'VERIFIED' if cert['verification']['lyapunov_verified'] else 'FAILED'}",
        f"-- V > 0: {cert['verification']['V_positive']}",
        f"-- Vdot < 0: {cert['verification']['Vdot_negative']}",
    ]
    return "\n".join(lines)


def format_lean_full(
    cert: Dict[str, Any],
    model_name: str = "model",
    lean_namespace: str = "NN.MLTheory.CROWN.Lyapunov.Generated",
) -> str:
    """Generate a Lean file with imports, certificate constants, and (when verified) proof stubs."""
    V_lo = cert['V_bounds']['lo']
    V_hi = cert['V_bounds']['hi']
    Vdot_lo = cert['Vdot_bounds']['lo']
    Vdot_hi = cert['Vdot_bounds']['hi']
    region = cert['region']
    n = region['dim']
    verified = cert['verification']['lyapunov_verified']
    
    # Generate region bounds as Lean code
    region_lo_cases = " else ".join(
        f"if i.val = {i} then {region['lo'][i]}" for i in range(n)
    ) + f" else {region['lo'][0]}"
    
    region_hi_cases = " else ".join(
        f"if i.val = {i} then {region['hi'][i]}" for i in range(n)
    ) + f" else {region['hi'][0]}"
    
    # Safe name for Lean (remove path, extension, special chars)
    safe_name = Path(model_name).stem.replace("-", "_").replace(".", "_")

    # Namespace that will contain the generated declarations.
    # This is a module-level organization choice, not a proof assumption.
    lean_namespace = lean_namespace.strip().strip(".")
    if not lean_namespace:
        lean_namespace = "NN.MLTheory.CROWN.Lyapunov.Generated"
    full_ns = f"{lean_namespace}.{safe_name}"
    
    lean_code = f'''/-
Auto-generated Lyapunov verification file.
Generated by: crown_verifier.py
Method: {cert['method']}
Dynamics: {cert['dynamics']}
Status: {'VERIFIED' if verified else 'FAILED'}

DO NOT EDIT - Regenerate with:
  python crown_verifier.py verify --model {model_name} --region "{region['lo'][0]},{region['hi'][0]}]x..." --dynamics {cert['dynamics']} --format lean-full
-/
import NN.MLTheory.CROWN.Lyapunov.Verification
import Mathlib.Tactic

open NN.MLTheory.CROWN.Lyapunov
open NN.MLTheory.CROWN
open Spec

namespace {full_ns}

/-!
# Certificate Values from Python CROWN

These values were computed by IBP/CROWN verification.
-/

-- State dimension
def n : Nat := {n}

-- Certificate with verified bounds
noncomputable def cert : RealCert n where
  V_lo := {V_lo}
  V_hi := {V_hi}
  Vdot_lo := {Vdot_lo}
  Vdot_hi := {Vdot_hi}
  region_lo := fun i => {region_lo_cases}
  region_hi := fun i => {region_hi_cases}

-- Convert to LyapunovCert for framework
noncomputable def lyapCert : LyapunovCert ℝ n := mkCert cert

/-!
## Trust boundary

`Lyapunov.Verification` exposes a single oracle axiom `crown_oracle` that, given a trusted witness
`CrownOracleWitness lyap lyapCert`, connects the numeric certificate bounds to the (external)
Lyapunov function `lyap`. In the Python-only pipeline, `lyap` represents the real network semantics
used by the external tool; we keep it abstract in Lean.
-/

axiom lyap : NeuralLyapunov ℝ n

/-- Trusted witness that the numeric bounds in `lyapCert` are sound for `lyap`. -/
opaque lyapCertWitness : CrownOracleWitness lyap lyapCert

/-!
# Verification Proofs (only for conditions that are satisfied)
-/
'''
    
    V_positive_ok = cert['verification']['V_positive']
    Vdot_negative_ok = cert['verification']['Vdot_negative']
    
    if V_positive_ok:
        lean_code += f'''
-- Certificate V_lo is positive (verified)
theorem cert_V_positive : cert.V_lo > 0 := by
  simp only [cert]
  norm_num

-- LyapunovCert V_lo is positive (verified)
theorem lyapCert_V_positive : lyapCert.V_lo > 0 := by
  simp only [lyapCert, mkCert, cert]
  norm_num
'''
    else:
        lean_code += f'''
-- V_lo = {V_lo} is NOT positive, so the condition `cert.V_lo > 0` is false and cannot be proven.
-- (We intentionally do not emit unverifiable Lean proofs here.)
'''
    
    if Vdot_negative_ok:
        lean_code += f'''
-- Certificate Vdot_hi is negative (verified)
theorem cert_Vdot_negative : cert.Vdot_hi < 0 := by
  simp only [cert]
  norm_num

-- LyapunovCert Vdot_hi is negative (verified)
theorem lyapCert_Vdot_negative : lyapCert.Vdot_hi < 0 := by
  simp only [lyapCert, mkCert, cert]
  norm_num
'''
    else:
        lean_code += f'''
-- Vdot_hi = {Vdot_hi} is NOT negative, so the condition `cert.Vdot_hi < 0` is false and cannot be proven.
-- (We intentionally do not emit unverifiable Lean proofs here.)
'''
    
    lean_code += '''
'''
    
    # Add the main stability theorem only if verified
    if verified:
        lean_code += f'''
/-!
# Main Stability Theorem

Lyapunov stability verified.
-/

theorem lyapunov_stable :
    (∀ x, Box.contains lyapCert.region x → lyap.V x > 0) ∧
    (∀ x, Box.contains lyapCert.region x → lyap.Vdot x < 0) :=
  Real.lyapunov_conditions lyap lyapCert lyapCertWitness lyapCert_V_positive lyapCert_Vdot_negative

-- Individual conditions
theorem V_positive (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains lyapCert.region x) : lyap.V x > 0 :=
  Real.V_positive lyap lyapCert lyapCertWitness lyapCert_V_positive x hx

theorem Vdot_negative (x : Tensor ℝ (.dim n .scalar))
    (hx : Box.contains lyapCert.region x) : lyap.Vdot x < 0 :=
  Real.Vdot_negative lyap lyapCert lyapCertWitness lyapCert_Vdot_negative x hx

#check @lyapunov_stable
'''
    else:
        lean_code += f'''
/-!
# Verification Status: FAILED

The certificate does not prove Lyapunov stability:
- V > 0: {cert['verification']['V_positive']}
- V̇ < 0: {cert['verification']['Vdot_negative']}

Try:
- Smaller verification region
- Different dynamics
- Re-train the Lyapunov network
-/
'''
    
    lean_code += f'''
end {full_ns}
'''
    
    return lean_code


# ============================================================================
# CLI
# ============================================================================

def cmd_verify(args):
    """Verify Lyapunov conditions"""
    # Load network
    if args.model.endswith('.json'):
        net = IBPNetwork.from_json(args.model)
    else:
        net = IBPNetwork.from_pytorch(args.model, prefix=args.prefix)
    
    # Only print status for non-lean-full formats (to not corrupt the output)
    if args.format != 'lean-full':
        print(f"Loaded network: {net.num_params} parameters", file=sys.stderr)
    
    # Parse region
    lo, hi = parse_region(args.region)
    
    # Verify
    cert = verify_lyapunov(net, lo, hi, args.dynamics, use_crown=not args.no_crown)
    
    # Output
    if args.format == 'json':
        print(json.dumps(cert, indent=2))
    elif args.format == 'lean':
        print(format_lean(cert))
    elif args.format == 'lean-full':
        print(format_lean_full(cert, args.model, args.lean_namespace))
    else:
        print(f"Method: {cert['method']}")
        print(f"Region: {lo} to {hi}")
        print(f"V(x) in [{cert['V_bounds']['lo']:.6f}, {cert['V_bounds']['hi']:.6f}]")
        print(f"Vdot(x) in [{cert['Vdot_bounds']['lo']:.6f}, {cert['Vdot_bounds']['hi']:.6f}]")
        print(f"V > 0: {cert['verification']['V_positive']} {'[OK]' if cert['verification']['V_positive'] else '[FAIL]'}")
        print(f"Vdot < 0: {cert['verification']['Vdot_negative']} {'[OK]' if cert['verification']['Vdot_negative'] else '[FAIL]'}")
        if cert['verification']['lyapunov_verified']:
            print("LYAPUNOV VERIFIED")
        else:
            print("Verification failed")
    
    # Write to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(cert, f, indent=2)
        print(f"Certificate written to: {args.output}", file=sys.stderr)
    
    return 0 if cert['verification']['lyapunov_verified'] else 1


def cmd_export(args):
    """Export network to JSON"""
    net = IBPNetwork.from_pytorch(args.model, prefix=args.prefix)
    net.to_json(args.output)
    print(f"Exported {net.num_params} params to {args.output}")
    return 0


def cmd_info(args):
    """Show network info"""
    if args.model.endswith('.json'):
        net = IBPNetwork.from_json(args.model)
    else:
        net = IBPNetwork.from_pytorch(args.model, prefix=args.prefix)
    
    print(f"Parameters: {net.num_params}")
    print(f"Layers: {len(net.weights)}")
    for i, (W, b, act) in enumerate(zip(net.weights, net.biases, net.activations)):
        print(f"  Layer {i}: {len(W[0])} -> {len(W)}, activation={act}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Unified CROWN/IBP verifier")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # verify command
    p_verify = subparsers.add_parser('verify', help='Verify Lyapunov conditions')
    p_verify.add_argument('--model', required=True, help='Network file (.json or .pth)')
    p_verify.add_argument('--region', required=True, help='Region spec: "[-1,1]x[-1,1]"')
    p_verify.add_argument('--dynamics', required=True, choices=list(DYNAMICS.keys()))
    p_verify.add_argument('--prefix', default='lyapunov.', help='State dict prefix for .pth')
    p_verify.add_argument('--no-crown', action='store_true', help='Use IBP only')
    p_verify.add_argument('--format', choices=['json', 'lean', 'lean-full', 'text'], default='text')
    p_verify.add_argument(
        '--lean-namespace',
        default='NN.MLTheory.CROWN.Lyapunov.Generated',
        help='Namespace prefix used by --format lean-full',
    )
    p_verify.add_argument('--output', '-o', help='Output certificate file')
    p_verify.set_defaults(func=cmd_verify)
    
    # export command
    p_export = subparsers.add_parser('export', help='Export .pth to .json')
    p_export.add_argument('--model', required=True, help='PyTorch checkpoint')
    p_export.add_argument('--output', '-o', required=True, help='Output JSON file')
    p_export.add_argument('--prefix', default='lyapunov.', help='State dict prefix')
    p_export.set_defaults(func=cmd_export)
    
    # info command
    p_info = subparsers.add_parser('info', help='Show network info')
    p_info.add_argument('--model', required=True, help='Network file')
    p_info.add_argument('--prefix', default='lyapunov.', help='State dict prefix for .pth')
    p_info.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
