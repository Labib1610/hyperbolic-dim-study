"""Geometry core: Poincaré-ball and Euclidean primitives with a shared interface.

This stage implements the geometry we optimize over — the Poincaré ball (curvature
c = 1) — plus a trivial Euclidean geometry exposing the same method names, so later
training code can be geometry-agnostic. The formulas are implemented from scratch
(deliberately, so we can debug them) and cross-checked against ``geoopt`` in the
tests.

**Correctness and numerical stability are the whole point here.** Points live in
the open unit ball (‖x‖ < 1); the factor ``(1 - ‖x‖²)`` vanishes at the boundary
and must be clamped everywhere it appears, and the ``arccosh`` argument is clamped
above 1 before the sqrt. See the numerical-stability notes on each method.

All points are torch tensors of shape ``(..., d)``.
"""

from __future__ import annotations

import torch

# Small margin keeping points strictly inside the ball and denominators nonzero.
EPS = 1e-5

# Floor on the sqrt argument inside arccosh: keeps the derivative 1/sqrt(x²-1)
# finite at x → 1 without perturbing the forward value (arccosh(1) stays ~0).
# NOTE: the STAGE_2 spec suggests clamping the arccosh *argument* to >= 1 + 1e-7,
# but that makes dist(p, p) ≈ 4.5e-4, which contradicts the spec's own T1
# (dist == 0, atol 1e-6). We instead clamp the argument to >= 1.0 (so coincident
# points give exactly 0) and floor the sqrt argument for gradient safety.
ARCCOSH_SQRT_FLOOR = 1e-15


def arccosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically-safe inverse hyperbolic cosine.

    ``arccosh(x) = log(x + sqrt(x² - 1))``. The argument is clamped to ``>= 1``
    (avoiding sqrt of a negative from float error) and the sqrt argument is
    floored to ``ARCCOSH_SQRT_FLOOR`` so the gradient stays finite as ``x → 1``.
    """
    x = x.clamp_min(1.0)
    return torch.log(x + torch.sqrt((x * x - 1.0).clamp_min(ARCCOSH_SQRT_FLOOR)))


class PoincareBall:
    """Poincaré ball model of hyperbolic space with curvature c = 1."""

    def __init__(self, eps: float = EPS):
        self.eps = eps

    def sq_norm(self, x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Squared Euclidean norm along the last dim: ``sum(x*x, dim=-1)``."""
        return torch.sum(x * x, dim=-1, keepdim=keepdim)

    def _one_minus_sq_norm(self, x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """``(1 - ‖x‖²)`` clamped to ``>= eps`` (safe to divide by)."""
        return (1.0 - self.sq_norm(x, keepdim=keepdim)).clamp_min(self.eps)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Rescale any point with ‖x‖ >= 1 - eps back to norm ``1 - eps``.

        Points already strictly inside are returned unchanged.
        """
        max_norm = 1.0 - self.eps
        norm = torch.norm(x, dim=-1, keepdim=True)
        scaled = x / norm.clamp_min(1e-15) * max_norm
        return torch.where(norm >= max_norm, scaled, x)

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Poincaré distance between ``u`` and ``v`` (shape ``(...,)``)."""
        diff2 = self.sq_norm(u - v)
        denom = self._one_minus_sq_norm(u) * self._one_minus_sq_norm(v)
        arg = 1.0 + 2.0 * diff2 / denom
        return arccosh(arg)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor ``lambda_x = 2 / (1 - ‖x‖²)`` (shape ``(...,)``)."""
        return 2.0 / self._one_minus_sq_norm(x)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Möbius addition ``x ⊕ y`` (curvature 1)."""
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        x2 = self.sq_norm(x, keepdim=True)
        y2 = self.sq_norm(y, keepdim=True)
        num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
        den = 1.0 + 2.0 * xy + x2 * y2
        return num / den.clamp_min(self.eps)

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map at ``x`` in tangent direction ``v``.

        Reduces to ``exp_0(v) = tanh(‖v‖) * v / ‖v‖`` at the origin. The result is
        projected back into the ball.
        """
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(self.eps)
        lam = self.conformal_factor(x).unsqueeze(-1)
        coef = torch.tanh(lam * v_norm / 2.0) / v_norm
        return self.project(self.mobius_add(x, coef * v))

    def riemannian_grad(self, x: torch.Tensor, euclid_grad: torch.Tensor) -> torch.Tensor:
        """Rescale a Euclidean gradient to the Riemannian gradient at ``x``.

        ``rgrad = ((1 - ‖x‖²)² / 4) * euclid_grad`` (inverse metric factor).
        """
        factor = ((1.0 - self.sq_norm(x)) ** 2 / 4.0).unsqueeze(-1)
        return factor * euclid_grad


class EuclideanGeometry:
    """Trivial Euclidean geometry with the same interface as :class:`PoincareBall`."""

    def __init__(self, eps: float = EPS):
        self.eps = eps

    def sq_norm(self, x: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        return torch.sum(x * x, dim=-1, keepdim=keepdim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.norm(u - v, dim=-1)

    def conformal_factor(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y

    def expmap(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return x + v

    def riemannian_grad(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return g
