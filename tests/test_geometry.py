"""Geometry-core tests: hand-computed ground truth + geoopt cross-check + stability.

Every value here was derived by hand (see STAGE_2 spec). A failure means the
geometry is wrong — do not proceed past this stage.
"""

import math

import torch

from src.geometry import EuclideanGeometry, PoincareBall

ball = PoincareBall()


def _pt(*coords):
    return torch.tensor(coords, dtype=torch.float64)


# --- T1: center-to-center is 0 -------------------------------------------------
def test_t1_center_to_center_zero():
    d = ball.dist(_pt(0.0, 0.0), _pt(0.0, 0.0))
    assert torch.allclose(d, torch.tensor(0.0, dtype=torch.float64), atol=1e-6)


# --- T2: small step near center ------------------------------------------------
def test_t2_small_step_near_center():
    d = ball.dist(_pt(0.0, 0.0), _pt(0.1, 0.0))
    assert abs(d.item() - 0.2003) < 1e-3


# --- T3: same-size step near boundary costs more -------------------------------
def test_t3_boundary_stretch():
    # NOTE: the spec's hand value 0.7719 is a hand-math error. The true distance
    # is 2*(atanh(0.9) - atanh(0.8)) = 0.74721, which our impl and geoopt both give.
    d = ball.dist(_pt(0.8, 0.0), _pt(0.9, 0.0))
    assert abs(d.item() - 0.74721) < 1e-3
    euclid_step = 0.1  # same-length Euclidean displacement
    assert d.item() > 3.0 * euclid_step


# --- T4: dog/cat collision fix (key test) --------------------------------------
def test_t4_dog_cat_collision():
    mammal = _pt(0.0, 0.5)
    dog = _pt(-0.53, 0.53)
    cat = _pt(0.53, 0.53)
    dc = ball.dist(dog, cat)
    md = ball.dist(mammal, dog)
    assert abs(dc.item() - 3.23) < 2e-2
    assert abs(md.item() - 1.65) < 2e-2
    assert abs((dc / md).item() - 1.96) < 5e-2


# --- T5: conformal factor values ----------------------------------------------
def test_t5_conformal_factor():
    assert abs(ball.conformal_factor(_pt(0.0, 0.0)).item() - 2.0) < 1e-4
    assert abs(ball.conformal_factor(_pt(0.9, 0.0)).item() - 10.526) < 1e-2


# --- T6: Riemannian gradient rescale + one retraction step ---------------------
def test_t6_riemannian_grad_step():
    x = _pt(0.6, 0.0)
    g = _pt(0.5, 0.0)
    rgrad = ball.riemannian_grad(x, g)
    assert torch.allclose(rgrad, _pt(0.0512, 0.0), atol=1e-3)
    x_new = x - 1.0 * rgrad
    assert torch.allclose(x_new, _pt(0.5488, 0.0), atol=1e-3)
    assert torch.norm(x_new).item() < 1.0


# --- T7: exp map at origin -----------------------------------------------------
def test_t7_expmap_origin():
    out = ball.expmap(_pt(0.0, 0.0), _pt(0.3, 0.0))
    assert torch.allclose(out, _pt(math.tanh(0.3), 0.0), atol=1e-4)


# --- T8: projection keeps points inside ---------------------------------------
def test_t8_projection():
    projected = ball.project(_pt(1.5, 0.0))
    assert projected.norm().item() < 1.0
    assert abs(projected.norm().item() - (1.0 - ball.eps)) < 1e-6
    inside = _pt(0.2, 0.1)
    assert torch.allclose(ball.project(inside), inside, atol=1e-6)


# --- T9: cross-check against geoopt -------------------------------------------
def test_t9_geoopt_crosscheck():
    import geoopt

    gball = geoopt.PoincareBall()
    torch.manual_seed(0)
    for _ in range(20):
        u = torch.rand(5, dtype=torch.float64) * 2 - 1
        v = torch.rand(5, dtype=torch.float64) * 2 - 1
        u = u / u.norm() * (torch.rand(1, dtype=torch.float64) * 0.9)
        v = v / v.norm() * (torch.rand(1, dtype=torch.float64) * 0.9)
        ours = ball.dist(u, v)
        theirs = gball.dist(u, v)
        assert torch.allclose(ours, theirs, atol=1e-4), (ours, theirs)


# --- T10: batching + boundary safety ------------------------------------------
def test_t10_batching_and_boundary():
    torch.manual_seed(1)
    batch_u = torch.rand(16, 5, dtype=torch.float64) * 0.5
    batch_v = torch.rand(16, 5, dtype=torch.float64) * 0.5
    d = ball.dist(batch_u, batch_v)
    assert d.shape == (16,)
    assert torch.isfinite(d).all()

    # Points AT the boundary (norm 0.9999) must still give finite distances.
    a = _pt(0.9999, 0.0)
    b = _pt(0.0, 0.9999)
    assert torch.isfinite(ball.dist(a, b)).all()


# --- Stability: finite gradients near the boundary (most important) ------------
def test_finite_gradients_near_boundary():
    x = torch.tensor([0.99, 0.0], dtype=torch.float64, requires_grad=True)
    target = _pt(0.1, 0.1)
    d = ball.dist(x, target)
    d.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# --- Euclidean interface parity ------------------------------------------------
def test_euclidean_interface():
    euc = EuclideanGeometry()
    u, v = _pt(0.3, 0.4), _pt(0.0, 0.0)
    assert abs(euc.dist(u, v).item() - 0.5) < 1e-6
    assert torch.allclose(euc.project(u), u)
    assert torch.allclose(euc.expmap(u, v), u + v)
    assert torch.allclose(euc.riemannian_grad(u, v), v)
