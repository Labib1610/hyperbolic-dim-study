# Stage 2 — Geometry Core (Poincaré ball, tested against hand-computed values)

## Context (read first)
Project: **a controlled study of dimension–distortion trade-offs of hyperbolic vs Euclidean embeddings of linguistic hierarchies, as a function of hierarchy structure.**

Stages 0 (env) and 1 (data) are complete. This stage implements the **geometry core**: the Poincaré-ball operations we will optimize over, plus a matching Euclidean interface so later code can treat both geometries uniformly. **Correctness and numerical stability are the whole point of this stage.** Every core function is unit-tested against numbers we computed BY HAND (listed below) so we catch geometry bugs immediately.

**Do NOT implement embeddings, training loops, data loading, or metrics here.** Only geometry primitives.

## Environment
- Repo `hyperbolic-dim-study/`, package `src`, venv active, torch (cu128, sm_120 verified), `geoopt` available.
- We will USE `geoopt` for the actual optimizer later, but in THIS stage we implement the core formulas ourselves in `src/geometry.py` AND cross-check them against `geoopt`'s `PoincareBall` where possible. Implementing them ourselves is deliberate: we must understand and be able to debug them.

---

## Mathematical spec (implement exactly these; curvature c = 1 unless noted)

All points are torch tensors of shape `(..., d)` inside the open unit ball (‖x‖ < 1).

**1. Norms / safety.** Work with `sq_norm(x) = sum(x*x, dim=-1)`. Everywhere `(1 - sq_norm)` appears, it can approach 0 near the boundary and cause NaN/inf. Clamp: maintain a small `EPS = 1e-5` and clamp `sq_norm` to at most `1 - EPS` (i.e. keep points strictly inside). Provide `project(x)` that rescales any point with ‖x‖ ≥ 1−EPS back to norm `1−EPS`.

**2. Poincaré distance** between `u`, `v`:
```
diff2   = sq_norm(u - v)                      # ||u-v||^2
denom   = (1 - sq_norm(u)) * (1 - sq_norm(v)) # clamp each factor to >= EPS
arg     = 1 + 2 * diff2 / denom
dist    = arccosh(arg)                         # = log(arg + sqrt(arg^2 - 1))
```
Implement `arccosh(x) = torch.log(x + torch.sqrt(x*x - 1))` with `x` clamped to `>= 1 + 1e-7` before the sqrt to avoid NaN at the boundary `x → 1`.

**3. Conformal factor** at `x`: `lambda_x = 2 / (1 - sq_norm(x))` (clamp denom).

**4. Möbius addition** (needed for exp map) of `x ⊕ y`:
```
xy   = sum(x*y, dim=-1, keepdim=True)
x2   = sq_norm(x, keepdim=True)
y2   = sq_norm(y, keepdim=True)
num  = (1 + 2*xy + y2) * x + (1 - x2) * y
den  = 1 + 2*xy + x2*y2
mob  = num / den
```

**5. Exponential map** at `x` in direction `v` (tangent vector), curvature 1:
```
v_norm = clamp_min(norm(v), EPS)
lam    = lambda_x                      # conformal factor at x (scalar per point)
coef   = tanh(lam * v_norm / 2) / v_norm
exp    = mobius_add(x, coef * v)
return project(exp)
```
At `x = 0` this must reduce to `exp_0(v) = tanh(||v||) * v / ||v||`. Include that as a test.

**6. Riemannian gradient rescale** (for reference / used by trainer later):
`riemannian_grad(x, euclid_grad) = ((1 - sq_norm(x))**2 / 4).unsqueeze(-1) * euclid_grad`.

Provide a parallel trivial **Euclidean geometry** interface with the SAME function names in a class, so the trainer can be geometry-agnostic:
- `EuclideanGeometry.dist(u,v) = ||u - v||`
- `EuclideanGeometry.project(x) = x` (no-op)
- `EuclideanGeometry.expmap(x, v) = x + v`

Structure `src/geometry.py` as two classes with a shared interface:
```
class PoincareBall:
    def __init__(self, eps=1e-5): ...
    def sq_norm(self, x): ...
    def project(self, x): ...
    def dist(self, u, v): ...
    def conformal_factor(self, x): ...
    def mobius_add(self, x, y): ...
    def expmap(self, x, v): ...
    def riemannian_grad(self, x, euclid_grad): ...

class EuclideanGeometry:
    def dist(self, u, v): ...
    def project(self, x): return x
    def expmap(self, x, v): return x + v
    def riemannian_grad(self, x, g): return g
```

---

## Task — Hand-computed ground-truth tests (`tests/test_geometry.py`)
These exact numbers were derived by hand and MUST match (tolerance `atol=1e-2` for the multi-step ones, `1e-4` for simple ones). If any fail, the geometry is wrong — do not proceed.

**T1 — center-to-center is 0.** `dist((0,0),(0,0)) == 0` (atol 1e-6).

**T2 — small step near center.** `dist((0,0),(0.1,0)) ≈ 0.2003` (atol 1e-3).

**T3 — same-size step near boundary costs more.** `dist((0.8,0),(0.9,0)) ≈ 0.7719` (atol 1e-3). Assert this is > 3× the value from a same-length Euclidean step (0.1), i.e. confirms boundary stretching.

**T4 — the dog/cat collision fix (key test).** With
`mammal=(0,0.5)`, `dog=(-0.53,0.53)`, `cat=(0.53,0.53)`:
- `dist(dog, cat) ≈ 3.23` (atol 2e-2)
- `dist(mammal, dog) ≈ 1.65` (atol 2e-2)
- assert `dist(dog,cat) / dist(mammal,dog) ≈ 1.96` (atol 5e-2) — geometry tracks the 2:1 tree-distance ratio.

**T5 — conformal factor values.** `conformal_factor((0,0)) == 2.0`; `conformal_factor((0.9,0)) ≈ 10.526` (atol 1e-2).

**T6 — Riemannian gradient rescale (hand-worked step).** For `x=(0.6,0)` and euclid grad `g=(0.5,0)`:
- `riemannian_grad(x,g) ≈ (0.0512, 0)` (atol 1e-3).
- Then one retraction step `x_new = x - 1.0 * riemannian_grad(x,g)` gives `x_new ≈ (0.5488, 0)` (atol 1e-3), and assert `||x_new|| < 1`.

**T7 — exp map at origin.** `expmap((0,0),(0.3,0)) ≈ (tanh(0.3), 0) ≈ (0.2913, 0)` (atol 1e-4).

**T8 — projection keeps points inside.** `project((1.5, 0))` has norm `≈ 1 - eps < 1`; `project((0.2,0.1))` is unchanged (atol 1e-6).

**T9 — cross-check against geoopt.** For 20 random point pairs inside radius 0.9, assert our `PoincareBall.dist` matches `geoopt.PoincareBall().dist(u,v)` (atol 1e-4). (This guards against sign/convention mistakes.)

**T10 — batching.** All functions accept batched inputs `(N, d)` and return correct shapes; verify `dist` on a `(16, 5)` batch runs and is finite, and that feeding points AT the boundary (norm 0.9999) yields finite (non-NaN) distances thanks to clamping.

---

## Numerical-stability requirements (must implement, not optional)
- Clamp `(1 - sq_norm)` factors to `>= EPS` before dividing.
- Clamp `arccosh` argument to `>= 1 + 1e-7` before `sqrt`.
- `project()` applied after every `expmap`.
- Add a test that a forward+backward pass through `dist` on parameters near the boundary produces **finite gradients** (no NaN) — create a leaf tensor at norm ~0.99, compute `dist` to another point, `.backward()`, assert grad is finite. This is the single most important stability test for later training.

---

## Definition of Done — checklist
- [ ] `src/geometry.py` implements `PoincareBall` and `EuclideanGeometry` with the shared interface.
- [ ] All clamping / stability measures implemented.
- [ ] `tests/test_geometry.py` implements T1–T10; **all pass**.
- [ ] The dog/cat test (T4) passes — confirms the collision fix numerically.
- [ ] The Riemannian-step test (T6) reproduces (0.5488, 0).
- [ ] geoopt cross-check (T9) passes.
- [ ] Finite-gradient-near-boundary test passes (no NaN).

## Report back to the human
Report: (1) all test results (which passed), (2) the actual computed values for T4 (dog/cat and mammal/dog distances) and T6 (the rescaled grad and x_new) so we can eyeball them against the hand math, (3) confirmation the geoopt cross-check matched, and (4) any place you had to deviate from the spec (e.g. geoopt API differences) and why.
```
