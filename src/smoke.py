"""Smoke test: verify Python, PyTorch (with sm_120 GPU support), and geoopt.

Run with:  python -m src.smoke
"""

import platform
import sys

import torch

from src.config import get_device


def main() -> int:
    # 1. Python version
    print(f"Python version: {platform.python_version()}")

    # 2. PyTorch version + CUDA build tag
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch built with CUDA: {torch.version.cuda}")

    # 3. CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")

    # 4. Compiled arch list (must contain sm_120 for the RTX 5060 Ti)
    arch_list = torch.cuda.get_arch_list()
    print(f"torch.cuda.get_arch_list(): {arch_list}")
    sm120_supported = "sm_120" in arch_list

    # 5. GPU name + compute capability
    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("GPU name: <none — CUDA not available>")

    # 6. Real GPU tensor op, guarded against sm_120 incompatibility.
    gpu_op_ok = False
    gpu_op_reason = ""
    if cuda_available:
        try:
            device = get_device()
            a = torch.rand(1024, 1024, device=device)
            b = torch.rand(1024, 1024, device=device)
            result = (a * b).sum()
            torch.cuda.synchronize()
            print(f"GPU tensor op (a*b).sum() on {device}: {result.item():.4f}")
            gpu_op_ok = True
        except Exception as e:  # noqa: BLE001 - report, do not crash
            gpu_op_reason = str(e).splitlines()[0] if str(e) else repr(e)
            print(f"GPU tensor op FAILED: {gpu_op_reason}")
    else:
        gpu_op_reason = "CUDA not available"

    # 7. geoopt Poincare-ball sanity op
    geoopt_ok = False
    geoopt_reason = ""
    try:
        import geoopt

        print(f"geoopt version: {geoopt.__version__}")
        ball = geoopt.PoincareBall()
        x = torch.tensor([0.1, 0.2])
        y = torch.tensor([-0.3, 0.4])
        x = ball.projx(x)
        y = ball.projx(y)
        dist = ball.dist(x, y)
        print(f"geoopt PoincareBall dist(x, y): {dist.item():.6f}")
        geoopt_ok = True
    except Exception as e:  # noqa: BLE001
        geoopt_reason = str(e).splitlines()[0] if str(e) else repr(e)
        print(f"geoopt sanity op FAILED: {geoopt_reason}")

    # 8. Pass / fail verdict
    if cuda_available and sm120_supported and gpu_op_ok and geoopt_ok:
        print("SMOKE TEST PASSED")
        return 0

    # Build a specific reason + remediation.
    if not cuda_available:
        reason = "torch.cuda.is_available() is False"
        remediation = "install the cu128 torch build; check driver + GPU visibility."
    elif not sm120_supported:
        reason = "sm_120 not in torch.cuda.get_arch_list()"
        remediation = (
            "reinstall torch from the cu128 channel (try nightly: "
            "pip install --pre torch --index-url "
            "https://download.pytorch.org/whl/nightly/cu128)."
        )
    elif not gpu_op_ok:
        reason = f"GPU tensor op failed: {gpu_op_reason}"
        remediation = "install nightly cu128 torch with sm_120 kernels."
    else:
        reason = f"geoopt sanity op failed: {geoopt_reason}"
        remediation = "pip install geoopt>=0.5.0."

    print(f"SMOKE TEST FAILED — {reason}. Remediation: {remediation}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
