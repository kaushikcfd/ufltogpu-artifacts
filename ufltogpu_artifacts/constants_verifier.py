from __future__ import annotations

import numpy as np

import firedrake as fd
from tsfc import compile_form

from ufltogpu_artifacts.core import Op, flops_per_cell
from ufltogpu_artifacts.weak_forms import get_bilinear_form


def verify_flops_per_cell() -> None:
    """
    Verifies the correctness of entries in :data:`ufltogpu_artifacts.flops_per_cell`.
    """
    for (op, dim, p), ref_flops in flops_per_cell.items():
        A, V = get_bilinear_form(1, dim, op, p)
        x = fd.Function(V)
        expr = fd.action(A, x)
        (knl,) = compile_form(expr)
        empirical_flops = knl.flop_count
        if np.testing.assert_array_equal(empirical_flops, ref_flops):
            raise RuntimeError(
                f"For {op=}, {dim=}, {p=}: "
                f"Expected = {ref_flops}, Measured = {empirical_flops}"
            )

    print("Verified flops_per_cell for all reference values. ✨ 🦾 ✨")


def get_p_lo_hi(dim: int) -> tuple[int, int]:
    if dim == 2:
        return (1, 8)
    elif dim == 3:
        return (1, 6)
    else:
        raise NotImplementedError(f"{dim=}")


def print_flops() -> None:
    """
    Prints the :data:`ufltogpu_artifacts.flops_per_cell` as a python dict.
    """
    import black

    BLACK_MODE = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=84)

    code = ""

    code += "flops_per_cell = {\n"

    for op in [
        Op.MASS,
        Op.LAPLACE,
        Op.HELMHOLTZ,
        Op.ELASTICITY,
        Op.HYPERELASTICITY,
    ]:
        for dim in [2, 3]:
            p_lo, p_hi = get_p_lo_hi(dim)
            for p in range(p_lo, p_hi + 1):
                A, V = get_bilinear_form(1, dim, op, p)
                x = fd.Function(V)
                expr = fd.action(A, x)
                (knl,) = compile_form(expr)
                code += f"({op}, {dim}, {p}): {int(knl.flop_count)},\n"
        code += "\n"

    code += "}"
    print(black.format_file_contents(code, fast=False, mode=BLACK_MODE))


if __name__ == "__main__":
    # print_flops()
    verify_flops_per_cell()
