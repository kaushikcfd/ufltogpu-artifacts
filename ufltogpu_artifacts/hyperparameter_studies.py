from __future__ import annotations

import argparse
import logging
import os
from typing import Sequence


# firedrake_clean() rmtree's the pytools cache dir, but loopy keeps a sqlite
# connection into it open for the process lifetime (opened at import time).
# Deleting the dir out from under that connection breaks later writes. Disabling
# loopy's disk cache before firedrake/loopy are imported avoids it.
os.environ["LOOPY_NO_CACHE"] = "1"

import pyop2.transforms.auto_tiling
from tabulate import tabulate

import firedrake as fd  # ruff: ignore[unused-import] (import for PETSc to init ctx.)

from ufltogpu_artifacts.core import Op, get_nel1d_for_reported_data, name_to_op, op_name
from ufltogpu_artifacts.timings_recorder import get_flops, get_runtime_in_s


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)


def firedrake_clean() -> None:
    import os
    import shutil
    import sys

    from pyop2.configuration import configuration as pyop2_configuration

    try:
        import platformdirs as appdirs
    except ImportError:
        import appdirs

    pyop2_cache = pyop2_configuration["cache_dir"]
    pytools_cache = appdirs.user_cache_dir("pytools", "pytools")

    for cache in [pyop2_cache, pytools_cache]:
        if os.path.exists(cache):
            shutil.rmtree(cache, ignore_errors=True)

    global_kernel = sys.modules.get("pyop2.global_kernel")
    if global_kernel is not None:
        global_kernel.AbstractGlobalKernel._cache.clear()

    assemble = sys.modules.get("firedrake.assemble")
    if assemble is not None:
        assemble._make_global_kernel.cache.clear()


def main(
    *,
    operators: Sequence[Op],
    dims: Sequence[int],
    p_lo: int,
    p_hi: int,
) -> None:
    eta_alias_max_values = [0, 0.5, 0.6, 0.7, 0.8, 0.9]
    eta_simd_max_values = [0, 0.5, 0.7, 0.9, 0.97, 0.99]
    wg_max_values = [64, 128, 256, 512]

    perf_table: dict[str, dict[float, str]] = {}

    for wg_max in wg_max_values:
        firedrake_clean()
        pyop2.transforms.auto_tiling.NcNwi_MAX = wg_max

        for dim in dims:
            nel_1d = get_nel1d_for_reported_data(dim)
            for op in operators:
                for p in range(p_lo, p_hi + 1):
                    t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel_1d=nel_1d)
                    nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)

                    row_label = f"{op_name(op)}.{dim}D.P{p}"
                    perf_table.setdefault(row_label, {})[wg_max] = (
                        f"{1e-9 * (nflops / t_op):.1f}"
                    )
                    logger.critical(f"Done with {wg_max=}, {op=}, {dim=}, {p=}")

    print(
        tabulate(
            [
                (row_label, *(gflops[wg_max] for wg_max in wg_max_values))
                for row_label, gflops in perf_table.items()
            ],
            headers=(
                "Operator",
                *(f"WG_MAX={wg_max}\nGFLOPS" for wg_max in wg_max_values),
            ),
            tablefmt="fancy_grid",
        )
    )

    perf_table: dict[str, dict[float, str]] = {}

    for eta_simd_max in eta_simd_max_values:
        firedrake_clean()
        pyop2.transforms.auto_tiling.ETA_SIMD_MAX = eta_simd_max

        for dim in dims:
            nel_1d = get_nel1d_for_reported_data(dim)
            for op in operators:
                for p in range(p_lo, p_hi + 1):
                    t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel_1d=nel_1d)
                    nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)

                    row_label = f"{op_name(op)}.{dim}D.P{p}"
                    perf_table.setdefault(row_label, {})[eta_simd_max] = (
                        f"{1e-9 * (nflops / t_op):.1f}"
                    )
                    logger.critical(f"Done with {eta_simd_max=}, {op=}, {dim=}, {p=}")

    print(
        tabulate(
            [
                (row_label, *(gflops[eta] for eta in eta_simd_max_values))
                for row_label, gflops in perf_table.items()
            ],
            headers=(
                "Operator",
                *(f"ETA_SIMD_MAX={eta}\nGFLOPS" for eta in eta_simd_max_values),
            ),
            tablefmt="fancy_grid",
        )
    )

    perf_table: dict[str, dict[float, str]] = {}

    for eta_alias_max in eta_alias_max_values:
        firedrake_clean()
        pyop2.transforms.auto_tiling.ETA_ALIAS_MAX = eta_alias_max

        for dim in dims:
            nel_1d = get_nel1d_for_reported_data(dim)
            for op in operators:
                for p in range(p_lo, p_hi + 1):
                    t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel_1d=nel_1d)
                    nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)

                    row_label = f"{op_name(op)}.{dim}D.P{p}"
                    perf_table.setdefault(row_label, {})[eta_alias_max] = (
                        f"{1e-9 * (nflops / t_op):.1f}"
                    )
                    logger.critical(f"Done with {eta_alias_max=}, {op=}, {dim=}, {p=}")

    print(
        tabulate(
            [
                (row_label, *(gflops[eta] for eta in eta_alias_max_values))
                for row_label, gflops in perf_table.items()
            ],
            headers=(
                "Operator",
                *(f"ETA_ALIAS_MAX={eta}\nGFLOPS" for eta in eta_alias_max_values),
            ),
            tablefmt="fancy_grid",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Utility to obtain parameter studies of Kulkarni, "
            "Kloeckner's auto-tiling strategy."
        )
    )
    parser.add_argument(
        "--op",
        type=str,
        nargs="*",
        default=[],
        help="List of operators whose performance is to be evaluated.",
        choices=[
            op_name(Op.MASS),
            op_name(Op.LAPLACE),
            op_name(Op.HELMHOLTZ),
            op_name(Op.ELASTICITY),
            op_name(Op.HYPERELASTICITY),
        ],
        required=True,
    )
    parser.add_argument(
        "--dim",
        choices=[2, 3],
        type=int,
        nargs="*",
        help="Toplogical dimension on which the function spaces are to be defined.",
        required=True,
    )
    parser.add_argument(
        "--p_range",
        type=int,
        nargs=2,
        help=(
            "Operators corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are evaluated."
        ),
        metavar=("X", "Y"),
        required=True,
    )

    args = parser.parse_args()

    assert (
        isinstance(args.dim, list)
        and (args.op, list)
        and isinstance(args.p_range, list)
    )

    p_lo, p_hi = args.p_range
    assert isinstance(p_lo, int) and isinstance(p_hi, int)

    main(
        operators=[name_to_op(o) for o in args.op],
        dims=args.dim,
        p_lo=p_lo,
        p_hi=p_hi,
    )
