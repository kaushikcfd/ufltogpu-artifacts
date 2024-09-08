from __future__ import annotations  # noqa: I001

# Disabled isort to init firedrake before anything else.

import firedrake as fd
import argparse
import logging
import sqlite3 as sql
from typing import Any, Sequence

from tabulate import tabulate

from pyop2.backends.cuda import cuda_backend
from pyop2.op2 import AbstractParloop
from pyop2.transforms.auto_tiling import AutotilingFallback

from ufltogpu_artifacts.constants import flops_per_cell
from ufltogpu_artifacts.core import (
    Op,
    get_active_cuda_device_and_version,
    name_to_op,
    op_name,
    get_nel1d_for_reported_data,
    get_num_cells,
    get_roofline_gflops,
)
from ufltogpu_artifacts.sql_utils import create_or_verify_db, record_runtime
from ufltogpu_artifacts.weak_forms import get_bilinear_form

import warnings

warnings.filterwarnings("ignore", category=AutotilingFallback)


fd.set_offloading_backend(cuda_backend)
logger = logging.getLogger(__name__)


def _get_parloop_and_args(
    assembler: fd.assemble.ParloopFormAssembler, out_func: fd.Function
) -> tuple[AbstractParloop, Sequence[Any]]:
    (parloop,) = assembler.parloops(out_func)
    iterset_part = parloop.iterset.core_part
    return parloop, (
        parloop.comm,
        iterset_part.offset,
        iterset_part.size,
        *parloop.arglist,
    )


def get_runtime_in_s(*, op: Op, dim: int, p: int, nel_1d: int) -> float:
    """
    Returns the average runtime of applying operator *op* in dimension *dim*
    with polynomial discretization function spaces of degree *p*. The spatial
    discretization is a tesselated cube in *dim*-dimensions.
    """
    import pycuda.driver as cuda

    fd.parameters["pyop2_options"]["gpu_strategy"] = "auto_tiling"

    N_WARMUP = 5
    NMIN_TIMING_ROUNDS = 1000
    NROUNDS_BATCH = 100
    NMIN_RUNTIME_IN_MS = 1000

    A, V = get_bilinear_form(nel_1d, dim, op, p)
    x = fd.Function(V)
    y = fd.Function(V)
    expr = fd.action(A, x)

    x_ref = fd.Function(V)
    y_ref = fd.Function(V)
    fd.assemble(fd.action(A, x_ref), tensor=y_ref)

    with fd.offloading():
        fd.assemble(expr, tensor=y)

    l2_err = fd.norm(y - y_ref)
    assert l2_err < 1e-6

    with fd.offloading():
        assembler = fd.get_assembler(expr)
        assembler.assemble(tensor=y)
        parloop, arglist = _get_parloop_and_args(assembler, y)

        for _ in range(N_WARMUP):
            parloop.global_kernel(*arglist)

        irounds = 0
        acc_runtime_in_ms = 0.0

        while irounds < NMIN_TIMING_ROUNDS or acc_runtime_in_ms < NMIN_RUNTIME_IN_MS:
            y.zero()  # zero out to avoid any overflow issues.
            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()
            for _ in range(NROUNDS_BATCH):
                parloop.global_kernel(*arglist)
            end_evt.record()
            end_evt.synchronize()

            irounds += NROUNDS_BATCH
            acc_runtime_in_ms += end_evt.time_since(start_evt)

    mean_runtime_in_ms = acc_runtime_in_ms / irounds
    return mean_runtime_in_ms * 1e-3


def get_flops(*, op: Op, dim: int, p: int, nel_1d: int) -> int:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    return get_num_cells(dim, nel_1d) * flops_per_cell[(op, dim, p)]


def main(
    *,
    conn: sql.Connection | None,
    operators: Sequence[Op],
    dims: Sequence[int],
    p_lo: int,
    p_hi: int,
) -> None:
    timings_table: list[tuple[str, float, float]] = []
    cursor = conn.cursor() if conn else None

    device, cuda_sdk_version = get_active_cuda_device_and_version()

    for dim in dims:
        nel_1d = get_nel1d_for_reported_data(dim)
        num_cells = get_num_cells(dim, nel_1d)
        for op in operators:
            for p in range(p_lo, p_hi + 1):
                t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel_1d=nel_1d)
                nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)
                roofline_gflops = get_roofline_gflops(
                    op, dim, p, num_cells, get_active_cuda_device_and_version()[0]
                )
                timings_table.append(
                    (
                        f"{op_name(op)}.{dim}D.P{p}",
                        1e-9 * (nflops / t_op),
                        roofline_gflops,
                    )
                )

                record_runtime(
                    cursor=cursor,
                    op=op,
                    dim=dim,
                    p=p,
                    num_cells=num_cells,
                    runtime_in_s=t_op,
                    device=device,
                    cuda_sdk_version=cuda_sdk_version,
                )

            if cursor is not None:
                cursor.connection.commit()

    print(
        tabulate(
            timings_table,
            headers=("Operator", "GFLOPS", "Roofline\nGFLOPS"),
            tablefmt="fancy_grid",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Utility to obtain throughput of Kulkarni, Kloeckner's auto-tiling strategy"
            " and write them to a SQL-database."
        )
    )
    parser.add_argument(
        "--op",
        type=str,
        nargs="*",
        default=[],
        help="",
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
        "--db_path",
        type=str,
        nargs=1,
        help=(
            "Path to the SQL database which is to be updated. A new one will be created"
            " if none exists at the provided path."
        ),
        required=False,
        default=None,
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
    conn: sql.Connection | None = None

    if args.db_path is not None:
        assert isinstance(args.db_path, str)
        conn = create_or_verify_db(args.db_path)

    p_lo, p_hi = args.p_range
    assert isinstance(p_lo, int) and isinstance(p_hi, int)

    main(
        conn=conn,
        operators=[name_to_op(o) for o in args.op],
        dims=args.dim,
        p_lo=p_lo,
        p_hi=p_hi,
    )
