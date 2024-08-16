from __future__ import annotations

import argparse
import logging
import sqlite3 as sql
from datetime import datetime
from typing import Sequence

import pytz
from pyop2.backends.cuda import cuda_backend
from tabulate import tabulate

import firedrake as fd

from ufltogpu_artifacts.core import (
    Op,
    device_name,
    flops_per_cell,
    get_active_cuda_device_and_version,
    op_name,
)
from ufltogpu_artifacts.weak_forms import get_bilinear_form


fd.set_offloading_backend(cuda_backend)
logger = logging.getLogger(__name__)


def create_or_verify_db(
    path: str,
) -> sql.Connection:

    conn = sql.connect(path)
    cursor = conn.cursor()

    cursor.execute(
        " SELECT name FROM sqlite_master WHERE (type='table' AND"
        " name='AUTOTILING_TIMES');"
    )

    if cursor.fetchall():
        # Found the table.
        # Verify the columns
        cursor.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE"
            " TABLE_NAME='AUTOTILING_TIMES'"
        )
        if cursor.fetchall() != [
            "ID",
            "op",
            "dim",
            "degree",
            "device_name",
            "n_cells",
            "runtime_in_sec",
            "cuda_sdk_version",
            "timestamp",
        ]:
            raise RuntimeError(
                f"Provided database ('{path}') contains a table with incorrect columns."
            )
    else:
        cursor.execute(
            "CREATE TABLE AUTOTILING_TIMES ("
            " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            " op TEXT,"
            " dim INT,"
            " degree INT,"
            " device_name TEXT,"
            " n_cells INT,"
            " runtime_in_sec REAL,"
            " cuda_sdk_version TEXT,"
            " timestamp TEXT"
            ")"
        )
        conn.commit()

    return conn


def get_runtime_in_s(*, op: Op, dim: int, p: int, nel_1d: int) -> float:
    """
    Returns the average runtime of applying operator *op* in dimension *dim*
    with polynomial discretization function spaces of degree *p*. The spatial
    discretization is a tesselated cube in *dim*-dimensions.
    """
    import pycuda.driver as cuda
    import pyop2.op2 as op2

    N_WARMUP = 5
    NMIN_TIMING_ROUNDS = 40
    NROUNDS_BATCH = 10
    NMIN_RUNTIME_IN_MS = 1000

    op2.configuration.reconfigure(gpu_strategy="auto_tiling")

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
        for _ in range(N_WARMUP):
            fd.assemble(expr, tensor=y)

        irounds = 0
        acc_runtime_in_ms = 0.0

        while irounds < NMIN_TIMING_ROUNDS and acc_runtime_in_ms < NMIN_RUNTIME_IN_MS:

            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()
            for _ in range(NROUNDS_BATCH):
                fd.assemble(expr, tensor=y)
            end_evt.record()
            end_evt.synchronize()

            irounds += NROUNDS_BATCH
            acc_runtime_in_ms += end_evt.time_since(start_evt)

    mean_runtime_in_ms = acc_runtime_in_ms / irounds
    return mean_runtime_in_ms * 1e-3


def _get_nel1d(dim: int) -> int:
    if dim == 2:
        return 256
    elif dim == 3:
        raise NotImplementedError
    else:
        raise NotImplementedError


def _get_num_cells(dim: int, nel_1d: int) -> int:
    """
    Returns the number of cells with a hypercube in *dim*-dimensions with
    *nel_1d+1* vertices along each edge of the cube. We choose Firedrake's
    "UnitCubeMesh" spatial discretization for computing the total number
    of cells in the mesh.
    """
    if dim == 2:
        return 2 * nel_1d * nel_1d
    elif dim == 3:
        return 6 * nel_1d * nel_1d * nel_1d
    else:
        raise ValueError(f"{dim=}")


def get_flops(*, op: Op, dim: int, p: int, nel_1d: int) -> int:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    return _get_num_cells(dim, nel_1d) * flops_per_cell[(op, dim, p)]


def record_runtime(
    *,
    cursor: sql.Cursor | None,
    op: Op,
    dim: int,
    p: int,
    num_cells: int,
    runtime_in_s: float,
) -> None:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    if cursor is None:
        return

    device, cuda_sdk_version = get_active_cuda_device_and_version()

    timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y_%m_%d_%H%M%S"
    )

    cursor.execute(
        "INSERT INTO AUTOTILING_TIMES"
        "(op, dim, degree, device_name, n_cells, cuda_sdk_version, runtime_in_sec)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        op_name(op),
        dim,
        p,
        device_name(device),
        num_cells,
        runtime_in_s,
        cuda_sdk_version,
        timestamp,
    )


def main(
    *,
    conn: sql.Connection | None,
    operators: Sequence[Op],
    dims: Sequence[int],
    p_lo: int,
    p_hi: int,
) -> None:
    timings_table: list[tuple[str, float]] = []
    cursor = conn.cursor() if conn else None

    for dim in dims:
        nel_1d = _get_nel1d(dim)
        num_cells = _get_num_cells(dim, nel_1d)
        for op in operators:
            for p in range(p_lo, p_hi + 1):
                t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel1_d=nel_1d)
                nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)
                timings_table.append(
                    (f"{op_name(op)}.{dim}D.P{p}", 1e-9 * (nflops / t_op))
                )

                record_runtime(
                    cursor=cursor,
                    op=op,
                    dim=dim,
                    p=p,
                    num_cells=num_cells,
                    runtime_in_s=t_op,
                )

            cursor.commit()

    print(
        tabulate(timings_table, headers=("Operator", "GFLOPS"), tablefmt="fancy_grid")
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
        nargs="+",
        help="Toplogical dimension on which the function spaces are to be defined.",
        required=True,
    )
    parser.add_argument(
        "--p_range",
        type=int,
        nargs=2,
        help=(
            "Two integers of the form `--p_range X Y`. Operators corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are evaluated."
        ),
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
    assert isinstance(args.p_lo, int) and isinstance(args.p_hi, int)

    main(
        conn=conn,
        operators=args.op,
        dims=args.dims,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
    )
