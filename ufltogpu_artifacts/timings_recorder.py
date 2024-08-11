from __future__ import annotations

import argparse
import logging
import sqlite3 as sql
from datetime import datetime
from typing import Sequence

import pytz

from .core import Op


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
            " timestamp TEXT"
            ")"
        )
        conn.commit()

    return conn


def get_runtime_in_s(*, op: Op, dim: int, p: int, num_cells: int) -> float:
    """
    Returns the average runtime of applying operator *op* in dimension *dim*
    with polynomial discretization function spaces of degree *p*.
    """
    raise NotImplementedError


def get_flops(*, op: Op, dim: int, p: int, num_cells: int) -> float:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    raise NotImplementedError


def record_runtime(
    *, cursor: sql.Cursor, op: Op, dim: int, p: int, num_cells: int, runtime_in_s: float
) -> None:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    device_name = 1/0

    timestamp = (datetime
                 .now(pytz.timezone("America/Chicago"))
                 .strftime("%Y_%m_%d_%H%M%S"))

    cursor.execute(
        "INSERT INTO AUTOTILING_TIMES"
        "(op, dim, degree, device_name, n_cells, runtime_in_sec)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        op_name(op),
        dim,
        p,
        device_name,
        num_cells,
        runtime_in_s,
        timestamp,
    )
    cursor.commit()


def _get_nel1d(dim: int) -> int:
    if dim == 2:
        return 256
    elif dim == 3:
        1/0
    else:
        raise NotImplementedError





def main(
    *,
    conn: sql.Connection,
    operators: Sequence[Op],
    dims: Sequence[int],
    p_lo: int,
    p_hi: int,
) -> None:
    cursor = conn.cursor()
    for dim in dims:
        num_cells = 1 / 0
        for op in operators:
            for p in range(p_lo, p_hi+1):
                t_op = get_runtime_in_s(op=op, dim=dim, p=p)
                record_runtime(
                    cursor=cursor,
                    op=op,
                    dim=dim,
                    p=p,
                    num_cells=num_cells,
                    runtime_in_s=t_op,
                )

    conn.close()


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
    )
    parser.add_argument(
        "--db_path",
        type=str,
        nargs=1,
        help=(
            "Path to the SQL database which is to be updated. A new one will be created"
            " if none exists at the provided path."
        ),
    )
    parser.add_argument(
        "--dim",
        choices=[2, 3],
        type=int,
        nargs="+",
        help="Toplogical dimension on which the function spaces are to be defined.",
    )
    parser.add_argument(
        "--p_range",
        type=int,
        nargs=2,
        help=(
            "Two integers of the form `--p_range X Y`. Operators corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are evaluated."
        ),
    )

    args = parser.parse_args()

    assert (
        isinstance(args.dim, list)
        and (args.op, list)
        and isinstance(args.p_range, list)
    )
    assert isinstance(args.db_path, str)

    p_lo, p_hi = args.p_range
    assert isinstance(args.p_lo, int) and isinstance(args.p_hi, int)

    conn = create_or_verify_db(args.db_path)

    main(
        conn=conn,
        operators=args.op,
        dims=args.dims,
        p_lo=args.p_lo,
        p_hi=args.p_hi,
    )
