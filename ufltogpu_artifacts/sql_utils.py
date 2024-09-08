from __future__ import annotations

import sqlite3 as sql
from datetime import datetime
from os import PathLike

import pytz

from ufltogpu_artifacts.core import (
    Device,
    Op,
    device_name,
    op_name,
)


StrOrBytesPath = str | bytes | PathLike[str] | PathLike[bytes]


def create_or_verify_db(
    path: StrOrBytesPath, create_if_does_not_exist: bool = True
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
        cursor.execute("PRAGMA table_info(AUTOTILING_TIMES);")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]
        if column_names != [
            "ID",
            "op",
            "dim",
            "degree",
            "n_cells",
            "device_name",
            "cuda_sdk_version",
            "runtime_in_sec",
            "timestamp",
        ]:
            raise RuntimeError(
                f"Provided database ('{path}') contains a table with incorrect columns."
            )
    else:
        if not create_if_does_not_exist:
            raise RuntimeError(
                f"Provided database ('{path}') either does not exist or does not"
                " contain the timings table."
            )
        cursor.execute(
            "CREATE TABLE AUTOTILING_TIMES ("
            " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            " op TEXT,"
            " dim INT,"
            " degree INT,"
            " n_cells INT,"
            " device_name TEXT,"
            " cuda_sdk_version TEXT,"
            " runtime_in_sec REAL,"
            " timestamp TEXT"
            ")"
        )
        conn.commit()

    return conn


def record_runtime(
    *,
    cursor: sql.Cursor | None,
    op: Op,
    dim: int,
    p: int,
    num_cells: int,
    runtime_in_s: float,
    device: Device,
    cuda_sdk_version: str,
) -> None:
    """
    Returns the runtime of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    if cursor is None:
        return

    timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y_%m_%d_%H%M%S"
    )

    cursor.execute(
        "INSERT INTO AUTOTILING_TIMES(op, dim, degree, n_cells, device_name,"
        " cuda_sdk_version, runtime_in_sec, timestamp) VALUES (?, ?, ?, ?, ?, ?,"
        " ?, ?)",
        (
            op_name(op),
            dim,
            p,
            num_cells,
            device_name(device),
            cuda_sdk_version,
            runtime_in_s,
            timestamp,
        ),
    )


def fetch_flops(
    cursor: sql.Cursor,
    device: Device,
    op: Op,
    dim: int,
    degree: int,
    num_cells: int,
) -> float:
    """
    Returns the GFLOPS recorded in the SQL database active in *cursor* for
    running the action operator *op* on the device *device*.
    """
    from .constants import flops_per_cell

    cursor.execute(
        "SELECT runtime_in_sec FROM AUTOTILING_TIMES WHERE (op = ?  AND dim = ? AND"
        " degree = ? AND device_name = ? AND n_cells = ?);",
        (op_name(op), dim, degree, device_name(device), num_cells),
    )
    runtimes = cursor.fetchall()
    if len(runtimes) > 1:
        raise NotImplementedError("Mulitple data points, not sure which to pick.")

    return (1e-9 * num_cells * flops_per_cell[(op, dim, degree)]) / runtimes[0][0]


def record_gflops(
    *,
    cursor: sql.Cursor,
    op: Op,
    dim: int,
    p: int,
    num_cells: int,
    gflops: float,
    device: Device,
    cuda_sdk_version: str,
) -> None:
    """
    Returns the Floating-point operations of applying operator *op* in dimension
    *dim* with polynomial discretization function spaces of degree *p*.
    """
    from ufltogpu_artifacts.constants import flops_per_cell

    ngflops = 1e-9 * num_cells * flops_per_cell[(op, dim, p)]
    runtime_in_s = ngflops / gflops
    record_runtime(
        cursor=cursor,
        op=op,
        dim=dim,
        p=p,
        num_cells=num_cells,
        runtime_in_s=runtime_in_s,
        device=device,
        cuda_sdk_version=cuda_sdk_version,
    )
