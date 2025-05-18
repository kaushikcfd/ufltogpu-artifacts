from __future__ import annotations

import argparse
import sqlite3 as sql

import matplotlib.pyplot as plt
import numpy as np

from ufltogpu_artifacts.constants import flops_per_cell
from ufltogpu_artifacts.core import get_roofline_gflops, name_to_device, name_to_op
from ufltogpu_artifacts.sql_utils import create_or_verify_db


plt.style.use("seaborn-v0_8")
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)


def main(conn: sql.Connection, file_to_save_in: None | str):
    cursor = conn.cursor()
    windows = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    yticks = [rf"\Large {int(100*window)}" for window in windows]
    markers = ["o-", "^-", "s-", "p-", "*-"]

    cursor.execute("SELECT device_name FROM AUTOTILING_TIMES;")
    devices: list[str] = sorted({data_pt[0] for data_pt in cursor.fetchall()})
    assert len(markers) >= len(devices)

    for device, marker in zip(devices, markers):
        flop_ratios = []

        cursor.execute(
            "SELECT op, dim, degree, runtime_in_sec, n_cells "
            "FROM AUTOTILING_TIMES WHERE (device_name = ?);",
            (device,),
        )
        for op_name, dim, p, runtime_in_sec, n_cells in cursor.fetchall():
            assert isinstance(op_name, str)
            assert isinstance(dim, int)
            assert isinstance(p, int)
            assert isinstance(runtime_in_sec, float)
            assert isinstance(n_cells, int)
            op = name_to_op(op_name)
            empirical_gflops = (
                1e-9 * n_cells * flops_per_cell[op, dim, p]
            ) / runtime_in_sec
            roofline_gflops = get_roofline_gflops(
                name_to_op(op_name),
                dim=dim,
                deg=p,
                num_cells=n_cells,
                device=name_to_device(device),
            )
            flop_ratios.append(empirical_gflops / roofline_gflops)

        flop_ratios_np = np.array(flop_ratios)

        cases_in_a_window = []
        for window in windows:
            cases_in_a_window.append(
                100 * ((flop_ratios_np >= window).mean())
            )

        plt.plot(cases_in_a_window, windows, marker, label=r"\Large " + device)

    plt.xlabel(r"\Large $x$\% of test cases")
    plt.ylabel(r"\Large $y$\% of roofline")
    plt.title(r"\Large $x$\% test cases performing at least $y$\% of the roofline")
    plt.yticks(windows, yticks)

    plt.legend()
    if file_to_save_in:
        plt.savefig(file_to_save_in)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to plot aggregated relative roofline performance."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        help=(
            "Path to the SQL database which is to be updated. A new one will be created"
            " if none exists at the provided path."
        ),
        required=True,
    )

    parser.add_argument(
        "-o",
        type=str,
        help=(
            "Place the generated plots in <file>. Plot is displayed using X-window if"
            " none passed."
        ),
        metavar="<file>",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    assert isinstance(args.db_path, str)
    conn = create_or_verify_db(args.db_path, False)

    main(
        conn=conn,
        file_to_save_in=args.o,
    )
