from __future__ import annotations

import argparse
import sqlite3 as sql

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import rc

from ufltogpu_artifacts.core import (
    Device,
    Op,
    create_or_verify_db,
    device_name,
    fetch_flops,
    get_nel1d_for_reported_data,
    get_num_cells,
    get_roofline_gflops,
    name_to_device,
    name_to_op,
    op_name,
)


rc("text", usetex=True)

plt.style.use("seaborn")


def main(
    conn: sql.Connection,
    device: Device,
    operators: list[Op],
    p_lo_2d: int,
    p_hi_2d: int,
    p_lo_3d: int,
    p_hi_3d: int,
    file_to_save_in: str | None,
) -> None:
    cursor = conn.cursor()

    _, plt_axes = plt.subplots(
        len(operators),
        1,
        figsize=(5, 6 * len(operators)),
        squeeze=False,
        gridspec_kw={"hspace": 0.3},
    )

    for plt_ax, op in zip(plt_axes, operators):
        expt_flops = []
        roofline_flops = []

        xticks = [f"{op_name(op)[:5]}.2D.P{p}" for p in range(p_lo_2d, p_hi_2d + 1)] + [
            f"{op_name(op)[:5]}.3D.P{p}" for p in range(p_lo_3d, p_hi_3d + 1)
        ]
        expt_flops = [
            fetch_flops(
                cursor,
                device,
                op,
                2,
                p,
                get_num_cells(2, get_nel1d_for_reported_data(2)),
            )
            for p in range(p_lo_2d, p_hi_2d + 1)
        ] + [
            fetch_flops(
                cursor,
                device,
                op,
                3,
                p,
                get_num_cells(3, get_nel1d_for_reported_data(3)),
            )
            for p in range(p_lo_3d, p_hi_3d + 1)
        ]
        roofline_flops = [
            get_roofline_gflops(
                op, 2, p, get_num_cells(2, get_nel1d_for_reported_data(2))
            )
            for p in range(p_lo_2d, p_hi_2d + 1)
        ] + [
            get_roofline_gflops(
                op, 3, p, get_num_cells(3, get_nel1d_for_reported_data(3))
            )
            for p in range(p_lo_3d, p_hi_3d + 1)
        ]

        bar_width = 0.2
        r1 = np.arange(len(xticks))
        r2 = [x + bar_width for x in r1]
        pos_list = [r + 0.5 * bar_width for r in range(len(xticks))]

        plt_ax.bar(
            r1, expt_flops, width=bar_width, edgecolor="black", label="Experiment"
        )
        plt_ax.bar(
            r2, roofline_flops, width=bar_width, edgecolor="black", label="Roofline"
        )

        plt_ax.xaxis.set_major_locator(ticker.FixedLocator(pos_list))
        plt_ax.xaxis.set_major_formatter(ticker.FixedFormatter(xticks))

        plt_ax.set_ylabel("GFLOPS")
        plt.setp(plt_ax.get_xticklabels(), rotation=15, fontsize=6)

    plt_axes[0].legend(bbox_to_anchor=(0.0, 1.04, 1.0, 0.3), loc="lower center", ncol=3)

    if file_to_save_in is not None:
        plt.savefig(file_to_save_in)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to plot throughputs stored  in the SQL-database."
    )
    parser.add_argument(
        "--op",
        type=str,
        nargs="+",
        default=[],
        help="",
        choices=[op_name(op) for op in Op],
        required=True,
    )
    parser.add_argument(
        "--device", type=str, nargs=1, choices=[device_name(dev) for dev in Device]
    )
    parser.add_argument(
        "--db_path",
        type=str,
        nargs=1,
        help=(
            "Path to the SQL database which is to be updated. A new one will be created"
            " if none exists at the provided path."
        ),
        required=True,
    )
    parser.add_argument(
        "--p_range_2d",
        type=int,
        nargs=2,
        help=(
            "2-D Operators corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are evaluated."
        ),
        metavar=("X", "Y"),
        required=True,
    )

    parser.add_argument(
        "--p_range_3d",
        type=int,
        nargs=2,
        help=(
            "3-D Operators corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are evaluated."
        ),
        metavar=("X", "Y"),
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
        default=False,
    )

    args = parser.parse_args()

    assert (
        isinstance(args.dim, list)
        and (args.op, list)
        and isinstance(args.p_range_2d, list)
        and isinstance(args.p_range_3d, list)
    )

    conn = create_or_verify_db(args.db_path, False)

    p_lo_2d, p_hi_2d = args.p_range_2d
    p_lo_3d, p_hi_3d = args.p_range_3d
    assert isinstance(p_lo_2d, int) and isinstance(p_hi_2d, int)
    assert isinstance(p_lo_3d, int) and isinstance(p_hi_3d, int)

    main(
        conn=conn,
        device=name_to_device(args.device),
        operators=[name_to_op(o) for o in args.op],
        p_lo_2d=p_lo_2d,
        p_hi_2d=p_hi_2d,
        p_lo_3d=p_lo_3d,
        p_hi_3d=p_hi_3d,
    )
