from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
from matplotlib import rc

from ufltogpu_artifacts.core import (
    Op,
    name_to_op,
    op_name,
)
from ufltogpu_artifacts.constants import num_transform_candidates


rc("text", usetex=True)
plt.style.use("seaborn-v0_8")

def main(
    operators: list[Op],
    dim: int,
    p_lo: int,
    p_hi: int,
    file_to_save_in: str | None,
) -> None:
    markers = ["o", "^", "s", "p", "*"]
    assert len(markers) >= len(operators)
    for op, marker in zip(operators, markers):
        xs = list(range(p_lo, p_hi+1))
        ys = [num_transform_candidates[(op, dim, p)]
              for p in range(p_lo, p_hi+1)]
        plt.semilogy(xs, ys, label=op_name(op), marker=marker)

    plt.xlabel("$p$")
    plt.ylabel("Number of transform candidates")
    plt.legend()

    if file_to_save_in is not None:
        plt.savefig(file_to_save_in)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to plot number of transform candidates."
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
        "--dim", type=int, choices=[2, 3]
    )
    parser.add_argument(
        "--p_range",
        type=int,
        nargs=2,
        help=(
            "Number of transform candidates corresponding to"
            " polynomial degrees {X, X+1, ..., Y} are plotted."
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
        default=None,
    )

    args = parser.parse_args()

    assert (
        isinstance(args.op, list)
        and isinstance(args.p_range, list)
    )
    p_lo, p_hi = args.p_range
    assert isinstance(p_lo, int) and isinstance(p_hi, int)
    assert isinstance(args.dim, int)

    main(
        operators=[name_to_op(o) for o in args.op],
        dim=args.dim,
        p_lo=p_lo,
        p_hi=p_hi,
        file_to_save_in=args.o,
    )
