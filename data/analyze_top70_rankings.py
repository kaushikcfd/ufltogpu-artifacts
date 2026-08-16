from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from tabulate import tabulate

from ufltogpu_artifacts.core import Device, Op, name_to_op


@dataclass(frozen=True)
class TestCase:
    op: Op
    dim: int
    p: int
    device: Device

    def is_low_order(self) -> bool:
        if self.dim == 2:
            return self.p <= 4
        else:
            assert self.dim == 3
            return self.p <= 3


RE_TEST_CASE = re.compile(r"^Op.(\w+).(\d)D.P(\d).(\w+)$")


@cache
def _parse_test_case(s: str) -> TestCase:
    match = RE_TEST_CASE.match(s)
    assert match, f"{s!r}"
    opname, dim, p, device_name = match.groups()
    assert device_name in ("H200NVL", "TITANV")
    device = Device.H200NVL if device_name == "H200NVL" else Device.TITANV
    return TestCase(name_to_op(opname), int(dim), int(p), device)


def main() -> None:
    df = pd.read_csv("top70_rankings.csv", index_col=0).loc[:, "Rank-1":]
    regret_table: list[list[float]] = []
    topks = {k: df.loc[:, "Rank-1":f"Rank-{k}"].min(axis=1) for k in [1, 5, 10, 20, 70]}

    cols = {
        "q_titanv_top1": (
            (lambda row: _parse_test_case(row).device == Device.TITANV),
            topks[1],
        ),
        "q_titanv_top5": (
            (lambda row: _parse_test_case(row).device == Device.TITANV),
            topks[5],
        ),
        "q_titanv_top10": (
            (lambda row: _parse_test_case(row).device == Device.TITANV),
            topks[10],
        ),
        "q_titanv_top20": (
            (lambda row: _parse_test_case(row).device == Device.TITANV),
            topks[20],
        ),
        "q_h200_top1": (
            (lambda row: _parse_test_case(row).device == Device.H200NVL),
            topks[1],
        ),
        "q_h200_top5": (
            (lambda row: _parse_test_case(row).device == Device.H200NVL),
            topks[5],
        ),
        "q_h200_top10": (
            (lambda row: _parse_test_case(row).device == Device.H200NVL),
            topks[10],
        ),
        "q_h200_top20": (
            (lambda row: _parse_test_case(row).device == Device.H200NVL),
            topks[20],
        ),
        "q_both_top1": ((lambda row: True), topks[1]),
        "q_both_top5": ((lambda row: True), topks[5]),
        "q_both_top10": ((lambda row: True), topks[10]),
        "q_both_top20": ((lambda row: True), topks[20]),
    }
    rows = {
        "all": lambda row: True,
        "twod": lambda row: _parse_test_case(row).dim == 2,
        "threed": lambda row: _parse_test_case(row).dim == 3,
        "loworder": lambda row: _parse_test_case(row).is_low_order(),
        "highorder": lambda row: not _parse_test_case(row).is_low_order(),
        "mass": lambda row: _parse_test_case(row).op == Op.MASS,
        "laplace": lambda row: _parse_test_case(row).op == Op.LAPLACE,
        "helmholtz": lambda row: _parse_test_case(row).op == Op.HELMHOLTZ,
        "elasticity": lambda row: _parse_test_case(row).op == Op.ELASTICITY,
        "hyperelasticity": lambda row: _parse_test_case(row).op == Op.HYPERELASTICITY,
    }

    for row, rowfilter in rows.items():
        regret_row: list[str] = [row]
        for colfilter, topk in cols.values():
            topk_filtered = topk[
                topk.index.map(
                    lambda x, rowfilter=rowfilter, colfilter=colfilter: (
                        rowfilter(x) and colfilter(x)
                    )
                )
            ]
            top70_filtered = topks[70][
                topk.index.map(
                    lambda x, rowfilter=rowfilter, colfilter=colfilter: (
                        rowfilter(x) and colfilter(x)
                    )
                )
            ]
            ratio_np = (top70_filtered / topk_filtered).to_numpy()
            assert isinstance(ratio_np, np.ndarray) and ratio_np.ndim == 1
            regret_row.append(f"{np.exp(np.log(ratio_np).mean()):.2f}")
        regret_table.append(regret_row)
    print(tabulate(regret_table, headers=["", *list(cols)], tablefmt="fancy_grid"))
    with open("topk_regret.csv", "w") as fp:
        fp.write(", ".join(["", *list(cols)]))
        fp.write("\n")
        for row in regret_table:
            fp.write(", ".join(row))
            fp.write("\n")


if __name__ == "__main__":
    main()
