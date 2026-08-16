from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.stats import spearmanr
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


RE_TEST_CASE = re.compile(r"^(\w+).(\d)D.P(\d).(\w+)$")


@cache
def _parse_test_case(s: str) -> TestCase:
    match = RE_TEST_CASE.match(s)
    opname, dim, p, device_name = match.groups()
    assert device_name in ("H200NVL", "TITANV")
    device = Device.H200NVL if device_name == "H200NVL" else Device.TITANV
    return TestCase(name_to_op(opname), int(dim), int(p), device)


def get_test_case_to_rho() -> Mapping[TestCase, float]:
    candidate_times: dict[TestCase, tuple[float, ...]] = {}
    rhos: dict[TestCase, float] = {}
    with open("random_rankings.csv") as fp:
        for row in fp:
            cells = row.split(", ")
            candidate_times[_parse_test_case(cells[0])] = tuple(
                float(cell) for cell in cells[1:]
            )
    for test_case, times in candidate_times.items():
        if len(times) > 5:
            rho = spearmanr(np.arange(len(times)), np.array(times)).statistic
            rhos[test_case] = rho

    return rhos


def main() -> None:
    rhos = get_test_case_to_rho()

    cols: Mapping[str, Callable[[TestCase], bool]] = {
        "rho_titanv": lambda k: k.device == Device.TITANV,
        "rho_h200": lambda k: k.device == Device.H200NVL,
        "rho_both": lambda k: True,
    }

    rows: Mapping[str, Callable[[TestCase], bool]] = {
        "all": lambda k: True,
        "twod": lambda k: k.dim == 2,
        "threed": lambda k: k.dim == 3,
        "loworder": lambda k: k.is_low_order(),
        "highorder": lambda k: not k.is_low_order(),
        "mass": lambda k: k.op == Op.MASS,
        "laplace": lambda k: k.op == Op.LAPLACE,
        "helmholtz": lambda k: k.op == Op.HELMHOLTZ,
        "elasticity": lambda k: k.op == Op.ELASTICITY,
        "hyperelasticity": lambda k: k.op == Op.HYPERELASTICITY,
    }
    spearman_table: list[list[str]] = []
    headers = ["", *cols]
    for row, rowfilter in rows.items():
        spearman_row = [row]
        for colfilter in cols.values():
            mean_spearman = np.mean(
                [v for k, v in rhos.items() if rowfilter(k) and colfilter(k)]
            )
            spearman_row.append(f"{mean_spearman:.2f}")
        spearman_table.append(spearman_row)

    print(tabulate(spearman_table, headers=headers, tablefmt="fancy_grid"))
    with open("spearman.csv", "w") as fp:
        fp.write(", ".join(headers))
        fp.write("\n")
        for row in spearman_table:
            fp.write(", ".join(row))
            fp.write("\n")


if __name__ == "__main__":
    main()
