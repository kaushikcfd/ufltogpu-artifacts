from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ufltogpu_artifacts.core import Device, Op, name_to_op


@dataclass(frozen=True)
class TestCase:
    op: Op
    dim: int
    p: int
    device: Device


RE_TEST_CASE = re.compile(r"^(\w+).(\d)D.P(\d).(\w+)$")


def _parse_test_case(s: str) -> TestCase:
    match = RE_TEST_CASE.match(s)
    opname, dim, p, device_name = match.groups()
    assert device_name in ("H200NVL", "TITANV")
    device = Device.H200NVL if device_name == "H200NVL" else Device.TITANV
    return TestCase(name_to_op(opname), int(dim), int(p), device)


def main():
    candidate_times: dict[TestCase, tuple[float, ...]] = {}
    rhos: dict[TestCase, float] = {}
    with open("random_rankings.csv") as fp:
        for row in fp:
            cells = row.split(", ")
            print(cells[0])
            candidate_times[_parse_test_case(cells[0])] = tuple(
                float(cell) for cell in cells[1:]
            )
    for test_case, times in candidate_times.items():
        print(f"{test_case}, {len(times)}", end="")
        if len(times) > 5:
            rho = spearmanr(
                np.arange(len(times)), np.array(times)
            ).statistic
            rhos[test_case] = rho
            print(f", {rho}", end="")
        print()

    print(
        f"Average rho (across {len(rhos)} cases) = {np.average(list(rhos.values()))}."
    )


if __name__ == "__main__":
    main()
