from __future__ import annotations

import logging
import os
from typing import Sequence

# firedrake_clean() rmtree's the pytools cache dir, but loopy keeps a sqlite
# connection into it open for the process lifetime (opened at import time).
# Deleting the dir out from under that connection breaks later writes. Disabling
# loopy's disk cache before firedrake/loopy are imported avoids it.
os.environ["LOOPY_NO_CACHE"] = "1"

import pyop2.transforms.auto_tiling
from tabulate import tabulate

import firedrake as fd  # ruff: ignore[unused-import] (import for PETSc to init ctx.)

from ufltogpu_artifacts.core import Op, get_nel1d_for_reported_data, op_name
from ufltogpu_artifacts.timings_recorder import get_flops, get_runtime_in_s

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.CRITICAL)

CONFIGS: Sequence[tuple[Op, int, int]] = (
    (Op.MASS, 2, 2),
    (Op.HYPERELASTICITY, 2, 4),
    (Op.ELASTICITY, 3, 2),
)


def firedrake_clean() -> None:
    import os
    import shutil
    import sys

    from pyop2.configuration import configuration as pyop2_configuration

    try:
        import platformdirs as appdirs
    except ImportError:
        import appdirs

    pyop2_cache = pyop2_configuration["cache_dir"]
    pytools_cache = appdirs.user_cache_dir("pytools", "pytools")

    for cache in [pyop2_cache, pytools_cache]:
        if os.path.exists(cache):
            shutil.rmtree(cache, ignore_errors=True)

    global_kernel = sys.modules.get("pyop2.global_kernel")
    if global_kernel is not None:
        global_kernel.AbstractGlobalKernel._cache.clear()

    assemble = sys.modules.get("firedrake.assemble")
    if assemble is not None:
        assemble._make_global_kernel.cache.clear()


def main() -> None:
    scpwi_w_values = [32, 64, 96, 128]

    perf_table: dict[str, dict[float, str]] = {}
    pyop2.transforms.auto_tiling.NUM_TILING_CANDIDATES = 0

    for scpwi_w in scpwi_w_values:
        firedrake_clean()
        pyop2.transforms.auto_tiling.SCPWI_W = scpwi_w

        for op, dim, p in CONFIGS:
            nel_1d = get_nel1d_for_reported_data(dim)
            t_op = get_runtime_in_s(op=op, dim=dim, p=p, nel_1d=nel_1d)
            nflops = get_flops(op=op, dim=dim, p=p, nel_1d=nel_1d)

            row_label = f"{op_name(op)}.{dim}D.P{p}"
            perf_table.setdefault(row_label, {})[
                scpwi_w
            ] = f"{1e-9 * (nflops / t_op):.1f}"
            logger.critical(f"Done with {scpwi_w=}, {op=}, {dim=}, {p=}")

    print(
        tabulate(
            [
                (row_label, *(gflops[scpwi_w] for scpwi_w in scpwi_w_values))
                for row_label, gflops in perf_table.items()
            ],
            headers=(
                "Operator",
                *(f"SCPWI_WG_SIZE={scpwi_w}" for scpwi_w in scpwi_w_values),
            ),
            tablefmt="fancy_grid",
        )
    )


if __name__ == "__main__":
    main()
