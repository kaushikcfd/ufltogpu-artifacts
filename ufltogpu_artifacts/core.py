from __future__ import annotations

import enum
from functools import cache

import numpy as np


@enum.unique
class Op(enum.Enum):
    MASS = 0
    LAPLACE = 1
    HELMHOLTZ = 2
    ELASTICITY = 3
    HYPERELASTICITY = 4


@enum.unique
class Device(enum.Enum):
    K40M = 0
    K40C = 1
    TITANV = 2
    H200NVL = 3


def op_name(op: Op) -> str:
    if op == Op.MASS:
        return "Mass"
    elif op == Op.LAPLACE:
        return "Laplace"
    elif op == Op.HELMHOLTZ:
        return "Helmholtz"
    elif op == Op.ELASTICITY:
        return "Elasticity"
    elif op == Op.HYPERELASTICITY:
        return "Hyperelasticity"
    else:
        raise AssertionError("unreachable")


def device_name(device: Device) -> str:
    if device == Device.K40M:
        return "Tesla K40m"
    elif device == Device.K40C:
        return "Tesla K40c"
    elif device == Device.TITANV:
        return "NVIDIA TITAN V"
    elif device == Device.H200NVL:
        return "NVIDIA H200 NVL"
    else:
        raise AssertionError("unreachable")


def name_to_op(name: str) -> Op:
    name = name.capitalize()
    for op in Op:
        if op_name(op) == name:
            return op
    raise AssertionError("unreachable")


def name_to_device(name: str) -> Device:
    name = name.lower()
    for dev in Device:
        if device_name(dev).lower() == name:
            return dev
    raise AssertionError(f"unknown device={name}")


@cache
def get_active_cuda_device_and_version() -> tuple[Device, str]:
    import firedrake  # noqa: F401, I001 (import for PETSc to init ctx.)
    from pycuda.autoprimaryctx import context
    import pycuda.driver as cuda

    return name_to_device(context.get_device().name()), ".".join(
        str(v) for v in cuda.get_version()
    )


def get_nel1d_for_reported_data(dim: int) -> int:
    """
    Returns the number of elements along each dimension of a unit
    *dim*-dimension hypercube for the reported data in the paper.
    """
    if dim == 2:
        return 512
    elif dim == 3:
        return 32
    else:
        raise NotImplementedError


def get_num_cells(dim: int, nel_1d: int) -> int:
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


def get_roofline_gflops(
    op: Op, dim: int, deg: int, num_cells: int, device: Device
) -> float:
    """
    Returns the roofline FLOPS (as per Sec 5.1 of the paper)
    """
    from .constants import (
        flops_per_cell,
        local_nbytes_accesses_per_cell,
        nfootprint_bytes_per_cell,
    )

    if device in [Device.K40M, Device.K40C]:
        fpeak = 1430  # GFlops/s
        beta_peak_global = 288  # GB/s
        beta_peak_shared = 1000  # GB/s
    elif device == Device.TITANV:
        fpeak = 6144  # GFlops/s
        beta_peak_global = 653  # GB/s
        beta_peak_shared = 13800  # GB/s
    elif device == Device.H200NVL:
        # See <https://www.nvidia.com/en-in/data-center/h200/>.
        fpeak = 30_000  # GFlops/s (no tensor core)
        beta_peak_global = 3800  # GB/s from SHOC::writeCoalescedGlobalMemory.
        beta_peak_shared = 23_000  # GB/s  from SHOC::readLocalMemory
    else:
        raise ValueError(f"Unknown device {device_name(device)}.")

    flops_per_cell_for_op = flops_per_cell[op, dim, deg]
    nglobal_nbytes_per_cell = nfootprint_bytes_per_cell[op, dim, deg]
    nlocal_nbytes_per_cell = local_nbytes_accesses_per_cell[op, dim, deg]

    flops_global_bw = flops_per_cell_for_op * (
        beta_peak_global / nglobal_nbytes_per_cell
    )

    if not np.isnan(nlocal_nbytes_per_cell):
        flops_local_bw = flops_per_cell_for_op * (
            beta_peak_shared / nlocal_nbytes_per_cell
        )
    else:
        flops_local_bw = np.inf

    return min(flops_global_bw, flops_local_bw, fpeak)
