from __future__ import annotations

import enum
from functools import cache

import numpy as np


__doc__ = """
.. data:: flops_per_cell

    A mapping from ``(Operator, dim, degree)`` to number of floating point
    operations required to compute the corresponding action for one cell in the
    mesh.

.. data:: nfootprint_bytes

    A mapping from ``(Operator, dim, degree, num_cells)`` to footprint
    bytes accessed by the corresponding action operator.

.. data:: local_nbytes_accesses_per_cell

    A mapping from ``(Operator, dim, degree)`` to the number of local memory
    accesses as per the roofline model kernel in Sec 5.1 of the paper.
"""


@enum.unique
class Op(enum.Enum):
    MASS = 0
    LAPLACE = 1
    HELMHOLTZ = 2
    ELASTICITY = 3
    HYPERELASTICITY = 4


@enum.unique
class Device(enum.Enum):
    K40 = 0
    TITANV = 1


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
    if device == Device.K40:
        return "NVIDIA TESLA K40"
    elif device == Device.TITANV:
        return "NVIDIA TITAN V"
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
    raise AssertionError("unreachable")


@cache
def get_active_cuda_device_and_version() -> tuple[Device, str]:
    import firedrake  # noqa: F401, I001 (import for PETSc to init ctx.)
    from pycuda.autoprimaryctx import context
    import pycuda.driver as cuda
    return name_to_device(context.get_device().name()), ".".join(
        str(v) for v in cuda.get_version()
    )


flops_per_cell = {
    (Op.MASS, 2, 1): 47,
    (Op.MASS, 2, 2): 164,
    (Op.MASS, 2, 3): 512,
    (Op.MASS, 2, 4): 1000,
    (Op.MASS, 2, 5): 2158,
    (Op.MASS, 2, 6): 3770,
    (Op.MASS, 2, 7): 6140,
    (Op.MASS, 2, 8): 10018,
    (Op.MASS, 3, 1): 96,
    (Op.MASS, 3, 2): 486,
    (Op.MASS, 3, 3): 1910,
    (Op.MASS, 3, 4): 6272,
    (Op.MASS, 3, 5): 16748,
    (Op.MASS, 3, 6): 41260,
    (Op.LAPLACE, 2, 1): 42,
    (Op.LAPLACE, 2, 2): 202,
    (Op.LAPLACE, 2, 3): 583,
    (Op.LAPLACE, 2, 4): 1633,
    (Op.LAPLACE, 2, 5): 2941,
    (Op.LAPLACE, 2, 6): 5988,
    (Op.LAPLACE, 2, 7): 10012,
    (Op.LAPLACE, 2, 8): 15763,
    (Op.LAPLACE, 3, 1): 115,
    (Op.LAPLACE, 3, 2): 670,
    (Op.LAPLACE, 3, 3): 3068,
    (Op.LAPLACE, 3, 4): 10496,
    (Op.LAPLACE, 3, 5): 31118,
    (Op.LAPLACE, 3, 6): 77162,
    (Op.HELMHOLTZ, 2, 1): 86,
    (Op.HELMHOLTZ, 2, 2): 541,
    (Op.HELMHOLTZ, 2, 3): 1645,
    (Op.HELMHOLTZ, 2, 4): 3149,
    (Op.HELMHOLTZ, 2, 5): 6713,
    (Op.HELMHOLTZ, 2, 6): 11629,
    (Op.HELMHOLTZ, 2, 7): 18829,
    (Op.HELMHOLTZ, 2, 8): 30593,
    (Op.HELMHOLTZ, 3, 1): 194,
    (Op.HELMHOLTZ, 3, 2): 2199,
    (Op.HELMHOLTZ, 3, 3): 8219,
    (Op.HELMHOLTZ, 3, 4): 26234,
    (Op.HELMHOLTZ, 3, 5): 68948,
    (Op.HELMHOLTZ, 3, 6): 168292,
    (Op.ELASTICITY, 2, 1): 87,
    (Op.ELASTICITY, 2, 2): 440,
    (Op.ELASTICITY, 2, 3): 1247,
    (Op.ELASTICITY, 2, 4): 3437,
    (Op.ELASTICITY, 2, 5): 6113,
    (Op.ELASTICITY, 2, 6): 12342,
    (Op.ELASTICITY, 2, 7): 20510,
    (Op.ELASTICITY, 2, 8): 32147,
    (Op.ELASTICITY, 3, 1): 290,
    (Op.ELASTICITY, 3, 2): 2119,
    (Op.ELASTICITY, 3, 3): 9677,
    (Op.ELASTICITY, 3, 4): 32585,
    (Op.ELASTICITY, 3, 5): 95543,
    (Op.ELASTICITY, 3, 6): 235235,
    (Op.HYPERELASTICITY, 2, 1): 185,
    (Op.HYPERELASTICITY, 2, 2): 1712,
    (Op.HYPERELASTICITY, 2, 3): 6078,
    (Op.HYPERELASTICITY, 2, 4): 16481,
    (Op.HYPERELASTICITY, 2, 5): 35379,
    (Op.HYPERELASTICITY, 2, 6): 64083,
    (Op.HYPERELASTICITY, 2, 7): 112350,
    (Op.HYPERELASTICITY, 2, 8): 182864,
    (Op.HYPERELASTICITY, 3, 1): 580,
    (Op.HYPERELASTICITY, 3, 2): 10780,
    (Op.HYPERELASTICITY, 3, 3): 66715,
    (Op.HYPERELASTICITY, 3, 4): 283705,
    (Op.HYPERELASTICITY, 3, 5): 2521666,
    (Op.HYPERELASTICITY, 3, 6): 6616456,
}

nfootprint_bytes = {
    (Op.MASS, 2, 1, 524288): 16818216,
    (Op.MASS, 2, 2, 524288): 48300072,
    (Op.MASS, 2, 3, 524288): 88170536,
    (Op.MASS, 2, 4, 524288): 142721064,
    (Op.MASS, 2, 5, 524288): 211951656,
    (Op.MASS, 2, 6, 524288): 295862312,
    (Op.MASS, 2, 7, 524288): 394453032,
    (Op.MASS, 2, 8, 524288): 507723816,
    (Op.MASS, 3, 1, 196608): 4870704,
    (Op.MASS, 3, 2, 196608): 18463536,
    (Op.MASS, 3, 3, 196608): 41641008,
    (Op.MASS, 3, 4, 196608): 83053872,
    (Op.MASS, 3, 5, 196608): 148207152,
    (Op.MASS, 3, 6, 196608): 242605872,
    (Op.LAPLACE, 2, 1, 524288): 16818216,
    (Op.LAPLACE, 2, 2, 524288): 48300072,
    (Op.LAPLACE, 2, 3, 524288): 88170536,
    (Op.LAPLACE, 2, 4, 524288): 142721064,
    (Op.LAPLACE, 2, 5, 524288): 211951656,
    (Op.LAPLACE, 2, 6, 524288): 295862312,
    (Op.LAPLACE, 2, 7, 524288): 394453032,
    (Op.LAPLACE, 2, 8, 524288): 507723816,
    (Op.LAPLACE, 3, 1, 196608): 4870704,
    (Op.LAPLACE, 3, 2, 196608): 18463536,
    (Op.LAPLACE, 3, 3, 196608): 41641008,
    (Op.LAPLACE, 3, 4, 196608): 83053872,
    (Op.LAPLACE, 3, 5, 196608): 148207152,
    (Op.LAPLACE, 3, 6, 196608): 242605872,
    (Op.HELMHOLTZ, 2, 1, 524288): 16818216,
    (Op.HELMHOLTZ, 2, 2, 524288): 48300072,
    (Op.HELMHOLTZ, 2, 3, 524288): 88170536,
    (Op.HELMHOLTZ, 2, 4, 524288): 142721064,
    (Op.HELMHOLTZ, 2, 5, 524288): 211951656,
    (Op.HELMHOLTZ, 2, 6, 524288): 295862312,
    (Op.HELMHOLTZ, 2, 7, 524288): 394453032,
    (Op.HELMHOLTZ, 2, 8, 524288): 507723816,
    (Op.HELMHOLTZ, 3, 1, 196608): 4870704,
    (Op.HELMHOLTZ, 3, 2, 196608): 18463536,
    (Op.HELMHOLTZ, 3, 3, 196608): 41641008,
    (Op.HELMHOLTZ, 3, 4, 196608): 83053872,
    (Op.HELMHOLTZ, 3, 5, 196608): 148207152,
    (Op.HELMHOLTZ, 3, 6, 196608): 242605872,
    (Op.ELASTICITY, 2, 1, 524288): 23134272,
    (Op.ELASTICITY, 2, 2, 524288): 73515072,
    (Op.ELASTICITY, 2, 3, 524288): 144867392,
    (Op.ELASTICITY, 2, 4, 524288): 243482688,
    (Op.ELASTICITY, 2, 5, 524288): 369360960,
    (Op.ELASTICITY, 2, 6, 524288): 522502208,
    (Op.ELASTICITY, 2, 7, 524288): 702906432,
    (Op.ELASTICITY, 2, 8, 524288): 910573632,
    (Op.ELASTICITY, 3, 1, 196608): 6595680,
    (Op.ELASTICITY, 3, 2, 196608): 31645536,
    (Op.ELASTICITY, 3, 3, 196608): 85449312,
    (Op.ELASTICITY, 3, 4, 196608): 186094944,
    (Op.ELASTICITY, 3, 5, 196608): 348524640,
    (Op.ELASTICITY, 3, 6, 196608): 587680608,
    (Op.HYPERELASTICITY, 2, 1, 524288): 27344992,
    (Op.HYPERELASTICITY, 2, 2, 524288): 90325088,
    (Op.HYPERELASTICITY, 2, 3, 524288): 182665312,
    (Op.HYPERELASTICITY, 2, 4, 524288): 310657120,
    (Op.HYPERELASTICITY, 2, 5, 524288): 474300512,
    (Op.HYPERELASTICITY, 2, 6, 524288): 673595488,
    (Op.HYPERELASTICITY, 2, 7, 524288): 908542048,
    (Op.HYPERELASTICITY, 2, 8, 524288): 1179140192,
    (Op.HYPERELASTICITY, 3, 1, 196608): 7458184,
    (Op.HYPERELASTICITY, 3, 2, 196608): 38236552,
    (Op.HYPERELASTICITY, 3, 3, 196608): 107353480,
    (Op.HYPERELASTICITY, 3, 4, 196608): 237615496,
    (Op.HYPERELASTICITY, 3, 5, 196608): 448683400,
    (Op.HYPERELASTICITY, 3, 6, 196608): 760217992,
}


local_nbytes_accesses_per_cell = {
    (Op.MASS, 2, 1): np.nan,
    (Op.MASS, 2, 2): 576,
    (Op.MASS, 2, 3): 1920,
    (Op.MASS, 2, 4): 3840,
    (Op.MASS, 2, 5): 8400,
    (Op.MASS, 2, 6): 14784,
    (Op.MASS, 2, 7): 24192,
    (Op.MASS, 2, 8): 39600,
    (Op.MASS, 3, 1): 256,
    (Op.MASS, 3, 2): 1760,
    (Op.MASS, 3, 3): 7360,
    (Op.MASS, 3, 4): 24640,
    (Op.MASS, 3, 5): 66304,
    (Op.MASS, 3, 6): 163968,
    (Op.LAPLACE, 2, 1): np.nan,
    (Op.LAPLACE, 2, 2): 576,
    (Op.LAPLACE, 2, 3): 1920,
    (Op.LAPLACE, 2, 4): 5760,
    (Op.LAPLACE, 2, 5): 10752,
    (Op.LAPLACE, 2, 6): 22400,
    (Op.LAPLACE, 2, 7): 38016,
    (Op.LAPLACE, 2, 8): 60480,
    (Op.LAPLACE, 3, 1): np.nan,
    (Op.LAPLACE, 3, 2): 1920,
    (Op.LAPLACE, 3, 3): 10560,
    (Op.LAPLACE, 3, 4): 38640,
    (Op.LAPLACE, 3, 5): 118272,
    (Op.LAPLACE, 3, 6): 298368,
    (Op.HELMHOLTZ, 2, 1): np.nan,
    (Op.HELMHOLTZ, 2, 2): 1728,
    (Op.HELMHOLTZ, 2, 3): 5760,
    (Op.HELMHOLTZ, 2, 4): 11520,
    (Op.HELMHOLTZ, 2, 5): 25200,
    (Op.HELMHOLTZ, 2, 6): 44352,
    (Op.HELMHOLTZ, 2, 7): 72576,
    (Op.HELMHOLTZ, 2, 8): 118800,
    (Op.HELMHOLTZ, 3, 1): np.nan,
    (Op.HELMHOLTZ, 3, 2): 7040,
    (Op.HELMHOLTZ, 3, 3): 29440,
    (Op.HELMHOLTZ, 3, 4): 98560,
    (Op.HELMHOLTZ, 3, 5): 265216,
    (Op.HELMHOLTZ, 3, 6): 655872,
    (Op.ELASTICITY, 2, 1): np.nan,
    (Op.ELASTICITY, 2, 2): 576,
    (Op.ELASTICITY, 2, 3): 1920,
    (Op.ELASTICITY, 2, 4): 5760,
    (Op.ELASTICITY, 2, 5): 10752,
    (Op.ELASTICITY, 2, 6): 22400,
    (Op.ELASTICITY, 2, 7): 38016,
    (Op.ELASTICITY, 2, 8): 60480,
    (Op.ELASTICITY, 3, 1): np.nan,
    (Op.ELASTICITY, 3, 2): 1920,
    (Op.ELASTICITY, 3, 3): 10560,
    (Op.ELASTICITY, 3, 4): 38640,
    (Op.ELASTICITY, 3, 5): 118272,
    (Op.ELASTICITY, 3, 6): 298368,
    (Op.HYPERELASTICITY, 2, 1): np.nan,
    (Op.HYPERELASTICITY, 2, 2): 1152,
    (Op.HYPERELASTICITY, 2, 3): 5120,
    (Op.HYPERELASTICITY, 2, 4): 15840,
    (Op.HYPERELASTICITY, 2, 5): 36960,
    (Op.HYPERELASTICITY, 2, 6): 70784,
    (Op.HYPERELASTICITY, 2, 7): 129024,
    (Op.HYPERELASTICITY, 2, 8): 216000,
    (Op.HYPERELASTICITY, 3, 1): np.nan,
    (Op.HYPERELASTICITY, 3, 2): 5280,
    (Op.HYPERELASTICITY, 3, 3): 42240,
    (Op.HYPERELASTICITY, 3, 4): 204960,
    (Op.HYPERELASTICITY, 3, 5): 1959552,
    (Op.HYPERELASTICITY, 3, 6): 5366592,
}


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


def get_roofline_flops(op: Op, dim: int, deg: int, device: Device) -> float:
    # if device == Device.K40:
    #     fpeak = 1430  # GFlops/s
    #     beta_peak_global = 288  # GB/s
    #     beta_peak_shared = 1000  # GB/s
    # elif device == Device.TITANV:
    #     fpeak = 6144  # GFlops/s
    #     beta_peak_global = 653  # GB/s
    #     beta_peak_shared = 13800  # GB/s
    # else:
    #     raise ValueError(f"Unknown device {dev_name}.")

    raise NotImplementedError("No idea what the heck is going on over here.")
    # AI_global = arithmetic_intensity[(prob.upper(), cell_type, fem_space, deg)]
    # Fs_to_F = fs_by_f_ratios[(prob.upper(), cell_type, fem_space, deg)]
    # AI_local = 0.25 / Fs_to_F
    # return min(AI_global * beta_peak_global, AI_local * beta_peak_shared, fpeak)
