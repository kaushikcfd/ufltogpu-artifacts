from __future__ import annotations

import enum
from functools import cache


__doc__ = """
.. data:: flops_per_cell

    A mapping from ``(Operator, dim, degree)`` to number of floating point
    operations required to compute the corresponding action for one cell in the
    mesh.

.. data:: nfootprint_bytes

    A mapping from ``(Operator, dim, degree, num_cells)`` to footprint
    bytes accessed by the corresponding action operator.

.. data:: fs_by_f_ratio

    A mapping from ``(Operator, dim, degree)`` to number of ratio of
      FLOP-arithmetic
    intensity required to compute the corresponding action.
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

fs_by_f_ratios = {
    (Op.MASS, 2, 1): 0.7959183673469388,
    (Op.MASS, 2, 2): 0.8470588235294118,
    (Op.MASS, 2, 3): 0.9266409266409267,
    (Op.MASS, 2, 4): 0.959079283887468,
    (Op.MASS, 2, 5): 0.9723472668810289,
    (Op.MASS, 2, 6): 0.98,
    (Op.MASS, 2, 7): 0.9848258174823681,
    (Op.MASS, 2, 8): 0.9880726484142044,
    (Op.MASS, 3, 1): 0.6336633663366337,
    (Op.MASS, 3, 2): 0.9076175040518638,
    (Op.MASS, 3, 3): 0.9614421632448673,
    (Op.MASS, 3, 4): 0.9843073288711401,
    (Op.MASS, 3, 5): 0.9905619817790972,
    (Op.MASS, 3, 6): 0.9938342402317981,
    (Op.LAPLACE, 2, 1): 0.255319148,
    (Op.LAPLACE, 2, 2): 0.6857142857142857,
    (Op.LAPLACE, 2, 3): 0.8121827411167513,
    (Op.LAPLACE, 2, 4): 0.8775137111517367,
    (Op.LAPLACE, 2, 5): 0.9138381201044387,
    (Op.LAPLACE, 2, 6): 0.9349565217391305,
    (Op.LAPLACE, 2, 7): 0.9491525423728814,
    (Op.LAPLACE, 2, 8): 0.9591607343574372,
    (Op.LAPLACE, 3, 1): 0.1875,
    (Op.LAPLACE, 3, 2): 0.7038123167155426,
    (Op.LAPLACE, 3, 3): 0.8610968733982574,
    (Op.LAPLACE, 3, 4): 0.9195402298850575,
    (Op.LAPLACE, 3, 5): 0.9511300330630916,
    (Op.LAPLACE, 3, 6): 0.9670868533965834,
    (Op.HELMHOLTZ, 2, 1): 0.56043956,
    (Op.HELMHOLTZ, 2, 2): 0.7868852459016393,
    (Op.HELMHOLTZ, 2, 3): 0.8711433756805808,
    (Op.HELMHOLTZ, 2, 4): 0.9144482828693355,
    (Op.HELMHOLTZ, 2, 5): 0.9382562829661806,
    (Op.HELMHOLTZ, 2, 6): 0.9533846777462505,
    (Op.HELMHOLTZ, 2, 7): 0.9635799672393963,
    (Op.HELMHOLTZ, 2, 8): 0.9707703575471069,
    (Op.HELMHOLTZ, 3, 1): 0.4444444444444444,
    (Op.HELMHOLTZ, 3, 2): 0.8011444921316166,
    (Op.HELMHOLTZ, 3, 3): 0.8944793850454228,
    (Op.HELMHOLTZ, 3, 4): 0.9403420158246127,
    (Op.HELMHOLTZ, 3, 5): 0.9620902556148776,
    (Op.HELMHOLTZ, 3, 6): 0.9744833140617635,
    (Op.ELASTICITY, 2, 1): 0.258064516,
    (Op.ELASTICITY, 2, 2): 0.6428571428571429,
    (Op.ELASTICITY, 2, 3): 0.7649402390438247,
    (Op.ELASTICITY, 2, 4): 0.8359941944847605,
    (Op.ELASTICITY, 2, 5): 0.8795811518324608,
    (Op.ELASTICITY, 2, 6): 0.9074438755415518,
    (Op.ELASTICITY, 2, 7): 0.9267748079070073,
    (Op.ELASTICITY, 2, 8): 0.940696131468817,
    (Op.ELASTICITY, 3, 1): 0.236065573,
    (Op.ELASTICITY, 3, 2): 0.675739089629282,
    (Op.ELASTICITY, 3, 3): 0.8187799528876615,
    (Op.ELASTICITY, 3, 4): 0.889124106906589,
    (Op.ELASTICITY, 3, 5): 0.9287754537915783,
    (Op.ELASTICITY, 3, 6): 0.9514378979291298,
    (Op.HYPERELASTICITY, 2, 1): 0.13043782,
    (Op.HYPERELASTICITY, 2, 2): 0.5023255813953489,
    (Op.HYPERELASTICITY, 2, 3): 0.6317784563546383,
    (Op.HYPERELASTICITY, 2, 4): 0.7207943447881339,
    (Op.HYPERELASTICITY, 2, 5): 0.7834948661356875,
    (Op.HYPERELASTICITY, 2, 6): 0.8284209346632299,
    (Op.HYPERELASTICITY, 2, 7): 0.8613039656931852,
    (Op.HYPERELASTICITY, 2, 8): 0.8859010488630936,
    (Op.HYPERELASTICITY, 3, 1): 0.120401337,
    (Op.HYPERELASTICITY, 3, 2): 0.5511409200262448,
    (Op.HYPERELASTICITY, 3, 3): 0.7126191657604967,
    (Op.HYPERELASTICITY, 3, 4): 0.8128349353388286,
    (Op.HYPERELASTICITY, 3, 5): 0.8742178819024475,
    (Op.HYPERELASTICITY, 3, 6): 0.912483216120746,
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
