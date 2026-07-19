from __future__ import annotations  # noqa: I001

import argparse
import numpy as np

import firedrake as fd
import pyop2.op2 as op2
from tsfc import compile_form

from ufltogpu_artifacts.constants import (
    flops_per_cell,
    local_nbytes_accesses_per_cell,
    nfootprint_bytes_per_cell,
    num_transform_candidates
)
from ufltogpu_artifacts.core import (
    Op,
    get_nel1d_for_reported_data,
    get_num_cells,
)
from ufltogpu_artifacts.weak_forms import get_bilinear_form


def verify_flops_per_cell() -> None:
    """
    Verifies the correctness of entries in :data:`ufltogpu_artifacts.flops_per_cell`.
    """
    for (op, dim, p), ref_flops in flops_per_cell.items():
        A, V = get_bilinear_form(1, dim, op, p)
        x = fd.Function(V)
        expr = fd.action(A, x)
        (knl,) = compile_form(expr)
        empirical_flops = knl.flop_count
        if np.testing.assert_array_equal(empirical_flops, ref_flops):
            raise RuntimeError(
                f"For {op=}, {dim=}, {p=}: "
                f"Expected = {ref_flops}, Measured = {empirical_flops}"
            )

    print("Verified flops_per_cell for all reference values. ✨ 🦾 ✨")


def get_p_lo_hi(dim: int) -> tuple[int, int]:
    if dim == 2:
        return (1, 8)
    elif dim == 3:
        return (1, 6)
    else:
        raise NotImplementedError(f"{dim=}")


def print_flops() -> None:
    """
    Prints the :data:`ufltogpu_artifacts.flops_per_cell` as a python dict.
    """
    import black

    BLACK_MODE = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=84)

    code = ""

    code += "flops_per_cell = {\n"

    for op in [
        Op.MASS,
        Op.LAPLACE,
        Op.HELMHOLTZ,
        Op.ELASTICITY,
        Op.HYPERELASTICITY,
    ]:
        for dim in [2, 3]:
            p_lo, p_hi = get_p_lo_hi(dim)
            for p in range(p_lo, p_hi + 1):
                A, V = get_bilinear_form(1, dim, op, p)
                x = fd.Function(V)
                expr = fd.action(A, x)
                (knl,) = compile_form(expr)
                code += f"({op}, {dim}, {p}): {int(knl.flop_count)},\n"
        code += "\n"

    code += "}"
    print(black.format_file_contents(code, fast=False, mode=BLACK_MODE))


def _get_parloop_nfootprint_bytes(p: op2.AbstractParloop) -> int:
    from more_itertools import zip_equal

    seen_maps: set[op2.Map] = set()
    nbytes: list[int] = []
    for arg, access in zip_equal(p.arguments, p.accesses):
        # {{{ Arg's nbytes

        if access == op2.READ:
            nbytes.append(arg.data.nbytes)
        elif access == op2.INC:
            # Incrementing involves both reading and writing.
            nbytes.append(2 * arg.data.nbytes)
        else:
            raise NotImplementedError(f"{access=}")

        # }}}

        # {{{ Arg's L2G maps' nbytes

        for map_ in arg.maps:
            if map_ not in seen_maps:
                seen_maps.add(map_)
                nbytes.append(map_._values.nbytes)
        # }}}

    assert len(nbytes) == len(p.arglist)
    return sum(nbytes)


def print_nfootprint_bytes() -> None:
    """
    Prints the :data:`ufltogpu_artifacts.nfootprint_bytes_per_cell` as a
    python dict.
    """
    import black

    BLACK_MODE = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=84)

    code = ""

    code += "nfootprint_bytes_per_cell = {\n"

    for op in [
        Op.MASS,
        Op.LAPLACE,
        Op.HELMHOLTZ,
        Op.ELASTICITY,
        Op.HYPERELASTICITY,
    ]:
        for dim in [2, 3]:
            p_lo, p_hi = get_p_lo_hi(dim)
            for p in range(p_lo, p_hi + 1):
                nel_1d = get_nel1d_for_reported_data(dim)
                num_cells = get_num_cells(dim, nel_1d)
                A, V = get_bilinear_form(nel_1d, dim, op, p)
                x = fd.Function(V)
                y = fd.Function(V)
                assembler = fd.get_assembler(fd.action(A, x))
                (parloop,) = assembler.parloops(y)
                nbytes = _get_parloop_nfootprint_bytes(parloop)
                code += f"({op}, {dim}, {p}): {nbytes / num_cells},\n"
        code += "\n"

    code += "}"
    print(black.format_file_contents(code, fast=False, mode=BLACK_MODE))


def verify_nfootprint_bytes() -> None:
    """
    Verifies the correctness of entries in
    :data:`ufltogpu_artifacts.nfootprint_bytes_per_cell`.
    """
    for (op, dim, p), ref_nbytes_per_cell in nfootprint_bytes_per_cell.items():
        nel_1d = get_nel1d_for_reported_data(dim)
        num_cells = get_num_cells(dim, nel_1d)
        A, V = get_bilinear_form(nel_1d, dim, op, p)
        x = fd.Function(V)
        y = fd.Function(V)
        assembler = fd.get_assembler(fd.action(A, x))
        (parloop,) = assembler.parloops(y)
        empirical_nbytes_per_cell = _get_parloop_nfootprint_bytes(parloop) / num_cells
        if np.testing.assert_array_equal(empirical_nbytes_per_cell, ref_nbytes_per_cell):
            raise RuntimeError(
                f"For {op=}, {dim=}, {p=}: "
                f"Expected = {ref_nbytes_per_cell}, "
                f"Measured = {empirical_nbytes_per_cell}"
            )

    print("Verified nfootprint_bytes_per_cell for all reference values. ✨ 🦾 ✨")


def _get_roofline_local_nbytes_per_cell_for_fem_kernel(p: op2.AbstractParloop) -> float:
    from pyop2.codegen.rep2loopy import generate
    from pyop2.transforms.auto_tiling import (
        MetadataMismatchError,
        _preprocess_tunit_for_autotiling,
        inference_which_should_ideally_be_done_by_passing_metadata,
    )
    from pyop2.transforms.gpu_utils import preprocess_t_unit_for_gpu

    t_unit = generate(p.global_kernel.builder)
    t_unit = preprocess_t_unit_for_gpu(t_unit)
    t_unit = _preprocess_tunit_for_autotiling(t_unit)
    kernel = t_unit.default_entrypoint

    try:
        _, metadata = inference_which_should_ideally_be_done_by_passing_metadata(kernel)
    except MetadataMismatchError:
        return np.nan

    return sum(
        kernel.temporary_variables[deriv_mat].nbytes
        for mv_stage in metadata.matvec_stage_descrs
        for deriv_mat in mv_stage.deriv_matrices
    )


def print_roofline_local_nbytes_reads() -> None:
    """
    Prints the :data:`ufltogpu_artifacts.ai_shared` as a python dict.
    """
    import black

    BLACK_MODE = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=84)

    code = ""

    code += "local_nbytes_accesses_per_cell = {\n"

    for op in [
        Op.MASS,
        Op.LAPLACE,
        Op.HELMHOLTZ,
        Op.ELASTICITY,
        Op.HYPERELASTICITY,
    ]:
        for dim in [2, 3]:
            p_lo, p_hi = get_p_lo_hi(dim)
            for p in range(p_lo, p_hi + 1):
                A, V = get_bilinear_form(1, dim, op, p)
                x = fd.Function(V)
                y = fd.Function(V)
                assembler = fd.get_assembler(fd.action(A, x), tensor=y)
                (parloop,) = assembler.parloops(y)
                local_nbytes = _get_roofline_local_nbytes_per_cell_for_fem_kernel(
                    parloop
                )
                code += f"({op}, {dim}, {p}): {local_nbytes},\n"
                print(f"Done with {op=}, {dim=}, {p=}")
        code += "\n"

    code += "}"
    print(black.format_file_contents(code, fast=False, mode=BLACK_MODE))


def verify_local_nbytes_accesses_per_cell() -> None:
    """
    Verifies the correctness of entries in
    :data:`ufltogpu_artifacts.local_nbytes_accesses_per_cell`.
    """
    for (op, dim, p), ref_nbytes in local_nbytes_accesses_per_cell.items():
        A, V = get_bilinear_form(1, dim, op, p)
        x = fd.Function(V)
        y = fd.Function(V)
        assembler = fd.get_assembler(fd.action(A, x), tensor=y)
        (parloop,) = assembler.parloops(y)
        empirical_nbytes = _get_roofline_local_nbytes_per_cell_for_fem_kernel(parloop)
        if np.testing.assert_array_equal(empirical_nbytes, ref_nbytes):
            raise RuntimeError(
                f"For {op=}, {dim=}, {p=}: "
                f"Expected = {ref_nbytes}, Measured = {empirical_nbytes}"
            )

    print("Verified local_nbytes_accesses_per_cell for all reference values. ✨ 🦾 ✨")


def _get_number_of_transform_candidates(p: op2.AbstractParloop) -> int:
    from pyop2.codegen.rep2loopy import generate
    from pyop2.transforms.auto_tiling import (
        MetadataMismatchError,
        ParametricTilingCandidateGenerator,
        _preprocess_tunit_for_autotiling,
    )
    from pyop2.transforms.gpu_utils import preprocess_t_unit_for_gpu

    t_unit = generate(p.global_kernel.builder)
    t_unit = preprocess_t_unit_for_gpu(t_unit)
    t_unit = _preprocess_tunit_for_autotiling(t_unit)

    try:
        candidate_generator = ParametricTilingCandidateGenerator(t_unit, 10)
        return len(candidate_generator._get_all_configurations()) + 1
    except MetadataMismatchError:
        # Only SWIPC possible.
        return 1


def print_number_of_transform_candidates() -> None:
    """
    Prints the :data:`ufltogpu_artifacts.num_transform_candidates` as a python dict.
    """
    import black

    BLACK_MODE = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=84)

    code = ""

    code += "num_transform_candidates = {\n"

    for op in [
        Op.MASS,
        Op.LAPLACE,
        Op.HELMHOLTZ,
        Op.ELASTICITY,
        Op.HYPERELASTICITY,
    ]:
        for dim in [2, 3]:
            p_lo, p_hi = get_p_lo_hi(dim)
            for p in range(p_lo, p_hi + 1):
                A, V = get_bilinear_form(1, dim, op, p)
                x = fd.Function(V)
                y = fd.Function(V)
                assembler = fd.get_assembler(fd.action(A, x), tensor=y)
                (parloop,) = assembler.parloops(y)
                num_candidates = _get_number_of_transform_candidates(parloop)
                code += f"({op}, {dim}, {p}): {num_candidates},\n"
                print(f"Done with {op=}, {dim=}, {p=}")
        code += "\n"

    code += "}"
    print(black.format_file_contents(code, fast=False, mode=BLACK_MODE))


def verify_num_transform_candidates() -> None:
    """
    Verifies the correctness of entries in
    :data:`ufltogpu_artifacts.num_transform_candidates`.
    """
    for (op, dim, p), ref_ncandidates in num_transform_candidates.items():
        A, V = get_bilinear_form(1, dim, op, p)
        x = fd.Function(V)
        y = fd.Function(V)
        assembler = fd.get_assembler(fd.action(A, x), tensor=y)
        (parloop,) = assembler.parloops(y)
        empirical_ncanidates = _get_number_of_transform_candidates(parloop)
        if np.testing.assert_array_equal(empirical_ncanidates, ref_ncandidates):
            raise RuntimeError(
                f"For {op=}, {dim=}, {p=}: "
                f"Expected = {ref_ncandidates}, Measured = {empirical_ncanidates}"
            )

    print("Verified num_transform_candidates for all reference values. ✨ 🦾 ✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to verify the tabulated data used in Roofline computation."
    )
    parser.add_argument(
        "--no-verify-flops_per_cell",
        action="store_true",
        help="Do not verify the #FLOPS per cell for the action operators",
    )
    parser.add_argument(
        "--no-verify-nfootprint_bytes",
        action="store_true",
        help="Do not verify the footprint memory accesses for the action operators.",
    )
    parser.add_argument(
        "--no-verify-local_bytes_accesses_per_cell",
        action="store_true",
        help=(
            "Do not verify the local memory accesses per cell for the action operators."
        ),
    )

    parser.add_argument(
        "--no-verify-num-transform-candidates",
        action="store_true",
        help=(
            "Do not verify the number of transform candidates for the action operators."
        ),
    )

    args = parser.parse_args()

    if not args.no_verify_flops_per_cell:
        verify_flops_per_cell()
    if not args.no_verify_nfootprint_bytes:
        verify_nfootprint_bytes()
    if not args.no_verify_local_bytes_accesses_per_cell:
        verify_local_nbytes_accesses_per_cell()
    if not args.no_verify_num_transform_candidates:
        verify_num_transform_candidates()
