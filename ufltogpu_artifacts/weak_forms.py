from __future__ import annotations

import ufl
from firedrake import (
    Function,
    FunctionSpace,
    MeshGeometry,
    TestFunction,
    TrialFunction,
    UnitCubeMesh,
    UnitSquareMesh,
    VectorFunctionSpace,
    dot,
    dx,
    grad,
    inner,
    transpose,
)

from ufltogpu_artifacts.core import Op


def mass(*, mesh: MeshGeometry, p: int) -> tuple[ufl.Form, ufl.FunctionSpace]:
    V = FunctionSpace(mesh, "CG", p)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = v * u * dx
    return A, V


def poisson(*, mesh: MeshGeometry, p: int) -> tuple[ufl.Form, ufl.FunctionSpace]:
    V = FunctionSpace(mesh, "CG", p)
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(grad(v), grad(u)) * dx, V


def helmholtz(*, mesh: MeshGeometry, p: int) -> tuple[ufl.Form, ufl.FunctionSpace]:
    V = FunctionSpace(mesh, "CG", p)
    u = TrialFunction(V)
    v = TestFunction(V)
    return dot(grad(v), grad(u)) - 1.0 * v * u, V


def elasticity(*, mesh: MeshGeometry, p: int) -> tuple[ufl.Form, ufl.FunctionSpace]:
    V = VectorFunctionSpace(mesh, "CG", p)
    u = TrialFunction(V)
    v = TestFunction(V)

    def eps(v):
        return grad(v) + transpose(grad(v))

    return 0.25 * inner(eps(v), eps(u)) * dx, V


def hyperelasticity(
    *, mesh: MeshGeometry, p: int
) -> tuple[ufl.Form, ufl.FunctionSpace]:
    from firedrake import Constant, Identity, derivative, diff, tr, variable

    V = VectorFunctionSpace(mesh, "CG", p)
    v = TestFunction(V)
    du = TrialFunction(V)  # Incremental displacement
    u = Function(V)  # Displacement from previous iteration
    B = Function(V)  # Body force per unit mass
    # Kinematics
    iden = Identity(mesh.geometric_dimension())
    F = iden + grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor
    E = (C - iden) / 2  # Euler-Lagrange strain tensor
    E = variable(E)
    # Material constants
    mu = Constant(1.0)  # Lame's constants
    lmbda = Constant(0.001)
    # Strain energy function (material model)
    psi = lmbda / 2 * (tr(E) ** 2) + mu * tr(E * E)
    S = diff(psi, E)  # Second Piola-Kirchhoff stress tensor
    PK = F * S  # First Piola-Kirchoff stress tensor
    # Variational problem
    return derivative((inner(PK, grad(v)) - inner(B, v)) * dx, u, du), V


def get_bilinear_form(
    nel_1d: int, dim: int, op: Op, p: int
) -> tuple[ufl.Form, ufl.FunctionSpace]:
    """
    Returns a tuple ``(A, V)``, where,

        - ``A`` is the bilinear form corresponding to operator ``op`` with a
           polynomial discretization of degree *p*.
        - ``V`` is the function space used in building the bilinear-form ``a``.
    """
    if dim == 2:
        mesh = UnitSquareMesh(nel_1d, nel_1d)
    elif dim == 3:
        mesh = UnitCubeMesh(nel_1d, nel_1d, nel_1d)
    else:
        raise ValueError(f"Invalid {dim=}.")

    if op == Op.MASS:
        return mass(mesh=mesh, p=p)
    elif op == Op.LAPLACE:
        return poisson(mesh=mesh, p=p)
    elif op == Op.HELMHOLTZ:
        return helmholtz(mesh=mesh, p=p)
    elif op == Op.ELASTICITY:
        return elasticity(mesh=mesh, p=p)
    elif op == Op.HYPERELASTICITY:
        return hyperelasticity(mesh=mesh, p=p)
    else:
        raise NotImplementedError(f"{op=}")