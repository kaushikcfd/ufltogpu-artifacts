## UFL-to-GPU Reproducer toolkit

This repository contains the necessary code to reproduce the results presented
in the paper "[UFL-to-GPU: Near the Roofline](#)". Follow the steps below to
reproduce the reported results.

1. Install `firedrake` project as follows, noting that
   several sub-packages use branches that deviate from
   'mainline' Firedrake. Note that the bulk of the work
   described in the paper exists on those branches.
    ```console
    > curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/gpu/scripts/firedrake-install

    > python3 firedrake-install --venv-name firedrake_gpu --minimal-petsc --no-package-manager --package-branch firedrake gpu --package-branch tsfc gpu --package-branch pyop2 auto_tiling --cuda --mpicc=/usr/bin/mpicc.mpich --mpicxx=/usr/bin/mpicxx.mpich --mpif90=/usr/bin/mpif90.mpich --mpiexec=/usr/bin/mpiexec.mpich
    ```
4. Clone this repository and build it.
   ```console
   > git clone https://github.com/kaushikcfd/ufltogpu-artifacts

   > cd ufltogpu-artifacts

   > source ${FIREDRAKE_VENV}/bin/activate
   > pip install -e .
   ```

5. Run the experiments
   ```console
   > cd ufltogpu-artifacts
   > python timings_recorder.py \
        --op Mass Laplace Helmholtz Elasticity Helmholtz Hyperelasticity \
        --dim 2 \
        --p_range 1 8
   > python timings_recorder.py \
        --op Mass Laplace Helmholtz Elasticity Helmholtz Hyperelasticity \
        --dim 3 \
        --p_range 1 6
   ```

### HOWTO: Run a subset of the benchmark suite

```console
> python timings_recorder.py -h
usage: timings_recorder.py [-h] --op [{Mass,Laplace,Helmholtz,Elasticity,Hyperelasticity} ...] [--db_path DB_PATH] --dim [{2,3} ...] --p_range X Y

Utility to obtain throughput of Kulkarni, Kloeckner's auto-tiling strategy and optionally write them to a SQL-database.

options:
  -h, --help            show this help message and exit
  --op [{Mass,Laplace,Helmholtz,Elasticity,Hyperelasticity} ...]
  --db_path DB_PATH     Path to the SQL database which is to be updated. A new one will be created if none exists at the provided path.
  --dim [{2,3} ...]     Toplogical dimension on which the function spaces are to be defined.
  --p_range X Y         Operators corresponding to polynomial degrees {X, X+1, ..., Y} are evaluated.
```

### HOWTO: Verify the constants used in Roofline calculations

```console
> $ python constants_verifier.py -h
usage: constants_verifier.py [-h] [--no-verify-flops_per_cell] [--no-verify-nfootprint_bytes] [--no-verify-local_bytes_accesses_per_cell]

Utility to verify the tabulated data used in Roofline computation.

options:
  -h, --help            show this help message and exit
  --no-verify-flops_per_cell
                        Do not verify the #FLOPS per cell for the action operators
  --no-verify-nfootprint_bytes
                        Do not verify the footprint memory accesses for the action operators.
  --no-verify-local_bytes_accesses_per_cell
                        Do not verify the local memory accesses per cell for the action operators.
```

### Software Contributions

As part of this work, we have proposed the following patches to enable GPU
support in Firedrake:
[Firedrake#1605](https://github.com/firedrakeproject/firedrake/pull/1605) and
[PyOP2#574](https://github.com/OP2/PyOP2/pull/574). Additionally, our [patch for
the auto-tiling
algorithm](https://github.com/kaushikcfd/PyOP2/compare/gpu...kaushikcfd:PyOP2:auto_tiling)
builds upon the proposed changes to PyOP2.

### Additional Information
Please refer to the paper for more details on the experiments. If you encounter
any issues or have questions, feel free to open an issue in this repository.
