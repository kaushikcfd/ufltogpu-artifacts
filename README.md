## UFL-to-GPU Reproducer toolkit

This repository contains the necessary code to reproduce the results presented
in the paper "[UFL-to-GPU: Near the Roofline](#)". Follow the steps below to
reproduce the reported results.

1. Install `firedrake` project.
    ```console
    > curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/gpu/scripts/firedrake-install

    > python3 firedrake-install --venv-name firedrake_gpu --minimal-petsc --no-package-manager --package-branch firedrake gpu --package-branch tsfc gpu --package-branch pyop2 auto_tiling --cuda --mpicc=/usr/bin/mpicc.mpich --mpicxx=/usr/bin/mpicxx.mpich --mpif90=/usr/bin/mpif90.mpich --mpiexec=/usr/bin/mpiexec.mpich
    ```
2. Clone this repository and build it.
   ```console

   > git clone https://github.com/kaushikcfd/ufltogpu-artifacts

   > cd ufltogpu-artifacts

   > source ${FIREDRAKE_VENV}/bin/activate
   > pip install -e .
   ```

3. Run the experiments
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

### Additional Information
Please refer to the paper for more details on the experiments. If you encounter
any issues or have questions, feel free to open an issue in this repository.