import csv

import numpy as np
import pandas as pd

from ufltogpu_artifacts.constants import flops_per_cell
from ufltogpu_artifacts.core import Op, get_num_cells


h200_rankings_df = pd.read_csv("h200_rankings.csv", index_col=0)
titanv_rankings_df = pd.read_csv("titanv_rankings.csv", index_col=0)
h200_times = h200_rankings_df.loc[
    [
        "Op.MASS.2D.P3",
        "Op.HYPERELASTICITY.2D.P6",
        "Op.ELASTICITY.3D.P4",
    ],
    "Rank-1":,
].to_numpy().T
titanv_times = titanv_rankings_df.loc[
    [
        "Op.MASS.2D.P3",
        "Op.HYPERELASTICITY.2D.P6",
        "Op.ELASTICITY.3D.P4",
    ],
    "Rank-1":,
].to_numpy().T
h200_ncells = np.array(
    [get_num_cells(2, 1280), get_num_cells(2, 1280), get_num_cells(3, 56)]
)
titanv_ncells = np.array(
    [get_num_cells(2, 512), get_num_cells(2, 512), get_num_cells(3, 32)]
)
flops_per_cell = np.array(
    [
        flops_per_cell[Op.MASS, 2, 3],
        flops_per_cell[Op.HYPERELASTICITY, 2, 6],
        flops_per_cell[Op.ELASTICITY, 3, 4],
    ]
)
h200_gflops = 1e-9 * flops_per_cell * h200_ncells / h200_times
titanv_gflops = 1e-9 * flops_per_cell * titanv_ncells / titanv_times

bs = [1, 3, 5, 20, 50, 70]


default_hyperparams_h200 = np.max(h200_gflops[:9], axis=0)
vary_hyperparams_h200 = np.array([
    # eta_alias_min
    [4129.6, 9392.6, 9729],
    [4127.4, 9281.5, 9742.2],
    [4118.5, 9280.5, 9732.3],
    [4126.3, 8968.8, 9729.4],
    [4115.1, 8970.6, 9737.1],
    # eta_simd_min
    [4128.6, 8999.7, 9735.7],
    [4129.5, 9003, 9729.3],
    [4128.7, 9001, 9736.9],
    [4128.1, 8998.9, 9734.9],
    [4121.4, 8971.3, 8159.2],
    # wg_max
    [4128.5, 5188.3, 10287.2],
    [4125.2, 9407.2, 10417.1],
    [4128, 8691.6, 9729.4],
    # Top-bs
    *[np.max(h200_gflops[:b], axis=0) for b in bs]
])

default_hyperparams_titanv = np.max(titanv_gflops[:9], axis=0)
vary_hyperparams_titanv = np.array([
    # eta_alias_min
    [1194.6, 3234.7, 2973.9],
    [1200, 3167.3, 2987.5],
    [1226.7, 3166.8, 2988.8],
    [1154.4, 3237.7, 3069.1],
    [1158.3, 3303.1, 3126.6],
    # eta_simd_min
    [1191.5, 3133.9, 3163],
    [1188.1, 3127.5, 3197.8],
    [1183.6, 3130.2, 3196.4],
    [1186.3, 3129, 3161.6],
    [1159, 3183.7, 3408.2],
    # wg_max
    [1140, 3284.4, 2545.6],
    [1145.3, 4057.1, 3463.3],
    [1146.8, 3084.1, 2492.7],
    # Top-bs
    *[np.max(titanv_gflops[:b], axis=0) for b in bs]
])

ratio_titanv = vary_hyperparams_titanv / default_hyperparams_titanv
ratio_h200 = vary_hyperparams_h200 / default_hyperparams_h200

workloads = [
    "mass_p3_2d",
    "hyperelasticity_p6_2d",
    "elasticity_p4_3d",
]
configurations = [
    "ETA_ALIAS_MIN=0",
    "ETA_ALIAS_MIN=0.5",
    "ETA_ALIAS_MIN=0.6",
    "ETA_ALIAS_MIN=0.7",
    "ETA_ALIAS_MIN=0.9",
    "ETA_SIMD_MIN=0",
    "ETA_SIMD_MIN=0.5",
    "ETA_SIMD_MIN=0.7",
    "ETA_SIMD_MIN=0.9",
    "ETA_SIMD_MIN=0.99",
    "WG_MAX=64",
    "WG_MAX=128",
    "WG_MAX=512",
    *[f"{b=}" for b in bs]
]

for filename, ratios in (
    ("hyperparam_ratio_titanv.csv", ratio_titanv),
    ("hyperparam_ratio_h200.csv", ratio_h200),
):
    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Configuration", *workloads])
        for configuration, values in zip(configurations, ratios, strict=True):
            writer.writerow([configuration, *values])
