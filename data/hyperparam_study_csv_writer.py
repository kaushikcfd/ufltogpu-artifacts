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
    [4128.7, 8684.9, 6506.3],
    [4130.7, 8689.2, 6509.2],
    [4125.1, 8690.5, 6508.3],
    [4127.5, 8683.9, 6510],
    [4123.6, 8688.8, 6508.7],
    [4124.5, 8904.5, 6510],
    # eta_simd_min
    [4124.2, 9015.8, 9731.6],
    [4125, 8999.6, 9728.2],
    [4124.7, 9011.1, 9732.3],
    [4124.2, 8687.4, 9730.3],
    [4123.9, 8691.4, 9731.4],
    [4125.5, 8683.3, 6507.9],
    # wg_max
    [4128.5, 5188.3, 10287.2],
    [4125.2, 9407.2, 10417.1],
    [4126.7, 9403.3, 9736.2],
    [4128, 8691.6, 9729.4],
    # Top-bs
    *[np.max(h200_gflops[:b], axis=0) for b in bs]
])

default_hyperparams_titanv = np.max(titanv_gflops[:9], axis=0)
vary_hyperparams_titanv = np.array([
    # eta_alias_min
    [1153.4, 3080.7, 2399.9],
    [1156.8, 3078.7, 2414.5],
    [1147.4, 3086.5, 2413.8],
    [1144.8, 3087.3, 2437.9],
    [1139.5, 3082.9, 2445.6],
    [1156.6, 3335.2, 2450.6],
    # eta_simd_min
    [1191.5, 3129, 3186],
    [1181, 3127.8, 3164.9],
    [1189.1, 3129.6, 3169.3],
    [1185.2, 3136.6, 3197.9],
    [1156.7, 3084.1, 2454.9],
    [1145.2, 3085.3, 2441.4],
    # wg_max
    [1140, 3284.4, 2545.6],
    [1145.3, 4057.1, 3463.3],
    [1160.2, 3183, 3090.8],
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
    "ETA_ALIAS_MIN=0.8",
    "ETA_ALIAS_MIN=0.9",
    "ETA_SIMD_MIN=0",
    "ETA_SIMD_MIN=0.5",
    "ETA_SIMD_MIN=0.7",
    "ETA_SIMD_MIN=0.9",
    "ETA_SIMD_MIN=0.97",
    "ETA_SIMD_MIN=0.99",
    "WG_MAX=64",
    "WG_MAX=128",
    "WG_MAX=256",
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
