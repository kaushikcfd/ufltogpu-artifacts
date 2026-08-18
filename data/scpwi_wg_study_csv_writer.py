import csv

import numpy as np

# test-cases: Mass.2D.P2, Hyperelasticity.2D.P4, Elasticity.3D.P2

pref_h200 = np.array(
    [
        2921.5,
        13579,
        5694,
    ]
)
pref_titanv = np.array(
    [
        770,
        3868,
        1694,
    ]
)
ratio_h200 = (
    np.array(
        [
            # wg=64
            [2927.5, 13477, 5843],
            # wg=96
            [2928, 11642, 5709],
            # wg=128
            [2931, 13415, 5868],
        ]
    )
    / pref_h200
)

ratio_titanv = (
    np.array(
        [
            # wg=64
            [762, 3817, 1737],
            # wg=96
            [757, 3602, 1705],
            # wg=128
            [755, 3833, 1718],
        ]
    )
    / pref_titanv
)

with open("scwpi_wg_studies.csv", "w") as fp:
    fp.write(
        ", ".join(
            [
                "W",
                "titanv_mass_2dp2",
                "titanv_hyperelasticity_2dp4",
                "titanv_elasticity_3d_p2",
                "h200_mass_2dp2",
                "h200_hyperelasticity_2dp4",
                "h200_elasticity_3d_p2",
            ]
        )
    )
    fp.write("\n")
    for wg_size, r_titanv, r_h200 in zip(
        ["64", "96", "128"], ratio_titanv, ratio_h200, strict=True
    ):
        fp.write(f"{wg_size}, ")
        fp.write(", ".join(f"{r:.2f}" for r in np.concatenate([r_titanv, r_h200])))
        fp.write("\n")
