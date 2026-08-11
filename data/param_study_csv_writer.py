import csv

import numpy as np


ptuned_h200 = np.array([4160, 9407, 9730])
psubset_h200 = np.array([
    # tile size changes
    [4129.3, 6882, 9009],
    [4129.3, 6510, 4479],
    [4129.3, 8886.8, 9115],
    # nc-nwi
    [4129.3, 5373, 6417],
    [4129.3, 5303, 9473],
    [4129, 10507, 9949],
    [4129, 9431, 10235],
    [4129, 8531, 7161],
    # t^Q
    [4129, 8534, 9730],
    [4129, 8972, 6297]
])

ptuned_titanv = np.array([1158.6, 3185.7, 3119])
psubset_titanv = np.array([
    # tile size changes
    [1159, 1814, 2444],
    [1139, 1127, 1738],
    [1140, 2169, 2375],
    # nc-nwi
    [1139, 951.7, 681.3],
    [1140, 3341.6, 2584.3],
    [1139, 3589, 2861.7],
    [1139, 4516, 3464],
    [1140, 3542, 2696],
    # t^Q
    [1229, 3545.3, 3117.9],
    [1141, 3185.6, 2071.3]
])

ratio_titanv = psubset_titanv / ptuned_titanv
ratio_h200 = psubset_h200 / ptuned_h200

workloads = [
    "Mass (P3, 2D)",
    "Hyperelasticity (P6, 2D)",
    "Elasticity (P4, 3D)",
]
configurations = [
    "Full extents",
    "Uniform tile size 4",
    "Uniform tile size 8",
    "(Nc, Nwi) = (32, 1)",
    "(Nc, Nwi) = (16, 4)",
    "(Nc, Nwi) = (8, 8)",
    "(Nc, Nwi) = (16, 8)",
    "(Nc, Nwi) = (16, 16)",
    "Tq=Q",
    "Tq=ceil(Q/2)",
]

for filename, ratios in (
    ("ratio_titanv.csv", ratio_titanv),
    ("ratio_h200.csv", ratio_h200),
):
    with open(filename, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Configuration", *workloads])
        for configuration, values in zip(configurations, ratios, strict=True):
            writer.writerow([configuration, *values])
