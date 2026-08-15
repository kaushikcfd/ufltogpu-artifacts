import pandas as pd


def main() -> None:
    df = pd.read_csv("top70_rankings.csv", index_col=0).loc[:, "Rank-1":]
    regret_table: list[list[float]] = []
    top1 = df.loc[:, "Rank-1":"Rank-1"].min(axis=1)
    top5 = df.loc[:, "Rank-1":"Rank-5"].min(axis=1)
    top10 = df.loc[:, "Rank-1":"Rank-10"].min(axis=1)
    top20 = df.loc[:, "Rank-1":"Rank-20"].min(axis=1)
    top70 = df.loc[:, "Rank-1":"Rank-70"].min(axis=1)

    column_headers = [
        "q_titanv_top1",
        "q_titanv_top5",
        "q_titanv_top10",
        "q_titanv_top20",
        "q_h200_top1",
        "q_h200_top5",
        "q_h200_top10",
        "q_h200_top20",
        "q_both_top1",
        "q_both_top5",
        "q_both_top10",
        "q_both_top20",
    ]
    row_headers = [
        "all",
        "twod",
        "threed",
        "loworder",
        "highorder",
        "mass",
        "laplace",
        "helmholtz",
        "elasticity",
        "hyperlelasiticity",
    ]


if __name__ == "__main__":
    main()
