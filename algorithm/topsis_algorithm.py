import pandas as pd
import numpy as np
import random


def normalize(df, benefit_cols, cost_cols):
    norm = df.copy()
    for col in benefit_cols:
        norm[col] = df[col] / np.sqrt((df[col] ** 2).sum())
    for col in cost_cols:
        norm[col] = (1 / df[col]) / np.sqrt(((1 / df[col]) ** 2).sum())
    return norm


def topsis_algorithm(data, SCHOLARSHIPS, nilai_weight, pendapatan_weight):
    df = pd.DataFrame(data)
    weights = {"nilai": nilai_weight, "pendapatan": pendapatan_weight}

    norm_df = normalize(df, benefit_cols=["nilai"], cost_cols=["pendapatan"])
    for k in weights:
        norm_df[k] = norm_df[k] * weights[k]

    ideal_best = norm_df[weights.keys()].max()
    ideal_worst = norm_df[weights.keys()].min()

    dist_best = np.sqrt(((norm_df[weights.keys()] - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((norm_df[weights.keys()] - ideal_worst) ** 2).sum(axis=1))

    df["topsis_score"] = dist_worst / (dist_best + dist_worst)
    df["rank"] = df["topsis_score"].rank(ascending=False)

    beasiswa_per_kelas = SCHOLARSHIPS // 3
    sisa = SCHOLARSHIPS % 3
    kelas_list = [10, 11, 12]
    random.shuffle(kelas_list)

    selected = pd.DataFrame()

    for kelas in kelas_list:
        kelas_df = df[df["kelas"] == kelas].sort_values("topsis_score", ascending=False)
        jumlah = beasiswa_per_kelas

        if sisa > 0 and len(kelas_df) > jumlah:
            jumlah += 1
            sisa -= 1

        selected = pd.concat([selected, kelas_df.head(jumlah)])

    if sisa > 0:
        already_selected_ids = set(selected["id"])
        remaining_candidates = df[~df["id"].isin(already_selected_ids)]
        extra = remaining_candidates.sort_values("topsis_score", ascending=False).head(
            sisa
        )
        selected = pd.concat([selected, extra])

    return selected.reset_index(drop=True)
