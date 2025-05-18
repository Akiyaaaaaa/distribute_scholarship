import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random


def c45_algorithm(students, scholarship_quota, nilai_weight, pendapatan_weight):
    df = pd.DataFrame(students)

    df["skor"] = (df["nilai"] * nilai_weight) + (
        (1 - df["pendapatan"] / df["pendapatan"].max()) * pendapatan_weight * 100
    )

    threshold = df["skor"].quantile(1 - (scholarship_quota / len(df)))
    df["label"] = df["skor"].apply(
        lambda x: "Layak" if x >= threshold else "Tidak Layak"
    )

    X = df[["nilai", "pendapatan"]]
    y = df["label"]
    model = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    model.fit(X, y)

    df["prediksi"] = model.predict(X)

    beasiswa_per_kelas = scholarship_quota // 3
    sisa = scholarship_quota % 3
    kelas_list = [10, 11, 12]
    random.shuffle(kelas_list)

    selected_df = pd.DataFrame()

    for kelas in kelas_list:
        jml = beasiswa_per_kelas + (1 if sisa > 0 else 0)
        if sisa > 0:
            sisa -= 1

        subset = df[df["kelas"] == kelas]
        layak_df = (
            subset[subset["prediksi"] == "Layak"]
            .sort_values("skor", ascending=False)
            .head(jml)
        )

        if len(layak_df) < jml:
            kurang = jml - len(layak_df)
            tidak_layak_df = (
                subset[~subset.index.isin(layak_df.index)]
                .sort_values("skor", ascending=False)
                .head(kurang)
            )
            combined = pd.concat([layak_df, tidak_layak_df])
        else:
            combined = layak_df

        selected_df = pd.concat([selected_df, combined])

    selected_df = selected_df.sort_values(["kelas", "skor"], ascending=[True, False])

    return selected_df, model, df
