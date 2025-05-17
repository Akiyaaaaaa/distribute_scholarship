import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import random


def decision_tree_algorithm(
    students_data, scholarships, nilai_weight=0.5, pendapatan_weight=0.5
):
    df = pd.DataFrame(students_data)

    df["score"] = (df["nilai"] * nilai_weight) + (
        (1 / df["pendapatan"]) * pendapatan_weight * 1e6
    )

    scaler = StandardScaler()
    features = scaler.fit_transform(df[["nilai", "pendapatan"]])

    df["label"] = 0
    selected = pd.DataFrame()
    beasiswa_per_kelas = scholarships // 3
    sisa = scholarships % 3

    kelas_list = [10, 11, 12]
    random.shuffle(kelas_list)

    for kelas in kelas_list:
        jumlah = beasiswa_per_kelas + (1 if sisa > 0 else 0)
        sisa -= 1 if sisa > 0 else 0

        top_kelas = df[df["kelas"] == kelas].nlargest(jumlah, "score")
        df.loc[top_kelas.index, "label"] = 1
        selected = pd.concat([selected, top_kelas])

    model = DecisionTreeClassifier(random_state=0, max_depth=3)
    model.fit(features, df["label"])

    df["predicted"] = model.predict(features)

    selected = df[df["label"] == 1]

    return selected, model, scaler, df
