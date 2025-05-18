from sklearn.tree import plot_tree
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import numpy as np
import os
from algorithm.genetic_algorithm import genetic_algorithm
from algorithm.topsis_algorithm import topsis_algorithm
from algorithm.decession_tree_algorithm import decision_tree_algorithm
from algorithm.c45_algorithm import c45_algorithm

st.set_page_config(page_title="Dashboard Distribusi Beasiswa", layout="wide")


@st.cache_data
def generate_students(total_students):
    students = []
    id_counter = 1
    classes = [10, 11, 12]
    students_per_class = total_students // len(classes)

    for kelas in classes:
        for i in range(students_per_class):
            students.append(
                {
                    "id": id_counter,
                    "nama": f"Siswa_{kelas}_{i+1}",
                    "kelas": kelas,
                    "nilai": random.randint(70, 100),
                    "pendapatan": random.randint(1500000, 4000000),
                }
            )
            id_counter += 1
    return students


# Sidebar
st.sidebar.title("Pengaturan Seleksi Beasiswa")
algorithm = st.sidebar.selectbox(
    "Pilih Algoritma", ["Genetika", "TOPSIS", "Decision Tree", "C4.5"]
)
SCHOLARSHIPS = st.sidebar.slider("Jumlah Beasiswa", 10, 300, 100, 10)
number_of_students = st.sidebar.number_input("Jumlah Siswa", 100, 3000, 900, step=100)
students_data = generate_students(number_of_students)

st.sidebar.title("Bobot Kriteria")
nilai_weight = st.sidebar.slider("Bobot Nilai (0.0 - 1.0)", 0.0, 1.0, 0.5, 0.05)
pendapatan_weight = 1.0 - nilai_weight
st.sidebar.markdown(f"**Bobot Pendapatan Otomatis:** {pendapatan_weight:.2f}")

if algorithm == "Genetika":
    st.sidebar.title("Parameter Algoritma Genetika")
    number_of_generation = st.sidebar.slider("Jumlah Generasi", 10, 200, 50, 10)
    population_size = st.sidebar.slider("Ukuran Populasi", 5, 50, 10, 5)

# Eksekusi
st.title("🎓 Dashboard Distribusi Beasiswa")
run_button = st.button("🔁 Jalankan Seleksi Beasiswa")

if run_button:
    if algorithm == "Genetika":
        selected_students, log = genetic_algorithm(
            students_data,
            SCHOLARSHIPS,
            nilai_weight,
            pendapatan_weight,
            number_of_generation,
            population_size,
        )
        df_result = pd.DataFrame(selected_students)
        st.success(f"📋 Jumlah siswa terpilih (Genetika): {len(df_result)}")
    elif algorithm == "TOPSIS":
        result = topsis_algorithm(
            students_data, SCHOLARSHIPS, nilai_weight, pendapatan_weight
        )
        df_result = pd.DataFrame(result)
        st.success(f"📋 Jumlah siswa terpilih (TOPSIS): {len(df_result)}")
    elif algorithm == "Decision Tree":
        selected, model, scaler, full_df = decision_tree_algorithm(
            students_data, SCHOLARSHIPS, nilai_weight, pendapatan_weight
        )
        df_result = selected
        st.success(f"📋 Jumlah siswa terpilih (Decision Tree): {len(df_result)}")
    elif algorithm == "C4.5":
        selected, model, full_df = c45_algorithm(
            students_data, SCHOLARSHIPS, nilai_weight, pendapatan_weight
        )
        df_result = selected
        st.success(f"📋 Jumlah siswa terpilih (C4.5): {len(df_result)}")

    distribusi = df_result["kelas"].value_counts().sort_index()
    st.subheader("Distribusi Penerima Beasiswa per Kelas")
    st.bar_chart(distribusi)

    st.subheader("Data Penerima Beasiswa")
    st.dataframe(df_result.sort_values(by="kelas"))

    if algorithm == "Genetika":
        fitness_log = [entry[1] for entry in log]
        st.subheader("📈 Grafik Perkembangan Fitness")
        fig, ax = plt.subplots()
        ax.plot(fitness_log, marker="o")
        ax.set_xlabel("Generasi")
        ax.set_ylabel("Fitness")
        ax.set_title("Perkembangan Fitness Tiap Generasi")
        st.pyplot(fig)
    elif algorithm == "TOPSIS":
        st.subheader("📈 Grafik TOPSIS Score per Kelas")
        fig, ax = plt.subplots(figsize=(10, 6))
        for kelas in [10, 11, 12]:
            data = df_result[df_result["kelas"] == kelas].sort_values("rank")
            ax.plot(
                range(1, len(data) + 1),
                data["topsis_score"].values,
                marker="o",
                label=f"Kelas {kelas}",
            )
        ax.set_title("Skor TOPSIS Penerima Beasiswa")
        ax.set_xlabel("Peringkat")
        ax.set_ylabel("Skor")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    elif algorithm == "Decision Tree" or algorithm == "C4.5":
        st.subheader("📊 Visualisasi Pohon Keputusan")
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_tree(
            model,
            feature_names=["nilai", "pendapatan"],
            class_names=["Tidak Layak", "Layak"],
            filled=True,
            ax=ax,
        )
        st.pyplot(fig)

    # Simpan hasil
    os.makedirs("./output", exist_ok=True)
    csv_path = "./output/penerima_beasiswa.csv"
    df_result.to_csv(csv_path, index=False)
    st.download_button(
        "📥 Download CSV",
        data=df_result.to_csv(index=False),
        file_name="penerima_beasiswa.csv",
        mime="text/csv",
    )
