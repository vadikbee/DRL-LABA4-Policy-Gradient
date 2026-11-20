from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Лабораторная работа 4", layout="wide")
st.title("Лабораторная работа 4 — методы градиента политики")
st.write("CartPole-v1: REINFORCE, базовая линия, VPG и TRPO")
data = json.loads((Path(__file__).parent / "results" / "metrics.json").read_text(encoding="utf-8"))
selected = st.sidebar.multiselect("Алгоритмы", list(data), default=list(data))
page = st.sidebar.radio("Раздел", ["Сравнение", "Отдельные запуски", "TRPO: доверительная область"])

if page == "Сравнение":
    rows = []
    fig, ax = plt.subplots(figsize=(11, 5))
    for name in selected:
        runs = data[name]
        scores = np.array([r["final_score"] for r in runs.values()])
        success = np.mean([r["final_success"] for r in runs.values()])
        rows.append({"Метод": name, "Средняя награда": scores.mean(), "std между зернами": scores.std(ddof=1), "Доля ≥ 475": success})
        grid = np.linspace(2048, 30000, 100)
        curves = np.array([np.interp(grid, r["eval_steps"], r["eval_mean"]) for r in runs.values()])
        ax.plot(grid, curves.mean(axis=0), label=name)
        ax.fill_between(grid, curves.mean(axis=0)-curves.std(axis=0), curves.mean(axis=0)+curves.std(axis=0), alpha=.15)
    ax.axhline(475, linestyle="--", color="firebrick", label="475")
    ax.set(xlabel="Шаги обучения", ylabel="Средняя оценочная награда", ylim=(0, 520))
    if selected:
        ax.legend()
        st.dataframe(pd.DataFrame(rows).style.format({"Средняя награда": "{:.1f}", "std между зернами": "{:.1f}", "Доля ≥ 475": "{:.1%}"}), hide_index=True)
    st.pyplot(fig)
    st.caption("Линии — среднее по пяти зернам; полоса — одно стандартное отклонение. Оценка в контрольных точках проводится на 10 эпизодах.")
elif page == "Отдельные запуски":
    method = st.selectbox("Метод", list(data))
    seed = st.selectbox("Зерно", list(data[method]))
    run = data[method][seed]
    st.metric("Итоговая оценка на 30 эпизодах", f'{run["final_score"]:.1f}')
    st.metric("Доля эпизодов с наградой ≥ 475", f'{run["final_success"]:.1%}')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(run["eval_steps"], run["eval_mean"], marker="o")
    ax.set(xlabel="Шаги обучения", ylabel="Средняя оценочная награда", ylim=(0, 520))
    st.pyplot(fig)
else:
    fig, ax = plt.subplots(figsize=(11, 5))
    for seed, run in data["TRPO"].items():
        ax.plot(run["train_steps"], run["kl"], marker="o", label=f"seed {seed}")
    ax.axhline(.01, color="firebrick", linestyle="--", label="Ограничение 0,01")
    ax.set(xlabel="Шаги обучения", ylabel="Среднее KL")
    ax.legend()
    st.pyplot(fig)
    st.caption("После линейного поиска сохранённое среднее KL не превышает выбранную границу.")
