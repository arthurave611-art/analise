import streamlit as st

# No topo do app.py
st.sidebar.success("Selecione uma análise acima.")

import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import numpy as np
import re
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="CalculaAí - Bioestatística", layout="wide")

# ---------------------------
# FUNÇÃO LIMPEZA DATASUS
# ---------------------------
def limpar_valores(col):

    return (
        col.astype(str)
        .str.replace('-', '0')
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )

# ---------------------------
# APP
# ---------------------------
st.title("📊 Mann-Kendall - Séries Epidemiológicas")

file = st.file_uploader("Upload CSV DATASUS", type=['csv'])

if file:

    df = pd.read_csv(
        file,
        sep=';',
        encoding='ISO-8859-1'
    )

    col_geo = df.columns[0]

    cols_tempo = [
        c for c in df.columns
        if re.match(r'^\d{6}', str(c))
    ]

    df_long = df.melt(
        id_vars=[col_geo],
        value_vars=cols_tempo,
        var_name='Periodo',
        value_name='Casos'
    )

    df_long['Casos'] = limpar_valores(df_long['Casos'])

    df_long['Casos'] = pd.to_numeric(
        df_long['Casos'],
        errors='coerce'
    ).fillna(0)

    # DATA MENSAL
    df_long['Data'] = pd.to_datetime(
        df_long['Periodo'],
        format='%Y%m'
    )

    # ---------------------------
    # SIDEBAR
    # ---------------------------
    st.sidebar.header("Configurações")

    tipo_serie = st.sidebar.radio(
        "Tipo de agregação",
        ("Mensal","Anual","STL")
    )

    usar_hr = st.sidebar.checkbox(
        "Usar Hamed-Rao",
        True
    )

    periodo_inicio = st.sidebar.text_input(
        "Mês inicial (AAAA-MM)",
        "2014-01"
    )

    periodo_fim = st.sidebar.text_input(
        "Mês final (AAAA-MM)",
        "2023-12"
    )

    cor = st.sidebar.color_picker(
        "Cor da linha",
        "#1f77b4"
    )

    largura = st.sidebar.slider(
        "Espessura",
        1,
        5,
        2
    )

    # ---------------------------
    # FILTRO PERÍODO
    # ---------------------------
    inicio = pd.to_datetime(periodo_inicio)
    fim = pd.to_datetime(periodo_fim)

    df_long = df_long[
        (df_long['Data'] >= inicio) &
        (df_long['Data'] <= fim)
    ]

    # ---------------------------
    # SÉRIE TEMPORAL
    # ---------------------------
    if tipo_serie == "Mensal":

        serie = (
            df_long.groupby('Data')['Casos']
            .sum()
            .sort_index()
        )

        eixo = serie.index
        serie_values = serie.values

    elif tipo_serie == "Anual":

        serie = (
            df_long.groupby(df_long['Data'].dt.year)['Casos']
            .sum()
            .sort_index()
        )

        eixo = serie.index
        serie_values = serie.values

    else:

        serie = (
            df_long.groupby('Data')['Casos']
            .sum()
            .sort_index()
        )

        stl = STL(serie, period=12)
        res = stl.fit()

        serie_values = res.trend.dropna().values
        eixo = np.arange(len(serie_values))

    if len(serie_values) < 4:
        st.warning("Série muito curta")
        st.stop()

    # ---------------------------
    # MANN-KENDALL
    # ---------------------------
    if usar_hr:
        resultado = mk.hamed_rao_modification_test(
            serie_values
        )
    else:
        resultado = mk.original_test(
            serie_values
        )

    res_orig = mk.original_test(serie_values)

    tau_val = getattr(
        res_orig,
        "tau",
        getattr(res_orig,"Tau",np.nan)
    )

    tabela = pd.DataFrame({
        "Métrica":[
            "Tendência",
            "h",
            "Valor-p",
            "Z",
            "Tau",
            "Sen slope"
        ],
        "Resultado":[
            resultado.trend,
            resultado.h,
            resultado.p,
            resultado.z,
            tau_val,
            resultado.slope
        ]
    })

    st.subheader("Resultados")
    st.table(tabela)

    # ---------------------------
    # GRÁFICO
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        eixo,
        serie_values,
        color=cor,
        linewidth=largura,
        marker='o'
    )

    x = np.arange(len(serie_values))

    intercept = (
        np.mean(serie_values)
        - resultado.slope*np.mean(x)
    )

    ax.plot(
        eixo,
        resultado.slope*x + intercept,
        linestyle='--'
    )

    ax.grid(True)

    st.pyplot(fig)

    st.download_button(
        "Baixar resultados",
        tabela.to_csv(index=False),
        file_name="resultado_mk.csv"
    )