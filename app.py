import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import numpy as np
import re

st.set_page_config(page_title="CalculaAí - Bioestatística", layout="wide")

# ---------------------------
# MAPEAMENTO IBGE
# ---------------------------
MAPA_ESTADOS = {
'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO',
'21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL',
'28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR',
'42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'
}

MAPA_REGIOES = {
'1':'Norte','2':'Nordeste','3':'Sudeste','4':'Sul','5':'Centro-Oeste'
}

# ---------------------------
# FUNÇÕES
# ---------------------------
def extrair_geo(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)

    if codigo:
        cod_uf = codigo.group(1)
        return (
            MAPA_REGIOES.get(cod_uf[0], 'Outros'),
            MAPA_ESTADOS.get(cod_uf, 'Outros'),
            re.sub(r'^\d+\s*', '', nome)
        )

    return 'Brasil', 'Brasil', 'Brasil'


def processar_tabela(df):
    col_geo = df.columns[0]

    cols_tempo = [
        c for c in df.columns
        if re.match(r'^\d{4}', str(c))
    ]

    df_long = df.melt(
        id_vars=[col_geo],
        value_vars=cols_tempo,
        var_name='Periodo',
        value_name='Casos'
    )

    # limpeza padrão DATASUS
    df_long['Casos'] = (
        df_long['Casos']
        .astype(str)
        .str.replace('-', '0')
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )

    df_long['Casos'] = pd.to_numeric(
        df_long['Casos'],
        errors='coerce'
    ).fillna(0)

    df_long['Ano'] = df_long['Periodo'].str[:4].astype(int)

    geos = df_long[col_geo].apply(extrair_geo)
    df_long[['Regiao','Estado','Municipio']] = pd.DataFrame(
        geos.tolist(),
        index=df_long.index
    )

    return df_long


def montar_serie_temporal(df, periodo):
    serie = (
        df.groupby('Ano')['Casos']
        .sum()
        .sort_index()
    )

    serie = serie.loc[periodo[0]:periodo[1]]

    # garantir continuidade temporal
    idx = pd.date_range(
        start=str(periodo[0]),
        end=str(periodo[1]),
        freq='YS'
    )

    serie.index = pd.to_datetime(serie.index, format='%Y')
    serie = serie.reindex(idx, fill_value=0)

    return serie.values.astype(float), idx.year


# ---------------------------
# APP
# ---------------------------
st.title("📊 Teste Mann-Kendall Modificado (Hamed & Rao)")

file = st.file_uploader("Upload CSV DATASUS", type=['csv'])

if file:

    df_raw = pd.read_csv(
        file,
        sep=';',
        encoding='ISO-8859-1'
    )

    df_raw = df_raw[
        ~df_raw.iloc[:,0]
        .astype(str)
        .str.contains('Total|TOTAL|Incompleto', na=False)
    ]

    df_final = processar_tabela(df_raw)

    # ---------------------------
    # FILTROS
    # ---------------------------
    st.sidebar.header("Filtros")

    nivel = st.sidebar.radio(
        "Nível geográfico",
        ("Brasil","Região","Estado","Município")
    )

    if nivel == "Brasil":
        df_temp = df_final
        local = "Brasil"

    elif nivel == "Região":
        r = st.sidebar.selectbox(
            "Região",
            sorted(df_final['Regiao'].unique())
        )
        df_temp = df_final[df_final['Regiao']==r]
        local = r

    elif nivel == "Estado":
        e = st.sidebar.selectbox(
            "Estado",
            sorted(df_final['Estado'].unique())
        )
        df_temp = df_final[df_final['Estado']==e]
        local = e

    else:
        uf = st.sidebar.selectbox(
            "UF",
            sorted(df_final['Estado'].unique())
        )

        mun = st.sidebar.selectbox(
            "Município",
            sorted(
                df_final[
                    df_final['Estado']==uf
                ]['Municipio'].unique()
            )
        )

        df_temp = df_final[
            df_final['Municipio']==mun
        ]

        local = mun

    anos = sorted(df_final['Ano'].unique())

    periodo = st.sidebar.select_slider(
        "Período",
        options=anos,
        value=(2014,2023)
    )

    # ---------------------------
    # SÉRIE TEMPORAL
    # ---------------------------
    serie_values, anos_plot = montar_serie_temporal(
        df_temp,
        periodo
    )

    if len(serie_values) < 4:
        st.warning("Série muito curta.")
        st.stop()

    # ---------------------------
    # TESTE MANN-KENDALL
    # ---------------------------
    res_hr = mk.hamed_rao_modification_test(
        serie_values,
        alpha=0.05,
        lag=1
    )

    res_orig = mk.original_test(serie_values)

    # CORREÇÃO DO TAU (compatível com qualquer versão)
    tau_val = getattr(res_orig, "tau", getattr(res_orig, "Tau", np.nan))

    st.subheader(f"Resultados - {local}")

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
            res_hr.trend,
            res_hr.h,
            res_hr.p,
            res_hr.z,
            tau_val,
            res_hr.slope
        ]
    })

    st.table(tabela)

    # ---------------------------
    # GRÁFICO
    # ---------------------------
    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        anos_plot,
        serie_values,
        marker='o',
        linewidth=2
    )

    x = np.arange(len(serie_values))
    intercept = np.mean(serie_values) - res_hr.slope*np.mean(x)

    ax.plot(
        anos_plot,
        res_hr.slope*x + intercept,
        linestyle='--',
        linewidth=2
    )

    ax.set_title(f"Dengue - {local}")
    ax.grid(True)

    st.pyplot(fig)