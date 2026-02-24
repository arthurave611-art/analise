import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import numpy as np
import re
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="CalculaAí - Bioestatística", layout="wide")

# ---------------------------
# MAPAS IBGE
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


def serie_anual(df, periodo):
    serie = (
        df.groupby('Ano')['Casos']
        .sum()
        .sort_index()
    )
    serie = serie.loc[periodo[0]:periodo[1]]
    return serie.values.astype(float), serie.index.values


def serie_mensal(df, periodo):
    df['Data'] = pd.to_datetime(df['Periodo'], format='%Y%m')
    serie = (
        df.groupby('Data')['Casos']
        .sum()
        .sort_index()
    )
    serie = serie[
        (serie.index.year >= periodo[0]) &
        (serie.index.year <= periodo[1])
    ]
    return serie


# ---------------------------
# APP
# ---------------------------
st.title("📊 Análise de Tendência Epidemiológica")

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
    # SIDEBAR
    # ---------------------------
    st.sidebar.header("Configurações")

    nivel = st.sidebar.radio(
        "Nível",
        ("Brasil","Região","Estado")
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

    else:
        e = st.sidebar.selectbox(
            "Estado",
            sorted(df_final['Estado'].unique())
        )
        df_temp = df_final[df_final['Estado']==e]
        local = e

    anos = sorted(df_final['Ano'].unique())

    periodo = st.sidebar.select_slider(
        "Período",
        options=anos,
        value=(2014,2023)
    )

    tipo_serie = st.sidebar.radio(
        "Tipo de série",
        ("Anual","Mensal","STL (artigo)")
    )

    usar_hr = st.sidebar.checkbox(
        "Usar Hamed-Rao (corrigir autocorrelação)",
        value=True
    )

    cor_linha = st.sidebar.color_picker(
        "Cor da série",
        "#1f77b4"
    )

    largura = st.sidebar.slider(
        "Espessura da linha",
        1,
        5,
        2
    )

    tamanho = st.sidebar.slider(
        "Tamanho gráfico",
        6,
        18,
        12
    )

    # ---------------------------
    # SÉRIE
    # ---------------------------
    if tipo_serie == "Anual":

        serie_values, eixo = serie_anual(
            df_temp,
            periodo
        )

    else:

        serie = serie_mensal(
            df_temp,
            periodo
        )

        if tipo_serie == "STL (artigo)":
            stl = STL(serie, period=12)
            res = stl.fit()
            serie_values = res.trend.dropna().values
            eixo = np.arange(len(serie_values))
        else:
            serie_values = serie.values
            eixo = np.arange(len(serie_values))

    if len(serie_values) < 4:
        st.warning("Série insuficiente.")
        st.stop()

    # ---------------------------
    # TESTE
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

    st.subheader(f"Resultados - {local}")
    st.table(tabela)

    # ---------------------------
    # GRÁFICO
    # ---------------------------
    fig, ax = plt.subplots(
        figsize=(tamanho,5)
    )

    ax.plot(
        eixo,
        serie_values,
        linewidth=largura,
        color=cor_linha,
        marker='o'
    )

    x = np.arange(len(serie_values))
    intercept = np.mean(serie_values) - resultado.slope*np.mean(x)

    ax.plot(
        eixo,
        resultado.slope*x + intercept,
        linestyle='--'
    )

    ax.grid(True)

    st.pyplot(fig)

    # ---------------------------
    # EXPORTAR
    # ---------------------------
    st.download_button(
        "Baixar resultados CSV",
        tabela.to_csv(index=False),
        file_name="resultado_mann_kendall.csv"
    )