import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import re

st.set_page_config(page_title="Decomposição STL", layout="wide")

st.title("📈 Decomposição de Série Temporal (STL)")
st.markdown("Separação de **Tendência**, **Sazonalidade** e **Resíduo**.")

# --- LÓGICA DE LEITURA (MESMA DO ANTERIOR PARA CONSISTÊNCIA) ---
def processar_dados_mensais(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    
    # Criar coluna de data real para o STL (YYYY-MM-01)
    # Transforma '2014/Jan' em '2014-01-01'
    meses_map = {'Jan':'01','Fev':'02','Mar':'03','Abr':'04','Mai':'05','Jun':'06',
                 'Jul':'07','Ago':'08','Set':'09','Out':'10','Nov':'11','Dez':'12'}
    
    def formatar_data(p):
        ano, mes_nome = p.split('/')
        return f"{ano}-{meses_map[mes_nome]}-01"
    
    df_long['Data'] = pd.to_datetime(df_long['Periodo'].apply(formatar_data))
    return df_long

uploaded_file = st.sidebar.file_uploader("Upload do CSV", type=['csv'])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
    df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
    df_final = processar_dados_mensais(df_raw)
    
    # Agrupamento para "País Inteiro" (como você pediu)
    serie_mensal = df_final.groupby('Data')['Casos'].sum().sort_index()
    
    # O STL exige uma frequência definida (Mensal = 12)
    serie_mensal.index.freq = 'MS' 

    if len(serie_mensal) > 24: # STL precisa de pelo menos 2 ciclos
        st.subheader("Visualização dos Componentes")
        
        # Execução do STL
        # period=12 porque os dados são mensais e a sazonalidade é anual
        res = STL(serie_mensal, period=12).fit()
        
        # Criando o gráfico no estilo profissional
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        ax1.plot(serie_mensal, color='black', lw=1.5)
        ax1.set_title('Série Observada (Dados Brutos)')
        
        ax2.plot(res.trend, color='blue', lw=1.5)
        ax2.set_title('Tendência (Componente de Longo Prazo)')
        
        ax3.plot(res.seasonal, color='green', lw=1.5)
        ax3.set_title('Sazonalidade (Padrão Repetitivo Anual)')
        
        ax4.scatter(serie_mensal.index, res.resid, color='red', s=10)
        ax4.axhline(0, color='black', lw=1, ls='--')
        ax4.set_title('Resíduo (Ruído e Anomalias)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.info("💡 **Interpretação:** A Tendência mostra se a doença está crescendo indepedente do mês. A Sazonalidade confirma os picos anuais (comum na Dengue).")
    else:
        st.warning("Dados insuficientes para decomposição STL (mínimo 24 meses).")