import pandas as pd
import numpy as np
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from lightgbm import LGBMClassifier
from plots import *
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import pandas as pd


# Configuração da página
st.set_page_config(page_title="Detecção de Fraudes", layout="centered")

# Carregar objetos
modelo = load("objects/modelo_fraude.pkl")
seletor = load("objects/seletor.pkl")
colunas_selecionadas = load("objects/colunas_selecionadas.pkl")
accuracy, confusion = load("objects/metricas.pkl")

# Interface lateral
with st.sidebar:
    st.markdown("## 📊 Acurácia do Modelo")
    st.markdown(f"**{accuracy*100:.2f}%** baseado nos dados de validação")
    st.markdown("---")
    st.markdown("## 🔗 Desenvolvedor: Ronivan")
    # (links e rodapé mantidos)

# Interface principal
st.title("🕵️‍♂️ Avaliação de Risco de Fraude")

# Entrada dinâmica de dados
dataset = pd.read_csv("Fraud_transactions.csv")
dataset.rename(columns=colunas_traduzidas, inplace=True)
mapeamentos = {0: 'Não', 1: 'Sim'}
dataset['autenticacao_3ds'] = dataset['autenticacao_3ds'].map(mapeamentos)
dataset['promocao_usada'] = dataset['promocao_usada'].map(mapeamentos)
dataset['resultado_cvv'] = dataset['resultado_cvv'].map(mapeamentos)
dataset['verificacao_endereco'] = dataset['verificacao_endereco'].map(mapeamentos)  

# Salvalmento de objetos com streamlit
st.session_state['modelo'] = modelo
st.session_state['colunas_selecionadas'] = colunas_selecionadas


entrada = {}
for coluna in colunas_selecionadas:
    valores_unicos = dataset[coluna].unique()
    if dataset[coluna].dtype == 'object':
        entrada[coluna] = st.selectbox(f"{coluna}", valores_unicos)
    else:        
        entrada[coluna] = st.number_input(f"{coluna}", value=dataset[coluna].median()
                                           if dataset[coluna].dtype in ['float'] else int(dataset[coluna].median()))

# Limpar Memória
if dataset is not None:
    del dataset
        

# Botão de previsão
if st.button("Avaliar Transação"):
    dados_novos = pd.DataFrame([entrada])
    for coluna in colunas_selecionadas:
        if dados_novos[coluna].dtype == 'object':        
            le = load('objects/label_encoder_' + coluna + '.pkl')
            dados_novos[coluna] = le.transform(dados_novos[coluna].astype(str))       

    probabilidade = modelo.predict_proba(dados_novos)[0][1] * 100
    classe = "Fraude" if probabilidade >= 30 else "Legítima"

    if classe == "Fraude":
        #probabilidade = 100 - probabilidade
        st.error(f"🚨 Transação suspeita! Probabilidade de fraude: {probabilidade:.2f}%")
    else:
        st.success(f"✅ Transação legítima. Probabilidade de fraude: {probabilidade:.2f}%")