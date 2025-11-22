import streamlit as st
import pandas as pd
import shap
from shap import TreeExplainer
from sklearn.model_selection import train_test_split
from plots import colunas_traduzidas
from joblib import load
import streamlit.components.v1 as components

# --- Configuração Inicial ---
st.set_page_config(layout="wide", page_title="Análise de Risco de Fraude em Tempo Real", page_icon="🕵️‍♂️")
st.title("Justificativa de Decisão do Modelo de Fraude")

# --- Carregamento de objetos e dados ---
modelo = st.session_state['modelo']
colunas_selecionadas = st.session_state['colunas_selecionadas']
seletor = load("objects/seletor.pkl")

df = pd.read_csv('Fraud_transactions.csv')
df.rename(columns=colunas_traduzidas, inplace=True)
y = df['fraude']
X_raw = df.drop(columns=['fraude', 'hora_transacao', 'id_transacao', 'id_usuario'])

# Codificação das variáveis categóricas
for coluna in X_raw.select_dtypes(include='object').columns:
    le = load(f'objects/label_encoder_{coluna}.pkl')
    X_raw[coluna] = le.transform(X_raw[coluna].astype(str))

# Aplicar seletor de características
X = seletor.transform(X_raw)
_, X_test, _, y_teste = train_test_split(X, y, test_size=0.25, random_state=1432)

# --- Sidebar ---
st.sidebar.header("Pesquisa de Transação")
transaction_ids = list(range(len(X_test)))
selected_id = st.sidebar.number_input("Selecione o ID da Transação:", min_value=0,
                                      max_value=len(transaction_ids) - 1, value=0, step=1)

# --- Lógica de Explicabilidade ---
if st.sidebar.button("Analisar Transação"):
    transaction_data = pd.DataFrame([X_test[selected_id]], columns=colunas_selecionadas)

    # Previsão
    prediction_proba = modelo.predict_proba(transaction_data)[:, 1][0]
    prediction_class = modelo.predict(transaction_data)[0]

    # Explicação SHAP
    explainer = TreeExplainer(modelo)
    shap_values_local = explainer.shap_values(transaction_data)

    # Apresentação
    st.subheader(f"Análise para Transação ID: {selected_id}")
    if prediction_class == 1:
        st.error(f"⚠️ **SINALIZADA COMO FRAUDE** (Probabilidade: {prediction_proba:.2%})")
    else:
        st.success(f"✅ **LEGÍTIMA** (Probabilidade: {prediction_proba:.2%})")

    import plotly.graph_objects as go
    import shap

    # Calcular valores SHAP
    shap_values_local = explainer.shap_values(transaction_data)[0]  # para modelos binários
    feature_names = transaction_data.columns.tolist()

    # Criar gráfico de barras ordenado
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values_local
    }).sort_values(by='shap_value', key=abs, ascending=False)

    fig = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker=dict(color=shap_df['shap_value'], colorscale='RdBu'),
    ))

    fig.update_layout(
        title="Contribuição de cada variável para a previsão",
        xaxis_title="Valor SHAP",
        yaxis_title="Variável",
        height=600,
        margin=dict(l=100, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)