import streamlit as st
import pandas as pd
from shap import TreeExplainer
from sklearn.model_selection import train_test_split
from plots import colunas_traduzidas, markdown
from joblib import load
import plotly.graph_objects as go


# --- Configuração Inicial ---
st.set_page_config(layout="wide", page_title="Análise de Risco de Fraude em Tempo Real", page_icon="🕵️‍♂️")

# --- Carregamento de objetos e dados ---
modelo = st.session_state['modelo']
colunas_selecionadas = st.session_state['colunas_selecionadas']
seletor = load("objects/seletor.pkl")

df = pd.read_csv('Fraud_transactions.csv')
df.rename(columns=colunas_traduzidas, inplace=True)
mapeamentos = {0: 'Não', 1: 'Sim'}
df['autenticacao_3ds'] = df['autenticacao_3ds'].map(mapeamentos)
df['promocao_usada'] = df['promocao_usada'].map(mapeamentos)
df['resultado_cvv'] = df['resultado_cvv'].map(mapeamentos)
df['verificacao_endereco'] = df['verificacao_endereco'].map(mapeamentos)  
y = df['fraude']
X_raw = df.drop(columns=['fraude', 'hora_transacao', 'id_transacao', 'id_usuario'])

# Codificação das variáveis categóricas
original_data = X_raw.copy()
original_data = original_data[colunas_selecionadas]  # Dados originais para exibição posterior
for coluna in X_raw.select_dtypes(include='object').columns:
    le = load(f'objects/label_encoder_{coluna}.pkl')
    X_raw[coluna] = le.transform(X_raw[coluna].astype(str))

# Aplicar seletor de características
X = seletor.transform(X_raw)
_, X_test, _, y_teste = train_test_split(X, y, test_size=0.25, random_state=1432)


st.header("Justificativa de Decisão do Modelo de Fraude", divider="green")    
# --- Sidebar ---
st.sidebar.header("Pesquisa de Transação")
transaction_ids = list(range(len(X_test)))
selected_id = st.sidebar.number_input("Selecione o ID da Transação:", min_value=0,
                                      max_value=len(transaction_ids) - 1, value= 18, step=1)


# --- Lógica de Explicabilidade ---
if st.sidebar.button("Analisar Transação", use_container_width=True, 
                     type='primary', help="Clique para gerar a interpretação"):
    progress = st.sidebar.progress(50, "Aguarde.... Gerando Explicabilidade do Modelo")
    transaction_data = pd.DataFrame([X_test[selected_id]], columns=colunas_selecionadas)
    original_data_row = original_data.iloc[selected_id:selected_id+1]
    y_true = y_teste.iloc[selected_id]


    # Previsão
    prediction_proba = (modelo.predict_proba(transaction_data)[:, 1][0])
    prediction_class = modelo.predict(transaction_data)[0]

    # Explicação SHAP
    explainer = TreeExplainer(modelo)
    shap_values_local = explainer.shap_values(transaction_data)


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
        marker=dict(color=['#e74c3c' if val > 0 else '#2ecc71' for val in shap_df['shap_value']], showscale=False),
        hovertemplate='Feature: %{y}<br>SHAP Value: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Contribuição para a previsão",
        xaxis_title="Valor da contribuição",
        yaxis_title="Características da Transação",
        height=500,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),        
    )

    progress.progress(80, "Quase lá... Finalizando a análise")
    col1, col2 = st.columns([0.65, 0.35], gap="medium", border=True)
    
        
    progress.progress(100, "Geração da análise Concluída!")
    # Apresentação do gráfico SHAP
    with col1:
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("-----")
        st.markdown("### 🧠 Interpretação da Decisão do Modelo")
        st.markdown("""
        Este painel detalha os fatores que mais influenciaram na decisão final do Modelo.
        A análise é baseada em **contribuições individuais de cada variável**, permitindo:        
        - **Aumentar a confiança nas previsões do modelo** através de lógicas de decisão transparentes.
        - **Identificar padrões recorrentes** em fraudes para ajustes futuros.
        """)       
            
        
        # Apresentação dos resultados
        with col2:
            st.subheader(f"Análise para Transação ID: {selected_id}")

            if y_true ==prediction_class:
                st.success("*A previsão do Modelo está:* **Correta**")
            else:
                st.error("**A previsão do Modelo está:* **Incorreta**")
            st.markdown("**Dados originais da transação:**")        

            st.code(original_data_row.T, language='plaintext')
            
            if prediction_class == 1:
                st.error(f"⚠️ **SINALIZADA COMO FRAUDE** (Probabilidade: {prediction_proba:.2%})")
            else:
                prediction_proba = 1 - prediction_proba
                st.success(f"✅ **LEGÍTIMA** (Probabilidade: {prediction_proba:.2%})")

            st.markdown("-----")
            st.markdown("### 🔍 Contribuições para a Previsão")
            st.markdown("""
            O gráfico de barras mostra o **impacto de cada característica da transação** na decisão final:
            - **Barras vermelhas**: aumentam a probabilidade de fraude.
            - **Barras verdes**: reduzem a probabilidade de fraude.
                
        """)

st.sidebar.markdown(markdown, unsafe_allow_html=True)
