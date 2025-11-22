import pandas as pd
import plotly.express as px
import streamlit as st

def plot_fraud_segments(df, target_col='Is_Fraud', n=10, col = 'Age'):
    """
    Gera gráficos de barras interativos com Plotly para mostrar onde ocorrem mais fraudes.
    
    Parâmetros:
    - df: DataFrame com os dados
    - target_col: nome da coluna binária de fraude (1 = fraude, 0 = normal)
    - top_n: número de categorias mais frequentes a exibir por gráfico

    Retorna:
    - Exibe os gráficos diretamente no Streamlit
    """
    # Detectar colunas categóricas
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes('number').columns.tolist()

    if col in num_cols:
        # Categorizar a coluna numérica em bins
        df[col] = pd.qcut(df[col], q=n * 3, duplicates='drop')
        df[col] = df[col].round().astype(str)        
            

    if col in cat_cols or col in num_cols:
        # Agrupar por categoria e calcular taxa de fraude
        fraud_stats = df.groupby(col)[target_col].agg(['count', 'sum'])
        fraud_stats['fraude_rate'] = fraud_stats['sum'] / fraud_stats['count']
        fraud_stats = fraud_stats.sort_values('fraude_rate', ascending=False).reset_index()
        fraud_stats_max = fraud_stats.nlargest(n, 'fraude_rate').reset_index()
        fraud_stats_min = fraud_stats.nsmallest(n, 'fraude_rate').reset_index()

        # Gráfico de barras com Plotly
        fig = px.bar(
            fraud_stats_max,
            x=col,
            y='fraude_rate',
            color='fraude_rate',
            text='fraude_rate',
            title=f'Taxa de Fraude por {col}',
            labels={col: col, 'fraude_rate': 'Taxa de Fraude'},
            color_continuous_scale='Reds'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='inside')
        fig.update_layout(xaxis_tickangle=-45)

        fig2 = px.bar(
            fraud_stats_min,
            x=col,
            y='fraude_rate',
            color='fraude_rate',
            text='fraude_rate',
            title=f'Taxa de Fraude Mínima por {col}',
            labels={col: col, 'fraude_rate': 'Taxa de Fraude'},
            color_continuous_scale='Greens_r'
        )
        fig2.update_traces(texttemplate='%{text:.2%}', textposition='inside')        
        fig2.update_layout(xaxis_tickangle=-45)

    else:
        raise ValueError(f"A coluna '{col}' não é categórica ou numérica.")

    # Exibir no Streamlit
    return fig, fig2