import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Funções de Verificação
def check_missing_values(df):
    """
    Identifica e imprime as colunas com valores nulos em um DataFrame.

    Parâmetros:
        df (pd.DataFrame): O DataFrame a ser analisado.

    Retorna:
        None: A função imprime o resultado diretamente.
    """
    print(" ============== Resumo de Valores Nulos ============== ")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Coluna': df.columns,
        'Valores Nulos': missing_data,
        'Porcentagem (%)': missing_percentage.round(2)
    })
    missing_summary = missing_summary[missing_summary['Valores Nulos'] > 0].reset_index(drop=True)
    
    if missing_summary.empty:
        print("Não há valores nulos neste DataFrame.")
    else:
        print(missing_summary.to_string(index=False))

    print("\nTotal de colunas com valores nulos:", len(missing_summary))

def check_outliers_column(df, column, factor=1.5):
    """
    Identifica e exibe informações sobre outliers em uma coluna de um DataFrame.

    Parâmetros:
        df (pd.DataFrame): O DataFrame a ser analisado.
        column (str): O nome da coluna em que serão detectados os outliers.
        factor (float, opcional): O fator multiplicador do IQR para definir os limites dos outliers (padrão: 1.5).

    Retorna:
        dict: Dicionário contendo informações sobre os outliers.
    """
    print(f" ============== Resumo de Outliers: {column} ============== ")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - factor * IQR)
    upper_bound = Q3 + factor * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    
    total_rows = len(df)
    num_outliers = len(outliers)
    percentage_outliers = (num_outliers / total_rows) * 100
    
    if num_outliers == 0:
        print("Não há outliers nesta coluna.")
        return None
    
    print(f"Limite inferior: {lower_bound:.2f}")
    print(f"Limite superior: {upper_bound:.2f}")
    print(f"\nNúmero de outliers: {num_outliers}")
    print(f"Porcentagem de outliers: {percentage_outliers:.2f}%")
    
    print("\nResumo estatístico dos outliers:")
    print(outliers.describe())
    
    print("\nPrimeiros 10 valores outliers:")
    print(outliers.head(10).tolist())
    
    return

# Funções de Tratamento
def treat_missing_values_column_median(df, column):
    """
    Trata valores ausentes em uma coluna de um DataFrame usando a mediana.

    Esta função identifica valores ausentes (NaN) em uma coluna específica
    e os substitui pela mediana dos valores não ausentes da mesma coluna.

    Parâmetros:
        df (pd.DataFrame): O DataFrame a ser tratado.
        column (str): O nome da coluna em que serão tratados os valores ausentes.

    Retorna:
        None: A função modifica o DataFrame in-place e não retorna nenhum valor.

    Efeitos colaterais:
        - Modifica o DataFrame original, substituindo valores ausentes pela mediana.
        - Imprime uma mensagem confirmando o tratamento dos valores ausentes.
    """
    median = df[column].median()
    df[column] = df[column].fillna(median)
    print(f"Valores ausentes na coluna '{column}' foram tratados com sucesso usando a mediana ({median}).")
    
    return


def treat_outliers_column_median(df, column):
    """
    Trata outliers em uma coluna de um DataFrame substituindo-os pela mediana.

    Esta função identifica outliers usando o método do Intervalo Interquartil (IQR)
    e substitui os valores identificados como outliers pela mediana da coluna.

    Parâmetros:
        df (pd.DataFrame): O DataFrame a ser tratado.
        column (str): O nome da coluna em que serão tratados os outliers.

    Retorna:
        None: A função modifica o DataFrame in-place e não retorna nenhum valor.

    Efeitos colaterais:
        - Modifica o DataFrame original, substituindo outliers pela mediana.
        - Imprime uma mensagem confirmando o tratamento dos outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    median = df[column].median()
    df[column] = np.where((df[column] < lower_bound) | (df[column] > upper_bound), median, df[column])
    print(f"Outliers na coluna '{column}' foram tratados com sucesso usando a mediana ({median}).")
    
    return

# Funções de Visualizações de Dados

def create_frequency_table(df, column_name):
    """
    Cria e exibe uma tabela de frequência para uma coluna específica de um DataFrame.

    Esta função calcula a contagem e a porcentagem de cada valor único em uma coluna
    especificada, ordena os resultados por contagem decrescente e exibe a tabela.

    Parâmetros:
        df (pd.DataFrame): O DataFrame contendo os dados a serem analisados.
        column_name (str): O nome da coluna para a qual a tabela de frequência será criada.

    Retorna:
        None: A função não retorna nenhum valor, mas imprime a tabela de frequência.

    Efeitos colaterais:
        - Imprime a tabela de frequência no console, mostrando a contagem e a porcentagem
          para cada valor único na coluna especificada.
    """
    counts = df[column_name].value_counts()
    percentages = df[column_name].value_counts(normalize=True).mul(100).round(2)
    
    result = pd.concat([counts, percentages], axis=1, keys=['Contagem', 'Porcentagem'])
    result['Porcentagem'] = result['Porcentagem'].astype(str) + '%'
    
    result = result.sort_values('Contagem', ascending=False)
    
    print(f"\nDistribuição de {column_name}:")
    print(result)
    
    return


def plot_categorical(df, column):
    """
    Cria e exibe um gráfico de barras para uma coluna categórica de um DataFrame.

    Esta função gera um gráfico de barras mostrando a distribuição de valores
    em uma coluna categórica. Cada barra representa um valor único, com sua altura
    indicando a contagem desse valor. As barras são rotuladas com a contagem e a
    porcentagem correspondente.

    Parâmetros:
        df (pd.DataFrame): O DataFrame contendo os dados a serem visualizados.
        column (str): O nome da coluna categórica a ser plotada.

    Retorna:
        None: A função não retorna nenhum valor, mas exibe o gráfico.

    Efeitos colaterais:
        - Cria e exibe um gráfico de barras usando matplotlib e seaborn.
        - O gráfico mostra a distribuição de valores na coluna especificada,
          com contagens e percentagens para cada categoria.
    """
    plt.figure(figsize=(8, 8))
    
    value_counts = df[column].value_counts()
    
    ax = sns.barplot(x=value_counts.index, y=value_counts.values, order=value_counts.index)
    
    plt.title(f'Distribuição de {column}')
    plt.xlabel(column)
    plt.ylabel('Contagem')
    
    total = len(df)
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        percentage = f'{100 * height / total:.1f}%'
        label = f'{int(height)}\n({percentage})'
        ax.text(p.get_x() + p.get_width()/2, height, 
                label, 
                ha='center', va='bottom',
                rotation=0)
    
    plt.tight_layout()
    plt.show()

    
def plot_histogram(df, column, bins=30):
    """
    Cria e exibe um histograma para uma coluna numérica de um DataFrame.

    Esta função gera um histograma mostrando a distribuição de valores
    em uma coluna numérica. O histograma divide os dados em um número
    especificado de bins, permitindo visualizar a frequência de valores
    em diferentes intervalos.

    Parâmetros:
        df (pd.DataFrame): O DataFrame contendo os dados a serem visualizados.
        column (str): O nome da coluna numérica a ser plotada.
        bins (int, opcional): O número de bins para o histograma. Padrão é 30.

    Retorna:
        None: A função não retorna nenhum valor, mas exibe o histograma.

    Efeitos colaterais:
        - Cria e exibe um histograma usando matplotlib.
        - O histograma mostra a distribuição de valores na coluna especificada,
          divididos no número de bins definido.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df[column], bins=bins, edgecolor='black')
    plt.title(f'Distribuição de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()
