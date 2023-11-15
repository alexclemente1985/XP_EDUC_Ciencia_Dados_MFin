import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import yfinance as yf

def pandas_lesson():
    # Criação de uma Série
    s = pd.Series([1, 3, 5, 6, 8, "ações", "Stocks", "ETFs"])
    print("Exemplo de uma série contendo dados do tipo numéricos e strings:")
    print("A primeira coluna se refere ao índice e a segunda aos dados")
    print(s)

    print()

    # Criação de um Dataframe
    data = {'Acao': ['Ambev', 'Wege', 'Vale'],
            'Preco': [15, 35, 68],
            'Dy': ['4%', '2%', '8%']}
    df_basic = pd.DataFrame(data)
    print("Exemplo de um dataframe:")
    print("A primeira coluna se refere ao índice e as demais aos dados")
    print(df_basic)

    # Importação de dados de um arquivo CSV

    df = pd.read_csv(r'dataenv/california_housing_test.csv')
    print("Exemplo de leitura de um arquivo CSV com Pandas:\n\n", df)

    # Exploração de dados

    print("Exemplo do uso de .head: (Pega as primeiras 5 linhas)\n")
    print(df.head())

    print("\nExemplo do uso de .tail: (Pega as ultimas 5 linhas)\n")
    print(df.tail())

    print("\nExemplo do uso de .sample: (Pega 1 linha aleatória)\n")
    print(df.sample())

    print("\nExemplo do uso de .info:\n")
    print(df.info())

    print("OBS: Aqui conseguimos verificar se temos valores NULOS para limpar")
    print("Observe o total de entradas de 3000 registros (entries - 0 a 2999) e o total de valores não nulos (3000)")
    print("Se tivesse valores null em alguma coluna este valor (3000 non-null) estaria menor")

    print("\nExemplo do uso de .describe:")
    print(
        "Oferece estatísticas descritivas, como média, desvio padrão e quartis, para todas as colunas numéricas do dataframe.\n")
    print(df.describe())

    # Visualização de dados

    print("Exemplo do uso do comando PRINT (Mais Simples)\n")
    print(df)
    print("\nExemplo do uso do comando DISPLAY (Mais jeito de tabela)\n")
    print("Inclusive com botão de sugestão de gráficos\n")
    display(df)

    # Exploração de dados (Limpeza e Preparação dos dados)

    df.dropna()  # Remove linhas com valores ausentes

    df.drop_duplicates()  # Remove linhas duplicadas

    # DataFrame de exemplo com dados do mercado de renda variável
    data = {
        'Data': ['2023-07-01', '2023-07-02', '2023-07-03', '2023-07-04'],
        'Ação': ['AAPL', 'GOOG', None, 'MSFT'],
        'Preço': [150.5, None, 350.2, 280.7],
        'Volume': [1000000, 1500000, None, 800000],
        'Variação': [0.02, 0.01, 0.03, None]
    }

    df = pd.DataFrame(data)

    # Exibindo o DataFrame original
    print("DataFrame original:")
    print(df)

    # Remoção de valores nulos
    df_dropna = df.dropna()  # Remove linhas com pelo menos um valor nulo

    # Preenchimento de valores nulos
    df_fillna = df.fillna(0)  # Preenche os valores nulos com 0
    df_ffill = df.ffill() #df.fill(
        #method='ffill')  # Preenche os valores nulos com o último valor válido (preenchimento para frente)
    df_bfill = df.bfill() #df.fill(
        #method='bfill')  # Preenche os valores nulos com o próximo valor válido (preenchimento para trás)

    # Verificação de valores nulos
    df_isnull = df.isnull()  # DataFrame booleano mostrando as células com valores nulos

    # Exibindo os DataFrames resultantes
    print("\nDataFrame após a remoção de valores nulos:")
    print(df_dropna)

    print("\nDataFrame após o preenchimento de valores nulos:")
    print(df_fillna)

    print("\nDataFrame após o preenchimento para frente de valores nulos:")
    print(df_ffill)

    print("\nDataFrame após o preenchimento para trás de valores nulos:")
    print(df_bfill)

    print("\nDataFrame com a verificação de valores nulos:")
    print(df_isnull)

    # Análise de dados
    # Dicionário com os dados
    data = {
        'coluna1': ['A', 'B', 'A', 'B', 'A'],
        'coluna2': [10, 20, 30, 40, 50]
    }

    # Criando o DataFrame
    df = pd.DataFrame(data)
    print("Dataframe original:")
    print(df)

    # Usando groupby para agrupar pela coluna1 e calculando a média das outras colunas
    grupo_media = df.groupby('coluna1').mean()

    print("\nGroupby media:")
    print(grupo_media)

    # Exemplo de Groupby e Soma dos valores de cada grupo:

    data = {
        'coluna1': ['A', 'B', 'A', 'B', 'A'],
        'coluna2': [10, 20, 30, 40, 50]
    }

    df = pd.DataFrame(data)
    print("Dataframe original:")
    print(df)

    grupo_soma = df.groupby('coluna1').sum()

    print("\nGroupby soma:")
    print(grupo_soma)

    # Exemplo de Groupby e Média para cada combinação de valores de colunas:

    data = {
        'coluna1': ['A', 'B', 'A', 'B', 'A', 'B', 'A'],
        'coluna2': [10, 20, 10, 20, 30, 30, 30],
        'coluna3': [5, 15, 25, 35, 45, 55, 65]
    }

    df = pd.DataFrame(data)
    print("Dataframe original:")
    print(df)

    grupo_media_combinada = df.groupby(['coluna1', 'coluna2']).mean()

    print("\nGroupby media combinada:")
    print(grupo_media_combinada)

    # O Pandas possui integração com a biblioteca Matplotlib para criação de gráficos.

    #df = pd.read_csv('dados.csv')
    #df.plot(x='Data', y='Valor', kind='line')
    #plt.show()

    # Exemplo de importação de dados de ações do Yahoo Finance

    df = yf.download('AAPL', start='2022-01-01', end='2022-12-31')
    print(df)
    print("Visualizando com DISPLAY")
    display(df)

if __name__ == '__main__':
    pandas_lesson()