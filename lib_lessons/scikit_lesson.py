import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Importa a classe StandardScaler do módulo preprocessing da biblioteca scikit-learn.
# Essa classe é usada para padronizar os dados antes da clusterização.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression


def scikit_lesson():

    # Modelo Regressão Linear

    # Dados de entrada
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Treinar o modelo de regressão linear
    reg = LinearRegression().fit(X, y)

    # Realizar a previsão para novos valores
    novos_valores = np.array([[6], [7]])
    previsao = reg.predict(novos_valores)

    # Imprimir a previsão
    print("O resultado numérico desta Regressão Linear é:")
    print()  # Este print sem argumento gera uma linha em branco
    print(previsao)
    print()  # Este print sem argumento gera uma linha em branco

    # Vamos ver o exemplo através de um gráfico para melhor entendimento

    print("O resultado gráfico desta Regressão Linear é:")
    print()  # Este print sem argumento gera uma linha em branco

    # Plotar o gráfico de dispersão e a linha de regressão
    plt.scatter(X, y, color='black')
    plt.plot(X, reg.predict(X), color='blue', linewidth=3, label='Regressão Linear')

    # Plotar a previsão como uma nova linha
    plt.plot(novos_valores, previsao, color='green', linestyle='--', linewidth=3, label='Previsão')

    # Configurar as legendas do gráfico
    plt.title('Regressão linear simples')
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.legend()
    plt.show()


    # Regressão Linear Exemplo prático

    # Importando a função make_regression do módulo sklearn.datasets.
    # Essa função é utilizada para criar dados sintéticos para problemas de regressão.

    # Gerando uma massa de dados:
    x, y = make_regression(n_samples=200, n_features=1, noise=30)
    # Estamos gerando 200 amostras, cada uma com uma única feature (1 característica) e adicionando ruído de valor 30.
    # O conjunto de dados gerado é armazenado nas variáveis x e y, onde x contém as features e y contém os valores alvo (rótulos).

    # Mostrando no gráfico:
    # Criando um gráfico de dispersão (scatter). O gráfico mostra os pontos de dados do conjunto x em relação aos valores alvo y.
    plt.scatter(x, y)
    print("Modelo de Regressão Linear\n")
    print("ETAPA 01: Gráfico de Dispersão com os dados gerados aleatóriamente para o nosso modelo de Regressão\n")
    plt.show()  # Exibe o gráfico

    # Criação do modelo
    modelo = LinearRegression()

    # Aqui, estamos ajustando o modelo aos dados x e y.
    # Ou seja, estamos treinando o modelo com os dados de entrada x e os valores alvo y.
    modelo.fit(x, y)

    # Nesta linha, estamos utilizando o modelo treinado para fazer previsões com base nos dados de entrada x.
    # O resultado é uma previsão para cada ponto em x.
    modelo.predict(x)

    # Nesta linha, estamos novamente criando um gráfico de dispersão com os pontos de dados x em relação aos valores alvo y.
    plt.scatter(x, y)
    # Aqui, estamos usando a função plot do matplotlib.pyplot para traçar uma linha que representa as previsões feitas pelo modelo em relação aos dados x.
    # A linha é desenhada em vermelho (color='red') com uma espessura de linha de 3 pixels (linewidth=3).
    plt.plot(x, modelo.predict(x), color='red', linewidth=3)
    print("ETAPA 02: Gráfico de Dispersão com a reta gerada pelo nosso modelo de regressão baseado nos pontos\n")
    plt.show()  # Exibe o gráfico
    # Esta linha exibe o gráfico contendo os pontos de dados originais x em relação aos valores alvo y e
    # a linha de previsão em vermelho feita pelo modelo de regressão linear.


    # Clusterização 1

    # Este código é um exemplo simples de como usar a biblioteca Scikit-learn em Python para executar o algoritmo de clusterização K-means em um conjunto de dados.

    # Primeiramente, o código importa as bibliotecas necessárias:
    # NumPy para trabalhar com matrizes e o KMeans do Scikit-learn para executar o algoritmo de clusterização.

    # Depois, um conjunto de dados é definido como a variável X, que é uma matriz NumPy de seis pontos em um espaço bidimensional.

    # Dados de entrada
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])

    # O próximo passo é treinar o modelo de clusterização usando a função KMeans do Scikit-learn, que recebe como argumento o número de clusters que deseja-se identificar.
    # Neste caso, o número de clusters é definido como 2.

    # Treinar o modelo de clusterização
    kmeans = KMeans(n_clusters=2).fit(X)

    # Obter os rótulos de cluster para cada objeto
    rotulos = kmeans.labels_

    # Imprimir os rótulos de cluster
    print("O resultado numérico desta Análise de Cluster é:")
    print()  # Este print sem argumento gera uma linha em branco
    print(rotulos)
    print()  # Este print sem argumento gera uma linha em branco

    # Vamos ver o exemplo através de um gráfico para melhor entendimento

    print("O resultado gráfico desta Análise de Cluster é:")
    print()  # Este print sem argumento gera uma linha em branco

    # Criar uma figura e um eixo
    fig, ax = plt.subplots()

    # Adicionar os pontos ao eixo
    ax.scatter(X[:, 0], X[:, 1], c=rotulos)

    # Exibir o gráfico
    plt.title('Análise de Cluster')
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.show()


    # Clusterização 2

    # Cria um dicionário com dados fictícios de indicadores fundamentalistas para seis empresas diferentes.
    dados = {
        'Empresa': ['Empresa A', 'Empresa B', 'Empresa C', 'Empresa D', 'Empresa E', 'Empresa F'],
        'Receita (milhões)': [100, 200, 150, 300, 250, 180],
        'Lucro Líquido (milhões)': [20, 30, 25, 40, 35, 28],
        'Margem de Lucro (%)': [20, 15, 16.67, 13.33, 14, 15.56],
    }

    # Converter os dados em um DataFrame
    df = pd.DataFrame(dados)

    # Seleciona apenas os indicadores fundamentalistas (Receita, Lucro Líquido e Margem de Lucro) para a clusterização.
    # A coluna "Empresa" é removida porque não é relevante para a análise de padrões.
    X = df.drop('Empresa', axis=1)

    # Padronizar os dados para que tenham média zero e desvio padrão igual a um
    # Cria uma instância do StandardScaler para padronizar os dados.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Realizar a clusterização com k-means (vamos assumir 2 clusters)
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Adicionar os rótulos dos clusters aos dados originais
    df['Cluster'] = clusters

    # Visualizar os resultados
    print(df)

    # Plotar os clusters
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='red')
    plt.xlabel('Receita (padronizado)')
    plt.ylabel('Lucro Líquido (padronizado)')
    plt.title('Clusters de Empresas')
    plt.show()

if __name__ == '__main__':
    scikit_lesson()
