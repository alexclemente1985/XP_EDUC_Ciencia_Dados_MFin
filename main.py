import pandas as pd
import yfinance as yf
import numpy as np
import warnings

import plotly.express as px  #Criação de graficos dinâmnicos
import plotly.offline as py
import plotly.graph_objects as go #Para criação e concatenização de graficos

from plotly.subplots import make_subplots
from pandas_datareader import data as pdr
from sklearn.cluster import KMeans

import pandas_datareader, sklearn, plotly
warnings.filterwarnings("ignore")
yf.pdr_override()

#Etapas da implementação do algoritmo K-means para conjunto financeiro de ativos

# 1) Coletar dados do dataset com a lista de ativos alvos
# 2) Coletar dados online do yahoo finances
# 3) Realizar o enriquecimento de dados
# 4) Aplicar modelo ML de K-means
# 5) Visualizar os resultados obtidos

def aula_interativa_2():
    print(f''' Relação das bibliotecas e suas respectivas versões
    ------------------------------------------------------------
      pandas: {pd.__version__}
      numpy: {np.__version__}
      yahoo ficances: {yf.__version__}
      pandas_datareader: {pandas_datareader.__version__}
      plotly: {plotly.__version__}
      sklearn: {sklearn.__version__}
    ------------------------------------------------------------
            '''
          )
    #1) Coletar dados do dataset com a lista de ativos alvos
    dados_ativos = pd.read_csv(r'./dataset/ativos.csv',
                               sep=';', encoding='latin1'
                               )

    print(dados_ativos.head())

    #2) Coletar dados online do yahoo finances

    lista_df = []

    for ativo in dados_ativos['ativo']:
        try:
            df = pdr.get_data_yahoo(ativo,
                                    start="2022-01-01",
                                    end="2022-12-31")

            #Criação de coluna com o nome do ativo
            df['ativo'] = ativo
            #Adição dos dados em uma lista
            lista_df.append(df)
        except (e):
            print(f'Foi encontrado um erro no ativo: {ativo} - erro: {e}')

    df_ativos = pd.concat(lista_df)

    print(f'O dataset coletado possui {len(df_ativos)} linhas')

    print(df_ativos.head())

    #3) Realizar o enriquecimento de dados
    ## Uso de coluna Adj Close (fechamento ajustado), criando duas novas features

    ###Fechamento ajustado: é o preço de fechamento após ajustes para todas as divisões e distribuições de dividendos aplicáveis.

    ###Cálculo da diferença de percentual da coluna "Adj Close" para cada linha
    df_ativos['dif_percentual'] = df_ativos['Adj Close'].pct_change()

    print(df_ativos.head())

    ###Enriquecimento da base de dados
    ####Retorno: indica qual retorno financeiro o ativo proporciona para a pessoa
    ####Volatividade: identifica o quanto que o ativo oscila; calculado com o desvio padrão

    retorno = (df_ativos.groupby(['ativo'])
               .agg(retorno=('dif_percentual','mean'))*252)

    volatividade = (df_ativos.groupby(['ativo'])
                    .agg(volatividade = ('dif_percentual', 'std'))*np.sqrt(252))

    analise_ativos = pd.merge(retorno, volatividade, how='inner', on='ativo')

    ###Resetando o index da nova tabela e visualizando dados gerados
    analise_ativos.reset_index(inplace=True)
    print("Análise de ativos (tabela enriquecida)")
    print(analise_ativos.head())

    ###Função para calcular valores de WCSS
    def calcular_wcss(dados_ativos):
        wcss = []

        for k in range(1,11):
            kmeans = KMeans(n_clusters=k, random_state=0, init='k-means++')
            kmeans.fit(X=dados_ativos)
            wcss.append(kmeans.inertia_)

        return wcss

    ### Seleção das variáveis para uso na clusterização
    dados_ativos = analise_ativos[['retorno','volatividade']]

    ### Cálculo do WCSS
    wcss_ativos = calcular_wcss(dados_ativos)

    ### Visualização dos dados obtidos do WCSS
    for i in range(len(wcss_ativos)):
        print(f'O cluster {i} possui valor de WCSS de: {wcss_ativos[i]}')

    ### Gráfico WCSS x Número de Cluster (gráfico de cotovelo)

    grafico_wcss = px.line(x=range(1,11),
                           y=wcss_ativos)

    fig = go.Figure(grafico_wcss)

    fig.update_layout(title='Calculando o WCSS',
                      xaxis_title='Número de clusters',
                      yaxis_title='Valor do Wcss',
                      template='plotly_white'
                      )

    fig.show()

    #4) Aplicar modelo ML de K-means
    ## Análise do gráfico WCSS x No Clusters -> número de clusters = 5 seria o ideal

    kmeans_ativos = KMeans(n_clusters=5,
                           random_state=0,
                           init='k-means++')

    analise_ativos['cluster'] = kmeans_ativos.fit_predict(dados_ativos)

    print('Análise ativos (k-means)')
    print(analise_ativos.head())

    #5) Visualização dos resultados obtidos

    fig = make_subplots(rows=1, cols= 1, shared_xaxes=True, vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=analise_ativos['volatividade'],
                             y=analise_ativos['retorno'],
                             name="", mode="markers",
                             text=analise_ativos['ativo'],
                             marker=dict(size=14, color=analise_ativos["cluster"])))

    fig.update_layout(height=600, width=900,
                      title_text='Análise de Clusters',
                      xaxis_title='Volatividade',
                      yaxis_title='Retorno')

    fig.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    aula_interativa_2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
