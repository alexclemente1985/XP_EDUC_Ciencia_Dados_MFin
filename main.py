from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import graphviz
import pandas as pd
import collections
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def arvore_decisao(name):
    dados_diabetes = pd.read_csv('dataset/diabetes3.csv',sep=",")

    print(dados_diabetes.info())

    print(dados_diabetes.head())

    #Verificar se a pessoa tem diabetes ou não

    ##Verificando dados nulos
    print(dados_diabetes.isna().sum()) ###Sem dados nulos

    ##Contar número de registros de cada classe
    print(dados_diabetes.groupby(['resultado'])['resultado'].count())

    ##Separando conjunto de dados de features e target

    #Z_dados = dados_diabetes['diabetes'].values
    X_dados = dados_diabetes.drop('resultado', axis=1).values
    y_dados = dados_diabetes['resultado'].values

    print("X_dados")
    print(X_dados)
    print("Y_dados")
    print(y_dados)

    ##Criando função para balancear os dados
    def balanceamento_dados(X_dados, y_dados):
        under_sample = RandomUnderSampler(random_state=42)

        ###Fazendo o fit dos dados
        X_under, y_under = under_sample.fit_resample(X_dados, y_dados)

        tl = TomekLinks(sampling_strategy='all')
        X_under, y_under = tl.fit_resample(X_under, y_under)

        return X_under, y_under

    X_dados_balanceados, y_dados_balanceados = balanceamento_dados(X_dados, y_dados)

    print(len(X_dados_balanceados), len(y_dados_balanceados))

    ##Verificando lista com dados balanceados -> separa certo os 0's e 1's
    print(collections.Counter(y_dados_balanceados))

    print("y balanceados")
    print(y_dados_balanceados)

    ##Criando funão para separar conjunto de treinamento do de teste

    def separa_treino_teste(X_dados_balanceados, y_dados_balanceados):
        X_train, X_test, y_train, y_test = train_test_split(X_dados_balanceados,
                                                                  y_dados_balanceados,
                                                                  random_state=42,
                                                                  test_size=0.2) #80% para treino e 20% para teste

        return X_train, X_test, y_train, y_test

    ##Separando conjuntos de treino e teste
    X_train, X_test, y_train, y_test = separa_treino_teste(X_dados_balanceados, y_dados_balanceados)

    ##Criando árvore de decisão

    algoritmo_arvore = tree.DecisionTreeClassifier()

    modelo = algoritmo_arvore.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("y_pred")
    print(y_pred)
    print(modelo.feature_importances_)

    print(dados_diabetes.columns)

    importancias_df = pd.DataFrame(modelo.feature_importances_, columns=['importancia'],
                 index=['gravidez', 'glicose', 'pressao_sanguinea', 'espessura_pele',
                        'insulina', 'bmi', 'funcao_pedigree_diabetes', 'idade']).sort_values('importancia', ascending=False)

    print(importancias_df)

    ##Verificando e analisando as classes do modelo
    print(modelo.classes_)

    nome_features = ['gravidez', 'glicose', 'pressao_sanguinea', 'espessura_pele',
                        'insulina', 'bmi', 'funcao_pedigree_diabetes', 'idade']
    nomes_classes = ['NÃO DIABETICO', 'DIABETICO']

    ##Criando função para visualização da árvore gerada
    def visualiza_arvore(modelo):
        arvore = tree.export_graphviz(modelo,
                                      feature_names=nome_features,
                                      class_names=nomes_classes,
                                      rounded=True,
                                      special_characters=True,
                                      filled=True)

        graph = graphviz.Source(arvore)
        return graph

    ##Visualizando a árvore completa
    os.environ["PATH"] += os.pathsep + "D:/Program Files/Graphviz/bin"
    visualiza_arvore(modelo).view()

    ##Visualizando a estrutura da árvore
    #tree.plot_tree(modelo)
    #plt.show()

    ##Calculando a precisão do modelo

    print(accuracy_score(y_test, y_pred))

    ##Criando função que gera features de importância para o modelo

    def gera_features_importantes(modelo):
        df = pd.DataFrame(modelo.feature_importances_, columns=['importancia'],
                          index=['gravidez', 'glicose', 'pressao_sanguinea', 'espessura_pele',
                        'insulina', 'bmi', 'funcao_pedigree_diabetes', 'idade']).sort_values('importancia', ascending=False)

        return df

    print(gera_features_importantes(modelo))

    ##Criando função que gera árvore de decisão
    ###Cria diretório para salvar imagens das árvores geradas

    PATH_IMG = r'./dataset'

    def cria_modelo(X_train, y_train, y_test, tamanho_arvore):
        algoritmo_arvore = tree.DecisionTreeClassifier(max_depth=tamanho_arvore, random_state=42)

        modelo = algoritmo_arvore.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        arvore = visualiza_arvore(modelo)

        ###Salva arquivo pdf
        arvore.render(f'{PATH_IMG}/modelo_tamanho_n{tamanho_arvore}')

        print(f'Features com maior importância para o modelo:'
              f'{gera_features_importantes(modelo)}')
        print('o valor da acuracia é: ', accuracy_score(y_test, y_pred))

        return arvore

    ##Gerando resultados com poda de árvore

    tamanho_arvore = 2
    arvore = cria_modelo(X_train, y_train, y_test, tamanho_arvore)
    arvore.view()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arvore_decisao('PyCharm')
