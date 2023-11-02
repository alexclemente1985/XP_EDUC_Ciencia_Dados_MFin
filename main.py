import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import collections
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz


def desafio():
    dados_carros = pd.read_csv('./dataset/cars_efficient.csv', sep=',')

    print(dados_carros.info())
    print(dados_carros.describe())
    print(dados_carros.shape)
    print("dados duplicados")
    print(dados_carros[dados_carros.duplicated()]) #retorna empty (sem dados duplicados)
    dados_carros.drop_duplicates(inplace=True)
    print(dados_carros.shape) #shape não alterou linhas e colunas restantes

    print("análise de dados ausentes")
    print(dados_carros.isnull().sum()) #cubinches 2 | weightlbs 3
    print("Cubic Inches nulos (2 formas de obter)")
    print(dados_carros[dados_carros['cubicinches'].isna()]) #retorna dados de cubicinches nulos
    print(dados_carros.query('cubicinches.isna()'))

    print("Weight lbs nulos (2 formas de obter)")
    print(dados_carros[dados_carros['weightlbs'].isna()]) #retorna dados de weightlbs nulos
    print(dados_carros.query('weightlbs.isna()'))

    print("contagem de registros de cada classe")
    print(dados_carros.groupby(['brand'])['brand'].count()) #Evidência de desbalanceamento -> 169 para eficiencia 0 | 99 para eficiencia 1

    #tratamento dados nulos
    print("remoção de dados nulos")
    print(dados_carros.groupby(['cubicinches'])['cubicinches'].count())
    print(dados_carros.groupby(['weightlbs'])['weightlbs'].count())

    #media_cubicinches = round(dados_carros['cubicinches'].mean(),2)
    #print("média cubicinches ", media_cubicinches)
    #dados_carros.fillna(value=media_cubicinches, inplace=True)
    #print("lista cubicinches nulos")
    #print(dados_carros.query('cubicinches.isna()')) #sem cubicinches nulo

    #media_weightlbs = round(dados_carros['weightlbs'].mean(), 2)
    #print("média weightlbs ", media_weightlbs)
    #dados_carros.fillna(value=media_weightlbs, inplace=True)
    #print("lista weightlbs nulos")
    #print(dados_carros.query('weightlbs.isna()')) #sem weightlbs nulo

    dados_carros.dropna(subset=['cubicinches'],inplace=True)
    dados_carros.dropna(subset=['weightlbs'],inplace=True)
    print("Verificação da remoção de nulos nas colunas cubicinches e weightlbs")
    print(dados_carros.query('cubicinches.isna()'))
    print(dados_carros.query('weightlbs.isna()'))


    print("Verificando balanceamento dos dados")
    X_dados = dados_carros.drop('brand', axis=1).values
    y_dados = dados_carros['brand'].values

    print(X_dados)
    print(y_dados)

    print(collections.Counter(y_dados)) #não está balanceado (quantidade de US é muito maior que a dos demais)

    #função para balanceamento
    def balanceamento_dados(X_dados, y_dados):
        under_sample = RandomUnderSampler(random_state=42)

        #fit dos dados
        X_under, y_under = under_sample.fit_resample(X_dados, y_dados)

        tl = TomekLinks(sampling_strategy='all')
        X_under, y_under = tl.fit_resample(X_under, y_under)

        return X_under, y_under

    X_dados_balanceados, y_dados_balanceados = balanceamento_dados(X_dados, y_dados)

    #Vericando balanceamento
    print(collections.Counter(y_dados_balanceados))

    #Criando função para separar conjunto de treinamento do de teste

    def separa_treino_teste(X_dados, y_dados):
    #def separa_treino_teste(X_dados_balanceados, y_dados_balanceados):
        X_train, X_test, y_train, y_test = train_test_split(X_dados,
                                                            y_dados,
                                                            random_state= 42,
                                                            test_size=0.3)

        return X_train, X_test, y_train, y_test

    #Separando conjunto de treino e teste
    X_train, X_test, y_train, y_test = separa_treino_teste(X_dados,y_dados)#(X_dados_balanceados,y_dados_balanceados)

    #criando árvore de decisão

    algoritmo_arvore = tree.DecisionTreeClassifier()

    modelo = algoritmo_arvore.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    #Verificando as importâncias

    nome_features = ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year', 'is_efficient']
    nome_features_2 = ['cubicinches', 'weightlbs', 'time-to-60', 'mpg', 'year', 'hp', 'cylinders']

    importancias_df = pd.DataFrame(modelo.feature_importances_, columns=['importancia'],
                                   index=nome_features).sort_values('importancia', ascending=False)

    print("Importâncias")
    print(importancias_df)

    #Verificando e analisando classes do modelo
    print(modelo.classes_)


    nome_classes = modelo.classes_

    #criando função para visualizar árvore
    def visualiza_arvore(modelo):
        arvore = tree.export_graphviz(modelo,
                                      feature_names=nome_features,
                                      class_names=nome_classes,
                                      rounded=True,
                                      special_characters=True,
                                      filled=True)

        graph = graphviz.Source(arvore)
        return graph

    #visualizando a árvore
    os.environ["PATH"] += os.pathsep + "D:/Program Files/Graphviz/bin"
    visualiza_arvore(modelo).view()

    #Calculando a precisão do modelo

    print("Precisão do modelo: ",accuracy_score(y_test, y_pred))

    #Criando função que gera features de importância para o modelo

    def gera_features_importantes(modelo):
        df = pd.DataFrame(modelo.feature_importances_, columns=['importancia'],
                                   index=nome_features).sort_values('importancia', ascending=False)

        return df

    #Criando função que gera árvore de decisão
    PATH_IMG = r'./dataset'

    def cria_modelo(X_train, y_train, y_test, tamanho_arvore):
        algoritmo_arvore = tree.DecisionTreeClassifier(max_depth=tamanho_arvore, random_state=42)

        modelo = algoritmo_arvore.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        arvore = visualiza_arvore(modelo)

        ###Salva arquivo pdf
        arvore.render(f'{PATH_IMG}/modelo_tamanho_n{tamanho_arvore}')

        print(f'Features com maior importância para o modelo tamanho n = {tamanho_arvore}:')
        print(f'{gera_features_importantes(modelo)}')
        print(f'o valor da acuracia para tamanho n = {tamanho_arvore} é: ', accuracy_score(y_test, y_pred))

        return arvore

    #Gerando resultados com poda da árvore

    #tamanho_arvore= 2
    #arvore = cria_modelo(X_train,y_train,y_test,tamanho_arvore)
    #arvore.view()

    #Histograma da coluna potências
    plt.figure(figsize=(20,7))
    plt.title('Distribuição de potências em HP', size=20)
    sns.histplot(
        data=dados_carros,
        x='hp',
        bins=25
    )
    plt.show()

    #Boxplot de brands e potências
    plt.figure(figsize=(20,7))
    plt.title('Gráfico Boxplot Marcas x Potências em HP', size=20)

    sns.boxplot(
        data=dados_carros,
        x='hp',
        y='brand',
        orient='h'
    )

    plt.show()

    #Questão 7 criando arvore com apenas parâmetro random_state=42
    def cria_modelo_sem_poda(X_train, y_train, y_test):
        algoritmo_arvore = tree.DecisionTreeClassifier(random_state=42)

        modelo = algoritmo_arvore.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        arvore = visualiza_arvore(modelo)

        ###Salva arquivo pdf
        arvore.render(f'{PATH_IMG}/modelo_tamanho_total')

        print(f'Features com maior importância para o modelo arvore total:')
        print(f'{gera_features_importantes(modelo)}')
        print('o valor da acuracia é: ', accuracy_score(y_test, y_pred))

        return arvore

    arvore_sem_poda = cria_modelo_sem_poda(X_train,y_train,y_test)
    arvore_sem_poda.view()





    arvore_tamanho_1 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=1)
    arvore_tamanho_1.view()

    arvore_tamanho_2 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=2)
    arvore_tamanho_2.view()

    arvore_tamanho_3 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=3)
    arvore_tamanho_3.view()

    arvore_tamanho_4 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=4)
    arvore_tamanho_4.view()

    arvore_tamanho_5 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=5)
    arvore_tamanho_5.view()

    arvore_tamanho_6 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=6)
    arvore_tamanho_6.view()

    arvore_tamanho_7 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=7)
    arvore_tamanho_7.view()

    arvore_tamanho_8 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=8)
    arvore_tamanho_8.view()

    arvore_tamanho_9 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=9)
    arvore_tamanho_9.view()

    arvore_tamanho_10 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=10)
    arvore_tamanho_10.view()

    arvore_tamanho_11 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=11)
    arvore_tamanho_11.view()

    arvore_tamanho_12 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=12)
    arvore_tamanho_12.view()

    arvore_tamanho_13 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=13)
    arvore_tamanho_13.view()

    arvore_tamanho_14 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=14)
    arvore_tamanho_14.view()

    arvore_tamanho_15 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=15)
    arvore_tamanho_15.view()

    arvore_tamanho_16 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=16)
    arvore_tamanho_16.view()

    arvore_tamanho_17 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=17)
    arvore_tamanho_17.view()

    arvore_tamanho_18 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=18)
    arvore_tamanho_18.view()

    arvore_tamanho_19 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=19)
    arvore_tamanho_19.view()

    arvore_tamanho_20 = cria_modelo(X_train, y_train, y_test, tamanho_arvore=20)
    arvore_tamanho_20.view()

    report = classification_report(y_test, y_pred)
    print(report)





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    desafio()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
