import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import collections
from sklearn.ensemble import RandomForestClassifier

def alg_balanceado():
    dados_diabetes = pd.read_csv('./dataset/diabetes.csv', sep=',')

    #Verificando dados do dataset

    print(dados_diabetes.info())

    ##Contando registros de cada classe
    print(dados_diabetes.groupby(['diabetes'])['diabetes'].count())

    #Separando conjunto de dados
    X_dados = dados_diabetes.drop('diabetes', axis=1).values
    y_dados = dados_diabetes['diabetes'].values

    #Função de balanceamento de dados

    def balanceamento_dados(X_dados, y_dados):
        undersample = RandomUnderSampler(random_state=42)
        X_under, y_under = undersample.fit_resample(X_dados,y_dados)

        tl = TomekLinks(sampling_strategy='all')
        X_under, y_under = tl.fit_resample(X_under,y_under)

        return X_under, y_under

    #Aplicando técnica de balanceamento na base de dados

    X_dados_balanceados, y_dados_balanceados = balanceamento_dados(X_dados, y_dados)

    #Verificando balanceamento realizado

    print(len(X_dados_balanceados), len(y_dados_balanceados))

    ##Outra forma de verificar o balanceamento
    print(collections.Counter(y_dados_balanceados))

    #Separação de bases de treino e de teste

    def separa_treino_teste(X_dados_balanceados, y_dados_balanceados):
        X_train, X_test, y_train, y_test = train_test_split(X_dados_balanceados,
                                                            y_dados_balanceados,
                                                            random_state=42,
                                                            test_size=0.2)

        return X_train, X_test, y_train, y_test

    #Separando conjuntos de treino e teste
    X_train, X_test, y_train, y_test = separa_treino_teste(X_dados_balanceados, y_dados_balanceados)

    print("conjuntos de treino")
    print(X_train)
    print(y_train)
    print("conjuntos de teste")
    print(X_test)
    print(y_test)

    #Criando modelo de RandomForest

    # Criando modelo Random Forest

    ##Criação classificador Random Forest
    classifier = RandomForestClassifier(random_state=42)

    ##Criação do modelo de classificação
    modelo = classifier.fit(X_train, y_train)

    ##Realizando predições
    y_pred = modelo.predict(X_test)
    print("y_pred")
    print(y_pred)

    # Features Importance

    print(pd.DataFrame(modelo.feature_importances_, columns=['importancia'],
                       index=['gravidez', 'glucose', 'pressao_sanguinea', 'espessura_pele', 'insulina', 'imc',
                              'predisposicao_diabetes', 'idade']).sort_values('importancia', ascending=False))

    # Cálculo da acurácia do modelo
    acuracia = accuracy_score(y_test, y_pred)

    print("precisão: ", acuracia)

    #Criando matriz de confusão

    cm = confusion_matrix(y_test, y_pred)

    #Visualizando matriz de confusão

    labels=['Não possui diabetes', 'Possui diabetes']

    plt.figure(figsize=(6,3))
    sns.heatmap(data=cm,
                annot=True,
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)

    plt.xlabel('Valor Previsto')
    plt.ylabel('Valor Real')
    plt.title("Matriz de Confusão")
    plt.show()

    #Calculando métricas de avaliação do modelo
    ##Gerando relatório de classificação

    report = classification_report(y_test, y_pred)
    print(report)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    alg_balanceado()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
