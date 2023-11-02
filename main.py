from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import warnings

warnings.simplefilter('ignore')

#Coletando dados da base de risco

df_risco = pd.read_csv('./dataset/base_risco.csv', sep=';', encoding='latin1')

print(df_risco)

#Selecionando colunas que vamos utilizar para criar o indicador

colunas_selecionadas = ['renda', 'tipo_divida', 'fiador', 'historico_credito','risco']

#Preparação dos dados
## Naive Bayes - transformar variáveis categóricas em identificadores numéricos

LE = LabelEncoder()

###Criando nova coluna baseado no encoder criado

for coluna in colunas_selecionadas:
    df_risco[f'id_{coluna}'] = LE.fit_transform(df_risco[coluna])

###Organizando as colunas no dataframe

df_risco = df_risco[['id_renda','renda', 'id_tipo_divida', 'tipo_divida', 'id_fiador', 'fiador', 'id_historico_credito', 'historico_credito', 'risco']]

print("novo df_risco")
print(df_risco.head())

###Selecionando apenas as colunas do modelo (os IDs que representam as variáveis categóricas)

colunas_modelo = ['id_renda', 'id_tipo_divida', 'id_fiador', 'id_historico_credito']

print(df_risco[colunas_modelo])

###Separando features do target
x_dados = df_risco[colunas_modelo]
y_dados = df_risco['risco']

print("x_dados")
print(x_dados)
print("y_dados")
print(y_dados)

###Criação do modelo Naive Bayes

model = GaussianNB()

####Treinamento do modelo

model.fit(x_dados.values, y_dados.values)
print("model")
print(model)

####Visualizando dados do modelo
print(model.classes_)
####Visualizando quantidades de cada classe
print(model.class_count_)
####Probabilidades de cada classe
print(model.class_prior_)
####Score do modelo (pontuação de acerto do modelo)
print(model.score(x_dados.values, y_dados.values))
#####Verificando funcionamento da pontuação
print(model.predict(X=x_dados))

######Avaliando com própria classe de dados
df_risco['classe_predita'] = model.predict(x_dados.values)

print(df_risco)

####Novo registro
##### renda acima de 40000 | tipo de dívida BAIXA | POSSUI fiador | histórico de crédito RUIM

dados_cliente = [[0, 1 ,1, 2]]
print("Tipo de risco do cliente acima:")
print(model.predict(dados_cliente))

#####Criação de novos registros para classificação do algoritmo

novos_registros = [
    [1,0,1,0], #entre 13 e 40K | dívida alta | possui fiador | crédito bom
    [2,1,0,0], #menor que 13k | dívida baixa | não possui fiador | crédito bom
    [0,0,0,0], #acima de 40K | dívida alta | não possui fiador | crédito bom
    [1,1,1,1],  #entre 13 e 40K | dívida baixa | possui fiador | não possui
    [0,1,1,2]  #acima de 40K | dívida baixa | possui fiador | crédito ruim
]

df_predicao = pd.DataFrame(novos_registros, columns=[colunas_modelo])

print(df_predicao)

####Aplicação do algoritmo para um novo conjunto de dados
df_predicao['classe_predita'] = model.predict((df_predicao.values))

print(df_predicao)
