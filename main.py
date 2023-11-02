import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diretorio_arquivo = r'dataset/perfil_clientes.csv'

#Leitura do arquivo csv, separando registros por ponto-e-virgula
dataframe = pd.read_csv(diretorio_arquivo, sep=';')

#Retornando número de linhas e colunas da planilha
dfLinesCols = dataframe.shape
print('dataframe shape: {}'.format(dfLinesCols))

#Retorna uma tabela com informações úteis para Análise Descritiva
dataframe.info()
## classe_trabalho possui dados faltantes (menos que o total de registros)

#Visualização das 5 primeiras linhas
print(dataframe.head())

#Visualização das 5 últimas linhas
print(dataframe.tail())
## presença de valor NaN na coluna classe_trabalho

#Análise Estatística Descritiva - distribuição das variáveis numéricas no dataset
print(dataframe.describe())

#Visualização de linhas como colunas e colunas como linhas (plus: round para arredondar casas decimais)
print(round(dataframe.describe().transpose()))

#Exploração de variáveis categóricas (valores únicos, frequência e tabela cruzada)

##Valores Únicos
contagem_categorias = dataframe['estado_civil'].value_counts()
print(contagem_categorias)

##Mostrando gráfico de barras horizontal
contagem_categorias.plot.barh()
plt.show()

##Mostrando gráfico de barras vertical
contagem_categorias.plot.bar()
plt.show()

##Tabela cruzada (comparação entre duas variáveis)
dframeCrossTab = pd.crosstab(dataframe['estado_civil'], dataframe['região'])
print(dframeCrossTab)

##Agregação de Dados
mediaIdadeEstadosCivis = dataframe.groupby('estado_civil')['idade'].mean()
maxIdadeEstadosCivis = dataframe.groupby('estado_civil')['idade'].max()

print(mediaIdadeEstadosCivis)
print(maxIdadeEstadosCivis)

#Após análise descritiva, realizar a limpeza dos dados
##Trazendo dados duplicados
print("Dados duplicados")
print(dataframe[dataframe.duplicated()])

##Eliminando duplicatas
###inplace=True elimina as duplicatas e salva no objeto dataframe para próximo uso do mesmo
dataframe.drop_duplicates(inplace=True)
###verificando a remoção
print("Linhas e colunas restantes:")
print(dataframe.shape)

###Análise de dados ausentes
print(dataframe.isnull().sum())

###classe_trabalho tem muitos ausentes
print(dataframe[dataframe['classe_trabalho'].isna()])

###Pesquisas na planilha
####Onde valores retornam nulo
print(dataframe.query('qtde_filhos.isna()'))
print(dataframe.query('salario.isna()'))

####Busca com várias condições
print(dataframe.query('idade == 20 and estado_civil == "Solteiro" and UF == "MG"'))

####Média de idades em uma busca
print('Média de idade na busca de solteiros em MG: {}'.format(round(dataframe.query('estado_civil == "Solteiro" and UF == "MG"')['idade'].mean())))

####Correção de dados ausentes (categóricos)
#####Imputação por Moda
######Correção de classe_trabalho
moda = dataframe['classe_trabalho'].mode()[0]
######Imputação por Moda: substitui valores nulos pela moda da coluna
print('Antes da remoção dos dados nulos classe trabalho: ',dataframe.query('classe_trabalho.isna()'))
dataframe['classe_trabalho'].fillna(moda, inplace=True)
######Visualização das alterações
print('Após a remoção dos dados nulos classe trabalho: ',dataframe.query('classe_trabalho.isna()'))


####Correção de dados numéricos
#####Preferência pelo tratamento por regras de negócio
#####dropando dados ausentes em coluna
print('Antes da remoção dos dados nulos: ',dataframe.query('qtde_filhos.isna()'))
dataframe.dropna(subset=['qtde_filhos'], inplace=True)
print('Depois da remoção dos dados nulos: ',dataframe.query('qtde_filhos.isna()'))

#####Correção coluna salário
media = round(dataframe['salario'].mean(),2)
print('Antes da remoção dos dados nulos salario: ')
print(dataframe.query('salario.isna()'))
dataframe['salario'].fillna(value=media, inplace=True)
print('Após a remoção dos dados nulos salario: ')
print(dataframe.query('salario.isna()'))
print('soma de salários nulos')
print(dataframe.isna().sum())


#Visualizações Gráficas dos dados
##Histograma: identifica padrões, tendências e características dos dados

plt.figure(figsize=(20,7))
plt.title('Distribuição das Idades', size=20)

###Criação do histograma
sns.histplot(
    data=dataframe,
    x='idade',
    bins=25
)
plt.show()

####Distribuição dos salários
plt.figure(figsize=(20,7))
plt.title('Distribuição dos Salários', size=20)

sns.histplot(
    data=dataframe,
    x='salario',
    bins=25
)
plt.show()

###Criação de Boxplot (gráfico de caixa)
####Permite identificar outliers, mediana, etc

#####Distribuição de dados para estado_civil = "Solteiro"
plt.figure(figsize=(5,3))
plt.title('Gráfico de Boxplot', size=20)

sns.boxplot(
    data=dataframe.query('estado_civil == "Solteiro"'),
    x='idade',
    orient='h'
)
plt.show()

idades_solteiros = dataframe.query('estado_civil == "Solteiro"')
print('Idades solteiros')
print(idades_solteiros['idade'].describe())

#####Boxplot relacionando idades com classes de trabalho
plt.figure(figsize=(7,5))
plt.title('Gráfico de Boxplot', size=10)

sns.boxplot(
    data=dataframe,
    x='idade',
    y='classe_trabalho',
    orient='h',
    palette='pastel'
)
plt.show()

###Criação de gráfico de Dispersão
####Preparação de dados para dispersão

media_salario_anos_estudo = dataframe.groupby(['anos_estudo'])['salario'].mean()

media_estudo = pd.DataFrame(media_salario_anos_estudo)
media_estudo.reset_index(inplace=True)
print(media_estudo)

####Criando o gráfico

plt.figure(figsize=(7,5))
plt.title('Gráfico de Dispersão', size=10)
sns.scatterplot(
    data=media_estudo,
    x='anos_estudo',
    y='salario'
)
#####Mudando o nome dos eixos
plt.xlabel("Anos de Estudo")
plt.ylabel("Média de Salários")
plt.show()

###Criação de Gráfico de Barras

plt.figure(figsize=(10,7))
plt.title('Gráfico de Barras', size=10)

sns.countplot(
    data=dataframe,
    x='escolaridade',
)
####Rotação para melhorar visualização dos valores no eixo x
plt.xticks(rotation=45, ha="right")
plt.show()

####Análise comparativa por meio do gráfico de Barras

plt.figure(figsize=(10,7))
plt.title('Gráfico de Barras', size=10)

sns.barplot(
    data=media_estudo,
    x='anos_estudo',
    y='salario'
)

plt.xticks(rotation=45, ha="right")
plt.show()

###Exploração de Correlação entre Variáveis
#### Valores próximos a 1 -> correlação forte e positiva | Valores próximos a -1 -> correlação forte e negativa | Valores próximos a 0 -> sem correlação linear aparente

colunas = ['anos_estudo','salario','qtde_filhos','idade']
print(round(dataframe[colunas].corr(),2))

###Criação de Gráfico Heatmap
#### Ideal para visualização de distribuição e intensidade de valores em uma matriz

plt.figure(figsize=(10,7))
plt.title('Gráfico Heatmap', size=10)

dados_correlacao = round(dataframe[colunas].corr(),2)

sns.heatmap(
    data=dados_correlacao,
    cmap='coolwarm',
    linewidths=0.1,
    linecolor='white',
    annot=True
)

plt.show()

###Explorando Formas de Agregação

anos_estudo = dataframe.groupby('escolaridade')['anos_estudo'].mean()
group_by_regiao_sexo = dataframe.groupby(['região','sexo']).agg(
    total=('sexo','count'),
    media_idade=('idade','mean')
)

group_by_regiao_uf_sexo = dataframe.groupby(['região','UF','sexo']).agg(
    total=('sexo','count'),
    media_idade=('idade','mean')
)

print(group_by_regiao_sexo)
print(group_by_regiao_uf_sexo)
