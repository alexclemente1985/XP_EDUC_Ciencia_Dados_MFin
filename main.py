import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

diretorio_arquivo = r'dataset/dados_licencas_medicas_utf8.csv'

#Leitura do arquivo csv, separando registros por ponto-e-virgula
dataframe = pd.read_csv(diretorio_arquivo, sep=';')

#Questão 1 - dados duplicados
print('Dados duplicados:')
##print(dataframe[dataframe.duplicated()])
print('antes remoção')
##print(dataframe.shape)
##dataframe.drop_duplicates(inplace=True)
print("Linhas e colunas restantes:")
##print(dataframe.shape)

#Questão 2
print(dataframe.isnull().sum())
#print(dataframe.query('qtde_filhos.isna()'))
#print(dataframe.query('salario.isna()'))

#Questão 3
dataframe.drop_duplicates(inplace=True)

media_salario = dataframe['salario'].mean()
dataframe['salario'].fillna(value=media_salario, inplace=True)

moda_qtd_filhos = dataframe['qtd_filhos'].mode()[0]
dataframe['qtd_filhos'].fillna(value=moda_qtd_filhos, inplace=True)

moda_estado_civil = dataframe['estado_civil'].mode()[0]
dataframe['estado_civil'].fillna(value=moda_estado_civil, inplace=True)

print('Média salarial após tratamento dos dados: ',round(dataframe['salario'].mean(),2))

print('Profissional que prestou menor número de assistências: ')
print(dataframe.groupby(['nome_medico'])['nome_colaborador'].count())


print(dataframe.query('estado_civil=="Casado(a)" and estado_colaborador=="Espírito Santo"'))

print(dataframe.query('sexo_colaborador=="Masculino"')['salario'].mean())

minas = dataframe.query('estado_colaborador=="Minas Gerais"')
menos = minas['motivo_licença'].value_counts()
print('Menos MG', menos)

