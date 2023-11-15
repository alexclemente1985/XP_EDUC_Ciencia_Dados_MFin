import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import mplfinance as mpf
import random


def matplotlib_lesson():
    # Dados fictícios de preços de ações ao longo de 5 dias
    dias = [1, 2, 3, 4, 5]
    precos = [100, 120, 115, 130, 125]

    # Plotar o gráfico de linhas
    plt.plot(dias, precos, marker='o', linestyle='-', color='b')
    # (dias,precos)   = (x,y) do gráfico
    # marker='o'      >> especifica que queremos pontos marcados nos dados
    # linestyle='-'   >> especifica que as linhas devem ser contínuas (outra opção seria ='--')
    # color='b'       >> especifica a cor do gráfico em azul (b = blue)
    plt.xlabel('Dia')  # rótulo do eixo X
    plt.ylabel('Preço')  # rótulo do eixo Y
    plt.title('Evolução do Preço da Ação')  # Título do Gráfico
    plt.grid(True)  # Linhas de grades no gráfico
    plt.show()  # Exibe o Gráfico

    # Gráfico de barras

    # Dados fictícios de preços de ações A e B no último mês
    acoes = ['A', 'B']
    precos_mes_anterior = [150, 130]
    precos_mes_atual = [160, 140]

    # Plotar o gráfico de barras
    plt.bar(acoes, precos_mes_anterior, label='Mês Anterior', color='b', width=0.4)
    plt.bar(acoes, precos_mes_atual, label='Mês Atual', color='g', width=0.4, bottom=precos_mes_anterior)
    plt.xlabel('Ação')  # rótulo do eixo X
    plt.ylabel('Preço')  # rótulo do eixo Y
    plt.title('Comparação de Preços de Ações')  # Título do Gráfico
    plt.legend()  # é usado para mostrar as legendas
    plt.show()  # Exibe o Gráfico

    # Barras empilhadas

    # Dados de exemplo - Contribuição de cada setor para o total da carteira de investimentos (%)
    setores = ['Tecnologia', 'Saúde', 'Financeiro', 'Indústria']
    ano_2019 = [15, 25, 10, 20]
    ano_2020 = [20, 20, 15, 25]
    ano_2021 = [25, 15, 20, 15]

    # Plotar o Gráfico de Barras Empilhadas
    plt.figure(figsize=(10, 6))

    # Criar as barras empilhadas usando plt.bar()
    plt.bar(setores, ano_2019, label='2019')
    plt.bar(setores, ano_2020, label='2020', bottom=ano_2019)
    plt.bar(setores, ano_2021, label='2021', bottom=[a + b for a, b in zip(ano_2019, ano_2020)])
    # Utilizamos a função zip() para somar as listas ano_2019 e ano_2020 elemento a elemento e, em seguida,
    # usamos a lista resultante como o argumento bottom para a terceira chamada de plt.bar(), que representa o ano 2021.

    plt.xlabel('Setores')  # rótulo do eixo X
    plt.ylabel('Contribuição para o Total (%)')  # rótulo do eixo Y
    plt.title('Contribuição de Setores para a Carteira de Investimentos')  # Título do Gráfico
    plt.legend()  # para mostrar uma legenda que representa os anos (2019, 2020, 2021).
    plt.grid(axis='y')  # adiciona linhas de grade somente no eixo y para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # Exibe o Gráfico

    # Gráfico de pizza

    # Composição da carteira de investimentos (%)
    setores = ['Tecnologia', 'Saúde', 'Financeiro', 'Indústria']
    composicao = [30, 25, 20, 25]

    # Plotar o gráfico de pizza
    plt.pie(composicao, labels=setores, autopct='%1.1f%%', startangle=90)
    # autopct='%1.1f%%'  >>  Formato de string que define como as % devem ser formatadas e exibidas no gráfico.
    # Neste caso, "%1.1f%%" significa que as porcentagens serão exibidas com uma casa decimal.
    # startangle=90  >> Define o ângulo inicial do gráfico de pizza. Neste caso, coloca 'Tecnologia' no topo do gráfico.
    plt.axis(
        'equal')  # Define a proporção de aspecto do gráfico como igual, desenha em forma de círculo, em vez de uma elipse.
    # Outras opções 'auto' , 'off' , 'scaled' , 'tight'
    plt.title('Composição da Carteira de Investimentos')  # Título do Gráfico
    plt.show()  # Exibe o Gráfico

    # Gráfico de dispersão

    # Dados de exemplo - Preços hipotéticos das ações de duas empresas ao longo do tempo
    tempo = list(range(1, 11))  # 10 pontos de dados no tempo
    empresa_a_precos = [random.uniform(100, 200) for _ in tempo]  # Preços da Empresa A
    empresa_b_precos = [random.uniform(80, 180) for _ in tempo]  # Preços da Empresa B

    # Plotar o Gráfico de Dispersão
    plt.figure(figsize=(8, 6))

    plt.scatter(tempo, empresa_a_precos, label='Empresa A', color='red', marker='o')
    plt.scatter(tempo, empresa_b_precos, label='Empresa B', color='blue', marker='x')

    plt.xlabel('Tempo')  # rótulo do eixo X
    plt.ylabel('Preço da Ação')  # rótulo do eixo Y
    plt.title('Relação entre os Preços das Ações de Empresa A e Empresa B ao Longo do Tempo')  # Título do Gráfico
    plt.legend()  # é usado para mostrar as legendas no gráfico
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # Exibe o Gráfico

    # Gráfico de áreas

    # Dados de exemplo - Preços hipotéticos das ações de duas empresas ao longo do tempo
    tempo = list(range(1, 11))  # 10 pontos de dados no tempo
    empresa_a_precos = [random.uniform(100, 200) for _ in tempo]  # Preços da Empresa A
    empresa_b_precos = [random.uniform(80, 180) for _ in tempo]  # Preços da Empresa B

    # Calcular a soma acumulada dos preços das ações para cada empresa utilizando list comprehension e a função sum()
    empresa_a_acumulado = [sum(empresa_a_precos[:i + 1]) for i in range(len(empresa_a_precos))]
    empresa_b_acumulado = [sum(empresa_b_precos[:i + 1]) for i in range(len(empresa_b_precos))]

    # Plotar o Gráfico de Áreas
    plt.figure(figsize=(8, 6))  # define o tamanho do gráfico

    plt.stackplot(tempo, empresa_a_acumulado, empresa_b_acumulado, labels=['Empresa A', 'Empresa B'], alpha=0.7)
    # plt.stackplot(x(tempo), y(lista valores para áreas), labels = rótulos às legendas, alpha = define a transparência das áreas )

    plt.xlabel('Tempo')  # Rótulo eixo X
    plt.ylabel('Soma Acumulada do Preço da Ação')  # Rótulo eixo Y
    plt.title('Soma Acumulada dos Preços das Ações de Empresa A e Empresa B ao Longo do Tempo')  # Título do Gráfico
    plt.legend(loc='upper left')  # é usado para mostrar as legendas no canto superior esquerdo do gráfico.
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # exibe o gráfico

    # Histograma

    # Dados de exemplo - Retornos diários hipotéticos de uma ação
    # Criação de uma lista (retornos_diarios) com 100 valores hipotéticos de uma ação
    num_dias = 100
    retornos_diarios = [random.normalvariate(0.01, 0.02) for _ in range(num_dias)]
    # Os valores são gerados aleatoriamente utilizando a função random.normalvariate(), que gera números aleatórios
    # de acordo com uma distribuição normal com média de 0.01 e desvio padrão de 0.02.

    # Plotar o Histograma
    plt.figure(figsize=(8, 6))  # define o tamanho do gráfico

    plt.hist(retornos_diarios, bins=20, density=True, edgecolor='black', alpha=0.7)
    # <lista dos retornos diários> dados a serem plotados
    # <bins> define o número de bins (intervalos) no histograma, que influencia na resolução da distribuição.
    # <density> é usado para normalizar o histograma, para que a área total sob o histograma seja igual a 1 (densidade).
    # <edgecolor> define a cor das bordas dos retângulos do histograma.
    # <alpha> define a transparência das barras.

    plt.xlabel('Retornos Diários')  # Rótulo eixo X
    plt.ylabel('Densidade')  # Rótulo eixo Y
    plt.title('Histograma dos Retornos Diários de uma Ação')  # Título do Gráfico
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # exibe o gráfico

    # Gráfico Boxplot

    # Dados de exemplo - Retornos diários hipotéticos de duas ações (gerados aleatoriamente com a função random.normalvariate())
    # Criação de duas listas (retornos_acao1 e retornos_acao2) com 100 valores hipotéticos de cada ação
    num_dias = 100
    acoes = ['Ação 1', 'Ação 2']

    retornos_acao1 = [random.normalvariate(0.01, 0.02) for _ in range(num_dias)]  # Média: 0.01, Desvio padrão: 0.02
    retornos_acao2 = [random.normalvariate(0.02, 0.03) for _ in range(num_dias)]  # Média: 0.02, Desvio padrão: 0.03

    # Plotar o Gráfico de Boxplot
    plt.figure(figsize=(8, 6))  # define o tamanho do gráfico

    plt.boxplot([retornos_acao1, retornos_acao2], labels=acoes, vert=False, patch_artist=True)
    # <listas dos retornos diários das 2 ações> lista de dados que queremos comparar
    # <labels> é usado para adicionar rótulos aos boxplots (nomes das ações).
    # <vert> é usado para exibir os boxplots horizontalmente.
    # <patch_artist> é usado para preencher os boxplots com cores.

    plt.xlabel('Retornos Diários')  # Rótulo eixo X
    plt.title('Boxplot dos Retornos Diários de Duas Ações')  # Título do Gráfico
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # exibe o gráfico

    # Gráfico Violino

    # Dados de exemplo - Retornos diários hipotéticos de duas ações (gerados aleatoriamente com a função random.normalvariate())
    # Criação de duas listas (retornos_acao1 e retornos_acao2) com 100 valores hipotéticos de cada ação
    num_dias = 100
    acoes = ['Ação 1', 'Ação 2']

    retornos_acao1 = [random.normalvariate(0.01, 0.02) for _ in range(num_dias)]  # Média: 0.01, Desvio padrão: 0.02
    retornos_acao2 = [random.normalvariate(0.02, 0.03) for _ in range(num_dias)]  # Média: 0.02, Desvio padrão: 0.03

    # Plotar o Gráfico de Violino
    plt.figure(figsize=(8, 6))  # define o tamanho do gráfico

    plt.violinplot([retornos_acao1, retornos_acao2], showmedians=True, vert=False, widths=0.7)
    # <listas dos retornos diários das 2 ações> lista de dados que queremos comparar
    # <showmedians> é usado para exibir a linha que representa a mediana no gráfico de violino.
    # <vert> é usado para exibir os violinos horizontalmente.
    # <widths> define a largura dos violinos.

    plt.xlabel('Retornos Diários')  # Rótulo eixo X
    plt.title('Gráfico de Violino dos Retornos Diários de Duas Ações')  # Título do Gráfico
    plt.yticks([1, 2], acoes)  # é usado para adicionar os rótulos das ações no eixo y.
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # exibe o gráfico

    # Gráfico 3D

    # Utilizamos o módulo mpl_toolkits.mplot3d para criar um espaço 3D para plotar os dados.

    # Dados de exemplo - Preços hipotéticos das ações de duas empresas em diferentes momentos (10 momentos)
    tempo = list(range(1, 11))  # 10 pontos de dados no tempo
    empresa_a_precos = [random.uniform(100, 200) for _ in tempo]  # Preços da Empresa A
    empresa_b_precos = [random.uniform(80, 180) for _ in tempo]  # Preços da Empresa B

    # Plotar o Gráfico 3D
    fig = plt.figure(figsize=(8, 6))  # define o tamanho do gráfico
    ax = fig.add_subplot(111, projection='3d')

    # Usamos a função ax.plot() para criar os gráficos 3D
    ax.plot(tempo, empresa_a_precos, zs=0, zdir='z', label='Empresa A', color='red', marker='o')
    ax.plot(tempo, empresa_b_precos, zs=1, zdir='z', label='Empresa B', color='blue', marker='x')
    # O primeiro argumento é o eixo x (tempo).
    # O segundo argumento é o eixo y (preços das ações das empresas A e B).
    # <zs> é usado para especificar a posição no eixo z onde os dados serão plotados (0 para Empresa A e 1 para Empresa B).
    # <zdir='z'> define a direção do eixo z.
    # <label> é usado para adicionar rótulos às legendas.
    # <color> define as cores dos pontos (vermelho para Empresa A e azul para Empresa B).
    # <marker> especifica o símbolo usado para representar cada ponto de dado (círculo para Empresa A e 'x' para Empresa B).

    ax.set_xlabel('Tempo')  # Rótulo eixo X
    ax.set_ylabel('Preço da Ação')  # Rótulo eixo Y
    ax.set_zlabel('Empresa')  # Rótulo eixo Z
    ax.set_title('Gráfico 3D dos Preços das Ações de Empresa A e Empresa B ao Longo do Tempo')  # Título do Gráfico
    ax.legend()  # é usado para mostrar as legendas no gráfico 3D.
    plt.tight_layout()  # é usado para evitar que as legendas se sobreponham ao gráfico.
    plt.show()  # exibe o gráfico 3D

    # Gráficos Avançados

    # Gráfico de Linhas com Múltiplas Séries

    # Exemplo com múltiplas séries de preços de ações
    # Dados fictícios de preços de ações de duas empresas ao longo de 10 dias
    dias = list(range(1, 11))
    acao1 = [100, 105, 110, 115, 120, 115, 118, 130, 135, 140]
    acao2 = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125]

    # Plotar o gráfico de linhas com múltiplas séries
    plt.plot(dias, acao1, marker='o', linestyle='-', color='b', label='Ação 1')
    plt.plot(dias, acao2, marker='s', linestyle='--', color='g', label='Ação 2')
    plt.xlabel('Dia')  # rótulo do eixo X
    plt.ylabel('Preço')  # rótulo do eixo Y
    plt.title('Evolução do Preço das Ações')  # Título do Gráfico
    plt.legend()  # é usado para mostrar as legendas no gráfico
    plt.grid(True)  # adiciona linhas de grade ao gráfico para facilitar a leitura.
    plt.show()  # exibe o gráfico

    # Gráfico Candlestick (vela)

    # Dados fictícios de preços de ações para um período de tempo
    datas = ['2023-07-10', '2023-07-11', '2023-07-12', '2023-07-13', '2023-07-14', '2023-07-15']
    abertura = [100, 110, 114, 120, 118, 125]
    fechamento = [105, 115, 108, 128, 120, 115]
    maximo = [112, 120, 122, 130, 125, 130]
    minimo = [98, 105, 107, 118, 114, 110]

    dados = {
        'Date': datas,
        'Open': abertura,
        'Close': fechamento,
        'High': maximo,
        'Low': minimo
    }

    df = pd.DataFrame(dados)
    df['Date'] = pd.to_datetime(df['Date'])  # Convertendo a coluna de data para o tipo datetime

    # Configurar o índice do DataFrame como a coluna de data
    df.set_index('Date', inplace=True)

    # Criar um formatador de datas para o eixo x
    date_format = mdates.DateFormatter('%Y-%m-%d')

    # Plotar o gráfico de candlestick
    fig, ax = mpf.plot(df, type='candle', style='yahoo', title='Gráfico de Candlestick',
                       ylabel='Preço', xrotation=45, datetime_format='%Y-%m-%d', returnfig=True)

    # Adicionar título à figura
    fig.suptitle('Gráfico de Candlestick')

    plt.show()  # exibe o gráfico


if __name__ == '__main__':
    matplotlib_lesson()
