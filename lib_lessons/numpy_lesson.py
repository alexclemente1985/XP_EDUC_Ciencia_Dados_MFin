import numpy as np


def numpy_lesson():
    # Tipos de arrays
    vector = np.array([1, 2, 3, 4, 5])

    arr_bidimensional = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    linha_separadora = "-" * 35

    print("Array do tipo vetor:'\n", vector)
    print("\n" + linha_separadora + "\n")
    print("Array bidimensional: \n", arr_bidimensional)

    # Operações matemáticas e lógicas

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])

    ##Adição de arrays
    result = arr1 + arr2
    print("Soma: ", result)

    ##Multiplicação
    result = arr1 * arr2
    print("Multiplicação: ", result)

    ##Operações matemáticas
    result = np.sin(arr1)
    print("Seno: ", result)

    ##Somatório dos elementos do array
    result = sum(arr1)
    print("Somatório arr1: ", result)

    # Indexação e fatias com arrays

    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    ##Acessando um elemento pelo índice
    print(arr1[0])  # Output: 1

    ##Acessando elementos usando fatias
    print(arr1[1:4])  # Output: [2 3 4]

    ##Acessando elementos a partir do final do array
    print(arr1[-1])  # Output: 5

    ##Acessando elementos do array multidimensional
    print(arr2[0])
    print(arr2[1, 0])

    # Funções universais (UFuncs)

    ##Função de exponenciação > np.exp()
    arr = np.array([1, 2, 3, 4, 5])
    result = np.exp(arr)
    print(result)  # Saída: [ 2.71828183  7.3890561  20.08553692 54.59815003 148.4131591 ]

    ##Função de Seno > np.sin()
    arr = np.array([0, np.pi / 2, np.pi])
    result = np.sin(arr)
    print(result)  # Saída: [0.         1.         1.2246468e-16]

    ##Função de Cosseno > np.cos()
    arr = np.array([0, np.pi / 2, np.pi])
    result = np.cos(arr)
    print(result)  # Saída: [ 1.000000e+00  6.123234e-17 -1.000000e+00]

    ##Função de Raiz Quadrada > np.sqrt()
    arr = np.array([4, 9, 16])
    result = np.sqrt(arr)
    print(result)  # Saída: [2. 3. 4.]

    ##Função de Logaritmo > np.log()
    arr = np.array([1, np.e, np.e ** 2])
    result = np.log(arr)
    print(result)  # Saída: [0.         1.         2.        ]

    ##Função de soma > np.sum()
    arr = np.array([1, 2, 3, 4, 5])
    result = np.sum(arr)
    print(result)  # Saída: 15

    ##Função de elemento máximo > np.max()
    arr = np.array([1, 2, 3, 4, 5])
    result = np.max(arr)
    print(result)  # Saída: 5

    ##Função de elemento mínimo > np.min()
    arr = np.array([1, 2, 3, 4, 5])
    result = np.min(arr)
    print(result)  # Saída: 1

    # Manipulação de arrays

    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6]])

    ##Concatenação vertical de arrays
    result = np.concatenate((arr1, arr2), axis=0)
    ###(axis=0) Nesse caso, os arrays de entrada são concatenados ao longo do eixo das Linhas.
    print(result)

    print()  # Gera uma linha em branco

    ##Concatenação achatada de arrays
    result = np.concatenate((arr1, arr2), axis=None)
    ###(axis=None) (Padrão) Nesse caso, os arrays de entrada são achatados (flattened) e concatenados em um único array unidimensional.
    print(result)

    print()  # Gera uma linha em branco

    ##Transposição de um array
    result = arr1.T
    print(result)

    print()  # Gera uma linha em branco

    ##Transposição de um array
    result = arr2.T
    print(result)

    # Funções estatísticas
    arr = np.array([1, 2, 3, 4, 5, 6])

    ##Média dos elementos do array
    result = np.mean(arr)
    print("Média dos elementos do array:", result)

    ##Mediana dos elementos do array
    result = np.median(arr)
    print("Mediana dos elementos do array:", result)

    ##Desvio padrão dos elementos do array
    result = np.std(arr)
    print("Desvio padrão dos elementos do array:", result)

    print()

    ##Exemplo com Array Multidimensional
    ###Criando um array bidimensional
    arr = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

    ###Calculando a mediana ao longo do eixo 0 (Linhas)
    median_axis0 = np.median(arr, axis=0)
    print("Mediana ao longo do eixo 0:", median_axis0)

    ###Calculando a mediana ao longo do eixo 1 (Colunas)
    median_axis1 = np.median(arr, axis=1)
    print("Mediana ao longo do eixo 1:", median_axis1)

    ###Calculando a mediana de todos os elementos do array
    median_all = np.median(arr)
    print("Mediana de todos os elementos:", median_all)

    # Álgebra Linear

    ##Multiplicação de matrizes
    arr1 = np.array([[1, 2], [3, 4]])
    arr2 = np.array([[5, 6], [7, 8]])

    result = np.dot(arr1, arr2)
    print(result)

    print()

    ##Cálculo do determinante de uma matriz
    arr = np.array([[1, 2], [3, 4]])

    result = np.linalg.det(arr)
    print(result)


if __name__ == '__main__':
    numpy_lesson()
