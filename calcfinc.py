# This is a sample Python script.
from statistics import mean

from programs import vf, ci, t, i


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def calc_finc():
    print('Bem-vindo à calculadora de Juros Compostos!')
    escolha = input('O que você gostaria de calcular? (vf, ci, t, i)\n')

    calculadora_on = True

    while calculadora_on:

        match escolha.lower():
            case 'vf':
                print('Calculando Valor Futuro')
                vf.calculando_vf()
            case 'ci':
                print('Calculando Capital Inicial')
                ci.calculando_ci()
            case 't':
                print('Calculando Período')
                t.calculando_t()
            case 'i':
                print('Calculando Taxa')
                i.calculando_i()
            case _:
                print('Escolha incorreta')
                continue

        continua_calculo = input('Desejaria fazer mais algum cálculo? SIM(S) | NÃO(N) ')

        if(continua_calculo.upper() == 'S'):
            escolha = input('O que você gostaria de calcular? (vf, ci, t, i)\n')
        else:
            calculadora_on = False










# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    calc_finc()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
