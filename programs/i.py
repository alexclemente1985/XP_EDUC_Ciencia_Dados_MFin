
def calculando_i():
    ci = float(input('Informe o Capital Inicial: '))
    t = int(input('Informe o Período: '))
    vf = float(input('Informe o Valor Futuro: '))

    i = ((vf/ci)**(1/t))-1

    print(f'A taxa de juros é: {round(i,2)*100}%')


if __name__ == '__main__':
    calculando_i()