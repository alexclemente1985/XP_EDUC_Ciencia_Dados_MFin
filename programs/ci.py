def calculando_ci():
    vf = float(input('Informe o Valor Futuro: '))
    i = float(input('Informe a Taxa: '))
    t = int(input('Informe o Período: '))

    ci = vf/(1+i)**t

    print(f'O Capital Inicial é : R${round(ci, 2)}')


if __name__ == '__main__':
    calculando_ci()