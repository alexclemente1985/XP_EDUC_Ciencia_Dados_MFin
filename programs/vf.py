def calculando_vf():
    ci = float(input('Informe o Capital Inicial: '))
    i = float(input('Informe a Taxa: '))
    t = int(input('Informe o Período: '))

    vf = ci * (1 + i) ** t

    print(f'O Valor Futuro é : R${round(vf, 2)}')


if __name__ == '__main__':
    calculando_vf()