from math import log

def calculando_t():
    ci = float(input('Informe o Capital Inicial: '))
    i = float(input('Informe a Taxa: '))
    vf = float(input('Informe o Valor Futuro: '))

    t=log(vf/ci)/log(1+i)

    print(f'O Período é : {round(t,2)}')


if __name__ == '__main__':
    calculando_t()