# This is a sample Python script.
from lib_lessons.numpy_lesson import numpy_lesson
from lib_lessons.pandas_lesson import pandas_lesson


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def lib_lessons():
    aula = input("Escolha a aula -> Numpy(np) | Pandas(pd)")
    match aula.lower():
        case "np":
            numpy_lesson()
        case "pd":
            pandas_lesson()
        case other:
            print("Escolha inválida...")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lib_lessons()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
