import sys
import numpy as np
import shutil


class SimplexTable:
    table: np.ndarray[np.ndarray[np.float64]]
    func_coeffs: np.ndarray[np.float64]
    base_coeffs: np.ndarray[np.float64]
    func_vars: list[int]
    base_vars: list[int]

    def __init__(self, n: int, k: int, mode: bool = True):  # True for -> max
        self.func_coeffs = np.array(list(map(float, input('Введите коэффициенты целевой функции через пробел: ').split())))
        if len(self.func_coeffs) != n:
            raise ValueError('Неправильно указано количество переменных целевой функции или неверно указаны коэффициенты.')
        if not mode:
            self.func_coeffs = -self.func_coeffs
        
        self.func_vars = [i for i in range(n)]
        self.base_coeffs = np.zeros(shape=(k), dtype=np.int32)
        self.base_vars = [n + i for i in range(k)]
        self.table = np.zeros(shape=(k + 1, n + 1), dtype=np.float64)

        for i in range(k):
            print(f'Введите коэффициенты в {i + 1}-м ограничении')
            for j in range(n):
                self.table[i, j] = float(input(f'Введите коэффициент при x{j + 1}: '))
            self.table[i, n] = float(input('Введите свободный коэффициент: '))
        
        for j in range(n):
            self.table[k, j] = -self.func_coeffs[j]

    def step(self):
        """ Iterates the simplex method """
        col = np.argmin(self.table[-1])
        if np.max(self.table[:, col]) <= 0: # Отношения составляются только к положительным элементам разрешающего столбца
            exit('Оптимальное решение найти невозможно')

        min_quotient = None
        for i in range(len(self.table) - 1):
            if self.table[i, col] > 0 and (not min_quotient or min_quotient and self.table[i, -1] / self.table[i, col] < min_quotient):
                resolving_element = self.table[i, col]
                row = i
                min_quotient = self.table[i, -1] / self.table[i, col]

        for i in range(len(self.table)):
            for j in range(len(self.table[0])):
                if i == row or j == col:
                    continue
                self.table[i, j] = ((self.table[i, j] * resolving_element) - (self.table[row, j] * self.table[i, col])) / resolving_element

        self.table[row, :] /= resolving_element
        self.table[:, col] /= -resolving_element
        self.table[row][col] = 1 / resolving_element
        self.base_coeffs[row], self.func_coeffs[col] = self.func_coeffs[col], self.base_coeffs[row]
        self.base_vars[row], self.func_vars[col] = self.func_vars[col], self.base_vars[row]

    def is_optimal(self) -> bool:
        """ Checks if solution is optimal """
        if np.min(self.table[-1]) >= 0:
            return True
        return False

    def __call__(self):
        print('\nИсходная симплекс-таблица:')
        print(self)

        iter = 1
        while True:
            self.step()
            header = f'Итерация {iter}'
            console_width = shutil.get_terminal_size().columns
            filler = "-" * ((console_width - len(header)) // 2)
            print(f"\n{filler}{header}{filler}\n" + str(self))
            iter += 1

            if self.is_optimal():
                print('Решение найдено')
                for i in range(len(self.base_coeffs)):
                    if self.base_coeffs[i] != 0:
                        print(f'X{self.base_vars[i] + 1} -- {self.table[i, -1]: .3f}')
                print(f'Значение целевой функции: {self.table[-1, -1]: .3f} [ден.ед.]')
                break

    def __str__(self) -> str:
        cell_width = 7
        result = []
        result.append(['', 'c_j', *list(map(lambda x: round(x, 3), self.func_coeffs))])
        result.append(['C_v', '', *[f'X_{str(coeff + 1)}' for coeff in self.func_vars], 'A_0'])

        for (i, row) in enumerate(self.table):
            if i == len(self.table) - 1:
                result.append(['', 'f', *list(map(lambda x: round(x, 3), row))])
            else:
                result.append([self.base_coeffs[i], f'X_{self.base_vars[i] + 1}', *list(map(lambda x: round(x, 3), row))])

        result.append(['', '', *[f'∆_{i + 1}' for i in range(len(self.func_vars))], 'Q'])
        return '\n'.join(list(map(lambda lst: ''.join([str(col).rjust(cell_width) for col in lst]), result)))


def main():
    sys.stdin = open('simplex.txt', 'r')
    n = int(input('Введите количество переменных: '))
    k = int(input('Введите количество ограничений: '))
    mode = input('1. Максимизация\n2. Минимизация\n') == '1'

    solution = SimplexTable(n, k, mode)
    solution()


if __name__ == "__main__":
    main()
