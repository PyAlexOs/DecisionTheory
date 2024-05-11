import pandas as pd
import numpy as np
from fractions import Fraction
from typing import Iterable
import shutil

KEYS = ["Стоимость машины",
        "Расход топлива",
        "Вместительность багажника",
        "Мощность двигателя",
        "Клиренс"]
TABLE = [['Mazda CX-5', 4.1, 7.1, 565, 165, 235],
         ['Toyota Land Cruiser Prado', 4.1, 11.0, 550, 163, 235],
         ['Audi A4', 4.1, 7.1, 570, 245, 235],
         ['Kia Sportage', 3.2, 8.2, 540, 150, 181],
         ['Volvo XC90', 5.6, 5.7, 721, 249, 235],
         ['Subaru Outback', 5.6, 7.3, 522, 188, 213],
         ['Nissan X-Trail', 2.2, 5.3, 497, 225, 230],
         ['Mercedes GLC', 5.1, 5.5, 620, 197, 180],
         ['Jeep Wrangler', 4.9, 11.3, 142, 225, 220],
         ['УАЗ Патриот', 1.9, 11.2, 1130, 225, 225]]
ASPIRATIONS = [False, False, True, True, True]


class HierarchyAnalyzer:
    def __call__(self, matrix: np.ndarray[np.ndarray[float]]) -> bool:
        if len(matrix) <= 2:
            print("Not enough alternatives.")
            return False

        random_consistency = [0.0, 0.0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]
        priority_vector = self.__get_priority_vector(matrix)
        consistency_index = self.__get_consistency_index(matrix=matrix, priority_vector=priority_vector)
        consistency = consistency_index / random_consistency[len(priority_vector)]

        print(f"Consistency: {consistency}")
        if consistency <= 0.1:
            print("The matrix is consistent")
            return True
        
        print("The matrix is not consistent")
        return False

    def __g_mean(self, elements: np.ndarray[float]) -> float:
        return np.exp(np.log(elements).mean())
    
    def __normalize(self, g_means: np.ndarray[float]) -> np.ndarray[float]:
        return g_means / sum(g_means)
    
    def __get_priority_vector(self, matrix: np.ndarray[np.ndarray[float]]) -> np.ndarray[float]:
        return self.__normalize(np.array([self.__g_mean(row) for row in matrix]))
    
    def __get_consistency_index(self, matrix: np.ndarray[np.ndarray[float]], priority_vector: np.ndarray[float]) -> float:
        return (sum(matrix.sum(0) * priority_vector) - len(priority_vector)) / (len(priority_vector) - 1)
    
    def make_report():
        pass


def show(self: pd.DataFrame,
         header: str = ""):
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


def main():
    data = dict(zip(KEYS, [[TABLE[row][col]
                            for row in range(len(TABLE))]
                           for col in range(len(TABLE[0]))][1:]))

    df = pd.DataFrame(data=data, index=[f'A{i}' for i in range(1, 11)])
    df.show("Alternatives")

    analyzer = HierarchyAnalyzer()
    """print(analyzer(np.array([[Fraction(1, 1), Fraction(3, 1), Fraction(3, 1), Fraction(5, 1), Fraction(5, 1)],
                             [Fraction(1, 3), Fraction(1, 1), Fraction(3, 1), Fraction(3, 1), Fraction(3, 1)],
                             [Fraction(1, 3), Fraction(1, 3), Fraction(1, 1), Fraction(3, 1), Fraction(5, 1)],
                             [Fraction(1, 5), Fraction(1, 3), Fraction(1, 3), Fraction(1, 1), Fraction(7, 1)],
                             [Fraction(1, 5), Fraction(1, 3), Fraction(1, 5), Fraction(1, 7), Fraction(1, 1)]])))"""


if __name__ == '__main__':
    pd.DataFrame.show = show
    main()