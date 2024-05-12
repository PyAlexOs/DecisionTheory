import numpy as np
from fractions import Fraction
import functools


class HierarchyAnalyzer:
    random_consistency = [0.0, 0.0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]

    def __call__(self, matrix: np.ndarray[np.ndarray[float]]) -> bool:
        if len(matrix) <= 2:
            print("Not enough alternatives.")
            return False

        priority_vector = self.__get_priority_vector(matrix)
        consistency_index = self.__get_consistency_index(matrix, priority_vector)
        consistency = consistency_index / self.random_consistency[len(priority_vector)]

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

    def __get_consistency_index(self, matrix: np.ndarray[np.ndarray[float]], priority_vec: np.ndarray[float]) -> float:
        return (sum(matrix.sum(0) * priority_vec) - len(priority_vec)) / (len(priority_vec) - 1)

    def make_report(self, matrix: np.ndarray[np.ndarray[Fraction]], is_criteria: str = "", criteria_number: str = ""):
        print(f"{"-" * 10}Criteria #{criteria_number}{"-" * 10}\n" if criteria_number != "" else "", end="")
        priority_vector = list()
        for (i, row) in enumerate(matrix, start=1):
            priority_vector.append(round((functools.reduce(lambda a, b: a * b, row)) ** (1 / len(row)), 3))
            print(f"V_{{{is_criteria}{criteria_number}{i}}}=({" * ".join(str(element)
                                                                         if element.denominator == 1
                                                                         else f"\\frac{{{element.numerator}}}"
                                                                              f"{{{element.denominator}}}"
                                                                         for element in row)})^\\frac{{1}}"
                  f"{{{len(row)}}}={priority_vector[-1]}")

        coeff = sum(priority_vector)
        print(f"∑V_{{{is_criteria}{criteria_number}{"Y" if is_criteria != "" else "i"}}}"
              f"={" + ".join([f"V_{{{is_criteria}{criteria_number}{i}}}" for i in range(1, len(priority_vector) + 1)])}"
              f"={" + ".join(list(map(lambda x: str(x), priority_vector)))}={coeff}")

        for (i, row) in enumerate(matrix, start=1):
            print(f"W_{{{f"3{is_criteria}{criteria_number}{i}" if is_criteria != "" else
                  f"2{i}"}}}=\\frac{{{priority_vector[i - 1]}}}{{∑V_i}} = {round(priority_vector[i - 1] / coeff, 3)}")
            priority_vector[i - 1] = round(priority_vector[i - 1] / coeff, 3)

        print(f"W_{{{f"3{is_criteria}{criteria_number}Y" if is_criteria != "" else "2i"}}}"
              f"=({"; ".join(list(map(lambda x: str(x), priority_vector)))})")

        print(f"{"-" * 10}Consistency{"-" * 10}")
        col_sum = list()
        for (i, col) in enumerate(matrix.T, start=1):
            col_sum.append(sum(col))
            print(f"S_{{{i}{is_criteria}{criteria_number}}}={" + ".join(str(element) if element.denominator == 1
                                                                        else f"\\frac{{{element.numerator}}}"
                                                                             f"{{{element.denominator}}}"
                                                                        for element in col)}="
                  f"{str(col_sum[-1]) if col_sum[-1].denominator == 1 else f"\\frac{{{col_sum[-1].numerator}}}"
                                                                           f"{{{col_sum[-1].denominator}}}"
                                                                           f"\\approx{round(float(col_sum[-1]), 3)}"}")

        preferences = list()
        for (i, col) in enumerate(col_sum, start=1):
            preferences.append(round(col * priority_vector[i - 1], 3))
            print(f"P_{{{i}{is_criteria}{criteria_number}}}=S_{{{i}{is_criteria}{criteria_number}}}*W_"
                  f"{{{"3" if is_criteria != "" else "2"}{is_criteria}"
                  f"{criteria_number}{i}}}={preferences[-1]}")

        print(f"λ_{{max{is_criteria}{criteria_number}}}={" + ".join([f"P_{{{i}{is_criteria}{criteria_number}}}"
                                                                     for i in range(1, len(preferences) + 1)])}="
              f"{round(sum(preferences), 3)}")
        index = round((round(sum(preferences), 3) - len(preferences)) / (len(preferences) - 1), 3)
        print(f"ИС{"_" if is_criteria != "" else ""}{{{is_criteria}{criteria_number}}}="
              f"\\frac{{λ_{{max{is_criteria}{criteria_number}}} - n}}{{n - 1}}="
              f"\\frac{{{round(sum(preferences), 3)} - {len(preferences)}}}{{{len(preferences)} - 1}}="
              f"{index}")
        print(f"ОС{"_" if is_criteria != "" else ""}{{{is_criteria}{criteria_number}}}="
              f"\\frac{{ИС{"_" if is_criteria else ""}{{{is_criteria}"
              f"{criteria_number}}}}}{{СИ}}=\\frac{{{index}}}{{{self.random_consistency[len(preferences) - 1]}}}="
              f"{round(index / self.random_consistency[len(preferences) - 1], 3)}")


def parse_table_from_word(string: str) -> list[list[Fraction]]:
    table = [row.split("\t") for row in string.split("\n")]
    for row in range(len(table)):
        for col in range(len(table[0])):
            if "/" in table[row][col]:
                numerator, denominator = table[row][col].split("/")
                table[row][col] = Fraction(int(numerator), int(denominator))
            else:
                table[row][col] = Fraction(int(table[row][col]), 1)

    return table


def main():
    # insert data from docx table to variable
    criteria = """1	3	7	5	9
                  1/3	1	5	3	7
                  1/7	1/5	1	1/3	1
                  1/5	1/3	3	1	5
                  1/9	1/7	1	1/5	1"""
    criteria_for_alternatives = ["""1	1/3	5	1/5	3
                                    3	1	7	1/3	5
                                    1/5	1/7	1	1/9	1/3
                                    5	3	9	1	7
                                    1/3	1/5	3	1/7	1""",
                                 """1	2	1/5	1/4	1/6
                                    1/2	1	1/7	1/7	1/6
                                    5	7	1	1/3	1/2
                                    4	7	3	1	2
                                    6	6	2	1/2	1""",
                                 """1	3	1/5	5	1/4
                                    1/3	1	1/7	3	1/6
                                    5	7	1	9	3
                                    1/5	1/3	1/9	1	1/7
                                    4	6	1/3	7	1""",
                                 """1	9	1/2	3	6
                                    1/9	1	1/9	1/7	1/3
                                    2	9	1	4	7
                                    1/3	7	1/4	1	4
                                    1/6	3	1/7	1/4	1""",
                                 """1	7	1	3	7
                                    1/7	1	1/7	1/6	1
                                    1	7	1	3	7
                                    1/3	6	1/3	1	6
                                    1/7	1	1/7	1/6	1"""]

    criteria = parse_table_from_word(criteria)
    criteria_for_alternatives = list(map(parse_table_from_word, criteria_for_alternatives))

    analyzer = HierarchyAnalyzer()
    analyzer.make_report(np.array(criteria))
    for (i, criteria_for_alternative) in enumerate(criteria_for_alternatives, start=1):
        print()
        analyzer.make_report(np.array(criteria_for_alternative), is_criteria="К", criteria_number=str(i))

    if analyzer(np.array(criteria, dtype=float)):
        pass
    for (i, criteria_for_alternative) in enumerate(criteria_for_alternatives, start=1):
        if analyzer(np.array(criteria_for_alternative, dtype=float)):
            pass

if __name__ == '__main__':
    main()
