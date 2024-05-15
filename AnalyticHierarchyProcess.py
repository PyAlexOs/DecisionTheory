import numpy as np
from fractions import Fraction
import functools
import shutil

NAMES = ['Audi A4', 'Kia Sportage', 'Volvo XC90', 'Nissan X-Trail', 'Mercedes GLC']
CRITERIA_2_LEVEL = [
    [Fraction(1, 1), Fraction(3, 1), Fraction(7, 1), Fraction(5, 1), Fraction(9, 1)],
    [Fraction(1, 3), Fraction(1, 1), Fraction(5, 1), Fraction(3, 1), Fraction(7, 1)],
    [Fraction(1, 7), Fraction(1, 5), Fraction(1, 1), Fraction(1, 3), Fraction(1, 1)],
    [Fraction(1, 5), Fraction(1, 3), Fraction(3, 1), Fraction(1, 1), Fraction(5, 1)],
    [Fraction(1, 9), Fraction(1, 7), Fraction(1, 1), Fraction(1, 5), Fraction(1, 1)]
              ]
CRITERIA_3_LEVEL = [
    [
        [Fraction(1, 1), Fraction(1, 3), Fraction(5, 1), Fraction(1, 5), Fraction(3, 1)],
        [Fraction(3, 1), Fraction(1, 1), Fraction(7, 1), Fraction(1, 3), Fraction(5, 1)],
        [Fraction(1, 5), Fraction(1, 7), Fraction(1, 1), Fraction(1, 9), Fraction(1, 3)],
        [Fraction(5, 1), Fraction(3, 1), Fraction(9, 1), Fraction(1, 1), Fraction(7, 1)],
        [Fraction(1, 3), Fraction(1, 5), Fraction(3, 1), Fraction(1, 7), Fraction(1, 1)]
    ],
    [
        [Fraction(1, 1), Fraction(2, 1), Fraction(1, 5), Fraction(1, 4), Fraction(1, 6)],
        [Fraction(1, 2), Fraction(1, 1), Fraction(1, 7), Fraction(1, 7), Fraction(1, 6)],
        [Fraction(5, 1), Fraction(7, 1), Fraction(1, 1), Fraction(1, 3), Fraction(1, 2)],
        [Fraction(4, 1), Fraction(7, 1), Fraction(3, 1), Fraction(1, 1), Fraction(2, 1)],
        [Fraction(6, 1), Fraction(6, 1), Fraction(2, 1), Fraction(1, 2), Fraction(1, 1)]
    ],
    [
        [Fraction(1, 1), Fraction(3, 1), Fraction(1, 5), Fraction(5, 1), Fraction(1, 4)],
        [Fraction(1, 3), Fraction(1, 1), Fraction(1, 7), Fraction(3, 1), Fraction(1, 6)],
        [Fraction(5, 1), Fraction(7, 1), Fraction(1, 1), Fraction(9, 1), Fraction(3, 1)],
        [Fraction(1, 5), Fraction(1, 3), Fraction(1, 9), Fraction(1, 1), Fraction(1, 7)],
        [Fraction(4, 1), Fraction(6, 1), Fraction(1, 3), Fraction(7, 1), Fraction(1, 1)]
    ],
    [
        [Fraction(1, 1), Fraction(9, 1), Fraction(1, 2), Fraction(3, 1), Fraction(6, 1)],
        [Fraction(1, 9), Fraction(1, 1), Fraction(1, 9), Fraction(1, 7), Fraction(1, 3)],
        [Fraction(2, 1), Fraction(9, 1), Fraction(1, 1), Fraction(4, 1), Fraction(7, 1)],
        [Fraction(1, 3), Fraction(7, 1), Fraction(1, 4), Fraction(1, 1), Fraction(4, 1)],
        [Fraction(1, 6), Fraction(3, 1), Fraction(1, 7), Fraction(1, 4), Fraction(1, 1)]
    ],
    [
        [Fraction(1, 1), Fraction(7, 1), Fraction(1, 1), Fraction(3, 1), Fraction(7, 1)],
        [Fraction(1, 7), Fraction(1, 1), Fraction(1, 7), Fraction(1, 6), Fraction(1, 1)],
        [Fraction(1, 1), Fraction(7, 1), Fraction(1, 1), Fraction(3, 1), Fraction(7, 1)],
        [Fraction(1, 3), Fraction(6, 1), Fraction(1, 3), Fraction(1, 1), Fraction(6, 1)],
        [Fraction(1, 7), Fraction(1, 1), Fraction(1, 7), Fraction(1, 6), Fraction(1, 1)]
    ]
]


class MatrixPipeline:
    """ Class containing methods for converting matrix and calculating its consistency """
    random_consistency = [0.0, 0.0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49, 1.51, 1.48, 1.56, 1.57, 1.59]

    @staticmethod
    def check_consistency(matrix: np.ndarray[np.ndarray[float]],
                          output: bool = False) -> bool:
        """ Checks the consistency of the matrix """
        if len(matrix) <= 2:
            if output:
                print("Not enough alternatives.")
            return False

        priority_vector = MatrixPipeline.get_priority_vector(matrix)
        consistency_index = MatrixPipeline.get_consistency_index(matrix, priority_vector)
        consistency = consistency_index / MatrixPipeline.random_consistency[len(priority_vector)]

        if output:
            print(f"Consistency: {consistency}")
        if consistency <= 0.1:
            if output:
                print("The matrix is consistent")
            return True

        if output:
            print("The matrix is not consistent")
        return False

    @staticmethod
    def g_mean(elements: np.ndarray[float]) -> float:
        """ Calculates the geometric mean of the array """
        return np.exp(np.log(elements).mean())

    @staticmethod
    def normalize(g_means: np.ndarray[float]) -> np.ndarray[float]:
        """ Normalizes the values in the array """
        return g_means / sum(g_means)

    @staticmethod
    def get_priority_vector(matrix: np.ndarray[np.ndarray[float]]) -> np.ndarray[float]:
        """ Calculates the priority vector for the matrix """
        return MatrixPipeline.normalize(np.array([MatrixPipeline.g_mean(row) for row in matrix]))

    @staticmethod
    def get_consistency_index(matrix: np.ndarray[np.ndarray[float]],
                              priority_vector: np.ndarray[float]) -> float:
        """ Calculates the consistency index of the matrix """
        return (sum(matrix.sum(0) * priority_vector) - len(priority_vector)) / (len(priority_vector) - 1)

    @staticmethod
    def make_report(matrix: np.ndarray[np.ndarray[Fraction]],
                    is_criteria: str = "",
                    criteria_number: str = ""):
        """ Outputs all formulas for the report """
        console_width = shutil.get_terminal_size().columns
        print(f"{"-" * ((console_width - 12) // 2)}Criteria #{criteria_number}{"-" * ((console_width - 12) // 2)}\n"
              if criteria_number != "" else "", end="")
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

        print(f"{"-" * ((console_width - 11) // 2)}Consistency{"-" * ((console_width - 11) // 2)}")
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
              f"{criteria_number}}}}}{{СИ}}=\\frac{{{index}}}{{{MatrixPipeline.random_consistency[len(preferences) - 1]}}}="
              f"{round(index / MatrixPipeline.random_consistency[len(preferences) - 1], 3)}")


def rank(priority_vector: np.ndarray[float],
         criteria_priority_vectors: np.ndarray[np.ndarray[float]],
         show_report: bool = True):
    """ Calculates the best alternatives using a second-level priority vector and an array of third-level priority vectors """
    if show_report:
        console_width = shutil.get_terminal_size().columns
        print(f"\n{"-" * ((console_width - 11) // 2)}Priorities{"-" * ((console_width - 11) // 2)}")
        for (i, vector) in enumerate(criteria_priority_vectors, start=1):
            print(f"W_{i}={" + ".join(f"W_{{2{j}}}*W_{{3К{j}{i}}}" for j in range(1, len(priority_vector) + 1))}="
                  f"{" + ".join([f"{round(priority_vector[j], 3)}*{round(vector[j], 3)}"
                                 for j in range(len(priority_vector))])}="
                  f"{round(sum([priority_vector[j] * vector[j] for j in range(len(priority_vector))]), 3)}")

    return [np.dot(vector, priority_vector) for vector in criteria_priority_vectors]


def parse_table_from_word(string: str) -> np.ndarray[np.ndarray[Fraction]]:
    """ Converts the table inserted from the word into an array of fractions """
    splited_string = [row.split("\t") for row in string.strip("\n").strip().split("\n")]
    table: np.ndarray[np.ndarray[Fraction]] = np.full((len(splited_string), len(splited_string[0])), Fraction())
    for row in range(len(splited_string)):
        for col in range(len(splited_string[0])):
            if "/" in splited_string[row][col]:
                numerator, denominator = list(map(lambda x: int(x), splited_string[row][col].split("/")))
                table[row][col].numerator, table[row][col].denominator = numerator, denominator
            else:
                table[row][col].numerator, table[row][col].denominator = int(splited_string[row][col]), 1

    return table


def main():
    # CRITERIA_2_LEVEL = parse_table_from_word(""" Table from word """)
    # CRITERIA_3_LEVEL = list(map(parse_table_from_word, [""" Table from word """, """ Another table from word """, ...]))
    
    pipeline = MatrixPipeline()
    pipeline.make_report(np.array(CRITERIA_2_LEVEL))
    for (i, criteria_for_alternative) in enumerate(CRITERIA_3_LEVEL, start=1):
        print()
        pipeline.make_report(np.array(criteria_for_alternative), is_criteria="К", criteria_number=str(i))

    priority_vector = pipeline.get_priority_vector(np.array(CRITERIA_2_LEVEL, dtype=float))
    criteria_vectors = np.array([pipeline.get_priority_vector(np.array(criteria_vec, dtype=float))
                                for criteria_vec in CRITERIA_3_LEVEL])
    
    preferences = rank(priority_vector, criteria_vectors)
    print("\n", *sorted([(name, round(rate, 3)) for name, rate in zip(NAMES, preferences)],
                  reverse=True, key=lambda x: x[1]), sep="\n")


if __name__ == '__main__':
    main()
