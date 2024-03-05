import pandas as pd
import shutil

KEYS = ["Стоимость машины",
        "Расход топлива",
        "Вместительность багажника",
        "Мощность двигателя",
        "Клиренс"]
TABLE = [['Mazda CX-5', 4.2, 7.1, 565, 165, 240],
         ['Toyota Land Cruiser Prado', 4.3, 11.0, 550, 163, 235],
         ['Audi A4', 4.1, 7.1, 570, 245, 245],
         ['Kia Sportage', 3.2, 8.2, 540, 150, 181],
         ['Volvo XC90', 5.6, 5.7, 721, 249, 235],
         ['Subaru Outback', 5.6, 7.3, 522, 188, 213],
         ['Nissan X-Trail', 2.2, 5.3, 497, 225, 230],
         ['Mercedes GLC', 5.1, 5.5, 620, 197, 180],
         ['Jeep Wrangler', 4.9, 11.3, 142, 225, 220],
         ['УАЗ Патриот', 1.9, 11.2, 1130, 225, 225]]


def main():
    data = dict(zip(KEYS, [[TABLE[row][col]
                            for row in range(len(TABLE))]
                           for col in range(len(TABLE[0]))][1:]))

    dataframe = pd.DataFrame(data=data, index=[f'A{i}' for i in range(1, 11)])
    dataframe.show("Alternatives")

    aspirations = [False, False, True, True, True]
    dataframe.normalize(aspirations)
    dataframe.show("Normalized")

    optimized_elements = dataframe.optimize()
    optimized = dataframe.iloc[optimized_elements]
    optimized.normalize(aspirations)
    optimized.show("Optimized")

    test_border_optimization(dataframe, optimized_elements)


def test_border_optimization(dataframe: pd.DataFrame, optimized_elements: list[int]):
    border_optimized_elements = dataframe.border_optimize(
        {"bigger": [[4, 230]]}
    )

    if len(border_optimized_elements) == 0:
        print("Failed to optimize the set using boundaries")
        return

    border_optimized = dataframe.iloc[border_optimized_elements]
    border_optimized.show("Border optimized")

    elements = list(set(border_optimized_elements) & set(optimized_elements))
    if len(elements) > 0:
        dataframe.iloc[elements].show("Pareto- and border-optimized set intersection")
    else:
        print("\nPareto- and border-optimized set intersection is empty")


def border_optimize(self, borders: dict[str, list[...]]) -> list[int]:
    elements = set()

    if len(borders) == 0:
        raise ValueError('Empty borders.')

    for (key, value) in borders.items():
        if key not in ["bigger", ">", "smaller", "<", "equals", "="]:
            raise KeyError('The keys can only be as follows: ["bigger", ">", "smaller", "<", "equals", "="]')

        if not isinstance(value, list) or len(value) == 0:
            raise ValueError('The value must be a dict like {condition: [[param_index_1, number_1], ...], ...}')

        for condition in value:
            if (not isinstance(condition, list) or
                    len(condition) != 2 or
                    condition[0] not in range(0, self.shape[1]) or
                    not all([i.isdigit() or i == "." for i in str(condition[1])])):
                raise ValueError('The value must be a list like: [param_index, number]')

            for row in range(self.shape[0]):
                if key in ["bigger", ">"]:
                    if self.iloc[row, condition[0]] > condition[1]:
                        elements.add(row)
                elif key in ["smaller", "<"]:
                    if self.iloc[row, condition[0]] < condition[1]:
                        elements.add(row)
                else:
                    if self.iloc[row, condition[0]] == condition[1]:
                        elements.add(row)

    return list(elements)


def optimize(self, show_comparison: bool = True) -> list[int]:
    elements = set()

    def compare() -> list[list[str]]:
        result = [["x" for _ in range(self.shape[0])] for _ in range(self.shape[0])]
        for alt_1 in range(self.shape[0]):
            for alt_2 in range(alt_1 + 1, self.shape[0]):
                if (self.iloc[alt_1].values >= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_1]
                    elements.add(alt_1)
                elif (self.iloc[alt_1].values <= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_2]
                    elements.add(alt_2)
                else:
                    result[alt_2][alt_1] = "н"
        return result

    compared = pd.DataFrame(compare(),
                            columns=[f'A{i}' for i in range(1, 11)],
                            index=[f'A{i}' for i in range(1, 11)])

    if show_comparison:
        compared.show("Pairwise comparison")

    return list(elements)


def normalize(self, aspirations: list[bool]):
    for col, aspiration in enumerate(aspirations):
        if not aspiration:
            for row in range(self.shape[0]):
                self.iloc[row, col] = (1 / self.iloc[row, col])


def show(self, header: str = ""):
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    pd.DataFrame.border_optimize = border_optimize
    pd.DataFrame.optimize = optimize
    pd.DataFrame.normalize = normalize
    pd.DataFrame.show = show
    main()
