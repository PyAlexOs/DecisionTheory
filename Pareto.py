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

    elements = dataframe.optimize()
    optimized = dataframe.iloc[elements]
    optimized.normalize(aspirations)
    optimized.show("Optimized")

    elements = dataframe.optimize()
    optimized = dataframe.iloc[elements]
    optimized.normalize(aspirations)
    elements = optimized.border_optimize({"bigger": [[1, 1]]})
    if len(elements) > 0:
        border_optimized = optimized.iloc[elements]
        border_optimized.show("Border optimized")
    else:
        print("Failed to optimize the set using boundaries")


def border_optimize(self, borders: dict[str, list[...]]) -> list[str]:
    """ Narrows down the set of optimal options using borders """
    elements = set()

    for (key, value) in borders.items():
        if key not in ["bigger", ">", "smaller", "<", "equals", "="]:
            raise KeyError('The keys can only be as follows: ["bigger", ">", "smaller", "<", "equals", "="]')

        if not isinstance(value, list) or len(value) == 0:
            raise ValueError('The value must be a list of two numbers [index, condition]')

        for condition in value:
            if (not isinstance(condition, list) or
                    len(condition) != 2 or
                    condition[0] not in range(0, self.shape[0]) or
                    not all([i.isdigit() or i == "." for i in str(condition[1])])):
                raise ValueError('The value must be a list of two numbers [index, condition]')

            for row in range(self.shape[0]):
                if key in ["bigger", ">"]:
                    print(self.index)
                    print(row, condition[0])
                    print(self.loc[row, condition[0]])
                    if self.loc[row, condition[0]] > condition[1]:
                        elements.add(row)
                elif key in ["smaller", "<"]:
                    if self.loc[row, condition[0]] < condition[1]:
                        elements.add(row)
                else:
                    if self.loc[row, condition[0]] == condition[1]:
                        elements.add(row)

    return list(elements)


def optimize(self) -> list[str]:
    """ Finds the optimal Pareto set """
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
    compared.show("Pairwise comparison")

    return list(elements)


def normalize(self, aspirations: list[bool]):
    """ Normalizes values whose indexes in the transmitted list are false """
    for col, aspiration in enumerate(aspirations):
        if not aspiration:
            for row in range(self.shape[0]):
                self.iloc[row, col] = (1 / self.iloc[row, col])


def show(self, header: str = ""):
    """ Prints dataframe with header """
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    import pandas as pd
    import shutil

    pd.DataFrame.border_optimize = border_optimize
    pd.DataFrame.optimize = optimize
    pd.DataFrame.normalize = normalize
    pd.DataFrame.show = show
    main()
