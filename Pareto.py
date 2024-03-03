KEYS = ["Стоимость машины",
        "Расход топлива",
        "Вместительность багажника",
        "Мощность двигателя",
        "Клиренс"]
TABLE = [['Mazda CX-5', 4.35, 7.1, 442, 156, 210],
         ['Toyota Land Cruiser Prado', 4.32, 11.0, 621, 163, 215],
         ['Audi A4', 3.99, 7.2, 495, 245, 180],
         ['Kia Sportage', 3.13, 8.2, 540, 150, 181],
         ['Volvo XC90', 5.58, 5.7, 721, 249, 238],
         ['Subaru Outback', 5.63, 7.3, 522, 188, 213],
         ['Nissan X-Trail', 2.21, 5.3, 497, 130, 210],
         ['Mercedes GLC', 5.12, 5.5, 620, 197, 180],
         ['Jeep Wrangler', 4.92, 11.3, 142, 272, 277],
         ['УАЗ Патриот', 1.94, 11.5, 1130, 150, 210]]


def main():
    data = dict(zip(KEYS, [[TABLE[row][col]
                            for row in range(len(TABLE))]
                           for col in range(len(TABLE[0]))][1:]))

    dataframe = pd.DataFrame(data=data, index=[f'A{i}' for i in range(1, 11)])
    dataframe.show("Alternatives")

    aspirations = [False, False, True, True, True]
    dataframe.normalize(aspirations)
    dataframe.show("Normalized")

    dataframe.optimize()


def optimize(self):
    def compare() -> list[list[...]]:
        result = [["x" for _ in range(self.shape[0])] for _ in range(self.shape[0])]
        for alt_1 in range(self.shape[0]):
            for alt_2 in range(alt_1 + 1, self.shape[0]):
                if (self.iloc[alt_1].values >= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_1]
                elif (self.iloc[alt_1].values <= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_2]
                else:
                    result[alt_2][alt_1] = "н"

        return result

    compared = pd.DataFrame(compare(),
                            columns=[f'A{i}' for i in range(1, 11)],
                            index=[f'A{i}' for i in range(1, 11)])
    compared.show("Pairwise comparison")


def normalize(self, aspirations: list[bool]):
    """ Normalizes values whose indexes in the transmitted list are false """
    for col, aspiration in enumerate(aspirations):
        if not aspiration:
            for row in range(self.shape[0]):
                self.iloc[row, col] = (1 / self.iloc[row, col])


def show(self, header: str):
    """ Prints dataframe with header """
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import shutil

    pd.DataFrame.optimize = optimize
    pd.DataFrame.normalize = normalize
    pd.DataFrame.show = show
    main()
