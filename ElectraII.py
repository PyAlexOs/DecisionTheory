import pandas as pd
import shutil

KEYS = ["Стоимость машины",
        "Расход топлива",
        "Вместительность багажника",
        "Мощность двигателя",
        "Клиренс"]
COST = [5, 4, 3, 3, 2]
RANKS = [{2.5, 4.5},
         {6., 8.5},
         {500., 600.},
         {190., 230.},
         {220., 230.}]
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


def main():
    data = dict(zip(KEYS, [[TABLE[row][col]
                            for row in range(len(TABLE))]
                           for col in range(len(TABLE[0]))][1:]))

    dataframe = pd.DataFrame(data=data, index=[f'A{i}' for i in range(1, 11)])
    dataframe.show("Alternatives")

    dataframe.range_rank(ASPIRATIONS, RANKS)


def normalize(self: pd.DataFrame,
              aspirations: list[bool]):
    for col, aspiration in enumerate(aspirations):
        if not aspiration:
            for row in range(self.shape[0]):
                self.iloc[row, col] = (1 / self.iloc[row, col])


def range_rank(self: pd.DataFrame,
               aspirations: list[bool],
               cost: list[int],
               ranks: list[set[float]],
               show_rank: bool = True):
    for col, aspiration in enumerate(aspirations):
        for row, _ in self.iterrows():
            for i, border in enumerate(sorted(ranks[col], reverse=not aspiration)):
                if 
                    self.loc[row, :].iloc[:, col] = cost[col] * (i + 1)
                    continue



    if show_rank:
        self.show("Ranked")


def show(self: pd.DataFrame,
         header: str = ""):
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    pd.DataFrame.normalize = normalize
    pd.DataFrame.range_rank = range_rank
    pd.DataFrame.show = show
    main()
