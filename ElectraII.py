import pandas as pd
import numpy as np
import graphviz as gv
import shutil

KEYS = ["Стоимость машины",
        "Расход топлива",
        "Вместительность багажника",
        "Мощность двигателя",
        "Клиренс"]
COST = [5, 4, 3, 3, 2]
RANKS = [{2.5, 3.5, 4.5, 5.5},
         {6., 7.5, 9.},
         {500., 600.},
         {170., 200., 230.},
         {200., 220., 230.}]
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

    dataframe.range_rank(RANKS, COST)
    print()
    preferences = dataframe.get_matrix(COST, ASPIRATIONS, show_weights=False)  # show weights for report
    make_graph(preferences, threshold=1.5)


def threshold_preferences(preferences: np.ndarray,
                          threshold: float) -> np.ndarray:
    """ Dilutes the matrix by a threshold value """
    for row in range(preferences.shape[0]):
        for col in range(preferences.shape[1]):
            if preferences[row, col] < threshold:
                preferences[row, col] = 0

    return preferences


def make_graph(preferences: np.ndarray,
               threshold: float = 1):
    """ Generates a preference graph """
    graph = gv.Digraph(name='preference graph')
    for row in range(preferences.shape[0]):
        graph.node(str(row + 1))
        for col in range(preferences.shape[1]):
            if preferences[row, col] >= threshold:
                graph.edge(str(row + 1), str(col + 1))

    graph.render(f'files/preference_graph(threshold-{threshold}).gv', view=True)


def get_matrix(self: pd.DataFrame,
               cost: list[int],
               aspirations: list[bool],
               show_matrix: bool = True,
               show_weights: bool = True) -> np.ndarray:
    """ Getting a matrix with preference weights """
    matrix = np.full((self.shape[0], self.shape[0]), "x", dtype=np.dtype("U4"))
    for i, (_, alt_1) in enumerate(self.iterrows()):
        for j, (_, alt_2) in enumerate(self.iloc[i + 1:].iterrows(), i + 1):
            if show_weights:
                print(f"Рассмотрим альтернативы {i + 1} и {j + 1} (i={i + 1}, j={j + 1}):")

            p, n = [], []
            for col in range(len(self.columns.values)):
                if ((alt_1.iloc[col] > alt_2.iloc[col] and aspirations[col]) or
                        (alt_1.iloc[col] < alt_2.iloc[col] and not aspirations[col])):
                    p.append(cost[col])
                    n.append(0)
                elif ((alt_1.iloc[col] < alt_2.iloc[col] and aspirations[col]) or
                        (alt_1.iloc[col] > alt_2.iloc[col] and not aspirations[col])):
                    n.append(cost[col])
                    p.append(0)

                else:
                    n.append(0)
                    p.append(0)

            if sum(p) == sum(n):
                matrix[i, j] = "–"
                matrix[j, i] = "–"
                if show_weights:
                    print(f"P{i + 1}{j + 1} = {" + ".join(list(map(lambda x: str(x), p)))} = {sum(p)}")
                    print(f"N{i + 1}{j + 1} = {" + ".join(list(map(lambda x: str(x), n)))} = {sum(n)}")
                    print(f"D{i + 1}{j + 1} = P{i + 1}{j + 1}/N{i + 1}{j + 1} = "
                          f"{sum(p)}/{sum(n)} = 1 – отбрасываем, согласно правилу D ≤ 1\n"
                          f"Тогда отбрасывается и D{j + 1}{i + 1}")
                continue

            result = np.inf
            if sum(n) != 0:
                result = round(sum(p) / sum(n), 2)

            if sum(p) > sum(n):
                matrix[i, j] = str(result) if result != np.inf else "∞"
                matrix[j, i] = "–"

            if show_weights:
                print(f"P{i + 1}{j + 1} = {" + ".join(list(map(lambda x: str(x), p)))} = {sum(p)}")
                print(f"N{i + 1}{j + 1} = {" + ".join(list(map(lambda x: str(x), n)))} = {sum(n)}")
                print(f"D{i + 1}{j + 1} = P{i + 1}{j + 1}/N{i + 1}{j + 1} = "
                      f"{sum(p)}/{sum(n)} = {result if result != np.inf else "∞"} " +
                      (("> 1 " if result != np.inf else "") + "– принимаем." if sum(p) > sum(n)
                       else "< 1 – отбрасываем."))

            result = np.inf
            if sum(p) != 0:
                result = round(sum(n) / sum(p), 2)

            if sum(n) > sum(p):
                matrix[i, j] = "–"
                matrix[j, i] = str(result) if result != np.inf else "∞"

            if show_weights:
                print(f"P{j + 1}{i + 1} = {" + ".join(list(map(lambda x: str(x), n)))} = {sum(n)}")
                print(f"N{j + 1}{i + 1} = {" + ".join(list(map(lambda x: str(x), p)))} = {sum(p)}")
                print(f"D{j + 1}{i + 1} = P{j + 1}{i + 1}/N{j + 1}{i + 1} = "
                      f"{sum(n)}/{sum(p)} = {result if result != np.inf else "∞"} " +
                      (("> 1 " if result != np.inf else "") + "– принимаем." if sum(n) > sum(p)
                       else "≤ 1 – отбрасываем."))

    if show_matrix:
        pd.DataFrame(matrix,
                     index=[f"A{i + 1}" for i in range(matrix.shape[0])],
                     columns=[f"A{i + 1}" for i in range(matrix.shape[1])]).show("Preference weights")

    preferences = np.full(matrix.shape, 0., dtype=np.float_)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col] in ["–", "x"]:
                continue
            if matrix[row, col] == "∞":
                preferences[row, col] = np.inf
            else:
                preferences[row, col] = matrix[row, col]

    return preferences


def range_rank(self: pd.DataFrame,
               ranks: list[set[float]],
               cost: list[int],
               show_rank: bool = True):
    """ Ranks the criteria according to the cost scale, taking into account the boundaries """
    ranged = np.full((self.shape[0], self.shape[1]), False)
    for col, rank in enumerate(ranks):
        rank = sorted(list(rank))
        rank.insert(0, 0)
        rank.append(np.inf)

        for border in range(len(rank) - 1):
            for row_number, (_, row) in enumerate(self.iterrows()):
                if not ranged[row_number, col]:
                    if rank[border] <= row.iloc[col] < rank[border + 1]:
                        self.iloc[row_number, col] = (np.arange(start=cost[col],
                                                                stop=(cost[col] * len(rank)),
                                                                step=cost[col]))[border]
                        ranged[row_number, col] = True

    if show_rank:
        self.show("Ranked")


def show(self: pd.DataFrame,
         header: str = ""):
    """ Displays a table with the name """
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    pd.DataFrame.get_matrix = get_matrix
    pd.DataFrame.range_rank = range_rank
    pd.DataFrame.show = show
    main()
