import pandas as pd
import shutil
from numbers import Number

ASPIRATIONS = [False, False, True, True, True]


def main():
    dataframe = pd.read_csv('alternatives.csv')
    dataframe.index = [f'A{i}' for i in range(1, 11)]
    dataframe = dataframe.iloc[:,1:]
    dataframe.show("Alternatives")
    dataframe.normalize(ASPIRATIONS)
    dataframe.show("Normalized")

    dataframe.copy().border_optimization(
        borders={
            "<": [[0, 4.5], [3, 200]],
            "bigger": [[2, 500], [4, 200]]
        }
    )

    dataframe.copy().sub_optimization(
        borders={"bigger": [[2, 550], [3, 200]]},
        key_criterion_index=1
    )

    dataframe.copy().lexicographic_optimization([4, 0, 1, 2, 3])


def lexicographic_optimization(self: pd.DataFrame,
                               criterion_importance_index: list,
                               show_result: bool = True) -> pd.DataFrame | None:
    """ Searches for the optimal solution in the given order of importance of the criteria """
    if len(set(criterion_importance_index)) != self.shape[1]:
        raise ValueError("Wrong criterion importance list")

    result = pd.DataFrame(self)
    for criterion in criterion_importance_index:
        if result.count().iloc[0] == 1:
            break

        best = result.loc[result.iloc[:, criterion].idxmax()]
        for index, row in result.iterrows():
            if row.iloc[criterion] < best.iloc[criterion]:
                result.drop(index, inplace=True)

    if show_result:
        result.normalize(ASPIRATIONS)
        result.show("Lexicographic optimized")

    return result


def sub_optimization(self: pd.DataFrame,
                     borders: dict[str, list[list[float, float]]],
                     key_criterion_index: int,
                     show_border_optimized: bool = True,
                     show_result: bool = True) -> pd.DataFrame | None:
    """ Finds the optimal solution according to the main criterion, which is suitable for additional conditions """
    border_optimized_elements = self.get_optimal_range(borders)
    if len(border_optimized_elements) == 0:
        print("Failed to optimize the set using boundaries")
        return

    border_optimized = self.iloc[border_optimized_elements]
    if show_border_optimized:
        border_optimized.normalize(ASPIRATIONS)
        border_optimized.show("Border optimized")

    result_element = border_optimized.iloc[:, key_criterion_index].idxmax()
    if not ASPIRATIONS[key_criterion_index]:
        result_element = border_optimized.iloc[:, key_criterion_index].idxmin()

    result = border_optimized.loc[[result_element]]
    if show_result:
        self.normalize(ASPIRATIONS)
        result.show("Suboptimization")

    return result


def border_optimization(self: pd.DataFrame,
                        borders: dict[str, list[list[int, float]]],
                        show_optimized: bool = True,
                        show_border_optimized: bool = True,
                        show_result: bool = True) -> pd.DataFrame | None:
    """ Finds Pareto optimal alternatives that satisfy additionally specified conditions """
    optimized_elements = self.optimize()
    if show_optimized:
        optimized = self.iloc[optimized_elements]
        optimized.normalize(ASPIRATIONS)
        optimized.show("Optimized")

    border_optimized_elements = self.get_optimal_range(borders)
    if len(border_optimized_elements) == 0:
        print("Failed to optimize the set using boundaries")
        return

    if show_border_optimized:
        border_optimized = self.iloc[border_optimized_elements]
        border_optimized.normalize(ASPIRATIONS)
        border_optimized.show("Border optimized")

    elements = list(set(border_optimized_elements) & set(optimized_elements))
    if len(elements) == 0:
        print("\nPareto- and border-optimized set intersection is empty")
        return

    if show_result:
        result = self.iloc[elements]
        result.normalize(ASPIRATIONS)
        result.show("Pareto- and border-optimized set intersection")

    return self.iloc[elements]


def get_optimal_range(self: pd.DataFrame,
                      borders: dict[str, list[list[int, float]]]) -> list[int]:
    """ Filters out alternatives that are not suitable for the specified conditions """
    if len(borders) == 0:
        raise ValueError('Empty borders.')

    result = [i for i in range(self.shape[0])]  # indexes of alternatives within the specified constraints
    for (key, value) in borders.items():
        if key not in ["bigger", ">", "smaller", "<", "equals", "="]:
            raise KeyError('The keys can only be as follows: ["bigger", ">", "smaller", "<", "equals", "="]')

        if not isinstance(value, list) or len(value) == 0:
            raise ValueError('The value must be a dict like {condition: [[param_index_1, border_1], ...], ...}')

        for condition in value:
            if (not isinstance(condition, list) or
                    len(condition) != 2 or
                    condition[0] not in range(0, self.shape[1]) or
                    not isinstance(condition[1], Number)):
                raise ValueError('The value must be a list like: [param_index, border]')

            convert = False
            if not ASPIRATIONS[condition[0]]:
                condition[1] = 1 / condition[1]
                convert = True

            for row in range(self.shape[0]):
                if key in ["bigger", ">"] or (convert and key in ["smaller", "<"]):
                    if not self.iloc[row, condition[0]] > condition[1] and row in result:
                        result.remove(row)
                elif key in ["smaller", "<"] or (convert and key in ["bigger", ">"]):
                    if not self.iloc[row, condition[0]] < condition[1] and row in result:
                        result.remove(row)
                else:
                    if not self.iloc[row, condition[0]] == condition[1] and row in result:
                        result.remove(row)

    return result


def optimize(self: pd.DataFrame,
             show_comparison: bool = True) -> list[int]:
    """ Finds incomparable and Pareto optimal solutions """
    elements = set()  # indexes of incomparable and Pareto optimal solutions

    def compare() -> list[list[str]]:
        """ Compares alternatives by criteria in pairs, returns a matrix reflecting the superiority of row over col, taking into account incomparable alternatives """
        result = [["x" for _ in range(self.shape[0])] for _ in range(self.shape[0])]
        incomparable_elements = [i for i in range(self.shape[0])]

        def mark_as_comparable(alt):
            if alt in incomparable_elements:
                incomparable_elements.remove(alt)

        for alt_1 in range(self.shape[0]):
            for alt_2 in range(alt_1 + 1, self.shape[0]):
                if (self.iloc[alt_1].values >= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_1]
                    elements.add(alt_1)
                    mark_as_comparable(alt_1)
                    mark_as_comparable(alt_2)

                elif (self.iloc[alt_1].values <= self.iloc[alt_2].values).all():
                    result[alt_2][alt_1] = self.index.values[alt_2]
                    elements.add(alt_2)
                    mark_as_comparable(alt_1)
                    mark_as_comparable(alt_2)

                else:
                    result[alt_2][alt_1] = "н"

        for element in incomparable_elements:
            elements.add(element)

        return result

    compared = pd.DataFrame(compare(),
                            columns=[f'A{i}' for i in range(1, 11)],
                            index=[f'A{i}' for i in range(1, 11)])

    if show_comparison:
        compared.show("Pairwise comparison")

    return list(elements)


def normalize(self: pd.DataFrame,
              aspirations: list[bool]):
    """ Converts the values of criteria with negative aspirations """
    for col, aspiration in enumerate(aspirations):
        if not aspiration:
            self.iloc[:, col] = (1 / self.iloc[:, col])


def show(self: pd.DataFrame,
         header: str = ""):
    """ Outputs the dataframe with the given name """
    console_width = shutil.get_terminal_size().columns
    filler = "-" * ((console_width - len(header)) // 2)
    print(f"\n{filler}{header}{filler}\n" + self.to_string())


if __name__ == "__main__":
    pd.DataFrame.lexicographic_optimization = lexicographic_optimization
    pd.DataFrame.sub_optimization = sub_optimization
    pd.DataFrame.border_optimization = border_optimization
    pd.DataFrame.get_optimal_range = get_optimal_range
    pd.DataFrame.optimize = optimize
    pd.DataFrame.normalize = normalize
    pd.DataFrame.show = show
    main()
