import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum


FUNC = "x_1 + 2x_2 -> max"
INEQUALITIES = [
    "5x_1 - 2x_2 <= 7",
    "-x_1 + 2x_2 <= 5",
    "x_2 + x_2 >= 6",
    "x_1 >= 0",
    "x_2 >= 0"
]


class Sign(Enum):
    """ Storing the inequality sign """
    L   = "<"
    LE  = "<="
    EQ  = "="
    GE  = ">="
    G   = ">"


@dataclass
class Inequality:
    """ Inequality as the sum of terms and the ratio of the inequation to zero """
    summand: list[tuple[float, str]]
    sign: Sign


@dataclass
class Function:
    """ Function as the sum of terms and aspiration """
    summand: list[tuple[float, str]]
    aspiration: bool


class Optimizer:
    func: Inequality
    equations: list[Inequality]

    def __init__(self,
                 func: str,
                 acceptable_values: list[str]):
        self.func = self.__parse_func(func)
        self.aspiration = self.func.sign
        self.inequality = list(map(self.__parse_inequality, acceptable_values))
        print(self.func, self.inequality)

    def __parse_func(self,
                     func: str):
        if "min" in func:
            pass
            

    def __parse_inequality(self,
                           inequality: str) -> Inequality:
        """ Converts an inequality from a string representation to a handy one """
        signs = ["<", "<=", "=", ">=", ">"]
        for sign in signs:
            if sign in inequality:
                inequality = " ".join(inequality.split(sign))
                inequality_sign = Sign(sign)
                break

        return Inequality([tuple([0, inequality])], inequality_sign)
    
    def __parse_terms(self,
                      )




def main():
    optimizer = Optimizer(FUNC, INEQUALITIES)


if __name__ == "__main__":
    main()
