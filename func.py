import numpy as np


class Func():
    def __init__(self, func_num):
        self.func_num = func_num

    def get_range(self):
        # [lb, ub]
        return {
            1: [-5.12, 5.12],
            2: [-100, 100],
            23: [-500, 500],
        }.get(self.func_num, "nothing")

    def get_dimention(self):
        # dimention
        return {
            1: 5,
            2: 10,
            23: 10,
        }.get(self.func_num, "nothing")

    def func(self, x):
        """x is position """
        #f = self.func_map(str(self.func_num))(x)
        f = self.func_map()(x)
        return f

    def func_map(self):
        return {
            1: self.F1,
            2: self.F2,
            # '3': self.F3,
            # '4': self.F4,
            # '5': self.F5,
            # '6': self.F6,
            # '7': self.F7,
            23: self.F23,

        }.get(self.func_num, 'nothing')

    # F1
    def F1(self, x):
        f = 25
        f += sum(np.ceil(x))
        return f

    # F2
    def F2(self, x):
        f = sum(x**2)
        return f

    # F23
    def F23(self, x):
        f = 418.9829 * self.get_dimention()
        # print(f)
        for i in range(self.get_dimention()):
            f -= (x[i] * np.sin(np.power(np.abs(x[i]), 0.5)))
        # print(f)
        return f
