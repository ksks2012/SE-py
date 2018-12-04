import numpy as np
import sys

import func


class Solution():

    def __init__(self, lower, upper, dimension):
        self.lower = lower
        self.upper = upper
        self.dimension = dimension

        self.fitness = sys.float_info.max

        self.region = 0

        self.sol_init()
        # self.sol_random(lower, upper, dimension)

    def sol_init(self):
        self.sol_list = np.zeros(self.dimension)

    def sol_random(self):
        self.sol_list = np.random.uniform(
            self.lower, self.upper, self.dimension)

    def sol_print(self):
        print("region ", self.region)
        print("x ",  self.sol_list)
        print("fitness ", self.fitness)

    def set_identity_bit(self, idx, lower, upper, change=False):
        if self.sol_list[idx] < lower or self.sol_list[idx] > upper or change:
            self.sol_list[idx] = np.random.uniform(
                lower, upper)

    def change_bit(self, idx, velocity):
        #rand = np.random.uniform(-0.5, 0.5)
        #rand = np.random.uniform(0, 1)
        #rand = np.random.uniform(-1.5, 1.5)
        #rand = np.random.uniform(-1, 2)
        rand = np.random.uniform(-2, 2)

        #rand = 0.7

        self.sol_list[idx] = (1 - rand) * self.sol_list[idx] + rand * velocity

        if self.sol_list[idx] < self.lower or self.sol_list[idx] > self.upper:
            self.rand_bit(idx)

    def change_bit_fix(self, idx, velocity, rand):

        self.sol_list[idx] = (1 - rand) * self.sol_list[idx] + rand * velocity

        if self.sol_list[idx] < self.lower or self.sol_list[idx] > self.upper:
            self.rand_bit(idx)

    def rand_bit(self, idx):
        self.sol_list[idx] = np.random.uniform(
            self.lower, self.upper)

    def change_identity_bit(self, idx, lower, upper):
        self.sol_list[idx] = np.random.uniform(
            lower, upper)


if __name__ == "__main__":

    sol = Solution(-1, 5, 10)
    print(sol.sol_list)
