import sys
import numpy as np
from copy import deepcopy

from solution import Solution
from func import Func


def standardization(data):
    if (data.std() != 0):
        data = (data - data.mean()) / data.std()
        data = offset(data)
    return data


def normalization(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def offset(data):
    data = (data - data.min() * 2)
    return data


class SE():
    def __init__(self, parameter, func):
        # user setting
        self.parameter = parameter

        self.p_size = parameter["regions"] * parameter["samples"]

        # size of sampleV (regions * samples * searchers * 2)
        self.v_size = self.p_size * parameter["searchers"] * 2

        # function
        self.func = func
        self.lower, self.upper = self.func.get_range()
        self.dimention = self.func.get_dimention()

        self.sample = self.malloc_sol(self.p_size)
        self.sampleV = self.malloc_sol(self.v_size)
        self.searcher = self.malloc_sol(parameter["searchers"])

        self.identity_bit = 0
        self.identity_range = []

        self.region_choose = np.zeros(parameter["regions"], dtype=float)
        self.region_un_choose = np.zeros(parameter["regions"], dtype=float)

        # the time that region been chosen
        self.T = np.zeros(parameter["regions"])
        self.V = np.zeros((parameter["searchers"],
                           parameter["regions"]), dtype=float)
        self.M = np.zeros(parameter["regions"])
        self.expected_value = np.zeros(
            parameter["regions"] * parameter["searchers"])

        self.region_best = self.malloc_sol(parameter["regions"])
        self.best_sol = Solution(self.lower, self.upper, self.dimention)
        # self.best_sol.fitness = sys.float_info.max

    # intitial solution
    def malloc_sol(self, size):

        sol = [Solution(self.lower, self.upper, self.dimention)
               for _ in range(size)]

        return sol

    # 1 Resource Arrangement
    def resource_arrangment(self):

        # 1.1
        self.devide_sol()

        # 1.1.1
        for i in range(self.parameter["searchers"]):
            self.searcher[i] = deepcopy(
                self.assign_sol_region(self.searcher[i], i))
            # print("searcher")
            # self.searcher[i].sol_print()
        self.record_best_sol()

        # 1.1.2 initialize the sample solutions
        for i in range(self.parameter["regions"]):
            for j in range(self.parameter["samples"]):
                idx = i * self.parameter["samples"] + j
                self.sample[idx] = deepcopy(
                    self.assign_sol_region(self.sample[idx], i))
                # print("sample", i)
                # self.sample[idx].sol_print()

        # 1.2
        for i in range(self.parameter["regions"]):
            for j in range(self.parameter["samples"]):
                idx = i * self.parameter["samples"] + j
                # print(self.region_best[i].fitness,
                #       self.sample[idx].fitness, idx)
                if self.sample[idx].fitness < self.region_best[i].fitness:
                    self.region_best[i] = deepcopy(self.sample[idx])

            self.region_choose[i] += 1
            self.region_un_choose[i] = 1
        # print("region_best")
        # self.region_best[i].sol_print()

        pass

    def devide_sol(self):
        region = self.parameter["regions"]
        for i in range(region + 1):
            self.identity_range.append(
                self.lower + (self.upper - self.lower) / region * i)

        # print(self.identity_range)

    def assign_sol_region(self, sol, idx):

        sol.region = idx

        sol.sol_random()

        sol.set_identity_bit(
            self.identity_bit, self.identity_range[idx], self.identity_range[idx + 1])
        sol.fitness = self.cal_fitness(sol)

        return sol

    # 2 Vision Search
    def vision_search(self):
        # 2.1
        self.transit()

        # 2.2
        self.compute_expected_value()

        # 2.3 sampleV -> sample
        self.select()

        # 2.4 vision_selection (searcher select region)
        self.vision_selection()
        return

    # 2.1
    def transit(self):
        for i in range(self.parameter["searchers"]):
            for j in range(self.p_size):

                searcher = deepcopy(self.searcher)
                sample = deepcopy(self.sample)

                v_idx = (i * self.p_size + j) * 2

                self.sampleV[v_idx], self.sampleV[v_idx +
                                                  1] = deepcopy(self.crossover(searcher[i], sample[j]))

                # self.sampleV[v_idx].sol_print()
                # self.sampleV[v_idx + 1].sol_print()
                # print("regions ", v_idx,
                #       self.sampleV[v_idx].region, self.sampleV[v_idx+1].region)
            # print("searcher")
            # self.searcher[i].sol_print()
        pass

    def crossover(self, searcher, sample):

        # print(id(searcher))

        if searcher.region == sample.region:
            searcher.sol_list[self.identity_bit] = (
                searcher.sol_list[self.identity_bit] * 0.7 + sample.sol_list[self.identity_bit]) * 0.3
            sample.sol_list[self.identity_bit] = (
                searcher.sol_list[self.identity_bit] * 0.3 + sample.sol_list[self.identity_bit]) * 0.7
            # searcher.change_bit(self.identity_bit,
            #                     sample.sol_list[self.identity_bit])
            # sample.change_bit(self.identity_bit,
            #                   searcher.sol_list[self.identity_bit])
        else:
            searcher.sol_list[self.identity_bit] = sample.sol_list[self.identity_bit]
            searcher.region = sample.region

        # sol = [searcher, sample, deepcopy(searcher)]
        sol = [searcher, sample, (searcher)]

        # print(id(sol[0]), id(sol[2]))
        rand = np.random.uniform(-2, 2)
        for i in range(2):
            rand_idx = np.random.randint(
                self.identity_bit + 1, self.dimention)

            sol[i].change_bit(rand_idx, sol[i+1].sol_list[rand_idx])
            #sol[i].change_bit_fix(rand_idx, sol[i+1].sol_list[rand_idx], rand)
            sol[i].fitness = self.cal_fitness(sol[i])

            # for j in range(self.identity_bit + 1, self.dimention):
            #     sol[i].change_bit_fix(j, sol[i+1].sol_list[j], rand)
            #     sol[i].fitness = self.cal_fitness(sol[i])

            # mutation
            if np.random.uniform(0, 1) > 0.8:
                rand_idx = np.random.randint(
                    self.identity_bit + 1, self.dimention)
                sol[i].rand_bit(rand_idx)

        return sol[0], sol[1]

    # 2.2
    def compute_expected_value(self):

        # for i in range(self.p_size):
        #     print(self.sample[i].fitness)

        # 3.2.1 M
        for i in range(self.parameter["regions"]):
            # start = i * self.parameter["samples"]
            # end = (i + 1) * self.parameter["samples"]

            all_sample_fitness = sum(
                Solution.fitness for Solution in self.sample)
            # print("sum ", all_sample_fitness)

            self.M[i] = 1 - self.region_best[i].fitness / all_sample_fitness

        self.M = standardization(self.M)
        # self.M = offset(self.M)
        # self.M = normalization(self.M)

        # 3.2.2 V
        # for i in range(self.v_size):
        #     self.sampleV[i].sol_print()
        region_sampleV = self.v_size / self.parameter["regions"]
        for i in range(self.parameter["searchers"]):
            for j in range(self.parameter["regions"]):

                start = i * region_sampleV + j
                end = start + self.p_size

                all_sampleV_fitness = sum(
                    Solution.fitness for Solution in self.sampleV[int(start):int(end)])
                # print(all_sampleV_fitness, 2.0 * self.parameter["samples"])
                self.V[i][j] = all_sampleV_fitness / \
                    (2.0 * self.parameter["samples"])

        self.V = standardization(1 / self.V)
        # self.V = offset(self.V)
        # self.V = normalization(1 / self.V)

        # 3.2.3 T
        self.T = self.region_un_choose / self.region_choose
        # self.T = normalization(T)
        self.T = standardization(self.T)
        # self.T = offset(self.T)
        # 3.2.4 EV
        for i in range(self.parameter["searchers"]):
            for j in range(self.parameter["regions"]):
                idx = j * self.parameter["regions"] + i
                self.expected_value[idx] = self.T[i] * self.V[i][j] * self.M[j]
        # print(self.M)
        # print(self.V)
        # print(self.T, self.region_un_choose, self.region_choose)
        # print("EV ", self.expected_value)

    # 2.3
    def select(self):
        for i in range(self.parameter["searchers"]):
            for j in range(self.parameter["regions"]):
                for k in range(self.parameter["samples"]):
                    idx = j * self.parameter["samples"] + k

                    v_idx = (idx << 1) + i * self.p_size * 2
                    # print(v_idx, idx)
                    # print("regions ",
                    #       self.sampleV[v_idx].region, self.sample[idx].region)

                    if (self.sampleV[v_idx].fitness < self.sample[idx].fitness):
                        self.sample[idx] = deepcopy(self.sampleV[v_idx])
                    if (self.sampleV[v_idx + 1].fitness < self.sample[idx].fitness):
                        self.sample[idx] = deepcopy(self.sampleV[v_idx + 1])

                    if self.sample[idx].fitness < self.region_best[j].fitness:
                        self.region_best[j] = deepcopy(self.sample[idx])

                # print("region best", j)
                # self.region_best[j].sol_print()
        pass

    # 2.4
    def vision_selection(self):
        self.region_un_choose += 1
        # for i in range(self.parameter["regions"]):
        #     print(self.region_un_choose[i])

        for i in range(self.parameter["searchers"]):
            # np.random.randint(0, self.parameter["regions"])
            # print("3.4 self.searcher[i].region ", i, self.searcher[i].region)
            play_idx = self.searcher[i].region

            for j in range(self.parameter["players"]):
                rand_idx = np.random.randint(0, self.parameter["regions"])

                play_idx_offset = play_idx + i * self.parameter["regions"]
                rand_idx_ofset = rand_idx + i * self.parameter["regions"]

                # print("3.4 r_idx ", rand_idx, play_idx)
                # print("3.4 ev",
                #       self.expected_value[rand_idx_ofset], self.expected_value[play_idx_offset])

                if self.expected_value[rand_idx_ofset] > self.expected_value[play_idx_offset]:
                    play_idx = rand_idx

            start = play_idx * self.parameter["samples"]
            stop = start + self.parameter["samples"]
            for j in range(start, stop):
                # print("3.4 reigons",
                #       self.sample[j].region, self.searcher[i].region)
                if self.sample[j].fitness < self.searcher[i].fitness:
                    self.searcher[i] = deepcopy(self.sample[j])

            # print(i, play_idx, self.region_un_choose, self.region_choose)
            self.region_choose[play_idx] += 1
            self.region_un_choose[play_idx] = 1
            # print()

    # 3 Marketing Research

    def marketing_research(self):
        # for i in range(self.parameter["searchers"]):
        #     print(i, self.searcher[i].region)

        # 4.1
        for i in range(self.parameter["regions"]):
            if self.region_un_choose[i] > 1:
                self.region_choose[i] = 1

        # 4.2
        for i in range(self.parameter["searchers"]):
            if self.searcher[i].fitness < self.best_sol.fitness:
                self.best_sol = deepcopy(self.searcher[i])
        return

    # use fitness function
    def cal_fitness(self, sol):
        return self.func.func(sol.sol_list)

    def record_best_sol(self):
        for i in range(self.parameter["searchers"]):
            if self.searcher[i].fitness < self.best_sol.fitness:
                self.best_sol = deepcopy(self.searcher[i])


if __name__ == "__main__":

    # user input
    parameter = {
        "runs": int(sys.argv[1]),
        "itr": int(sys.argv[2]),
        "searchers": int(sys.argv[3]),
        "regions": int(sys.argv[4]),
        "samples": int(sys.argv[5]),
        "players": int(sys.argv[6]),
        "eva": int(sys.argv[7]),
        "func": int(sys.argv[8]),
    }

    func = Func(parameter["func"])

    se = SE(parameter, func)

    se.resource_arrangment()

    for i in range(parameter["itr"]):
        # for i in range(10010):
        se.vision_search()
        se.marketing_research()
        print(i)
        se.best_sol.sol_print()
