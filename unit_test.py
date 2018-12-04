import unittest
import solution
import func


class SolutionTestCase(unittest.TestCase):
    def setUp(self):
        self.sol = solution(0, 1, 1)
        self.sol.sol_random()
        # return super(SolutionTestCase, self).setUp()

    def tearDown(self):
        return super(SolutionTestCase, self).tearDown()

    def test_cal_fitness(self):
        #func = Func()
        #fitness = func
        pass

    def test_change_bit(self):
        self.sol.change_bit(0, 0.1)


if __name__ == "__main__":

    pass
