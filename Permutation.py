import random
import math
import numpy as np
from global_processes import ACTUAL_CLASS
from global_processes import TEST_CLASS
from global_processes import TRAIN_STR
from global_processes import TEST_STR
from global_processes import NUM_WEIGHT



class Permutation:
    def __init__(self):
        self.permutation = self.random_order()
        self.fitness = None
        self.actual_classification = ACTUAL_CLASS
        self.test_classification = TEST_CLASS
        self.num_weights = NUM_WEIGHT

    def random_p(self):
        p = [random.uniform(-1, 1) for i in self.num_weights]
        return p

    def upgrade_fitness(self): #calculate and upgrade fitness
        y_get = self.cal_classification(self, TRAIN_STR)
        MSE = np.square(np.subtract(self.actual_classification, y_get)).mean()
        RMSE = math.sqrt(MSE)
        self.fitness = 1 / (1 + RMSE)

    def cal_test_fitness(self):
        y_get = self.cal_classification(self, TEST_STR)
        MSE = np.square(np.subtract(self.test_classification, y_get)).mean()
        RMSE = math.sqrt(MSE)
        return 1 / (1 + RMSE)

    def cal_classification(self, strings):
        return 0
    #TODO : write



