import random
import math
import numpy as np
from global_processes import ACTUAL_CLASS
from global_processes import TEST_CLASS
from global_processes import TRAIN_STR
from global_processes import TEST_STR
from global_processes import NUM_WEIGHT


def sig(x):
    return 1 / (1 + np.exp(-x))


class Weights:
    def __init__(self):
        self.fitness = None
        self.actual_classification = ACTUAL_CLASS
        self.test_classification = TEST_CLASS
        self.num_weights = NUM_WEIGHT
        self.weights = self.random_w()

    def random_w(self):
        w = [random.uniform(-1, 1) for i in range(self.num_weights)]
        return w

    def upgrade_fitness(self): #calculate and upgrade fitness
        y_get = self.cal_classification(TRAIN_STR)
        MSE = np.square(np.subtract(self.actual_classification, y_get)).mean()
        RMSE = math.sqrt(MSE)
        self.fitness = 1 / (1 + RMSE)

    def cal_test_fitness(self):
        y_get = self.cal_classification(TEST_STR)
        MSE = np.square(np.subtract(self.test_classification, y_get)).mean()
        RMSE = math.sqrt(MSE)
        return 1 / (1 + RMSE)

    def cal_classification(self, strings):
        class_list = []
        # build the neuron network by the weights
        hidden_layer_1 = [None for i in range(8)]
        hidden_layer_2 = [None for i in range(8)]
        for string in strings:
            index_weights = 0
            # cal hidden layer 1
            for i, n in enumerate(hidden_layer_1):
                summ = 0
                for j in range(len(string)):
                    summ = summ + (float(string[j]) * self.weights[index_weights])
                    index_weights += 1
                summ = summ + (1 * self.weights[index_weights])  # bias
                index_weights += 1
                output = sig(summ)
                hidden_layer_1[i] = output

            # cal hidden layer 2
            for i, n in enumerate(hidden_layer_2):
                summ = 0
                for j in range(len(hidden_layer_1)):
                    summ = summ + (hidden_layer_1[j] * self.weights[index_weights])
                    index_weights += 1
                summ = summ + (1 * self.weights[index_weights])  # bias
                index_weights += 1
                output = sig(summ)
                hidden_layer_2[i] = output

            # cal output neuron
            summ = 0
            for j in range(len(hidden_layer_2)):
                summ = summ + (hidden_layer_2[j] * self.weights[index_weights])
                index_weights += 1
            summ = summ + (1 * self.weights[index_weights])  # bias
            index_weights += 1
            output = sig(summ)
            output_neuron = output
            if output_neuron > 0.5:
                classification = 1
            else:
                classification = 0
            class_list.append(classification)
        return class_list










