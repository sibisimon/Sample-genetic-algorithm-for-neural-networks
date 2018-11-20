'''
Created on 31-Oct-2018
@author: Sibi Simon - 18201381
'''

# Importing packages
import math
import textwrap
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.preprocessing as skp

N_GENERATION = 500


def start_process():
    """
    Main sequential process to be run
    """

    # Fetching and modifying data
    test_df, training_df = fetch_and_modify_data()

    # Initializing GA object
    ga = GeneticAlgorithm(training_df=training_df)

    # Initial steps for GA
    ga.initialisation()
    ga.binaralize_population()
    ga.fitness_check()
    ga.selection()

    # Making n generations
    for index in range(0, N_GENERATION):
        print("Generation - " + str(index + 1))
        ga.cross_over()
        ga.mutation()
        ga.de_binarization()
        ga.remove_lowest_fits()
        ga.fitness_check()
        ga.selection()
        ga.set_max_weight()

    # Overall error calculation
    y_hat, error = calculate_yhat_error(ga, test_df)
    print('------------------------------- OVERALL PREDICTED ERROR ------------------------------\n' + str(error))

    # Scatter plot of highest fitness values
    data = ga.fitness_values_list
    show_fitness_scatter_plot(data)

    # Scatter Plot in 3D
    show_3d_scatter_plot(ga, test_df)


def calculate_yhat_error(ga, df):
    pr_body_fat = []
    weights_ar = ga.pr_weight.reshape(ga.wghts_matrix)
    features_x = np.matrix(df.iloc[:, [0, 1, 2, 3, 4]])
    target_y = np.matrix(df.iloc[:, [5]])
    for row in features_x:
        mul_matrix = row * weights_ar
        pr_body_fat.append(sum(map(ga._sigmoid, np.squeeze(np.array(np.sum(mul_matrix, axis=0))))))

    return pr_body_fat, sum([(a - b) ** 2 for a, b in zip(pr_body_fat, target_y)]) / len(target_y)


def show_fitness_scatter_plot(data):
    plt.scatter(range(1, len(data) + 1), data, alpha=0.5, label="Fitness Value")
    plt.title("Scatter Plot Of Highest Fitness Values")
    plt.xlabel("Generations")
    plt.ylabel("Fitness Values")
    plt.show()


def show_3d_scatter_plot(ga, df):
    x_ax = list(df.iloc[:, 0])
    y_ax = list(df.iloc[:, 1])
    orginal_y = list(df.iloc[:, 5])
    yhat, error = calculate_yhat_error(ga, df)

    d3_plot = plt.figure()
    d3 = d3_plot.add_subplot(111, projection='3d')
    d3.scatter(x_ax, y_ax, yhat, c='r', marker='o')
    d3.scatter(x_ax, y_ax, orginal_y, c='b', marker='^')
    d3.set_xlabel('Weight')
    d3.set_ylabel('Height')
    d3.set_zlabel('Y & Yhat')
    plt.show()


def fetch_and_modify_data():
    df = pd.read_csv('ga_dataset.csv')
    df = df.drop(df.columns[[5, 6, 7, 8, 9, 10, 11, 12, 14, 15]], axis=1)

    # Normalizing and splitting input and output
    df = pd.DataFrame(skp.normalize(df, norm='l2'))
    training_df = df.sample(frac=0.75)
    test_df = df.drop(training_df.index)

    return test_df, training_df


class GeneticAlgorithm(object):
    """
    Class for performing the Genetic Algorithm operations
    """
    def __init__(self, *args, **kwargs):
        self.features_X = np.matrix(kwargs['training_df'].iloc[:, [0, 1, 2, 3, 4]])
        self.target_Y = np.matrix(kwargs['training_df'].iloc[:, [5]])
        self.wghts_matrix = (5, 10)
        self.population = []
        self.y_hat = []
        self.fitness_values = []
        self.parent_chromosome = 0
        self.chromosomes = []
        self.highest_fitness_val = None
        self.prev_highest_fitness_val = None
        self.highest_fit_index = None
        self.max_weight = None
        self.cross_chromosomes = []
        self.fitness_values_list = []
        self.max_generation = 5000

    def initialisation(self):
        self.population = np.random.uniform(low=-1, high=1, size=(500, 50))
        self._normalization()

    def fitness_check(self):
        self.y_hat = self._calculate_body_fat()
        self.fitness_values = self._calculate_fitness()

    def selection(self):
        self.highest_fit_index, self.highest_fitness_val = self.fitness_values.index(max(self.fitness_values)), \
                                               max(self.fitness_values)
        if self.prev_highest_fitness_val:
            self.fitness_values_list.append(self.highest_fitness_val)
            if self.prev_highest_fitness_val <= self.highest_fitness_val:
                self.parent_chromosome = self.chromosomes[self.highest_fit_index]
        else:
            self.parent_chromosome = self.chromosomes[self.highest_fit_index]

    def binaralize_population(self):
        self.chromosomes = []
        decimal_pop = np.rint(np.array(self.population))
        for row in decimal_pop:
            self.chromosomes.append(''.join([bin(int(val*1000))[2:].zfill(10) for val in row]))

    def cross_over(self):
        self.cross_chromosomes = []

        for chromo in self.chromosomes:
            c_point1, c_point2 = np.random.random_integers(2, 500 - 1),  np.random.random_integers(2, 500 - 1)
            first_offspring = self.parent_chromosome[:c_point1] + chromo[c_point1:]
            second_offspring = self.parent_chromosome[:c_point2:] + chromo[c_point2:]
            self.cross_chromosomes.append(first_offspring)
            self.cross_chromosomes.append(second_offspring)

    def mutation(self):
        mutated_list = []
        for chromo in self.cross_chromosomes:
            mutated_list.append(self._mutate_chromosome(chromo))
        self.cross_chromosomes = mutated_list

    def de_binarization(self):
        bin_data = []
        d_data = []
        for chromo in self.cross_chromosomes:
            bin_data.append(textwrap.wrap(chromo, 10))
        for chromo in bin_data:
            d_data.append([(int(x, 2))/1000 for x in chromo])
        self.population = self._de_normalisation(d_data)

    def remove_lowest_fits(self):
        sorted_fittest_value = self.fitness_values.copy()
        sorted_fittest_value.sort()
        index_of_lowest_value = []
        pop = []
        self.chromosomes = []
        for val in sorted_fittest_value[:500]:
            index_of_lowest_value.append(self.fitness_values.index(val))
        for index in index_of_lowest_value:
            pop.append(self.population[index])
            self.chromosomes.append(self.cross_chromosomes[index])

        self.population = pop

        if not self.prev_highest_fitness_val:
            self.prev_highest_fitness_val = self.highest_fitness_val
            self.prev_parent_chromosome = self.parent_chromosome

        if self.prev_highest_fitness_val < self.highest_fitness_val:
            self.prev_parent_chromosome = self.parent_chromosome
            self.prev_highest_fitness_val = self.highest_fitness_val

    def set_max_weight(self):
        self.max_weight = self.population[self.highest_fit_index]

    def _calculate_body_fat(self):
        pr_body_fat = []
        for weights in self.population:
            rL = []
            weights_ar = np.array(weights)
            weights_ar = weights_ar.reshape(self.wghts_matrix)
            for row in self.features_X:
                mul_matrix = row * weights_ar
                rL.append(sum(map(self._sigmoid, np.squeeze(np.array(np.sum(mul_matrix, axis=0))))))
            pr_body_fat.append(rL)
        return pr_body_fat

    def _sigmoid(self, val):
        return 1 / (1 + math.exp(-val))

    def _calculate_fitness(self):
        fitness_values = []
        for y_hat in self.y_hat:
            y_yhat_c = (1 - sum([(a - b) ** 2 for a, b in zip(self.target_Y, y_hat)]) / len(self.features_X)) * 100
            fitness_values.append(y_yhat_c.item(0, 0))

        return fitness_values

    def _mutate_chromosome(self, chromo):
        chromo_list = np.array(list(chromo)).astype('int')
        random_index = random.sample(range(0, len(chromo)), 21)

        for index in random_index:
            chromo_list[index] = 0 if chromo_list[index] else 1

        return ''.join(chromo_list.astype('str'))

    def _normalization(self):
        for i in range(0, len(self.population)):
            self.population[i] = self._normalise_2d_mat(self.population[i].reshape(self.wghts_matrix)).flatten()

    def _normalise_2d_mat(self, matrix_2d):
        for i in range(0, self.wghts_matrix[0]):
            for j in range(0, self.wghts_matrix[1]):
                matrix_2d[i][j] = ((matrix_2d[i][j] + 1) / 2)
        return matrix_2d

    def _de_normalisation(self, data):
        for i in range(0, len(data)):
            data = np.array(data)
            data[i] = self._de_normalise_2d_mat(data[i].reshape(self.wghts_matrix)).flatten()
        return data

    def _de_normalise_2d_mat(self, matrix_2d):
        for i in range(0, self.wghts_matrix[0]):
            for j in range(0, self.wghts_matrix[1]):
                matrix_2d[i][j] = (matrix_2d[i][j] * 2) - 1
        return matrix_2d

    @property
    def pr_weight(self):
        return self.max_weight

    @property
    def fitness_list(self):
        return self.fitness_values_list


if __name__ == "__main__":
    start_process()



