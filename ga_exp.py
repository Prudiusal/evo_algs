from deap import tools, base
from multiprocessing import Pool
from ga_scheme import eaMuPlusLambda
# from deap.algorithms import eaMuPlusLambda
from numpy import random as rnd
import numpy as np
from deap import creator
from deap import benchmarks

# создает стратегию по максимизации первого значения фитнес функции
# (создан на основе base.Fitness)
creator.create("BaseFitness", base.Fitness, weights=(1.0,))
# класс индивидуал представляет собой расширение массива нампай
# в качестве атрибута хранится раннее определённая фитнес функция
#  (в которой есть веса, значения фитнеса и взвешенные значения фитнеса,
#  по которым будет сравниваться индивиды)
creator.create("Individual", np.ndarray, fitness=creator.BaseFitness)


def mutation(individual):
    n = len(individual)
    # для каждого индивида
    for i in range(n):
        # вероятность увеличивается с номером хромосомы???
        # чтоб реализовать просто мутацию, нужно убрать n
        if rnd.random() < n * 0.15:
            #
            # добавляет чуть-чуть к гена
            individual[i] += rnd.normal(0.0, 0.2)
            # подгоняет к стандартным границам
            # что - то такое во второй лабораторной работе

            individual[i] = np.clip(individual[i], -5, 5)
    return individual,


class SimpleGAExperiment:
    def factory(self):
        # создает хромосому
        # в качестве гена выступают float между 0 и 1.
        # изначально производится умножение на 0.00001
        #
        return rnd.random(self.dimension) * 10 - 4

    def __init__(self, function, dimension, pop_size, iterations):
        # size of population
        self.pop_size = pop_size
        # number of iterations
        self.iterations = iterations
        self.mut_prob = 0.6
        # self.cross_prob = 0
        self.cross_prob = 0.95
        # target function (for which we are searching the maximum)
        self.function = function
        # why dimension equal to 100?? ??
        self.dimension = dimension

        self.pool = Pool(8)
        self.engine = base.Toolbox()
        self.engine.register("map", self.pool.map)
        # self.engine.register("map", map)

        # создание одного индивида
        # на основе initRepeat (в init.py есть альтернативные)
        # каждый индивид имеет вид, как у класса creator.Individual, созданного ранее
        # для создания каждого индивида используется метод self.factory
        #
        #
        self.engine.register("individual", tools.initIterate, creator.Individual, self.factory)
        #
        # создание популяции. На основе initRepeat
        # все индивиды хранятся в списке list
        #
        self.engine.register("population", tools.initRepeat, list, self.engine.individual, self.pop_size)
        # одноточечный кроссовер
        # (другие в crossover.py)
        # self.engine.register("mate", tools.cxOnePoint)
        # self.engine.register("mate", tools.cxTwoPoint)
        # self.engine.register("mate", tools.cxUniform, indpb=0.5)
        self.engine.register("mate", tools.cxBlend, alpha=0.4)
        # eta - spreading factor
        # self.engine.register("mate", tools.cxSimulatedBinary, eta=5)

        #  мутация из тулбокса с параметрами гаус распр. + вероятность мутации
        # (другие в mutation.py)
        BOUND_LOW = -5.0
        BOUND_HIGH = 5.0
        # self.engine.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW,
        #                      up=BOUND_HIGH,
        #                      eta=30, indpb=0.6)

        self.engine.register("mutate", tools.mutGaussian, mu=0, sigma=0.7
                             , indpb=0.6)
        # альтернативная стратегия мутации на основе кастомной функции

        # self.engine.register("mutate", mutation)
        # алгоритм селекции (турир из 4)
        # (другие в selection.py)
        self.engine.register("select", tools.selTournament, tournsize=2)
        # self.engine.register("select", tools.selRoulette)
        # self.engine.register("select", tools.selStochasticUniversalSampling)
        #   функция, которая считает фитнес и записывает его в атрибут Индивида BaseFitness
        self.engine.register("evaluate", self.function)

    def run(self):
        pop = self.engine.population()
        #
        #
        #
        hof = tools.HallOfFame(10, np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = eaMuPlusLambda(pop, self.engine, mu=self.pop_size, lambda_=int(self.pop_size * 0.8
                                                                                  ),
                                  cxpb=self.cross_prob, mutpb=self.mut_prob,
                                  ngen=self.iterations,
                                  stats=stats, halloffame=hof, verbose=True)
        print("Best = {}".format(hof[0]))
        print("Best fit = {}".format(hof[0].fitness.values[0]))
        return log


from functions import rastrigin

if __name__ == "__main__":
    def function(x):
        res = rastrigin(x)
        return res,


    dimension = 100
    pop_size = 1000
    iterations = 400
    scenario = SimpleGAExperiment(function, dimension, pop_size, iterations)
    log = scenario.run()
    from draw_log import draw_log

    draw_log(log)
