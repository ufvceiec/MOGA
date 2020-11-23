import dask
import numpy as np
import random
import time
from distributed import as_completed, Client
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import DynamicAlgorithm, Algorithm
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.core.operator import Selection
from jmetal.core.problem import Problem, DynamicProblem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.solutions.comparator import DominanceComparator, Comparator, SolutionAttributeComparator, \
    MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion
from typing import List, TypeVar
from typing import TypeVar, List

S = TypeVar('S')
R = TypeVar('R')

"""
.. module:: NSGA-II
   :platform: Unix, Windows
   :synopsis: NSGA-II (Non-dominance Sorting Genetic Algorithm II) implementation.
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class MOGA(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(MultiComparator([SolutionAttributeComparator(key = "fitness")])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """
        MOGA implementation as described in
        * K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A fast and elitist
          multiobjective genetic algorithm: NSGA-II," in IEEE Transactions on Evolutionary Computation,
          vol. 6, no. 2, pp. 182-197, Apr 2002. doi: 10.1109/4235.996017
        NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The implementation of NSGA-II provided in jMetalPy follows the evolutionary
        algorithm template described in the algorithm module (:py:mod:`jmetal.core.algorithm`).
        .. note:: A steady-state version of this algorithm can be run by setting the offspring size to 1.
        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super(MOGA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        """ This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.
        :param population: Parent population.
        :param offspring_population: Offspring population.
        :return: New population after ranking and crowding distance selection is applied.
        """
        ranking = FastNonDominatedRanking(self.dominance_comparator)
        density_estimator = CrowdingDistance()
        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions
    
    def selection(self, population: List[S]):
        mating_population = []
        
        population = self.calcular_pesos_objetivos(population)
        population = self.calcular_fitness_escalar_lineal(population)
        
        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def calcular_pesos_objetivos(self, population: List[S]) -> List[List[S]]:
        for i in range(len(population)):
            for j in range(len(population[i].objectives)):
                population[i].attributes["weights"][j] = random.random()/(sum([random.random() for k in range(len(population[i].objectives))]))
        return population

    def calcular_fitness_escalar_lineal(self, population: List[S]) -> List[S]:
        #obtengo un array con el fitness de cada individuo
        for i in range(len(population)):
            suma = 0
            for j in range(len(population[i].attributes["weights"])):
                suma += population[i].objectives[j]*population[i].attributes["weights"][j]
            population[i].attributes["fitness"] = suma
        return population

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'MOGA'
