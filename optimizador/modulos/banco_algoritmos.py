"""
Módulo de banco de algoritmos. Aquí s epueden configurar y añadir diferentes algoritmos de optimización
que podrán ser utilizados por el optimizador.

"""
import datetime
import numpy as np
import pandas as pd
import pyowm
import random
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.random_search import RandomSearch
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.core.problem import Problem
from jmetal.operator.crossover import SBXCrossover, SPXCrossover, CompositeCrossover
from jmetal.operator.mutation import UniformMutation, BitFlipMutation, PolynomialMutation, CompositeMutation, \
    NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from typing import TypeVar, List

from modulos.moga import MOGA
from modulos.problem import HVAC


class BancoAlgoritmos():

    def __init__(self, problema: Problem, probabilidad: float, maxima_poblacion: int, evaluaciones: int):
        self.algoritmo = None
        self.problema = problema
        self.probabilidad = probabilidad
        self.maxima_poblacion = maxima_poblacion
        self.evaluaciones = evaluaciones

    def configurar_OMOPSO(self):


        algorithm = OMOPSO(
                problem=self.problema,
                swarm_size=self.maxima_poblacion,
                epsilon=0.0075,
                uniform_mutation=UniformMutation(probability=self.probabilidad, perturbation=0.5),
                non_uniform_mutation=NonUniformMutation(probability=self.probabilidad, perturbation=0.5),
                leaders=CrowdingDistanceArchive(100),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones))
        return algorithm


    def configurar_SMPSO(self):

        algorithm = SMPSO(
                problem=self.problema,
                swarm_size=self.maxima_poblacion,
                mutation=PolynomialMutation(probability=self.probabilidad, distribution_index=20),
                leaders=CrowdingDistanceArchive(100),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones))
        return algorithm

    def configurar_SPEA2(self):

        algorithm = SPEA2(
                problem=self.problema,
                population_size=self.maxima_poblacion,
                offspring_population_size=self.maxima_poblacion,
                mutation=PolynomialMutation(probability=self.probabilidad, distribution_index=0.20),
                crossover=SBXCrossover(probability=self.probabilidad, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones),
                dominance_comparator=DominanceComparator())
        return algorithm

    def configurar_RandomSearch(self):
        algorithm = RandomSearch(
                                problem = self.problema,
                                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones))
        return algorithm

    def configurar_NSGAIII(self):
        
        algorithm = NSGAIII(
                problem=self.problema,
                reference_directions = UniformReferenceDirectionFactory(5, n_points=91),
                population_size=self.maxima_poblacion,
                mutation=PolynomialMutation(probability = self.probabilidad , distribution_index=0.20),
                crossover=SBXCrossover(probability= self.probabilidad, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones),
                dominance_comparator=DominanceComparator())
        return algorithm

    def configurar_MOGA(self):


        algorithm = MOGA(
                problem=self.problema,
                population_size=self.maxima_poblacion,
                offspring_population_size=self.maxima_poblacion,
                mutation=PolynomialMutation(probability = self.probabilidad),
                crossover=SBXCrossover(probability= self.probabilidad, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones),
                dominance_comparator=DominanceComparator())
        
        return algorithm

    def configurar_NSGAII(self):


        algorithm = NSGAII(
                problem=self.problema,
                population_size=self.maxima_poblacion,
                offspring_population_size=self.maxima_poblacion,
                mutation=PolynomialMutation(probability = self.probabilidad , distribution_index=0.20),
                crossover=SBXCrossover(probability= self.probabilidad, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=self.evaluaciones),
                dominance_comparator=DominanceComparator())
        return algorithm


