import datetime
import datetime
import numpy as np
import random
import random
import sys
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.core.problem import Problem
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.lab.visualization.chord_plot import chord_diagram
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.operator.mutation import UniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations
from matplotlib import pyplot as plt
from typing import Generic, TypeVar, List

from modulos.moga import MOGA
from modulos.operadores.crossover import SBXCrossoverD
from modulos.operadores.mutation import IntegerPolynomialMutationD, NonUniformMutation
from modulos.problem import HVAC


class HVAcOptimizer():
    def __init__(self,):
        
        self.soluciones: List[S] = []
        self.algoritmo : Algorithm = None
        self.objetivos =  {"Label" : ["Confort", "Consumo", "Coste", "Rendimiento", "Vt", "Tshow - Tend"],
                         "Value": np.arange(0, 7)}

        self.problem: Problem = None

    
    def configurar_algoritmo(self, hora_show : datetime, hora_minima: datetime = datetime.time(0, 0), 
                           algoritmo: str = 'MOGA', mutation_probability: float = 0.25,
                          max_evaluations: int = 500, population: int = 100):
        hora_minima, hora_show  = self.restricciones_hora(hora_minima, hora_show)
        if(hora_minima == 0):
            hora_minima = hora_show - 180
        restricciones_baja = list([hora_minima, -100, -100, 0, 0])
        restricciones_alta = list([hora_show, 0, 0, 100, 100])
        
        self.problem = HVAC(lower_bound = restricciones_baja, upper_bound = restricciones_alta, number_of_configurations = 3)
        
        print("algoritmo: ", algoritmo)
        if algoritmo == 'MOGA':
            algorithm = MOGA(
                        problem=self.problem,
                        population_size = population,
                        offspring_population_size = population,
                        mutation = IntegerPolynomialMutationD(probability = mutation_probability, distribution_index=20),
                        crossover = SBXCrossoverD(probability=mutation_probability, distribution_index = 20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations),
                        dominance_comparator = DominanceComparator())
                
        elif algoritmo == "NSGAII":
            algorithm = NSGAII(
                        problem=self.problem,
                        population_size = population,
                        offspring_population_size = population,
                        mutation = IntegerPolynomialMutationD(probability = mutation_probability, distribution_index=20),
                        crossover = SBXCrossoverD(probability = mutation_probability, distribution_index = 20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations),
                        dominance_comparator = DominanceComparator())
            
        elif algoritmo == 'OMOPSO':
            algorithm = OMOPSO(
                        problem=self.problem,
                        swarm_size=population,
                        epsilon=0.0075,
                        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
                        non_uniform_mutation=NonUniformMutation(probability = mutation_probability, perturbation=0.5),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations))
        
        elif algoritmo == 'SMPSO':
            algorithm = SMPSO(
                        problem=self.problem,
                        swarm_size=population,
                        mutation=IntegerPolynomialMutation(probability = mutation_probability, distribution_index=20),
                        leaders=CrowdingDistanceArchive(100),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations))

        elif algoritmo == 'SPEA2':
            algorithm = SPEA2(
                        problem=self.problem,
                        population_size=population,
                        offspring_population_size=population,
                        mutation=IntegerPolynomialMutationD(probability=mutation_probability, distribution_index=20),
                        crossover=SBXCrossoverD(probability=mutation_probability, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations),
                        dominance_comparator=DominanceComparator())
            
        else:
            print("Algoritmo no válido. Creando MOGA por defecto...")
            algorithm = MOGA(
                        problem=self.problem,
                        population_size = population,
                        offspring_population_size = population,
                        mutation = IntegerPolynomialMutationD(probability = mutation_probability, distribution_index=20),
                        crossover = SBXCrossoverD(probability=mutation_probability, distribution_index = 20),
                        termination_criterion=StoppingByEvaluations(max=max_evaluations),
                        dominance_comparator = DominanceComparator())
        self.algoritmo = algorithm
        self.algoritmo.observable.register(observer=ProgressBarObserver(max=max_evaluations))

        return algorithm
    
    def ejecutar_algoritmo(self):

        self.algoritmo.run()
        self.soluciones = self.algoritmo.get_result()

        print('Algorithm (continuous problem): ' + self.algoritmo.get_name())
        print('Problem: ' + self.problem.get_name())
        print('Computing time: ' + str(self.algoritmo.total_computing_time))

    def restricciones_hora(self, hora_inicio, hora_fin):
        minutos_inicio = int(
            datetime.timedelta(hours=hora_inicio.hour, minutes=hora_inicio.minute).total_seconds() / 60)
        minutos_fin = int(datetime.timedelta(hours=hora_fin.hour, minutes=hora_fin.minute).total_seconds() / 60) - 1
        return minutos_inicio, minutos_fin

    """tener en cuenta si lo paso como minutos o como formato hora"""

    def comparar_hora(self, hora_inicio, hora_fin, minutos):
        hora_configuracion = hora_inicio + minutos
        return hora_configuracion < hora_fin

    def mostrar_chord_diagram(self):
        chord_diagram(self.soluciones, nbins = "auto", obj_labels = self.objetivos["Label"])

    def mostrar_frente_2d(self, labels: List[str], filename: str = None, format: str = 'eps'):

        if len(labels) == 0 or len(labels) > 2:
            print("Introduzca 2 soluciones a mostrar")
            return
        indices = []
        for j in range(2):
            for i in range(len(self.objetivos["Label"])):
                if labels[j] in self.objetivos["Label"][i] or labels[j] in self.objetivos["Label"][i]:
                    indices.append(i)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        for i in range(len(self.soluciones)):
            ax.scatter(self.soluciones[i].objectives[indices[0]], self.soluciones[i].objectives[indices[1]])
        ax.relim()
        ax.autoscale_view(True, True)

        ax.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)

        plt.show()
        #plt.close(fig)
    
    def mostrar_frente_3d(self, labels: List[str], filename: str = None, format: str = 'eps'):

        if len(labels) != 3:
            print("Introduzca 3 soluciones a mostrar")
            return
        indices = []
        for j in range(3):
            for i in range(len(self.objetivos["Label"])):
                if labels[j] in self.objetivos["Label"][i] or labels[j] in self.objetivos["Label"][i]:
                    indices.append(i)

        if len(indices) != 3:
            print("labels incorrectos.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        for i in range(0, len(self.soluciones)):
            ax.scatter(self.soluciones[i].objectives[indices[0]],
                       self.soluciones[i].objectives[indices[1]],
                       self.soluciones[i].objectives[indices[2]])
        ax.relim()
        ax.autoscale_view(True, True)
        ax.view_init(elev=30.0, azim=15.0)
        ax.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)

        plt.show()
        #plt.close(fig)
        
    def mostrar_frentes_2d(self, labels: List[str], filename: str = None, format: str = 'eps'):

        if len(labels) == 0 or len(labels) > 2:
            print("Introduzca 2 soluciones a mostrar")
            return
        indices = []
        for i in range(len(labels)):
            if(labels == self.objetivos["Label"]):
                indices.append(i)
                soluciones_objetivo.append(self.soluciones)
        front = [x.objectives[indices] for x in self.soluciones]
        
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

        for i in range(0, len(front)):
            ax.scatter(front[i][0], front[i][1])
        ax.relim()
        ax.autoscale_view(True, True)

        ax.locator_params(nbins=4)

        ay = fig.add_subplot(212)
        ay.set_xlabel(labels[0])
        ay.set_ylabel(labels[1])

        for i in range(0, len(front)):
            ay.scatter(front[i][0], front[i][2])
        ay.relim()
        ay.autoscale_view(True, True)

        ay.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)

        plt.show()
        plt.close(fig)
        
    


