#!/usr/bin/env python
# coding: utf-8

# In[1]:

import datetime
import numpy as np
import random
from jmetal.core.problem import Problem
from jmetal.core.solution import IntegerSolution
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.lab.visualization.chord_plot import chord_diagram
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations
from typing import List

#Prueba de crear distintas horas
"""m_inicio, m_fin = restricciones_hora(datetime.time(10, 30), datetime.time(14, 50))
restricciones_bajo = list([m_inicio, 0, 0, 0, 0])
restricciones_alto = list([m_fin, 100, 100, 100, 100])
print(restricciones_bajo)
print(restricciones_alto)
"""

"""
Definición de la clase problema HVAC
"""
class HVAC(Problem[IntegerSolution]):
   
    def __init__(self, lower_bound: List[int], upper_bound: List[int],number_of_configurations: int=3):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(HVAC, self).__init__()
        self.number_of_variables = number_of_configurations*5
        
        self.number_of_objectives = 6
        self.number_of_constraints = 0
        self.number_of_configurations = number_of_configurations
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MAXIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Confort", "Consumo", "Coste", "Rendimiento", "Vt", "Tshow - Tend"]
        self.lower_bound = []
        self.upper_bound = []
        
        for i in range(0, self.number_of_variables):
            self.lower_bound.append(lower_bound[i%len(lower_bound)])
            self.upper_bound.append(upper_bound[i%len(upper_bound)])
            
    def ordenar_solucion(self, solution: IntegerSolution) -> IntegerSolution:
        solution.variables = [int(i) for i in solution.variables]
        for i in range(0, int(solution.number_of_variables/5)):
            for j in range(i, int(solution.number_of_variables/5)):
                if (solution.variables[j * 5] < solution.variables[i * 5]):
                    aux = solution.variables[j * 5: j * 5 + 5]
                    solution.variables[j * 5: j * 5 + 5] = solution.variables[i * 5: i * 5 + 5]
                    solution.variables[i * 5: i * 5 + 5] = aux[:]
        return solution
     
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        """falta evaluar los objetivos"""
        solution = self.ordenar_solucion(solution)
        #Ejemplos de evaluación
        solution.objectives[0] = sum(solution.variables)
        solution.objectives[1] = solution.variables[1] + solution.variables[2]
        solution.objectives[2] = solution.variables[1] / (solution.variables[3]**2 + 1)
        solution.objectives[3] = sum(solution.variables[:4])**2
        solution.objectives[4] = (sum(solution.variables[2:5])+1)**2
        solution.objectives[5] = sum(solution.variables[4:])**2
        return solution

    def get_name(self):
        return 'HVAC Problem'
    
    

    def create_solution(self) -> IntegerSolution:
        
            
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        
        new_solution.number_of_variables = self.number_of_variables
        new_solution.attributes = {"fitness": [0], "variables": [],"weights": np.zeros(self.number_of_objectives)}
        new_solution.variables = ([int(random.uniform(self.lower_bound[j], self.upper_bound[j])) for j in range(0, new_solution.number_of_variables)])
        return new_solution
    
 



