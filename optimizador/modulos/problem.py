#!/usr/bin/env python
# coding: utf-8

"""

Módulo problema. Contiene la definición del problema de optimización.
"""

# In[1]:

from typing import List

import keras
import numpy as np
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
import pickle
from modulos.datos_externos import DatosExternos


class HVAC(Problem[FloatSolution]):
   
    def __init__(self, datos_externos: DatosExternos, minimos_hvac: List[int], maximos_hvac: List[int], min_configurations: int=1):
        """ :param number_of_variables: Number of decision variables of the problem.
        """
        super(HVAC, self).__init__()

        self.number_of_variables = 5*(datos_externos.intervalos)
        self.number_of_objectives = 5
        self.number_of_constraints = 0
        self.datos_externos = datos_externos

        self.modelo = keras.models.load_model('modelos/modelo_hvac.h5')
        #Si queremos cargar al modelo de random forest
        #fich = open('modelos/forest_model.sav', 'rb')
        #self.modelo = pickle.load(fich)

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Confort", "Consumo", "Rendimiento", "Vt", "Tshow - Tend"]

        self.lower_bound = []
        self.upper_bound = []

        for i in range(0, datos_externos.intervalos):
            self.lower_bound.append(0)
            self.upper_bound.append(1)
            for j in range(4):
                self.lower_bound.append(float(minimos_hvac[j]))
                self.upper_bound.append(float(maximos_hvac[j]))
            

    #Transforma las variables del individuo para que las interprete el modelo
    #Los datos están normalizados, por lo que es necesaria la transformación
    def obtener_intervalos(self, solucion: FloatSolution):
        #solution.variables = [int(i) for i in solution.variables]
        intervalos = []
        for i in range(0, self.datos_externos.intervalos):
            
            hora = 1 if (solucion.variables[i*5] >= 0.5) else 0
            if hora == 1:
                hora = i
                intervalos.append(hora)
        i -= 3
        intervalos = []
        if len(intervalos) < 3:
            solucion.variables[i*5] == 0.5
            hora = i 
            intervalos.append(hora)
            i+=1
            solucion.variables[i*5] == 0.5
            hora = i 
            intervalos.append(hora)
            i += 1
            solucion.variables[i * 5] == 0.5
            hora = i
            intervalos.append(hora)
        return  intervalos
        


    def evaluate(self, solution: FloatSolution) -> FloatSolution:

        intervalos = self.obtener_intervalos(solution)
        numero_configuraciones = len(intervalos) - 1
       
        for i in range(self.number_of_objectives):
            solution.objectives[i] = 0

        configuracion = -1
        temperatura_interior = self.datos_externos.temperatura_inicial
        temperatura_inicial_configuracion = temperatura_interior

        for intervalo in range(intervalos[0], self.datos_externos.intervalos, 1):
            
            if (configuracion != numero_configuraciones):
                if (intervalos[configuracion + 1] == intervalo):
                    configuracion += 1
                    temperatura_inicial_configuracion = temperatura_interior
                    hora_intervalo = self.datos_externos.horas_intervalos[intervalo]
            
            entradas_modelo = np.array([solution.variables[configuracion*5 + 1], solution.variables[configuracion*5 + 2],
                                solution.variables[configuracion*5 + 3], solution.variables[configuracion* 5 + 4], temperatura_interior,
                                self.datos_externos.temperaturas_intervalos[intervalo], self.datos_externos.aforos[intervalo]]).reshape(1, 7)
            salidas_modelo = self.modelo.predict(entradas_modelo)
           
            # Evaluamos el objetivo Consumo
            solution.objectives[1] += salidas_modelo[0][0]
            # Evaluamos el objetivo rendimiento
            solution.objectives[2] += salidas_modelo[0][2]/(salidas_modelo[0][0]+1)
            
            # Evaluamos el objetivo vt
            solution.objectives[3] = abs(salidas_modelo[0][1])
            temperatura_interior = temperatura_interior  + salidas_modelo[0][1]
            intervalo += 1

        
        solution.attributes["temperatura"] = temperatura_interior

        # Evaluamos el objetivo Confort
        solution.objectives[0] = abs(temperatura_interior - self.datos_externos.temperatura_objetivo)

        #Evaluamos el rendimiento
        solution.objectives[2] = -1*(solution.objectives[2] / (self.datos_externos.intervalos - intervalos[0]))

        # Evaluamos el objetivo tshow - tend
        solution.objectives[4] = float(((self.datos_externos.fecha_evento - hora_intervalo).total_seconds() / 60))

        #solution.attributes["vt"] = (temperatura_interior - temperatura_inicial_configuracion) / solution.objectives[4]       
        return solution

    def get_name(self):
        return 'HVAC Problem'
    
    def create_solution(self) -> FloatSolution:

        new_solution = FloatSolution(
            lower_bound = self.lower_bound,
            upper_bound = self.upper_bound,
            number_of_objectives = self.number_of_objectives,
            number_of_constraints = self.number_of_constraints)

        new_solution.number_of_variables = self.number_of_variables
        new_solution.attributes = {"fitness": 0,
                                   "weights": np.zeros(self.number_of_objectives),
                                   "confort" : 0,
                                   "temperatura" : 0,
                                   "vt" : 0}
        new_solution.variables = [np.random.uniform(self.lower_bound[i], self.upper_bound[i]) for i in range(self.datos_externos.intervalos*5)]


        return new_solution
    
 



