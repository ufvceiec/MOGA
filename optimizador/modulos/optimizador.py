"""
Módulo optimizador. 

Autor: Raúl Jiménez Juárez

Descripción: la clase HVAcOptimizer contiene la definición del problema y el algoritmo que lo resolverá.
Se encargará de configurar el problema para después ejecutarlo y obtener las soluciones del problema.


"""


import datetime
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pyowm
import sys
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import Problem
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from typing import TypeVar, List



from modulos.banco_algoritmos import BancoAlgoritmos
from modulos.datos_externos import DatosExternos
from modulos.problem import HVAC
from modulos.visualizacion import MostrarFrente
from pickle import dump, load

class HVAcOptimizer():
    
    """Constructor"""
    def __init__(self,):
        
        self.soluciones = None
        self.algoritmo : Algorithm = None
        self.objetivos =  {"Label" : ["Confort", "Consumo",  "Rendimiento", "Vt", "Tshow - Tend"],
                         "Value": np.arange(0, 6)}

        self.problem: Problem = None
        self.visualizador = None
        self.datos_externos = None

    def configurar_optimizador_simulacion(self,
                             datos_externos: DatosExternos = None,
                             nombre_algoritmo: str = 'MOGA',
                             probabilidad_mutacion: float = None,
                             max_evaluaciones: int = 500, 
                             poblacion_maxima: int = 100):

        minimos_hvac = list([-100, -100, -100, -100])
        maximos_hvac = list([0, 0, 100, 100])

        
        if probabilidad_mutacion == None:
            probabilidad_mutacion = float(poblacion_maxima / max_evaluaciones)
        if probabilidad_mutacion > 0.5:
            probabilidad_mutacion = 0.5
        
        self.datos_externos = datos_externos

        self.problem = HVAC(datos_externos = self.datos_externos,
                            minimos_hvac = minimos_hvac,
                            maximos_hvac = maximos_hvac)
        
        self.algoritmo = self.configurar_algoritmo(nombre_algoritmo, poblacion_maxima, probabilidad_mutacion, max_evaluaciones)

        #Desde aqui añadimos los nuevos observers
        self.algoritmo.observable.register(observer=ProgressBarObserver(max=max_evaluaciones))

        return self.algoritmo
        


    def configurar_optimizador(self,
                             fecha_inicio: datetime = None,
                             fecha_evento : datetime = None,
                             temperatura_inicial: int = None,
                             temperatura_objetivo:int = None,
                             nombre_algoritmo: str = 'MOGA',
                             probabilidad_mutacion: float = None,
                             max_evaluaciones: int = 500, 
                             poblacion_maxima: int = 100):

        if fecha_evento == None:
            print("Introduzca el paarámetro de fecha show. Error al crear el problema")
            return
        if fecha_inicio == None:
            fecha_inicio = fecha_evento - datetime.timedelta(minutes=180)


        self.datos_externos = DatosExternos(fecha_inicio, fecha_evento, temperatura_inicial,temperatura_objetivo)
        self.datos_externos.calcular_datos_externos()

        minimos_hvac = list([-100, -100, -100, -100])
        maximos_hvac = list([0, 0, 100, 100])

        if probabilidad_mutacion == None:
            probabilidad_mutacion = float(poblacion_maxima / max_evaluaciones)
        if probabilidad_mutacion > 0.5:
            probabilidad_mutacion = 0.5
        

        self.problem = HVAC(datos_externos = self.datos_externos,
                            minimos_hvac = minimos_hvac,
                            maximos_hvac = maximos_hvac)
        
        self.algoritmo = self.configurar_algoritmo(nombre_algoritmo, poblacion_maxima, probabilidad_mutacion, max_evaluaciones)

        #Desde aqui añadimos los nuevos observers
        self.algoritmo.observable.register(observer=ProgressBarObserver(max=max_evaluaciones))
        #self.algoritmo.observable.register(observer=PrintObjectivesObserver(frequency=50))
        
        print("\nOptimzador creado: ")
        print("\talgoritmo: ", nombre_algoritmo)
        print("\tpoblación: ", poblacion_maxima)
        print("\tprobabilidad de mutación: ", probabilidad_mutacion)
        print("\tmáximas evaluaciones: ", max_evaluaciones)
        print("\nDatos del problema: ")
        print("\thora mínima: ", fecha_inicio)
        print("\thora show: ", fecha_evento)
        print("\tconfiguraciones máximas: ", self.datos_externos.intervalos)
        return self.algoritmo

    

    def ejecutar_algoritmo(self):
        self.algoritmo.run()
        self.soluciones = self.algoritmo.get_result()
        self.visualizador = MostrarFrente(self.soluciones, axis_labels = self.objetivos["Label"])
        #self.visualizador.mostrar_frentes_2d(configuraciones = self.obtener_configuraciones())

        print('Algorithm: ' + self.algoritmo.get_name())
        print('Problem: ' + self.problem.get_name())
        print('Computing time: ' + str(self.algoritmo.total_computing_time))


    def configurar_algoritmo(self, algoritmo, population, mutation_probability,  max_evaluations):

        banco_algoritmos = BancoAlgoritmos(self.problem, mutation_probability, population, max_evaluations)
        if algoritmo == 'MOGA':
            algorithm = banco_algoritmos.configurar_MOGA()
            
        elif algoritmo == "NSGAII":
            algorithm = banco_algoritmos.configurar_NSGAII()

        elif algoritmo == 'OMOPSO':
            algorithm = banco_algoritmos.configurar_OMOPSO()
        
        elif algoritmo == 'RandomSearch':
            algorithm = banco_algoritmos.configurar_RandomSearch()

        elif algoritmo == 'NSGAIII':
            algorithm = banco_algoritmos.configurar_NSGAIII()

        elif algoritmo == 'SMPSO':
            algorithm = banco_algoritmos.configurar_SMPSO()

        elif algoritmo == 'SPEA2':
           algorithm = banco_algoritmos.configurar_SPEA2()
        else:
            print("Algoritmo no válido. Creando NSGAII por defecto...")
            algorithm = banco_algoritmos.configurar_NSGAII()
        return algorithm
        
    
    def obtener_soluciones(self):
       
        num_solucion = 1
        lista_configuraciones = []
        resultados = pd.DataFrame()

        for solucion in self.soluciones:

            intervalos = self.problem.obtener_intervalos(solucion)
            horas_configuracion = []
            numero_configuraciones = len(intervalos)
            hvacs = []

            for i in intervalos:        
                horas_configuracion.append(self.datos_externos.horas_intervalos[i])
                hvacs.append(solucion.variables[i*5 +1:i*5 + 5])             
            hvacs = np.array(hvacs)
            confort = solucion.attributes["temperatura"] - self.datos_externos.temperatura_objetivo 

            horas_configuracion = np.array(horas_configuracion)

            
            df_configuraciones = pd.DataFrame({"Hora": horas_configuracion[:],
                                        "Climatizador 1":  hvacs[:,0],
                                        "Climatizador 2" : hvacs[:,1],
                                        "Climatizador Carlos" : hvacs[:,2],
                                        "Climatizador Felipe" : hvacs[:,3]})
            
            resultados = resultados.append(pd.DataFrame({"Consumo" : [solucion.objectives[1]],
                                        "COP" : [abs(solucion.objectives[2])],
                                        "Confort" : [confort],
                                        "Temperatura final" : [solucion.attributes["temperatura"]],
                                        "Vt" : [solucion.attributes["vt"]],
                                        "Tshow - Tend" : [solucion.objectives[4]],
                                        "Programa" : [df_configuraciones]}), ignore_index = True)
            num_solucion += 1

        return resultados




    

    
        
    