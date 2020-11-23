"""
Módulo de visualización.
No se trata de ningún módulo funcional, ya que es encargado de mostrar Frentes de Pareto.
"""

import logging
import numpy as np
import pandas as pd
from jmetal.lab.visualization.chord_plot import chord_diagram
from jmetal.lab.visualization.plotting import Plot
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from plotly import io as pio
from plotly import offline
from typing import TypeVar, List

S = TypeVar('S')

class MostrarFrente():
    
    def __init__(self, frente: List[S] = None,
                         axis_labels: list = None):
        self.frente = frente
        self.labels = axis_labels

    def mostrar_chord_diagram(self):
        chord_diagram(self.frente, nbins = "auto", obj_labels = self.labels)
        
    def mostrar_frente_2d(self, labels: List[str], filename: str = None, format: str = 'eps', title: str = "Pareto front 2d"):

        if len(labels) == 0 or len(labels) > 2:
            print("Introduzca 2 soluciones a mostrar")
            return
        indices = []
        for j in range(2):
            for i in range(len(self.labels)):
                if labels[j] in self.labels[i] or labels[j] in self.labels[i]:
                    indices.append(i)


        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        for i in range(len(self.frente)):
            ax.scatter(self.frente[i].objectives[indices[0]], self.frente[i].objectives[indices[1]])

        ax.autoscale_view(True, True)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)
        plt.grid(True)
        plt.show()
        plt.close(fig)

    def mostrar_frente_3d(self, labels: List[str], filename: str = None, format: str = 'eps', title: str = "Pareto front 3d"):

        if len(labels) != 3:
            print("Introduzca 3 soluciones a mostrar")
            return
        indices = []
        for j in range(3):
            for i in range(len(self.labels)):
                if labels[j] in self.labels[i] or labels[j] in self.labels[i]:
                    indices.append(i)

        if len(indices) != 3:
            print("labels incorrectos.")
            return
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        for i in range(0, len(self.frente)):
            ax.scatter(self.frente[i].objectives[indices[0]],
                       self.frente[i].objectives[indices[1]],
                       self.frente[i].objectives[indices[2]])
        ax.relim()
        ax.autoscale_view(True, True)
        ax.view_init(elev=30.0, azim=15.0)
        ax.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)

        plt.show()
        #plt.close(fig)

    def mostrar_frentes_2d(self, filename: str = None, format: str = 'eps'):
        
        
        fig = plt.figure(figsize=(20,13))
        #plt.subplots_adjust(hspace=0.35)


        labels = self.labels
        #Confort - consumo
        ax = fig.add_subplot(231)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        
        for i in range(0, len(self.frente)):
            ax.scatter(self.frente[i].objectives[0], self.frente[i].objectives[1])
            
        ax.autoscale_view(True, True)
        
        #Confort - rendimiento
        ay = fig.add_subplot(232)
        ay.set_xlabel(labels[0])
        ay.set_ylabel(labels[2])
        for i in range(0, len(self.frente)):
            ay.scatter(self.frente[i].objectives[0], self.frente[i].objectives[2])
        
        ay.autoscale_view(True, True)
        
        #Vt  -  Tshow - Tend
        az = fig.add_subplot(233)
        az.set_xlabel(labels[3])
        az.set_ylabel(labels[4])

        for i in range(0, len(self.frente)):
            az.scatter(self.frente[i].objectives[3], self.frente[i].objectives[4])
        
        az.autoscale_view(True, True)

            
       #Consumo - Rendimiento
        aa = fig.add_subplot(234)
        aa.set_xlabel(labels[1])
        aa.set_ylabel(labels[2])

        for i in range(0, len(self.frente)):
            aa.scatter(self.frente[i].objectives[1], self.frente[i].objectives[2])
        
        aa.autoscale_view(True, True)
    
        #Consumo - Coste
        ab = fig.add_subplot(235)
        ab.set_xlabel(labels[1])
        ab.set_ylabel(labels[2])

        for i in range(0, len(self.frente)):
            ab.scatter(self.frente[i].objectives[1], self.frente[i].objectives[2])
        
        ab.autoscale_view(True, True)
        
        plt.show()
        plt.close(fig)


