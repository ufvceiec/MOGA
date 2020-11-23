"""
Módulo de datos externos.
En este módulo se gestiona la información de contexto del problema necesaria para su resolución.
"""
import datetime
import pyowm
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class DatosExternos():

    def __init__(self, fecha_inicio, fecha_evento, temperatura_inicial, temperatura_objetivo):
        self.temperaturas_intervalos = None
        self.horas_intervalos = None
        self.aforos = None
        self.fecha_inicio = fecha_inicio
        self.fecha_evento = fecha_evento
        self.temperatura_inicial = temperatura_inicial
        self.temperatura_objetivo = temperatura_objetivo
        self.intervalos = None

    
    def calcular_datos_externos(self):

        self.calcular_numero_intervalos()
        self.calcular_aforo_intervalos()
        self.calcular_temperaturas_externas()
        self.calcular_horas_intervalos()

        
    def calcular_numero_intervalos(self):
        minutos_inicio, minutos_fin = self.calcular_restricciones_hora()
        self.intervalos = int ((minutos_fin - minutos_inicio) / 15)


    def calcular_aforo_intervalos(self):

        var_aforo = 1700
        aumento_aforo = (var_aforo/self.intervalos) ** (1/3)
        aforos = list()
        aforo_actual = 0
        for i in range(self.intervalos):
            aforos.append(aforo_actual)
            aforo_actual = round(i*(aumento_aforo**3))

        self.aforos = aforos


    def calcular_horas_intervalos(self):

        hora_intervalo = self.fecha_inicio
        horas_intervalos = list()

        for i in range(self.intervalos):
            horas_intervalos.append(hora_intervalo)
            hora_intervalo = hora_intervalo + datetime.timedelta(minutes=15)
        self.horas_intervalos = horas_intervalos


    def calcular_temperaturas_externas(self):
        owm = pyowm.OWM('221815e6266b3c5ba9407990ae21eafa')
        fc = owm.three_hours_forecast_at_coords(40.41824, -3.71037)
        f = fc.get_forecast()
        horas = list()
        temperaturas = list()
        contador = 0

        for weather in f:
            hora = weather.get_reference_time('date').replace(tzinfo=None)
            if ((hora >= self.fecha_inicio and hora <= self.fecha_evento) or (contador == 1 and hora > self.fecha_evento)):
                horas.append(hora)
                temperaturas.append(weather.get_temperature('celsius')["temp"])
                contador += 1

        var_temperatura = temperaturas[-1] - temperaturas[0]
        incremento_temperatura = var_temperatura / self.intervalos
        temperatura_intervalo = temperaturas[0]
        temperaturas_intervalos = list()

        for i in range(self.intervalos):
            temperaturas_intervalos.append(temperatura_intervalo)
            temperatura_intervalo += incremento_temperatura

        self.temperaturas_intervalos = temperaturas_intervalos

    def calcular_restricciones_hora(self):

        minutos_fin = int(
            datetime.timedelta(hours=self.fecha_evento.hour, minutes=self.fecha_evento.minute).total_seconds() / 60)
        if self.fecha_inicio == None:
            minutos_inicio = minutos_fin - 180
        else:
            minutos_inicio = int(
                datetime.timedelta(hours=self.fecha_inicio.hour, minutes=self.fecha_inicio.minute).total_seconds() / 60)
        return minutos_inicio, minutos_fin

