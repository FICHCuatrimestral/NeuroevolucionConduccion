"""Mapa de calles.

El mapa parte de un grafo de osmnx: los nodos son las esquinas y las aristas son las cuadras.
Con ese grafo, la clase Mapa arma tres cosas:
  1. calles: la superficie por donde se puede andar, como un polígono de shapely
  2. intersecciones: las zonas de los cruces, donde no se mide el carril
  3. matriz: la misma superficie en celdas. Los sensores y los choques la consultan

Los mapas se abren con santa_fe: son calles reales, bajadas de OpenStreetMap por descargar_mapa.py
y descargar_ciudad.py. manzana da el mapa 3x3 con su manzana del medio, para la primera etapa.
"""

import math
from pathlib import Path

import networkx as nx
import numpy as np
import osmnx as ox
import shapely

from .. import config


# -------------------------------------------------------------------------
# Mapa: dónde hay calle y dónde no.
#
# Recibe: el grafo de calles (formato de osmnx, en metros) y un nombre.
# Guarda: el grafo, la superficie de calle, las zonas de cruce y la matriz.
# -------------------------------------------------------------------------
class Mapa:

    def __init__(self, grafo, nombre=""):
        self.grafo = grafo
        self.nombre = nombre

        # osmnx pasa el grafo a dos tablas: una con la forma de cada esquina y otra con la de cada cuadra
        esquinas, cuadras = ox.convert.graph_to_gdfs(grafo)

        # 1. Calles. Cada cuadra es una línea; se ensancha un carril a cada lado y se unen todas.
        self.calles = cuadras.buffer(config.ANCHO_CARRIL).union_all()

        # 2. Cruces. Ahí no hay línea divisoria y el auto puede pasar por el centro al doblar,
        #    así que no se mide si invade el carril. El radio cubre la esquina de dos calles de 7 m.
        self.radio_interseccion = config.ANCHO_CARRIL * math.sqrt(2)
        self.intersecciones = esquinas.buffer(self.radio_interseccion).union_all()

        # Se prepara la geometría porque se consulta muchas veces durante el entrenamiento
        shapely.prepare(self.intersecciones)

        # 3. La misma superficie, en celdas
        self.armar_matriz()

    # ---------------------------------------------------------------------
    # Posición (x, y) de una esquina del grafo, en metros.
    # ---------------------------------------------------------------------
    def posicion(self, nodo):
        datos = self.grafo.nodes[nodo]
        return datos["x"], datos["y"]

    # ---------------------------------------------------------------------
    # Pasa la superficie de calle a una matriz de celdas de RESOLUCION_MAPA de lado.
    # matriz[fila, columna] es True si el centro de esa celda está sobre la calle.
    #
    # Se deja un margen de ALCANCE_RAYOS alrededor, para que los rayos de los
    # sensores puedan salir del mapa sin quedar fuera de la matriz.
    # ---------------------------------------------------------------------
    def armar_matriz(self):
        x_min, y_min, x_max, y_max = self.calles.bounds

        # Esquina de abajo a la izquierda de la matriz
        self.x0 = x_min - config.ALCANCE_RAYOS
        self.y0 = y_min - config.ALCANCE_RAYOS

        columnas = math.ceil((x_max - x_min + 2 * config.ALCANCE_RAYOS) / config.RESOLUCION_MAPA)
        filas = math.ceil((y_max - y_min + 2 * config.ALCANCE_RAYOS) / config.RESOLUCION_MAPA)

        # Centro de cada celda
        xs = self.x0 + (np.arange(columnas) + 0.5) * config.RESOLUCION_MAPA
        ys = self.y0 + (np.arange(filas) + 0.5) * config.RESOLUCION_MAPA
        X, Y = np.meshgrid(xs, ys)

        # shapely responde, para cada centro, si está adentro de la calle
        self.matriz = shapely.contains_xy(self.calles, X, Y)

    # ---------------------------------------------------------------------
    # ¿Estos puntos están sobre la calle?
    #
    # Recibe: arreglos de coordenadas xs e ys, de cualquier forma.
    # Devuelve: un arreglo de la misma forma, con True donde hay calle.
    #
    # Se consulta la matriz y no shapely porque es mucho más rápido: con shapely,
    # una generación tarda entre 1,7 y 2,4 veces más.
    # ---------------------------------------------------------------------
    def es_calle(self, xs, ys):
        # 1. De metros a número de celda
        columnas = np.floor((np.asarray(xs) - self.x0) / config.RESOLUCION_MAPA).astype(int)
        filas = np.floor((np.asarray(ys) - self.y0) / config.RESOLUCION_MAPA).astype(int)

        # 2. Lo que cae fuera de la matriz no es calle
        filas_totales, columnas_totales = self.matriz.shape
        dentro = (columnas >= 0) & (columnas < columnas_totales) & (filas >= 0) & (filas < filas_totales)

        # 3. Lo que cae adentro se lee de la matriz
        resultado = np.zeros(columnas.shape, dtype=bool)
        resultado[dentro] = self.matriz[filas[dentro], columnas[dentro]]
        return resultado


# -------------------------------------------------------------------------
# Archivo donde está guardada una zona. Por ejemplo, "2x2" da mapas/santa_fe_2x2.graphml.
# -------------------------------------------------------------------------
def archivo_santa_fe(zona):
    return Path(__file__).parents[2] / "mapas" / f"santa_fe_{zona}.graphml"


# -------------------------------------------------------------------------
# Abre una zona de Santa Fe guardada en mapas/. No necesita internet.
#
# Zonas: "2x2", "3x3", "centro", "candioti", "guadalupe" y los recortes de
# recortes.json, como "n1_01_20_de_junio".
# -------------------------------------------------------------------------
def santa_fe(zona="2x2"):
    grafo = ox.load_graphml(archivo_santa_fe(zona))
    return Mapa(grafo, f"santa fe {zona}")


# -------------------------------------------------------------------------
# El mapa de la primera etapa: el 3x3 entero, para dar vueltas a su manzana del medio.
#
# Se usa la manzana del medio porque sus 4 esquinas son cruces completos. En una manzana
# suelta, o en el borde del mapa, la calle de afuera termina en la esquina y no hay lugar
# para doblar por el lado de afuera.
#
# Devuelve: el Mapa, con las 4 esquinas de esa manzana, en orden, en esquinas_manzana.
# -------------------------------------------------------------------------
def manzana():
    grafo = ox.load_graphml(archivo_santa_fe("3x3"))
    calles = nx.Graph(grafo.to_undirected())

    # 1. Cada manzana es un ciclo de 4 esquinas en el grafo.
    #    La del medio es la única con sus 4 esquinas de 4 calles cada una.
    ciclos = nx.minimum_cycle_basis(calles)
    medio = [ciclo for ciclo in ciclos
             if len(ciclo) == 4 and all(calles.degree(esquina) == 4 for esquina in ciclo)][0]

    # 2. Las esquinas en el orden en que se recorren
    mapa = Mapa(grafo, "manzana")
    mapa.esquinas_manzana = [esquina for esquina, _ in nx.find_cycle(calles.subgraph(medio))]
    return mapa
