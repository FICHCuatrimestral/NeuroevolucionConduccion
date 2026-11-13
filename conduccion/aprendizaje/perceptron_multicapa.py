"""Perceptrón multicapa con tangente hiperbólica.

No se entrena con retropropagación: los pesos los busca el algoritmo genético.
Por eso la red solo sabe propagar una entrada y pasar sus pesos a un cromosoma y de vuelta.
"""

import numpy as np

from .. import config


# -------------------------------------------------------------------------
# Perceptron: una red de capas totalmente conectadas, con tanh en todas las neuronas.
#
#   capas     cuántas neuronas tiene cada capa, por ejemplo [16, 16, 2]
#   pesos     una matriz por capa, con una fila por entrada y una columna por neurona
#   umbrales  un vector por capa, con un umbral por neurona
#
# Recibe: el tamaño de cada capa y, de forma opcional, un cromosoma.
# Con cromosoma, la red usa esos pesos. Sin cromosoma, usa pesos chicos al azar
# con desvío DESVIO_PESOS_INICIALES (Montana y Davis: las buenas soluciones suelen
# tener pesos chicos).
# -------------------------------------------------------------------------
class Perceptron:

    def __init__(self, capas, rng=None, cromosoma=None):
        self.capas = list(capas)

        # Una matriz de pesos y un vector de umbrales por cada par de capas consecutivas
        pares = list(zip(self.capas[:-1], self.capas[1:]))
        self.pesos = [np.empty((entradas, salidas)) for entradas, salidas in pares]
        self.umbrales = [np.empty(salidas) for _, salidas in pares]

        if cromosoma is not None:
            self.cargar_cromosoma(cromosoma)
            return

        if rng is None:
            rng = np.random.default_rng()
        for pesos, umbrales in zip(self.pesos, self.umbrales):
            pesos[:] = rng.normal(0, config.DESVIO_PESOS_INICIALES, pesos.shape)
            umbrales[:] = rng.normal(0, config.DESVIO_PESOS_INICIALES, umbrales.shape)

    # ---------------------------------------------------------------------
    # Pasa una entrada por la red, capa por capa. En cada neurona j:
    #
    #     v_j = suma_i (w_ij · y_i) + umbral_j
    #     y_j = tanh(v_j)
    #
    # np.dot hace la suma de todas las neuronas de la capa a la vez.
    #
    # Devuelve: las salidas de la última capa, entre -1 y 1.
    # ---------------------------------------------------------------------
    def propagar(self, entrada):
        y = entrada
        for pesos, umbrales in zip(self.pesos, self.umbrales):
            y = np.tanh(np.dot(y, pesos) + umbrales)
        return y

    # ---------------------------------------------------------------------
    # Todos los pesos y umbrales en fila: el cromosoma.
    # Capa por capa: primero los pesos de la capa y después sus umbrales.
    # ---------------------------------------------------------------------
    def a_cromosoma(self):
        partes = []
        for pesos, umbrales in zip(self.pesos, self.umbrales):
            partes.extend((pesos.ravel(), umbrales))
        return np.concatenate(partes)

    # ---------------------------------------------------------------------
    # Lo contrario de a_cromosoma: reparte el cromosoma en las matrices de pesos y los umbrales.
    # ---------------------------------------------------------------------
    def cargar_cromosoma(self, cromosoma):
        inicio = 0
        for capa, pesos in enumerate(self.pesos):
            filas, columnas = pesos.shape

            # Los pesos de la capa
            fin = inicio + filas * columnas
            self.pesos[capa] = cromosoma[inicio:fin].reshape(filas, columnas).copy()
            inicio = fin

            # Los umbrales de la capa: uno por neurona
            fin = inicio + columnas
            self.umbrales[capa] = cromosoma[inicio:fin].copy()
            inicio = fin

    # ---------------------------------------------------------------------
    # Todas las neuronas con pesos de entrada (las ocultas y las de salida), como pares
    # (capa, número). Las usan el cruce y la mutación por neurona.
    # ---------------------------------------------------------------------
    def neuronas(self):
        return [(capa, j) for capa in range(len(self.pesos)) for j in range(self.capas[capa + 1])]


# -------------------------------------------------------------------------
# Largo del cromosoma. Con [16, 16, 2]: 16·16 + 16 + 16·2 + 2 = 306.
# -------------------------------------------------------------------------
def cantidad_de_pesos(capas):
    return sum(entradas * salidas + salidas for entradas, salidas in zip(capas[:-1], capas[1:]))
