"""Aptitud (fitness): simula la población y le pone una nota a cada individuo.

La nota de una ruta es el progreso de Togelius y Lucas, con un descuento por invadir el carril:

    F_r = (k + p) / K · (1 - PENALIZACION_CARRIL · fracción del tiempo fuera del carril)

  k = puntos de control pasados: los alcanzados y los que se salteó antes de alcanzar uno posterior
  p = progreso hacia el siguiente punto, de 0 a 1
  K = total de puntos de la ruta

Vale 1 si el auto llegó a B sin salir de su carril. PENALIZACION_CARRIL es menor que 1 para que un
auto que invade el carril conserve parte de su progreso: así la evolución tiene una señal útil desde
las primeras generaciones.

Con PREMIO_TIEMPO = lambda mayor que 0, también se premia llegar rápido:

    F_r = (1 - lambda) · F_r + lambda · L · (1 - t / t_max)

L vale 1 si llegó y 0 si no. Así, chocar temprano no da premio.

La nota del individuo es el promedio de F_r sobre las rutas que manejó.
"""

import numpy as np

from .. import config
from ..simulacion.simulacion import correr_episodio
from .perceptron_multicapa import Perceptron


# -------------------------------------------------------------------------
# Hace manejar a toda la población y le pone una nota a cada individuo.
#
# Recibe: la población (lista de cromosomas), los casos (pares mapa, ruta) y el
#         generador al azar para el ruido de los sensores. simular permite que los
#         episodios los corra la ventana de pygame en lugar de este bucle.
# Devuelve: las notas y los resultados de cada individuo, en el orden de la población.
#           None si se cerró la ventana de pygame.
# -------------------------------------------------------------------------
def evaluar(poblacion, casos, rng, simular=None):
    if simular is None:
        resultados_poblacion = []
        for cromosoma in poblacion:
            red = Perceptron(config.CAPAS, cromosoma=cromosoma)
            resultados = [correr_episodio(mapa, ruta, red, rng=rng) for mapa, ruta in casos]
            resultados_poblacion.append(resultados)
    else:
        resultados_poblacion = simular(poblacion, casos, rng)
        if resultados_poblacion is None:
            return None

    notas = [nota(resultados) for resultados in resultados_poblacion]
    return notas, resultados_poblacion


# -------------------------------------------------------------------------
# Nota de un individuo: el promedio de sus notas por ruta.
# -------------------------------------------------------------------------
def nota(resultados):
    return sum(nota_de_ruta(resultado) for resultado in resultados) / len(resultados)


# -------------------------------------------------------------------------
# Nota de una ruta, F_r, con las fórmulas de arriba del archivo.
# -------------------------------------------------------------------------
def nota_de_ruta(resultado):
    # 1. Progreso: puntos alcanzados más el avance hacia el siguiente, sobre el total
    progreso = (resultado.puntos_alcanzados + resultado.progreso) / resultado.total_puntos

    # 2. Descuento por el tiempo fuera del carril
    fraccion_fuera = 0.0
    if resultado.tiempo > 0:
        fraccion_fuera = min(1.0, resultado.tiempo_fuera_del_carril / resultado.tiempo)
    nota_base = progreso * (1 - config.PENALIZACION_CARRIL * fraccion_fuera)

    # 3. Premio por llegar rápido: solo si llegó
    premio = 0.0
    if resultado.motivo == "llegada":
        premio = 1 - resultado.tiempo / resultado.tiempo_maximo
    return (1 - config.PREMIO_TIEMPO) * nota_base + config.PREMIO_TIEMPO * premio


# -------------------------------------------------------------------------
# Cuántas de las rutas terminaron en B.
# -------------------------------------------------------------------------
def contar_llegadas(resultados):
    return sum(resultado.motivo == "llegada" for resultado in resultados)


# -------------------------------------------------------------------------
# Mide un cromosoma en las rutas de referencia.
#
# Usa siempre el mismo ruido (una semilla fija), así dos redes de distintas
# generaciones se comparan en igualdad de condiciones.
#
# Devuelve: la nota, el tiempo total y la cantidad de llegadas.
# -------------------------------------------------------------------------
def medir_en_referencia(cromosoma, casos, semilla):
    rng = np.random.default_rng(semilla + 2_000_003)
    notas, resultados = evaluar([cromosoma], casos, rng)
    return float(notas[0]), sum(r.tiempo for r in resultados[0]), contar_llegadas(resultados[0])
