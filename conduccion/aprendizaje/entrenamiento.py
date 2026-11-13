"""Entrenamiento directo: prepara o recupera una corrida y guarda cada generación.

El bucle de generaciones está en genetico.py (algoritmo_evolutivo). Este archivo se encarga de lo
que rodea al bucle: armar la población inicial o cargar una guardada, elegir las rutas de referencia
y guardar el estado después de cada generación.
"""

import numpy as np

from .. import config
from ..mapa.ruta import ruta_al_azar
from .genetico import algoritmo_evolutivo, poblacion_inicial
from .guardado import (cargar, cargar_poblacion, donde_se_guarda, guardar, nuevo_entrenamiento,
                       tiene_guardado, valores_de_config)


# -------------------------------------------------------------------------
# Las rutas de referencia de un mapa: RUTAS_REFERENCIA rutas fijas para comparar
# redes de distintas generaciones.
#
# Usan su propia semilla, distinta de la del entrenamiento, así son siempre las
# mismas para un mapa y una semilla, sin importar cuántas generaciones se corrieron.
# -------------------------------------------------------------------------
def rutas_de_referencia(mapa, semilla):
    rng = np.random.default_rng(semilla + 1_000_003)
    return [ruta_al_azar(mapa, rng) for _ in range(config.RUTAS_REFERENCIA)]


# -------------------------------------------------------------------------
# Recupera una corrida guardada para seguir entrenándola.
#
# Para que la continuación tenga sentido, tienen que ser iguales el tamaño de la
# población, el cruce, la mutación y todos los parámetros de config.py. Si cambió
# el mapa, la población se conserva pero la historia empieza de nuevo.
#
# Devuelve: la población, la mejor red, la historia, el generador al azar en el
#           estado en que quedó, y la semilla.
# -------------------------------------------------------------------------
def continuar_entrenamiento(identificador, mapa, tamano_poblacion, tipo_cruce, tipo_mutacion):
    mejor, historia, datos = cargar(identificador)
    poblacion, estado = cargar_poblacion(identificador)

    # 1. Lo que se elige al entrenar tiene que coincidir
    if len(poblacion) != tamano_poblacion or datos["parametros"]["tipo_cruce"] != tipo_cruce:
        raise ValueError("Para continuar conservá el tamaño de población y el cruce.")
    if datos["parametros"]["tipo_mutacion"] != tipo_mutacion:
        raise ValueError("Para continuar conservá el tipo de mutación.")

    # 2. Y los parámetros de config.py también
    comprobar_config(datos["config"])

    # 3. En un mapa nuevo la historia empieza de cero
    if datos["parametros"]["mapa"] != mapa.nombre:
        historia = []

    # 4. El generador al azar sigue exactamente donde quedó
    semilla = datos["parametros"]["semilla"]
    rng = np.random.default_rng(semilla)
    rng.bit_generator.state = estado["rng"]
    return poblacion, mejor, historia, rng, semilla


# -------------------------------------------------------------------------
# Comprueba que config.py tenga los mismos valores que cuando se guardó el
# entrenamiento, salvo los que se eligen al entrenar. Si alguno cambió, da error.
#
# Recibe: los valores de config.py guardados con el entrenamiento.
# -------------------------------------------------------------------------
def comprobar_config(guardada):
    elegidos_al_entrenar = {"GENERACIONES", "TAMANO_POBLACION", "TIPO_CRUCE", "TIPO_MUTACION"}
    for nombre, valor in valores_de_config().items():
        if nombre in elegidos_al_entrenar or nombre not in guardada:
            continue
        if guardada[nombre] != valor:
            raise ValueError(f"Cambió {nombre} respecto del entrenamiento guardado; no se puede continuar.")


# -------------------------------------------------------------------------
# Entrena en un solo mapa.
#
# Recibe: el mapa, la semilla, la cantidad de generaciones (None = hasta interrumpir),
#         el tamaño de la población, los tipos de cruce y de mutación, y opcionalmente:
#           mostrar            la función que muestra el avance (por defecto, print)
#           simular_poblacion  para evaluar en la ventana de pygame
#           continuar_desde    el identificador de una corrida guardada para seguirla
# Devuelve: la mejor red y la historia.
#
# Continuar una corrida no la modifica: crea un guardado nuevo. Si se interrumpe con
# Ctrl+C, se conserva la última generación completa.
# -------------------------------------------------------------------------
def entrenar(mapa, semilla=0, generaciones=config.GENERACIONES, tamano_poblacion=config.TAMANO_POBLACION,
             tipo_cruce=config.TIPO_CRUCE, tipo_mutacion=config.TIPO_MUTACION, mostrar=print,
             simular_poblacion=None, continuar_desde=None):
    if generaciones is not None and generaciones < 1:
        raise ValueError("La cantidad de generaciones debe ser positiva o None.")

    # 1. Población nueva, o la de una corrida guardada
    if continuar_desde is None:
        rng = np.random.default_rng(semilla)
        poblacion = poblacion_inicial(tamano_poblacion, config.CAPAS, rng)
        mejor, historia = None, []
    else:
        poblacion, mejor, historia, rng, semilla = continuar_entrenamiento(
            continuar_desde, mapa, tamano_poblacion, tipo_cruce, tipo_mutacion)

    # 2. La carpeta donde se guarda, y lo que se anota de la corrida
    identificador = nuevo_entrenamiento(semilla)
    parametros = {"semilla": semilla, "mapa": mapa.nombre, "generaciones": generaciones,
                  "tamano_poblacion": tamano_poblacion, "tipo_cruce": tipo_cruce,
                  "tipo_mutacion": tipo_mutacion, "origen": continuar_desde}
    mostrar(f"Guardando {identificador} en {donde_se_guarda()}")

    # Se llama al final de cada generación
    def guardar_generacion(poblacion, mejor, historia, rng):
        estado = {"generaciones_completadas": len(historia), "rng": rng.bit_generator.state}
        guardar(identificador, mejor, historia, parametros, poblacion, estado)

    # 3. El bucle de generaciones
    try:
        mejor, _, historia = algoritmo_evolutivo(
            poblacion, mapa, rutas_de_referencia(mapa, semilla), rng, max_generaciones=generaciones,
            tipo_cruce=tipo_cruce, tipo_mutacion=tipo_mutacion, semilla=semilla, historia=historia,
            mejor=mejor, simular_poblacion=simular_poblacion, al_guardar=guardar_generacion, mostrar=mostrar)

    # 4. Si se interrumpe, se devuelve lo último que se guardó
    except KeyboardInterrupt:
        mostrar("Entrenamiento detenido. Se conserva la última generación completada.")
        if tiene_guardado(identificador):
            mejor, historia, _ = cargar(identificador)
        else:
            mejor, historia = poblacion[0].copy(), []
    return mejor, historia
