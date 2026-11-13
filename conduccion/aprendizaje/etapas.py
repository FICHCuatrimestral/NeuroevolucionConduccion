"""Entrenamiento por etapas: de lo simple a lo complejo (evolución incremental, Gomez y Miikkulainen 1996).

Las etapas de etapas_santa_fe salen de un diccionario de ajustes, en su orden. Las posibles son:
  1. manzana: vueltas a la manzana del medio del 3x3, una doblando siempre a la izquierda y otra a la derecha
  2. la grilla de 2 × 2 manzanas
  3. la grilla de 3 × 3 manzanas
  4. el centro de Santa Fe, con curvas, diagonales y ramales (el informe no lo usa como etapa)

En cada etapa, entrenar_por_etapas hace esto:
  1. La población sigue de la etapa anterior: lo aprendido no se pierde
  2. σ queda fijo en el valor sigma de la etapa. Cada etapa usa uno igual o más chico que la anterior,
     para afinar lo que la red ya sabe en vez de desarmarlo
  3. Cada generación maneja las rutas de la etapa y una ruta de una etapa anterior: así no se olvida
  4. Pasa a la etapa siguiente cuando la mejor red llega a B en una fracción `umbral` de las rutas de
     referencia de la etapa, o cuando se acaban sus generaciones. La última etapa usa todas las suyas

La mejor red es la que llega en más rutas de referencia. Si empatan, la de mayor nota, y después la más rápida.
(En el entrenamiento directo, en cambio, se compara primero la nota.)

Cada etapa se guarda como un entrenamiento común, en su propia carpeta de resultados/.
Así se puede abrir la mejor red de cualquier etapa.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .. import config
from ..mapa.mapa import manzana, santa_fe
from ..mapa.ruta import ruta_al_azar, vueltas_a_la_manzana
from .entrenamiento import rutas_de_referencia
from .fitness import contar_llegadas, evaluar, medir_en_referencia
from .genetico import nueva_generacion, poblacion_inicial
from .guardado import cargar, cargar_poblacion, guardar, nuevo_entrenamiento, tiene_guardado


# -------------------------------------------------------------------------
# Etapa: un mapa y cómo se entrena en él.
#
#   nombre               "manzana", "2x2", "3x3", "centro", ...
#   mapa                 el Mapa
#   elegir_rutas         función que recibe el generador al azar y devuelve las rutas de una generación
#   referencia           rutas fijas: eligen la mejor red y deciden cuándo pasar de etapa
#   sigma                σ de la mutación en esta etapa
#   umbral               fracción de las rutas de referencia en la que la mejor red tiene que llegar
#   minimo_generaciones  no pasa de etapa antes de esta cantidad de generaciones
#   maximo_generaciones  pasa de etapa después de esta cantidad, llegue o no al umbral
# -------------------------------------------------------------------------
@dataclass
class Etapa:
    nombre: str
    mapa: object
    elegir_rutas: Callable
    referencia: list
    sigma: float
    umbral: float
    minimo_generaciones: int
    maximo_generaciones: int

    # ---------------------------------------------------------------------
    # Mide un cromosoma en las rutas de referencia de la etapa.
    # Devuelve: la nota, el tiempo total y la cantidad de llegadas.
    # ---------------------------------------------------------------------
    def medir(self, cromosoma, semilla):
        casos = [(self.mapa, ruta) for ruta in self.referencia]
        return medir_en_referencia(cromosoma, casos, semilla)


# -------------------------------------------------------------------------
# Estado: lo que cambia mientras se entrena una etapa.
#
#   poblacion        la población actual
#   mejor            la mejor red encontrada
#   historia         una fila por generación
#   nota_mejor       nota de la mejor red en las rutas de referencia de la etapa
#   tiempo_mejor     tiempo total de la mejor red en esas rutas
#   llegadas_mejor   en cuántas de esas rutas llega la mejor red
#
# Al crearlo, la mejor red se mide en las rutas de referencia de la etapa.
# -------------------------------------------------------------------------
class Estado:

    def __init__(self, poblacion, mejor, historia, etapa, semilla):
        self.poblacion = poblacion
        self.mejor = mejor
        self.historia = list(historia)
        self.etapa = etapa
        self.semilla = semilla
        self.nota_mejor, self.tiempo_mejor, self.llegadas_mejor = etapa.medir(mejor, semilla)

    # ---------------------------------------------------------------------
    # Compara un candidato con la mejor red, en las rutas de referencia. Si gana, lo guarda.
    #
    # Primero cuentan las llegadas: una red con nota alta que se queda sin tiempo cerca
    # de B no reemplaza a una que llega. Si llegan en las mismas rutas, gana la de mayor
    # nota, y si también empatan, la más rápida. Python compara las tuplas en ese orden;
    # el tiempo va con signo menos porque menos tiempo es mejor.
    # ---------------------------------------------------------------------
    def probar_candidato(self, candidato):
        nota, tiempo, llegadas = self.etapa.medir(candidato, self.semilla)
        if (llegadas, nota, -tiempo) > (self.llegadas_mejor, self.nota_mejor, -self.tiempo_mejor):
            self.mejor = candidato.copy()
            self.nota_mejor, self.tiempo_mejor, self.llegadas_mejor = nota, tiempo, llegadas

    # ---------------------------------------------------------------------
    # ¿La población pasa a la etapa siguiente?
    #
    # Pasa si la mejor red llega en al menos una fracción `umbral` de las rutas de
    # referencia y ya se hicieron minimo_generaciones. La última etapa nunca pasa:
    # usa todas sus generaciones.
    # ---------------------------------------------------------------------
    def pasa_de_etapa(self, ultima):
        supera_umbral = self.llegadas_mejor >= self.etapa.umbral * len(self.etapa.referencia)
        return not ultima and supera_umbral and len(self.historia) >= self.etapa.minimo_generaciones

    # ---------------------------------------------------------------------
    # Agrega a la historia la fila de una generación.
    #
    # Recibe: el número de generación, las notas y los resultados de la población,
    #         y cuál es el de mayor nota.
    # ---------------------------------------------------------------------
    def registrar_generacion(self, generacion, notas, resultados, indice_mejor):
        self.historia.append({"generacion": generacion,
                              "etapa": self.etapa.nombre,
                              "mejor_nota": float(notas[indice_mejor]),
                              "nota_media": float(np.mean(notas)),
                              "llegadas_del_mejor": contar_llegadas(resultados[indice_mejor]),
                              "sigma": self.etapa.sigma,
                              "mejor_referencia": self.nota_mejor,
                              "llegadas_referencia": self.llegadas_mejor})


# Los ajustes de cada etapa para entrenar desde el notebook, hasta el centro. Los del
# informe (100, 50 y 50 generaciones, sin el centro) están en ETAPAS, en comparaciones.py.
AJUSTES_ETAPAS = {
    "manzana": {"sigma": 0.3, "umbral": 1.0, "minimo_generaciones": 10, "maximo_generaciones": 150},
    "2x2": {"sigma": 0.2, "umbral": 0.9, "minimo_generaciones": 10, "maximo_generaciones": 250},
    "3x3": {"sigma": 0.15, "umbral": 0.9, "minimo_generaciones": 10, "maximo_generaciones": 300},
    "centro": {"sigma": 0.15, "umbral": 1.0, "minimo_generaciones": 10, "maximo_generaciones": 600},
}


# -------------------------------------------------------------------------
# Arma las etapas, en el orden del diccionario de ajustes.
#
# "manzana" da las vueltas a la manzana; cualquier otro nombre es una zona de santa_fe().
# Sacar un nombre del diccionario saltea esa etapa.
# -------------------------------------------------------------------------
def etapas_santa_fe(semilla=0, ajustes=AJUSTES_ETAPAS):
    etapas = []
    for nombre, valores in ajustes.items():
        if nombre == "manzana":
            # En la manzana no hay rutas al azar: se usan las 8 vueltas posibles
            mapa = manzana()
            izquierda, derecha = vueltas_a_la_manzana(mapa)
            etapa = Etapa(nombre, mapa, vueltas_de(izquierda, derecha), izquierda + derecha, **valores)
        else:
            mapa = santa_fe(nombre)
            etapa = Etapa(nombre, mapa, rutas_al_azar_de(mapa), rutas_de_referencia(mapa, semilla), **valores)
        etapas.append(etapa)
    return etapas


# -------------------------------------------------------------------------
# Cómo elige sus rutas la etapa de la manzana: en cada generación, una vuelta
# hacia cada lado, cada una desde una esquina al azar.
# -------------------------------------------------------------------------
def vueltas_de(izquierda, derecha):
    # Recibe el generador al azar y devuelve las dos vueltas de la generación
    def elegir(rng):
        return [izquierda[rng.integers(len(izquierda))], derecha[rng.integers(len(derecha))]]
    return elegir


# -------------------------------------------------------------------------
# Cómo eligen sus rutas las demás etapas: RUTAS_POR_GENERACION rutas al azar del mapa.
# -------------------------------------------------------------------------
def rutas_al_azar_de(mapa):
    # Recibe el generador al azar y devuelve las rutas de la generación
    def elegir(rng):
        return [ruta_al_azar(mapa, rng) for _ in range(config.RUTAS_POR_GENERACION)]
    return elegir


# -------------------------------------------------------------------------
# Los casos de una generación: las rutas de la etapa actual y, desde la segunda
# etapa, una ruta de una etapa anterior elegida al azar, para no olvidarla.
#
# Devuelve: una lista de pares (mapa, ruta).
# -------------------------------------------------------------------------
def casos_de_la_generacion(etapas, numero_etapa, rng):
    etapa = etapas[numero_etapa]
    casos = [(etapa.mapa, ruta) for ruta in etapa.elegir_rutas(rng)]
    if numero_etapa > 0:
        anterior = etapas[rng.integers(numero_etapa)]
        rutas = anterior.elegir_rutas(rng)
        casos.append((anterior.mapa, rutas[rng.integers(len(rutas))]))
    return casos


# =========================================================================
# ENTRENAR
# =========================================================================

# -------------------------------------------------------------------------
# Entrena la población etapa por etapa (los pasos 1 a 4 de arriba del archivo).
#
# Recibe: las etapas, la semilla, el tamaño de la población, los tipos de cruce y
#         de mutación, y opcionalmente:
#           mostrar          la función que muestra el avance (por defecto, print)
#           continuar_desde  identificador de una etapa guardada, para seguir desde ahí
#           desde_etapa      número de la etapa con que se sigue (0 = la primera).
#                            None = la misma etapa donde se cortó
# Devuelve: la mejor red y los identificadores de las etapas guardadas, uno por etapa.
#
# Continuar en la misma etapa donde se cortó da el mismo resultado que si no se hubiera
# cortado: se recuperan la población, la mejor red, la historia de la etapa y el estado
# del azar, y se sigue desde la generación siguiente. Continuar en otra etapa sirve para
# repetir solo las últimas etapas: esa etapa empieza de cero con la población guardada.
# -------------------------------------------------------------------------
def entrenar_por_etapas(etapas, semilla=0, tamano_poblacion=config.TAMANO_POBLACION, tipo_cruce=config.TIPO_CRUCE,
                        tipo_mutacion=config.TIPO_MUTACION, mostrar=print, continuar_desde=None, desde_etapa=None):
    rng = np.random.default_rng(semilla)

    # Una población nueva, o la de una etapa guardada
    if continuar_desde is None:
        poblacion = poblacion_inicial(tamano_poblacion, config.CAPAS, rng)
        mejor = poblacion[0].copy()
        historia_guardada = []
        if desde_etapa is None:
            desde_etapa = 0
    else:
        poblacion, mejor, historia_guardada, desde_etapa = retomar(continuar_desde, etapas, semilla, desde_etapa, rng)

    identificadores = []
    origen = str(continuar_desde) if continuar_desde else None

    try:
        for numero_etapa in range(desde_etapa, len(etapas)):
            etapa = etapas[numero_etapa]
            ultima = numero_etapa == len(etapas) - 1

            # 1. La población sigue de la etapa anterior. La mejor red se vuelve a medir en las rutas nuevas.
            #    Si se retoma una etapa cortada, se conserva su historia.
            historia = historia_guardada if numero_etapa == desde_etapa else []
            estado = Estado(poblacion, mejor, historia, etapa, semilla)

            identificador = nuevo_entrenamiento(semilla)
            identificadores.append(identificador)
            parametros = parametros_de_etapa(etapas, numero_etapa, semilla, len(poblacion), tipo_cruce,
                                             tipo_mutacion, origen)
            mostrar(f"Etapa {numero_etapa + 1} de {len(etapas)}: {etapa.nombre}. Guardando en {identificador}")

            # Guarda la etapa tal como está; se llama al final de cada generación
            def guardar_etapa():
                guardar(identificador, estado.mejor, estado.historia, parametros, estado.poblacion,
                        {"generaciones_completadas": len(estado.historia), "rng": rng.bit_generator.state})

            # Una etapa retomada pudo haber alcanzado el umbral justo en la última generación guardada
            if estado.historia and estado.pasa_de_etapa(ultima):
                primera_generacion = etapa.maximo_generaciones
                mostrar_paso_de_etapa(estado, mostrar)
            else:
                primera_generacion = len(estado.historia)

            for generacion in range(primera_generacion, etapa.maximo_generaciones):
                # 2 y 3. Rutas de la etapa y una de una etapa anterior; se evalúa la población
                casos = casos_de_la_generacion(etapas, numero_etapa, rng)
                notas, resultados = evaluar(estado.poblacion, casos, rng)

                # El de mayor nota es el candidato a mejor red
                indice_mejor = int(np.argmax(notas))
                estado.probar_candidato(estado.poblacion[indice_mejor])

                # Registrar la generación
                estado.registrar_generacion(generacion, notas, resultados, indice_mejor)
                mostrar_generacion(estado, len(casos), mostrar)

                # La generación siguiente, y guardar
                estado.poblacion = nueva_generacion(estado.poblacion, notas, config.CAPAS, etapa.sigma, rng,
                                                    tipo_cruce, tipo_mutacion, mejor=estado.mejor)
                guardar_etapa()

                # 4. ¿Pasa de etapa?
                if estado.pasa_de_etapa(ultima):
                    mostrar_paso_de_etapa(estado, mostrar)
                    break

            # Una etapa retomada que ya había terminado no corre generaciones nuevas:
            # se guarda igual, para que su carpeta exista
            if primera_generacion >= etapa.maximo_generaciones:
                guardar_etapa()

            # La etapa siguiente arranca con lo que dejó esta
            poblacion, mejor, origen = estado.poblacion, estado.mejor, str(identificador)
            mostrar_olvido(mejor, etapas[:numero_etapa + 1], semilla, mostrar)

    except KeyboardInterrupt:
        mostrar("Entrenamiento detenido. Cada etapa conserva su última generación completa.")
        guardados = [identificador for identificador in identificadores if tiene_guardado(identificador)]
        if guardados:
            mostrar(f'Para seguir desde donde quedó: continuar_desde="{guardados[-1]}"')

    return mejor, identificadores


# -------------------------------------------------------------------------
# Lo que hace falta para continuar desde una etapa guardada. Deja el azar
# exactamente donde quedó.
#
# Recibe: el identificador guardado, las etapas, la semilla, la etapa con que se
#         sigue (None = la misma donde se cortó) y el generador al azar.
# Devuelve: la población, la mejor red, la historia guardada y el número de la
#           etapa con que se sigue. Si se sigue en otra etapa, la historia está vacía.
# -------------------------------------------------------------------------
def retomar(continuar_desde, etapas, semilla, desde_etapa, rng):
    mejor, historia, datos = cargar(continuar_desde)
    poblacion, guardado = cargar_poblacion(continuar_desde)
    if datos["parametros"]["semilla"] != semilla:
        raise ValueError(f"Para continuar usá la misma semilla: {datos['parametros']['semilla']}.")
    rng.bit_generator.state = guardado["rng"]

    # Sin desde_etapa, se sigue en la etapa donde se cortó
    nombres = [etapa.nombre for etapa in etapas]
    etapa_guardada = datos["parametros"]["nombre_etapa"]
    if desde_etapa is None:
        if etapa_guardada not in nombres:
            raise ValueError(f"La etapa guardada, {etapa_guardada}, no está entre las etapas elegidas.")
        desde_etapa = nombres.index(etapa_guardada)

    # En otra etapa, la historia empieza de cero
    if nombres[desde_etapa] != etapa_guardada:
        historia = []
    return poblacion, mejor, historia, desde_etapa


# -------------------------------------------------------------------------
# Los parámetros de una etapa que se guardan junto con ella.
# origen es el identificador de la etapa de donde vino la población, o None.
# -------------------------------------------------------------------------
def parametros_de_etapa(etapas, numero_etapa, semilla, tamano_poblacion, tipo_cruce, tipo_mutacion, origen):
    etapa = etapas[numero_etapa]
    return {"semilla": semilla, "mapa": etapa.mapa.nombre, "generaciones": etapa.maximo_generaciones,
            "tamano_poblacion": tamano_poblacion, "tipo_cruce": tipo_cruce,
            "tipo_mutacion": tipo_mutacion, "origen": origen,
            "etapa": numero_etapa + 1, "nombre_etapa": etapa.nombre,
            "etapas": [otra.nombre for otra in etapas], "sigma_etapa": etapa.sigma,
            "umbral_etapa": etapa.umbral}


# -------------------------------------------------------------------------
# Muestra en una línea cómo le fue a la última generación de la etapa.
# -------------------------------------------------------------------------
def mostrar_generacion(estado, cantidad_casos, mostrar):
    fila = estado.historia[-1]
    mostrar(f"{fila['etapa']}  generación {fila['generacion']}"
            f"  mejor {fila['mejor_nota']:.3f}  media {fila['nota_media']:.3f}"
            f"  llegadas del mejor {fila['llegadas_del_mejor']}/{cantidad_casos}"
            f"  mejor referencia {fila['mejor_referencia']:.3f}"
            f"  llega en {fila['llegadas_referencia']}/{len(estado.etapa.referencia)}  sigma {fila['sigma']}")


# -------------------------------------------------------------------------
# Avisa que la población pasa a la etapa siguiente.
# -------------------------------------------------------------------------
def mostrar_paso_de_etapa(estado, mostrar):
    mostrar(f"La mejor red llega en {estado.llegadas_mejor} de {len(estado.etapa.referencia)} "
            "rutas de referencia: pasa a la etapa siguiente.")


# -------------------------------------------------------------------------
# Al terminar una etapa, muestra en cuántas rutas de referencia de cada etapa ya
# hecha llega la mejor red. Sirve para ver si olvidó lo que aprendió antes.
# -------------------------------------------------------------------------
def mostrar_olvido(mejor, etapas, semilla, mostrar):
    partes = []
    for etapa in etapas:
        _, _, llegadas = etapa.medir(mejor, semilla)
        partes.append(f"{etapa.nombre} {llegadas}/{len(etapa.referencia)}")
    mostrar("Llegadas de la mejor red en las referencias: " + ", ".join(partes))
