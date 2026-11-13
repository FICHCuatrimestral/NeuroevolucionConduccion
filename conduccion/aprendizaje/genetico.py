"""Algoritmo genético de pesos reales.

Un individuo es un cromosoma: el vector con los 306 pesos y umbrales de un perceptrón 16 → 16 → 2.

Para armar la generación siguiente, nueva_generacion hace esto:
  1. Elitismo: los ELITE mejores pasan sin cambios
  2. Selección: padre y madre, cada uno por torneo
  3. Cruce: el hijo mezcla a los dos padres. TIPO_CRUCE elige cómo:
       "neurona"  cada neurona copia de un padre todos sus pesos de entrada y su umbral (Montana y Davis)
       "pesos"    cruce uniforme: cada peso se copia del padre o de la madre
       "blx"      BLX-alfa: cada peso se sortea en un intervalo alrededor de los de los padres
       "ninguna"  sin cruce: el hijo es una copia del padre
  4. Mutación: al hijo se le suman pequeños cambios. TIPO_MUTACION elige cómo:
       "gaussiana" a cada peso, con probabilidad PROBABILIDAD_MUTACION, se le suma un valor normal
       "neurona"   a NEURONAS_A_MUTAR neuronas al azar se les cambian todos sus pesos de entrada y su umbral
  5. Si la mejor red encontrada no quedó en la generación nueva, ocupa el lugar del último hijo

σ, el desvío de la mutación, queda fijo durante toda la corrida: SIGMA en el entrenamiento
directo, o el de cada etapa en el entrenamiento por etapas.

algoritmo_evolutivo repite esos pasos generación tras generación: evalúa la población con fitness.py,
elige la mejor red en las rutas de referencia y arma la generación siguiente.
"""

import numpy as np

from .. import config
from ..mapa.ruta import ruta_al_azar
from . import fitness
from .perceptron_multicapa import Perceptron

TIPOS_CRUCE = ("neurona", "pesos", "blx", "ninguna")
TIPOS_MUTACION = ("gaussiana", "neurona")


# -------------------------------------------------------------------------
# La primera generación: tamano redes con pesos chicos al azar, pasadas a cromosoma.
# -------------------------------------------------------------------------
def poblacion_inicial(tamano, capas, rng):
    return [Perceptron(capas, rng).a_cromosoma() for _ in range(tamano)]


# -------------------------------------------------------------------------
# Elitismo: copia los ELITE individuos con mejor nota.
# -------------------------------------------------------------------------
def elegir_elite(poblacion, notas):
    de_mayor_a_menor = np.argsort(notas)[::-1]
    elite = []
    for k in de_mayor_a_menor[:config.ELITE]:
        elite.append(poblacion[k].copy())
    return elite


# -------------------------------------------------------------------------
# Arma la generación siguiente con los pasos 1 a 5 de arriba del archivo.
#
# Recibe: la población, sus notas, las capas de la red, el σ de la mutación,
#         el generador al azar, los tipos de cruce y de mutación, y la mejor red
#         encontrada hasta ahora (para no perderla).
# Devuelve: la población nueva, del mismo tamaño.
# -------------------------------------------------------------------------
def nueva_generacion(poblacion, notas, capas, sigma, rng, tipo_cruce=config.TIPO_CRUCE,
                     tipo_mutacion=config.TIPO_MUTACION, mejor=None):
    # 1. Elitismo: los mejores pasan sin cambios
    nueva = elegir_elite(poblacion, notas)

    # El resto de la población son hijos
    while len(nueva) < len(poblacion):
        # 2. Selección: un torneo para el padre y otro para la madre
        padre = seleccion(poblacion, notas, rng)
        madre = seleccion(poblacion, notas, rng)

        # 3. Cruce: el hijo mezcla a los dos padres
        hijo = cruzar(padre, madre, capas, tipo_cruce, rng)

        # 4. Mutación: el hijo cambia un poco y entra en la población
        hijo = mutar(hijo, capas, sigma, tipo_mutacion, rng)
        nueva.append(hijo)

    # 5. La mejor red encontrada no se pierde
    if mejor is not None and not any(np.array_equal(individuo, mejor) for individuo in nueva):
        nueva[-1] = mejor.copy()
    return nueva


# -------------------------------------------------------------------------
# Selección por torneo: se eligen TAMANO_TORNEO individuos al azar, sin repetir,
# y gana el de mayor nota. Si empatan, gana el primero que se sorteó.
#
# Devuelve: el cromosoma ganador.
# -------------------------------------------------------------------------
def seleccion(poblacion, notas, rng):
    elegidos = rng.choice(len(poblacion), size=config.TAMANO_TORNEO, replace=False)
    ganador = elegidos[0]
    for k in elegidos:
        if notas[k] > notas[ganador]:
            ganador = k
    return poblacion[ganador]


# -------------------------------------------------------------------------
# Cruce de dos padres.
#
# Con probabilidad PROBABILIDAD_CRUCE hay cruce, del tipo tipo_cruce.
# Si no hay cruce, o el tipo es "ninguna", el hijo es una copia del padre.
#
# Devuelve: el cromosoma del hijo.
# -------------------------------------------------------------------------
def cruzar(padre, madre, capas, tipo_cruce, rng):
    if tipo_cruce not in TIPOS_CRUCE:
        raise ValueError("El cruce tiene que ser uno de estos: " + ", ".join(TIPOS_CRUCE))
    if tipo_cruce == "ninguna" or rng.random() >= config.PROBABILIDAD_CRUCE:
        return padre.copy()

    if tipo_cruce == "neurona":
        return cruce_por_neurona(padre, madre, capas, rng)
    if tipo_cruce == "pesos":
        return cruce_por_pesos(padre, madre, rng)
    return cruce_blx(padre, madre, rng)


# -------------------------------------------------------------------------
# Cruce por neurona (Montana y Davis, 1989).
#
# Los pesos que entran a una misma neurona trabajan juntos. Por eso cada neurona
# del hijo, oculta o de salida, se copia entera de uno de los padres: todos sus
# pesos de entrada (una columna de la matriz) y su umbral.
# -------------------------------------------------------------------------
def cruce_por_neurona(padre, madre, capas, rng):
    hijo = Perceptron(capas, cromosoma=padre)
    red_madre = Perceptron(capas, cromosoma=madre)
    for capa, j in hijo.neuronas():
        # Con probabilidad 1/2, la neurona j viene de la madre; si no, queda la del padre
        if rng.random() < 0.5:
            hijo.pesos[capa][:, j] = red_madre.pesos[capa][:, j]
            hijo.umbrales[capa][j] = red_madre.umbrales[capa][j]
    return hijo.a_cromosoma()


# -------------------------------------------------------------------------
# Cruce uniforme: cada peso y cada umbral se copia del padre o de la madre,
# con igual probabilidad.
# -------------------------------------------------------------------------
def cruce_por_pesos(padre, madre, rng):
    hijo = padre.copy()
    for i in range(len(hijo)):
        if rng.random() < 0.5:
            hijo[i] = madre[i]
    return hijo


# -------------------------------------------------------------------------
# Cruce BLX-alfa (Eshelman y Schaffer, 1993).
#
# Cada gen del hijo se sortea en el intervalo entre los genes de los padres,
# agrandado ALFA_BLX veces su ancho hacia cada lado. Así el hijo puede tener
# valores intermedios, y también un poco fuera del rango de los padres.
# -------------------------------------------------------------------------
def cruce_blx(padre, madre, rng):
    hijo = padre.copy()
    for i in range(len(hijo)):
        menor = min(padre[i], madre[i])
        mayor = max(padre[i], madre[i])
        ancho = mayor - menor
        hijo[i] = rng.uniform(menor - config.ALFA_BLX * ancho, mayor + config.ALFA_BLX * ancho)
    return hijo


# -------------------------------------------------------------------------
# Mutación: suma ruido normal de desvío sigma.
#
# Devuelve: el cromosoma mutado. El original no cambia.
# -------------------------------------------------------------------------
def mutar(cromosoma, capas, sigma, tipo_mutacion, rng):
    if tipo_mutacion not in TIPOS_MUTACION:
        raise ValueError("La mutación tiene que ser una de estas: " + ", ".join(TIPOS_MUTACION))
    if tipo_mutacion == "neurona":
        return mutacion_por_neurona(cromosoma, capas, sigma, rng)
    return mutacion_gaussiana(cromosoma, sigma, rng)


# -------------------------------------------------------------------------
# Mutación gaussiana: cada peso y cada umbral muta, con probabilidad
# PROBABILIDAD_MUTACION, sumándole un valor normal de desvío sigma.
# -------------------------------------------------------------------------
def mutacion_gaussiana(cromosoma, sigma, rng):
    hijo = cromosoma.copy()
    for i in range(len(hijo)):
        if rng.random() < config.PROBABILIDAD_MUTACION:
            hijo[i] = hijo[i] + rng.normal(0, sigma)
    return hijo


# -------------------------------------------------------------------------
# Mutación por neurona (Montana y Davis, 1989).
#
# Se eligen NEURONAS_A_MUTAR neuronas al azar, ocultas o de salida, y a cada una
# se le suma ruido normal en todos sus pesos de entrada y en su umbral.
# -------------------------------------------------------------------------
def mutacion_por_neurona(cromosoma, capas, sigma, rng):
    hijo = Perceptron(capas, cromosoma=cromosoma)
    neuronas = hijo.neuronas()

    # Se eligen algunas neuronas, sin repetir, y se les suma ruido a sus pesos de entrada y a su umbral
    for k in rng.choice(len(neuronas), size=config.NEURONAS_A_MUTAR, replace=False):
        capa, j = neuronas[k]
        hijo.pesos[capa][:, j] = hijo.pesos[capa][:, j] + rng.normal(0, sigma, capas[capa])
        hijo.umbrales[capa][j] = hijo.umbrales[capa][j] + rng.normal(0, sigma)
    return hijo.a_cromosoma()


# -------------------------------------------------------------------------
# Entrenamiento directo: evoluciona la población en un solo mapa.
#
# En cada generación:
#   1. Se sortean RUTAS_POR_GENERACION rutas, las mismas para toda la población
#   2. Se evalúa la población en esas rutas
#   3. El de mayor nota se mide en las rutas de referencia. Si ahí le gana a la mejor
#      red encontrada (mayor nota, o igual nota y menos tiempo), pasa a ser la mejor
#   4. Se registra la generación en la historia
#   5. Se arma la generación siguiente
#
# Recibe: la población, el mapa, las rutas de referencia, el generador al azar, la
#         cantidad de generaciones (None = hasta interrumpir), los tipos de cruce y de
#         mutación, la semilla, y para continuar una corrida, su historia y su mejor red.
#         simular_poblacion permite evaluar en la ventana de pygame; al_guardar se llama
#         al final de cada generación con la población, la mejor red, la historia y el azar.
# Devuelve: la mejor red, su nota de referencia y la historia.
# -------------------------------------------------------------------------
def algoritmo_evolutivo(poblacion, mapa, rutas_fijas, rng, max_generaciones=config.GENERACIONES,
                        tipo_cruce=config.TIPO_CRUCE, tipo_mutacion=config.TIPO_MUTACION, semilla=0,
                        historia=None, mejor=None, simular_poblacion=None, al_guardar=None, mostrar=print):
    historia = [] if historia is None else historia
    referencia = [(mapa, ruta) for ruta in rutas_fijas]

    # Si se continúa una corrida, la mejor red se vuelve a medir. Si no, cualquier candidato le gana.
    if mejor is not None:
        mejor = mejor.copy()
        nota_mejor, tiempo_mejor, _ = fitness.medir_en_referencia(mejor, referencia, semilla)
    else:
        mejor = poblacion[0].copy()
        nota_mejor, tiempo_mejor = float("-inf"), float("inf")

    sigma = config.SIGMA
    generacion = 0
    while max_generaciones is None or generacion < max_generaciones:
        # 1. Las mismas rutas para todos los individuos de esta generación
        rutas = [ruta_al_azar(mapa, rng) for _ in range(config.RUTAS_POR_GENERACION)]
        casos = [(mapa, ruta) for ruta in rutas]

        # 2. Evaluar. Si se cerró la ventana de pygame, esta generación se descarta.
        evaluacion = fitness.evaluar(poblacion, casos, rng, simular=simular_poblacion)
        if evaluacion is None:
            break
        notas, resultados = evaluacion

        # 3. El de mayor nota es el candidato; se lo compara con la mejor red en las rutas de referencia
        indice_mejor = int(np.argmax(notas))
        candidato = poblacion[indice_mejor]
        nota_candidato, tiempo, _ = fitness.medir_en_referencia(candidato, referencia, semilla)
        if le_gana(nota_candidato, tiempo, nota_mejor, tiempo_mejor):
            mejor, nota_mejor, tiempo_mejor = candidato.copy(), nota_candidato, tiempo

        # 4. Registrar la generación
        fila = fila_de_generacion(len(historia), notas, resultados, indice_mejor, sigma, nota_candidato, nota_mejor)
        historia.append(fila)
        mostrar_generacion(fila, len(rutas), mostrar)

        # 5. La generación siguiente
        poblacion = nueva_generacion(poblacion, notas, config.CAPAS, sigma, rng, tipo_cruce, tipo_mutacion, mejor=mejor)
        if al_guardar is not None:
            al_guardar(poblacion, mejor, historia, rng)
        generacion += 1

    return mejor, nota_mejor, historia


# -------------------------------------------------------------------------
# ¿El candidato le gana a la mejor red en las rutas de referencia?
# Gana si tiene mayor nota, o la misma nota y menos tiempo.
# -------------------------------------------------------------------------
def le_gana(nota, tiempo, nota_mejor, tiempo_mejor):
    return nota > nota_mejor or (nota == nota_mejor and tiempo < tiempo_mejor)


# -------------------------------------------------------------------------
# La fila de la historia de una generación.
#
# Recibe: el número de generación, las notas y los resultados de la población,
#         cuál es el de mayor nota, el σ, la nota de referencia del candidato
#         y la de la mejor red.
# Devuelve: la fila, como diccionario.
# -------------------------------------------------------------------------
def fila_de_generacion(generacion, notas, resultados, indice_mejor, sigma, nota_candidato, nota_mejor):
    return {"generacion": generacion,
            "mejor_nota": float(notas[indice_mejor]),
            "nota_media": float(np.mean(notas)),
            "llegadas_del_mejor": fitness.contar_llegadas(resultados[indice_mejor]),
            "sigma": sigma,
            "nota_referencia_candidato": nota_candidato,
            "mejor_referencia": nota_mejor}


# -------------------------------------------------------------------------
# Muestra en una línea cómo le fue a una generación.
# -------------------------------------------------------------------------
def mostrar_generacion(fila, cantidad_rutas, mostrar):
    mostrar(f"generación {fila['generacion']}"
            f"  mejor {fila['mejor_nota']:.3f}  media {fila['nota_media']:.3f}"
            f"  llegadas del mejor {fila['llegadas_del_mejor']}/{cantidad_rutas}"
            f"  mejor referencia {fila['mejor_referencia']:.3f}  sigma {fila['sigma']}")
