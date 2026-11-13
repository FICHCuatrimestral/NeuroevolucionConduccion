"""Evaluar una red entrenada.

En un mapa: rutas_del_mapa da sus rutas en un orden fijo, y manejar registra cómo maneja la red cada una.
tipo_de_ruta las separa por sus giros, y tabla_por_tipo resume las llegadas de cada tipo.

En muchos mapas, para medir si la red generaliza: evaluar_red. Cada mapa se prueba con sus
RUTAS_REFERENCIA rutas de referencia. Se maneja sin ruido en los sensores, así todas las redes
se comparan igual.

Los mapas salen de mapas/recortes.json: 58 recortes de 1 km de Santa Fe, con su nivel de dificultad (1 a 4)
y su uso ("entrenamiento" o "prueba"). Las redes del informe no entrenan en ningún recorte: entrenan en
la manzana, el 2x2 y el 3x3. Los niveles sirvieron para elegir los recortes; el informe no los usa.
También se pueden agregar otros mapas: "2x2", "3x3", "centro", "candioti" o "guadalupe".

Cada mapa se evalúa en un proceso aparte: el proceso abre el mapa, maneja las rutas y lo suelta.
Así nunca hay más de un mapa por proceso en memoria (unos 20 MB cada uno).

El resultado se guarda en resultados/evaluaciones/<identificador de la red>_<fecha>.json.

Uso desde la terminal, parado en la carpeta del proyecto:
    python -m conduccion.aprendizaje.evaluacion                       la red más nueva, todos los recortes
    python -m conduccion.aprendizaje.evaluacion --red <identificador> --niveles 1 2 --uso prueba --procesos 4
"""

import argparse
import itertools
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config
from ..mapa.mapa import santa_fe
from ..mapa.ruta import calcular_ruta, ruta_sirve
from ..parametros import capas_validas, usar_parametros_de
from ..simulacion.simulacion import Episodio, correr_episodio, salidas_a_acciones
from .entrenamiento import rutas_de_referencia
from .fitness import nota_de_ruta
from .guardado import cargar, entrenamientos_guardados
from .perceptron_multicapa import Perceptron, cantidad_de_pesos

CARPETA_MAPAS = Path(__file__).resolve().parents[2] / "mapas"
CARPETA_EVALUACIONES = Path(__file__).resolve().parents[2] / "resultados" / "evaluaciones"
TIPOS_DE_RUTA = ("sin giros", "solo izquierda", "solo derecha", "izquierda y derecha")


# =========================================================================
# UN MAPA: todas sus rutas, separadas por tipo de giro
# =========================================================================

# -------------------------------------------------------------------------
# Las rutas del mapa que sirven para entrenar, en un orden al azar pero fijo.
# Son las mismas que acepta el entrenamiento (ver ruta_sirve en ruta.py): de
# LARGO_MINIMO_RUTA o más y con todo su carril sobre la calle.
#
# Recibe: el mapa, cuántas rutas como mucho (los mapas grandes tienen miles) y la
#         semilla del orden.
# Devuelve: la lista de rutas.
# -------------------------------------------------------------------------
def rutas_del_mapa(mapa, maximo=200, semilla=0):
    # Todos los pares (A, B) de esquinas distintas, mezclados siempre igual
    pares = list(itertools.permutations(mapa.grafo.nodes, 2))
    np.random.default_rng(semilla).shuffle(pares)

    rutas = []
    for a, b in pares:
        ruta = calcular_ruta(mapa, a, b)
        if ruta_sirve(mapa, ruta):
            rutas.append(ruta)
            if len(rutas) == maximo:
                break
    return rutas


# -------------------------------------------------------------------------
# El tipo de una ruta, según hacia dónde dobla en sus esquinas.
#
# En cada esquina se mira el ángulo entre la cuadra que llega y la que sale.
# Más de 30° es un giro a la izquierda; menos de -30°, a la derecha.
#
# Devuelve: uno de TIPOS_DE_RUTA.
# -------------------------------------------------------------------------
def tipo_de_ruta(mapa, ruta):
    posiciones = [mapa.posicion(esquina) for esquina in ruta.esquinas]
    izquierda = derecha = 0

    # Cada esquina b, con la anterior a y la siguiente c
    for a, b, c in zip(posiciones, posiciones[1:], posiciones[2:]):
        # u: la cuadra que llega; v: la que sale
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - b[0], c[1] - b[1]

        # El ángulo de u a v, con signo: positivo es antihorario, o sea a la izquierda
        angulo = math.degrees(math.atan2(ux * vy - uy * vx, ux * vx + uy * vy))
        izquierda += angulo > 30
        derecha += angulo < -30

    if izquierda + derecha == 0:
        return "sin giros"
    if derecha == 0:
        return "solo izquierda"
    if izquierda == 0:
        return "solo derecha"
    return "izquierda y derecha"


# -------------------------------------------------------------------------
# La red maneja una ruta sin ruido, y se registra cada paso.
#
# Devuelve: un diccionario con la ruta, su tipo, el Resultado y la lista de pasos.
#           Cada paso tiene el tiempo, la posición, la velocidad en km/h, el volante
#           en grados, la aceleración, el ángulo al próximo punto y la calle libre al frente.
# -------------------------------------------------------------------------
def manejar(mapa, ruta, red):
    episodio = Episodio(mapa, ruta)
    pasos = []
    while not episodio.terminado():
        entradas = episodio.entradas()
        volante, aceleracion = salidas_a_acciones(red.propagar(entradas))

        # entradas[8] y entradas[9] son el seno y el coseno de la brújula;
        # entradas[3] es el rayo que mira hacia adelante, sobre el alcance
        pasos.append({"t": episodio.tiempo,
                      "x": episodio.auto.x,
                      "y": episodio.auto.y,
                      "velocidad": 3.6 * episodio.auto.velocidad,
                      "volante": math.degrees(volante),
                      "aceleracion": aceleracion,
                      "brujula": math.degrees(math.atan2(entradas[8], entradas[9])),
                      "calle_al_frente": entradas[3] * config.ALCANCE_RAYOS})
        episodio.paso(volante, aceleracion)
    return {"ruta": ruta, "tipo": tipo_de_ruta(mapa, ruta), "resultado": episodio.resultado(), "pasos": pasos}


# -------------------------------------------------------------------------
# Resume las corridas de manejar(): una fila por tipo de ruta y una de total.
#
# Devuelve: una lista de filas con rutas, llegadas, choques, sin tiempo,
#           % del tiempo fuera del carril, velocidad media y % del tiempo frenando.
# -------------------------------------------------------------------------
def tabla_por_tipo(corridas):
    filas = []
    for tipo in TIPOS_DE_RUTA + ("total",):
        grupo = [corrida for corrida in corridas if tipo in (corrida["tipo"], "total")]
        if not grupo:
            continue

        resultados = [corrida["resultado"] for corrida in grupo]
        motivos = [resultado.motivo for resultado in resultados]
        velocidades = [paso["velocidad"] for corrida in grupo for paso in corrida["pasos"]]
        frenando = [paso["aceleracion"] < 0 for corrida in grupo for paso in corrida["pasos"]]
        tiempo_fuera = sum(r.tiempo_fuera_del_carril for r in resultados)
        tiempo_total = sum(r.tiempo for r in resultados)

        filas.append({"tipo": tipo,
                      "rutas": len(grupo),
                      "llegan": motivos.count("llegada"),
                      "chocan": motivos.count("choque"),
                      "sin tiempo": motivos.count("tiempo"),
                      "% fuera del carril": round(100 * tiempo_fuera / tiempo_total, 1),
                      "km/h media": round(float(np.mean(velocidades)), 1),
                      "% frenando": round(100 * float(np.mean(frenando)), 1)})
    return filas


# =========================================================================
# MUCHOS MAPAS: los recortes de la ciudad
# =========================================================================

# -------------------------------------------------------------------------
# Los recortes de recortes.json que se pueden abrir, filtrados por nivel y por uso.
#
# Recibe: una lista de niveles (None = todos) y un uso: "entrenamiento", "prueba" o None.
# Devuelve: los datos de cada recorte elegido.
# -------------------------------------------------------------------------
def recortes(niveles=None, uso=None):
    datos = json.loads((CARPETA_MAPAS / "recortes.json").read_text(encoding="utf-8"))
    elegidos = []
    for recorte in datos["recortes"]:
        # Los recortes descartados al bajar la ciudad no tienen zona (por ejemplo, tenían pocas calles)
        if not recorte.get("zona"):
            continue
        if niveles is not None and recorte["nivel"] not in niveles:
            continue
        if uso is not None and recorte["uso"] != uso:
            continue
        elegidos.append(recorte)
    return elegidos


# -------------------------------------------------------------------------
# Un mapa que no es recorte, como "candioti", con los mismos datos que un recorte.
# -------------------------------------------------------------------------
def mapa_suelto(zona):
    return {"zona": zona, "barrio": zona, "nivel": "sin nivel", "uso": "original", "entropia": None, "rotondas": None}


# -------------------------------------------------------------------------
# Corre en un proceso aparte: la red maneja las rutas de referencia de un mapa.
#
# Recibe: el identificador de la red, los datos del mapa y la semilla de las rutas.
# Devuelve: los datos del mapa, más llegadas, choques, sin tiempo, nota media,
#           avance medio, fracción fuera del carril y velocidad media de las que llegan.
# -------------------------------------------------------------------------
def evaluar_mapa(identificador, recorte, semilla):
    # 1. La red, con los mismos parámetros con los que se entrenó
    usar_parametros_de(identificador)
    cromosoma, _, _ = cargar(identificador)
    red = Perceptron(config.CAPAS, cromosoma=cromosoma)

    # 2. Las rutas de referencia del mapa, manejadas sin ruido
    mapa = santa_fe(recorte["zona"])
    rutas = rutas_de_referencia(mapa, semilla)
    resultados = [correr_episodio(mapa, ruta, red, rng=None) for ruta in rutas]

    # 3. El resumen, junto con los datos del mapa
    return {**recorte, **resumir_rutas(resultados)}


# -------------------------------------------------------------------------
# Resume cómo le fue a una red en varias rutas.
#
# Recibe: el Resultado de cada ruta.
# Devuelve: rutas, llegadas, choques, sin tiempo, nota media, avance medio,
#           fracción fuera del carril y velocidad media de las que llegan.
# -------------------------------------------------------------------------
def resumir_rutas(resultados):
    motivos = Counter(resultado.motivo for resultado in resultados)
    llegadas = [resultado for resultado in resultados if resultado.motivo == "llegada"]
    avance = [(r.puntos_alcanzados + r.progreso) / r.total_puntos for r in resultados]
    fuera = [r.tiempo_fuera_del_carril / r.tiempo for r in resultados if r.tiempo > 0]
    return {"rutas": len(resultados),
            "llegadas": motivos["llegada"],
            "choques": motivos["choque"],
            "sin_tiempo": motivos["tiempo"],
            "nota_media": float(np.mean([nota_de_ruta(resultado) for resultado in resultados])),
            "avance_medio": float(np.mean(avance)),
            "fuera_del_carril": float(np.mean(fuera)),
            "velocidad_media": velocidad_media(llegadas)}


# -------------------------------------------------------------------------
# Velocidad media de las rutas en que se llegó. La de cada ruta es el largo de
# su recorrido dividido por el tiempo.
# Devuelve: la velocidad media en m/s, o None si no se llegó en ninguna.
# -------------------------------------------------------------------------
def velocidad_media(llegadas):
    if not llegadas:
        return None
    velocidades = [sum(np.hypot(*np.diff(np.array(r.recorrido), axis=0).T)) / r.tiempo for r in llegadas]
    return float(np.mean(velocidades))


# -------------------------------------------------------------------------
# Comprueba que una red guardada tenga las entradas que usa el código actual.
# Si no, da error: una red vieja no puede leer los sensores de ahora.
# -------------------------------------------------------------------------
def comprobar_entradas(identificador):
    cromosoma, _, datos = cargar(identificador)
    capas = datos["config"]["CAPAS"]
    if not capas_validas(capas) or len(cromosoma) != cantidad_de_pesos(capas):
        raise ValueError(f"La red {identificador} tiene {capas[0]} entradas y el código actual usa "
                         f"{config.CANTIDAD_ENTRADAS}.")


# -------------------------------------------------------------------------
# Evalúa la red en cada mapa de la lista, un mapa por proceso.
#
# Recibe: el identificador de la red, la lista de mapas (de recortes() o mapa_suelto()),
#         la semilla de las rutas y la cantidad de procesos (None = todos los núcleos).
# Devuelve: el resumen de cada mapa, en el mismo orden que la lista.
# -------------------------------------------------------------------------
def evaluar_red(identificador, mapas, semilla=0, procesos=None, mostrar=print):
    # 1. La red tiene que tener las entradas que usa el código actual
    comprobar_entradas(identificador)

    # 2. No tiene sentido abrir más procesos que mapas
    if procesos is None:
        procesos = os.cpu_count() or 1
    procesos = max(1, min(procesos, len(mapas)))
    mostrar(f"Evaluando {identificador} en {len(mapas)} mapas con {procesos} procesos.")

    # 3. Cada mapa en un proceso. Los resultados llegan en cualquier orden y se guardan en su lugar.
    resultados = [None] * len(mapas)
    with ProcessPoolExecutor(procesos) as pool:
        tareas = {pool.submit(evaluar_mapa, identificador, recorte, semilla): i for i, recorte in enumerate(mapas)}
        for hechas, tarea in enumerate(as_completed(tareas), start=1):
            i = tareas[tarea]
            resultados[i] = tarea.result()
            r = resultados[i]
            mostrar(f"{hechas}/{len(mapas)}  {r['zona']}  llega {r['llegadas']}/{r['rutas']}"
                    f"  nota {r['nota_media']:.3f}")
    return resultados


# -------------------------------------------------------------------------
# Suma llegadas y rutas por nivel o por uso.
#
# Recibe: los resultados de evaluar_red y la clave: "nivel" o "uso".
# Devuelve: {valor: {"mapas", "llegadas", "rutas", "fraccion"}}.
# -------------------------------------------------------------------------
def resumen(resultados, clave):
    grupos = {}
    for r in resultados:
        grupo = grupos.setdefault(str(r[clave]), {"mapas": 0, "llegadas": 0, "rutas": 0})
        grupo["mapas"] += 1
        grupo["llegadas"] += r["llegadas"]
        grupo["rutas"] += r["rutas"]
    for grupo in grupos.values():
        grupo["fraccion"] = grupo["llegadas"] / grupo["rutas"]
    return dict(sorted(grupos.items()))


# -------------------------------------------------------------------------
# Guarda la evaluación en resultados/evaluaciones/, con el resumen por nivel y por uso.
# Devuelve: la ruta del archivo.
# -------------------------------------------------------------------------
def guardar_evaluacion(identificador, resultados, semilla):
    CARPETA_EVALUACIONES.mkdir(parents=True, exist_ok=True)
    _, historia, datos = cargar(identificador)
    salida = {"red": str(identificador),
              "fecha": datetime.now().isoformat(timespec="seconds"),
              "entrenada_en": datos["parametros"]["mapa"],
              "generaciones": len(historia),
              "semilla_rutas": semilla,
              "por_nivel": resumen(resultados, "nivel"),
              "por_uso": resumen(resultados, "uso"),
              "mapas": resultados}
    nombre = f"{Path(str(identificador)).name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.json"
    archivo = CARPETA_EVALUACIONES / nombre
    archivo.write_text(json.dumps(salida, indent=1, ensure_ascii=False), encoding="utf-8")
    return archivo


# -------------------------------------------------------------------------
# Muestra las llegadas por nivel y por uso, en una tabla de texto.
# -------------------------------------------------------------------------
def mostrar_resumen(resultados, mostrar=print):
    for clave in ("nivel", "uso"):
        mostrar(f"Por {clave}:")
        for valor, grupo in resumen(resultados, clave).items():
            mostrar(f"  {valor:<14}{grupo['llegadas']:>5}/{grupo['rutas']:<5}"
                    f" ({100 * grupo['fraccion']:.0f} %)  en {grupo['mapas']} mapas")


# =========================================================================
# TERMINAL
# =========================================================================

# -------------------------------------------------------------------------
# Lee los argumentos de la terminal, evalúa, muestra el resumen y lo guarda.
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evalúa una red entrenada en muchos mapas.")
    parser.add_argument("--red", help="identificador de la red; por defecto, la más nueva")
    parser.add_argument("--niveles", type=int, nargs="*", help="niveles de recortes, de 1 a 4; por defecto, todos")
    parser.add_argument("--uso", choices=("entrenamiento", "prueba"), help="solo recortes de este uso")
    parser.add_argument("--extra", nargs="*", default=[], help='mapas que no son recortes, por ejemplo "centro candioti"')
    parser.add_argument("--semilla", type=int, default=0, help="semilla de las rutas de referencia")
    parser.add_argument("--procesos", type=int, help="por defecto, todos los núcleos")
    argumentos = parser.parse_args()

    identificador = argumentos.red or entrenamientos_guardados()[-1]
    mapas = recortes(argumentos.niveles, argumentos.uso) + [mapa_suelto(zona) for zona in argumentos.extra]
    resultados = evaluar_red(identificador, mapas, argumentos.semilla, argumentos.procesos)
    mostrar_resumen(resultados)
    print("Guardado en", guardar_evaluacion(identificador, resultados, argumentos.semilla))


if __name__ == "__main__":
    main()
