"""Comparaciones para el informe: se cambian factores del entrenamiento y se mide cuánto aporta cada uno.

FACTORES tiene los valores posibles de cada factor y BASE, la configuración de base del informe.
Hay dos formas de armar las combinaciones:
  - "uno_por_vez": la base y, por cada factor, la base con ese factor cambiado. Dice cuánto aporta
    cada factor contra la base. Son 6 corridas.
  - "todo": todas las combinaciones de todos los factores. Muestra además si dos factores se
    afectan entre sí, pero son 16 corridas.
Para agregar un factor alcanza con sumar una línea a FACTORES, otra a BASE y decir en
parametros_de cómo se aplica.

Todas las corridas entrenan por etapas, en ETAPAS, con cada una de las SEMILLAS. Una corrida
(combinación, semilla) guarda todo en:
    resultados/comparaciones/<combinación>/semilla<n>/
      corrida.json          el número de la corrida, su combinación y su semilla
      <fecha>_semilla<n>/   un entrenamiento guardado por etapa
      registro.txt          lo que se mostró durante el entrenamiento
      entrenamiento.json    se escribe al terminar de entrenar: el identificador de cada etapa
      evaluacion.json       se escribe al terminar de probar: una fila por ruta de prueba en el centro
y resultados/comparaciones/indice.json e indice.xlsx tienen una fila por corrida terminada.

Cada corrida tiene un número al azar de 4 cifras, para nombrarla en el informe.

El nombre de la carpeta de una combinación solo depende de sus valores: si una combinación aparece
en las dos formas, se corre una sola vez.

Una corrida que tiene evaluacion.json ya terminó. Si se cortó a la mitad, correr_corrida la
sigue desde la última generación guardada, con el mismo resultado que si no se hubiera cortado.

Se corren con scripts/correr_comparaciones.py.
"""

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

from .. import config
from ..mapa.mapa import santa_fe
from ..mapa.ruta import distancias_a_giros
from ..parametros import usar_parametros, volver_a_config
from ..simulacion.simulacion import Episodio
from .entrenamiento import rutas_de_referencia
from .etapas import AJUSTES_ETAPAS, entrenar_por_etapas, etapas_santa_fe
from .fitness import nota_de_ruta
from .guardado import cargar, cargar_configuracion, entrenamientos_guardados, usar_carpeta
from .perceptron_multicapa import Perceptron

CARPETA_COMPARACIONES = Path(__file__).resolve().parents[2] / "resultados" / "comparaciones"

# Los factores y sus valores posibles
FACTORES = {
    "cruce": ["neurona", "pesos", "blx", "ninguna"],
    "mutacion": ["gaussiana", "neurona"],
    "agarre": [True, False],
}

# La configuración de base del informe
BASE = {"cruce": "neurona", "mutacion": "gaussiana", "agarre": True}

# Las semillas de todas las combinaciones
SEMILLAS = [0]

# Tamaño de la población en todas las corridas
POBLACION = 50

# Las etapas: manzana → 2x2 → 3x3, con los ajustes del informe y menos generaciones,
# para que las 6 corridas de "uno_por_vez" entren en una noche en una PC de 4 núcleos.
# El centro no se entrena: es el mapa de prueba.
ETAPAS = {
    "manzana": {**AJUSTES_ETAPAS["manzana"], "maximo_generaciones": 100},
    "2x2": {**AJUSTES_ETAPAS["2x2"], "maximo_generaciones": 50},
    "3x3": {**AJUSTES_ETAPAS["3x3"], "maximo_generaciones": 50},
}

# El mapa donde se prueba la red al terminar de entrenar
MAPA_DE_PRUEBA = "centro"

# Semilla de las rutas de prueba: la misma para todas las redes, así manejan las mismas rutas
SEMILLA_PRUEBA = 0

# Para la velocidad cerca y lejos de los giros: a cuántos metros de un giro como mucho
# se está "cerca", y desde cuántos metros se está "lejos"
CERCA_DE_UN_GIRO = 15.0
LEJOS_DE_UN_GIRO = 30.0

# Segundos que tarda una ruta en cada mapa, y la prueba final completa, medidos en una
# PC de 4 núcleos. Solo se usan para estimar cuánto tarda una corrida.
SEGUNDOS_POR_RUTA = {"manzana": 0.3, "2x2": 0.55, "3x3": 0.55, "centro": 2.0}
SEGUNDOS_DE_PRUEBA = 60


# =========================================================================
# LAS COMBINACIONES
# =========================================================================

# -------------------------------------------------------------------------
# Las combinaciones de factores.
#
# Recibe: "uno_por_vez" o "todo".
# Devuelve: una lista de diccionarios {factor: valor}.
# -------------------------------------------------------------------------
def combinaciones(forma):
    if forma == "todo":
        return [dict(zip(FACTORES, valores)) for valores in itertools.product(*FACTORES.values())]
    if forma == "uno_por_vez":
        lista = [dict(BASE)]
        for factor, valores in FACTORES.items():
            for valor in valores:
                if valor != BASE[factor]:
                    lista.append({**BASE, factor: valor})
        return lista
    raise ValueError('La forma tiene que ser "uno_por_vez" o "todo".')


# -------------------------------------------------------------------------
# El nombre de una combinación, que es también el de su carpeta.
# Por ejemplo: "neurona_gaussiana_agarre".
# -------------------------------------------------------------------------
def nombre_de(combinacion):
    agarre = "agarre" if combinacion["agarre"] else "sin-agarre"
    return f"{combinacion['cruce']}_{combinacion['mutacion']}_{agarre}"


# -------------------------------------------------------------------------
# Los valores de config.py que cambian respecto de la base, según la combinación.
# -------------------------------------------------------------------------
def parametros_de(combinacion):
    return {"ACELERACION_LATERAL_MAXIMA": 6.0 if combinacion["agarre"] else None}


# -------------------------------------------------------------------------
# Cuántas rutas maneja cada individuo por generación en una etapa: en la manzana,
# 2 vueltas; en las demás, RUTAS_POR_GENERACION. Desde la segunda etapa, una más
# de una etapa anterior.
# -------------------------------------------------------------------------
def rutas_por_generacion(numero_etapa, nombre):
    rutas = 2 if nombre == "manzana" else config.RUTAS_POR_GENERACION
    return rutas + 1 if numero_etapa > 0 else rutas


# =========================================================================
# UNA CORRIDA
# =========================================================================

# -------------------------------------------------------------------------
# La carpeta de una corrida.
# -------------------------------------------------------------------------
def carpeta_de_corrida(combinacion, semilla):
    return CARPETA_COMPARACIONES / nombre_de(combinacion) / f"semilla{semilla}"


# -------------------------------------------------------------------------
# ¿Ya terminó esta corrida?
# -------------------------------------------------------------------------
def corrida_terminada(combinacion, semilla):
    return (carpeta_de_corrida(combinacion, semilla) / "evaluacion.json").exists()


# -------------------------------------------------------------------------
# Los números de todas las corridas que ya tienen uno.
# -------------------------------------------------------------------------
def numeros_usados():
    return {json.loads(archivo.read_text(encoding="utf-8"))["numero"]
            for archivo in CARPETA_COMPARACIONES.glob("*/semilla*/corrida.json")}


# -------------------------------------------------------------------------
# Le da a una corrida un número al azar de 4 cifras, distinto de los demás, y lo
# guarda en corrida.json. Si la corrida ya tiene número, lo deja.
#
# Devuelve: el número de la corrida.
# -------------------------------------------------------------------------
def numerar(combinacion, semilla):
    carpeta = carpeta_de_corrida(combinacion, semilla)
    archivo = carpeta / "corrida.json"
    if archivo.exists():
        return json.loads(archivo.read_text(encoding="utf-8"))["numero"]
    usados = numeros_usados()
    numero = random.randint(1000, 9999)
    while numero in usados:
        numero = random.randint(1000, 9999)
    carpeta.mkdir(parents=True, exist_ok=True)
    corrida = {"numero": numero, "combinacion": combinacion, "semilla": semilla}
    archivo.write_text(json.dumps(corrida, indent=1), encoding="utf-8")
    return numero


# -------------------------------------------------------------------------
# Corre una corrida completa: entrena (o sigue entrenando) y prueba la mejor red.
# Está pensada para correr en un proceso aparte: cambia config y la carpeta de guardado.
#
# Recibe: la combinación y la semilla.
# Devuelve: un texto corto con cómo terminó.
# -------------------------------------------------------------------------
def correr_corrida(combinacion, semilla):
    if corrida_terminada(combinacion, semilla):
        return "ya estaba terminada"
    carpeta = carpeta_de_corrida(combinacion, semilla)
    numero = numerar(combinacion, semilla)

    # 1. Los entrenamientos de esta corrida se guardan en su carpeta, con sus parámetros
    usar_carpeta(carpeta)
    volver_a_config()
    usar_parametros(**parametros_de(combinacion))
    mostrar = registro_en(carpeta / "registro.txt")

    # 2. Entrenar, salvo que ya esté
    archivo_entrenamiento = carpeta / "entrenamiento.json"
    if not archivo_entrenamiento.exists():
        identificadores = entrenar_etapas(combinacion, semilla, mostrar)
        archivo_entrenamiento.write_text(json.dumps({"identificadores": identificadores}, indent=1),
                                         encoding="utf-8")
    identificadores = json.loads(archivo_entrenamiento.read_text(encoding="utf-8"))["identificadores"]

    # 3. Probar la mejor red, la de la última etapa, en el mapa de prueba
    filas = probar(identificadores[-1], mostrar)
    evaluacion = {"numero": numero, "combinacion": combinacion, "semilla": semilla, "red": identificadores[-1],
                  "rutas": filas}
    (carpeta / "evaluacion.json").write_text(json.dumps(evaluacion, indent=1, ensure_ascii=False), encoding="utf-8")
    llegadas = sum(fila["motivo"] == "llegada" for fila in filas)
    return f"llega en {llegadas} de {len(filas)} rutas de prueba"


# -------------------------------------------------------------------------
# Una función que escribe cada mensaje al final de un archivo. Cada corrida tiene
# su registro: así los mensajes de corridas en paralelo no se mezclan.
# -------------------------------------------------------------------------
def registro_en(archivo):
    def mostrar(*partes):
        with open(archivo, "a", encoding="utf-8") as registro:
            print(*partes, file=registro)
    return mostrar


# -------------------------------------------------------------------------
# Entrena por etapas. Si hay entrenamientos guardados en la carpeta, sigue desde
# el último, en la misma etapa y desde la generación siguiente.
#
# Devuelve: el identificador del último guardado de cada etapa, en orden.
#           Si el entrenamiento se cortó antes de terminar, da error.
# -------------------------------------------------------------------------
def entrenar_etapas(combinacion, semilla, mostrar):
    guardados = entrenamientos_guardados()
    entrenar_por_etapas(etapas_santa_fe(semilla, ETAPAS), semilla=semilla, tamano_poblacion=POBLACION,
                        tipo_cruce=combinacion["cruce"], tipo_mutacion=combinacion["mutacion"], mostrar=mostrar,
                        continuar_desde=guardados[-1] if guardados else None)

    # El último guardado de cada etapa. La última etapa siempre usa todas sus generaciones.
    ultimo = {}
    for identificador in entrenamientos_guardados():
        ultimo[cargar_configuracion(identificador)["parametros"]["nombre_etapa"]] = identificador
    ultima_etapa = list(ETAPAS)[-1]
    necesarias = ETAPAS[ultima_etapa]["maximo_generaciones"]
    if ultima_etapa not in ultimo or generaciones_hechas(ultimo[ultima_etapa]) < necesarias:
        raise RuntimeError("El entrenamiento se cortó antes de terminar.")
    return [ultimo[nombre] for nombre in ETAPAS if nombre in ultimo]


# -------------------------------------------------------------------------
# Cuántas generaciones tiene la historia de un entrenamiento guardado.
# -------------------------------------------------------------------------
def generaciones_hechas(identificador):
    _, historia, _ = cargar(identificador)
    return len(historia)


# =========================================================================
# LA PRUEBA
# =========================================================================

# -------------------------------------------------------------------------
# Los mapas donde se prueba: solo MAPA_DE_PRUEBA, que no es parte del entrenamiento.
# -------------------------------------------------------------------------
def mapas_de_prueba():
    return [{"zona": MAPA_DE_PRUEBA, "uso": "prueba"}]


# -------------------------------------------------------------------------
# La red maneja, sin ruido, las rutas de referencia de cada mapa de prueba.
#
# Recibe: el identificador de la red y la función mostrar.
# Devuelve: una fila por ruta, con el mapa y las medidas de medir_ruta.
# -------------------------------------------------------------------------
def probar(identificador, mostrar):
    cromosoma, _, _ = cargar(identificador)
    red = Perceptron(config.CAPAS, cromosoma=cromosoma)
    filas = []
    for recorte in mapas_de_prueba():
        mapa = santa_fe(recorte["zona"])
        del_mapa = [{"zona": recorte["zona"], "ruta": numero, **medir_ruta(mapa, ruta, red)}
                    for numero, ruta in enumerate(rutas_de_referencia(mapa, SEMILLA_PRUEBA))]
        filas += del_mapa
        llegadas = sum(fila["motivo"] == "llegada" for fila in del_mapa)
        mostrar(f"prueba {recorte['zona']} ({recorte['uso']}): llega en {llegadas}/{len(del_mapa)}")
    return filas


# -------------------------------------------------------------------------
# La red maneja una ruta sin ruido, y se mide cómo manejó.
#
# Devuelve: cómo terminó, la nota, el tiempo, la fracción del tiempo fuera del carril,
#           la velocidad media (km/h) cerca y lejos de los giros, y la aceleración
#           lateral máxima (m/s²). Si la ruta no pasa cerca o lejos de un giro, esa
#           velocidad es None.
# -------------------------------------------------------------------------
def medir_ruta(mapa, ruta, red):
    giros = distancias_a_giros(ruta.carril)
    episodio = Episodio(mapa, ruta)
    cerca, lejos, laterales = [], [], []
    while not episodio.terminado():
        episodio.paso_con_red(red)
        auto = episodio.auto
        laterales.append(aceleracion_lateral(auto.velocidad, auto.volante))

        # Los metros hasta el giro más cercano, medidos a lo largo del carril
        sobre_el_carril = ruta.carril.project(shapely.Point(auto.x, auto.y))
        al_giro = min((abs(sobre_el_carril - giro) for giro in giros), default=math.inf)
        if al_giro <= CERCA_DE_UN_GIRO:
            cerca.append(3.6 * auto.velocidad)
        elif al_giro >= LEJOS_DE_UN_GIRO:
            lejos.append(3.6 * auto.velocidad)

    resultado = episodio.resultado()
    return {"motivo": resultado.motivo,
            "nota": nota_de_ruta(resultado),
            "tiempo": resultado.tiempo,
            "fuera_del_carril": resultado.tiempo_fuera_del_carril / resultado.tiempo,
            "velocidad_cerca": float(np.mean(cerca)) if cerca else None,
            "velocidad_lejos": float(np.mean(lejos)) if lejos else None,
            "lateral_maxima": max(laterales)}


# -------------------------------------------------------------------------
# Aceleración lateral del modelo de bicicleta: a_y = v² · sin(beta) / l_r.
# -------------------------------------------------------------------------
def aceleracion_lateral(velocidad, volante):
    beta = math.atan(config.L_R / (config.L_F + config.L_R) * math.tan(volante))
    return velocidad ** 2 * abs(math.sin(beta)) / config.L_R


# =========================================================================
# EL ÍNDICE Y CUÁNTO TARDA
# =========================================================================

# -------------------------------------------------------------------------
# Escribe resultados/comparaciones/indice.json e indice.xlsx: una fila por corrida
# terminada, con su número, su combinación, su semilla, su red, y las llegadas
# en los mapas de prueba: en total y en cada mapa.
#
# Devuelve: un aviso si no pudo escribir el Excel (por ejemplo, porque está
#           abierto), o None.
# -------------------------------------------------------------------------
def actualizar_indice():
    filas = []
    for archivo in sorted(CARPETA_COMPARACIONES.glob("*/semilla*/evaluacion.json")):
        evaluacion = json.loads(archivo.read_text(encoding="utf-8"))
        fila = {"numero": evaluacion["numero"], **evaluacion["combinacion"], "semilla": evaluacion["semilla"],
                "red": evaluacion["red"], "carpeta": archivo.parent.relative_to(CARPETA_COMPARACIONES).as_posix()}
        rutas = evaluacion["rutas"]
        fila["llegadas"] = sum(ruta["motivo"] == "llegada" for ruta in rutas)
        fila["rutas"] = len(rutas)
        for zona in dict.fromkeys(ruta["zona"] for ruta in rutas):
            fila[zona] = sum(ruta["motivo"] == "llegada" for ruta in rutas if ruta["zona"] == zona)
        filas.append(fila)
    CARPETA_COMPARACIONES.mkdir(parents=True, exist_ok=True)
    (CARPETA_COMPARACIONES / "indice.json").write_text(json.dumps(filas, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
    try:
        pd.DataFrame(filas).to_excel(CARPETA_COMPARACIONES / "indice.xlsx", index=False)
    except PermissionError:
        return "no se pudo escribir indice.xlsx: ¿está abierto en Excel? Se escribe de nuevo con la próxima corrida."
    return None


# -------------------------------------------------------------------------
# Segundos que tarda como mucho una corrida en un núcleo, si todas las etapas usan
# todas sus generaciones. Incluye la prueba final.
# -------------------------------------------------------------------------
def segundos_como_mucho():
    entrenamiento = sum(ajustes["maximo_generaciones"] * POBLACION * rutas_por_generacion(numero, nombre)
                        * SEGUNDOS_POR_RUTA[nombre]
                        for numero, (nombre, ajustes) in enumerate(ETAPAS.items()))
    return entrenamiento + SEGUNDOS_DE_PRUEBA
