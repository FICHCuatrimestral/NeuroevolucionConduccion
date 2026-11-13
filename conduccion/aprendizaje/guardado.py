"""Guardar y cargar entrenamientos.

Cada entrenamiento tiene su carpeta: resultados/<fecha y hora>_semilla<n>/. El nombre de la carpeta es su
identificador. Se actualiza en cada generación:
  - checkpoint.npz  todo lo necesario para continuar: población, mejor red, historia y estado del azar
  - mejor.npy       el mejor cromosoma encontrado hasta el momento
  - historia.json   mejor nota, nota media, llegadas y σ de cada generación
  - config.json     la semilla, los parámetros del entrenamiento y todos los valores de config.py

Los últimos tres son copias fáciles de abrir. Para continuar solo se usa checkpoint.npz.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import config

CARPETA_RESULTADOS = Path(__file__).resolve().parents[2] / "resultados"


# -------------------------------------------------------------------------
# La carpeta donde se guardan todos los entrenamientos.
# -------------------------------------------------------------------------
def donde_se_guarda():
    return CARPETA_RESULTADOS


# -------------------------------------------------------------------------
# Cambia la carpeta donde se guardan los entrenamientos. La usan los
# experimentos: cada corrida guarda en su propia carpeta.
# -------------------------------------------------------------------------
def usar_carpeta(carpeta):
    global CARPETA_RESULTADOS
    CARPETA_RESULTADOS = Path(carpeta)


# -------------------------------------------------------------------------
# Identificador de un entrenamiento nuevo: la fecha y hora, y la semilla.
# Empieza con la fecha, así las carpetas quedan ordenadas de la más vieja a la más nueva.
# -------------------------------------------------------------------------
def nuevo_entrenamiento(semilla):
    return f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}_semilla{semilla}"


# -------------------------------------------------------------------------
# La carpeta de un entrenamiento. Acepta el identificador solo o con su ruta,
# por ejemplo "resultados/2026-11-05_..._semilla0".
# -------------------------------------------------------------------------
def carpeta_de(identificador):
    carpeta = Path(identificador)
    if carpeta.is_dir():
        return carpeta
    return CARPETA_RESULTADOS / carpeta.name


# -------------------------------------------------------------------------
# Todos los parámetros de config.py, como diccionario. Son los nombres en mayúscula.
# -------------------------------------------------------------------------
def valores_de_config():
    return {nombre: getattr(config, nombre) for nombre in dir(config) if nombre.isupper()}


# -------------------------------------------------------------------------
# Guarda el estado de un entrenamiento. Se llama al final de cada generación.
#
# Recibe: el identificador, la mejor red, la historia, los parámetros de la corrida,
#         la población y el estado (generaciones completadas y estado del azar).
# -------------------------------------------------------------------------
def guardar(identificador, mejor, historia, parametros, poblacion, estado):
    carpeta = carpeta_de(identificador)
    carpeta.mkdir(parents=True, exist_ok=True)
    datos = {"parametros": parametros, "config": valores_de_config()}

    # 1. El checkpoint. Primero se escribe un archivo temporal y después se renombra:
    #    si el programa se corta a la mitad, el checkpoint anterior sigue sano.
    temporal = carpeta / ".checkpoint.tmp"
    with temporal.open("wb") as archivo:
        np.savez(archivo, mejor=mejor, poblacion=np.asarray(poblacion), historia=json.dumps(historia),
                 config=json.dumps(datos), estado=json.dumps(estado))
    temporal.replace(carpeta / "checkpoint.npz")

    # 2. Las copias fáciles de abrir
    np.save(carpeta / "mejor.npy", mejor)
    (carpeta / "historia.json").write_text(json.dumps(historia, indent=1), encoding="utf-8")
    (carpeta / "config.json").write_text(json.dumps(datos, indent=1, ensure_ascii=False), encoding="utf-8")


# -------------------------------------------------------------------------
# ¿Existe este entrenamiento?
# -------------------------------------------------------------------------
def tiene_guardado(identificador):
    return (carpeta_de(identificador) / "checkpoint.npz").exists()


# -------------------------------------------------------------------------
# Carga un entrenamiento.
# Devuelve: el mejor cromosoma, la historia y la configuración.
# -------------------------------------------------------------------------
def cargar(identificador):
    with np.load(carpeta_de(identificador) / "checkpoint.npz", allow_pickle=False) as checkpoint:
        mejor = checkpoint["mejor"].copy()
        historia = json.loads(checkpoint["historia"].item())
        datos = json.loads(checkpoint["config"].item())
    return mejor, historia, datos


# -------------------------------------------------------------------------
# Los parámetros y la configuración con los que se entrenó la red.
# Lee config.json, que es más liviano que el checkpoint.
# -------------------------------------------------------------------------
def cargar_configuracion(identificador):
    return json.loads((carpeta_de(identificador) / "config.json").read_text(encoding="utf-8"))


# -------------------------------------------------------------------------
# Lo que hace falta, además de cargar(), para continuar un entrenamiento.
# Devuelve: la población y el estado (generaciones completadas y estado del azar).
# -------------------------------------------------------------------------
def cargar_poblacion(identificador):
    with np.load(carpeta_de(identificador) / "checkpoint.npz", allow_pickle=False) as checkpoint:
        poblacion = list(checkpoint["poblacion"])
        estado = json.loads(checkpoint["estado"].item())
    return poblacion, estado


# -------------------------------------------------------------------------
# Los identificadores de todos los entrenamientos guardados, del más viejo al más nuevo.
# -------------------------------------------------------------------------
def entrenamientos_guardados():
    if not CARPETA_RESULTADOS.exists():
        return []
    return [carpeta.name for carpeta in sorted(CARPETA_RESULTADOS.iterdir()) if tiene_guardado(carpeta)]
