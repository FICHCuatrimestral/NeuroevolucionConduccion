"""Cambiar parámetros desde un notebook, sin tocar config.py.

Todos los módulos leen los parámetros como config.NOMBRE en el momento de usarlos. Por eso,
cambiar un valor en config alcanza para que todo el paquete use el valor nuevo.

Solo se pueden cambiar los parámetros de CAMBIABLES. Los demás quedan fijos porque se usan para
calcular otros valores al importar (como ALCANCE_RAYOS) o son valores por defecto de una función (como ALFA_BLX).

CAPAS se elige antes de crear una población o al cargar una red guardada: cambiarla no transforma
las redes que ya existen.
"""

from . import config
from .aprendizaje.guardado import cargar_configuracion

# Los parámetros que se pueden cambiar, agrupados como en config.py
CAMBIABLES = (
    # Auto
    "ACELERACION_LATERAL_MAXIMA", "GIRO_VOLANTE_MAXIMO", "ACELERACION_MAXIMA", "FRENADA_MAXIMA", "VELOCIDAD_MAXIMA",
    # Ruta
    "SEPARACION_PUNTOS", "PUNTOS_EN_GIROS", "ANGULO_GIRO", "RADIO_PUNTO", "PUNTOS_SALTEABLES",
    "VELOCIDAD_REFERENCIA", "TIEMPO_EXTRA",
    # Aptitud
    "PENALIZACION_CARRIL", "PREMIO_TIEMPO",
    # Red
    "CAPAS", "DESVIO_PESOS_INICIALES",
    # Algoritmo genético
    "TAMANO_TORNEO", "ELITE", "PROBABILIDAD_CRUCE", "PROBABILIDAD_MUTACION", "NEURONAS_A_MUTAR", "SIGMA",
    # Entrenamiento
    "RUTAS_POR_GENERACION", "RUTAS_REFERENCIA",
)

# Los valores de config.py, para poder volver a ellos
ORIGINALES = {nombre: getattr(config, nombre) for nombre in CAMBIABLES}


# -------------------------------------------------------------------------
# ¿Estas capas sirven para el código actual? Tienen que ser enteros positivos,
# con CANTIDAD_ENTRADAS neuronas de entrada y 2 de salida (volante y acelerador).
# -------------------------------------------------------------------------
def capas_validas(capas):
    if not isinstance(capas, (list, tuple)) or len(capas) < 2:
        return False
    if any(type(n) is not int or n < 1 for n in capas):
        return False
    return capas[0] == config.CANTIDAD_ENTRADAS and capas[-1] == 2


# -------------------------------------------------------------------------
# Cambia parámetros. Por ejemplo: usar_parametros(ELITE=4, PREMIO_TIEMPO=0.1).
#
# Recibe: los parámetros como argumentos con nombre. Solo valen los de CAMBIABLES.
# -------------------------------------------------------------------------
def usar_parametros(**valores):
    for nombre in valores:
        if nombre not in CAMBIABLES:
            raise ValueError(f"{nombre} no se puede cambiar desde acá. Se pueden cambiar: {', '.join(CAMBIABLES)}")

    if "CAPAS" in valores:
        if not capas_validas(valores["CAPAS"]):
            raise ValueError(f"CAPAS debe tener tamaños enteros positivos, {config.CANTIDAD_ENTRADAS} entradas "
                             "y 2 salidas.")
        valores["CAPAS"] = list(valores["CAPAS"])

    for nombre, valor in valores.items():
        setattr(config, nombre, valor)


# -------------------------------------------------------------------------
# Deja todos los parámetros como están en config.py.
# -------------------------------------------------------------------------
def volver_a_config():
    usar_parametros(**ORIGINALES)


# -------------------------------------------------------------------------
# Usa los parámetros con los que se hizo un entrenamiento guardado. Así una red
# maneja con la misma física con la que se entrenó.
#
# Devuelve: los parámetros que son distintos de config.py.
# -------------------------------------------------------------------------
def usar_parametros_de(identificador):
    guardados = cargar_configuracion(identificador)["config"]
    valores = {nombre: guardados.get(nombre, ORIGINALES[nombre]) for nombre in CAMBIABLES}
    usar_parametros(**valores)
    return {nombre: valor for nombre, valor in valores.items() if valor != ORIGINALES[nombre]}
