"""Menú para ver manejar una red o entrenar una nueva, sin abrir notebooks.

Los mapas son calles reales de Santa Fe. Cada entrenamiento se guarda en su carpeta de resultados/.

Se corre desde la carpeta del proyecto: python scripts/simulador.py
"""

import sys
from pathlib import Path

# No se crean carpetas __pycache__, y el paquete conduccion se busca en la carpeta de arriba
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conduccion import config
from conduccion.aprendizaje.genetico import TIPOS_CRUCE, TIPOS_MUTACION
from conduccion.aprendizaje.guardado import cargar, donde_se_guarda, entrenamientos_guardados
from conduccion.aprendizaje.perceptron_multicapa import Perceptron, cantidad_de_pesos
from conduccion.mapa.mapa import santa_fe
from conduccion.parametros import capas_validas, usar_parametros, usar_parametros_de, volver_a_config

# Los mapas del menú, con el nombre que se muestra
ZONAS = {"2x2": "2x2 manzanas",
         "3x3": "3x3 manzanas",
         "centro": "centro, unos 900 x 900 m",
         "candioti": "barrio Candioti",
         "guadalupe": "barrio Guadalupe (prueba)"}

NEURONAS_OCULTAS = (8, 16, 32)

# Variantes del auto y de la nota que se comparan en el informe
CONFIGURACIONES = [
    ("Base del informe: agarre de 6 m/s², sin premio por tiempo",
     {"ACELERACION_LATERAL_MAXIMA": 6.0, "PREMIO_TIEMPO": 0.0}),
    ("Agarre de 6 m/s² y premio por llegar rápido",
     {"ACELERACION_LATERAL_MAXIMA": 6.0, "PREMIO_TIEMPO": 0.1}),
    ("Sin límite de agarre y sin premio",
     {"ACELERACION_LATERAL_MAXIMA": None, "PREMIO_TIEMPO": 0.0}),
]


# =========================================================================
# PREGUNTAS
# =========================================================================

# -------------------------------------------------------------------------
# Pide un número entero. Con Enter se usa el valor sugerido.
# -------------------------------------------------------------------------
def pedir_entero(nombre, valor_inicial, minimo):
    while True:
        texto = input(f"{nombre} [{valor_inicial}]: ").strip()
        if texto == "":
            return valor_inicial
        if texto.isdecimal() and int(texto) >= minimo:
            return int(texto)
        print(f"Escribí un número entero mayor o igual a {minimo}.")


# -------------------------------------------------------------------------
# Muestra las opciones numeradas desde 1. Con Enter se elige la inicial.
# Devuelve: la posición de la opción elegida, desde 0.
# -------------------------------------------------------------------------
def elegir(titulo, opciones, inicial=0):
    print(titulo)
    for numero, opcion in enumerate(opciones, 1):
        print(f"  {numero}) {opcion}")
    while True:
        texto = input(f"Elegí una opción [{inicial + 1}]: ").strip()
        if texto == "":
            return inicial
        if texto.isdecimal() and 1 <= int(texto) <= len(opciones):
            return int(texto) - 1
        print(f"Elegí un número del 1 al {len(opciones)}.")


# -------------------------------------------------------------------------
# Pregunta en qué mapa. Devuelve: la zona, por ejemplo "3x3".
# -------------------------------------------------------------------------
def elegir_zona(inicial="2x2"):
    zonas = list(ZONAS)
    return zonas[elegir("Mapa:", list(ZONAS.values()), zonas.index(inicial))]


# -------------------------------------------------------------------------
# Pregunta cuántas generaciones. Devuelve: un número, o None para seguir hasta que se detenga.
# -------------------------------------------------------------------------
def pedir_generaciones():
    if elegir("Duración:", ["Una cantidad de generaciones", "Hasta que lo detengas"]) == 1:
        return None
    return pedir_entero("Generaciones", 30, 1)


# =========================================================================
# REDES GUARDADAS
# =========================================================================

# -------------------------------------------------------------------------
# La zona de un entrenamiento guardado: "santa fe 3x3" da "3x3".
# Devuelve: None si no es una zona del menú (por ejemplo, la etapa de la manzana).
# -------------------------------------------------------------------------
def zona_de(datos):
    zona = datos["parametros"]["mapa"].split()[-1]
    return zona if zona in ZONAS else None


# -------------------------------------------------------------------------
# Carga una red guardada y revisa que el código actual la pueda usar:
# 16 entradas, 2 salidas, el largo de cromosoma correcto y una zona del menú.
#
# Devuelve: un diccionario con el identificador, el cromosoma, la configuración
#           y la historia, o None si la red no sirve.
# -------------------------------------------------------------------------
def leer_red(identificador):
    try:
        cromosoma, historia, datos = cargar(identificador)
        capas = datos["config"]["CAPAS"]
        if not capas_validas(capas):
            return None
    except (OSError, ValueError, EOFError, KeyError, TypeError):
        return None

    if not historia or cromosoma.ndim != 1 or len(cromosoma) != cantidad_de_pesos(capas):
        return None
    if zona_de(datos) is None:
        return None
    return {"identificador": identificador, "cromosoma": cromosoma, "datos": datos, "historia": historia}


# -------------------------------------------------------------------------
# Una línea que describe una red guardada: zona, capa oculta, generaciones,
# agarre y nota de referencia.
# -------------------------------------------------------------------------
def describir(red):
    datos = red["datos"]
    agarre = datos["config"].get("ACELERACION_LATERAL_MAXIMA")
    ocultas = " → ".join(str(n) for n in datos["config"]["CAPAS"][1:-1])
    con_o_sin = "con agarre" if agarre is not None else "sin agarre"
    referencia = red["historia"][-1]["mejor_referencia"]
    return (f"{zona_de(datos)}, {ocultas} ocultas, {len(red['historia'])} generaciones, {con_o_sin}, "
            f"referencia {referencia:.3f}")


# -------------------------------------------------------------------------
# Las redes guardadas que se pueden usar, de la más nueva a la más vieja.
# -------------------------------------------------------------------------
def redes_disponibles():
    redes = [leer_red(identificador) for identificador in reversed(entrenamientos_guardados())]
    return [red for red in redes if red is not None]


# -------------------------------------------------------------------------
# Muestra las redes, cada una con la fecha y su descripción, y pregunta cuál.
# Devuelve: la red elegida.
# -------------------------------------------------------------------------
def elegir_red(titulo, redes):
    nombres = [f"{red['identificador'][:19]}: {describir(red)}" for red in redes]
    return redes[elegir(titulo, nombres)]


# =========================================================================
# LAS OPCIONES DEL MENÚ
# =========================================================================

# -------------------------------------------------------------------------
# Opción 1: elegir una red guardada y un mapa, y verla manejar.
# -------------------------------------------------------------------------
def ver_auto():
    redes = redes_disponibles()
    if not redes:
        print("No hay redes para ver. Entrená una con la opción 2 o 3.")
        return

    red = elegir_red("Red:", redes)
    zona = elegir_zona(zona_de(red["datos"]))

    # La red maneja con los mismos parámetros con los que se entrenó, como el límite de agarre
    volver_a_config()
    usar_parametros_de(red["identificador"])

    # pygame se importa recién acá: solo hace falta en las opciones que abren una ventana
    from conduccion.visualizacion.simulador_pygame import ver_simulacion

    print("Se abre una ventana. Puede quedar detrás de esta.")
    print("Teclas: N = otra ruta, R = reiniciar, Espacio = pausa, flechas = más o menos rápido, "
          "S = sensores, Esc = salir.")
    ver_simulacion(santa_fe(zona), Perceptron(config.CAPAS, cromosoma=red["cromosoma"]))


# -------------------------------------------------------------------------
# Pregunta qué entrenamiento guardado continuar.
# Devuelve: los datos de la red elegida, o None si no hay ninguno.
# -------------------------------------------------------------------------
def elegir_entrenamiento_guardado():
    redes = redes_disponibles()
    if not redes:
        print("Todavía no hay entrenamientos guardados.")
        return None
    return elegir_red("Entrenamiento:", redes)


# -------------------------------------------------------------------------
# Pregunta todo lo de un entrenamiento desde cero y aplica los parámetros.
# Devuelve: la zona y las opciones para entrenar().
# -------------------------------------------------------------------------
def pedir_entrenamiento_nuevo():
    zona = elegir_zona()

    # 1. El auto y la nota
    opcion = elegir("Auto y nota:", [nombre for nombre, _ in CONFIGURACIONES])
    volver_a_config()
    usar_parametros(**CONFIGURACIONES[opcion][1])

    # 2. El tamaño de la capa oculta
    ocultas = NEURONAS_OCULTAS[elegir("Neuronas en la capa oculta:",
                                      [f"{n} neuronas" for n in NEURONAS_OCULTAS], inicial=1)]
    usar_parametros(CAPAS=[config.CANTIDAD_ENTRADAS, ocultas, 2])

    # 3. Semilla, población, cruce y mutación
    print("Presioná Enter para usar los valores sugeridos.")
    return zona, {"semilla": pedir_entero("Semilla", 0, 0),
                  "tamano_poblacion": pedir_entero("Autos por generación", 20, 3),
                  "tipo_cruce": TIPOS_CRUCE[elegir("Cruce:", TIPOS_CRUCE)],
                  "tipo_mutacion": TIPOS_MUTACION[elegir("Mutación:", TIPOS_MUTACION)],
                  "continuar_desde": None}


# -------------------------------------------------------------------------
# Pregunta qué entrenamiento continuar y en qué mapa, y usa sus mismos parámetros.
# Devuelve: la zona y las opciones para entrenar(), o (None, None) si no hay ninguno.
# -------------------------------------------------------------------------
def pedir_continuacion():
    red = elegir_entrenamiento_guardado()
    if red is None:
        return None, None

    zona_anterior = zona_de(red["datos"])
    zona = elegir_zona(zona_anterior)
    parametros = red["datos"]["parametros"]
    volver_a_config()
    usar_parametros_de(red["identificador"])

    print("Se retoma la población completa en un guardado nuevo. El original no cambia.")
    if zona != zona_anterior:
        print("Mapa nuevo: se conserva la población y se inicia un historial para este mapa.")
    return zona, {"semilla": parametros["semilla"],
                  "tamano_poblacion": parametros["tamano_poblacion"],
                  "tipo_cruce": parametros["tipo_cruce"],
                  "tipo_mutacion": parametros["tipo_mutacion"],
                  "continuar_desde": red["identificador"]}


# -------------------------------------------------------------------------
# Opciones 2 y 3: entrenar desde cero o continuar uno guardado.
# Las dos entrenan igual; con ventana, además se ve a toda la población manejar.
# -------------------------------------------------------------------------
def entrenar_red(con_ventana):
    if elegir("Entrenamiento:", ["Desde cero", "Continuar uno guardado"]) == 0:
        zona, opciones = pedir_entrenamiento_nuevo()
    else:
        zona, opciones = pedir_continuacion()
        if zona is None:
            return

    generaciones = pedir_generaciones()
    mapa = santa_fe(zona)
    print("Se entrena en:", mapa.nombre)
    print("Los resultados se guardan en", donde_se_guarda())
    print("Para detener: Esc en la ventana." if con_ventana else "Para detener: Ctrl+C.")

    if con_ventana:
        from conduccion.visualizacion.simulador_pygame import ver_entrenamiento
        _, historia = ver_entrenamiento(mapa, generaciones=generaciones, **opciones)
    else:
        from conduccion.aprendizaje.entrenamiento import entrenar
        _, historia = entrenar(mapa, generaciones=generaciones, **opciones)

    if historia:
        print(f"Listo: {len(historia)} generaciones. Elegí 1 en el menú para verla manejar.")
    else:
        print("No se completó ninguna generación; no se guardó una red nueva.")


# -------------------------------------------------------------------------
# El menú principal. Se repite hasta elegir Salir.
# -------------------------------------------------------------------------
def main():
    while True:
        print("\n=== Conducción autónoma por neuroevolución ===")
        print("1) Ver manejar una red entrenada (ventana)")
        print("2) Entrenar viendo a toda la población (ventana)")
        print("3) Entrenar rápido (sin ventana)")
        print("4) Salir")

        try:
            opcion = input("Elegí una opción: ").strip()
            if opcion == "1":
                ver_auto()
            elif opcion == "2":
                entrenar_red(con_ventana=True)
            elif opcion == "3":
                entrenar_red(con_ventana=False)
            elif opcion == "4":
                return
            else:
                print("Elegí un número del 1 al 4.")
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo del menú.")
            return
        except ValueError as error:
            print("No se pudo continuar:", error)


if __name__ == "__main__":
    main()
