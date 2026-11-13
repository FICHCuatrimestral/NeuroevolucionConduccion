"""Corre las comparaciones del informe (ver conduccion/aprendizaje/comparaciones.py).

Uso, parado en la carpeta del proyecto:
    python scripts/correr_comparaciones.py uno_por_vez                    las 6 corridas a la vez
    python scripts/correr_comparaciones.py uno_por_vez --procesos 3       tres corridas a la vez
    python scripts/correr_comparaciones.py uno_por_vez --semillas 0 1 2   solo esas semillas
    python scripts/correr_comparaciones.py todo --estimar                 cuánto tarda, sin correr nada

Cada corrida recibe un número al azar de 4 cifras antes de empezar. Al terminar cada corrida se
actualizan resultados/comparaciones/indice.json e indice.xlsx.

Las corridas van semilla por semilla: primero todas las combinaciones con la semilla 0,
después con la 1, etc. Así, si hay que cortar, quedan semillas completas.

Se puede cortar con Ctrl+C en cualquier momento. Al volver a correr lo mismo, se saltean las
corridas terminadas y las cortadas siguen desde la última generación guardada.

Para repartir el trabajo entre varias computadoras: cada una corre otras semillas, y después se
juntan las carpetas resultados/comparaciones/ de todas.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# La carpeta del proyecto, para poder importar el paquete conduccion
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conduccion.aprendizaje.comparaciones import (SEMILLAS, actualizar_indice, combinaciones, corrida_terminada,
                                                  correr_corrida, nombre_de, numerar, segundos_como_mucho)

# Cuántas corridas van a la vez si no se dice otra cosa. Con 6, las corridas de
# "uno_por_vez" van todas juntas y terminan más o menos a la vez.
PROCESOS = 6


# -------------------------------------------------------------------------
# Las corridas que faltan, semilla por semilla.
# Devuelve: una lista de (combinación, semilla).
# -------------------------------------------------------------------------
def corridas_pendientes(forma, semillas):
    return [(combinacion, semilla) for semilla in semillas for combinacion in combinaciones(forma)
            if not corrida_terminada(combinacion, semilla)]


# -------------------------------------------------------------------------
# Muestra cada corrida pendiente con su número, y cuántas horas tardan como
# mucho, si cada etapa usa todas sus generaciones. Cada corrida usa un solo
# núcleo: con más procesos que núcleos, se reparten los núcleos.
# -------------------------------------------------------------------------
def estimar(pendientes, procesos):
    horas = segundos_como_mucho() / 3600
    for combinacion, semilla in pendientes:
        print(f"  corrida {numerar(combinacion, semilla)}: {nombre_de(combinacion)}/semilla{semilla}")
    nucleos = min(procesos, os.cpu_count() or 1)
    total = len(pendientes) * horas
    print(f"{len(pendientes)} corridas de hasta {horas:.1f} h cada una en un núcleo.")
    print(f"Con {procesos} procesos y {nucleos} núcleos: hasta unas {max(total / nucleos, horas):.1f} h.")


# -------------------------------------------------------------------------
# Corre las corridas pendientes, `procesos` a la vez. Muestra cada una al
# terminar y actualiza el índice.
# -------------------------------------------------------------------------
def correr(pendientes, procesos):
    inicio = time.time()
    with ProcessPoolExecutor(procesos) as pool:
        tareas = {pool.submit(correr_corrida, *corrida): corrida for corrida in pendientes}
        for hechas, tarea in enumerate(as_completed(tareas), start=1):
            combinacion, semilla = tareas[tarea]
            try:
                estado = tarea.result()
            except Exception as error:
                estado = f"ERROR: {error}"
            aviso = actualizar_indice()
            horas = (time.time() - inicio) / 3600
            print(f"[{hechas}/{len(pendientes)}  {horas:.1f} h]  corrida {numerar(combinacion, semilla)} "
                  f"({nombre_de(combinacion)}/semilla{semilla}): {estado}", flush=True)
            if aviso:
                print(f"  {aviso}", flush=True)


# -------------------------------------------------------------------------
# Lee los argumentos y corre o estima.
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Corre las comparaciones del informe.")
    parser.add_argument("forma", choices=["uno_por_vez", "todo"])
    parser.add_argument("--semillas", nargs="+", type=int, default=SEMILLAS)
    parser.add_argument("--procesos", type=int, default=PROCESOS)
    parser.add_argument("--estimar", action="store_true", help="solo mostrar cuánto tarda")
    argumentos = parser.parse_args()

    pendientes = corridas_pendientes(argumentos.forma, argumentos.semillas)
    if not pendientes:
        print("No queda nada por correr.")
        return
    estimar(pendientes, argumentos.procesos)
    if argumentos.estimar:
        return

    print("El avance de cada corrida queda en resultados/comparaciones/<combinación>/semilla<n>/registro.txt",
          flush=True)
    try:
        correr(pendientes, argumentos.procesos)
    except KeyboardInterrupt:
        print("\nDetenido. Para seguir, corré lo mismo: lo terminado se saltea y lo cortado sigue.")


if __name__ == "__main__":
    main()
