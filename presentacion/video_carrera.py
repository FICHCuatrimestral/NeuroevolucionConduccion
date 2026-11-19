"""Video de las cinco redes de los operadores manejando la misma ruta del centro, a la vez.

Uso, parado en la carpeta del proyecto (hacen falta las corridas de resultados/comparaciones/):
    python presentacion/video_carrera.py

Es la comparación de la tabla de operadores, pero vista: las cinco redes en el mismo mapa
(el centro, donde se midió la tabla) y en la misma ruta, cada una de un color. Al costado, el
orden en que llegan y en cuántas de las 20 rutas del centro llega cada una.

La ruta es la RUTA de las 20 fijas del centro (826 m, 5 giros): la más compleja en la que llegan
las cinco, y en el mismo orden que su tiempo medio en la tabla. Es un ejemplo; los números de la comparación están en la tabla.
Sale img/carrera.mp4. Hace falta ffmpeg (por ejemplo, el del paquete imageio-ffmpeg).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon

# La carpeta del proyecto, para poder importar el paquete conduccion y las figuras de la presentación
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conduccion import config
from conduccion.aprendizaje.entrenamiento import rutas_de_referencia
from conduccion.mapa.mapa import santa_fe
from figuras_presentacion import SALIDA, dibujar_calles, ffmpeg, manejar

CARPETA = RAIZ / "resultados" / "comparaciones"

# Cada red: su carpeta, su nombre en el video y su color (los mismos de la tabla)
CORRIDAS = [("neurona_gaussiana_agarre", "Cruce por neurona", "#2a78d6"),
            ("pesos_gaussiana_agarre", "Cruce por pesos", "#1baf7a"),
            ("blx_gaussiana_agarre", "Cruce BLX-α", "#8e5bd0"),
            ("ninguna_gaussiana_agarre", "Sin cruce", "#b8960a"),
            ("neurona_neurona_agarre", "Mutación por neurona", "#d6438a")]

ZONA = "centro"
RUTA = 7

# Cuántas veces más rápido que el tiempo real va el video
VELOCIDAD_VIDEO = 6

# Los autos se dibujan más grandes que en la realidad, para que se vean. Solo cambia el dibujo.
AGRANDAR = 5.5

TINTA, GRIS = "#222222", "#8a8a85"


# -------------------------------------------------------------------------
# La red final de una corrida y cómo le fue en las 20 rutas del centro.
# -------------------------------------------------------------------------
def red_de(corrida):
    carpeta = CARPETA / corrida / "semilla0"
    evaluacion = json.loads((carpeta / "evaluacion.json").read_text(encoding="utf-8"))
    llegadas = sum(fila["motivo"] == "llegada" for fila in evaluacion["rutas"] if fila["zona"] == ZONA)
    return str(carpeta / evaluacion["red"]), llegadas


def agrandar(esquinas):
    esquinas = np.asarray(esquinas, dtype=float)
    centro = esquinas.mean(axis=0)
    return centro + AGRANDAR * (esquinas - centro)


# -------------------------------------------------------------------------
# Arma el video: el mapa a la izquierda, con la ruta y los cinco autos; a la
# derecha, cada red con su color, y cuando llega, su puesto y su tiempo.
#
# Recibe: el mapa, la ruta y una lista de (nombre, color, llegadas en el centro, pasos, motivo).
# -------------------------------------------------------------------------
def video(mapa, ruta, redes):
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg()
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 0.58, 1.0))
    panel = fig.add_axes((0.58, 0.0, 0.42, 1.0))
    panel.axis("off")

    # El mapa, encuadrado en la ruta
    dibujar_calles(ax, mapa)
    ax.plot(*ruta.carril.xy, color=GRIS, lw=1.5, ls="--", zorder=1)
    xs, ys = ruta.carril.xy
    ax.plot(xs[0], ys[0], "o", color=TINTA, ms=10, zorder=2)
    ax.plot(xs[-1], ys[-1], "s", color=TINTA, ms=10, zorder=2)
    x0, y0, x1, y1 = ruta.carril.bounds
    margen = 40
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ancho, alto = x1 - x0 + 2 * margen, y1 - y0 + 2 * margen
    forma = (0.58 * 12.8) / 7.2
    ancho, alto = max(ancho, alto * forma), max(alto, ancho / forma)
    ax.set_xlim(cx - ancho / 2, cx + ancho / 2)
    ax.set_ylim(cy - alto / 2, cy + alto / 2)
    ax.set_aspect("equal")
    ax.axis("off")

    # El panel de la derecha
    panel.text(0.04, 0.93, "La misma ruta del centro", fontsize=24, fontweight="bold", va="top")
    reloj = panel.text(0.04, 0.83, "", fontsize=18, va="top", color=GRIS)
    panel.text(0.67, 0.72, "llega en el\ncentro", fontsize=12, va="bottom", ha="center", color=GRIS)
    panel.text(0.88, 0.72, "en esta\nruta", fontsize=12, va="bottom", ha="center", color=GRIS)
    puestos = []
    for n, (nombre, color, llegadas, _, _) in enumerate(redes):
        y = 0.64 - 0.11 * n
        panel.text(0.04, y, "■ " + nombre, fontsize=14, color=color, fontweight="bold", va="center")
        panel.text(0.67, y, f"{llegadas} / 20", fontsize=14, color=TINTA, va="center", ha="center")
        puestos.append(panel.text(0.88, y, "", fontsize=14, color=TINTA, va="center", ha="center", fontweight="bold"))
    panel.text(0.04, 0.06, f"Video a {VELOCIDAD_VIDEO}× la velocidad real.\nAutos dibujados más grandes que en la realidad.",
               fontsize=11, color=GRIS, va="bottom")

    # Sin rastros: van todos por la misma línea y se taparían
    autos = []
    for nombre, color, _, pasos, _ in redes:
        auto = Polygon(agrandar(pasos[0]["esquinas"]), fc=color, ec=TINTA, lw=1, zorder=5)
        ax.add_patch(auto)
        autos.append(auto)

    # El orden de llegada, para escribir el puesto cuando cada una termina
    finales = [len(pasos) for _, _, _, pasos, _ in redes]
    orden = sorted(range(len(redes)), key=lambda j: finales[j])
    puesto_de = {j: p + 1 for p, j in enumerate(orden)}

    cuadros = list(range(0, max(finales), 2))
    cuadros += [max(finales) - 1] * 40

    def actualizar(i):
        reloj.set_text(f"tiempo: {(i + 1) * config.DT:.0f} s")
        for j, auto in enumerate(autos):
            pasos = redes[j][3]
            k = min(i, len(pasos) - 1)
            auto.set_xy(agrandar(pasos[k]["esquinas"]))
            # El que termina sale del mapa, para no tapar a los que vienen; su puesto queda en el panel
            auto.set_visible(i < len(pasos) - 1)
            if i >= len(pasos) - 1:
                if redes[j][4] == "llegada":
                    puestos[j].set_text(f"{puesto_de[j]}º · {len(pasos) * config.DT:.0f} s")
                else:
                    puestos[j].set_text("no llega")
        return autos

    animacion = FuncAnimation(fig, actualizar, frames=cuadros, blit=False)
    animacion.save(SALIDA / "carrera.mp4", writer=FFMpegWriter(fps=VELOCIDAD_VIDEO / (2 * config.DT), bitrate=1800,
                   extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print("-> carrera.mp4", len(cuadros), "cuadros")


if __name__ == "__main__":
    mapa = santa_fe(ZONA)
    ruta = rutas_de_referencia(mapa, 0)[RUTA]
    redes = []
    for corrida, nombre, color in CORRIDAS:
        red, llegadas = red_de(corrida)
        pasos, motivo = manejar(red, mapa, ruta)
        print(f"{nombre}: {motivo}, {len(pasos) * config.DT:.1f} s; llega en {llegadas} de 20 del centro")
        redes.append((nombre, color, llegadas, pasos, motivo))
    video(mapa, ruta, redes)
