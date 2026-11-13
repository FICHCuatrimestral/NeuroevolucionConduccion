"""Videos y figura de velocidad de la presentación, con las redes finales de las comparaciones.

Uso, parado en la carpeta del proyecto (hacen falta las corridas de resultados/comparaciones/):
    python presentacion/figuras_presentacion.py

La red base (con límite de agarre) y la red sin límite manejan la misma ruta del centro, sin ruido
en los sensores. Es la misma ruta de la figura de velocidad del informe (informe/figuras/figuras_resultados.py):
  - manejo_sin.mp4 y manejo_con.mp4: el auto visto de arriba, con su velocidad y su aceleración lateral
  - velocidad.png: la velocidad de las dos a lo largo de la ruta

Hace falta ffmpeg: el del paquete imageio-ffmpeg, si está instalado, o el que esté en el PATH.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon

# La carpeta del proyecto, para poder importar el paquete conduccion
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
SALIDA = Path(__file__).resolve().parent / "img"
SALIDA.mkdir(exist_ok=True)

from conduccion import config
from conduccion.parametros import usar_parametros_de
from conduccion.aprendizaje.guardado import cargar
from conduccion.aprendizaje.perceptron_multicapa import Perceptron
from conduccion.aprendizaje.entrenamiento import rutas_de_referencia
from conduccion.mapa.mapa import santa_fe
from conduccion.mapa.ruta import distancias_a_giros
from conduccion.simulacion.simulacion import Episodio

CARPETA = RAIZ / "resultados" / "comparaciones"

# Qué redes se muestran: la carpeta de su corrida y su título
REDES = {"sin": ("neurona_gaussiana_sin-agarre", "Sin límite de agarre"),
         "con": ("neurona_gaussiana_agarre", "Con límite de 6 m/s²")}
ZONA = "centro"

# Cuál de las 20 rutas fijas del centro se maneja: la de más giros a la que llegan las dos redes
RUTA = 7

# Cuántas veces más rápido que el tiempo real va el video
VELOCIDAD_VIDEO = 4

# Metros que se ven a cada lado del auto en el video
VENTANA = 45.0

AZ, NA, TINTA, GRIS, CALLE = "#2a78d6", "#eb6834", "#222222", "#8a8a85", "#dcdcd8"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 1.0, "lines.linewidth": 2.4})


# -------------------------------------------------------------------------
# La carpeta de la red final de una corrida de las comparaciones.
# -------------------------------------------------------------------------
def red_final(corrida):
    carpeta = CARPETA / corrida / "semilla0"
    evaluacion = json.loads((carpeta / "evaluacion.json").read_text(encoding="utf-8"))
    return str(carpeta / evaluacion["red"])


# -------------------------------------------------------------------------
# Dónde está ffmpeg: el del paquete imageio-ffmpeg si está instalado; si no,
# el que esté en el PATH.
# -------------------------------------------------------------------------
def ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


# -------------------------------------------------------------------------
# La red guardada maneja la ruta, con la física con la que se entrenó.
#
# Devuelve: una lista con un diccionario por paso (posición, esquinas del auto,
#           velocidad, aceleración lateral y metros recorridos) y el motivo del final.
# -------------------------------------------------------------------------
def manejar(identificador, mapa, ruta):
    usar_parametros_de(identificador)
    cromosoma, _, _ = cargar(identificador)
    red = Perceptron(config.CAPAS, cromosoma=cromosoma)
    episodio = Episodio(mapa, ruta)
    pasos = []
    recorrido = 0.0
    while not episodio.terminado():
        x, y = episodio.auto.x, episodio.auto.y
        episodio.paso_con_red(red)
        auto = episodio.auto
        recorrido += math.dist((x, y), (auto.x, auto.y))

        # Aceleración lateral del modelo de bicicleta: v² · sin(beta) / l_r
        beta = math.atan(config.L_R / (config.L_F + config.L_R) * math.tan(auto.volante))
        lateral = auto.velocidad ** 2 * abs(math.sin(beta)) / config.L_R
        pasos.append({"x": auto.x, "y": auto.y, "esquinas": auto.esquinas(), "v": auto.velocidad,
                      "lateral": lateral, "recorrido": recorrido})
    return pasos, episodio.motivo


# -------------------------------------------------------------------------
# Pinta las calles del mapa. Los huecos del polígono (las manzanas) van en blanco.
# -------------------------------------------------------------------------
def dibujar_calles(ax, mapa):
    partes = mapa.calles.geoms if hasattr(mapa.calles, "geoms") else [mapa.calles]
    for parte in partes:
        ax.fill(*parte.exterior.xy, color=CALLE, lw=0, zorder=0)
        for hueco in parte.interiors:
            ax.fill(*hueco.xy, color="white", lw=0, zorder=0)


# -------------------------------------------------------------------------
# Arma el video de una red: la cámara sigue al auto, que deja un rastro, y
# abajo se ven su velocidad y su aceleración lateral.
#
# Recibe: el nombre del archivo, el título, el color, los pasos de manejar(),
#         el mapa y la ruta.
# -------------------------------------------------------------------------
def video(nombre, titulo, color, pasos, mapa, ruta):
    fig = plt.figure(figsize=(6.4, 6.4), dpi=110)
    ax = fig.add_axes((0, 0, 1, 1))
    dibujar_calles(ax, mapa)
    ax.plot(*ruta.carril.xy, color=GRIS, lw=1.2, ls="--", zorder=1)
    rastro, = ax.plot([], [], color=color, lw=3, zorder=2)
    auto = Polygon(pasos[0]["esquinas"], fc=color, ec=TINTA, lw=1, zorder=3)
    ax.add_patch(auto)
    caja = dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9)
    ax.text(0.04, 0.95, titulo, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top", bbox=caja)
    texto = ax.text(0.04, 0.05, "", transform=ax.transAxes, fontsize=20, va="bottom", bbox=caja)
    ax.set_aspect("equal")
    ax.axis("off")

    # Se muestra un paso de cada dos. Al final, el auto se queda quieto un momento.
    cuadros = list(range(0, len(pasos), 2)) + [len(pasos) - 1] * 10

    # ---------------------------------------------------------------------
    # Dibuja el cuadro i: mueve el auto, alarga el rastro, centra la cámara
    # y escribe la velocidad y la aceleración lateral.
    # ---------------------------------------------------------------------
    def actualizar(i):
        p = pasos[i]
        auto.set_xy(p["esquinas"])
        rastro.set_data([q["x"] for q in pasos[:i + 1]], [q["y"] for q in pasos[:i + 1]])
        ax.set_xlim(p["x"] - VENTANA, p["x"] + VENTANA)
        ax.set_ylim(p["y"] - VENTANA, p["y"] + VENTANA)
        texto.set_text(f"{3.6 * p['v']:.0f} km/h      lateral {p['lateral']:.0f} m/s²".replace(".", ","))
        return auto, rastro, texto

    animacion = FuncAnimation(fig, actualizar, frames=cuadros, blit=False)
    cuadros_por_segundo = VELOCIDAD_VIDEO / (2 * config.DT)
    animacion.save(SALIDA / f"manejo_{nombre}.mp4", writer=FFMpegWriter(fps=cuadros_por_segundo, bitrate=900,
                   extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print("->", f"manejo_{nombre}.mp4", len(cuadros), "cuadros")


# -------------------------------------------------------------------------
# La velocidad (arriba) y la aceleración lateral (abajo) de cada red a lo largo
# de la ruta, con los giros marcados, el tope de 6 m/s² y el tiempo total al
# final de cada línea de velocidad.
#
# Recibe: una lista de (título, color, pasos) y la ruta.
# -------------------------------------------------------------------------
def velocidad(corridas, ruta):
    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)
    for d in distancias_a_giros(ruta.carril):
        for ax in (arriba, abajo):
            ax.axvspan(d - 12, d + 12, color="#f0efe9", zorder=0)
        arriba.text(d, 47, "giro", ha="center", va="top", color=GRIS, fontsize=13)
    for (titulo, color, pasos) in corridas:
        metros = [p["recorrido"] for p in pasos]
        arriba.plot(metros, [3.6 * p["v"] for p in pasos], color=color, label=titulo)
        arriba.text(metros[-1] + 6, 3.6 * pasos[-1]["v"], f"{0.1 * len(pasos):.0f} s".replace(".", ","),
                    color=color, va="center", fontsize=13)
        abajo.plot(metros, [p["lateral"] for p in pasos], color=color, lw=1.6)
    abajo.axhline(6, color=TINTA, lw=1, ls="--")
    abajo.text(ruta.largo + 58, 6.8, "6 m/s²", ha="right", va="bottom", fontsize=13)
    arriba.set_ylabel("velocidad (km/h)")
    arriba.set_ylim(0, 48)
    abajo.set_ylabel("aceleración\nlateral (m/s²)")
    abajo.set_xlabel("metros recorridos")
    abajo.set_xlim(0, ruta.largo + 60)
    arriba.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(SALIDA / "velocidad.png", dpi=160, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("-> velocidad.png")


if __name__ == "__main__":
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg()
    mapa = santa_fe(ZONA)
    ruta = rutas_de_referencia(mapa, 0)[RUTA]
    corridas = []
    for nombre, color in (("sin", NA), ("con", AZ)):
        corrida, titulo = REDES[nombre]
        pasos, motivo = manejar(red_final(corrida), mapa, ruta)
        print(titulo, "|", motivo, "|", round(0.1 * len(pasos), 1), "s | velocidad mínima tras arrancar",
              round(3.6 * min(p["v"] for p in pasos[30:]), 1), "km/h | lateral máxima",
              round(max(p["lateral"] for p in pasos), 1), "m/s²")
        video(nombre, titulo, color, pasos, mapa, ruta)
        corridas.append((titulo, color, pasos))
    velocidad(corridas, ruta)
