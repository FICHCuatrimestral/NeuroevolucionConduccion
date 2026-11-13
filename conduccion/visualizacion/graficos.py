"""Gráficos con matplotlib para los notebooks."""

import math

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon

from ..aprendizaje.evaluacion import TIPOS_DE_RUTA
from ..auto.sensores import Sensores
from .. import config

# Colores del mapa

# Fuera de la calle
VERDE = "#a8c99a"
# Calle
GRIS = "#6b6b6b"
# Carril de la ruta
CELESTE = "#2a9fd6"
# Puntos de control
AMARILLO = "#f2c12e"
# Choque o destino
ROJO = "#d03b3b"

# Colores de las series de los gráficos, en orden de uso
AZUL = "#2a78d6"
NARANJA = "#eb6834"
AGUA = "#1baf7a"
OCRE = "#eda100"
SERIES = [AZUL, NARANJA, AGUA, OCRE]
TINTA = "#0b0b0b"
GRIS_CLARO = "#c3c2b7"

# Escala de rojo (negativo) a azul (positivo), para los pesos de la red
DIVERGENTE = LinearSegmentedColormap.from_list("rojo_azul", ["#e34948", "#f0efec", AZUL])

# El nombre de cada una de las 16 entradas de la red, en orden
NOMBRES_ENTRADAS = ([f"rayo {round(math.degrees(a))}°" for a in config.ANGULOS_RAYOS]
                    + ["velocidad", "brújula seno", "brújula coseno", "distancia al punto", "desvío del carril",
                       "carril a 3 m seno", "carril a 3 m coseno", "carril a 6 m seno", "carril a 6 m coseno"])


# -------------------------------------------------------------------------
# El mismo estilo para todos los gráficos de los notebooks.
# -------------------------------------------------------------------------
def aplicar_estilo():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRIS_CLARO,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e1e0d9", "grid.linewidth": 0.7, "axes.axisbelow": True,
                         "lines.linewidth": 2, "font.size": 10, "axes.titlesize": 11,
                         "axes.titlelocation": "left", "legend.frameon": False})


# =========================================================================
# MAPA, RUTA Y AUTO
# =========================================================================

# -------------------------------------------------------------------------
# Dibuja la matriz del mapa: gris = calle, verde = fuera de la calle.
# Los puntos blancos son las esquinas.
#
# Los ejes no llevan números: las coordenadas son UTM, en millones de metros, y no aportan.
# -------------------------------------------------------------------------
def dibujar_mapa(ax, mapa):
    filas, columnas = mapa.matriz.shape
    extension = [mapa.x0, mapa.x0 + columnas * config.RESOLUCION_MAPA, mapa.y0, mapa.y0 + filas * config.RESOLUCION_MAPA]
    ax.imshow(mapa.matriz, origin="lower", extent=extension, cmap=ListedColormap([VERDE, GRIS]))
    for nodo in mapa.grafo.nodes:
        x, y = mapa.posicion(nodo)
        ax.plot(x, y, "o", color="white", markersize=4)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


# -------------------------------------------------------------------------
# Dibuja el carril de la ruta, sus puntos de control, la salida A y el destino B.
# Los primeros `alcanzados` puntos van en verde.
# -------------------------------------------------------------------------
def dibujar_ruta(ax, ruta, alcanzados=0):
    xs, ys = ruta.carril.xy
    ax.plot(xs, ys, color=CELESTE, linewidth=1)
    for i, (x, y) in enumerate(ruta.puntos):
        ax.plot(x, y, "o", color="limegreen" if i < alcanzados else AMARILLO, markersize=4)
    x0, y0, _ = ruta.inicio
    ax.plot(x0, y0, "s", color="white", markersize=7, label="A")
    ax.plot(ruta.puntos[-1][0], ruta.puntos[-1][1], "*", color=ROJO, markersize=14, label="B")


# -------------------------------------------------------------------------
# Dibuja el rectángulo del auto.
# -------------------------------------------------------------------------
def dibujar_auto(ax, auto, color=AZUL):
    ax.add_patch(Polygon(auto.esquinas(), closed=True, color=color))


# -------------------------------------------------------------------------
# Dibuja los 7 rayos, desde el centro del auto hasta donde tocan el borde de la calle.
# -------------------------------------------------------------------------
def dibujar_rayos(ax, auto, mapa):
    for angulo_relativo, distancia in zip(config.ANGULOS_RAYOS, Sensores(mapa).medir_rayos(auto)):
        angulo = auto.angulo + angulo_relativo
        x = auto.x + math.cos(angulo) * distancia
        y = auto.y + math.sin(angulo) * distancia
        ax.plot([auto.x, x], [auto.y, y], color="gold", linewidth=1)
        ax.plot(x, y, "o", color="orange", markersize=3)


# -------------------------------------------------------------------------
# Dibuja el recorrido de un episodio. Si chocó, marca el choque con una cruz.
# -------------------------------------------------------------------------
def dibujar_recorrido(ax, resultado, color=AZUL, etiqueta=None):
    xs = [x for x, y in resultado.recorrido]
    ys = [y for x, y in resultado.recorrido]
    ax.plot(xs, ys, color=color, linewidth=2, label=etiqueta)
    if resultado.motivo == "choque":
        ax.plot(xs[-1], ys[-1], "x", color=ROJO, markersize=12, markeredgewidth=3)


# -------------------------------------------------------------------------
# Un mapa por tipo de ruta, con todos los recorridos de ese tipo.
# Azul: llega. Naranja: no llega. La cruz marca dónde terminó.
#
# Recibe: el mapa y las corridas de evaluacion.manejar().
# -------------------------------------------------------------------------
def dibujar_recorridos_por_tipo(mapa, corridas):
    fig, ejes = plt.subplots(1, len(TIPOS_DE_RUTA), figsize=(15, 4), sharex=True, sharey=True)
    for ax, tipo in zip(ejes, TIPOS_DE_RUTA):
        dibujar_mapa(ax, mapa)
        grupo = [corrida["resultado"] for corrida in corridas if corrida["tipo"] == tipo]
        for resultado in grupo:
            xs, ys = zip(*resultado.recorrido)
            llego = resultado.motivo == "llegada"
            ax.plot(xs, ys, color=AZUL if llego else NARANJA, linewidth=1, alpha=0.8)
            if not llego:
                ax.plot(xs[-1], ys[-1], "x", color=TINTA, markersize=7, markeredgewidth=2)
        llegan = sum(resultado.motivo == "llegada" for resultado in grupo)
        ax.set_title(f"{tipo}: llegan {llegan} de {len(grupo)}")
    fig.suptitle("Azul: llega. Naranja: no llega. La cruz marca dónde terminó", x=0.01, ha="left")
    plt.show()


# =========================================================================
# ENTRENAMIENTO
# =========================================================================

# -------------------------------------------------------------------------
# Las tres curvas de un entrenamiento, por generación:
#   - la nota del mejor individuo en las rutas sorteadas de esa generación
#   - la nota media de la población
#   - la nota de la mejor red encontrada, en las rutas de referencia
# -------------------------------------------------------------------------
def graficar_entrenamiento(historia, titulo="Entrenamiento"):
    generaciones = [fila["generacion"] for fila in historia]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(generaciones, [fila["mejor_nota"] for fila in historia], color=GRIS_CLARO, linewidth=1,
            label="mejor (rutas de la generación)")
    ax.plot(generaciones, [fila["nota_media"] for fila in historia], color=AZUL, label="media de la población")
    ax.plot(generaciones, [fila["mejor_referencia"] for fila in historia], color=NARANJA,
            label="mejor red (rutas de referencia)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generación")
    ax.set_ylabel("nota")
    ax.set_title(titulo)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.show()


# -------------------------------------------------------------------------
# La nota de referencia de varios entrenamientos, en el mismo gráfico.
#
# Recibe: una lista de pares (identificador, historia).
# -------------------------------------------------------------------------
def comparar_entrenamientos(corridas):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for (identificador, historia), color in zip(corridas, SERIES * 3):
        ax.plot([fila["generacion"] for fila in historia], [fila["mejor_referencia"] for fila in historia],
                color=color, label=identificador[:19])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generación")
    ax.set_ylabel("mejor referencia")
    ax.set_title("La nota de la mejor red, en sus rutas de referencia")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    plt.show()
