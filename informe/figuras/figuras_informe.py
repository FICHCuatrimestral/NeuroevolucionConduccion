"""Figuras del informe, al tamaño final de la plantilla (ancho de texto 5,5 pulgadas).

Uso, parado en la carpeta del proyecto:
    python informe/figuras/figuras_informe.py                 todas
    python informe/figuras/figuras_informe.py mapas auto      solo esas

Cada figura se guarda en PDF (para el informe) y en PNG (para mirarla rápido).
Las figuras de métodos no dependen de ninguna red entrenada: usan redes al azar o la física del auto.
"""
import itertools
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import shapely

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, FancyArrowPatch, Patch, Polygon, Rectangle

# La carpeta del proyecto, para poder importar el paquete conduccion
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from conduccion import config
from conduccion.aprendizaje.genetico import cruzar, mutar, poblacion_inicial
from conduccion.aprendizaje.perceptron_multicapa import Perceptron
from conduccion.auto.auto import Auto
from conduccion.auto.sensores import Sensores
from conduccion.mapa.mapa import manzana, santa_fe
from conduccion.mapa.ruta import calcular_ruta, ruta_al_azar, vueltas_a_la_manzana

SALIDA = Path(__file__).resolve().parent

# Ancho de texto de la plantilla, en pulgadas
ANCHO = 5.5

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.formatter.use_locale": False, "pdf.fonttype": 42,
})

# Paleta: la misma en todas las figuras. Azul = el auto o lo principal, naranja = el segundo elemento.
AZ, NA, VE, AM, MA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TINTA, GRIS, GRIS_CLARO = "#222222", "#8a8a85", "#c8c8c4"
CALLE, FONDO = "#dcdcd8", "#ffffff"

# Fondo de las matrices de pesos: casilla sin cambio
SIN_CAMBIO = "#f2f2f0"


# =========================================================================
# AYUDAS PARA DIBUJAR
# =========================================================================

# -------------------------------------------------------------------------
# Guarda la figura en PDF y en PNG, con el nombre dado, y la cierra.
# -------------------------------------------------------------------------
def guardar(fig, nombre):
    for extension in ("pdf", "png"):
        fig.savefig(SALIDA / f"{nombre}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", nombre)


# -------------------------------------------------------------------------
# Dibuja una flecha de p a q.
# -------------------------------------------------------------------------
def flecha(ax, p, q, color=TINTA, lw=1.0, ms=7, ls="-", z=4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, linestyle=ls, zorder=z))


# -------------------------------------------------------------------------
# Escribe la letra del panel, por ejemplo "a)", arriba a la izquierda.
# -------------------------------------------------------------------------
def letra(ax, texto, x=-0.02, y=1.02):
    ax.text(x, y, texto, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


# -------------------------------------------------------------------------
# Pinta las calles del mapa. Los huecos del polígono (las manzanas) van en blanco.
# -------------------------------------------------------------------------
def dibujar_calles(ax, mapa):
    partes = mapa.calles.geoms if hasattr(mapa.calles, "geoms") else [mapa.calles]
    for parte in partes:
        ax.fill(*parte.exterior.xy, color=CALLE, lw=0, zorder=0)
        for hueco in parte.interiors:
            ax.fill(*hueco.xy, color=FONDO, lw=0, zorder=0)


# -------------------------------------------------------------------------
# Deja el panel sin ejes ni bordes, con la misma escala en x y en y.
# -------------------------------------------------------------------------
def sin_ejes(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for borde in ax.spines.values():
        borde.set_visible(False)


# -------------------------------------------------------------------------
# Dibuja una ruta: el carril, los puntos de control y la salida.
# -------------------------------------------------------------------------
def dibujar_ruta(ax, ruta, color=AZ, ms=2.2, lw=1.0):
    ax.plot(*ruta.carril.xy, color=color, lw=lw, zorder=2)
    px, py = zip(*ruta.puntos)
    ax.plot(px, py, "o", color=NA, ms=ms, mec="white", mew=0.3, zorder=3)
    x, y, _ = ruta.inicio
    ax.plot(x, y, "s", color=VE, ms=ms + 1.3, mec="white", mew=0.3, zorder=4)


# -------------------------------------------------------------------------
# Deja visible el borde gris de una matriz dibujada con imshow.
# -------------------------------------------------------------------------
def bordes_grises(ax):
    for borde in ax.spines.values():
        borde.set_visible(True)
        borde.set_linewidth(0.4)
        borde.set_color(GRIS)


# =========================================================================
# MAPAS: la manzana, el 3x3 con una ruta y el centro
# =========================================================================

# -------------------------------------------------------------------------
# a) La manzana, con una vuelta hacia cada lado.
# -------------------------------------------------------------------------
def panel_manzana(ax):
    mapa = manzana()
    izquierda, derecha = vueltas_a_la_manzana(mapa)
    dibujar_calles(ax, mapa)
    ax.plot(*izquierda[0].carril.xy, color=AZ, lw=1.0, zorder=2, label="vuelta a la izquierda")
    ax.plot(*derecha[0].carril.xy, color=NA, lw=1.0, zorder=2, label="vuelta a la derecha")
    xs, ys = zip(*[mapa.posicion(esquina) for esquina in mapa.esquinas_manzana])
    cx, cy = np.mean(xs), np.mean(ys)
    ax.set_xlim(cx - 85, cx + 85)
    ax.set_ylim(cy - 85, cy + 85)
    ax.set_title("Manzana")
    sin_ejes(ax)
    letra(ax, "a)")


# -------------------------------------------------------------------------
# b) El 3x3 con una ruta: el camino de A* por las esquinas, el carril y los puntos.
#    De 60 rutas al azar, se elige la de más esquinas entre las de menos de 380 m.
# -------------------------------------------------------------------------
def panel_3x3(ax):
    mapa = santa_fe("3x3")
    rng = np.random.default_rng(4)
    candidatas = [ruta_al_azar(mapa, rng) for _ in range(60)]
    ruta = max((candidata for candidata in candidatas if candidata.largo < 380), key=lambda c: len(c.esquinas))
    dibujar_calles(ax, mapa)
    for esquina, siguiente in zip(ruta.esquinas, ruta.esquinas[1:]):
        p, q = mapa.posicion(esquina), mapa.posicion(siguiente)
        ax.plot([p[0], q[0]], [p[1], q[1]], color=GRIS, lw=3.5, alpha=0.45, zorder=1, solid_capstyle="round")
    dibujar_ruta(ax, ruta, ms=1.0, lw=1.5)
    ax.set_title(r"3$\times$3 manzanas")
    sin_ejes(ax)
    letra(ax, "b)")
    print("mapas: ruta 3x3", round(ruta.largo), "m,", len(ruta.puntos), "puntos,", len(ruta.esquinas), "esquinas")


# -------------------------------------------------------------------------
# c) El centro, con dos rutas al azar.
# -------------------------------------------------------------------------
def panel_centro(ax):
    mapa = santa_fe("centro")
    rng = np.random.default_rng(1)
    dibujar_calles(ax, mapa)
    for _ in range(2):
        dibujar_ruta(ax, ruta_al_azar(mapa, rng), ms=0.7, lw=1.3)
    ax.set_title("Centro")
    sin_ejes(ax)
    letra(ax, "c)")


# -------------------------------------------------------------------------
# La figura de los mapas, con una leyenda común a los tres paneles.
# -------------------------------------------------------------------------
def mapas():
    fig = plt.figure(figsize=(ANCHO, 2.15))
    ejes = [fig.add_axes(lugar) for lugar in ((0.00, 0.13, 0.31, 0.78), (0.345, 0.13, 0.31, 0.78),
                                              (0.69, 0.13, 0.31, 0.78))]
    panel_manzana(ejes[0])
    panel_3x3(ejes[1])
    panel_centro(ejes[2])
    fig.legend(handles=[Line2D([], [], color=GRIS, lw=2.2, alpha=0.5, label="camino de A*"),
                        Line2D([], [], color=AZ, lw=1.2, label="carril / vuelta a la izquierda"),
                        Line2D([], [], color=NA, lw=1.2, label="vuelta a la derecha"),
                        Line2D([], [], color=NA, marker="o", ls="none", ms=3, label="puntos de control"),
                        Line2D([], [], color=VE, marker="s", ls="none", ms=3.5, label="salida")],
               loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, handlelength=1.4,
               columnspacing=1.0, labelspacing=0.3)
    guardar(fig, "fig_mapas")


# =========================================================================
# AUTO: geometría, trayectorias con y sin agarre, volante máximo
# =========================================================================

# -------------------------------------------------------------------------
# Dibuja una rueda como un trazo grueso, centrada en `centro` y girada `angulo`.
# -------------------------------------------------------------------------
def dibujar_rueda(ax, centro, angulo, color):
    medio = np.array([math.cos(angulo), math.sin(angulo)]) * 0.35
    ax.plot([centro[0] - medio[0], centro[0] + medio[0]], [centro[1] - medio[1], centro[1] + medio[1]],
            color=color, lw=4.5, solid_capstyle="butt")


# -------------------------------------------------------------------------
# a) La geometría del modelo de bicicleta. El auto va horizontal (ψ = 0) para
#    que se lea fácil; con el volante a 35° el centro de giro O queda cerca.
# -------------------------------------------------------------------------
def panel_bicicleta(ax):
    lf, lr = config.L_F, config.L_R
    delta = math.radians(35)
    beta = math.atan(lr / (lf + lr) * math.tan(delta))

    # 1. El eje del auto, con la rueda trasera recta y la delantera girada
    u = np.array([1.0, 0.0])
    n = np.array([0.0, 1.0])
    cm = np.zeros(2)
    delantera = cm + lf * u
    trasera = cm - lr * u
    ax.plot(*zip(trasera, delantera), color=TINTA, lw=1.6)
    dibujar_rueda(ax, trasera, 0, TINTA)
    dibujar_rueda(ax, delantera, delta, NA)

    # 2. El centro de giro O: donde se cruzan las perpendiculares a las dos ruedas
    direccion_delantera = np.array([math.cos(delta), math.sin(delta)])
    normal_delantera = np.array([-direccion_delantera[1], direccion_delantera[0]])
    s = np.linalg.solve(np.array([n, -normal_delantera]).T, delantera - trasera)
    centro_giro = trasera + s[0] * n
    for punto in (trasera, delantera, cm):
        ax.plot(*zip(punto, centro_giro), color=GRIS, lw=0.6, ls=":")
    ax.plot(*centro_giro, "x", color=TINTA, ms=5, mew=1.2)
    ax.text(centro_giro[0] + 0.2, centro_giro[1], "$O$", va="center")
    ax.text(*(0.5 * cm + 0.5 * centro_giro) + np.array([0.15, 0]), "$R$", color=GRIS, va="center")

    # 3. El centro de masa y la velocidad v, que forma el ángulo beta con el auto
    ax.plot(*cm, "o", color=AZ, ms=4.5, zorder=5)
    direccion_velocidad = np.array([math.cos(beta), math.sin(beta)])
    flecha(ax, cm, cm + 2.0 * direccion_velocidad, color=AZ, lw=1.2)
    ax.text(*(cm + 2.1 * direccion_velocidad), "$v$", color=AZ, va="bottom")
    ax.add_patch(Arc(cm, 1.6, 1.6, theta1=0, theta2=math.degrees(beta), color=AZ, lw=0.8))
    ax.text(*(cm + 1.05 * np.array([math.cos(beta / 2), math.sin(beta / 2)])), r"$\beta$",
            color=AZ, va="center", ha="center", fontsize=7)

    # 4. El ángulo del volante delta
    ax.plot(*zip(delantera, delantera + 1.5 * u), color=TINTA, lw=0.5, ls="--")
    ax.plot(*zip(delantera, delantera + 1.5 * direccion_delantera), color=NA, lw=0.5, ls="--")
    ax.add_patch(Arc(delantera, 2.2, 2.2, theta1=0, theta2=math.degrees(delta), color=NA, lw=0.8))
    ax.text(*(delantera + 1.4 * np.array([math.cos(delta / 2), math.sin(delta / 2)])), r"$\delta$",
            color=NA, va="center", ha="center")

    # 5. Las distancias del centro de masa a cada eje
    ax.text(*(0.5 * (cm + delantera) - 0.4 * n), r"$\ell_f$", ha="center", va="center")
    ax.text(*(0.5 * (cm + trasera) - 0.4 * n), r"$\ell_r$", ha="center", va="center")
    ax.text(*(cm - 0.45 * n), "CM", ha="center", va="center", fontsize=6, color=AZ)
    ax.set_aspect("equal")
    ax.axis("off")
    letra(ax, "a)", 0.0, 0.98)


# -------------------------------------------------------------------------
# El recorrido de 5 s de un auto que sale de (0, 0) con el volante fijo y el
# acelerador a fondo. Devuelve: las listas de x y de y.
# -------------------------------------------------------------------------
def trayectoria(grados):
    auto = Auto(0, 0, 0)
    xs, ys = [0], [0]
    for _ in range(50):
        auto.avanzar(math.radians(grados), 2.0)
        xs.append(auto.x)
        ys.append(auto.y)
    return xs, ys


# -------------------------------------------------------------------------
# b) Trayectorias con tres ángulos de volante, sin límite de agarre y con él.
# -------------------------------------------------------------------------
def panel_trayectorias(ax):
    agarre_original = config.ACELERACION_LATERAL_MAXIMA
    for grados, color in zip([5, 15, 35], [AZ, VE, NA]):
        for agarre, estilo in ((None, "-"), (6.0, "--")):
            config.ACELERACION_LATERAL_MAXIMA = agarre
            xs, ys = trayectoria(grados)
            ax.plot(xs, ys, color=color, ls=estilo, lw=1.1, label=f"{grados}°" if agarre is None else None)

    # Se deja el agarre como estaba, para las figuras que siguen
    config.ACELERACION_LATERAL_MAXIMA = agarre_original

    ax.plot([], [], color=TINTA, lw=1.1, label="sin límite")
    ax.plot([], [], color=TINTA, ls="--", lw=1.1, label="con límite")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.6, labelspacing=0.25, borderaxespad=0.1)
    letra(ax, "b)", -0.25, 1.04)


# -------------------------------------------------------------------------
# c) El volante máximo según la velocidad, con la ecuación del agarre del informe.
#    Mientras sin(beta) = a_max · l_r / v² pasa de 1, el auto puede doblar a fondo.
# -------------------------------------------------------------------------
def panel_volante_maximo(ax):
    lf, lr = config.L_F, config.L_R
    velocidades = np.linspace(0.1, 11, 400)
    maximos = []
    for v in velocidades:
        seno_beta = 6 * lr / v ** 2
        if seno_beta >= 1:
            maximos.append(35)
        else:
            maximos.append(min(35, math.degrees(math.atan((lf + lr) / lr * math.tan(math.asin(seno_beta))))))
    ax.axhline(35, color=GRIS, ls="--", lw=0.8)
    ax.plot(velocidades, maximos, color=AZ, lw=1.4)
    ax.text(10.9, 36.5, "sin límite", ha="right", va="bottom", color=GRIS, fontsize=7)
    ax.text(6.0, 7, r"$a_{\max} = 6$ m/s$^2$", color=AZ, fontsize=7)
    ax.set_xlabel("velocidad (m/s)")
    ax.set_ylabel(r"$\delta_{\max}$ (°)")
    ax.set_ylim(0, 42)
    ax.set_xlim(0, 11)
    letra(ax, "c)", -0.25, 1.04)


# -------------------------------------------------------------------------
# La figura del auto.
# -------------------------------------------------------------------------
def auto():
    fig = plt.figure(figsize=(ANCHO, 1.85))
    a = fig.add_axes((0.0, 0.02, 0.30, 0.90))
    b = fig.add_axes((0.38, 0.17, 0.27, 0.72))
    c = fig.add_axes((0.74, 0.17, 0.26, 0.72))
    panel_bicicleta(a)
    panel_trayectorias(b)
    panel_volante_maximo(c)
    guardar(fig, "fig_auto")


# =========================================================================
# RED: sensores sobre el mapa, la red y su cromosoma
# =========================================================================

# -------------------------------------------------------------------------
# La primera ruta del 2x2 con al menos 3 esquinas y más de 150 m.
# -------------------------------------------------------------------------
def ruta_de_ejemplo(mapa):
    for origen, destino in itertools.permutations(list(mapa.grafo.nodes), 2):
        ruta = calcular_ruta(mapa, origen, destino)
        if ruta is not None and len(ruta.esquinas) >= 3 and ruta.largo > 150:
            return ruta


# -------------------------------------------------------------------------
# Un auto a 22 m del inicio del carril, corrido 0,9 m hacia la izquierda y
# girado 12°, para que el desvío y las miradas al carril se vean.
# -------------------------------------------------------------------------
def auto_de_ejemplo(ruta):
    p = ruta.carril.interpolate(22.0)
    q = ruta.carril.interpolate(23.0)
    angulo = math.atan2(q.y - p.y, q.x - p.x)
    normal = np.array([-math.sin(angulo), math.cos(angulo)])
    auto = Auto(p.x + 0.9 * normal[0], p.y + 0.9 * normal[1], angulo + math.radians(12))
    auto.velocidad = 5.0
    return auto


# -------------------------------------------------------------------------
# a) Las entradas sobre el mapa: los rayos, el próximo punto, el desvío del
#    carril y las miradas al carril a 3 y 6 m.
# -------------------------------------------------------------------------
def panel_sensores(ax, mapa, ruta, auto):
    siguiente = next(punto for punto in ruta.puntos if math.dist(punto, (auto.x, auto.y)) > 8)
    distancias = Sensores(mapa).medir_rayos(auto)

    # 1. El mapa, el carril y los rayos
    dibujar_calles(ax, mapa)
    ax.plot(*ruta.carril.xy, color=AZ, lw=0.8, ls="--", zorder=1)
    for angulo_relativo, distancia in zip(config.ANGULOS_RAYOS, distancias):
        ux, uy = math.cos(auto.angulo + angulo_relativo), math.sin(auto.angulo + angulo_relativo)
        ax.plot([auto.x, auto.x + distancia * ux], [auto.y, auto.y + distancia * uy], color=GRIS, lw=0.8, zorder=2)
        ax.plot(auto.x + distancia * ux, auto.y + distancia * uy, "o", color=GRIS, ms=2, zorder=2)

    # 2. El auto y el próximo punto
    ax.add_patch(Polygon(np.array(auto.esquinas()), fc=AZ, ec=TINTA, lw=0.5, zorder=5))
    flecha(ax, (auto.x, auto.y), siguiente, color=NA, lw=0.9, ls="--", z=4)
    ax.plot(*siguiente, "o", color=NA, ms=4, zorder=6)

    # 3. El desvío: del auto al punto más cercano del carril
    cercano = ruta.carril.interpolate(ruta.carril.project(shapely.Point(auto.x, auto.y)))
    ax.plot([auto.x, cercano.x], [auto.y, cercano.y], color=VE, lw=1.8, zorder=6)

    # 4. Las miradas al carril, 3 y 6 m más adelante
    sobre_el_carril = ruta.carril.project(shapely.Point(auto.x, auto.y))
    for metros in (3, 6):
        mirada = ruta.carril.interpolate(sobre_el_carril + metros)
        ax.plot(mirada.x, mirada.y, "D", color=MA, ms=3, zorder=6)
        ax.plot([auto.x, mirada.x], [auto.y, mirada.y], color=MA, lw=0.7, zorder=4)

    # 5. Leyenda y encuadre
    ax.plot([], [], color=GRIS, lw=0.8, label="1–7: rayos al borde")
    ax.plot([], [], color=NA, ls="--", lw=0.9, label="9–11: próximo punto")
    ax.plot([], [], color=VE, lw=1.8, label="12: desvío del carril")
    ax.plot([], [], "D-", color=MA, ms=3, lw=0.7, label="13–16: carril a 3 y 6 m")
    ax.set_xlim(auto.x - 9, auto.x + 15)
    ax.set_ylim(auto.y - 6, auto.y + 18)
    sin_ejes(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, handlelength=1.3,
              columnspacing=0.8)
    letra(ax, "a)", 0.0, 1.0)


# -------------------------------------------------------------------------
# b) La red 16 → 16 → 2, con las entradas agrupadas y coloreadas como en a).
# -------------------------------------------------------------------------
def panel_red(ax):
    nombres = ["rayos (7)", "velocidad", "punto: sen, cos", "punto: dist.", "desvío", "carril 3 m: sen, cos",
               "carril 6 m: sen, cos"]
    grupos = [7, 1, 2, 1, 1, 2, 2]

    # 1. Altura de cada neurona. Entre un grupo de entradas y el siguiente se deja un poco de aire.
    y_entradas = np.linspace(1, -1, 16) * 1.6
    for inicio in np.cumsum(grupos)[:-1]:
        y_entradas[inicio:] -= 0.07
    y_ocultas = np.linspace(1, -1, 16) * 1.6
    y_salidas = np.array([0.5, -0.5])
    colores_entradas = sum(([color] * cantidad for color, cantidad in zip([GRIS, TINTA, NA, NA, VE, MA, MA], grupos)), [])

    # 2. Las conexiones y las neuronas
    for y_entrada in y_entradas:
        for y_oculta in y_ocultas:
            ax.plot([0, 2], [y_entrada, y_oculta], color=GRIS_CLARO, lw=0.15, zorder=1)
    for y_oculta in y_ocultas:
        for y_salida in y_salidas:
            ax.plot([2, 4], [y_oculta, y_salida], color=GRIS_CLARO, lw=0.3, zorder=1)
    ax.scatter(np.zeros(16), y_entradas, s=10, c=colores_entradas, zorder=3, edgecolors="white", linewidths=0.3)
    ax.scatter(np.full(16, 2), y_ocultas, s=10, c=AZ, zorder=3, edgecolors="white", linewidths=0.3)
    ax.scatter(np.full(2, 4), y_salidas, s=18, c=TINTA, zorder=3)

    # 3. Los nombres de cada grupo de entradas y de las salidas
    primera = 0
    for nombre, cantidad in zip(nombres, grupos):
        ax.text(-0.15, np.mean(y_entradas[primera:primera + cantidad]), nombre, ha="right", va="center", fontsize=6)
        primera += cantidad
    ax.text(4.15, y_salidas[0], r"$o_1 \rightarrow \delta$", va="center", fontsize=7)
    ax.text(4.15, y_salidas[1], r"$o_2 \rightarrow a$", va="center", fontsize=7)
    for x, texto in ((0, "16 entradas"), (2, "16 ocultas"), (4, "2 salidas")):
        ax.text(x, 1.85, texto, ha="center", va="bottom", fontsize=7)
    ax.set_xlim(-2.6, 5.2)
    ax.set_ylim(-2.15, 2.1)
    ax.axis("off")
    letra(ax, "b)", 0.0, 0.98)


# -------------------------------------------------------------------------
# La figura de la red: a) los sensores en una situación real y b) la red.
# -------------------------------------------------------------------------
def red():
    mapa = santa_fe("2x2")
    ruta = ruta_de_ejemplo(mapa)
    auto_ejemplo = auto_de_ejemplo(ruta)

    fig = plt.figure(figsize=(ANCHO, 1.85))
    a = fig.add_axes((0.0, 0.0, 0.36, 0.97))
    b = fig.add_axes((0.40, 0.06, 0.60, 0.9))
    panel_sensores(a, mapa, ruta, auto_ejemplo)
    panel_red(b)
    guardar(fig, "fig_red")


# =========================================================================
# OPERADORES: qué genes toca cada operador
# =========================================================================

# -------------------------------------------------------------------------
# La matriz de pesos entrada → oculta (16 × 16) de un cromosoma.
# -------------------------------------------------------------------------
def pesos_de_entrada(cromosoma):
    return Perceptron(config.CAPAS, cromosoma=cromosoma).pesos[0]


# -------------------------------------------------------------------------
# De qué padre viene cada peso del hijo: 0 = del padre, 1 = de la madre,
# nan = de ninguno de los dos.
# -------------------------------------------------------------------------
def de_quien(hijo, padre, madre):
    return np.where(np.isclose(hijo, padre), 0.0, np.where(np.isclose(hijo, madre), 1.0, np.nan))


# -------------------------------------------------------------------------
# Dónde cae cada peso del hijo entre los de los padres: 0 = el del padre,
# 1 = el de la madre, fuera de [0, 1] = más allá de los dos.
# -------------------------------------------------------------------------
def fraccion(hijo, padre, madre):
    diferencia = np.where(np.abs(madre - padre) < 1e-12, 1, madre - padre)
    return np.clip((hijo - padre) / diferencia, -0.5, 1.5)


# -------------------------------------------------------------------------
# Qué pesos cambió una mutación: 1 = cambió, 0 = quedó igual.
# -------------------------------------------------------------------------
def mutados(hijo, padre):
    return (np.abs(hijo - padre) > 1e-12).astype(float)


# -------------------------------------------------------------------------
# La guía de lectura de las matrices: una columna es una neurona y una
# casilla es un peso.
# -------------------------------------------------------------------------
def panel_como_se_lee(fig):
    ax = fig.add_axes((0.0, 0.3, 0.135, 0.6))
    ax.imshow(np.zeros((16, 16)), cmap=ListedColormap([SIN_CAMBIO]), interpolation="nearest")
    ax.add_patch(Rectangle((2.5, -0.5), 1, 16, fc=AZ, ec="none", alpha=0.35))
    ax.add_patch(Rectangle((11.5, 12.5), 1, 1, fc=TINTA, ec="none"))
    ax.set_title("cómo se lee", fontsize=7, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    bordes_grises(ax)
    ax.set_ylabel("16 entradas", fontsize=6.5, labelpad=1)
    ax.set_xlabel("16 neuronas ocultas", fontsize=6.5, labelpad=1)
    ax.annotate("columna =\nuna neurona", (3.4, 4.0), (9.5, 4.0), fontsize=6, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=TINTA, mutation_scale=5))
    ax.annotate("casilla =\nun peso", (12.0, 12.4), (9.5, 9.0), fontsize=6, ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", lw=0.5, color=TINTA, mutation_scale=5))


# -------------------------------------------------------------------------
# La figura de los operadores: la matriz entrada → oculta de lo que deja cada
# operador sobre dos redes al azar.
# -------------------------------------------------------------------------
def operadores():
    capas = config.CAPAS
    fig = plt.figure(figsize=(ANCHO, 1.6))

    # 1. Dos redes al azar, con los pesos más grandes para que se vea la mezcla
    rng = np.random.default_rng(7)
    padre, madre = poblacion_inicial(2, capas, rng)
    padre = padre * 4
    madre = madre * 4
    P = pesos_de_entrada(padre)
    M = pesos_de_entrada(madre)

    # 2. Lo que deja cada operador
    hijo_neurona = pesos_de_entrada(cruzar(padre, madre, capas, "neurona", np.random.default_rng(1)))
    hijo_pesos = pesos_de_entrada(cruzar(padre, madre, capas, "pesos", np.random.default_rng(1)))
    hijo_blx = pesos_de_entrada(cruzar(padre, madre, capas, "blx", np.random.default_rng(1)))
    mutado_gaussiana = pesos_de_entrada(mutar(padre, capas, 0.3, "gaussiana", np.random.default_rng(2)))
    mutado_neurona = pesos_de_entrada(mutar(padre, capas, 0.3, "neurona", np.random.default_rng(2)))

    # 3. Cada panel: la matriz, sus colores, su rango, el título y la explicación
    origen = ListedColormap([AZ, NA])
    escala = LinearSegmentedColormap.from_list("blx", [AZ, SIN_CAMBIO, NA])
    cambio = LinearSegmentedColormap.from_list("mut", [SIN_CAMBIO, TINTA])
    paneles = [(de_quien(hijo_neurona, P, M), origen, 0, 1, "cruce por neurona", "neuronas enteras\nde cada padre"),
               (de_quien(hijo_pesos, P, M), origen, 0, 1, "cruce por pesos", "cada peso de un\npadre al azar"),
               (fraccion(hijo_blx, P, M), escala, -0.5, 1.5, r"cruce BLX-$\alpha$", "valores entre\nlos dos padres"),
               (mutados(mutado_gaussiana, P), cambio, 0, 1, "mutación gaussiana", "pesos sueltos\nal azar"),
               (mutados(mutado_neurona, P), cambio, 0, 1, "mutación por neurona", "dos neuronas\nenteras")]

    # 4. La guía de lectura y las cinco matrices
    panel_como_se_lee(fig)
    for i, (matriz, colores, minimo, maximo, titulo, explicacion) in enumerate(paneles):
        ax = fig.add_axes((0.175 + 0.167 * i, 0.3, 0.135, 0.6))
        ax.imshow(matriz, cmap=colores, vmin=minimo, vmax=maximo, interpolation="nearest")
        ax.set_title(titulo, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(explicacion, fontsize=6.5, color="#555555", labelpad=2)
        bordes_grises(ax)

    # 5. La leyenda común
    fig.legend(handles=[Patch(color=AZ, label="peso del padre"), Patch(color=NA, label="peso de la madre"),
                        Patch(color=TINTA, label="peso mutado"),
                        Patch(color=SIN_CAMBIO, ec=GRIS, lw=0.4, label="sin cambio")],
               loc="lower center", bbox_to_anchor=(0.58, -0.04), ncol=4, frameon=False, handlelength=1.0)
    print("operadores: genes mutados gaussiana", int(mutados(mutado_gaussiana, P).sum()), "en 256;",
          "por neurona", int(mutados(mutado_neurona, P).sum()))
    guardar(fig, "fig_operadores")


if __name__ == "__main__":
    todas = {"mapas": mapas, "auto": auto, "red": red, "operadores": operadores}
    for nombre in (sys.argv[1:] or todas):
        todas[nombre]()
