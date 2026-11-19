"""Figuras de la presentación que muestran el problema y nuestro método.

Uso, parado en la carpeta del proyecto:
    python presentacion/figuras_metodo.py

  - problema.png       el centro de Santa Fe con un punto A, un punto B y la ruta entre los dos
  - mapas_etapas.png   los mapas de las etapas: manzana, 2x2, 3x3 y el centro
  - esquina.png        la curva más amplia que entra en un cruce, doblando a la izquierda y a la derecha
  - algoritmo.png      el diagrama de flujo del entrenamiento por etapas
  - cromosoma.png      la red y su cromosoma de 306 genes, con una neurona marcada
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

# La carpeta del proyecto, para poder importar el paquete conduccion
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
SALIDA = Path(__file__).resolve().parent / "img"

from conduccion.aprendizaje.entrenamiento import rutas_de_referencia
from conduccion.mapa.mapa import manzana, santa_fe

AZ, NA, TINTA, GRIS, CALLE = "#2a78d6", "#eb6834", "#222222", "#8a8a85", "#dcdcd8"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15})

# Ancho de cada calle en la figura de la esquina: dos manos de 3,5 m
ANCHO_CALLE = 7.0

# Aceleración lateral máxima del límite de agarre, en m/s²
ACELERACION_LATERAL = 6.0


# -------------------------------------------------------------------------
# Pinta las calles del mapa. Los huecos del polígono (las manzanas) van en blanco.
# -------------------------------------------------------------------------
def dibujar_calles(ax, mapa):
    partes = mapa.calles.geoms if hasattr(mapa.calles, "geoms") else [mapa.calles]
    for parte in partes:
        ax.fill(*parte.exterior.xy, color=CALLE, lw=0, zorder=0)
        for hueco in parte.interiors:
            ax.fill(*hueco.xy, color="white", lw=0, zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")


# -------------------------------------------------------------------------
# Guarda la figura en img/ y la cierra.
# -------------------------------------------------------------------------
def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("->", nombre)


# -------------------------------------------------------------------------
# El problema: el centro, con una ruta de A a B. Se elige la más larga de
# las rutas de referencia, para que cruce buena parte del mapa.
# -------------------------------------------------------------------------
def problema(centro):
    ruta = max(rutas_de_referencia(centro, 0), key=lambda r: r.largo)
    fig, ax = plt.subplots(figsize=(8, 8))
    dibujar_calles(ax, centro)
    ax.plot(*ruta.carril.xy, color=AZ, lw=3.5, zorder=2)
    for (x, y), letra in ((ruta.carril.coords[0], "A"), (ruta.carril.coords[-1], "B")):
        ax.plot(x, y, "o", color=NA, ms=16, zorder=3)
        ax.text(x, y, letra, color="white", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    guardar(fig, "problema.png")


# -------------------------------------------------------------------------
# Los mapas de las etapas, uno al lado del otro, de lo fácil a lo difícil.
# -------------------------------------------------------------------------
def mapas_etapas(centro):
    mapas = [("Manzana", manzana()), ("2×2", santa_fe("2x2")), ("3×3", santa_fe("3x3")),
             ("Centro (solo prueba)", centro)]
    fig, ejes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (titulo, mapa) in zip(ejes, mapas):
        dibujar_calles(ax, mapa)
        ax.set_title(titulo, fontsize=16, color=TINTA)

    # En la manzana se maneja solo la vuelta a la manzana del medio
    mapa = mapas[0][1]
    xs, ys = zip(*[mapa.posicion(esquina) for esquina in mapa.esquinas_manzana])
    ejes[0].plot(xs + xs[:1], ys + ys[:1], color=AZ, lw=4)
    guardar(fig, "mapas_etapas.png")


# -------------------------------------------------------------------------
# Un arco de circunferencia, de un ángulo a otro (en grados).
# -------------------------------------------------------------------------
def arco(ax, centro, radio, desde, hasta, color):
    angulos = np.radians(np.linspace(desde, hasta, 100))
    ax.plot(centro[0] + radio * np.cos(angulos), centro[1] + radio * np.sin(angulos), color=color, lw=3)


# -------------------------------------------------------------------------
# La esquina: un cruce de dos calles de 7 m. El auto llega desde abajo por el
# medio de su carril (x = 1,75) y dobla. La curva más amplia que no se mete en
# la manzana es tangente al carril de llegada y al de salida y pasa por la
# esquina de la manzana:
#   izquierda: centro en (1,75 − R; 1,75 − R) y esquina en (−3,5; −3,5) → R = 17,9 m
#   derecha:   centro en (1,75 + R; −1,75 − R) y esquina en (3,5; −3,5)  → R = 6,0 m
# A 6 m/s² de aceleración lateral, la velocidad máxima es raíz(6 · R).
# -------------------------------------------------------------------------
def esquina():
    medio = ANCHO_CALLE / 2
    carril = medio / 2
    izquierda = (medio + carril) * math.sqrt(2) / (math.sqrt(2) - 1)
    derecha = carril * math.sqrt(2) / (math.sqrt(2) - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    largo = 24
    ax.add_patch(Rectangle((-largo, -medio), 2 * largo, ANCHO_CALLE, color=CALLE, lw=0))
    ax.add_patch(Rectangle((-medio, -largo), ANCHO_CALLE, 2 * largo, color=CALLE, lw=0))
    ax.plot([-largo, -medio], [0, 0], color="white", lw=1.5, ls="--")
    ax.plot([medio, largo], [0, 0], color="white", lw=1.5, ls="--")
    ax.plot([0, 0], [-largo, -medio], color="white", lw=1.5, ls="--")
    ax.plot([0, 0], [medio, largo], color="white", lw=1.5, ls="--")

    # Las dos curvas, desde la llegada por el carril derecho hasta la salida
    arco(ax, (carril - izquierda, carril - izquierda), izquierda, 0, 90, AZ)
    arco(ax, (carril + derecha, -carril - derecha), derecha, 180, 90, NA)
    ax.annotate("", xy=(carril, -largo + 6), xytext=(carril, -largo), arrowprops={"arrowstyle": "-|>", "color": TINTA})
    for x, y in ((-medio, -medio), (medio, -medio)):
        ax.plot(x, y, "o", color=TINTA, ms=6)

    v_izquierda = 3.6 * math.sqrt(ACELERACION_LATERAL * izquierda)
    v_derecha = 3.6 * math.sqrt(ACELERACION_LATERAL * derecha)
    ax.text(0, largo + 5, f"izquierda: R = {izquierda:.1f} m → hasta {v_izquierda:.0f} km/h".replace(".", ","),
            color=AZ, fontsize=17, ha="center")
    ax.text(0, largo + 1.5, f"derecha: R = {derecha:.1f} m → hasta {v_derecha:.0f} km/h".replace(".", ","),
            color=NA, fontsize=17, ha="center")
    ax.set_xlim(-largo, largo)
    ax.set_ylim(-largo, largo)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "esquina.png")


# -------------------------------------------------------------------------
# La misma esquina, doblando a la derecha, a dos velocidades y con el límite de
# agarre. El radio más chico posible es v² / a_max:
#   a 15 km/h: R = 2,9 m, entra con más de 1 m de sobra (la esquina admite hasta 6,0 m)
#   a 40 km/h: R = 20,6 m, no entra: se mete en la manzana
# La de 15 km/h dobla para terminar en el medio de su carril; la de 40 km/h empieza a
# doblar donde empieza la curva más cerrada que entra (R = 6,0 m), y aun así se sale.
# Va como panel d) de la figura del auto, con su misma letra.
# -------------------------------------------------------------------------
def esquina_rapida():
    medio = ANCHO_CALLE / 2
    carril = medio / 2
    justa = carril * math.sqrt(2) / (math.sqrt(2) - 1)
    y0 = -carril - justa

    with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix",
                         "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]}):
        fig, ax = plt.subplots(figsize=(4.2, 4.6))
        largo = 11
        ax.add_patch(Rectangle((-largo, -medio), 2 * largo, ANCHO_CALLE, color=CALLE, lw=0))
        ax.add_patch(Rectangle((-medio, -largo), ANCHO_CALLE, 2 * largo, color=CALLE, lw=0))
        ax.plot([0, 0], [-largo, -medio], color="white", lw=1.2, ls="--")
        ax.plot([medio, largo], [0, 0], color="white", lw=1.2, ls="--")

        for kmh, color, sale in ((15, AZ, False), (40, NA, True)):
            v = kmh / 3.6
            radio = v ** 2 / ACELERACION_LATERAL
            inicio = y0 if sale else -carril - radio
            centro = (carril + radio, inicio)
            # Ángulo desde el centro: 180° es el punto de partida; baja hacia 90° al doblar a la derecha
            hasta = 90 if not sale else 180 - math.degrees(math.asin((largo - y0) / radio)) if radio > largo - y0 else 120
            angulos = np.radians(np.linspace(180, hasta, 200))
            ax.plot([carril, carril], [-largo, inicio], color=color, lw=2.5, zorder=2 if sale else 3)
            xs = centro[0] + radio * np.cos(angulos)
            ys = centro[1] + radio * np.sin(angulos)
            if not sale:
                xs = np.append(xs, largo)
                ys = np.append(ys, ys[-1])
            dentro = (xs <= largo) & (ys <= largo)
            ax.plot(xs[dentro], ys[dentro], color=color, lw=2.5, zorder=2 if sale else 3)
            if sale:
                # Dónde sale de la calle: el primer punto que no está en ninguna de las dos
                fuera = ~((np.abs(xs) <= medio) | (np.abs(ys) <= medio))
                k = int(np.argmax(fuera))
                ax.plot(xs[k], ys[k], "X", color=NA, ms=13, mec="white", mew=1.2, zorder=5)
        ax.text(-largo + 0.6, largo - 1.6, "15 km/h: dobla", color=AZ, fontsize=20)
        ax.text(-largo + 0.6, largo - 4.4, "40 km/h: se sale", color=NA, fontsize=20)
        ax.text(-largo, largo + 0.7, "d)", fontsize=15, fontweight="bold", va="bottom")
        ax.set_xlim(-largo, largo)
        ax.set_ylim(-largo, largo)
        ax.set_aspect("equal")
        ax.axis("off")
        guardar(fig, "esquina_rapida.png")


# -------------------------------------------------------------------------
# Una caja del diagrama, centrada en (x, y), con un título en negrita y un
# texto debajo. Las preguntas van en un rombo.
# -------------------------------------------------------------------------
def caja(ax, x, y, titulo, texto="", pregunta=False, ancho=2.9, alto=1.5):
    if pregunta:
        ax.add_patch(Polygon([(x - ancho / 2, y), (x, y + alto / 2), (x + ancho / 2, y), (x, y - alto / 2)],
                             closed=True, fc="white", ec=NA, lw=2))
        ax.text(x, y, titulo, ha="center", va="center", fontsize=13, color=TINTA)
        return
    ax.add_patch(FancyBboxPatch((x - ancho / 2, y - alto / 2), ancho, alto, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="white", ec=AZ, lw=2))
    ax.text(x, y + (0.3 if texto else 0), titulo, ha="center", va="center", fontsize=13, fontweight="bold",
            color=TINTA)
    if texto:
        ax.text(x, y - 0.25, texto, ha="center", va="center", fontsize=11, color=GRIS, linespacing=1.3)


# -------------------------------------------------------------------------
# Una flecha que sigue los puntos dados, con un rótulo opcional en el medio
# del primer tramo.
# -------------------------------------------------------------------------
def flecha(ax, puntos, rotulo=""):
    for a, b in zip(puntos[:-2], puntos[1:-1]):
        ax.plot(*zip(a, b), color=TINTA, lw=1.5)
    ax.annotate("", xy=puntos[-1], xytext=puntos[-2],
                arrowprops={"arrowstyle": "-|>", "color": TINTA, "lw": 1.5, "mutation_scale": 16})
    if rotulo:
        (x0, y0), (x1, y1) = puntos[0], puntos[1]
        if y0 == y1:
            ax.text((x0 + x1) / 2, y0 + 0.12, rotulo, fontsize=12, color=NA, ha="center", va="bottom")
        else:
            ax.text(x0 + 0.15, (y0 + y1) / 2, rotulo, fontsize=12, color=NA, ha="left", va="center")


# -------------------------------------------------------------------------
# El diagrama de flujo del entrenamiento: arriba, de izquierda a derecha, lo
# que pasa en cada generación hasta la pregunta de si pasa de etapa; abajo, la
# vuelta a la generación siguiente y el cambio de etapa.
# -------------------------------------------------------------------------
def algoritmo():
    fig, ax = plt.subplots(figsize=(16, 5.2))
    arriba, abajo = 3.4, 0.9
    caja(ax, 1.5, arriba, "Inicio", "50 redes al azar", ancho=2.6)
    caja(ax, 5.1, arriba, "Evaluar", "3 rutas + 1 de una etapa\nanterior; fitness = promedio", ancho=3.6)
    caja(ax, 9.0, arriba, "Confirmar la mejor", "en las rutas fijas\n(20; 8 en la manzana)", ancho=3.4)
    caja(ax, 12.5, arriba, "¿Pasa de\netapa?", pregunta=True, ancho=2.6, alto=1.8)
    caja(ax, 15.6, arriba, "¿Última\netapa?", pregunta=True, ancho=2.6, alto=1.8)
    caja(ax, 18.8, arriba, "Mejor red", "guardada", ancho=2.4)
    caja(ax, 9.0, abajo, "Generación siguiente", "élite 2 · torneo 3\ncruce 0,8 · mutación σ", ancho=3.9, alto=1.6)
    caja(ax, 15.6, abajo, "Etapa siguiente", "σ más chico", ancho=3.2)

    flecha(ax, [(2.8, arriba), (3.3, arriba)])
    flecha(ax, [(6.9, arriba), (7.3, arriba)])
    flecha(ax, [(10.7, arriba), (11.2, arriba)])
    flecha(ax, [(13.8, arriba), (14.3, arriba)], "sí")
    flecha(ax, [(16.9, arriba), (17.6, arriba)], "sí")
    flecha(ax, [(12.5, arriba - 0.9), (12.5, abajo), (10.95, abajo)], "no")
    flecha(ax, [(15.6, arriba - 0.9), (15.6, abajo + 0.75)], "no")
    flecha(ax, [(14.0, abajo), (12.5, abajo)])
    flecha(ax, [(7.05, abajo), (5.1, abajo), (5.1, arriba - 0.75)])

    ax.set_xlim(0, 20.2)
    ax.set_ylim(-0.1, 4.4)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "algoritmo.png")


# Partes del cromosoma, en orden: nombre, cantidad de genes, color claro (barra) y fuerte (red)
PARTES_CROMOSOMA = [("pesos entrada → oculta", 256, "#9cc3ef", "#7fb0e8"),
                    ("umbrales ocultos", 16, "#a6dcc6", "#1baf7a"),
                    ("pesos oculta → salida", 32, "#f6b99c", "#ee8a5c"),
                    ("umbrales de salida", 2, "#f2d27a", "#d9a800")]


# -------------------------------------------------------------------------
# Dibuja la red 16 → 16 → 2 con los colores de las partes del cromosoma. Si se
# pasa una neurona oculta, sus pesos de entrada van en azul fuerte.
# -------------------------------------------------------------------------
def dibujar_red(ax, x0, y0, alto, ancho, marcar=None, entradas=16, ocultas=16, salidas=2):
    def alturas(n):
        return [y0 + alto * (1 - (i + 0.5) / n) for i in range(n)]
    xe, xo, xs = x0, x0 + ancho / 2, x0 + ancho
    ye, yo, ys = alturas(entradas), alturas(ocultas), alturas(salidas)
    for j, b in enumerate(yo):
        for a in ye:
            fuerte = marcar == j
            ax.plot([xe, xo], [a, b], color=AZ if fuerte else PARTES_CROMOSOMA[0][3], lw=1.6 if fuerte else 0.5,
                    alpha=1 if fuerte else 0.8, zorder=3 if fuerte else 1)
    for a in yo:
        for b in ys:
            ax.plot([xo, xs], [a, b], color=PARTES_CROMOSOMA[2][3], lw=0.6, alpha=0.7, zorder=1)
    ax.scatter([xe] * entradas, ye, s=28, color=GRIS, zorder=4)
    ax.scatter([xo] * ocultas, yo, s=40, zorder=4,
               color=[AZ if marcar == j else PARTES_CROMOSOMA[1][3] for j in range(ocultas)],
               edgecolor=TINTA, linewidth=[1.2 if marcar == j else 0 for j in range(ocultas)])
    ax.scatter([xs] * salidas, ys, s=90, color=PARTES_CROMOSOMA[3][3], zorder=4)
    for x, texto in [(xe, f"{entradas} entradas"), (xo, f"{ocultas} ocultas"), (xs, f"{salidas} salidas")]:
        ax.text(x, y0 + alto + 0.25, texto, ha="center", va="bottom", fontsize=13, color=TINTA)


# -------------------------------------------------------------------------
# Dibuja el cromosoma como una barra de genes, con cada parte de su color. Los
# genes de `marcar` van en azul fuerte.
# -------------------------------------------------------------------------
def dibujar_barra(ax, x0, y, largo, alto, marcar=frozenset(), tam=11.5):
    total = sum(cantidad for _, cantidad, _, _ in PARTES_CROMOSOMA)
    g = largo / total
    gen = 0
    for nombre, cantidad, claro, _ in PARTES_CROMOSOMA:
        inicio = gen
        for _ in range(cantidad):
            ax.add_patch(Rectangle((x0 + gen * g, y), g, alto, lw=0, color=AZ if gen in marcar else claro))
            gen += 1
        ax.plot([x0 + gen * g] * 2, [y - 0.05, y + alto + 0.05], color="white", lw=1.5)
        medio = x0 + (inicio + gen) / 2 * g
        if cantidad >= 32:
            ax.text(medio, y - 0.15, f"{cantidad}\n{nombre}", ha="center", va="top", fontsize=tam, color=TINTA)
        else:
            # Las partes angostas llevan el rótulo arriba, cada una a otra altura
            altura = y + alto + (0.25 if cantidad == 16 else 0.75)
            ax.text(medio, altura, f"{cantidad} {nombre}", ha="center", va="bottom", fontsize=tam, color=TINTA)
            ax.plot([medio] * 2, [y + alto, altura], color=GRIS, lw=0.8)
    ax.text(x0 + largo + 0.15, y + alto / 2, f"{total} genes", va="center", fontsize=tam + 1.5, fontweight="bold",
            color=TINTA)


# -------------------------------------------------------------------------
# El cromosoma de nuestra red, 16 → 16 → 2: a la izquierda la red, a la derecha
# sus 306 genes en fila, con cada parte del mismo color. Se marca la neurona 3:
# el cromosoma lee la matriz de pesos fila por fila, así que sus 16 pesos de
# entrada quedan salteados cada 16 genes, y su umbral va aparte.
# -------------------------------------------------------------------------
def cromosoma():
    entradas, ocultas, neurona = 16, 16, 2
    # Los genes de una neurona oculta: su columna de la matriz de pesos (uno cada 16) y su umbral
    marcados = {fila * ocultas + neurona for fila in range(entradas)} | {entradas * ocultas + neurona}
    fig, ax = plt.subplots(figsize=(16, 4.2))
    dibujar_red(ax, 0.3, 0.2, 3.4, 3.2, marcar=neurona)
    ax.annotate("", xy=(5.0, 1.9), xytext=(3.9, 1.9), arrowprops={"arrowstyle": "-|>", "color": TINTA, "lw": 2})
    dibujar_barra(ax, 5.2, 1.55, 8.6, 0.7, marcar=marcados)
    ax.text(5.2, 0.15, "Una neurona (en azul) son sus 16 pesos de entrada y su umbral: 17 genes salteados.",
            fontsize=12.5, color=TINTA, va="center")
    ax.set_xlim(0, 15.6)
    ax.set_ylim(-0.1, 4.2)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "cromosoma.png")


if __name__ == "__main__":
    cromosoma()
    algoritmo()
    esquina()
    esquina_rapida()
    centro = santa_fe("centro")
    problema(centro)
    mapas_etapas(centro)
