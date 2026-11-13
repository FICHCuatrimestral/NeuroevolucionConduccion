"""Figuras de la presentación que muestran el problema y nuestro método.

Uso, parado en la carpeta del proyecto:
    python presentacion/figuras_metodo.py

  - problema.png       el centro de Santa Fe con un punto A, un punto B y la ruta entre los dos
  - mapas_etapas.png   los mapas de las etapas: manzana, 2x2, 3x3 y el centro
  - esquina.png        la curva más amplia que entra en un cruce, doblando a la izquierda y a la derecha
  - algoritmo.png      el diagrama de flujo del entrenamiento por etapas
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
    caja(ax, 5.1, arriba, "Evaluar", "3 rutas + 1 de una etapa\nanterior; nota = promedio", ancho=3.6)
    caja(ax, 9.0, arriba, "Confirmar la mejor", "en las 20 rutas fijas", ancho=3.4)
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


# -------------------------------------------------------------------------
# El cromosoma de nuestra red, 16 → 16 → 2. Arriba, los 306 genes en fila con
# sus cuatro partes. Abajo, los primeros 256 armados como la matriz de pesos de
# entrada a la capa oculta: una fila por entrada y una columna por neurona.
# El cromosoma lee la matriz fila por fila, así que los 16 pesos de una neurona
# (una columna) quedan salteados cada 16 genes. Se marca la neurona 3 en los dos.
# -------------------------------------------------------------------------
def cromosoma():
    entradas, ocultas, salidas = 16, 16, 2
    partes = [("pesos entrada → oculta", entradas * ocultas, "#9cc3ef"), ("umbrales ocultos", ocultas, "#a6dcc6"),
              ("pesos oculta → salida", ocultas * salidas, "#f6b99c"), ("umbrales de salida", salidas, "#f2d27a")]
    neurona = 2
    marcados = {fila * ocultas + neurona for fila in range(entradas)} | {entradas * ocultas + neurona}
    colores_fuertes = {"#9cc3ef": AZ, "#a6dcc6": "#1baf7a"}

    fig, ax = plt.subplots(figsize=(16, 6.6))
    ancho = 0.052
    gen = 0
    for nombre, cantidad, color in partes:
        x0 = gen * ancho
        for k in range(cantidad):
            fuerte = gen in marcados
            ax.add_patch(Rectangle(((gen) * ancho, 5.2), ancho, 0.7, lw=0,
                                   color=colores_fuertes.get(color, color) if fuerte else color))
            gen += 1
        ax.plot([gen * ancho, gen * ancho], [5.1, 6.0], color="white", lw=2)
        # Los umbrales son partes angostas: su rótulo va arriba de la barra, cada uno a otra altura
        if nombre.startswith("umbrales"):
            altura = 6.05 if nombre == "umbrales ocultos" else 6.5
            ax.text((x0 + gen * ancho) / 2, altura, f"{nombre}: {cantidad}", ha="center", va="bottom", fontsize=12.5,
                    color=TINTA)
            ax.plot([(x0 + gen * ancho) / 2] * 2, [5.95, altura], color=GRIS, lw=0.8)
        else:
            ax.text((x0 + gen * ancho) / 2, 4.9, f"{nombre}\n{cantidad}", ha="center", va="top", fontsize=12.5,
                    color=TINTA)
    ax.text(gen * ancho + 0.15, 5.55, "306 genes", va="center", fontsize=14, fontweight="bold", color=TINTA)

    # La matriz de 16 × 16: cada fila es una entrada; cada columna, una neurona oculta
    lado, x0, y0 = 0.2, 1.2, 0.2
    for fila in range(entradas):
        for columna in range(ocultas):
            color = AZ if columna == neurona else "#9cc3ef"
            ax.add_patch(Rectangle((x0 + columna * lado, y0 + (entradas - 1 - fila) * lado), lado * 0.92, lado * 0.92,
                                   color=color, lw=0))
    ax.text(x0 + ocultas * lado / 2, y0 + entradas * lado + 0.12, "neuronas ocultas (columnas)", ha="center",
            va="bottom", fontsize=12.5, color=TINTA)
    ax.text(x0 - 0.15, y0 + entradas * lado / 2, "entradas\n(filas)", ha="right", va="center", fontsize=12.5,
            color=TINTA)
    ax.annotate("", xy=(x0 + ocultas * lado / 2, y0 + entradas * lado + 0.55), xytext=(3.0, 5.15),
                arrowprops={"arrowstyle": "-|>", "color": GRIS, "lw": 1.2, "connectionstyle": "arc3,rad=0.15"})
    ax.text(5.0, 2.9, "Los primeros 256 genes son la matriz de pesos de entrada a la capa oculta,\n"
                      "leída fila por fila.\n\n"
                      "Una neurona es una columna: sus 16 pesos de entrada\n"
                      "quedan salteados cada 16 genes, y su umbral va aparte.\n"
                      "El cruce y la mutación por neurona eligen por columna, no por posición.",
            fontsize=13.5, color=TINTA, va="center", linespacing=1.4)
    ax.set_xlim(0, 17.8)
    ax.set_ylim(0, 6.9)
    ax.set_aspect("equal")
    ax.axis("off")
    guardar(fig, "cromosoma.png")


if __name__ == "__main__":
    cromosoma()
    algoritmo()
    esquina()
    centro = santa_fe("centro")
    problema(centro)
    mapas_etapas(centro)
