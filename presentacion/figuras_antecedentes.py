"""Figuras de los antecedentes, redibujadas a partir de los papers de docs/.

Uso, parado en la carpeta del proyecto:
    python presentacion/figuras_antecedentes.py

Cada figura copia una figura de un paper, con sus mismos números, en castellano y con
los colores de la presentación. En la diapositiva va "Basado en <autor> (<año>), fig. <n>".
  - kong_bicicleta.png      Kong et al. (2015), fig. 1: el modelo cinemático de bicicleta
  - togelius_pista.png      Togelius y Lucas (2006), figs. 1 y 2: pista, puntos de control y sensores
  - montana_red.png         Montana y Davis (1989), fig. 1: una red escrita como cromosoma
  - montana_operadores.png  Montana y Davis (1989), fig. 2: mutación y cruce
  - yao_permutacion.png     Yao (1999), figs. 3 y 4: dos redes iguales con cromosomas distintos
  - gomez_etapas.png        Gomez y Miikkulainen (1996), sección 5.3: tareas cada vez más difíciles
  - nelson_aptitud.png      Nelson et al. (2009), fig. 1: clases de funciones de aptitud
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyBboxPatch, Polygon, Rectangle

SALIDA = Path(__file__).resolve().parent / "img" / "antecedentes"
SALIDA.mkdir(parents=True, exist_ok=True)

AZ, NA, TINTA, GRIS, CALLE, AMARILLO = "#2a78d6", "#eb6834", "#222222", "#8a8a85", "#dcdcd8", "#e8b923"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0})


# -------------------------------------------------------------------------
# Una figura sin ejes, con la misma escala en x y en y.
# -------------------------------------------------------------------------
def lienzo(ancho, alto):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# -------------------------------------------------------------------------
# Guarda la figura en img/antecedentes/ y la cierra.
# -------------------------------------------------------------------------
def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("guardada", nombre)


# -------------------------------------------------------------------------
# Una flecha de a hasta b.
# -------------------------------------------------------------------------
def flecha(ax, a, b, color=TINTA, lw=1.8, estilo="-|>"):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": estilo, "color": color, "lw": lw,
                                                 "shrinkA": 0, "shrinkB": 0, "mutation_scale": 16})


# -------------------------------------------------------------------------
# Una rueda: un rectángulo redondeado centrado en `centro`, girado `angulo`.
# -------------------------------------------------------------------------
def rueda(ax, centro, angulo, color):
    largo, ancho = 1.1, 0.35
    u = np.array([math.cos(angulo), math.sin(angulo)])
    n = np.array([-u[1], u[0]])
    esquinas = [centro + a * largo / 2 * u + b * ancho / 2 * n for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    ax.add_patch(Polygon(esquinas, closed=True, color=color, zorder=4))


# =========================================================================
# KONG ET AL. (2015), FIG. 1
# =========================================================================

# -------------------------------------------------------------------------
# El modelo cinemático de bicicleta: el auto visto de arriba, con sus dos
# ruedas, el centro de masa en (x, y), el rumbo psi, el ángulo de la velocidad
# beta, el volante delta_f y el centro de giro O.
# -------------------------------------------------------------------------
def kong_bicicleta():
    fig, ax = lienzo(9, 6.5)
    lf, lr = 2.6, 2.9
    psi, delta = math.radians(18), math.radians(32)
    beta = math.atan(lr / (lf + lr) * math.tan(delta))
    cm = np.array([6.0, 3.2])
    u = np.array([math.cos(psi), math.sin(psi)])
    n = np.array([-u[1], u[0]])
    delantera, trasera = cm + lf * u, cm - lr * u

    # 1. Los ejes X, Y y la posición (x, y) del centro de masa
    flecha(ax, (0, 0), (11.5, 0))
    flecha(ax, (0, 0), (0, 8.2))
    ax.text(11.6, -0.1, "$X$", va="top")
    ax.text(-0.15, 8.2, "$Y$", ha="right")
    ax.plot([cm[0], cm[0]], [0, cm[1]], color=GRIS, lw=1, ls=":")
    ax.plot([0, cm[0]], [cm[1], cm[1]], color=GRIS, lw=1, ls=":")
    ax.text(cm[0], -0.2, "$x$", ha="center", va="top")
    ax.text(-0.2, cm[1], "$y$", ha="right", va="center")

    # 2. El auto: el eje, la rueda trasera recta y la delantera girada delta_f
    ax.plot(*zip(trasera, delantera), color=TINTA, lw=2)
    rueda(ax, trasera, psi, GRIS)
    rueda(ax, delantera, psi + delta, NA)
    ax.add_patch(Circle(cm, 0.18, color=TINTA, zorder=5))
    ax.text(*(trasera + np.array([-0.3, -0.75])), r"$\delta_r = 0$", ha="center", color=GRIS, fontsize=13)

    # 3. Las distancias l_r y l_f, al costado del auto
    for a, b, texto in ((trasera, cm, "$l_r$"), (cm, delantera, "$l_f$")):
        a2, b2 = a - 0.9 * n, b - 0.9 * n
        flecha(ax, a2, b2, lw=1, estilo="<|-|>")
        ax.text(*((a2 + b2) / 2 - 0.45 * n), texto, ha="center", va="center")

    # 4. El rumbo psi, medido desde una horizontal
    ax.plot(*zip(cm, cm + np.array([2.2, 0])), color=GRIS, lw=1, ls="--")
    ax.add_patch(Arc(cm, 3.4, 3.4, theta1=0, theta2=math.degrees(psi), color=TINTA, lw=1.2))
    ax.text(*(cm + 1.95 * np.array([math.cos(psi / 2), math.sin(psi / 2)])), r"$\psi$", va="center")

    # 5. La velocidad v, que forma el ángulo beta con el eje del auto
    direccion = np.array([math.cos(psi + beta), math.sin(psi + beta)])
    flecha(ax, cm, cm + 2.6 * direccion, color=AZ, lw=2.4)
    ax.text(*(cm + 2.8 * direccion), "$v$", color=AZ, fontsize=17)
    ax.add_patch(Arc(cm, 2.2, 2.2, theta1=math.degrees(psi), theta2=math.degrees(psi + beta), color=AZ, lw=1.2))
    ax.text(*(cm + 1.4 * np.array([math.cos(psi + beta / 2), math.sin(psi + beta / 2)])), r"$\beta$",
            color=AZ, va="center", ha="center")

    # 6. El volante delta_f, entre el eje del auto y la rueda delantera
    ax.plot(*zip(delantera, delantera + 1.8 * u), color=GRIS, lw=1, ls="--")
    direccion_rueda = np.array([math.cos(psi + delta), math.sin(psi + delta)])
    ax.plot(*zip(delantera, delantera + 1.8 * direccion_rueda), color=NA, lw=1, ls="--")
    ax.add_patch(Arc(delantera, 2.6, 2.6, theta1=math.degrees(psi), theta2=math.degrees(psi + delta), color=NA,
                     lw=1.2))
    ax.text(*(delantera + 1.65 * np.array([math.cos(psi + delta / 2), math.sin(psi + delta / 2)])),
            r"$\delta_f$", color=NA, va="center", ha="center")

    # 7. El centro de giro O: donde se cruzan las perpendiculares a las dos ruedas
    normal_rueda = np.array([-direccion_rueda[1], direccion_rueda[0]])
    s = np.linalg.solve(np.array([n, -normal_rueda]).T, delantera - trasera)
    centro_giro = trasera + s[0] * n
    for punto in (trasera, cm, delantera):
        ax.plot(*zip(punto, centro_giro), color=GRIS, lw=0.8, ls=":")
    ax.plot(*centro_giro, "o", color=TINTA, ms=5)
    ax.text(centro_giro[0] + 0.25, centro_giro[1], "$O$", va="center")

    ax.set_xlim(-0.8, 12.2)
    ax.set_ylim(-0.9, max(8.6, centro_giro[1] + 0.5))
    guardar(fig, "kong_bicicleta.png")


# =========================================================================
# TOGELIUS Y LUCAS (2006), FIGS. 1 Y 2
# =========================================================================

# -------------------------------------------------------------------------
# Distancia desde `origen`, en la dirección `direccion`, hasta la pared más
# cercana. Las paredes son segmentos. Si no toca ninguna antes de `alcance`,
# devuelve `alcance`.
# -------------------------------------------------------------------------
def distancia_a_pared(origen, direccion, paredes, alcance):
    mejor = alcance
    for a, b in paredes:
        a, b = np.array(a, float), np.array(b, float)
        matriz = np.array([direccion, a - b]).T
        if abs(np.linalg.det(matriz)) < 1e-9:
            continue
        t, s = np.linalg.solve(matriz, a - origen)
        if 0 < t < mejor and 0 <= s <= 1:
            mejor = t
    return mejor


# -------------------------------------------------------------------------
# Una pista cerrada con paredes, los puntos de control numerados en orden y el
# auto con sus sensores: tres que miden la distancia a la pared y uno que
# apunta al próximo punto de control.
# -------------------------------------------------------------------------
def togelius_pista():
    fig, ax = lienzo(10, 6.4)
    afuera = [(0, 0), (16, 0), (16, 10), (0, 10)]
    adentro = [(4, 3.5), (12, 3.5), (12, 6.5), (4, 6.5)]
    paredes = []
    for poligono in (afuera, adentro):
        ax.add_patch(Polygon(poligono, closed=True, fill=False, edgecolor=TINTA, lw=4))
        paredes += [(poligono[i], poligono[(i + 1) % 4]) for i in range(4)]

    # 1. Los puntos de control, en el orden en que hay que pasarlos
    puntos = [(8, 1.7), (14, 1.7), (14.2, 5), (14, 8.3), (8, 8.3), (2, 8.3), (1.8, 5), (2, 1.7)]
    for numero, punto in enumerate(puntos, start=1):
        ax.add_patch(Circle(punto, 0.32, color=AMARILLO, zorder=3))
        ax.text(punto[0], punto[1] + 0.55, str(numero), ha="center", va="bottom", fontsize=13, color=GRIS)

    # 2. El auto, que va hacia el punto 2
    posicion, rumbo = np.array([10.2, 1.4]), math.radians(8)
    u = np.array([math.cos(rumbo), math.sin(rumbo)])
    n = np.array([-u[1], u[0]])
    esquinas = [posicion + a * 0.6 * u + b * 0.3 * n for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
    ax.add_patch(Polygon(esquinas, closed=True, color=NA, zorder=5))

    # 3. Los sensores de pared: adelante, adelante a la izquierda y adelante a la derecha
    for angulo in (0, 40, -40):
        direccion = np.array([math.cos(rumbo + math.radians(angulo)), math.sin(rumbo + math.radians(angulo))])
        largo = distancia_a_pared(posicion, direccion, paredes, 7.0)
        ax.plot(*zip(posicion, posicion + largo * direccion), color=AZ, lw=2)
        ax.plot(*(posicion + largo * direccion), "o", color=AZ, ms=5)

    # 4. El sensor del punto de control: el ángulo entre el rumbo y el punto siguiente
    siguiente = np.array(puntos[1])
    ax.plot(*zip(posicion, siguiente), color=AMARILLO, lw=1.6, ls="--")

    ax.text(16.6, 7.6, "paredes", color=TINTA, fontsize=13, va="center")
    ax.text(16.6, 6.6, "puntos de control,\nen orden", color="#b08a10", fontsize=13, va="center")
    ax.text(16.6, 5.0, "sensores de pared", color=AZ, fontsize=13, va="center")
    ax.text(16.6, 4.0, "auto", color=NA, fontsize=13, va="center")
    ax.set_xlim(-0.5, 21.5)
    ax.set_ylim(-0.5, 10.5)
    guardar(fig, "togelius_pista.png")


# =========================================================================
# MONTANA Y DAVIS (1989), FIGS. 1 Y 2
# =========================================================================

# -------------------------------------------------------------------------
# Un número con coma decimal: 0.3 → "0,3".
# -------------------------------------------------------------------------
def coma(valor):
    return f"{valor:g}".replace(".", ",")


# -------------------------------------------------------------------------
# Una fila de casillas, una por gen, con el valor adentro.
#
# Recibe: los valores, dónde empieza (x, y), el color de cada casilla y un
#         título a la izquierda.
# -------------------------------------------------------------------------
def casillas(ax, valores, x, y, colores, titulo="", ancho=1.0):
    for i, (valor, color) in enumerate(zip(valores, colores)):
        ax.add_patch(Rectangle((x + i * ancho, y), ancho, 0.8, facecolor=color, edgecolor="white", lw=2))
        ax.text(x + i * ancho + ancho / 2, y + 0.4, coma(valor),
                ha="center", va="center", fontsize=13, color="white" if color != CALLE else TINTA)
    if titulo:
        ax.text(x - 0.3, y + 0.4, titulo, ha="right", va="center", fontsize=14)


# -------------------------------------------------------------------------
# Una red de 3 entradas, 2 neuronas ocultas y 1 salida, con sus pesos y sus
# umbrales, y abajo el cromosoma: la misma lista de números, en orden.
# -------------------------------------------------------------------------
def montana_red():
    fig, ax = lienzo(10, 6.2)
    entradas = [np.array([1.5, 0]), np.array([4, 0]), np.array([6.5, 0])]
    ocultas = [np.array([2.6, 2.6]), np.array([5.4, 2.6])]
    salida = np.array([4, 5])

    # 1. Los pesos: de cada entrada a cada oculta, y de cada oculta a la salida
    pesos_entrada = [[1.2, 0.8], [-0.3, -0.1], [0.7, -6.3]]
    for entrada, pesos in zip(entradas, pesos_entrada):
        for oculta, peso in zip(ocultas, pesos):
            flecha(ax, entrada + 0.45 * (oculta - entrada) / np.linalg.norm(oculta - entrada),
                   oculta - 0.45 * (oculta - entrada) / np.linalg.norm(oculta - entrada), lw=1.4)
            # Las flechas que se cruzan llevan el número cerca de la entrada; las otras, más arriba
            cruza = abs(oculta[0] - entrada[0]) > 2
            medio = entrada + (0.25 if cruza else 0.6) * (oculta - entrada)
            ax.text(*medio, coma(peso), fontsize=12, color=AZ, ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    for oculta, peso in zip(ocultas, (0.3, -0.4)):
        flecha(ax, oculta + 0.45 * (salida - oculta) / np.linalg.norm(salida - oculta),
               salida - 0.45 * (salida - oculta) / np.linalg.norm(salida - oculta), lw=1.4)
        medio = (oculta + salida) / 2
        ax.text(medio[0] + (-0.45 if oculta[0] < salida[0] else 0.45), medio[1], coma(peso),
                fontsize=12, color=NA, ha="center", va="center")

    # 2. Las neuronas, con su umbral adentro
    for centro, umbral in [(e, None) for e in entradas] + list(zip(ocultas, (1.7, -2.2))) + [(salida, 0.3)]:
        ax.add_patch(Circle(centro, 0.45, facecolor="white", edgecolor=TINTA, lw=1.8, zorder=3))
        if umbral is not None:
            ax.text(*centro, coma(umbral), ha="center", va="center", fontsize=12, zorder=4)

    # 3. El cromosoma, en el orden del paper
    cromosoma = [0.3, -0.4, 0.3, 1.2, 0.8, -0.3, -0.1, 0.7, -6.3, 1.7, -2.2]
    colores = [NA, NA, GRIS] + [AZ] * 6 + [GRIS] * 2
    casillas(ax, cromosoma, -1.5, -2.2, colores, ancho=1.0)
    ax.text(4, -2.7, "cromosoma: los 11 números de la red, uno detrás del otro", ha="center", va="top",
            fontsize=13, color=GRIS)
    ax.text(7.2, 4.6, "pesos a la salida", color=NA, fontsize=13)
    ax.text(7.2, 3.9, "pesos de entrada", color=AZ, fontsize=13)
    ax.text(7.2, 3.2, "umbrales", color=GRIS, fontsize=13)
    ax.set_xlim(-2, 10.5)
    ax.set_ylim(-3.4, 5.7)
    guardar(fig, "montana_red.png")


# -------------------------------------------------------------------------
# La mutación cambia algunos genes de un padre. El cruce arma un hijo con
# genes de dos padres, cada gen en su misma posición.
# -------------------------------------------------------------------------
def montana_operadores():
    fig, ax = lienzo(11, 5.6)
    padre = [0.3, -0.4, 1.2, 0.8, -0.3, -0.1, 0.7, -6.3]
    madre = [-0.7, -0.9, 1.3, 0.4, 1.8, 2.1, -0.2, -1.2]
    mutado = [0.3, -0.4, 1.2, -2.2, -0.3, -0.1, 1.2, -6.3]
    hijo = [-0.7, -0.4, 1.3, 0.4, 1.8, 2.1, 0.7, -6.3]

    # 1. Mutación: los genes que cambian, en naranja
    ax.text(-4.8, 7.0, "Mutación", fontsize=16, weight="bold", va="center")
    casillas(ax, padre, 0, 6.6, [AZ] * 8, "padre")
    casillas(ax, mutado, 0, 5.4, [NA if a != b else AZ for a, b in zip(padre, mutado)], "hijo")

    # 2. Cruce: cada gen del hijo viene de uno de los dos padres
    ax.text(-4.8, 3.4, "Cruce", fontsize=16, weight="bold", va="center")
    casillas(ax, padre, 0, 3.0, [AZ] * 8, "padre 1")
    casillas(ax, madre, 0, 1.8, [GRIS] * 8, "padre 2")
    casillas(ax, hijo, 0, 0.3, [AZ if a == b else GRIS for a, b in zip(padre, hijo)], "hijo")
    ax.set_xlim(-5, 8.5)
    ax.set_ylim(0, 7.8)
    guardar(fig, "montana_operadores.png")


# =========================================================================
# YAO (1999), FIGS. 3 Y 4
# =========================================================================

# -------------------------------------------------------------------------
# Una red de 2 entradas, 2 neuronas ocultas y 1 salida. `orden` dice cuál
# neurona va a la izquierda. Debajo, el cromosoma en decimal y en binario.
# -------------------------------------------------------------------------
def red_de_yao(ax, x0, orden, titulo):
    pesos = {"1": {"entrada": (4, 10), "salida": 7}, "2": {"entrada": (2, 0), "salida": 3}}
    colores = {"1": AZ, "2": NA}
    entradas = [np.array([x0 + 0.8, 0]), np.array([x0 + 3.2, 0])]
    ocultas = {nombre: np.array([x0 + 0.4 + 3.2 * i, 2.4]) for i, nombre in enumerate(orden)}
    salida = np.array([x0 + 2.0, 4.6])

    for nombre, oculta in ocultas.items():
        color = colores[nombre]
        for entrada, peso in zip(entradas, pesos[nombre]["entrada"]):
            if peso:
                ax.plot(*zip(entrada, oculta), color=color, lw=2)
                medio = entrada + 0.7 * (oculta - entrada)
                ax.text(*medio, str(peso), color=color, ha="center", va="center", fontsize=14,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=2)
        ax.plot(*zip(oculta, salida), color=color, lw=2)
        medio = (oculta + salida) / 2
        ax.text(medio[0] + (-0.3 if oculta[0] < salida[0] else 0.3), medio[1], str(pesos[nombre]["salida"]),
                color=color, ha="center", va="center", fontsize=14)
        ax.add_patch(Circle(oculta, 0.32, color=color, zorder=3))
        ax.text(*oculta, nombre, color="white", ha="center", va="center", fontsize=13, zorder=4)
    for punto in entradas + [salida]:
        ax.add_patch(Circle(punto, 0.18, color=TINTA, zorder=3))

    # El cromosoma: primero los pesos de entrada de cada neurona, después los de salida
    genes = [g for nombre in orden for g in pesos[nombre]["entrada"]] + [pesos[n]["salida"] for n in orden]
    duenos = [n for n in orden for _ in range(2)] + list(orden)
    for i, (gen, dueno) in enumerate(zip(genes, duenos)):
        ax.add_patch(Rectangle((x0 - 0.8 + i * 0.95, -1.6), 0.95, 0.7, facecolor=colores[dueno],
                               edgecolor="white", lw=2))
        ax.text(x0 - 0.8 + i * 0.95 + 0.475, -1.25, str(gen), color="white", ha="center", va="center",
                fontsize=13)
        ax.text(x0 - 0.8 + i * 0.95 + 0.475, -2.2, f"{gen:04b}", color=GRIS, ha="center", va="center",
                fontsize=10, family="monospace")
    ax.text(x0 + 2.0, 5.4, titulo, ha="center", fontsize=15)


# -------------------------------------------------------------------------
# Las dos redes de Yao: calculan lo mismo, pero con las neuronas ocultas en
# otro orden, así que sus cromosomas son distintos.
# -------------------------------------------------------------------------
def yao_permutacion():
    fig, ax = lienzo(11, 5.2)
    red_de_yao(ax, 0, "12", "Red A")
    red_de_yao(ax, 7.5, "21", "Red B")
    ax.text(5.8, 2.4, "=", fontsize=34, ha="center", va="center", color=GRIS)
    ax.set_xlim(-1.2, 13.2)
    ax.set_ylim(-2.6, 6.0)
    guardar(fig, "yao_permutacion.png")


# =========================================================================
# GOMEZ Y MIIKKULAINEN (1996), SECCIÓN 5.3
# =========================================================================

# -------------------------------------------------------------------------
# Una escalera de tareas, de la más fácil a la más difícil.
#
# Recibe: los textos de cada escalón, la altura donde empieza y un título.
# -------------------------------------------------------------------------
def escalera(ax, tareas, y0, titulo, color):
    ax.text(-0.3, y0 + 0.9, titulo, ha="right", va="center", fontsize=15, weight="bold")
    for i, tarea in enumerate(tareas):
        caja = FancyBboxPatch((i * 3.3, y0 + i * 0.45), 3.0, 1.0, boxstyle="round,pad=0.05,rounding_size=0.15",
                              facecolor=color, edgecolor="none")
        ax.add_patch(caja)
        ax.text(i * 3.3 + 1.5, y0 + i * 0.45 + 0.5, tarea, ha="center", va="center", color="white", fontsize=13)
        if i:
            flecha(ax, (i * 3.3 - 0.3, y0 + i * 0.45 + 0.3), (i * 3.3 - 0.02, y0 + i * 0.45 + 0.3), lw=1.4)


# -------------------------------------------------------------------------
# La evolución incremental de Gomez: la red aprende a escapar de un enemigo
# lento, después de dos, y los enemigos van cada vez más rápido. Abajo, las
# etapas de este trabajo, armadas con la misma idea.
# -------------------------------------------------------------------------
def gomez_etapas():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.axis("off")
    escalera(ax, ["1 enemigo\nvelocidad 0,3", "2 enemigos\nvelocidad 0,3", "2 enemigos\nvelocidad 0,4",
                  "2 enemigos\nvelocidad 0,5"], 3.2, "Gomez y\nMiikkulainen", GRIS)
    escalera(ax, ["manzana", "2×2", "3×3", "centro\n(solo prueba)"], 0, "Este trabajo", AZ)
    flecha(ax, (0, -0.6), (13, -0.6), color=GRIS, lw=1.2)
    ax.text(13.1, -0.6, "más difícil", color=GRIS, va="center", fontsize=13)
    ax.set_xlim(-4.2, 15.2)
    ax.set_ylim(-1.0, 5.8)
    guardar(fig, "gomez_etapas.png")


# =========================================================================
# NELSON ET AL. (2009), FIG. 1
# =========================================================================

# -------------------------------------------------------------------------
# Las clases de funciones de aptitud, según cuánto conocimiento previo del
# diseñador meten en la evolución. Los niveles son los de su tabla 1.
# La aptitud "a medida" mide cuánto avanzó en la tarea y además cómo se
# comportó: es la de este trabajo.
# -------------------------------------------------------------------------
def nelson_aptitud():
    clases = [("Agregada: solo si cumplió la tarea", 0, 0),
              ("Competitiva", 0, 2),
              ("Incremental por entorno", 2, 2),
              ("A medida: avance + comportamiento", 2, 2),
              ("Incremental por funciones", 2, 3),
              ("Por comportamiento", 3, 3),
              ("Con datos de entrenamiento", 4, 4)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for fila, (nombre, desde, hasta) in enumerate(clases):
        ax.barh(fila, hasta - desde + 0.8, left=desde - 0.4, height=0.65,
                color=AZ if nombre.startswith("A medida") else GRIS)
    ax.set_yticks(range(len(clases)), [nombre for nombre, _, _ in clases])
    ax.get_yticklabels()[3].set_color(AZ)
    ax.invert_yaxis()
    ax.set_xticks(range(5), ["muy bajo", "bajo", "moderado", "alto", "muy alto"])
    ax.set_xlabel("conocimiento previo que se le da a la evolución")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.5, 4.5)
    guardar(fig, "nelson_aptitud.png")


if __name__ == "__main__":
    kong_bicicleta()
    togelius_pista()
    montana_red()
    montana_operadores()
    yao_permutacion()
    gomez_etapas()
    nelson_aptitud()
