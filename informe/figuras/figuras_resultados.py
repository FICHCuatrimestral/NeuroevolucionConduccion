"""Números y figuras de la sección de resultados del informe.

Uso, parado en la carpeta del proyecto (hacen falta las corridas de resultados/comparaciones/
y, para los barrios, presentacion/img/barrios.json, que arma presentacion/video_comparaciones.py):
    python informe/figuras/figuras_resultados.py

Muestra los números de las tablas del informe y guarda dos figuras, en PDF y en PNG:
  - fig_entrenamiento   llegadas de la mejor red en las rutas fijas, generación por generación
  - fig_velocidad       velocidad y aceleración lateral a lo largo de una ruta del centro, con y sin agarre
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import beta as distribucion_beta

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# La carpeta del proyecto, para poder importar el paquete conduccion
RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from conduccion import config
from conduccion.aprendizaje.entrenamiento import rutas_de_referencia
from conduccion.aprendizaje.guardado import cargar
from conduccion.aprendizaje.perceptron_multicapa import Perceptron
from conduccion.mapa.mapa import santa_fe
from conduccion.mapa.ruta import distancias_a_giros
from conduccion.parametros import usar_parametros_de
from conduccion.simulacion.simulacion import Episodio

SALIDA = Path(__file__).resolve().parent
CARPETA = RAIZ / "resultados" / "comparaciones"
BARRIOS = RAIZ / "presentacion" / "img" / "barrios.json"

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
AZ, NA, TINTA, GRIS = "#2a78d6", "#eb6834", "#222222", "#8a8a85"

# Las seis corridas: carpeta, nombre y color
CORRIDAS = [("neurona_gaussiana_agarre", "Base (cruce por neurona)", "#2a78d6"),
            ("pesos_gaussiana_agarre", "Cruce por pesos", "#1baf7a"),
            ("blx_gaussiana_agarre", "Cruce BLX-α", "#8e5bd0"),
            ("ninguna_gaussiana_agarre", "Sin cruce", "#b8960a"),
            ("neurona_neurona_agarre", "Mutación por neurona", "#d6438a"),
            ("neurona_gaussiana_sin-agarre", "Sin límite de agarre", "#eb6834")]

# Rutas fijas de cada etapa: en la manzana son las 8 vueltas posibles
REFERENCIAS = {"manzana": 8, "2x2": 20, "3x3": 20}


# -------------------------------------------------------------------------
# Guarda la figura en PDF y en PNG, con el nombre dado, y la cierra.
# -------------------------------------------------------------------------
def guardar(fig, nombre):
    for extension in ("pdf", "png"):
        fig.savefig(SALIDA / f"{nombre}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", nombre)


# -------------------------------------------------------------------------
# Intervalo de Clopper-Pearson del 95 % para k llegadas en n rutas, en porcentaje.
# -------------------------------------------------------------------------
def intervalo(k, n):
    bajo = 0.0 if k == 0 else distribucion_beta.ppf(0.025, k, n - k + 1)
    alto = 1.0 if k == n else distribucion_beta.ppf(0.975, k + 1, n - k)
    return round(100 * bajo), round(100 * alto)


# -------------------------------------------------------------------------
# La carpeta de una corrida y lo que guardó: la evaluación en el centro y el
# identificador del último guardado de cada etapa.
# -------------------------------------------------------------------------
def corrida(nombre):
    carpeta = CARPETA / nombre / "semilla0"
    evaluacion = json.loads((carpeta / "evaluacion.json").read_text(encoding="utf-8"))
    etapas = json.loads((carpeta / "entrenamiento.json").read_text(encoding="utf-8"))["identificadores"]
    return carpeta, evaluacion, [str(carpeta / identificador) for identificador in etapas]


# -------------------------------------------------------------------------
# La historia completa de una corrida: las filas de todas sus etapas, en orden.
# -------------------------------------------------------------------------
def historia(etapas):
    filas = []
    for identificador in etapas:
        _, de_la_etapa, _ = cargar(identificador)
        filas += de_la_etapa
    return filas


# -------------------------------------------------------------------------
# Muestra los números de las tablas: el centro, las generaciones por etapa y los barrios.
# -------------------------------------------------------------------------
def tablas():
    print("\nCENTRO (20 rutas fijas, sin ruido)")
    for nombre, titulo, _ in CORRIDAS:
        _, evaluacion, etapas = corrida(nombre)
        rutas = evaluacion["rutas"]
        llegadas = [r for r in rutas if r["motivo"] == "llegada"]
        cerca = [r["velocidad_cerca"] for r in rutas if r["velocidad_cerca"] is not None]
        lejos = [r["velocidad_lejos"] for r in rutas if r["velocidad_lejos"] is not None]
        generaciones = {}
        for fila in historia(etapas):
            generaciones[fila["etapa"]] = generaciones.get(fila["etapa"], 0) + 1
        bajo, alto = intervalo(len(llegadas), len(rutas))
        print(f"{titulo:26} llega {len(llegadas):2d}/20 [{bajo}-{alto} %]  "
              f"choca {sum(r['motivo'] == 'choque' for r in rutas)}  sin tiempo {sum(r['motivo'] == 'tiempo' for r in rutas)}  "
              f"t medio {np.mean([r['tiempo'] for r in llegadas]):.0f} s  "
              f"v cerca/lejos {np.mean(cerca):.0f}/{np.mean(lejos):.0f} km/h  "
              f"a_y max {max(r['lateral_maxima'] for r in rutas):.1f}  "
              f"fuera {100 * np.mean([r['fuera_del_carril'] for r in rutas]):.1f} %  "
              f"generaciones {generaciones}")

    print("\nBARRIOS (20 rutas fijas de cada uno)")
    barrios = json.loads(BARRIOS.read_text(encoding="utf-8"))
    for nombre, titulo, _ in CORRIDAS:
        filas = barrios[nombre]["rutas"]
        llegadas = sum(f["motivo"] == "llegada" for f in filas)
        bajo, alto = intervalo(llegadas, len(filas))
        print(f"{titulo:26} {barrios[nombre]['barrio']:14} llega {llegadas:2d}/20 [{bajo}-{alto} %]  "
              f"choca {sum(f['motivo'] == 'choque' for f in filas)}  sin tiempo {sum(f['motivo'] == 'tiempo' for f in filas)}")


# -------------------------------------------------------------------------
# Fracción de las rutas fijas de su etapa en la que llega la mejor red guardada,
# generación por generación, para las seis corridas. Las líneas verticales
# finas marcan, para cada corrida, el paso a la etapa siguiente.
# -------------------------------------------------------------------------
def fig_entrenamiento():
    fig, ax = plt.subplots(figsize=(ANCHO, 2.1))
    for nombre, titulo, color in CORRIDAS:
        _, _, etapas = corrida(nombre)
        filas = historia(etapas)
        fraccion = [fila["llegadas_referencia"] / REFERENCIAS[fila["etapa"]] for fila in filas]
        ax.plot(range(1, len(filas) + 1), fraccion, color=color, lw=1, label=titulo)
        for g in range(1, len(filas)):
            if filas[g]["etapa"] != filas[g - 1]["etapa"]:
                ax.axvline(g + 0.5, color=color, lw=0.5, ls=":", alpha=0.8)
    ax.set_xlabel("generación")
    ax.set_ylabel("llegadas de la mejor\n(fracción de las rutas fijas)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    guardar(fig, "fig_entrenamiento")


# -------------------------------------------------------------------------
# Una red guardada maneja una ruta sin ruido, con la física con la que se entrenó.
# Devuelve: metros recorridos, velocidad (km/h) y aceleración lateral (m/s²) en
# cada paso, y cómo terminó.
# -------------------------------------------------------------------------
def manejar(identificador, mapa, ruta):
    usar_parametros_de(identificador)
    cromosoma, _, _ = cargar(identificador)
    red = Perceptron(config.CAPAS, cromosoma=cromosoma)
    episodio = Episodio(mapa, ruta)
    metros, velocidad, lateral = [], [], []
    recorrido = 0.0
    while not episodio.terminado():
        x, y = episodio.auto.x, episodio.auto.y
        episodio.paso_con_red(red)
        auto = episodio.auto
        recorrido += math.dist((x, y), (auto.x, auto.y))
        beta = math.atan(config.L_R / (config.L_F + config.L_R) * math.tan(auto.volante))
        metros.append(recorrido)
        velocidad.append(3.6 * auto.velocidad)
        lateral.append(auto.velocidad ** 2 * abs(math.sin(beta)) / config.L_R)
    return metros, velocidad, lateral, episodio.motivo


# -------------------------------------------------------------------------
# Velocidad y aceleración lateral a lo largo de una ruta del centro, para la
# base (con agarre) y la red sin límite. La ruta es, de las 20 fijas del centro,
# la de más giros a la que llegan las dos.
# -------------------------------------------------------------------------
def fig_velocidad():
    centro = santa_fe("centro")
    rutas = rutas_de_referencia(centro, 0)
    redes = {}
    for nombre in ("neurona_gaussiana_agarre", "neurona_gaussiana_sin-agarre"):
        _, evaluacion, etapas = corrida(nombre)
        redes[nombre] = (evaluacion["rutas"], etapas[-1])
    ambas = [i for i in range(len(rutas))
             if all(redes[nombre][0][i]["motivo"] == "llegada" for nombre in redes)]
    numero = max(ambas, key=lambda i: (len(distancias_a_giros(rutas[i].carril)), rutas[i].largo))
    ruta = rutas[numero]
    giros = distancias_a_giros(ruta.carril)

    fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(ANCHO, 3.0), sharex=True)
    for nombre, titulo, color in (("neurona_gaussiana_sin-agarre", "Sin límite de agarre", NA),
                                  ("neurona_gaussiana_agarre", "Con límite de 6 m/s² (base)", AZ)):
        metros, velocidad, lateral, motivo = manejar(redes[nombre][1], centro, ruta)
        arriba.plot(metros, velocidad, color=color, label=f"{titulo}: {len(metros) * config.DT:.0f} s")
        abajo.plot(metros, lateral, color=color, lw=0.9)
        print(f"velocidad: ruta {numero}, {titulo}, {motivo}, {len(metros) * config.DT:.0f} s")
    for ax in (arriba, abajo):
        for d in giros:
            ax.axvline(d, color=GRIS, lw=0.5, ls=":")
    abajo.axhline(6, color=TINTA, lw=0.6, ls="--")
    abajo.text(ruta.largo, 6.6, "6 m/s²", ha="right", va="bottom", fontsize=7)
    arriba.set_ylabel("velocidad (km/h)")
    abajo.set_ylabel("aceleración\nlateral (m/s²)")
    abajo.set_xlabel("metros recorridos (líneas punteadas: giros)")
    arriba.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    guardar(fig, "fig_velocidad")


if __name__ == "__main__":
    tablas()
    fig_entrenamiento()
    fig_velocidad()
