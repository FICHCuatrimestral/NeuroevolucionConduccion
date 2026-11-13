"""Video de las seis redes de las comparaciones, cada una manejando en un barrio distinto.

Uso, parado en la carpeta del proyecto (hacen falta las corridas de resultados/comparaciones/):
    python presentacion/video_comparaciones.py

Ningún barrio se usó para entrenar ni para la prueba del centro. Para cada red:
  1. Maneja, sin ruido en los sensores, las 20 rutas fijas de su barrio, con la física con la que
     se entrenó (con o sin límite de agarre). Los resultados se guardan en img/barrios.json.
  2. Se elige, de las rutas a las que llega, la que tiene más giros sin pasar de TIEMPO_MAXIMO.
El video muestra las seis a la vez, en una grilla de 2 × 3, y debajo de cada una cuántas de las
20 rutas de su barrio termina. Sale img/comparaciones.mp4. Hace falta ffmpeg (por ejemplo, el del
paquete imageio-ffmpeg).
"""
import json
import sys
from pathlib import Path

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
from conduccion.mapa.ruta import distancias_a_giros
from figuras_presentacion import SALIDA, dibujar_calles, manejar

CARPETA = RAIZ / "resultados" / "comparaciones"
RESUMEN = SALIDA / "barrios.json"

# Cada corrida: su carpeta, su nombre en el video, su color, el barrio y su nombre
CORRIDAS = [("neurona_gaussiana_agarre", "Base: cruce por neurona", "#2a78d6", "guadalupe", "Guadalupe"),
            ("pesos_gaussiana_agarre", "Cruce por pesos", "#1baf7a", "n2_02_roma", "Roma"),
            ("blx_gaussiana_agarre", "Cruce BLX-α", "#8e5bd0", "n3_09_schneider", "Schneider"),
            ("ninguna_gaussiana_agarre", "Sin cruce", "#b8960a", "n1_02_12_de_octubre", "12 de Octubre"),
            ("neurona_neurona_agarre", "Mutación por neurona", "#d6438a", "n4_14_la_guardia", "La Guardia"),
            ("neurona_gaussiana_sin-agarre", "Sin límite de agarre", "#eb6834", "n4_02_arenales", "Arenales")]

# La ruta que se muestra no dura más que esto, para que el video no sea largo
TIEMPO_MAXIMO = 120.0

# Cuántas veces más rápido que el tiempo real va el video
VELOCIDAD_VIDEO = 4

# Metros que se ven a cada lado del auto: la cámara lo sigue
VENTANA = 60.0


# -------------------------------------------------------------------------
# La carpeta de la red final de una corrida, la que se probó en el centro.
# -------------------------------------------------------------------------
def red_de(corrida):
    carpeta = CARPETA / corrida / "semilla0"
    evaluacion = json.loads((carpeta / "evaluacion.json").read_text(encoding="utf-8"))
    return str(carpeta / evaluacion["red"])


# -------------------------------------------------------------------------
# La red maneja las 20 rutas fijas de su barrio.
# Devuelve: una fila por ruta, con cómo terminó, cuánto tardó y cuántas esquinas tiene.
# -------------------------------------------------------------------------
def probar_en_barrio(red, mapa):
    filas = []
    for numero, ruta in enumerate(rutas_de_referencia(mapa, 0)):
        pasos, motivo = manejar(red, mapa, ruta)
        filas.append({"ruta": numero, "motivo": motivo, "tiempo": round(len(pasos) * config.DT, 1),
                      "esquinas": len(ruta.esquinas)})
    return filas


# -------------------------------------------------------------------------
# Las rutas de cada red en su barrio. Si ya están en RESUMEN, se leen de ahí.
# -------------------------------------------------------------------------
def resumen_de_barrios(mapas):
    resumen = json.loads(RESUMEN.read_text(encoding="utf-8")) if RESUMEN.exists() else {}
    for corrida, nombre, _, zona, barrio in CORRIDAS:
        if corrida not in resumen:
            filas = probar_en_barrio(red_de(corrida), mapas[zona])
            resumen[corrida] = {"barrio": barrio, "zona": zona, "rutas": filas}
            RESUMEN.write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
        filas = resumen[corrida]["rutas"]
        llegadas = sum(fila["motivo"] == "llegada" for fila in filas)
        choques = sum(fila["motivo"] == "choque" for fila in filas)
        print(f"{nombre} en {barrio}: llega en {llegadas} de 20, choca en {choques}", flush=True)
    return resumen


# -------------------------------------------------------------------------
# De las rutas a las que llegó, la más larga que no pase de TIEMPO_MAXIMO; si
# dos miden parecido, la de más giros. Si ninguna entra en ese tiempo, la más
# corta de las que llegó.
#
# Recibe: las filas de la red en su barrio y las 20 rutas del barrio.
# -------------------------------------------------------------------------
def ruta_para_mostrar(filas, rutas):
    llegadas = [fila for fila in filas if fila["motivo"] == "llegada"]
    cortas = [fila for fila in llegadas if fila["tiempo"] <= TIEMPO_MAXIMO]
    if cortas:
        return max(cortas, key=lambda fila: (round(rutas[fila["ruta"]].largo, -1),
                                             len(distancias_a_giros(rutas[fila["ruta"]].carril))))["ruta"]
    return min(llegadas, key=lambda fila: fila["tiempo"])["ruta"]


# -------------------------------------------------------------------------
# El texto de abajo de cada panel: cómo le fue a la red en las 20 rutas de su barrio.
# -------------------------------------------------------------------------
def balance(filas):
    llegadas = sum(fila["motivo"] == "llegada" for fila in filas)
    choques = sum(fila["motivo"] == "choque" for fila in filas)
    tiempo = sum(fila["motivo"] == "tiempo" for fila in filas)
    texto = f"llega en {llegadas} de 20 rutas"
    if choques:
        texto += f" · choca en {choques}"
    if tiempo:
        texto += f" · {tiempo} sin tiempo"
    return texto


# -------------------------------------------------------------------------
# Arma el video: una grilla de 2 × 3 con cada red manejando su ruta en su barrio.
# La cámara de cada panel sigue a su auto. Cuando un auto llega, queda quieto
# hasta que terminan todos.
#
# Recibe: una lista de (nombre, color, barrio, texto de abajo, mapa, ruta, pasos).
# -------------------------------------------------------------------------
def video(paneles):
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg()
    fig, ejes = plt.subplots(2, 3, figsize=(12.8, 7.2), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.06, wspace=0.08, hspace=0.32)
    autos, rastros = [], []
    for ax, (nombre, color, barrio, texto, mapa, ruta, pasos) in zip(ejes.flat, paneles):
        dibujar_calles(ax, mapa)
        ax.plot(*ruta.carril.xy, color="#8a8a85", lw=1.2, ls="--", zorder=1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{nombre} · {barrio}", fontsize=13, color=color, fontweight="bold", pad=6)
        ax.text(0.5, -0.03, texto, transform=ax.transAxes, ha="center", va="top", fontsize=12, color="#222222")
        rastro, = ax.plot([], [], color=color, lw=3, zorder=2)
        auto = Polygon(pasos[0]["esquinas"], fc=color, ec="#222222", lw=1, zorder=4)
        ax.add_patch(auto)
        autos.append(auto)
        rastros.append(rastro)

    cuadros = list(range(0, max(len(panel[-1]) for panel in paneles), 2))
    cuadros += [cuadros[-1]] * 15

    # ---------------------------------------------------------------------
    # Dibuja el cuadro del paso i: cada auto en su paso i, o en su último paso
    # si ya llegó, con su rastro, y la cámara centrada en él.
    # ---------------------------------------------------------------------
    def actualizar(i):
        for ax, panel, auto, rastro in zip(ejes.flat, paneles, autos, rastros):
            pasos = panel[-1]
            k = min(i, len(pasos) - 1)
            auto.set_xy(pasos[k]["esquinas"])
            rastro.set_data([p["x"] for p in pasos[:k + 1]], [p["y"] for p in pasos[:k + 1]])
            ax.set_xlim(pasos[k]["x"] - VENTANA, pasos[k]["x"] + VENTANA)
            ax.set_ylim(pasos[k]["y"] - VENTANA, pasos[k]["y"] + VENTANA)
        return autos + rastros

    animacion = FuncAnimation(fig, actualizar, frames=cuadros, blit=False)
    animacion.save(SALIDA / "comparaciones.mp4", writer=FFMpegWriter(fps=VELOCIDAD_VIDEO / (2 * config.DT), bitrate=1800,
                   extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print("-> comparaciones.mp4", len(cuadros), "cuadros")


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


if __name__ == "__main__":
    mapas = {zona: santa_fe(zona) for _, _, _, zona, _ in CORRIDAS}
    resumen = resumen_de_barrios(mapas)
    paneles = []
    for corrida, nombre, color, zona, barrio in CORRIDAS:
        filas = resumen[corrida]["rutas"]
        rutas = rutas_de_referencia(mapas[zona], 0)
        numero = ruta_para_mostrar(filas, rutas)
        ruta = rutas[numero]
        pasos, motivo = manejar(red_de(corrida), mapas[zona], ruta)
        print(f"{nombre}: ruta {numero}, {motivo}, {len(pasos) * config.DT:.0f} s")
        paneles.append((nombre, color, barrio, balance(filas), mapas[zona], ruta, pasos))
    video(paneles)
