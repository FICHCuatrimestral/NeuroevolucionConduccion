"""Video del entrenamiento: los 50 autos de una generación salen juntos, etapa por etapa.

Uso, parado en la carpeta del proyecto:
    python presentacion/video_entrenamiento.py grabar   (repite la corrida base y guarda cada generación; tarda horas)
    python presentacion/video_entrenamiento.py video    (arma img/entrenamiento.mp4 con lo grabado)

1. grabar: entrena igual que la corrida base de las comparaciones (cruce por neurona, mutación
   gaussiana, límite de agarre, semilla 0, población de 50, manzana → 2x2 → 3x3). Con la misma
   semilla, el entrenamiento es el mismo: al terminar se compara la mejor red con la del informe.
   Cada generación se guarda en cache/video_entrenamiento/generaciones.npz: la población que se
   evaluó, su fitness y la etapa.
2. video: para algunas generaciones de cada etapa, los 50 autos manejan juntos la misma ruta, sin
   ruido en los sensores. En gris, la población; en azul, el de mejor fitness de esa generación.
   Los que chocan quedan en rojo claro. Al costado, la etapa, la generación y cuántos llegan.
   Hace falta ffmpeg (por ejemplo, el del paquete imageio-ffmpeg).
"""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.patches import Polygon

# La carpeta del proyecto, para poder importar el paquete conduccion
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conduccion import config
from conduccion.parametros import usar_parametros, volver_a_config
from conduccion.aprendizaje.comparaciones import BASE, ETAPAS, POBLACION, parametros_de
from conduccion.aprendizaje.etapas import entrenar_por_etapas, etapas_santa_fe
from conduccion.aprendizaje.fitness import nota
from conduccion.aprendizaje.guardado import usar_carpeta
from conduccion.aprendizaje.perceptron_multicapa import Perceptron
from conduccion.simulacion.simulacion import Episodio, correr_episodio
from figuras_presentacion import SALIDA, dibujar_calles, ffmpeg

CACHE = Path(__file__).resolve().parent / "cache" / "video_entrenamiento"
GRABACION = CACHE / "generaciones.npz"
SEMILLA = 0

# La mejor red de la corrida base del informe, para comprobar que el entrenamiento es el mismo
BASE_INFORME = RAIZ / "resultados" / "comparaciones" / "neurona_gaussiana_agarre" / "semilla0"

# Nombres de las etapas en el video
NOMBRES = {"manzana": "Manzana", "2x2": "2×2 manzanas", "3x3": "3×3 manzanas"}

# Qué generaciones se muestran de cada etapa, como fracción de sus generaciones (0 = la primera, 1 = la última)
MOMENTOS = {"manzana": [0, 0.3, 1], "2x2": [0, 1], "3x3": [0, 1]}
# Rutas de referencia que se muestran en cada momento: las vueltas a la manzana, y en los
# otros mapas las más largas y con más giros, una distinta en cada momento
RUTAS = {"manzana": [0, 0, 4], "2x2": [14, 16], "3x3": [9, 4]}

# Cada tramo dura como mucho estos cuadros, más una pausa al final
CUADROS_POR_SEGUNDO = 25
CUADROS_POR_TRAMO = 140
PAUSA = 20

AZ, ROJO, TINTA, GRIS = "#2a78d6", "#d6402a", "#222222", "#8a8a85"
# Los autos se dibujan más grandes que en la realidad (miden 4 m en cuadras de 100 m),
# para que se vean en la diapositiva: tanto más cuanto más mapa entra en el cuadro.
# Solo cambia el dibujo, no la simulación.
AGRANDAR_CADA_100_M = 1.7


def agrandar(esquinas, factor):
    esquinas = np.asarray(esquinas, dtype=float)
    centro = esquinas.mean(axis=0)
    return centro + factor * (esquinas - centro)
plt.rcParams.update({"font.family": "sans-serif", "font.size": 15})


# =========================================================================
# 1. GRABAR
# =========================================================================

# -------------------------------------------------------------------------
# Deja config como en la corrida base.
# -------------------------------------------------------------------------
def configurar():
    volver_a_config()
    usar_parametros(**parametros_de(BASE))


# -------------------------------------------------------------------------
# Repite la corrida base y guarda la población de cada generación.
#
# La evaluación es la misma que hace evaluar() sin ventana, en el mismo orden,
# así el azar se usa igual y el entrenamiento da lo mismo que el del informe.
# -------------------------------------------------------------------------
def grabar():
    configurar()
    CACHE.mkdir(parents=True, exist_ok=True)
    usar_carpeta(CACHE / "etapas")
    etapas = etapas_santa_fe(SEMILLA, ETAPAS)
    etapa_de_mapa = {id(etapa.mapa): etapa.nombre for etapa in etapas}
    poblaciones, fitness, nombres = [], [], []

    # Evalúa la generación como evaluar() y la anota
    def evaluar_y_anotar(poblacion, casos, rng):
        resultados_poblacion = []
        for cromosoma in poblacion:
            red = Perceptron(config.CAPAS, cromosoma=cromosoma)
            resultados_poblacion.append([correr_episodio(mapa, ruta, red, rng=rng) for mapa, ruta in casos])
        poblaciones.append(np.array(poblacion))
        fitness.append([nota(resultados) for resultados in resultados_poblacion])
        nombres.append(etapa_de_mapa[id(casos[0][0])])
        print(f"generación {len(nombres)} ({nombres[-1]}): mejor fitness {max(fitness[-1]):.3f}", flush=True)
        return resultados_poblacion

    mejor, _ = entrenar_por_etapas(etapas, semilla=SEMILLA, tamano_poblacion=POBLACION, tipo_cruce=BASE["cruce"],
                                   tipo_mutacion=BASE["mutacion"], mostrar=lambda *a: None,
                                   simular_poblacion=evaluar_y_anotar)
    np.savez_compressed(GRABACION, poblaciones=np.array(poblaciones), fitness=np.array(fitness),
                        etapas=np.array(nombres))
    print("guardado", GRABACION, len(nombres), "generaciones")

    # ¿Es el mismo entrenamiento que el del informe?
    evaluacion = json.loads((BASE_INFORME / "evaluacion.json").read_text(encoding="utf-8"))
    del_informe = np.load(BASE_INFORME / evaluacion["red"] / "mejor.npy")
    igual = np.allclose(mejor, del_informe)
    print("la mejor red es la misma que la del informe:", "sí" if igual else "NO")


# =========================================================================
# 2. VIDEO
# =========================================================================

# -------------------------------------------------------------------------
# La ruta que manejan todos en una etapa. En la manzana, una vuelta a la
# izquierda; en las demás, la ruta fija con más giros de hasta 300 m.
# -------------------------------------------------------------------------
def ruta_de(etapa, orden):
    elegidas = RUTAS[etapa.nombre]
    return etapa.referencia[elegidas[orden % len(elegidas)]]


# -------------------------------------------------------------------------
# Los 50 autos manejan juntos la ruta, paso a paso.
#
# Devuelve: por paso, la lista de esquinas de cada auto, y por auto, su recorrido
#           y cómo terminó (o None si seguía cuando se cortó).
# -------------------------------------------------------------------------
def manejar_todos(poblacion, mapa, ruta):
    episodios = [Episodio(mapa, ruta) for _ in poblacion]
    redes = [Perceptron(config.CAPAS, cromosoma=cromosoma) for cromosoma in poblacion]
    pasos = [[e.auto.esquinas() for e in episodios]]
    recorridos = [[(e.auto.x, e.auto.y)] for e in episodios]
    while not all(e.terminado() for e in episodios):
        for i, (episodio, red) in enumerate(zip(episodios, redes)):
            if not episodio.terminado():
                episodio.paso_con_red(red)
                recorridos[i].append((episodio.auto.x, episodio.auto.y))
        pasos.append([e.auto.esquinas() for e in episodios])
    motivos = [e.motivo for e in episodios]
    # En qué paso terminó cada uno
    fin = [len(r) - 1 for r in recorridos]
    return pasos, recorridos, motivos, fin


# -------------------------------------------------------------------------
# Los tramos del video: (etapa, número de etapa, generación dentro de la etapa,
# total de generaciones de la etapa, índice en la grabación).
# -------------------------------------------------------------------------
def tramos(nombres_grabados):
    salida = []
    orden = list(dict.fromkeys(nombres_grabados))
    for numero, nombre in enumerate(orden):
        indices = [i for i, n in enumerate(nombres_grabados) if n == nombre]
        for fraccion in MOMENTOS[nombre]:
            k = round(fraccion * (len(indices) - 1))
            salida.append((nombre, numero, k, len(indices), indices[k]))
    return salida


def video():
    configurar()
    grabado = np.load(GRABACION)
    poblaciones, fitness, nombres = grabado["poblaciones"], grabado["fitness"], list(grabado["etapas"])
    etapas = {etapa.nombre: etapa for etapa in etapas_santa_fe(SEMILLA, ETAPAS)}
    cantidad_etapas = len(dict.fromkeys(nombres))

    # Simular todos los tramos antes de dibujar
    datos = []
    vistos = {}
    for nombre, numero, k, total, indice in tramos(nombres):
        etapa = etapas[nombre]
        orden = vistos.get(nombre, 0)
        vistos[nombre] = orden + 1
        ruta = ruta_de(etapa, orden)
        pasos, recorridos, motivos, fin = manejar_todos(poblaciones[indice], etapa.mapa, ruta)
        mejor = int(np.argmax(fitness[indice]))
        salto = max(1, math.ceil(len(pasos) / CUADROS_POR_TRAMO))
        cuadros = list(range(0, len(pasos), salto)) + [len(pasos) - 1] * PAUSA
        datos.append(dict(etapa=etapa, ruta=ruta, numero=numero, k=k, total=total, pasos=pasos,
                          recorridos=recorridos, motivos=motivos, fin=fin, mejor=mejor, cuadros=cuadros))
        llegan = sum(m == "llegada" for m in motivos)
        print(f"{nombre} generación {k + 1}/{total}: llegan {llegan} de {len(motivos)}, {len(cuadros)} cuadros")

    # Un cuadro del video = (tramo, paso)
    secuencia = [(t, paso) for t, d in enumerate(datos) for paso in d["cuadros"]]

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes((0.0, 0.0, 0.6, 1.0))
    panel = fig.add_axes((0.6, 0.0, 0.4, 1.0))
    panel.axis("off")
    titulo = panel.text(0.05, 0.86, "", fontsize=30, fontweight="bold", va="top")
    subtitulo = panel.text(0.05, 0.72, "", fontsize=24, va="top")
    cuenta = panel.text(0.05, 0.52, "", fontsize=22, va="top", linespacing=1.6)
    panel.text(0.05, 0.16, "■ el mejor de la generación", color=AZ, fontsize=16)
    panel.text(0.05, 0.10, "■ el resto de la población", color=GRIS, fontsize=16)
    panel.text(0.05, 0.04, "■ chocó o se salió", color=ROJO, fontsize=16)

    estado = {"tramo": None, "autos": [], "rastro": None}

    # Prepara el mapa y los autos de un tramo
    def preparar(d):
        ax.clear()
        if d["etapa"].nombre == "manzana":
            # Solo la cuadra de la vuelta: se recortan las calles alrededor de la ruta
            solo_la_cuadra = SimpleNamespace(calles=d["etapa"].mapa.calles.intersection(d["ruta"].carril.buffer(9)))
            dibujar_calles(ax, solo_la_cuadra)
        else:
            dibujar_calles(ax, d["etapa"].mapa)
        ax.plot(*d["ruta"].carril.xy, color=GRIS, lw=1.5, ls="--", zorder=1)
        xs, ys = d["ruta"].carril.xy
        ax.plot(xs[0], ys[0], "o", color=TINTA, ms=9, zorder=2)
        ax.plot(xs[-1], ys[-1], "s", color=TINTA, ms=9, zorder=2)
        # En la manzana, zoom sobre la cuadra; en los otros, el mapa entero.
        # Con margen, y estirado a la forma del recuadro
        if d["etapa"].nombre == "manzana":
            x0, y0, x1, y1 = d["ruta"].carril.bounds
            margen = 25
        else:
            x0, y0, x1, y1 = d["etapa"].mapa.calles.bounds
            margen = 8
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ancho, alto = x1 - x0 + 2 * margen, y1 - y0 + 2 * margen
        forma = (0.6 * 12.8) / 7.2
        ancho, alto = max(ancho, alto * forma), max(alto, ancho / forma)
        ax.set_xlim(cx - ancho / 2, cx + ancho / 2)
        ax.set_ylim(cy - alto / 2, cy + alto / 2)
        ax.set_aspect("equal")
        estado["factor"] = AGRANDAR_CADA_100_M * alto / 100
        ax.axis("off")
        autos = []
        for i in range(len(d["motivos"])):
            es_mejor = i == d["mejor"]
            autos.append(Polygon(agrandar(d["pasos"][0][i], estado["factor"]), closed=True, zorder=5 if es_mejor else 3,
                                 fc=AZ if es_mejor else GRIS, ec=TINTA, lw=1.2 if es_mejor else 0.5,
                                 alpha=1.0 if es_mejor else 0.8))
            ax.add_patch(autos[-1])
        rastro, = ax.plot([], [], color=AZ, lw=2.5, zorder=4)
        estado.update(autos=autos, rastro=rastro)
        titulo.set_text(f"Etapa {d['numero'] + 1} de {cantidad_etapas}")
        subtitulo.set_text(f"{NOMBRES[d['etapa'].nombre]}\ngeneración {d['k'] + 1} de {d['total']}")

    # Dibuja un cuadro
    def actualizar(i):
        t, paso = secuencia[i]
        d = datos[t]
        if estado["tramo"] != t:
            preparar(d)
            estado["tramo"] = t
        for j, auto in enumerate(estado["autos"]):
            auto.set_xy(agrandar(d["pasos"][paso][j], estado["factor"]))
            if paso >= d["fin"][j] and d["motivos"][j] != "llegada" and j != d["mejor"]:
                auto.set_facecolor(ROJO)
                auto.set_zorder(4)
        recorrido = d["recorridos"][d["mejor"]][:paso + 1]
        estado["rastro"].set_data([p[0] for p in recorrido], [p[1] for p in recorrido])
        llegaron = sum(1 for j, m in enumerate(d["motivos"]) if m == "llegada" and paso >= d["fin"][j])
        chocaron = sum(1 for j, m in enumerate(d["motivos"]) if m not in ("llegada", None) and paso >= d["fin"][j]
                       and m != "tiempo")
        cuenta.set_text(f"llegaron: {llegaron} de {len(d['motivos'])}\nchocaron o se salieron: {chocaron}")
        return []

    animacion = FuncAnimation(fig, actualizar, frames=len(secuencia), blit=False)
    animacion.save(SALIDA / "entrenamiento.mp4",
                   writer=FFMpegWriter(fps=CUADROS_POR_SEGUNDO, bitrate=1600, extra_args=["-pix_fmt", "yuv420p"]))
    plt.close(fig)
    print("-> entrenamiento.mp4", len(secuencia), "cuadros,", round(len(secuencia) / CUADROS_POR_SEGUNDO), "s")


if __name__ == "__main__":
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg()
    {"grabar": grabar, "video": video}[sys.argv[1] if len(sys.argv) > 1 else "video"]()
