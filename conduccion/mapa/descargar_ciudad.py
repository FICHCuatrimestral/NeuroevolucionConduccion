"""Baja la ciudad entera de Santa Fe y la parte en recortes de 1 km, ordenados de lo simple a lo complejo.

Se corre desde la carpeta del proyecto: python -m conduccion.mapa.descargar_ciudad
Tarda unos 20 minutos. Necesita internet solo la primera vez: osmnx guarda lo que baja en cache/.

Guarda en mapas/:
  - ciudad_santa_fe.graphml: toda la ciudad (16 MB, no está en el repositorio). No se abre con
    santa_fe(): la matriz de toda la ciudad no entra en memoria.
  - santa_fe_n1_01_<barrio>.graphml, ...: cada recorte que sirve. Se abre con santa_fe("n1_01_<barrio>").
  - recortes.json: las medidas, el nivel y el uso de cada recorte.

Diferencias con descargar_mapa.py:
  1. Se saca la autopista: el auto no pasa de 40 km/h. Las bajadas (los *_link) se dejan,
     porque sin ellas Colastiné y La Guardia quedan desconectados del resto.
  2. Los cruces se juntan a 3 m y no a 10 m. Una avenida con cantero, como Blas Parera, son dos calzadas.
     Con 10 m, sus dos esquinas se juntan en una sola, en el medio del cantero, y cada cuadra hace un
     diente para llegar a ella. Ahí el carril se parte: en algunos recortes, en el 40 % de las rutas.
     Con 3 m, cada calzada conserva sus esquinas y el carril se parte en el 1 %.

Nivel de cada recorte, de 1 a 4:
  - Por la entropía de orientación de las calles (Boeing, 2019): cuánto varía hacia dónde apuntan.
    Un damero perfecto da 1,39 (cuatro direcciones). Con calles en todas las direcciones, hasta 3,58.
  - Un recorte con rotondas va al nivel 4.

En cada nivel, uno de cada cinco recortes queda para prueba: la red nunca entrena ahí.
"""

import json
import unicodedata

import numpy as np
import osmnx as ox
from shapely import affinity
from shapely.geometry import Point, box

from .descargar_mapa import GIRO_CALLES, doble_mano
from .mapa import Mapa, archivo_santa_fe
from .ruta import calcular_ruta, ruta_sirve

# Relación de OpenStreetMap del municipio de Santa Fe. Buscándolo por nombre, a veces da Rosario.
CIUDAD = "R3498629"

# Tipos de calle que se sacan: la autopista
SIN_AUTOS = {"motorway", "motorway_link"}

# Los nodos más cerca que esto, en metros, forman un solo cruce
TOLERANCIA_CRUCES = 3

# Datos de cada cuadra que se guardan, además de nombre, tipo, mano, carriles y velocidad máxima
DATOS_EXTRA = ["surface", "junction", "lit"]

# Metros de lado de cada recorte, como el mapa centro
LADO = 1000

# Un recorte con menos kilómetros de calle que esto es casi todo río o campo
MINIMO_KM = 8

# Para ver si un recorte sirve, se sortean estas rutas y tiene que servir esta fracción
RUTAS_DE_MUESTRA = 40
MINIMO_RUTAS = 0.8

# Cortes de entropía entre niveles: hasta 2,0 es nivel 1; hasta 2,25, nivel 2; hasta 2,5, nivel 3
ENTROPIA_NIVELES = [2.0, 2.25, 2.5]

# En cada nivel, uno de cada tantos recortes queda para prueba
UNO_DE_PRUEBA_CADA = 5

CARPETA = archivo_santa_fe("x").parent


# -------------------------------------------------------------------------
# La ciudad entera, en metros, sin autopista, con los cruces juntos y todas
# las calles de doble mano.
# -------------------------------------------------------------------------
def bajar_ciudad():
    ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + DATOS_EXTRA
    limite = ox.geocode_to_gdf(CIUDAD, by_osmid=True).geometry.iloc[0]
    grafo = ox.graph_from_polygon(limite, network_type="drive")

    # 1. Sin autopista. Cuando una cuadra une dos calles, osmnx da una lista de tipos.
    autopista = []
    for u, v, clave, datos in grafo.edges(keys=True, data=True):
        tipos = datos["highway"] if isinstance(datos["highway"], list) else [datos["highway"]]
        if SIN_AUTOS.intersection(tipos):
            autopista.append((u, v, clave))
    grafo.remove_edges_from(autopista)
    grafo = ox.truncate.largest_component(grafo)

    # 2. De latitud y longitud a metros, y cada cruce partido en un solo nodo
    grafo = ox.project_graph(grafo)
    grafo = ox.simplification.consolidate_intersections(grafo, tolerance=TOLERANCIA_CRUCES)

    # 3. Doble mano, como en los otros mapas. El sentido real queda guardado en "oneway".
    doble_mano(grafo)
    return ox.truncate.largest_component(grafo, strongly=True)


# -------------------------------------------------------------------------
# Los barrios de OpenStreetMap, como puntos en metros. Sirven para ponerle
# nombre a cada recorte.
# -------------------------------------------------------------------------
def barrios(crs):
    limite = ox.geocode_to_gdf(CIUDAD, by_osmid=True).geometry.iloc[0]
    lugares = ox.features_from_polygon(limite, tags={"place": ["suburb", "neighbourhood", "quarter"]})
    lugares = lugares[lugares["name"].notna()]
    lugares = ox.projection.project_gdf(lugares, to_crs=crs)
    lugares["geometry"] = lugares.geometry.representative_point()
    return lugares


# -------------------------------------------------------------------------
# Parte la ciudad en cuadrados de LADO metros, girados como las calles del centro.
#
# Devuelve, uno por uno: (x, y, grafo del recorte), donde (x, y) es el centro del cuadrado.
# -------------------------------------------------------------------------
def recortes(grafo):
    esquinas = ox.convert.graph_to_gdfs(grafo, edges=False)
    x_min, y_min, x_max, y_max = esquinas.total_bounds
    for x in np.arange(x_min + LADO / 2, x_max, LADO):
        for y in np.arange(y_min + LADO / 2, y_max, LADO):
            cuadrado = affinity.rotate(box(x - LADO / 2, y - LADO / 2, x + LADO / 2, y + LADO / 2),
                                       GIRO_CALLES, origin=(x, y))
            # Si el cuadrado cae en el río, no tiene calles
            try:
                recorte = ox.truncate.truncate_graph_polygon(grafo, cuadrado)
            except ValueError:
                continue
            yield x, y, ox.truncate.largest_component(recorte, strongly=True)


# -------------------------------------------------------------------------
# Las medidas de un recorte.
#
# Devuelve: esquinas, km de calle, entropía de orientación, fracción de avenidas,
#           rotondas, y qué fracción de una muestra de rutas sirve para entrenar.
#           Las rutas de muestra usan las mismas funciones que el entrenamiento.
# -------------------------------------------------------------------------
def medir(recorte, rng):
    cuadras = ox.convert.graph_to_gdfs(recorte, nodes=False)
    tipos = cuadras["highway"].astype(str)
    largo = cuadras["length"].sum()

    # La entropía se calcula con las calles sin sentido y en grados
    en_grados = ox.project_graph(ox.convert.to_undirected(recorte), to_latlong=True)

    rotondas = []
    if "junction" in cuadras:
        rotondas = cuadras["junction"].astype(str).str.contains("roundabout|circular")

    # Una muestra de rutas al azar: ¿cuántas sirven?
    mapa = Mapa(recorte)
    esquinas = list(recorte.nodes)
    sirven = 0
    for _ in range(RUTAS_DE_MUESTRA):
        a, b = rng.choice(len(esquinas), size=2, replace=False)
        sirven += ruta_sirve(mapa, calcular_ruta(mapa, esquinas[a], esquinas[b]))

    return {
        "esquinas": recorte.number_of_nodes(),
        # Cada cuadra está dos veces, una por mano: por eso se divide por 2000 y no por 1000
        "km": round(largo / 2000, 1),
        "entropia": round(ox.bearing.orientation_entropy(ox.bearing.add_edge_bearings(en_grados)), 2),
        "avenidas": round(cuadras.loc[tipos.str.contains("trunk|primary|secondary"), "length"].sum() / largo, 2),
        # Cada cuadra de rotonda también está dos veces
        "rotondas": int(sum(rotondas) // 2),
        "rutas_que_sirven": round(sirven / RUTAS_DE_MUESTRA, 2),
    }


# -------------------------------------------------------------------------
# Nivel de dificultad de un recorte, de 1 a 4: 4 si tiene rotondas; si no,
# según en qué tramo de ENTROPIA_NIVELES cae su entropía.
# -------------------------------------------------------------------------
def nivel(medidas):
    if medidas["rotondas"] > 0:
        return 4
    return 1 + int(np.searchsorted(ENTROPIA_NIVELES, medidas["entropia"]))


# -------------------------------------------------------------------------
# Pasa un nombre a minúsculas, sin tildes y con guiones bajos, para usarlo en
# el nombre de un archivo. Por ejemplo, "Candioti Sud" da "candioti_sud".
# -------------------------------------------------------------------------
def sin_tildes(nombre):
    nombre = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    return "_".join("".join(letra if letra.isalnum() else " " for letra in nombre.lower()).split())


# -------------------------------------------------------------------------
# Mide cada recorte de la ciudad, le pone el nombre del barrio más cercano y
# marca como descartados los que no sirven.
#
# Devuelve: una fila por recorte, con sus medidas, su nivel y el grafo del recorte.
# -------------------------------------------------------------------------
def medir_recortes(grafo, rng):
    lugares = barrios(grafo.graph["crs"])
    filas = []
    for x, y, recorte in recortes(grafo):
        if recorte.number_of_nodes() < 2:
            continue
        medidas = medir(recorte, rng)
        cercano = lugares.distance(Point(x, y)).idxmin()
        fila = {"x": round(x), "y": round(y), "barrio": lugares.loc[cercano, "name"], **medidas,
                "nivel": nivel(medidas), "recorte": recorte}
        if medidas["km"] < MINIMO_KM:
            fila["descartado"] = "pocas calles"
        elif medidas["rutas_que_sirven"] < MINIMO_RUTAS:
            fila["descartado"] = "pocas rutas sirven"
        filas.append(fila)
        print(fila["barrio"], {k: v for k, v in fila.items() if k not in ("barrio", "recorte")}, flush=True)
    return filas


# -------------------------------------------------------------------------
# Numera y guarda los recortes que sirven, nivel por nivel. En cada nivel, uno
# de cada UNO_DE_PRUEBA_CADA queda para prueba. Le agrega a cada fila su zona y su uso.
# -------------------------------------------------------------------------
def guardar_recortes(filas, rng):
    for n in range(1, len(ENTROPIA_NIVELES) + 2):
        del_nivel = [fila for fila in filas if fila["nivel"] == n and "descartado" not in fila]
        prueba = set(rng.choice(len(del_nivel), size=len(del_nivel) // UNO_DE_PRUEBA_CADA, replace=False))
        for i, fila in enumerate(del_nivel):
            fila["zona"] = f"n{n}_{i + 1:02d}_{sin_tildes(fila['barrio'])}"
            fila["uso"] = "prueba" if i in prueba else "entrenamiento"
            ox.save_graphml(fila["recorte"], archivo_santa_fe(fila["zona"]))


# -------------------------------------------------------------------------
# Escribe recortes.json: el resumen de todos los recortes, guardados y descartados.
# -------------------------------------------------------------------------
def escribir_resumen(filas):
    for fila in filas:
        del fila["recorte"]
    resumen = {"lado": LADO, "tolerancia_cruces": TOLERANCIA_CRUCES, "entropia_niveles": ENTROPIA_NIVELES,
               "recortes": sorted(filas, key=lambda fila: fila.get("zona", "z"))}
    (CARPETA / "recortes.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Guardados:", sum("zona" in fila for fila in filas), " descartados:", sum("descartado" in fila for fila in filas))


# -------------------------------------------------------------------------
# Baja la ciudad, mide cada recorte, guarda los que sirven y escribe recortes.json.
# -------------------------------------------------------------------------
def descargar():
    rng = np.random.default_rng(0)

    # 1. La ciudad entera
    grafo = bajar_ciudad()
    ox.save_graphml(grafo, CARPETA / "ciudad_santa_fe.graphml")
    print("ciudad_santa_fe.graphml - esquinas:", grafo.number_of_nodes(), " aristas:", grafo.number_of_edges())

    # 2. Medir cada recorte y descartar los que no sirven
    filas = medir_recortes(grafo, rng)

    # 3. Guardar los que sirven
    guardar_recortes(filas, rng)

    # 4. El resumen de todos
    escribir_resumen(filas)


if __name__ == "__main__":
    descargar()
