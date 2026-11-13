"""Baja de OpenStreetMap las calles del centro de Santa Fe y guarda tres zonas en mapas/: 2x2, 3x3 y centro.

Solo hace falta correrlo para cambiar de zona: los mapas guardados están en el repositorio,
así todos entrenan con los mismos y no hace falta internet. La manzana de la primera etapa
no se descarga: sale del 3x3 (ver manzana() en mapa.py).

Los mapas candioti y guadalupe no salen de este script: se bajaron aparte y quedaron guardados
en mapas/. Guadalupe es el barrio de prueba de la red base en el informe.

Se corre desde la carpeta del proyecto: python -m conduccion.mapa.descargar_mapa
"""

import osmnx as ox
from shapely import affinity
from shapely.geometry import box

from .mapa import archivo_santa_fe

# Plaza 25 de Mayo, Santa Fe. Va fija porque buscarla por nombre devolvió la de Rosario.
CENTRO = (-31.65737, -60.7105)

# Metros alrededor del centro que se bajan
RADIO = 500

# Los nodos más cerca que esto, en metros, forman un solo cruce
TOLERANCIA_CRUCES = 10

# Cuántos grados están giradas las calles del centro respecto del norte
GIRO_CALLES = -14.5

# Cada recorte: (x, y, medio lado) de su cuadrado, en metros
RECORTES = {"2x2": (717348.3, 6495527.5, 150), "3x3": (717270.9, 6495482.3, 215)}


# -------------------------------------------------------------------------
# Agrega el sentido contrario a las cuadras de una sola mano. Así toda calle
# tiene carril derecho en los dos sentidos.
# -------------------------------------------------------------------------
def doble_mano(grafo):
    for u, v, datos in list(grafo.edges(data=True)):
        if not grafo.has_edge(v, u):
            # La misma cuadra, con su forma dada vuelta
            grafo.add_edge(v, u, **{**datos, "geometry": datos["geometry"].reverse()})


# -------------------------------------------------------------------------
# Guarda una zona en mapas/santa_fe_<zona>.graphml, con todas sus calles de doble mano.
# -------------------------------------------------------------------------
def guardar(grafo, zona):
    doble_mano(grafo)
    archivo = archivo_santa_fe(zona)
    archivo.parent.mkdir(exist_ok=True)
    ox.save_graphml(grafo, archivo)
    print(archivo.name, "- esquinas:", grafo.number_of_nodes(), " aristas:", grafo.number_of_edges())


# -------------------------------------------------------------------------
# Baja el centro y guarda las tres zonas.
# -------------------------------------------------------------------------
def descargar():
    # 1. Calles para autos alrededor del centro, en latitud y longitud
    grafo = ox.graph_from_point(CENTRO, dist=RADIO, network_type="drive")

    # 2. De latitud y longitud a metros
    grafo = ox.project_graph(grafo)

    # 3. Un cruce puede venir partido en varios nodos: se unen en uno
    grafo = ox.simplification.consolidate_intersections(grafo, tolerance=TOLERANCIA_CRUCES)

    # 4. Cada recorte es un cuadrado girado como las calles
    for zona, (x, y, medio) in RECORTES.items():
        cuadrado = affinity.rotate(box(x - medio, y - medio, x + medio, y + medio), GIRO_CALLES, origin=(x, y))
        guardar(ox.truncate.truncate_graph_polygon(grafo, cuadrado), zona)

    # 5. El centro es todo lo que se bajó
    guardar(grafo, "centro")


if __name__ == "__main__":
    descargar()
