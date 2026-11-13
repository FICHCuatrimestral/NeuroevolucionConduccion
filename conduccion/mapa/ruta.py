"""Ruta de A a B: el camino que el auto tiene que seguir.

Todo el trabajo geométrico lo hacen las librerías:
  - networkx busca las esquinas con A*
  - osmnx da la forma de esas cuadras y reparte los puntos de control
  - shapely une las cuadras y corre la línea hacia el carril derecho
"""

import math
from dataclasses import dataclass

import networkx as nx
import osmnx as ox
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import substring

from .. import config


# -------------------------------------------------------------------------
# Ruta: todo lo que el auto necesita saber del camino.
#
#   esquinas  las esquinas del grafo por las que pasa, de A a B
#   carril    la línea central del carril derecho (LineString de shapely)
#   puntos    los puntos de control (x, y) sobre el carril; el último es B
#   largo     metros desde donde arranca el auto hasta B
#   inicio    (x, y, ángulo) donde arranca el auto
# -------------------------------------------------------------------------
@dataclass
class Ruta:
    esquinas: list
    carril: LineString
    puntos: list
    largo: float
    inicio: tuple


# -------------------------------------------------------------------------
# Ruta más corta entre dos esquinas.
#
# Recibe: el mapa y las esquinas de origen y destino.
# Devuelve: la Ruta, o None si origen y destino son la misma esquina
#           o si en algún giro muy cerrado el carril se parte.
# -------------------------------------------------------------------------
def calcular_ruta(mapa, origen, destino):
    if origen == destino:
        return None

    # A* necesita una estimación de lo que falta. Se usa la distancia en línea recta,
    # que nunca es mayor que la distancia real por las calles.
    def distancia_recta(u, v):
        return math.dist(mapa.posicion(u), mapa.posicion(v))

    esquinas = nx.astar_path(mapa.grafo, origen, destino, heuristic=distancia_recta, weight="length")
    return ruta_por_esquinas(mapa, esquinas)


# -------------------------------------------------------------------------
# Ruta que pasa por estas esquinas, en orden. Sirve también para recorridos
# elegidos a mano, como la vuelta a una manzana.
#
# Recibe: el mapa y la lista de esquinas.
# Devuelve: la Ruta, o None si en algún giro muy cerrado el carril se parte.
# -------------------------------------------------------------------------
def ruta_por_esquinas(mapa, esquinas):
    # 1. Las cuadras del camino, unidas en una sola línea por el medio de la calle
    centro = linea_central(mapa, esquinas)

    # 2. El carril es esa línea corrida medio carril. Una distancia negativa va hacia la derecha.
    #    En un giro muy cerrado, la línea corrida puede partirse en varios pedazos: esa ruta no sirve.
    carril = centro.offset_curve(-config.ANCHO_CARRIL / 2, join_style="mitre")
    if carril.geom_type != "LineString" or carril.length <= config.LARGO_AUTO:
        return None

    # 3. El auto arranca un largo de auto adelante, para que la cola no quede fuera de la calle.
    #    Mira en la dirección del primer tramo del carril.
    recorrido = substring(carril, config.LARGO_AUTO, carril.length)
    (x, y), (x_siguiente, y_siguiente) = recorrido.coords[:2]
    inicio = (x, y, math.atan2(y_siguiente - y, x_siguiente - x))

    # 4. Los puntos de control, hasta B
    puntos = puntos_de_control(recorrido)
    if not puntos:
        return None

    return Ruta(esquinas, carril, puntos, recorrido.length, inicio)


# -------------------------------------------------------------------------
# Las cuadras que unen estas esquinas, como una sola línea por el medio de la calle.
# simplify(1.0) alisa los quiebres de menos de 1 m que traen los datos de OpenStreetMap.
#
# En una vuelta cerrada, line_merge cerraría un anillo que puede arrancar en cualquier
# esquina, y offset_curve no corre bien una línea cerrada. Por eso las cuadras se unen
# en el orden del camino, y la línea se corta el ancho de la calle antes de cerrarse.
# -------------------------------------------------------------------------
def linea_central(mapa, esquinas):
    cuadras = ox.routing.route_to_gdf(mapa.grafo, esquinas, weight="length")
    if esquinas[0] == esquinas[-1]:
        vuelta = LineString(shapely.get_coordinates(cuadras.geometry)).simplify(1.0)
        return substring(vuelta, 0, vuelta.length - 2 * config.ANCHO_CARRIL)
    lineas = shapely.multilinestrings(list(cuadras.geometry))
    return shapely.line_merge(lineas, directed=True).simplify(1.0)


# -------------------------------------------------------------------------
# Dónde gira una línea.
#
# Recibe: una LineString.
# Devuelve: los metros desde el inicio de la línea hasta cada quiebre que gira
#           ANGULO_GIRO grados o más.
# -------------------------------------------------------------------------
def distancias_a_giros(linea):
    coords = linea.coords
    distancias = []
    recorrido = 0.0

    # Cada vértice interior (x1, y1) se compara con el anterior y el siguiente
    for i in range(1, len(coords) - 1):
        (x0, y0), (x1, y1), (x2, y2) = coords[i - 1], coords[i], coords[i + 1]
        recorrido += math.dist((x0, y0), (x1, y1))

        # Ángulo del tramo que llega y del tramo que sale
        angulo_antes = math.atan2(y1 - y0, x1 - x0)
        angulo_despues = math.atan2(y2 - y1, x2 - x1)

        # La diferencia, llevada a [-180°, 180°] para que no importe el sentido
        giro = math.degrees(abs(math.remainder(angulo_despues - angulo_antes, math.tau)))
        if giro >= config.ANGULO_GIRO:
            distancias.append(recorrido)
    return distancias


# -------------------------------------------------------------------------
# Puntos de control sobre el recorrido.
#
# Sin PUNTOS_EN_GIROS: puntos parejos cada unos SEPARACION_PUNTOS metros.
# Con PUNTOS_EN_GIROS: los giros parten el recorrido en tramos. En cada tramo
# los puntos van parejos y el último cae justo en el giro.
#
# En los dos casos el primer punto sería donde arranca el auto, así que se descarta.
# El último punto es B.
# -------------------------------------------------------------------------
def puntos_de_control(recorrido):
    if not config.PUNTOS_EN_GIROS:
        return list(ox.utils_geo.interpolate_points(recorrido, config.SEPARACION_PUNTOS))[1:]

    cortes = [0.0] + distancias_a_giros(recorrido) + [recorrido.length]
    puntos = []
    for desde, hasta in zip(cortes, cortes[1:]):
        # Dos giros casi en el mismo lugar: alcanza con un punto
        if hasta - desde < 1.0:
            continue
        tramo = substring(recorrido, desde, hasta)
        puntos += list(ox.utils_geo.interpolate_points(tramo, config.SEPARACION_PUNTOS))[1:]
    return puntos


# Cuántos pares de esquinas se sortean, como mucho, antes de dar por perdido un mapa
MAXIMO_INTENTOS = 1000


# -------------------------------------------------------------------------
# Una ruta al azar que sirva para entrenar.
#
# Recibe: el mapa y el generador de números al azar.
# Devuelve: una Ruta entre dos esquinas sorteadas. Si la ruta sorteada no sirve,
#           sortea otra. Si después de MAXIMO_INTENTOS no encontró ninguna, el mapa
#           no tiene rutas útiles y se avisa con un error, en lugar de seguir para siempre.
# -------------------------------------------------------------------------
def ruta_al_azar(mapa, rng):
    esquinas = list(mapa.grafo.nodes)
    for _ in range(MAXIMO_INTENTOS):
        i, j = rng.choice(len(esquinas), size=2, replace=False)
        ruta = calcular_ruta(mapa, esquinas[i], esquinas[j])
        if ruta_sirve(mapa, ruta):
            return ruta
    raise RuntimeError(f"En el mapa {mapa.nombre!r} no se encontró ninguna ruta que sirva en "
                       f"{MAXIMO_INTENTOS} intentos.")


# -------------------------------------------------------------------------
# ¿La ruta sirve para entrenar?
# Sirve si existe, si no es demasiado corta y si todo su carril queda sobre la calle.
# -------------------------------------------------------------------------
def ruta_sirve(mapa, ruta):
    if ruta is None or ruta.largo < config.LARGO_MINIMO_RUTA:
        return False
    return shapely.contains(mapa.calles, ruta.carril)


# -------------------------------------------------------------------------
# Las vueltas completas a la manzana de mapa.esquinas_manzana (ver manzana() en mapa.py).
#
# Devuelve: dos listas de 4 rutas, una por esquina de salida.
#           izquierda: doblando siempre a la izquierda. derecha: siempre a la derecha.
# -------------------------------------------------------------------------
def vueltas_a_la_manzana(mapa):
    esquinas = list(mapa.esquinas_manzana)

    # Recorrer la manzana en sentido antihorario es doblar siempre a la izquierda
    contorno = Polygon([mapa.posicion(esquina) for esquina in esquinas]).exterior
    if not contorno.is_ccw:
        esquinas.reverse()

    izquierda, derecha = [], []
    for i in range(len(esquinas)):
        # Las mismas esquinas, empezando por la número i
        orden = esquinas[i:] + esquinas[:i]

        # Se agrega la primera esquina al final para cerrar la vuelta
        izquierda.append(ruta_por_esquinas(mapa, orden + orden[:1]))
        derecha.append(ruta_por_esquinas(mapa, orden[::-1] + orden[-1:]))
    return izquierda, derecha
