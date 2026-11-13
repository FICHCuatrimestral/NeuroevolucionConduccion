"""Sensores: lo que el auto percibe. Con eso arma las 16 entradas del perceptrón.

El auto no sabe dónde está en el mapa: solo ve lo que miden sus sensores.

     7  rayos al borde de la calle          distancia / alcance              0 a 1
     1  velocidad                           v / v máxima                     0 a 1
     2  brújula al próximo punto            seno y coseno del ángulo        -1 a 1
     1  distancia al próximo punto          distancia / separación, hasta 1  0 a 1
     1  desvío del carril                   distancia con signo / ancho     -1 a 1
     4  miradas al carril, a 3 m y a 6 m    seno y coseno de cada una       -1 a 1
"""

import math

import numpy as np
import shapely

from .. import config

# Distancias a las que cada rayo mira el mapa: 0,25 m, 0,5 m, ..., 30 m
DISTANCIAS_RAYOS = np.arange(1, round(config.ALCANCE_RAYOS / config.PASO_RAYOS) + 1) * config.PASO_RAYOS


# -------------------------------------------------------------------------
# Sensores de un auto en un episodio.
#
# Recibe: el mapa y la línea del carril que hay que seguir. Los dos quedan fijos
#         durante el episodio; en cada paso, los métodos reciben el auto.
#         Para medir solo los rayos, alcanza con el mapa.
# -------------------------------------------------------------------------
class Sensores:

    def __init__(self, mapa, carril=None):
        self.mapa = mapa
        self.carril = carril

    # ---------------------------------------------------------------------
    # Las 16 entradas de la red, en el orden de la tabla de arriba del archivo.
    #
    # Recibe: el auto, el próximo punto de control (x, y) y, si se quiere ruido,
    #         el generador al azar.
    # Devuelve: un arreglo de 16 números.
    # ---------------------------------------------------------------------
    def entradas(self, auto, siguiente, rng=None):
        entradas = []

        # 7 rayos al borde de la calle
        for distancia in self.medir_rayos(auto):
            entradas.append(distancia / config.ALCANCE_RAYOS)

        # Velocidad
        entradas.append(auto.velocidad / config.VELOCIDAD_MAXIMA)

        # Brújula al próximo punto
        entradas.extend(brujula(auto, siguiente))

        # Distancia al próximo punto
        entradas.append(min(1.0, math.dist((auto.x, auto.y), siguiente) / config.SEPARACION_PUNTOS))

        # Desvío del carril
        entradas.append(max(-1.0, min(1.0, self.desvio_del_carril(auto) / config.ANCHO_CARRIL)))

        # Miradas al carril, a 3 m y a 6 m
        for metros in config.MIRADAS_CARRIL:
            entradas.extend(self.mirada_al_carril(auto, metros))

        entradas = np.array(entradas)

        # Un poco de ruido, para que la red no dependa de valores exactos (Togelius y Lucas, 2006)
        if rng is not None:
            entradas = entradas + rng.normal(0, config.RUIDO_SENSORES, len(entradas))
        return entradas

    # ---------------------------------------------------------------------
    # Las distancias al borde de la calle, en metros, para los 7 rayos.
    #
    # Cada rayo es una fila de puntos, uno cada PASO_RAYOS metros. La distancia
    # del rayo es la del primer punto que no es calle; si no toca nada, el alcance.
    # ---------------------------------------------------------------------
    def medir_rayos(self, auto):
        # 1. Los puntos de los 7 rayos juntos: una fila por rayo, una columna por distancia
        angulos = auto.angulo + np.array(config.ANGULOS_RAYOS)
        xs = auto.x + np.outer(np.cos(angulos), DISTANCIAS_RAYOS)
        ys = auto.y + np.outer(np.sin(angulos), DISTANCIAS_RAYOS)

        # 2. Cuáles de esos puntos ya no son calle
        fuera_de_la_calle = ~self.mapa.es_calle(xs, ys)

        # 3. Cada rayo se queda con su primer contacto
        distancias = []
        for tocados in fuera_de_la_calle:
            if tocados.any():
                distancias.append(DISTANCIAS_RAYOS[np.argmax(tocados)])
            else:
                distancias.append(config.ALCANCE_RAYOS)
        return distancias

    # ---------------------------------------------------------------------
    # Metros de carril recorridos hasta el punto de la línea más cercano al auto.
    # ---------------------------------------------------------------------
    def lugar_en_el_carril(self, auto):
        return shapely.line_locate_point(self.carril, shapely.Point(auto.x, auto.y))

    # ---------------------------------------------------------------------
    # Brújula a un punto del carril que está esos metros más adelante que el auto.
    # Así la red ve hacia dónde va a doblar el carril antes de llegar al giro.
    # ---------------------------------------------------------------------
    def mirada_al_carril(self, auto, metros):
        punto = self.carril.interpolate(self.lugar_en_el_carril(auto) + metros)
        return brujula(auto, (punto.x, punto.y))

    # ---------------------------------------------------------------------
    # Distancia con signo del auto a la línea del carril. Positivo = a la izquierda.
    #
    # Se toma el punto de la línea más cercano al auto y otro 0,5 m más adelante.
    # El producto vectorial entre ese tramo y el vector que va de la línea al auto
    # da la distancia, con signo según el lado.
    # ---------------------------------------------------------------------
    def desvio_del_carril(self, auto):
        lugar = self.lugar_en_el_carril(auto)
        cercano = self.carril.interpolate(lugar)
        adelante = self.carril.interpolate(lugar + 0.5)

        # Dirección del carril en ese lugar
        dx = adelante.x - cercano.x
        dy = adelante.y - cercano.y
        largo = math.hypot(dx, dy)

        # Al final del carril los dos puntos coinciden y no hay dirección
        if largo == 0:
            return 0.0

        return (dx * (auto.y - cercano.y) - dy * (auto.x - cercano.x)) / largo


# -------------------------------------------------------------------------
# Brújula: seno y coseno del ángulo entre hacia dónde mira el auto y un punto.
#
# Se usan seno y coseno, y no el ángulo, para que no haya un salto de 180° a -180°
# cuando el punto queda detrás del auto.
# -------------------------------------------------------------------------
def brujula(auto, punto):
    angulo_al_punto = math.atan2(punto[1] - auto.y, punto[0] - auto.x) - auto.angulo
    return math.sin(angulo_al_punto), math.cos(angulo_al_punto)
