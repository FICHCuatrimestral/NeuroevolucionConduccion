"""Auto simulado: su física y su contorno. Lo que el auto percibe está en sensores.py.

FÍSICA. Modelo de bicicleta cinemático de Kong et al. (2015). Las dos ruedas de cada eje se juntan
en una sola y solo gira la delantera. La red decide el volante y la aceleración, y el modelo calcula
cómo se mueve el auto:

    beta    = arctan( l_r / (l_f + l_r) · tan(volante) )
    x'      = v · cos(angulo + beta)
    y'      = v · sin(angulo + beta)
    angulo' = v / l_r · sin(beta)
    v'      = aceleracion

(x, y) es el centro de masa y beta, el ángulo entre hacia dónde mira el auto y hacia dónde se mueve.
Se integra con Euler: cada variable suma su derivada por DT.

CONTORNO. Un rectángulo que gira con el auto. Sirve para los choques y para medir si invade el carril.
"""

import math

import numpy as np
import shapely

from .. import config

# -------------------------------------------------------------------------
# Auto: su estado es la posición (x, y), hacia dónde mira, la velocidad y el volante.
#
# Recibe: la posición inicial y el ángulo inicial, medido desde el eje x.
# Arranca quieto y con el volante derecho.
# -------------------------------------------------------------------------
class Auto:

    def __init__(self, x, y, angulo):
        self.x = x
        self.y = y
        self.angulo = angulo
        self.velocidad = 0.0
        self.volante = 0.0

    # =====================================================================
    # FÍSICA
    # =====================================================================

    # ---------------------------------------------------------------------
    # Mueve el auto un paso de DT segundos.
    #
    # Recibe: el volante (radianes) y la aceleración (m/s², negativa = freno)
    #         que decidió la red. Antes de usarlos se recortan a lo que el auto permite.
    # ---------------------------------------------------------------------
    def avanzar(self, volante, aceleracion):
        volante, aceleracion = self.aplicar_limites(volante, aceleracion)

        # 1. Las derivadas, con el estado actual
        beta = math.atan(config.L_R / (config.L_F + config.L_R) * math.tan(volante))
        dx = self.velocidad * math.cos(self.angulo + beta)
        dy = self.velocidad * math.sin(self.angulo + beta)
        dangulo = self.velocidad / config.L_R * math.sin(beta)
        dvelocidad = aceleracion

        # 2. Paso de Euler: cada variable suma su derivada por DT
        self.x = self.x + dx * config.DT
        self.y = self.y + dy * config.DT
        self.angulo = self.angulo + dangulo * config.DT

        # La velocidad no puede ser negativa (el auto no va marcha atrás) ni pasar la máxima
        self.velocidad = recortar(self.velocidad + dvelocidad * config.DT, 0.0, config.VELOCIDAD_MAXIMA)
        self.volante = volante

    # ---------------------------------------------------------------------
    # Recorta el volante y la aceleración a lo que el auto puede hacer.
    #
    # Devuelve: el volante y la aceleración que de verdad se aplican.
    # ---------------------------------------------------------------------
    def aplicar_limites(self, volante, aceleracion):
        # 1. El ángulo máximo del volante
        volante = recortar(volante, -config.VOLANTE_MAXIMO, config.VOLANTE_MAXIMO)

        # 2. El agarre: a esta velocidad, más volante haría patinar al auto
        maximo_con_agarre = self.volante_con_agarre()
        volante = recortar(volante, -maximo_con_agarre, maximo_con_agarre)

        # 3. Si hay límite, el volante no puede girar más de GIRO_VOLANTE_MAXIMO por segundo
        if config.GIRO_VOLANTE_MAXIMO is not None:
            cambio_maximo = config.GIRO_VOLANTE_MAXIMO * config.DT
            volante = recortar(volante, self.volante - cambio_maximo, self.volante + cambio_maximo)

        # 4. El freno y el acelerador a fondo
        aceleracion = recortar(aceleracion, -config.FRENADA_MAXIMA, config.ACELERACION_MAXIMA)
        return volante, aceleracion

    # ---------------------------------------------------------------------
    # Cuánto volante se puede usar a la velocidad actual sin pasar la aceleración lateral máxima.
    #
    # La aceleración lateral es a_y = v² · sin(beta) / l_r. Se despeja el sin(beta) que da
    # a_y = ACELERACION_LATERAL_MAXIMA, y con la fórmula de beta se pasa a volante.
    # Cuanto más rápido va el auto, menos volante tiene: para doblar cerrado hay que frenar.
    #
    # Devuelve: el volante máximo, en radianes.
    # ---------------------------------------------------------------------
    def volante_con_agarre(self):
        # Sin límite, o con el auto quieto, vale el volante máximo
        if config.ACELERACION_LATERAL_MAXIMA is None or self.velocidad == 0:
            return config.VOLANTE_MAXIMO

        seno_beta = config.ACELERACION_LATERAL_MAXIMA * config.L_R / self.velocidad ** 2

        # Si el seno da 1 o más, ni siquiera con el volante a fondo se llega al límite
        if seno_beta >= 1:
            return config.VOLANTE_MAXIMO

        # beta = arcsen(seno_beta), y de beta = arctan(l_r / (l_f + l_r) · tan(volante)) se despeja el volante
        beta = math.asin(seno_beta)
        volante = math.atan((config.L_F + config.L_R) / config.L_R * math.tan(beta))
        return min(config.VOLANTE_MAXIMO, volante)

    # =====================================================================
    # CONTORNO
    # =====================================================================

    # ---------------------------------------------------------------------
    # Las cuatro esquinas del rectángulo del auto, en este orden:
    # adelante izquierda, adelante derecha, atrás derecha, atrás izquierda.
    # ---------------------------------------------------------------------
    def esquinas(self):
        # Dos direcciones que giran con el auto: hacia adelante y hacia su izquierda
        adelante_x = math.cos(self.angulo)
        adelante_y = math.sin(self.angulo)
        izquierda_x = -adelante_y
        izquierda_y = adelante_x

        mitad_largo = config.LARGO_AUTO / 2
        mitad_ancho = config.ANCHO_AUTO / 2

        # Cada esquina es el centro, más medio largo hacia adelante o atrás,
        # más medio ancho hacia la izquierda o la derecha
        esquinas = []
        for largo, ancho in ((mitad_largo, mitad_ancho), (mitad_largo, -mitad_ancho),
                             (-mitad_largo, -mitad_ancho), (-mitad_largo, mitad_ancho)):
            esquinas.append((self.x + adelante_x * largo + izquierda_x * ancho,
                             self.y + adelante_y * largo + izquierda_y * ancho))
        return esquinas

    # ---------------------------------------------------------------------
    # ¿El auto invade otro carril?
    #
    # Es True si alguna esquina se aleja de la línea central de su carril más de medio carril.
    # Mientras el auto toca un cruce no se mide, porque al doblar puede pasar por el centro.
    #
    # Recibe: la línea del carril y las zonas de cruce del mapa.
    # ---------------------------------------------------------------------
    def invade_carril(self, linea_del_carril, intersecciones=None):
        esquinas = np.asarray(self.esquinas())

        # 1. En un cruce no se mide
        if intersecciones is not None and shapely.intersects(intersecciones, shapely.Polygon(esquinas)):
            return False

        # 2. Distancia de cada esquina a la línea del carril
        puntos = shapely.points(esquinas[:, 0], esquinas[:, 1])
        distancias = shapely.distance(linea_del_carril, puntos)
        return bool(np.any(distancias > config.ANCHO_CARRIL / 2 + config.MARGEN_CARRIL))


# -------------------------------------------------------------------------
# Deja valor entre minimo y maximo.
# -------------------------------------------------------------------------
def recortar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))

