"""Un episodio: un auto recorre una ruta hasta llegar, chocar o agotar el tiempo.

En cada paso:
  1. El auto avanza con el volante y la aceleración que decide la red
  2. Se suma el tiempo que pasa con alguna esquina fuera de su carril
  3. Si alguna esquina sale de la calle, choca
  4. Si no, se revisa si alcanzó el próximo punto de control, o uno de los PUNTOS_SALTEABLES
     siguientes; el último es B
  5. Si se acabó el tiempo, termina
"""

import math
from dataclasses import dataclass

from ..auto.auto import Auto
from ..auto.sensores import Sensores
from .. import config


# -------------------------------------------------------------------------
# Resultado de un episodio: lo que necesita la aptitud y lo que se dibuja.
#
#   puntos_alcanzados        k: puntos de control pasados (alcanzados o salteados)
#   progreso                 p: avance hacia el siguiente punto, de 0 a 1
#   total_puntos             K: puntos de control de la ruta
#   motivo                   "llegada", "choque" o "tiempo"
#   tiempo                   segundos que duró el episodio
#   tiempo_fuera_del_carril  segundos con alguna esquina fuera del carril
#   recorrido                posiciones (x, y) del auto, para dibujar
#   tiempo_maximo            segundos disponibles para esta ruta
# -------------------------------------------------------------------------
@dataclass
class Resultado:
    puntos_alcanzados: int
    progreso: float
    total_puntos: int
    motivo: str
    tiempo: float
    tiempo_fuera_del_carril: float
    recorrido: list
    tiempo_maximo: float


# -------------------------------------------------------------------------
# Pasa las dos salidas de la red, entre -1 y 1, a volante y aceleración.
#
# La primera salida es el volante. La segunda es el acelerador si es positiva
# y el freno si es negativa: se escala por la aceleración o la frenada máxima.
# -------------------------------------------------------------------------
def salidas_a_acciones(salidas):
    volante = config.VOLANTE_MAXIMO * salidas[0]
    if salidas[1] > 0:
        aceleracion = config.ACELERACION_MAXIMA * salidas[1]
    else:
        aceleracion = config.FRENADA_MAXIMA * salidas[1]
    return volante, aceleracion


# -------------------------------------------------------------------------
# Progreso entre dos puntos de control (Togelius y Lucas, 2005).
# Vale 0 sobre el punto anterior y 1 sobre el siguiente.
# -------------------------------------------------------------------------
def progreso_hacia(auto, anterior, siguiente):
    d_anterior = math.dist((auto.x, auto.y), anterior)
    d_siguiente = math.dist((auto.x, auto.y), siguiente)
    return d_anterior / (d_anterior + d_siguiente)


# -------------------------------------------------------------------------
# Episodio: un auto en una ruta. Se avanza de a un paso con paso(), así se puede
# manejar con la red, dibujar en pygame o registrar cada paso.
#
# Recibe: el mapa, la ruta y, si se quiere ruido en los sensores, el generador al azar.
# -------------------------------------------------------------------------
class Episodio:

    def __init__(self, mapa, ruta, rng=None):
        self.mapa = mapa
        self.ruta = ruta
        self.rng = rng

        x, y, angulo = ruta.inicio
        self.auto = Auto(x, y, angulo)
        self.sensores = Sensores(mapa, ruta.carril)

        # Último punto de control alcanzado. Al principio, la salida.
        self.anterior = (x, y)

        # Índice del próximo punto de control
        self.k = 0

        self.tiempo = 0.0
        self.tiempo_maximo = ruta.largo / config.VELOCIDAD_REFERENCIA + config.TIEMPO_EXTRA
        self.tiempo_fuera_del_carril = 0.0
        self.invasion_carril = False
        self.recorrido = [(x, y)]

        # Por qué terminó: queda en None mientras sigue
        self.motivo = None

    # ---------------------------------------------------------------------
    # El próximo punto de control.
    # ---------------------------------------------------------------------
    def siguiente(self):
        return self.ruta.puntos[self.k]

    # ---------------------------------------------------------------------
    # Las 16 entradas de la red en este momento.
    # ---------------------------------------------------------------------
    def entradas(self):
        return self.sensores.entradas(self.auto, self.siguiente(), self.rng)

    # ---------------------------------------------------------------------
    # ¿Alcanzó algún punto pendiente?
    #
    # Se mira el próximo punto y los PUNTOS_SALTEABLES siguientes, en orden.
    # Devuelve: el índice del primero que está a menos de RADIO_PUNTO, o None.
    #
    # Si el auto pasa lejos de un punto (por ejemplo, al abrirse en un giro) pero
    # alcanza el siguiente, sigue sumando: no queda trabado hasta agotar el tiempo.
    # ---------------------------------------------------------------------
    def punto_alcanzado(self):
        ultimo = min(self.k + config.PUNTOS_SALTEABLES, len(self.ruta.puntos) - 1)
        for indice in range(self.k, ultimo + 1):
            if math.dist((self.auto.x, self.auto.y), self.ruta.puntos[indice]) < config.RADIO_PUNTO:
                return indice
        return None

    # ---------------------------------------------------------------------
    # ¿Chocó? Choca si alguna esquina del auto sale de la calle.
    # ---------------------------------------------------------------------
    def choco(self):
        esquinas = self.auto.esquinas()
        xs = [x for x, y in esquinas]
        ys = [y for x, y in esquinas]
        return not self.mapa.es_calle(xs, ys).all()

    # ---------------------------------------------------------------------
    # Un paso de simulación con el volante y la aceleración dados.
    # ---------------------------------------------------------------------
    def paso(self, volante, aceleracion):
        # 1. El auto avanza
        self.auto.avanzar(volante, aceleracion)
        self.tiempo = self.tiempo + config.DT
        self.recorrido.append((self.auto.x, self.auto.y))

        # 2. Tiempo con alguna esquina fuera del carril
        self.invasion_carril = self.auto.invade_carril(self.ruta.carril, self.mapa.intersecciones)
        if self.invasion_carril:
            self.tiempo_fuera_del_carril = self.tiempo_fuera_del_carril + config.DT

        # 3. ¿Chocó?
        if self.choco():
            self.motivo = "choque"

        # 4. Si no chocó, ¿alcanzó algún punto?
        else:
            self.revisar_puntos()

        # 5. ¿Se acabó el tiempo?
        if self.motivo is None and self.tiempo >= self.tiempo_maximo:
            self.motivo = "tiempo"

    # ---------------------------------------------------------------------
    # Si el auto alcanzó un punto de control, avanza k hasta después de ese punto:
    # los que se salteó cuentan como pasados. Si el alcanzado era el último, llegó a B.
    # ---------------------------------------------------------------------
    def revisar_puntos(self):
        alcanzado = self.punto_alcanzado()
        if alcanzado is None:
            return
        self.anterior = self.ruta.puntos[alcanzado]
        self.k = alcanzado + 1
        if self.k == len(self.ruta.puntos):
            self.motivo = "llegada"

    # ---------------------------------------------------------------------
    # Un paso manejado por la red: la red mira las entradas de este momento,
    # decide el volante y la aceleración, y el auto avanza.
    # ---------------------------------------------------------------------
    def paso_con_red(self, red):
        volante, aceleracion = salidas_a_acciones(red.propagar(self.entradas()))
        self.paso(volante, aceleracion)

    # ---------------------------------------------------------------------
    # ¿Terminó el episodio?
    # ---------------------------------------------------------------------
    def terminado(self):
        return self.motivo is not None

    # ---------------------------------------------------------------------
    # El Resultado del episodio. Si llegó a B, no hay siguiente punto y el progreso vale 0.
    # ---------------------------------------------------------------------
    def resultado(self):
        if self.k == len(self.ruta.puntos):
            progreso = 0.0
        else:
            progreso = progreso_hacia(self.auto, self.anterior, self.siguiente())
        return Resultado(self.k, progreso, len(self.ruta.puntos), self.motivo, self.tiempo,
                         self.tiempo_fuera_del_carril, self.recorrido, self.tiempo_maximo)


# -------------------------------------------------------------------------
# La red maneja la ruta de principio a fin, sin dibujar nada.
#
# Recibe: el mapa, la ruta, la red y, si se quiere ruido en los sensores, el generador al azar.
# Devuelve: el Resultado.
# -------------------------------------------------------------------------
def correr_episodio(mapa, ruta, red, rng=None):
    episodio = Episodio(mapa, ruta, rng)
    while not episodio.terminado():
        episodio.paso_con_red(red)
    return episodio.resultado()
