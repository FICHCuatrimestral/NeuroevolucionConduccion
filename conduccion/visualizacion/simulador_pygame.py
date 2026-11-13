"""Ventanas de pygame.

ver_simulacion:     una red maneja en tiempo real.
                    Teclas: flechas arriba/abajo = ritmo, Espacio = pausa,
                    R = reiniciar la ruta, N = volver a rutas al azar, S = rayos, Esc = salir.
ver_entrenamiento:  toda la población maneja a la vez mientras el algoritmo genético la va evolucionando.
                    Teclas: Espacio = pausa, flechas arriba/abajo = más o menos rápido, Esc = terminar.
ver_entrenamiento_por_etapas: lo mismo, por etapas (manzana → 2x2 → 3x3), como en el informe.

Al entrenar con ventana, todos los autos manejan a la vez, así que el ruido de los sensores se
sortea en otro orden que sin ventana: con la misma semilla, los números no son exactamente los
mismos. Sin ruido en los sensores, las notas son idénticas.

En las dos ventanas, en pausa, se puede elegir una ruta con dos clics: la esquina de inicio y la de destino.
Al ver manejar una red, la red sigue en esa ruta hasta que N vuelve a las rutas al azar. En el entrenamiento,
la población la maneja una vez como exhibición, sin que cuente para la nota, y después sigue entrenando.
"""

import math

import networkx as nx
import numpy as np
import pygame

from ..aprendizaje.entrenamiento import entrenar
from ..aprendizaje.etapas import entrenar_por_etapas
from ..aprendizaje.fitness import contar_llegadas, nota, nota_de_ruta
from ..aprendizaje.guardado import cargar
from ..aprendizaje.perceptron_multicapa import Perceptron
from .. import config
from ..mapa.ruta import calcular_ruta, ruta_al_azar
from ..simulacion.simulacion import Episodio

# Alto del mapa en la ventana, en píxeles. El ancho sale de la forma del mapa.
ALTO_MAPA = 800

# Ancho del panel de texto, a la derecha del mapa
ANCHO_PANEL = 280

# Ritmos de reproducción posibles: 1.0 es tiempo real
VELOCIDADES_REPRODUCCION = (0.5, 1.0, 2.0, 4.0, 8.0)

# Segundos reales que se muestra un episodio terminado antes de pasar al siguiente
ESPERA_ENTRE_RUTAS = 1.0

# Colores, en RGB

# Fuera de la calle
VERDE = (168, 201, 154)
# Calle
GRIS = (107, 107, 107)
# Carril de la ruta
CELESTE = (42, 159, 214)
# Puntos de control pendientes
AMARILLO = (242, 193, 46)
# Separación de las dos manos de la calle
LINEA_CENTRAL = (207, 188, 112)
# Puntos de control alcanzados
VERDE_FUERTE = (50, 205, 50)
# Auto
AZUL = (42, 120, 214)
# Destino y choque
ROJO = (208, 59, 59)
# Se le acabó el tiempo
GRIS_OSCURO = (60, 60, 60)
# Rayos, o auto fuera de su carril
NARANJA = (255, 165, 0)
BLANCO = (240, 240, 240)
FONDO_PANEL = (30, 34, 40)


# -------------------------------------------------------------------------
# Cierra la ventana antes de devolver el control al menú o al notebook.
# -------------------------------------------------------------------------
def cerrar_ventana():
    pygame.display.quit()
    pygame.quit()


# -------------------------------------------------------------------------
# SelectorRuta: elegir una ruta a mano con dos clics, en pausa.
# Lo usan las dos ventanas. Cada clic se ajusta a la esquina más cercana.
#
#   ruta_manual  la ruta elegida, o None si se usan rutas al azar
#   origen       la esquina de inicio, mientras se espera el clic del destino
#   cambio       True cuando cambió la ruta y la ventana tiene que reiniciar el episodio
#   mensaje      el texto de ayuda que se muestra en el panel
# -------------------------------------------------------------------------
class SelectorRuta:

    def __init__(self, ventana):
        self.ventana = ventana
        self.ruta_manual = None
        self.origen = None
        self.cambio = False
        self.mensaje = ""
        self.boton_azar = pygame.Rect(ventana.ancho_mapa + 20, ALTO_MAPA - 50, ANCHO_PANEL - 40, 32)

    # ---------------------------------------------------------------------
    # Deja de usar la ruta manual y vuelve a las rutas al azar.
    # ---------------------------------------------------------------------
    def volver_al_azar(self):
        self.ruta_manual = None
        self.origen = None
        self.mensaje = "Rutas aleatorias activadas."
        self.cambio = True

    # ---------------------------------------------------------------------
    # Atiende un clic. El primero elige el inicio y el segundo, el destino.
    # Solo se puede elegir en pausa y haciendo clic cerca de una esquina.
    # ---------------------------------------------------------------------
    def clic(self, posicion, pausa):
        # El botón de volver al azar funciona siempre
        if self.boton_azar.collidepoint(posicion):
            self.volver_al_azar()
            return

        x, y = posicion
        if not pausa or not (0 <= x < self.ventana.ancho_mapa and 0 <= y < ALTO_MAPA):
            return

        # 1. La esquina más cercana al clic
        mapa = self.ventana.mapa
        esquina = self.esquina_cercana(posicion)
        if esquina is None:
            self.mensaje = "Hacé clic cerca de una esquina."
            return

        # 2. Primer clic: el inicio
        if self.origen is None:
            self.origen = esquina
            self.mensaje = ""
            return
        if esquina == self.origen:
            self.mensaje = "Elegí otra esquina de destino."
            return

        # 3. Segundo clic: el destino. Se aceptan rutas cortas, siempre que el carril quede en la calle.
        try:
            ruta = calcular_ruta(mapa, self.origen, esquina)
        except nx.NetworkXNoPath:
            ruta = None
        if ruta is None or not mapa.calles.contains(ruta.carril):
            self.mensaje = "No hay ruta válida. Elegí otro fin."
            return

        self.ruta_manual = ruta
        self.origen = None
        self.mensaje = "Ruta lista. Espacio para seguir."
        self.cambio = True

    # ---------------------------------------------------------------------
    # La esquina más cercana a un punto de la pantalla.
    # Devuelve: la esquina, o None si la más cercana está a más de 20 píxeles.
    # ---------------------------------------------------------------------
    def esquina_cercana(self, posicion):
        mapa = self.ventana.mapa

        # Distancia en píxeles de una esquina al clic
        def distancia_al_clic(nodo):
            return math.dist(posicion, self.ventana.a_pantalla(*mapa.posicion(nodo)))

        esquina = min(mapa.grafo.nodes, key=distancia_al_clic)
        if distancia_al_clic(esquina) > 20:
            return None
        return esquina

    # ---------------------------------------------------------------------
    # Dibuja la ayuda en el panel, el botón de volver al azar y las letras A y B.
    # En pausa, además, marca todas las esquinas que se pueden elegir.
    # ---------------------------------------------------------------------
    def dibujar(self, pausa):
        ventana = self.ventana
        x = ventana.ancho_mapa + 20
        y = ALTO_MAPA - 178

        # 1. La ayuda
        textos = ["Ruta: " + ("manual" if self.ruta_manual is not None else "aleatoria")]
        if pausa:
            textos.append("Clic: elegí el inicio" if self.origen is None else "Clic: elegí el destino")
            textos.append("Se ajusta a la esquina cercana.")
            for nodo in ventana.mapa.grafo.nodes:
                pygame.draw.circle(ventana.pantalla, BLANCO, ventana.a_pantalla(*ventana.mapa.posicion(nodo)), 4, 1)
        else:
            textos.append("Pausá para elegir una ruta.")
        if self.mensaje:
            textos.append(self.mensaje)
        for texto in textos:
            ventana.pantalla.blit(ventana.fuente_ayuda.render(texto, True, BLANCO), (x, y))
            y += 23

        # 2. El botón
        pygame.draw.rect(ventana.pantalla, GRIS_OSCURO, self.boton_azar, border_radius=5)
        ventana.pantalla.blit(ventana.fuente_ayuda.render("N: volver a aleatorio", True, BLANCO),
                              (self.boton_azar.x + 10, self.boton_azar.y + 6))

        # 3. Las letras A y B sobre las esquinas elegidas
        extremos = []
        if self.ruta_manual is not None:
            extremos = [("A", self.ruta_manual.esquinas[0], VERDE_FUERTE),
                        ("B", self.ruta_manual.esquinas[-1], ROJO)]
        if self.origen is not None:
            extremos.append(("A", self.origen, VERDE_FUERTE))
        for letra, nodo, color in extremos:
            px, py = ventana.a_pantalla(*ventana.mapa.posicion(nodo))
            pygame.draw.circle(ventana.pantalla, color, (px, py), 9, 2)
            ventana.pantalla.blit(ventana.fuente.render(letra, True, color), (px + 10, py - 20))


# -------------------------------------------------------------------------
# Ventana: el mapa a la izquierda y un panel de texto a la derecha.
# Tiene lo que comparten VentanaSimulacion y VentanaEntrenamiento.
# -------------------------------------------------------------------------
class Ventana:

    def __init__(self, mapa):
        pygame.init()
        pygame.display.set_caption("Conducción autónoma por neuroevolución")
        self.fuente = pygame.font.SysFont("arial", 18)
        self.fuente_grande = pygame.font.SysFont("arial", 24, bold=True)
        self.fuente_ayuda = pygame.font.SysFont("arial", 15)
        self.mapa = None
        self.usar_mapa(mapa)

    # ---------------------------------------------------------------------
    # Pasa a dibujar otro mapa: escala, tamaño de la ventana, fondo y selector
    # de rutas. En el entrenamiento por etapas se llama cada vez que cambia el mapa.
    # ---------------------------------------------------------------------
    def usar_mapa(self, mapa):
        if mapa is self.mapa:
            return
        self.mapa = mapa

        # El mapa ocupa ALTO_MAPA píxeles de alto; el ancho mantiene la proporción
        filas, columnas = mapa.matriz.shape
        self.escala = ALTO_MAPA / (filas * config.RESOLUCION_MAPA)
        self.ancho_mapa = round(columnas * config.RESOLUCION_MAPA * self.escala)
        self.pantalla = pygame.display.set_mode((self.ancho_mapa + ANCHO_PANEL, ALTO_MAPA))
        self.fondo = self.dibujar_fondo()
        self.selector = SelectorRuta(self)

    # ---------------------------------------------------------------------
    # El fondo: la matriz del mapa como imagen, con la línea central de cada calle.
    # Se dibuja una sola vez y se reutiliza en cada cuadro.
    # ---------------------------------------------------------------------
    def dibujar_fondo(self):
        # 1. La matriz como imagen. pygame usa [x, y] con la y hacia abajo;
        #    la matriz es [fila, columna] con las filas hacia arriba. Por eso se invierte y se traspone.
        es_calle = self.mapa.matriz[::-1, :].T
        colores = np.zeros(es_calle.shape + (3,), dtype=np.uint8)
        colores[es_calle] = GRIS
        colores[~es_calle] = VERDE
        imagen = pygame.surfarray.make_surface(colores)
        fondo = pygame.transform.smoothscale(imagen, (self.ancho_mapa, ALTO_MAPA))

        # 2. La línea central de cada cuadra
        self.dibujar_lineas_centrales(fondo)
        return fondo

    # ---------------------------------------------------------------------
    # Dibuja sobre el fondo la línea central de cada cuadra, sin entrar en los cruces.
    # Una misma cuadra aparece dos veces en el grafo (ida y vuelta): se dibuja una vez.
    # ---------------------------------------------------------------------
    def dibujar_lineas_centrales(self, fondo):
        dibujadas = set()
        for origen, destino in self.mapa.grafo.edges():
            cuadra = frozenset((origen, destino))
            if cuadra in dibujadas:
                continue
            dibujadas.add(cuadra)

            x0, y0 = self.mapa.posicion(origen)
            x1, y1 = self.mapa.posicion(destino)
            largo = math.hypot(x1 - x0, y1 - y0)

            # Toda esquina es un cruce: la línea se recorta en cada punta para no entrar en él
            recorte = self.mapa.radio_interseccion
            if largo <= 2 * recorte:
                continue

            # Dirección de la cuadra, de largo 1
            dx = (x1 - x0) / largo
            dy = (y1 - y0) / largo
            pygame.draw.line(fondo, LINEA_CENTRAL,
                             self.a_pantalla(x0 + dx * recorte, y0 + dy * recorte),
                             self.a_pantalla(x1 - dx * recorte, y1 - dy * recorte), 1)

    # ---------------------------------------------------------------------
    # De metros a píxeles. En pantalla la y crece hacia abajo.
    # ---------------------------------------------------------------------
    def a_pantalla(self, x, y):
        columna = (x - self.mapa.x0) * self.escala
        fila = ALTO_MAPA - (y - self.mapa.y0) * self.escala
        return round(columna), round(fila)

    # ---------------------------------------------------------------------
    # Dibuja la ruta: el carril, los puntos de control y el destino B.
    # Los primeros `alcanzados` puntos van en verde.
    # ---------------------------------------------------------------------
    def dibujar_ruta(self, ruta, alcanzados=0, radio_punto=3):
        pygame.draw.lines(self.pantalla, CELESTE, False, [self.a_pantalla(x, y) for x, y in ruta.carril.coords], 1)
        for i, punto in enumerate(ruta.puntos):
            color = VERDE_FUERTE if i < alcanzados else AMARILLO
            pygame.draw.circle(self.pantalla, color, self.a_pantalla(*punto), radio_punto)
        pygame.draw.circle(self.pantalla, ROJO, self.a_pantalla(*ruta.puntos[-1]), 7, 2)

    # ---------------------------------------------------------------------
    # Dibuja el rectángulo del auto.
    # ---------------------------------------------------------------------
    def dibujar_auto(self, auto, color):
        pygame.draw.polygon(self.pantalla, color, [self.a_pantalla(x, y) for x, y in auto.esquinas()])

    # ---------------------------------------------------------------------
    # Escribe el panel de la derecha.
    # Recibe: una lista de renglones (texto, fuente), de arriba hacia abajo.
    # ---------------------------------------------------------------------
    def escribir_panel(self, renglones):
        pygame.draw.rect(self.pantalla, FONDO_PANEL, (self.ancho_mapa, 0, ANCHO_PANEL, ALTO_MAPA))
        x, y = self.ancho_mapa + 20, 20
        for texto, fuente in renglones:
            self.pantalla.blit(fuente.render(texto, True, BLANCO), (x, y))
            y = y + fuente.get_height() + 6

# =========================================================================
# VER MANEJAR UNA RED
# =========================================================================

# -------------------------------------------------------------------------
# VentanaSimulacion: una red maneja rutas una atrás de otra, en tiempo real.
#
#   red               la red que maneja
#   rng               generador al azar de las rutas
#   episodio          el episodio que se está mostrando
#   pausa             True mientras está en pausa
#   ver_rayos         True si se dibujan los rayos
#   indice_velocidad  posición en VELOCIDADES_REPRODUCCION
#   acumulado         tiempo real que todavía falta simular
#   espera            tiempo que lleva mostrándose un episodio terminado
#   choques           choques desde que se abrió la ventana
# -------------------------------------------------------------------------
class VentanaSimulacion(Ventana):

    def __init__(self, mapa, red, semilla=0):
        super().__init__(mapa)
        self.red = red
        self.rng = np.random.default_rng(semilla)
        self.pausa = False
        self.ver_rayos = True
        self.indice_velocidad = 1
        self.choques = 0
        self.empezar_episodio(ruta_al_azar(mapa, self.rng))

    # ---------------------------------------------------------------------
    # Empieza un episodio nuevo en la ruta dada.
    # ---------------------------------------------------------------------
    def empezar_episodio(self, ruta):
        self.episodio = Episodio(self.mapa, ruta)
        self.acumulado = 0.0
        self.espera = 0.0

    # ---------------------------------------------------------------------
    # La ruta que sigue: la elegida a mano, si hay una, o una al azar.
    # ---------------------------------------------------------------------
    def ruta_siguiente(self):
        if self.selector.ruta_manual is not None:
            return self.selector.ruta_manual
        return ruta_al_azar(self.mapa, self.rng)

    # ---------------------------------------------------------------------
    # Atiende teclas y clics.
    # Devuelve: False si hay que terminar (Esc o ventana cerrada).
    # ---------------------------------------------------------------------
    def atender_teclas(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    return False
                elif evento.key == pygame.K_SPACE:
                    self.pausa = not self.pausa
                elif evento.key == pygame.K_UP:
                    self.indice_velocidad = min(self.indice_velocidad + 1, len(VELOCIDADES_REPRODUCCION) - 1)
                elif evento.key == pygame.K_DOWN:
                    self.indice_velocidad = max(self.indice_velocidad - 1, 0)
                elif evento.key == pygame.K_r:
                    self.empezar_episodio(self.episodio.ruta)
                elif evento.key == pygame.K_n:
                    self.selector.volver_al_azar()
                elif evento.key == pygame.K_s:
                    self.ver_rayos = not self.ver_rayos
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                self.selector.clic(evento.pos, self.pausa)
        return True

    # ---------------------------------------------------------------------
    # Avanza la simulación lo que corresponde al tiempo real transcurrido.
    #
    # En pausa no se simula. Un episodio terminado se muestra un rato y después
    # se pasa al siguiente. Si no, se simulan tantos pasos de DT como entren en el
    # tiempo acumulado: el paso de la física es siempre DT, y el ritmo de reproducción
    # solo cambia cuántos pasos se ven por segundo.
    # ---------------------------------------------------------------------
    def avanzar(self, transcurrido):
        self.acumulado = self.acumulado + transcurrido * VELOCIDADES_REPRODUCCION[self.indice_velocidad]
        if self.pausa:
            self.acumulado = 0.0
        elif self.episodio.terminado():
            self.acumulado = 0.0
            self.espera = self.espera + transcurrido
            if self.espera >= ESPERA_ENTRE_RUTAS:
                self.empezar_episodio(self.ruta_siguiente())

        while self.acumulado >= config.DT and not self.episodio.terminado():
            self.episodio.paso_con_red(self.red)
            if self.episodio.motivo == "choque":
                self.choques += 1
            self.acumulado = self.acumulado - config.DT

    # ---------------------------------------------------------------------
    # El bucle de la ventana. Sigue hasta que se cierra o se aprieta Esc.
    # ---------------------------------------------------------------------
    def correr(self):
        reloj = pygame.time.Clock()
        while True:
            # 1. Cuánto tiempo pasó desde el cuadro anterior, a 60 cuadros por segundo como mucho
            transcurrido = reloj.tick(60) / 1000

            # 2. Teclas y clics
            if not self.atender_teclas():
                return

            # 3. Si se eligió otra ruta, empieza un episodio nuevo
            if self.selector.cambio:
                self.empezar_episodio(self.ruta_siguiente())
                self.selector.cambio = False

            # 4. Simular y dibujar
            self.avanzar(transcurrido)
            self.dibujar()

    # ---------------------------------------------------------------------
    # Dibuja un cuadro: la ruta, el recorrido, los rayos, el auto y el panel.
    # ---------------------------------------------------------------------
    def dibujar(self):
        self.pantalla.blit(self.fondo, (0, 0))
        episodio = self.episodio

        # 1. La ruta, con los puntos ya pasados en verde
        self.dibujar_ruta(episodio.ruta, alcanzados=episodio.k)

        # 2. El recorrido hecho hasta ahora
        if len(episodio.recorrido) > 1:
            pygame.draw.lines(self.pantalla, AZUL, False, [self.a_pantalla(x, y) for x, y in episodio.recorrido], 1)

        # 3. Los rayos al borde de la calle
        if self.ver_rayos:
            self.dibujar_rayos()

        # 4. El auto: rojo si chocó, naranja si está fuera de su carril, azul si va bien
        if episodio.motivo == "choque":
            color = ROJO
        elif episodio.invasion_carril:
            color = NARANJA
        else:
            color = AZUL
        self.dibujar_auto(episodio.auto, color)

        # 5. El panel y la ayuda para elegir rutas
        self.dibujar_panel()
        self.selector.dibujar(self.pausa)
        pygame.display.flip()

    # ---------------------------------------------------------------------
    # Dibuja los rayos del auto, desde el auto hasta el borde de la calle.
    # ---------------------------------------------------------------------
    def dibujar_rayos(self):
        auto = self.episodio.auto
        for angulo_relativo, distancia in zip(config.ANGULOS_RAYOS, self.episodio.sensores.medir_rayos(auto)):
            angulo = auto.angulo + angulo_relativo
            fin = (auto.x + math.cos(angulo) * distancia, auto.y + math.sin(angulo) * distancia)
            pygame.draw.line(self.pantalla, NARANJA, self.a_pantalla(auto.x, auto.y), self.a_pantalla(*fin), 1)

    # ---------------------------------------------------------------------
    # El panel de la derecha: estado, velocidad, tiempo, puntos, nota y teclas.
    # ---------------------------------------------------------------------
    def dibujar_panel(self):
        episodio = self.episodio
        if episodio.motivo is not None:
            estado = episodio.motivo.upper()
        elif self.pausa:
            estado = "PAUSA"
        else:
            estado = "EN MARCHA"

        renglones = [
            (estado, self.fuente_grande),
            ("", self.fuente),
            (f"Auto: {episodio.auto.velocidad * 3.6:.0f} km/h", self.fuente),
            (f"Reproducción: x{VELOCIDADES_REPRODUCCION[self.indice_velocidad]:g}", self.fuente),
            (f"Volante: {math.degrees(episodio.auto.volante):.0f}°", self.fuente),
            (f"Tiempo: {episodio.tiempo:.1f} de {episodio.tiempo_maximo:.0f} s", self.fuente),
            (f"Puntos: {episodio.k} de {len(episodio.ruta.puntos)}", self.fuente),
            (f"Fuera del carril: {episodio.tiempo_fuera_del_carril:.1f} s", self.fuente),
            (f"Nota: {nota_de_ruta(episodio.resultado()):.3f}", self.fuente),
            (f"Choques: {self.choques}", self.fuente),
            ("", self.fuente),
            ("Espacio: pausa", self.fuente),
            ("Arriba/abajo: ritmo", self.fuente),
            ("R: reiniciar ruta", self.fuente),
            ("N: volver a aleatorio", self.fuente),
            ("S: rayos", self.fuente),
            ("Esc: salir", self.fuente),
        ]
        self.escribir_panel(renglones)


# -------------------------------------------------------------------------
# Ver manejar una red en tiempo real.
#
# Maneja rutas al azar una atrás de otra. Al terminar una, la muestra un segundo
# y pasa a la siguiente (o repite la manual, si se eligió una). La ventana queda
# abierta hasta que se cierra o se aprieta Esc.
#
# Recibe: el mapa, la red y la semilla de las rutas al azar.
# -------------------------------------------------------------------------
def ver_simulacion(mapa, red, semilla=0):
    try:
        VentanaSimulacion(mapa, red, semilla).correr()
    finally:
        cerrar_ventana()


# =========================================================================
# ENTRENAMIENTO EN VIVO: TODA LA POBLACIÓN A LA VEZ
# =========================================================================

# -------------------------------------------------------------------------
# VentanaEntrenamiento: toda la población maneja la misma ruta a la vez,
# cada auto en su propia copia del mapa (no chocan entre sí).
#
#   generacion        número de la generación que se está mostrando
#   pasos_por_cuadro  cuántos pasos de simulación se hacen antes de dibujar
#   pausa             True mientras está en pausa
#   anterior          (mejor nota, llegadas del mejor, rutas) de la generación anterior
#   etapas            en el entrenamiento por etapas, el nombre de la etapa de cada mapa
#   etapa             la etapa que se está entrenando, o None si se entrena en un solo mapa
# -------------------------------------------------------------------------
class VentanaEntrenamiento(Ventana):

    def __init__(self, mapa, etapas=None):
        super().__init__(mapa)
        pygame.display.set_caption("Conducción autónoma por neuroevolución - entrenamiento")
        self.generacion = 0
        self.pasos_por_cuadro = 2
        self.pausa = False
        self.anterior = None
        self.etapas = {id(etapa.mapa): etapa.nombre for etapa in etapas or []}
        self.etapa = None

    # ---------------------------------------------------------------------
    # Atiende teclas y clics.
    # Devuelve: False si hay que terminar (Esc o ventana cerrada).
    # ---------------------------------------------------------------------
    def atender_teclas(self):
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.display.quit()
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    pygame.display.quit()
                    return False
                elif evento.key == pygame.K_SPACE:
                    self.pausa = not self.pausa
                elif evento.key == pygame.K_UP:
                    self.pasos_por_cuadro = min(64, self.pasos_por_cuadro * 2)
                elif evento.key == pygame.K_DOWN:
                    self.pasos_por_cuadro = max(1, self.pasos_por_cuadro // 2)
                elif evento.key == pygame.K_n:
                    self.selector.volver_al_azar()
            elif evento.type == pygame.MOUSEBUTTONDOWN and evento.button == 1:
                self.selector.clic(evento.pos, self.pausa)
        return True

    # ---------------------------------------------------------------------
    # Hace manejar a toda la población en pantalla. Reemplaza al bucle de
    # fitness.evaluar: entrenar() la recibe como simular_poblacion.
    #
    # La población maneja las rutas sorteadas de la generación, una atrás de otra.
    # Si se elige una ruta a mano, se muestra como exhibición y no cuenta para la nota
    # (ver exhibir). Si se eligió a mitad de la generación, la generación vuelve a empezar
    # con el azar como estaba al principio: el resultado es el mismo que sin exhibición.
    #
    # Recibe: la población, los casos (pares mapa, ruta) y el generador al azar.
    # Devuelve: los resultados de cada individuo, o None si se cerró la ventana.
    # ---------------------------------------------------------------------
    def simular_poblacion(self, poblacion, casos, rng):
        redes = [Perceptron(config.CAPAS, cromosoma=cromosoma) for cromosoma in poblacion]

        # Por etapas, el primer caso es siempre de la etapa actual. Si cambió de etapa,
        # la numeración de las generaciones vuelve a empezar.
        etapa = self.etapas.get(id(casos[0][0]))
        if etapa != self.etapa:
            self.etapa, self.generacion, self.anterior = etapa, 0, None

        # El estado del azar al principio de la generación, para poder repetirla igual
        azar_inicial = rng.bit_generator.state

        while True:
            rng.bit_generator.state = azar_inicial
            resultados = [[] for _ in poblacion]
            pidio_exhibicion = False

            for numero_ruta, (mapa, ruta) in enumerate(casos):
                # 1. Un episodio por individuo, todos en la misma ruta. Por etapas, la última
                #    ruta puede ser de una etapa anterior: la ventana pasa a ese mapa.
                self.usar_mapa(mapa)
                episodios = [Episodio(mapa, ruta, rng=rng) for _ in poblacion]

                # 2. Simular hasta que todos terminen
                salida = self.correr_episodios(redes, episodios, numero_ruta, len(casos))
                if salida == "cerrada":
                    return None
                if salida == "exhibir":
                    pidio_exhibicion = True
                    break

                # 3. Guardar el resultado de cada individuo en esta ruta
                for k in range(len(poblacion)):
                    resultados[k].append(episodios[k].resultado())

            if not pidio_exhibicion:
                break

            # Se eligió una ruta a mano: se muestra, y la generación vuelve a empezar
            if not self.exhibir(redes, self.mapa):
                return None

        # 4. Lo que se muestra de esta generación en el panel de la siguiente
        self.anotar_generacion(resultados)
        return resultados

    # ---------------------------------------------------------------------
    # Anota la nota y las llegadas del mejor de la generación que terminó,
    # para mostrarlas en el panel, y pasa a la generación siguiente.
    # ---------------------------------------------------------------------
    def anotar_generacion(self, resultados):
        notas = [nota(resultados_individuo) for resultados_individuo in resultados]
        indice_mejor = int(np.argmax(notas))
        self.anterior = (notas[indice_mejor], contar_llegadas(resultados[indice_mejor]), len(resultados[0]))
        self.generacion = self.generacion + 1

    # ---------------------------------------------------------------------
    # Simula y dibuja una tanda de episodios hasta que terminen todos.
    #
    # Recibe: las redes, un episodio por red, y qué ruta es (para el panel).
    #         Con exhibicion=True, el panel lo indica y no se aceptan rutas nuevas.
    # Devuelve: "terminada", "cerrada" si se cerró la ventana, o "exhibir" si se
    #           eligió una ruta a mano y hay que mostrarla.
    # ---------------------------------------------------------------------
    def correr_episodios(self, redes, episodios, numero_ruta, cantidad_rutas, exhibicion=False):
        reloj = pygame.time.Clock()
        while self.pausa or not all(episodio.terminado() for episodio in episodios):
            reloj.tick(60)
            if not self.atender_teclas():
                return "cerrada"

            # ¿Se eligió una ruta a mano? Durante una exhibición se ignora.
            if self.selector.cambio:
                self.selector.cambio = False
                if self.selector.ruta_manual is not None and not exhibicion:
                    return "exhibir"

            if not self.pausa:
                for paso in range(self.pasos_por_cuadro):
                    # A velocidad alta, un cuadro puede simular muchos pasos.
                    # Cada 4 pasos se revisan las teclas, para que Esc y la pausa respondan rápido.
                    if paso > 0 and paso % 4 == 0:
                        if not self.atender_teclas():
                            return "cerrada"
                        if self.pausa or self.selector.cambio:
                            break

                    # Un paso para cada auto que sigue en marcha
                    for red, episodio in zip(redes, episodios):
                        if not episodio.terminado():
                            episodio.paso_con_red(red)
                    if all(episodio.terminado() for episodio in episodios):
                        break

            self.dibujar_poblacion(episodios, numero_ruta, cantidad_rutas, exhibicion)
        return "terminada"

    # ---------------------------------------------------------------------
    # Exhibición: toda la población maneja una vez la ruta elegida a mano.
    #
    # No cuenta para la nota y maneja sin ruido en los sensores, así no usa el azar
    # del entrenamiento. La ruta se muestra una sola vez; después el entrenamiento
    # sigue con sus rutas sorteadas.
    #
    # Devuelve: False si se cerró la ventana.
    # ---------------------------------------------------------------------
    def exhibir(self, redes, mapa):
        ruta = self.selector.ruta_manual
        self.selector.ruta_manual = None
        episodios = [Episodio(mapa, ruta) for _ in redes]
        if self.correr_episodios(redes, episodios, 0, 1, exhibicion=True) == "cerrada":
            return False
        self.selector.mensaje = "Fin de la exhibición. Sigue el entrenamiento."
        return True

    # ---------------------------------------------------------------------
    # Dibuja un cuadro del entrenamiento: la ruta, todos los autos y el panel.
    # En una exhibición, el panel lo indica en lugar de la generación.
    #
    # Colores de los autos: azul = en marcha, naranja = fuera de su carril,
    # verde = llegó a B, rojo = chocó, gris = se le acabó el tiempo.
    # ---------------------------------------------------------------------
    def dibujar_poblacion(self, episodios, numero_ruta, cantidad_rutas, exhibicion=False):
        self.pantalla.blit(self.fondo, (0, 0))
        ruta = episodios[0].ruta

        # 1. La ruta. Cada auto va por su punto, así que no se marca ninguno como pasado.
        self.dibujar_ruta(ruta, radio_punto=2)

        # 2. Primero los autos que terminaron, para que los que siguen en marcha queden arriba
        for episodio in episodios:
            if episodio.terminado():
                self.dibujar_auto(episodio.auto, color_al_terminar(episodio))

        # 3. Los que siguen en marcha
        for episodio in episodios:
            if not episodio.terminado():
                self.dibujar_auto(episodio.auto, NARANJA if episodio.invasion_carril else AZUL)

        # 4. El panel y la ayuda para elegir rutas
        self.dibujar_panel_poblacion(episodios, numero_ruta, cantidad_rutas, exhibicion)
        self.selector.dibujar(self.pausa)
        pygame.display.flip()

    # ---------------------------------------------------------------------
    # El panel de la derecha del entrenamiento: la generación, cuántos autos
    # siguen, llegaron, chocaron o invadieron el carril, la generación anterior
    # y las teclas.
    # ---------------------------------------------------------------------
    def dibujar_panel_poblacion(self, episodios, numero_ruta, cantidad_rutas, exhibicion):
        en_marcha = sum(not episodio.terminado() for episodio in episodios)
        llegaron = sum(episodio.motivo == "llegada" for episodio in episodios)
        chocaron = sum(episodio.motivo == "choque" for episodio in episodios)
        invaden = sum(episodio.tiempo_fuera_del_carril > 0 for episodio in episodios)

        if exhibicion:
            encabezado = [("EXHIBICIÓN", self.fuente_grande),
                          ("Ruta elegida a mano: no cuenta", self.fuente)]
        else:
            encabezado = [(f"GENERACIÓN {self.generacion}", self.fuente_grande),
                          (f"Ruta {numero_ruta + 1} de {cantidad_rutas}", self.fuente)]
        if self.etapa is not None:
            mapa_de_la_ruta = self.etapas.get(id(self.mapa), "")
            encabezado.insert(0, (f"Etapa: {self.etapa}", self.fuente_grande))
            if mapa_de_la_ruta != self.etapa:
                encabezado.append((f"Repaso de una etapa anterior: {mapa_de_la_ruta}", self.fuente))
        renglones = encabezado + [
            (f"Tiempo: {episodios[0].tiempo:.1f} s", self.fuente),
            ("", self.fuente),
            (f"En marcha: {en_marcha}", self.fuente),
            (f"Llegaron a B: {llegaron}", self.fuente),
            (f"Chocaron: {chocaron}", self.fuente),
            (f"Invadieron carril: {invaden}", self.fuente),
            ("", self.fuente),
        ]
        if self.anterior is not None:
            mejor_nota, llegadas_del_mejor, rutas_totales = self.anterior
            renglones.append(("Generación anterior", self.fuente_grande))
            renglones.append((f"Mejor nota: {mejor_nota:.3f}", self.fuente))
            renglones.append((f"El mejor llegó en {llegadas_del_mejor} de {rutas_totales}", self.fuente))
            renglones.append(("", self.fuente))
        if self.pausa:
            renglones.append(("PAUSA", self.fuente_grande))
        renglones.append((f"Velocidad: x{self.pasos_por_cuadro}", self.fuente))
        renglones.append(("Flechas arriba/abajo: velocidad", self.fuente))
        renglones.append(("Espacio: pausa", self.fuente))
        renglones.append(("Esc: terminar", self.fuente))
        self.escribir_panel(renglones)


# -------------------------------------------------------------------------
# Color de un auto que terminó: verde si llegó a B, naranja si llegó pero
# salió de su carril en algún momento, rojo si chocó y gris si se le acabó el tiempo.
# -------------------------------------------------------------------------
def color_al_terminar(episodio):
    if episodio.motivo == "llegada":
        return NARANJA if episodio.tiempo_fuera_del_carril > 0 else VERDE_FUERTE
    if episodio.motivo == "choque":
        return ROJO
    return GRIS_OSCURO


# -------------------------------------------------------------------------
# Entrenar mirando a toda la población. Al cerrar la ventana, el entrenamiento termina.
#
# Se guarda igual que entrenar() y recibe casi los mismos argumentos. Para continuar
# una corrida en el mismo mapa, la numeración de las generaciones sigue desde donde quedó.
#
# Devuelve: la mejor red y la historia.
# -------------------------------------------------------------------------
def ver_entrenamiento(mapa, semilla=0, generaciones=config.GENERACIONES, tamano_poblacion=config.TAMANO_POBLACION,
                      continuar_desde=None, tipo_cruce=None, tipo_mutacion=None):
    ventana = VentanaEntrenamiento(mapa)
    if continuar_desde is not None:
        _, historia_anterior, datos = cargar(continuar_desde)
        if datos["parametros"]["mapa"] == mapa.nombre:
            ventana.generacion = len(historia_anterior)

    # Si no se eligen cruce o mutación, entrenar() usa los de config.py
    opciones = {"simular_poblacion": ventana.simular_poblacion, "continuar_desde": continuar_desde}
    if tipo_cruce is not None:
        opciones["tipo_cruce"] = tipo_cruce
    if tipo_mutacion is not None:
        opciones["tipo_mutacion"] = tipo_mutacion

    try:
        mejor, historia = entrenar(mapa, semilla, generaciones, tamano_poblacion, **opciones)
    finally:
        cerrar_ventana()
    return mejor, historia


# -------------------------------------------------------------------------
# Entrenar por etapas mirando a toda la población, como en el informe:
# manzana → 2x2 → 3x3. La ventana cambia de mapa con cada etapa, y también
# para la ruta de repaso de una etapa anterior. Al cerrar la ventana, el
# entrenamiento termina; cada etapa queda guardada como en entrenar_por_etapas().
#
# Recibe: las etapas (de etapas_santa_fe) y los mismos argumentos que entrenar_por_etapas().
# Devuelve: la mejor red y los identificadores de las etapas guardadas.
# -------------------------------------------------------------------------
def ver_entrenamiento_por_etapas(etapas, semilla=0, tamano_poblacion=config.TAMANO_POBLACION,
                                 tipo_cruce=config.TIPO_CRUCE, tipo_mutacion=config.TIPO_MUTACION,
                                 continuar_desde=None):
    ventana = VentanaEntrenamiento(etapas[0].mapa, etapas)
    try:
        mejor, identificadores = entrenar_por_etapas(etapas, semilla, tamano_poblacion, tipo_cruce, tipo_mutacion,
                                                     continuar_desde=continuar_desde,
                                                     simular_poblacion=ventana.simular_poblacion)
    finally:
        cerrar_ventana()
    return mejor, identificadores
