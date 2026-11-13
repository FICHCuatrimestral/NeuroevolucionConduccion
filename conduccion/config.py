"""Parámetros del proyecto.

Todas las distancias están en metros, los tiempos en segundos y los ángulos en radianes.
Los valores son los de la configuración de base del informe.
Se pueden cambiar desde un notebook con parametros.usar_parametros, sin tocar este archivo.
"""

import math

# =========================================================================
# AUTO
# Modelo de bicicleta cinemático (Kong et al., 2015). Ver auto/auto.py.
# =========================================================================

# Distancia del centro de masa al eje delantero y al eje trasero
L_F = 1.1
L_R = 1.7

# Tamaño del rectángulo del auto. Solo se usa para los choques, el carril y el dibujo.
LARGO_AUTO = 4.5
ANCHO_AUTO = 1.8

# Duración de un paso de simulación
DT = 0.1

# Ángulo máximo del volante
VOLANTE_MAXIMO = math.radians(35)

# Aceleración con el acelerador a fondo y frenada con el freno a fondo, en m/s²
ACELERACION_MAXIMA = 2.0
FRENADA_MAXIMA = 4.0

# Velocidad máxima: 11 m/s son unos 40 km/h
VELOCIDAD_MAXIMA = 11.0

# Cuánto puede girar el volante por segundo, en rad/s. None = sin límite.
GIRO_VOLANTE_MAXIMO = None

# Agarre de las ruedas: la aceleración lateral máxima al doblar, en m/s². None = sin límite.
ACELERACION_LATERAL_MAXIMA = 6.0

# =========================================================================
# CALLES
# =========================================================================

# Ancho de cada carril. Todas las calles tienen dos carriles, uno por mano.
ANCHO_CARRIL = 3.5

# Tolerancia al medir si el auto salió de su carril
MARGEN_CARRIL = 0.05

# Lado de cada celda de la matriz del mapa
RESOLUCION_MAPA = 0.25

# =========================================================================
# SENSORES
# =========================================================================

# Los 7 rayos que miden la distancia al borde de la calle, respecto de hacia dónde mira el auto
ANGULOS_RAYOS = [math.radians(a) for a in (-90, -60, -30, 0, 30, 60, 90)]

# Hasta dónde llega cada rayo, y cada cuánto se mira la matriz a lo largo del rayo
ALCANCE_RAYOS = 30.0
PASO_RAYOS = 0.25

# Desvío del ruido normal que se suma a cada entrada durante el entrenamiento
RUIDO_SENSORES = 0.02

# Entradas de la red: 7 rayos + velocidad + 2 de brújula + distancia al punto + desvío + 4 miradas al carril
CANTIDAD_ENTRADAS = 16

# A cuántos metros por delante del auto se mira la dirección del carril
MIRADAS_CARRIL = [3.0, 6.0]

# =========================================================================
# RUTA Y PUNTOS DE CONTROL
# =========================================================================

# Distancia entre puntos de control
SEPARACION_PUNTOS = 10.0

# Si es True, además se pone un punto justo en cada giro del carril
PUNTOS_EN_GIROS = True

# Desde cuántos grados un quiebre del carril cuenta como giro
ANGULO_GIRO = 30.0

# A menos de esta distancia, el punto cuenta como alcanzado (Togelius y Lucas, 2006)
RADIO_PUNTO = 6.0

# Además del próximo punto, cuántos de los siguientes se miran. Si el auto alcanza uno de ellos,
# los que se salteó cuentan como pasados: así no queda trabado por abrirse en un giro.
PUNTOS_SALTEABLES = 2

# Las rutas más cortas que esto no se usan: serían de una sola cuadra
LARGO_MINIMO_RUTA = 150.0

# Tiempo máximo de una ruta = largo / VELOCIDAD_REFERENCIA + TIEMPO_EXTRA
VELOCIDAD_REFERENCIA = 4.0
TIEMPO_EXTRA = 10.0

# =========================================================================
# RED
# =========================================================================

# Neuronas de cada capa: entradas, capa oculta y 2 salidas (volante y acelerador)
CAPAS = [CANTIDAD_ENTRADAS, 16, 2]

# Desvío de los pesos al azar de la primera generación
DESVIO_PESOS_INICIALES = 0.1

# =========================================================================
# ALGORITMO GENÉTICO
# =========================================================================

TAMANO_POBLACION = 50

# Cuántos individuos compiten en cada torneo
TAMANO_TORNEO = 3

# Cuántos de los mejores pasan sin cambios a la generación siguiente
ELITE = 2

# Cruce: "neurona", "pesos", "blx" o "ninguna"
PROBABILIDAD_CRUCE = 0.8
TIPO_CRUCE = "neurona"

# Cuánto se agranda el intervalo entre los padres en el cruce BLX-alfa
ALFA_BLX = 0.5

# Mutación: "gaussiana" o "neurona"
TIPO_MUTACION = "gaussiana"

# Mutación gaussiana: probabilidad de que mute cada peso
PROBABILIDAD_MUTACION = 0.05

# Mutación por neurona: cuántas neuronas cambian (Montana y Davis, 1989)
NEURONAS_A_MUTAR = 2

# Desvío σ del ruido que suma la mutación. Queda fijo durante toda la corrida.
SIGMA = 0.3

# =========================================================================
# ENTRENAMIENTO
# =========================================================================

# Rutas que maneja cada individuo en cada generación. Son las mismas para toda la población.
RUTAS_POR_GENERACION = 3

# Rutas fijas para comparar redes de distintas generaciones y elegir la mejor
RUTAS_REFERENCIA = 20

GENERACIONES = 100

# =========================================================================
# APTITUD (fitness). Ver aprendizaje/fitness.py.
# =========================================================================

# Cuánto descuenta la nota el tiempo fuera del carril
PENALIZACION_CARRIL = 0.8

# Lambda: cuánto pesa llegar rápido. 0 = no se premia.
PREMIO_TIEMPO = 0.0
