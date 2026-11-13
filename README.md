# Conducción autónoma en calles de Santa Fe mediante neuroevolución

Proyecto final de Inteligencia Computacional (FICH, UNL).
Autores: M. Bertona, A. Greminger y A. Nanzer.

Un auto simulado aprende a ir de una esquina A a una esquina B por calles reales de Santa Fe, sin chocar y por su carril. Lo maneja un perceptrón multicapa cuyos pesos se buscan con un algoritmo genético; no hay retropropagación ni ejemplos de manejo.

- **Entorno:** calles de OpenStreetMap (osmnx), rutas con A\* y puntos de control sobre el carril derecho.
- **Vehículo:** modelo de bicicleta cinemático (Kong et al., 2015), con un límite de aceleración lateral.
- **Controlador:** perceptrón 16 → 16 → 2 con tanh. Entradas: 7 rayos al borde de la calle, velocidad, brújula al próximo punto, distancia al punto, desvío del carril y la dirección del carril 3 m y 6 m adelante. Salidas: volante y acelerador.
- **Algoritmo genético:** torneo de 3, élite de 2, cruce por neurona, por pesos o BLX-α, y mutación gaussiana o por neurona.
- **Entrenamiento:** directo en un mapa, o por etapas de dificultad creciente: manzana → 2×2 → 3×3 → centro.
- **Evaluación:** 58 recortes de 1 km de toda la ciudad, en 4 niveles de dificultad. 10 de ellos son de prueba: la red nunca entrena ahí.

## Estructura

```
conduccion/                 el paquete de Python
├── config.py               todos los parámetros (configuración de base del informe)
├── parametros.py           cambiar parámetros desde un notebook
├── mapa/                   mapas (grafo, superficie y matriz), rutas y descarga desde OpenStreetMap
├── auto/                   física del vehículo y sensores
├── simulacion/             un episodio: manejar una ruta hasta llegar, chocar o agotar el tiempo
├── aprendizaje/            red, aptitud, algoritmo genético, entrenamiento, etapas, evaluación y guardado
└── visualizacion/          gráficos (matplotlib) y simulador en tiempo real (pygame)
notebooks/
├── 1_componentes.ipynb     cada parte por separado, sin entrenar
├── 2_entrenamiento.ipynb   elegir parámetros y entrenar
└── 3_evaluacion.ipynb      analizar una red entrenada
scripts/simulador.py        menú para ver manejar una red o entrenar viendo a la población
mapas/                      mapas ya descargados (no hace falta internet)
resultados/                 un subdirectorio por entrenamiento
informe/                    informe en LaTeX y el script de sus figuras
presentacion/               presentación (Quarto) y el script de sus videos
docs/                       los artículos citados en el informe, con el nombre de su clave en refs.bib
```

## Instalación

Hace falta Python 3.11 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

Todos los comandos se corren desde la carpeta del proyecto.

### Notebooks

```bash
jupyter notebook notebooks/
```

Se leen en orden. El primero ya está ejecutado y muestra cómo funciona cada parte. El segundo entrena, y el tercero analiza el resultado.

### Entrenar

Desde `2_entrenamiento.ipynb`, o con el menú:

```bash
python scripts/simulador.py
```

Cada entrenamiento se guarda en `resultados/<fecha>_semilla<n>/` después de cada generación:

| Archivo | Contenido |
|---|---|
| `checkpoint.npz` | población, mejor red, historia y estado del generador al azar; sirve para continuar |
| `mejor.npy` | el mejor cromosoma |
| `historia.json` | notas, llegadas y σ de cada generación |
| `config.json` | semilla y todos los parámetros con que se entrenó |

Con 50 individuos, un entrenamiento por etapas tarda varias horas.

### Evaluar en los recortes de la ciudad

```bash
python -m conduccion.aprendizaje.evaluacion --red <identificador>
python -m conduccion.aprendizaje.evaluacion --red <identificador> --uso prueba --niveles 1 2
```

Sin `--red` se usa el entrenamiento más reciente. El resumen se guarda en `resultados/evaluaciones/`.

### Ver manejar una red

`python scripts/simulador.py`, opción 1, o la última sección de `3_evaluacion.ipynb`. Teclas: Espacio = pausa, N = ruta nueva, R = reiniciar, S = sensores, flechas = velocidad, Esc = salir.

### Reproducir las figuras

```bash
python informe/figuras/figuras_informe.py
python presentacion/figuras_presentacion.py      # necesita ffmpeg y las redes de REDES
```

### Volver a descargar los mapas

No hace falta: los mapas están en `mapas/`. Para regenerarlos se necesita internet.

```bash
python -m conduccion.mapa.descargar_mapa         # 2x2, 3x3 y centro (la manzana sale del 3x3)
python -m conduccion.mapa.descargar_ciudad       # los 58 recortes; tarda unos 20 minutos
```

## Referencias principales

- Togelius, J. y Lucas, S. M. (2005). *Evolving controllers for simulated car racing.*
- Togelius, J. y Lucas, S. M. (2006). *Evolving robust and specialized car racing skills.*
- Kong, J., Pfeiffer, M., Schildbach, G. y Borrelli, F. (2015). *Kinematic and dynamic vehicle models for autonomous driving control design.*
- Montana, D. J. y Davis, L. (1989). *Training feedforward neural networks using genetic algorithms.*
- Yao, X. (1999). *Evolving artificial neural networks.*
- Eshelman, L. J. y Schaffer, J. D. (1993). *Real-coded genetic algorithms and interval-schemata.*
- Gomez, F. y Miikkulainen, R. (1996). *Incremental evolution of complex general behavior.*
- Boeing, G. (2025). *Modeling and analyzing urban networks and amenities with OSMnx.*

La bibliografía completa está en `informe/refs.bib`.
