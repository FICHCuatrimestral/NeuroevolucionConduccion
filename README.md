# Conducción autónoma en calles de Santa Fe mediante neuroevolución

Proyecto final de Inteligencia Computacional (FICH, UNL).
Autores: M. Bertona, A. Greminger y A. Nanzer.

Un auto simulado aprende a ir de una esquina A a una esquina B por calles reales de Santa Fe, sin chocar y por su carril. Lo maneja un perceptrón multicapa cuyos pesos se buscan con un algoritmo genético; no hay retropropagación ni ejemplos de manejo.

- **Entorno:** calles de OpenStreetMap (osmnx), rutas con A\* y puntos de control sobre el carril derecho.
- **Vehículo:** modelo de bicicleta cinemático (Kong et al., 2015), con un límite de aceleración lateral.
- **Controlador:** perceptrón 16 → 16 → 2 con tanh. Entradas: 7 rayos al borde de la calle, velocidad, brújula al próximo punto, distancia al punto, desvío del carril y la dirección del carril 3 m y 6 m adelante. Salidas: volante y acelerador.
- **Algoritmo genético:** torneo de 3, élite de 2, cruce por neurona, por pesos o BLX-α, y mutación gaussiana o por neurona.
- **Entrenamiento:** directo en un mapa, o por etapas: manzana → 2×2 → 3×3. Desde el notebook también se puede sumar el centro como etapa.
- **Evaluación:** las 20 rutas fijas del centro y las de seis barrios de prueba, donde la red nunca entrena.
  Los barrios salen de 58 recortes de 1 km de toda la ciudad (`mapas/recortes.json`), salvo Guadalupe, que se bajó aparte.
  Los recortes tienen un nivel de dificultad según la entropía de orientación de sus calles (Boeing, 2019, *Urban spatial order*);
  sirvió para elegirlos, pero el informe no lo usa.

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
scripts/
├── simulador.py            menú: ver manejar una red, o entrenar en un mapa o por etapas, viendo a la población
├── Abrir simulador.bat     abre el menú con doble clic (Windows); Abrir simulador.command, en macOS
└── correr_comparaciones.py las seis corridas del informe
mapas/                      mapas ya descargados (no hace falta internet)
resultados/comparaciones/   las seis corridas del informe, con sus redes y evaluaciones
informe/                    informe en LaTeX (main.tex, Informe.pdf y el zip para Overleaf) y los scripts de sus figuras
presentacion/               presentación (Quarto), su guion y los scripts de sus figuras y videos
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

### El simulador

```bash
python scripts/simulador.py
```

En Windows también se abre con doble clic en `scripts/Abrir simulador.bat`. El menú tiene cinco opciones:

| Opción | Qué hace |
|---|---|
| 1 | Ver manejar una red guardada: las seis del informe o cualquiera entrenada desde el menú, en el mapa que se elija |
| 2 | Entrenar en un solo mapa, viendo a toda la población manejar a la vez |
| 3 | Lo mismo, rápido y sin ventana |
| 4 | Entrenar por etapas como en el informe (manzana → 2×2 → 3×3), viendo a toda la población; la ventana cambia de mapa con cada etapa |
| 5 | Lo mismo, rápido y sin ventana |

Al entrenar se eligen el auto y la nota (con o sin límite de agarre, con o sin premio por tiempo), el tamaño de la capa oculta
(8, 16 o 32), la semilla, la población, el cruce y la mutación. Por etapas, además, el máximo de generaciones: el del informe
(100, 50 y 50) o uno corto para mirar un rato (20, 10 y 10). En la ventana del entrenamiento, el panel muestra la etapa y
la generación, y avisa cuando la ruta es el repaso de una etapa anterior.

Con ventana, el ruido de los sensores se sortea en otro orden que sin ventana, así que con la misma semilla los números
no salen exactamente iguales; sin ruido, son idénticos.

### Entrenar

Desde `2_entrenamiento.ipynb`, o con el simulador (opciones 2 a 5).

Cada entrenamiento se guarda en `resultados/<fecha>_semilla<n>/` después de cada generación:

| Archivo | Contenido |
|---|---|
| `checkpoint.npz` | población, mejor red, historia y estado del generador al azar; sirve para continuar |
| `mejor.npy` | el mejor cromosoma |
| `historia.json` | notas, llegadas y σ de cada generación |
| `config.json` | semilla y todos los parámetros con que se entrenó |

Con 50 individuos, un entrenamiento por etapas tarda varias horas.

### Reproducir las corridas del informe

```bash
python scripts/correr_comparaciones.py uno_por_vez          # las 6 corridas a la vez, unas 4 horas y media
```

Cada corrida queda en `resultados/comparaciones/<combinación>/semilla0/`, con una carpeta por etapa.
En las corridas guardadas, la carpeta que empieza a las 06:38 es el comienzo de la manzana: ese primer
intento se cortó a las pocas generaciones y la manzana siguió desde ahí (ver `origen` en su `config.json`).

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
python informe/figuras/figuras_informe.py          # métodos del informe y algunas imágenes de la presentación
python presentacion/video_comparaciones.py         # video de los barrios y img/barrios.json
python informe/figuras/figuras_resultados.py       # números de las tablas y figuras de resultados
python presentacion/figuras_metodo.py              # problema, etapas, esquina, algoritmo y cromosoma
python presentacion/figuras_antecedentes.py        # figuras de los papers, redibujadas
python presentacion/figuras_presentacion.py        # videos de manejo y velocidad
quarto render presentacion/presentacion.qmd        # la presentación en HTML
```

Los de resultados necesitan las corridas de `resultados/comparaciones/`, y los videos, ffmpeg (por ejemplo, el de `imageio-ffmpeg`).

### Volver a descargar los mapas

No hace falta: los mapas están en `mapas/`. Para regenerarlos se necesita internet.

```bash
python -m conduccion.mapa.descargar_mapa         # 2x2, 3x3 y centro (la manzana sale del 3x3)
python -m conduccion.mapa.descargar_ciudad       # los 58 recortes; tarda unos 20 minutos
```

Los mapas de Guadalupe y Candioti no salen de estos scripts: se bajaron aparte y quedaron en `mapas/`.

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
