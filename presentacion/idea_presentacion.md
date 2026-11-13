# Idea de la presentación

Borrador para trabajar el contenido. Cada bloque es una diapositiva: qué se ve y qué se dice.

Regla del hilo: cada diapositiva usa solo lo que ya se presentó antes.

---

## Parte 1. La idea

### 1. Conducción autónoma

- **Se ve:** un mapa real de Santa Fe, con un punto A, un punto B y la ruta entre los dos.
- **Pregunta:** ¿se le puede enseñar a un auto a manejar sin decirle cómo?
- **Se dice:** queremos que un auto vaya solo de A a B por calles reales de Santa Fe, sin salirse de la calle y por su carril.

---

## Parte 2. Los antecedentes

### 2. Antecedentes

- **Se ve:** la lista de trabajos, cada uno con su año y la pregunta que responde.
  - Kong et al. (2015): ¿cómo se mueve un auto, con las cuentas más simples?
  - Togelius y Lucas (2005, 2006): ¿se puede enseñar a manejar con un algoritmo evolutivo?
  - Montana y Davis (1989): ¿se puede entrenar una red con un algoritmo genético?
  - Yao (1999): ¿cómo se cruzan y se mutan las redes?
  - Nelson et al. (2009): ¿qué nota se le pone a un viaje?
  - Gomez y Miikkulainen (1996): ¿cómo se aprende algo difícil?
- **Se dice:** de la cátedra traemos el perceptrón multicapa y el algoritmo genético. Lo demás lo sacamos de estos trabajos.

### 3. Kong et al. (2015): el modelo de bicicleta

- **Se ve:** su figura del modelo (redibujada) y las cinco ecuaciones.
- **Se dice:** junta las dos ruedas de cada eje en una. El estado es la posición, el rumbo y la velocidad. Tiene dos entradas, el volante y la aceleración: eso es lo que va a tener que decidir quien maneje, así que define las dos salidas de la red. Kong lo compara con datos de un auto real: predice casi tan bien como un modelo con neumáticos, con muchas menos cuentas.

### 4. Togelius y Lucas (2005, 2006): manejar con un algoritmo evolutivo

- **Se ve:** su pista con paredes, los puntos de control en orden y el auto con sus sensores (redibujada de sus figuras 1 y 2).
- **Se dice:** evolucionan redes que manejan autos de juguete simulados.
  - **La red:** un perceptrón multicapa con una capa oculta, dos salidas y tangente hiperbólica; en sus pruebas, la capa oculta tiene 5 o 7 neuronas (Togelius y Lucas, 2005). Nuestra red tiene la misma forma.
  - **Lo que ve:** la velocidad, la distancia a las paredes y el ángulo al próximo punto de control. Todo medido desde el auto.
  - **La nota:** cuántos puntos de control pasó en orden.
  - **El algoritmo:** una estrategia evolutiva, un algoritmo evolutivo que solo muta, sin cruce. Hay 50 redes; cada una tiene un hijo mutado, y de las 100 quedan las 50 mejores.
  - Entrenan en unas pistas y prueban en otras.

### 5. Montana y Davis (1989): una red entrenada con un algoritmo genético

- **Se ve:** su red escrita como cromosoma, con sus mismos números (redibujada de su figura 1).
- **Se dice:** ponen todos los pesos de la red en fila, como números reales, y los buscan con un algoritmo genético. Les va mejor que con retropropagación. La razón: la retropropagación necesita la respuesta correcta en cada paso y una nota derivable. Al manejar no hay volante correcto, solo una nota al final del viaje, y la nota salta cuando el auto choca. El genético solo necesita saber qué red salió mejor.

### 6. Montana y Davis (1989): mutar y cruzar

- **Se ve:** su figura de la mutación y el cruce sobre cromosomas de números reales (redibujada de su figura 2).
- **Se dice:** la mutación cambia algunos genes de un padre. El cruce arma un hijo con genes de los dos padres, cada uno en su misma posición.

### 7. Yao (1999): cómo se cruzan y se mutan las redes

- **Se ve:** sus dos redes que calculan lo mismo con las neuronas en otro orden, y sus cromosomas distintos (redibujada de sus figuras 3 y 4).
- **Se dice:**
  - **El problema de las permutaciones:** dos redes iguales pueden tener las neuronas en otro orden, y entonces sus cromosomas son distintos. Si se cruzan peso por peso, el hijo junta pedazos de neuronas distintas y puede no andar.
  - **Mutar:** con pesos reales, el operador principal es la mutación gaussiana: sumarle a cada peso un poco de ruido normal.
  - **No cruzar:** los algoritmos que solo mutan sufren menos el problema de las permutaciones. Es lo que hace Togelius.
  - **Cruzar por neurona:** la propuesta de Montana y Davis. Cada neurona del hijo viene completa de un padre.
- **De acá salen las opciones que comparamos:**

| Operador | Qué hace |
|---|---|
| Cruce por neurona (Montana y Davis) | Cada neurona del hijo se copia entera de uno de los dos padres: todos sus pesos de entrada y su umbral |
| Cruce por pesos (uniforme) | Cada peso del hijo se copia del padre o de la madre, al azar, uno por uno |
| Cruce BLX-α (Eshelman y Schaffer, 1993) | Cada peso del hijo se sortea entre los valores de los dos padres, con el intervalo agrandado α = 0,5 de su ancho hacia cada lado |
| Sin cruce (como Togelius) | El hijo es una copia de un padre; solo cambia por la mutación |
| Mutación gaussiana | Cada peso muta con probabilidad 0,05: se le suma un número normal de desvío σ |
| Mutación por neurona (Montana y Davis) | Se eligen 2 neuronas al azar y se les suma ruido normal a todos sus pesos de entrada y a su umbral |

- Con probabilidad 0,2 no hay cruce y el hijo es una copia de un padre, sea cual sea el tipo de cruce.

### 8. Nelson et al. (2009): qué nota ponerle a un viaje

- **Se ve:** su clasificación de notas según cuánto le dicen a la evolución de antemano (redibujada de su tabla 1).
- **Se dice:**
  - Revisan las notas que se usan en robótica evolutiva y las ordenan según cuánto le dicen a la evolución de antemano: desde "agregada" (solo si cumplió la tarea) hasta "con datos de entrenamiento" (copiar un manejo conocido).
  - **Poco conocimiento:** la nota agregada, "llegó o no llegó", deja que la evolución encuentre cualquier solución. Pero al principio ninguna red llega, todas sacan cero y no hay a quién elegir. Nelson lo llama *bootstrap problem*.
  - **Mucho conocimiento:** una nota que premia comportamientos puntuales le dice a la red cómo hacer la tarea, y la limita a lo que el diseñador imaginó.
  - **En el medio, la nota "a medida":** mide cuánto avanzó y además algo de cómo se comportó. La de Togelius, por puntos de control, es de este tipo: distingue redes aunque ninguna llegue.

### 9. Gomez y Miikkulainen (1996): de lo fácil a lo difícil

- **Se ve:** su secuencia de tareas: un enemigo lento, después dos, cada vez más rápidos.
- **Se dice:** una tarea difícil se aprende mejor si se arranca por una fácil y se sube de a poco, sin reiniciar la población.

---

## Parte 3. Nuestro trabajo

### 10. La red y su cromosoma

- **Se ve:** la red final (16 → 16 → 2), sus 16 entradas dibujadas sobre el auto y el cromosoma en fila.
- **Se dice:**
  - **Lo que ve (16 entradas):** 7 rayos al borde de la calle, la velocidad, la dirección y la distancia al próximo punto de control, el desvío del carril y la dirección del carril 3 y 6 m adelante. Todo medido desde el auto, como en Togelius.
  - **Lo que decide (2 salidas):** volante = 35° · o₁; acelerador = 2 m/s² · o₂ si o₂ > 0, y freno = 4 m/s² · o₂ si no. Son las dos entradas del modelo de Kong.
  - **Capa oculta:** 16 neuronas, todas con tangente hiperbólica.
  - **El cromosoma (diapositiva aparte, con su figura):** la red en fila, como Montana y Davis: 16 × 16 = 256 pesos de entrada a la capa oculta, 16 umbrales, 16 × 2 = 32 pesos a la salida y 2 umbrales. En total, 306 números reales.
  - **Se ve en la figura:** los 306 genes en fila con sus cuatro partes, y debajo los primeros 256 armados como la matriz de 16 × 16. La matriz se lee fila por fila; cada neurona es una columna, así que sus 16 pesos quedan salteados cada 16 genes y su umbral va aparte. Por eso el cruce y la mutación por neurona eligen por columna, no por posición.

### 11. Las permutaciones en nuestra red

- **Se ve:** dos padres que son la misma red con las neuronas en otro orden, y el hijo que sale de cruzarlos.
- **Se dice:** con 16 neuronas ocultas, nuestra red se puede escribir de 16! · 2^16 ≈ 1,4 · 10^18 formas distintas (el orden y el signo de cada neurona). El cruce por neurona no lo resuelve, porque empareja las neuronas por posición: solo evita desarmar cada neurona por dentro.

### 12. Nuestra nota

- **Se ve:** la fórmula completa de la nota de una ruta, la misma del informe:

  F = (1 − λ) · (k + p) / K · (1 − c · f) + λ · L · (1 − t / t_max), con c = 0,8. En nuestras corridas λ = 0.

- **Se dice:**
  - **k:** puntos de control pasados. Si el auto alcanza uno de los 2 siguientes al próximo, los que se salteó cuentan como pasados: así no queda trabado por abrirse en un giro.
  - **p:** cuánto avanzó hacia el punto siguiente, de 0 a 1.
  - **K:** total de puntos de la ruta. Llegar a B vale 1.
  - **f:** fracción del tiempo que anduvo fuera de su carril. Se descuenta 0,8 y no 1, para que avanzar por la contramano valga algo más que quedarse quieto.
  - **El viaje termina** si llega a B, si se sale de la calle o si se acaba el tiempo (largo de la ruta / 4 m/s + 10 s).
  - **La nota de la red** es el promedio de F en las rutas que manejó.
  - Es la nota de avance de Togelius con un descuento por carril: una nota "a medida", en palabras de Nelson.
  - Se le puede sumar un premio por llegar rápido, λ · (1 − t / t_max), solo al que llega. En nuestras corridas está apagado: λ = 0.

### 13. Qué buscamos

- **Se ve, primero en palabras:** buscamos los pesos de la red que den la mejor nota promedio en todas las rutas. Como no podemos probarlas todas al entrenar, usamos 3 rutas al azar. Esto hace que la nota sea inestable, que tenga ruido: depende de qué rutas salgan sorteadas. Lo resolvemos confirmando la mejor red con 20 rutas fijas.
- **Y después, la fórmula:** w\* = argmax J(w), con J(w) el promedio de la nota F_r(w) sobre todas las rutas.
- **Abajo, en chico:** w son los 306 pesos; F_r(w), la nota en la ruta r. Cada generación son 50 · 3 + 20 = 170 viajes simulados.

### 14. Nuestro algoritmo

- **Se ve:** un diagrama de flujo, lo más completo posible:
  1. **Inicio:** 50 redes con pesos chicos al azar (normales de desvío 0,1).
  2. **Evaluar:** cada red maneja 3 rutas al azar de la etapa, más 1 ruta de una etapa anterior, las mismas para todas. Su nota es el promedio.
  3. **Confirmar la mejor:** la de mejor nota maneja además las 20 rutas fijas de la etapa. Reemplaza a la mejor guardada si llega en más rutas; si empatan, si tiene mejor nota; y si siguen empatadas, si es más rápida.
  4. **Élite:** las 2 de mejor nota pasan sin cambios.
  5. **Selección:** cada padre sale de un torneo de 3.
  6. **Cruce**, con probabilidad 0,8, y **mutación**, con el σ de la etapa.
  7. **Reemplazo:** los hijos forman la generación siguiente.
  8. **¿Pasa de etapa?** Si la mejor red guardada llega en el 90 % de las 20 rutas fijas (en la manzana, en todas) y ya pasaron 10 generaciones, o si se acabaron las generaciones de la etapa. Si no, vuelve a 2.
  9. **Fin:** la última etapa usa todas sus generaciones. El resultado es la mejor red guardada.
- **Ajustes de cada etapa:**

| Etapa | σ de la mutación | Pasa si la mejor llega en | Generaciones |
|---|---|---|---|
| Manzana | 0,3 | todas las rutas fijas | de 10 a 100 |
| 2×2 | 0,2 | el 90 % | de 10 a 50 |
| 3×3 | 0,15 | (última etapa) | 50 |

- **Se dice, en una línea:** buscamos los 306 pesos de la red que dan la mejor nota promedio en todas las rutas. Como no se pueden probar todas, cada red maneja 3 rutas al azar. Eso le mete ruido a la nota; por eso la mejor se confirma con 20 rutas fijas.

### 15. Nuestras etapas

- **Se ve:** los mapas: la manzana, el recorte de 2×2, el de 3×3 y el centro de Santa Fe.
- **Se dice:** como Gomez, de lo fácil a lo difícil. La población arranca en la manzana, pasa al 2×2 y al 3×3, sin reiniciarse. El centro no es una etapa: queda para probar. Ojo: contiene a los recortes de entrenamiento (cerca del 22 % de sus calles); por eso también se prueba en barrios que no se usaron para nada.

### 16. Los parámetros

- **Se ve:** la tabla de la configuración de base: red 16 → 16 → 2 (pesos iniciales de desvío 0,1, ruido de sensores 0,02); población 50, torneo 3, élite 2; cruce por neurona con probabilidad 0,8; mutación gaussiana con probabilidad 0,05 por gen; 3 rutas por generación y 20 fijas; descuento por carril 0,8 y λ = 0; límite de agarre 6 m/s².

### 17. Lo que agregamos

- **Se ve:** las calles reales con su ruta y sus puntos de control, y el límite de agarre.
- **Lo que no tenían los antecedentes:** Togelius maneja en pistas de juguete con paredes, sin carriles y siempre en el mismo circuito. El modelo de Kong supone que las ruedas nunca patinan: deja doblar a fondo a cualquier velocidad.
- **Se dice:**
  - Calles reales de OpenStreetMap; A\* elige el camino; puntos de control sobre el carril derecho.
  - Un descuento en la nota por andar fuera del carril.
  - Y cuatro cambios para que doble bien, que vienen a continuación.

### 18. Lo que agregamos para que doble bien

- **El problema:** la primera versión doblaba mal, sobre todo a la derecha, que es la curva más cerrada. Llegaba rápido a la esquina y, con puntos de control lejos uno del otro, se salía.
- **Se ve:** la figura de la esquina: un cruce de dos calles de 7 m con la curva más amplia que entra doblando a cada lado (redibujar la del apunte, `p18-radios-en-la-esquina.png`).
- **Se dice:**
  1. **Cuánto se puede doblar en una esquina.** Doblando a la izquierda, la curva más amplia que entra en el cruce tiene 17,9 m de radio; doblando a la derecha, 6,0 m. Con 6 m/s² de aceleración lateral, la velocidad máxima es unos 37 km/h a la izquierda y unos 22 km/h a la derecha. A 40 km/h no entra ninguna esquina.
  2. **Límite de agarre.** Kong deja doblar a fondo a cualquier velocidad. La red sin límite lo aprovechaba: doblaba a 40 km/h con 27,7 m/s², casi 3 g; un auto real patina cerca de 8 m/s². Ahora la aceleración lateral no puede pasar de 6 m/s²: a más velocidad, menos volante. El límite no frena: si llega rápido, el volante no le alcanza y se sale. Frenar antes lo tiene que aprender la red.
  3. **Puntos de control en cada giro.** Antes había un punto cada 15 m, repartidos sin mirar la forma de la ruta, y a veces la esquina quedaba entre dos puntos, lejos de los dos: el auto podía cortarla y la nota no se enteraba. Pasaba en el 11 % de los giros del 2×2, el 16 % del 3×3 y el 18 % del centro. Ahora el carril se corta en cada giro y se reparten puntos cada unos 10 m dentro de cada tramo: siempre hay un punto justo en el giro.
  4. **Mirar el carril adelante.** El próximo punto podía estar hasta 15 m adelante y, en una esquina, quedaba en diagonal: no decía cómo era la curva. Se agregaron dos entradas que miran el carril 3 y 6 m adelante del auto: dicen hacia dónde va el carril y cuánto dobla.
  5. **Puntos que se pueden saltear.** Si al abrirse en un giro el auto alcanza uno de los 2 puntos siguientes, los que se salteó cuentan como pasados: así no queda trabado. (Está en el código, `PUNTOS_SALTEABLES = 2`, pero no en el apunte: falta confirmar cuándo y por qué se agregó.)

---

### 19. El límite de agarre, en cuentas

- **Se ve:** las ecuaciones: a_y = v²/R ≤ a_max, R = l_r / sen β, y de ahí el volante máximo a cada velocidad; y la figura con las curvas sin y con límite, y el volante máximo según la velocidad.
- **Se dice:** por debajo de 5,1 m/s el auto puede doblar a fondo; a 8 m/s el volante queda limitado a 14,9°.

## Parte 4. Resultados

### 20. Tres preguntas

1. ¿Qué hace la red si la física no la limita?
2. ¿Importa el operador?
3. ¿Maneja en calles donde nunca entrenó? (todas se prueban en el centro, que no es una etapa y es más difícil que los mapas de entrenamiento, y en seis barrios que no se usaron para entrenar)

### 21. P1: sin el límite, la red hace trampa

- **Se ve:** los dos videos, sin límite y con límite: la red base y la red sin límite, en la misma ruta del centro que la figura de velocidad del informe. Es la primera vez que se ve un auto entrenado.
- **Se dice:** sin límite dobla a 40 km/h, con 27,6 m/s² de aceleración lateral, casi 3 g. Con límite nunca pasa de 6,3 m/s².

### 22. P1 en números

- **Se ve:** la tabla del centro.

| | Llega | Tiempo medio | Velocidad cerca / lejos de un giro | Aceleración lateral máxima |
|---|---|---|---|---|
| Con límite | 15 / 20 | 111 s | 19 / 18 km/h | 6,3 m/s² |
| Sin límite | 20 / 20 | 56 s | 40 / 37 km/h | 27,6 m/s² |

- **Se dice:** sin límite llega siempre y en la mitad del tiempo, porque aprovecha una física imposible. Con límite va lenta todo el viaje, no solo en las esquinas: como la nota no premia el tiempo, ir lento no le cuesta nada. La red aprende lo que la nota le pide, y nada más.

### 23. P1: velocidad y aceleración lateral en la ruta

- **Se ve:** la velocidad (arriba) y la aceleración lateral (abajo) a lo largo de la ruta de los videos, sin y con límite, con los giros marcados. Es la misma figura del informe.
- **Se dice:** sin límite va a 40 km/h todo el viaje y en cada giro llega a 25 o 27 m/s². Con límite va a unos 19 km/h todo el viaje y nunca pasa de los 6 m/s²: no frena en las esquinas, va lenta siempre. En pruebas preliminares con premio por tiempo (λ = 0,1), una de dos redes sí aprendió a bajar la velocidad antes de cada giro: frenar es posible, pero esta nota no lo pide.

### 24. P2 y P3: operadores y calles nuevas

| Corrida | Qué cambia | Llega en el centro | Tiempo medio |
|---|---|---|---|
| 4743 | nada (la base) | 15 / 20 | 111 s |
| 9082 | cruce por pesos | 20 / 20 | 142 s |
| 8477 | cruce BLX-α | 19 / 20 | 128 s |
| 3623 | sin cruce | 19 / 20 | 135 s |
| 9009 | mutación por neurona | 17 / 20 | 129 s |

- **P3:** todas aprendieron en la manzana, el 2×2 y el 3×3, y llegan en 15 a 20 de 20 rutas del centro, que contiene a esos recortes pero suma diagonales, una avenida curva y cruces de más de cuatro calles.
- **P2:** con una sola semilla, esas diferencias pueden ser suerte. Todos los operadores aprenden; para ordenarlos hacen falta más semillas.

### 25. P3: cada red en un barrio que nunca vio

- **Se ve:** un video con las 6 redes a la vez, cada una manejando en un barrio distinto que no usó para entrenar ni para la prueba del centro. Debajo de cada una, cómo le fue en las 20 rutas fijas de su barrio:

| Red | Barrio (nivel) | Llega | Choca |
|---|---|---|---|
| Base: cruce por neurona | Guadalupe (sin nivel) | 10 de 20 | 10 |
| Cruce por pesos | Roma (2) | 19 de 20 | 1 |
| Cruce BLX-α | Schneider (3) | 12 de 20 | 6 (y 2 sin tiempo) |
| Sin cruce | 12 de Octubre (1) | 20 de 20 | 0 |
| Mutación por neurona | La Guardia (4) | 7 de 20 | 11 (y 2 sin tiempo) |
| Sin límite de agarre | Arenales (4) | 13 de 20 | 7 |

- **Se dice:** todas manejan en calles que nunca vieron; en los barrios más difíciles chocan más. Como cada red está en un barrio distinto, esto muestra que generalizan, no cuál operador es mejor: para eso está la tabla del centro.
- **Cómo se hace:** `python presentacion/video_comparaciones.py`. Elige, para cada red, la ruta más larga a la que llega en hasta 120 s, y la cámara de cada panel sigue a su auto (60 m a cada lado). Los resultados de las 120 simulaciones quedan en `img/barrios.json`.

---

## Parte 5. Cierre

### 26. Conclusiones

1. Sin límite de agarre, la red aprovecha la física. Con el límite maneja dentro de lo posible, pero lento.
2. Todos los operadores aprenden; con una semilla no alcanza para ordenarlos.
3. Lo aprendido en recortes chicos sirve en el centro.

- **Lo nuestro:** juntar las piezas de estos trabajos en calles reales, agregar el límite de agarre y medir qué aporta cada una.
- **Lo que sigue:** más semillas, premio por tiempo para que aprenda a frenar antes de las esquinas, tráfico y semáforos.

---

## Pendientes

- **Puntos que se pueden saltear (diapositiva 18, punto 5).** Está en el código (`PUNTOS_SALTEABLES = 2`), pero no aparece en el apunte. ¿Quién lo agregó y fue por el problema de doblar mal? En el borrador está marcado como «falta confirmar».
- **Diapositiva 17.** Quedó solo lo general: las calles reales y el descuento por carril. Los cambios para doblar bien pasaron a la 18.
- **Ya no hay diapositivas de respaldo:** las que había se integraron a la charla (6, 11, 13, 16, 19 y 23). Con el video de los barrios (25), son 26 diapositivas.
- **Barrios distintos:** en el video, «sin cruce» saca 20 de 20 en 12 de Octubre (nivel 1) y «mutación por neurona» 7 de 20 en La Guardia (nivel 4). La diferencia es sobre todo del barrio, no del operador: decirlo así en la charla.
