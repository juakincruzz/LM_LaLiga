# Hoja de ruta — LM LaLiga

Estudio de mejoras iniciado el 19-10-2026. Las cifras de esta página salen de datos reales
(BD local, modelo guardado y el experimento reproducible
[`experimentos/fiabilidad_modelo.py`](../experimentos/fiabilidad_modelo.py)), no de estimaciones.

---

## Hecho (19-10-2026)

- ✅ **Desajuste H2H entre entrenamiento y simulación.** El simulador reconstruía el
  estado solo con la temporada actual: veía 0,0 enfrentamientos previos por partido donde
  el entrenamiento veía 2,86. Ahora `scripts/season_simulator.py::build_current_state`
  recorre las mismas temporadas que el entrenamiento (`core.features.build_state`).
- ✅ **Tabla simulada incoherente.** Los puntos salían de una sola simulación y las
  probabilidades de otras 500 (un 4.º con más opciones de título que el 3.º). Ahora todo
  sale del mismo Monte Carlo: puntos y goles medios, ordenados por puntos esperados.
- ✅ **Temporada desfasada.** `current_season` pasa a la 26/27, que está en juego.
- ✅ **La temporada en curso parecía terminada.** El CSV de football-data solo trae
  partidos jugados, así que «pendientes» era siempre 0. Ahora se descarga también el
  calendario de openfootball (jornadas incluidas) y se fusiona por par local-visitante,
  de modo que los aplazados no se duplican.
- ✅ **Próxima jornada y cara a cara** en la portada (ver §3).
- ✅ **El estado de la temporada sumaba todas las temporadas** de la BD (760 partidos en
  una liga de 380), y la portada mostraba la proyección hecha desde la temporada anterior.
  Ahora ambos se filtran por temporada (`simulations.season`).
- ✅ **Predicción 3,6× más rápida.** Un modelo cargado predecía fila a fila con
  `n_jobs=-1`: 0,40 ms frente a 0,11 ms con un hilo. 500 simulaciones de la 26/27 bajan de
  ~6 min (y 1,5 h de CPU) a 39 s.


## Hecho (20-10-2026) — fase A del modelo y correcciones

- ✅ **Modelo nuevo** (pasos 1–4 del plan de §2): XGBoost regularizado, 13 features de Elo
  y forma continua sin reinicio de temporada, historia desde la 14/15 y métricas
  walk-forward en la UI. Resultado real, sobre las 4 últimas temporadas:

  | | Antes | Ahora | Mercado |
  |---|---|---|---|
  | Log-loss walk-forward (22/23–25/26) | 1,034 | **0,980** | 0,961 |
  | Log-loss, jornadas 1–6 | 1,062 | **0,938** | 0,919 |
  | Acierto (22/23–25/26) | 49,8 % | **52,6 %** | 54,6 % |
  | 26/27 hasta hoy (78 partidos): acierto / log-loss | 43,6 % / 1,083 | **52,6 % / 0,926** | — / 0,930 |

  Coincide con lo que predijo el estudio (0,978 y 0,936). En la 26/27 el modelo va a la par
  del mercado, pero con 78 partidos eso todavía es ruido.
- ✅ La tarjeta del modelo muestra la log-loss junto a la del mercado y la de «no saber
  nada», con el test explicado.
- ✅ **Correcciones de §1:** `ModelNotTrainedError` al simular sin modelo; la proyección lee
  por `core/dataset.py`; los descensos salen de `relegated` en `leagues.yaml`; al arrancar,
  los trabajos interrumpidos por un reinicio quedan como fallidos; `.venv/` recreado.
- ✅ **Goles esperados con la forma continua** (marcadores de la jornada y del cara a
  cara, y goles del Monte Carlo). Antes usaban las medias de la temporada y eran neutros
  hasta el 3.er partido. Medido sobre los goles reales de 22/23 a 26/27
  (log-verosimilitud por partido, más alta = mejor):

  | | Total | Jornadas 1–6 | Marcador dado el resultado |
  |---|---|---|---|
  | Medias de temporada (antes) | −2,884 | −3,098 | −1,857 |
  | **Forma continua (ahora)** | **−2,833** | **−2,931** | **−1,830** |
  | Forma continua, multiplicativa | −2,886 | −2,932 | −1,875 |
- ✅ **Cuotas descartadas** como variable del modelo (decisión del 20-10-2026): el modelo
  sigue siendo propio. Siguen usándose solo como referencia en las métricas.
- ✅ **Ficha de equipo, revisión:** fuertes/débiles exige |z| ≥ 0,75 y 5 partidos, y no
  repite medidas de la misma familia (goles totales, en casa y fuera); el posible 11 siempre
  tiene portero.

---

## 2. Fiabilidad del modelo — estudio

### Cómo se ha medido

Validación **walk-forward**: para cada temporada de test (22/23, 23/24, 24/25 y 25/26) se
entrena **solo con temporadas anteriores** y se evalúa la temporada entera; las cifras son
la media de las cuatro (~1.500 partidos). Métrica principal: **log-loss**, porque el Monte
Carlo y la próxima jornada usan las *probabilidades*, no solo el favorito. Más baja = mejor;
1,099 es no saber nada (⅓ a cada resultado).

### Resultados

| Configuración | Log-loss | Brier | Acierto | ECE* |
|---|---|---|---|---|
| Frecuencias fijas (referencia «no saber nada») | 1,062 | 0,641 | 46,3 % | — |
| **Modelo actual** (35 features, XGBoost, 5 temporadas) | **1,034** | 0,613 | 49,8 % | 0,055 |
| Mismas 35 features, XGBoost regularizado | 1,000 | 0,594 | 52,0 % | 0,049 |
| 35 features + Elo, XGBoost regularizado | 0,983 | 0,585 | 53,2 % | 0,042 |
| **Elo + forma continua (13 features)**, XGBoost reg., toda la historia | **0,978** | 0,582 | 52,6 % | 0,039 |
| Elo + forma continua, regresión logística, toda la historia | 0,981 | 0,582 | 54,5 % | 0,050 |
| Elo + forma continua + cuotas, regresión logística | 0,964 | 0,573 | 54,5 % | 0,045 |
| *Mercado (cuotas Bet365, sin margen)* | *0,961* | *0,570* | *54,6 %* | *0,048* |

\* ECE: error de calibración (cuánto se desvía «dice 60 %» de «acierta el 60 %»).

Por temporada (log-loss):

| | 22/23 | 23/24 | 24/25 | 25/26 |
|---|---|---|---|---|
| Frecuencias fijas | 1,053 | 1,076 | 1,071 | 1,049 |
| Modelo actual | **1,065** | 1,022 | 1,036 | 1,012 |
| XGBoost regularizado | 1,024 | 0,981 | 1,010 | 0,985 |
| 35 features + Elo | 1,012 | 0,969 | 0,985 | **0,968** |
| Elo + forma continua | **1,000** | **0,964** | **0,969** | 0,978 |
| Mercado | 0,980 | 0,947 | 0,953 | 0,964 |

Todas las alternativas mejoran al modelo actual en las cuatro temporadas, y en la 22/23 el
modelo actual fue incluso peor que las frecuencias fijas. Entre las dos mejores opciones sin
cuotas el orden varía (en la 25/26 gana «35 + Elo»): la diferencia entre ellas es pequeña;
la que hay con el modelo actual, no.

**Principio de temporada (jornadas 1–6)**, el momento en que estábamos al hacer el estudio:

| | Jornadas 1–6 | Jornadas 7–38 |
|---|---|---|
| Modelo actual | 1,062 (= no saber nada) | 1,029 |
| Elo + forma continua | 0,936 | 0,986 |
| Mercado | 0,919 | 0,969 |

Confirmado en vivo: reentrenado sobre la 26/27 (78 partidos), el modelo de entonces
acertaba un 43,6 %, por debajo de «siempre local» (44,9 %), con log-loss 1,083.

### Conclusiones

1. **El modelo actual sobreajusta.** Solo con regularizar XGBoost (árboles de
   profundidad 2, más árboles con menos aprendizaje, submuestreo) la log-loss baja de
   1,034 a 1,000, sin tocar las features.
2. **El reinicio por temporada lo ciega en las primeras jornadas.** Las 35 features
   empiezan cada verano a cero, y hasta la jornada 6 el modelo no sabe más que las
   frecuencias. Un **Elo** arrastrado entre temporadas (con regresión a la media) y una
   **forma continua** (medias exponenciales de goles y tiros a puerta, sin reinicio)
   resuelven exactamente eso: 1,062 → 0,936 en las jornadas 1–6.
3. **Menos features, mejor modelo.** Elo + forma continua (13 features) supera a las 35
   actuales + Elo. Muchas de las 35 son redundantes o ruido.
4. **Más historia ayuda.** Entrenar con todas las temporadas desde la 14/15 mejora
   frente a usar solo 5.
5. **Con cuotas se iguala al mercado, pero no se le gana** (0,964 frente a 0,961). Es
   una decisión de producto, no técnica:
   - *Sin cuotas*: un modelo propio, honesto, a un ~2 % del mercado.
   - *Con cuotas*: la mejor predicción posible, pero en la práctica repite al mercado.
6. **No hay que forzar empates.** El mercado casi nunca da el empate como favorito
   (0,3 % de los partidos) y los mejores modelos tampoco (1–4 %). El modelo actual
   predice empate en el 13 % y es justo el peor: predecir más empates no lo hace más fiable.
7. **La calibración no es el problema principal** (ECE 0,04–0,05, similar al mercado).
   Revisada con el modelo nuevo el 20-10-2026: ver «Calibración» más abajo. No se corrige.

### Plan para el modelo

| Paso | Cambio | Ganancia medida | Estado |
|---|---|---|---|
| 1 | XGBoost regularizado | 1,034 → 1,000 | ✅ |
| 2 | Elo y forma continua (goles y tiros a puerta) en `core/state.py`, sin reinicio de temporada | → 0,978; jornadas 1–6: 1,062 → 0,936 | ✅ |
| 3 | Entrenar con toda la historia disponible (desde la 14/15) | incluido en el 0,978 | ✅ |
| 4 | Métrica de la UI = log-loss walk-forward de varias temporadas, no solo la actual | número estable (~1.200 partidos en vez de 78) | ✅ |
| 5 | Decidir lo de las cuotas (conclusión 5) | → 0,964 si se incluyen | Descartado: modelo propio |

### Calibración (20-10-2026)

Estudio reproducible: [`experimentos/calibracion.py`](../experimentos/calibracion.py).
Predicciones fuera de muestra de cada temporada desde la 17/18 (cada una con un modelo
entrenado solo con las anteriores) y dos correcciones, cada una ajustada solo con
predicciones de temporadas anteriores a la evaluada.

| Media 22/23–25/26 | Log-loss | Brier | ECE local | ECE empate | ECE visitante |
|---|---|---|---|---|---|
| **Sin calibrar (el modelo actual)** | **0,9799** | **0,5829** | 0,056 | 0,038 | 0,048 |
| Calibración «Dirichlet» (logística sobre log-probabilidades) | 0,9802 | 0,5831 | 0,062 | 0,025 | 0,048 |
| Temperatura | 0,9819 | 0,5842 | 0,056 | 0,030 | 0,055 |

En la 26/27 (78 partidos) tampoco: 0,926 sin calibrar, 0,946 y 0,936 calibrando.

Diagrama de fiabilidad (22/23–25/26): cuando el modelo da al local un 70–80 %, gana el
76,7 % (predicho 75,7 %); al visitante un 60–70 %, gana el 64,5 % (64,4 %). Casi todos los
tramos caen dentro del margen del azar. La única desviación clara es el **empate**: cuando
el modelo le da un 30–40 % (317 partidos), sale el 26,8 % y no el 33,0 %, unos 2,5 errores
estándar. Corregirlo con Dirichlet arregla esa desviación (ECE del empate 0,038 → 0,025),
pero no mejora la log-loss: lo que gana en los empates lo pierde en las victorias locales.

**Decisión: no se calibra.** El modelo ya está bien calibrado, y las correcciones
ajustadas con temporadas anteriores empeoran ligeramente al aplicarlas en la siguiente.
La temperatura ajustada además va bajando (1,28 → 1,16): el sesgo que corrige es de las
temporadas antiguas, con menos historia de entrenamiento, y deja de existir.

### Elo de los ascendidos con datos de Segunda (20-10-2026)

Estudio reproducible: [`experimentos/elo_ascendidos.py`](../experimentos/elo_ascendidos.py)
(descarga Segunda, `SP2`, a `data/raw/SP2/`). Los nombres de equipo de Segunda coinciden
con los de Primera en los 37 ascensos desde la 15/16. Variantes del Elo con el que un
ascendido empieza su primera temporada; parámetros elegidos en validación (17/18–21/22) y
evaluados en test (22/23–25/26), walk-forward. La variante «actual» reproduce exactamente
la matriz de producción (comprobado en el script).

| Log-loss | Validación | Test | Partidos de ascendidos (test, 432) | Ascendidos, jornadas 1–6 |
|---|---|---|---|---|
| **Actual** (1430 si es nuevo; si vuelve, el Elo que dejó) | **1,0125** | **0,9799** | **0,9338** | 0,8978 |
| Fijo: 1430 siempre | 1,0126 | 0,9799 | 0,9352 | 0,8952 |
| 1430 + β · puntos por partido en Segunda (β = 100–400) | 1,0128–1,0153 | 0,9797–0,9812 | 0,9353–0,9375 | 0,8963–0,9028 |
| Elo compartido Primera + Segunda, como clubelo (X = 1300–1450) | 1,0130–1,0144 | 0,9799–0,9811 | 0,9380–0,9400 | 0,8994–0,9118 |

**Decisión: se descarta.** Ninguna variante mejora al Elo actual, ni en general ni en los
partidos de los propios ascendidos; usar Segunda incluso empeora un poco. Lo que hizo un
equipo en Segunda dice poco de cómo le irá en Primera, y el Elo y la forma continua ya se
corrigen en pocas jornadas (los partidos de ascendidos en jornadas 1–6 ya están en 0,90,
mejor que la media). Las diferencias entre variantes son del orden del ruido.

### xG de Understat (20-10-2026)

Comprobación de datos: Understat tiene xG de **todos** los partidos desde la 14/15
(4.638, 380 por temporada), y casan uno a uno con football-data, sin un gol ni una fecha
distintos. Bastan 13 peticiones (el JSON de liga trae el xG). Faltaban 4 nombres antiguos
en `UNDERSTAT` (Córdoba, Eibar, Sporting, Huesca): añadidos. El xG de football-data (solo
26/27) es otro modelo: correlación 0,93–0,95 con Understat, pero ~0,2 goles menos al local.
Si algún día se usa, siempre la misma fuente.

Estudio reproducible: [`experimentos/xg_understat.py`](../experimentos/xg_understat.py).
Forma continua de xG a favor/en contra (α = 0,06, el mejor en validación) añadida a las 13
features. Diferencia de log-loss partido a partido frente al modelo actual, con bootstrap
pareado (negativo = mejor con xG):

| | Media | IC 95 % | P(mejor con xG) |
|---|---|---|---|
| Validación 17/18–21/22 | −0,0002 | [−0,006, +0,006] | 53 % |
| Test 22/23–25/26 | −0,0043 | [−0,009, +0,000] | 97 % |
| Las 8 temporadas | −0,0023 | [−0,006, +0,002] | 88 % |
| 26/27 (78 partidos) | +0,0240 | [+0,006, +0,042] | 0 % |

Sustituir los tiros a puerta por xG da lo mismo (test 0,9745, validación peor). Además, el
**xG de Understat cambia de escala** con los años: 0,98 goles esperados por gol en la 14/15,
1,05–1,10 desde la 21/22 y 1,12 en la 25/26.

**Decisión: no se incorpora.** La mejora media es pequeña (una décima parte de la distancia
al mercado), no aparece en validación, se invierte en la temporada actual y el intervalo
incluye el cero. A cambio, el modelo pasaría a depender de una fuente no oficial con una
escala que se mueve. No se ha probado normalizar el xG por temporada, a propósito: después
de varias variantes, seguir probando hasta que una salga bien sería buscar ruido.

### Más adelante

- **xG en el modelo:** estudiado y descartado de momento (ver «xG de Understat» arriba).
  Revisar si se acumulan temporadas con una escala de xG estable.
- **Modelo de goles Dixon-Coles**, que unificaría los dos simuladores.
- **Comunicar la incertidumbre:** el Monte Carlo ya calcula la desviación de los puntos
  (`points_std`); mostrar un intervalo en la tabla en vez de un número seco.

---

## 3. Producto

Hecho:

- ✅ **Próxima jornada**: 1X2 por partido, marcador más probable y selector de jornada
  (los aplazados siguen accesibles desde el selector).
- ✅ **Cara a cara**: cualquier par de equipos, con pronóstico, los 5 marcadores más
  probables, goles esperados, forma, posición y precedentes.
- ✅ **Jornadas**: `matches.matchday` se rellena desde el calendario.
- ✅ **Ficha de equipo** (`/equipo/{nombre}`): forma, fuertes y débiles, posible 11,
  mejores jugadores (Understat) y palmarés.
- ✅ **Clasificación real actual** en la portada (antes solo había simulaciones), con
  zonas y forma; la simulada pasa a llamarse «Proyección final».
- ✅ **Historial de aciertos** jornada a jornada: cada partido jugado se predice con un
  modelo entrenado solo con temporadas anteriores, frente al mercado (aciertos y
  log-loss), con el detalle de cada partido. A día de hoy: 41/78 (53 %) frente a 55 % del
  mercado; log-loss 0,926 frente a 0,930.

Pendiente, por valor:

1. **Distribución de posiciones finales** (equipos × posición) con las simulaciones.
2. **Evolución por jornada** de las probabilidades de título, Europa y descenso.
3. **«¿Y si…?»**: fijar un resultado pendiente y volver a simular.
4. **Actualización automática** tras cada jornada y **zonas europeas configurables** en
   `leagues.yaml` (los descensos ya lo son).
5. **Desempate por enfrentamientos directos** en la clasificación real, como el
   reglamento de la Liga.

Descartado (20-10-2026): **cuotas junto a cada predicción**. La interfaz no muestra cuotas
por partido; el mercado solo aparece como referencia agregada (tarjeta del modelo e
historial de aciertos).

---

## 4. Técnico

- **Monte Carlo vectorizado:** sigue habiendo una llamada a `predict_proba` por partido y
  por simulación. Avanzar todas las simulaciones a la vez y predecir una matriz por partido
  permitiría pasar de 500 a 5.000–10.000 simulaciones.
- **Fuentes y htmx servidos en local**, para no depender de CDNs.
- **CI** (GitHub Actions) con `pytest` y `ruff`.
- **Limpiar `models/predictor.py`** (los `print` y los modelos que no se usan) y los
  métodos de compatibilidad de `SeasonSimulator`.

---

## Orden propuesto

| Fase | Contenido | Por qué |
|---|---|---|
| ~~A~~ | ~~Plan del modelo, pasos 1–4~~ | ✅ Hecho el 20-10-2026 |
| ~~B~~ | ~~Clasificación real e historial de aciertos~~ | ✅ Hecho el 20-10-2026 (cuotas junto a la predicción: descartado) |
| **C** | Monte Carlo vectorizado | |
| **D** | Resto de §3 y §4 | Profundidad |
