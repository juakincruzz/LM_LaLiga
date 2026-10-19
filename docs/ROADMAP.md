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

---

## 1. Correcciones pendientes

- `simulation_service.run_simulation` lanza `ValueError` sin modelo; debería ser
  `ModelNotTrainedError` (`core/errors.py`).
- `next_season_service._load_season_df` lee el CSV con `pd.read_csv` en vez de pasar por
  `core/dataset.py`.
- `SeasonSimulator.monte_carlo_simulation` fija `tail(3)` como descenso; debería salir de
  la configuración.
- Al reiniciar el servidor, un trabajo que estaba `running` se queda así para siempre (las
  `BackgroundTasks` no sobreviven al reinicio). Barrido en el `lifespan`: marcarlos `failed`.
- Los dos entornos virtuales del repo (`.venv/` y `venv/`) están rotos (ver CLAUDE.md).

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
| **Elo + forma continua (11 features)**, XGBoost reg., toda la historia | **0,978** | 0,582 | 52,6 % | 0,039 |
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

**Principio de temporada (jornadas 1–6)**, el momento en que estamos ahora:

| | Jornadas 1–6 | Jornadas 7–38 |
|---|---|---|
| Modelo actual | 1,062 (= no saber nada) | 1,029 |
| Elo + forma continua | 0,936 | 0,986 |
| Mercado | 0,919 | 0,969 |

Confirmado en vivo: reentrenado sobre la 26/27 (78 partidos), el modelo actual acierta un
43,6 %, por debajo de «siempre local» (44,9 %), con log-loss 1,083.

### Conclusiones

1. **El modelo actual sobreajusta.** Solo con regularizar XGBoost (árboles de
   profundidad 2, más árboles con menos aprendizaje, submuestreo) la log-loss baja de
   1,034 a 1,000, sin tocar las features.
2. **El reinicio por temporada lo ciega en las primeras jornadas.** Las 35 features
   empiezan cada verano a cero, y hasta la jornada 6 el modelo no sabe más que las
   frecuencias. Un **Elo** arrastrado entre temporadas (con regresión a la media) y una
   **forma continua** (medias exponenciales de goles y tiros a puerta, sin reinicio)
   resuelven exactamente eso: 1,062 → 0,936 en las jornadas 1–6.
3. **Menos features, mejor modelo.** Elo + forma continua (11 features) supera a las 35
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
   Revisarla después de los cambios anteriores, con un diagrama de fiabilidad.

### Plan recomendado para el modelo

| Paso | Cambio | Ganancia medida |
|---|---|---|
| 1 | XGBoost regularizado | 1,034 → 1,000 |
| 2 | Elo y forma continua (goles y tiros a puerta) en `core/state.py`, sin reinicio de temporada | → 0,978; jornadas 1–6: 1,062 → 0,936 |
| 3 | Entrenar con toda la historia disponible (desde la 14/15) | incluido en el 0,978 |
| 4 | Métrica de la UI = log-loss walk-forward de varias temporadas, no solo la actual | Hoy el test son 78 partidos: un número muy ruidoso |
| 5 | Decidir lo de las cuotas (conclusión 5) | → 0,964 si se incluyen |

Para los pasos 1–3, ojo con las invariantes de CLAUDE.md: las features nuevas van en
`LeagueState` (la única implementación) y el snapshot de `tests/test_feature_parity.py`
se regenera a propósito.

### Más adelante

- **xG real.** El CSV de la 26/27 trae por primera vez `HxG`/`AxG`, pero ninguna temporada
  anterior lo tiene, así que no se puede entrenar con él todavía. Conviene empezar a
  guardarlo; los tiros a puerta hacen de sustituto mientras tanto.
- **Ascendidos:** Elo inicial con datos de Segunda (`SP2` en football-data) en vez de un
  valor fijo.
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

Pendiente, por valor:

1. **Clasificación real actual** (hoy solo hay simulaciones): `LeagueState.standings()`.
2. **Cuotas junto a la predicción.** `football-data.co.uk/fixtures.csv` trae las cuotas de
   la jornada en curso, con los mismos nombres de equipo: «el modelo dice 76 %, el mercado 71 %».
3. **Historial de aciertos** jornada a jornada, público: cuántos favoritos acertó el
   modelo y su log-loss frente al mercado.
4. **Distribución de posiciones finales** (equipos × posición) con las simulaciones.
5. **Evolución por jornada** de las probabilidades de título, Europa y descenso.
6. **«¿Y si…?»**: fijar un resultado pendiente y volver a simular.
7. **Página de equipo**, **actualización automática** tras cada jornada y **zonas
   europeas configurables** en `leagues.yaml`.

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
| **A** | Plan del modelo, pasos 1–4 | Hoy el modelo no supera a «siempre local» a principio de temporada |
| **B** | §3.1–3.3 + §1 | Clasificación real y transparencia frente al mercado |
| **C** | Decisión sobre cuotas (paso 5) y Monte Carlo vectorizado | |
| **D** | Resto de §3 y §4 | Profundidad |
