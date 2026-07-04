# LM_LaLiga - Predicción y análisis de La Liga

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![pandas](https://img.shields.io/badge/Data-pandas-green)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-red)

Proyecto de **Machine Learning y análisis de datos** aplicado a partidos y clasificación de **La Liga española**.

El repositorio combina procesamiento de datos históricos, creación de características, entrenamiento de modelos de clasificación y proyección de la clasificación final a partir de datos actuales.

---

## Descripción

El objetivo del proyecto es analizar datos de partidos de La Liga y construir herramientas para:

- Cargar y limpiar datos históricos de partidos.
- Crear características estadísticas para modelos de Machine Learning.
- Entrenar modelos para predecir el resultado de partidos.
- Obtener o actualizar datos de clasificación.
- Proyectar la clasificación final de la temporada.
- Estimar un campeón probable a partir de puntos actuales y ritmo de puntuación.

> Nota: el modelo de Machine Learning está orientado a predecir resultados de partidos. La predicción del campeón se realiza mediante una proyección de clasificación basada en puntos por partido, no mediante una predicción directa entrenada sobre campeones históricos.

---

## Tecnologías utilizadas

- Python 3.8+
- pandas
- numpy
- scikit-learn
- XGBoost
- LightGBM
- requests
- BeautifulSoup4
- matplotlib
- seaborn
- Jupyter

---

## Componentes principales

| Componente | Descripción |
|---|---|---|
| `run_full_pipeline.py` | **Pipeline unificado**: descarga datos si faltan, entrena el modelo, proyecta el campeón y ejecuta Monte Carlo. |
| `run_pipeline.py` | Ejecuta el pipeline de Machine Learning con validación temporal. |
| `run_temporal_validation.py` | Ejecuta walk-forward validation para evaluar el modelo de forma robusta. |
| `run_monte_carlo.py` | Ejecuta simulaciones Monte Carlo para estimar probabilidades de campeón, Champions y descenso. |
| `scripts/data_manager.py` | Verifica la disponibilidad de datos locales y descarga automáticamente desde football-data.org si faltan. |
| `scripts/load_data.py` | Carga CSVs históricos, normaliza columnas, limpia datos y guarda datasets procesados. |
| `scripts/preprocess.py` | Genera características estadísticas de equipos y partidos sin data leakage. |
| `scripts/model_evaluation.py` | Proporciona división temporal simple y walk-forward validation. |
| `scripts/season_simulator.py` | Simula los partidos restantes de la temporada usando el modelo de ML. |
| `models/predictor.py` | Implementa modelos de clasificación con Random Forest, Gradient Boosting, XGBoost y LightGBM. |
| `scripts/fetch_updated_data.py` | Obtiene datos actualizados desde football-data.org y, opcionalmente, RapidAPI. |
| `scripts/scraper_laliga.py` | Obtiene clasificación desde ESPN/BBC y usa datos manuales como respaldo. |
| `predict_champion_latest.py` | Muestra clasificación actual y proyecta campeón según puntos por partido. |
| `update_and_predict.py` | Actualiza datos, calcula clasificación y proyecta el resultado final. |

---

## Flujo de trabajo

```text
Verificación de datos locales
        |
        v
Descarga automática desde football-data.org (si faltan datos)
        |
        v
Carga y limpieza de datos
        |
        v
Feature engineering
        |
        v
Entrenamiento de modelos
        |
        v
Evaluación e importancia de variables
        |
        v
Proyección de clasificación
```

---

## Modelos implementados

La clase `LaLigaPredictor` permite entrenar distintos modelos de clasificación:

| Modelo | Uso |
|---|---|
| Random Forest | Clasificador basado en ensamblado de árboles. |
| Gradient Boosting | Modelo secuencial de boosting. |
| XGBoost | Modelo principal usado en el pipeline completo. |

El objetivo del modelo es clasificar el resultado de un partido:

```text
-1 -> Victoria visitante
 0 -> Empate
 1 -> Victoria local
```

Para XGBoost, estas clases se codifican internamente como `0`, `1` y `2`.

---

## Características generadas

El módulo de preprocesamiento (`scripts/preprocess.py`) crea características usando **únicamente información disponible antes del partido**, evitando *data leakage*:

### Estadísticas acumuladas por equipo
- Partidos jugados, victorias, derrotas y empates.
- Puntos por partido.
- Goles a favor y en contra por partido.

### Forma reciente
- Puntos por partido en los últimos 3 y 5 partidos.
- Rendimiento como local del equipo local.
- Rendimiento como visitante del equipo visitante.

### Diferencias entre equipos
- Diferencia de puntos por partido.
- Diferencia de porcentaje de victorias.
- Diferencia de goles a favor y en contra.
- Diferencia de forma reciente.
- Factor campo (*home advantage*).

### Contexto adicional
- **Días de descanso**: días transcurridos desde el último partido de cada equipo.
- **Enfrentamientos directos (H2H)**: resultados históricos entre el local y el visitante.

> **Importante:** no se usan goles, tiros, corners, tarjetas ni ninguna estadística del propio partido para predecir su resultado. Esto garantiza que el modelo pueda usarse para predecir partidos futuros.

---

## Datos

El proyecto trabaja con datos en formato CSV dentro de `data/raw/`.

Si ejecutas `run_full_pipeline.py` o `run_pipeline.py`, los datos se descargan **automáticamente** desde `football-data.org` si no hay archivos locales. También puedes añadir manualmente datos históricos.

Archivos esperados por el cargador:

```text
data/raw/SP1.csv
data/raw/football_matches.csv
```

Fuentes de datos soportadas:

- `football-data.org`, sin clave de API (descarga automática).
- RapidAPI / API-Football, si se configura `RAPIDAPI_KEY`.
- ESPN o BBC Sport para obtener clasificación actual mediante scraping.

Los CSVs grandes y modelos entrenados no se versionan necesariamente en Git, ya que están ignorados en `.gitignore`.

---

## Estructura del proyecto

```text
LM_LaLiga/
├── README.md
├── requirements.txt
├── run_full_pipeline.py      # Pipeline unificado: datos + modelo + campeón + Monte Carlo
├── run_pipeline.py           # Pipeline de ML con split temporal
├── run_temporal_validation.py # Walk-forward validation
├── run_monte_carlo.py        # Simulaciones Monte Carlo
├── predict_champion_latest.py
├── update_and_predict.py
│
├── scripts/
│   ├── data_manager.py       # Verificación y descarga automática de datos
│   ├── model_evaluation.py   # Validación temporal
│   ├── season_simulator.py   # Simulación de temporada con ML
│   ├── load_data.py
│   ├── preprocess.py         # Feature engineering sin data leakage
│   ├── fetch_updated_data.py
│   └── scraper_laliga.py
│
├── models/
│   └── predictor.py
│
└── data/
    ├── raw/
    └── processed/
```

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/juakincruzz/LM_LaLiga.git
cd LM_LaLiga
```

Crear y activar un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate
```

En Windows:

```bash
venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Uso

### Ejecutar el pipeline completo (recomendado)

```bash
python run_full_pipeline.py
```

Este script unificado realiza todo el flujo en un solo comando:

1. Verifica si existen datos locales.
2. Descarga automáticamente la temporada actual desde football-data.org si faltan datos.
3. Carga y limpia los datos.
4. Genera características.
5. Entrena y compara **Random Forest, Gradient Boosting, XGBoost y LightGBM**.
6. Selecciona automáticamente el mejor modelo.
7. Evalúa el mejor modelo usando **validación temporal**.
8. Calcula la clasificación actual.
9. Simula los partidos restantes con el modelo de ML para predecir el campeón.
10. Ejecuta **Monte Carlo** (500 simulaciones) para estimar probabilidades de campeón, Champions y descenso.
11. Muestra también la proyección lineal clásica como comparación.
12. Guarda el modelo, la proyección, la simulación y las probabilidades en `data/processed/`.

> **Nota:** ya no es necesario descargar datos manualmente. Si no tienes CSVs locales, el pipeline los descarga automáticamente.

### Ejecutar el pipeline de Machine Learning

```bash
python run_pipeline.py
```

Este script realiza:

1. Verificación/descarga automática de datos si faltan.
2. Carga de datos.
3. Limpieza.
4. Feature engineering.
5. **División temporal train/test** (cronológica, no aleatoria).
6. Entrenamiento con XGBoost.
7. Evaluación del modelo.
8. Guardado del modelo entrenado.

### Validación temporal con walk-forward

```bash
python run_temporal_validation.py
```

Este script evalúa el modelo de forma más robusta mediante **walk-forward validation**:

- Entrena con una ventana inicial de datos históricos.
- Predice los siguientes `step` partidos.
- Avanza la ventana y repite el proceso.
- Muestra accuracy por fold, accuracy promedio, matriz de confusión global y reporte de clasificación.

Esto evita el data leakage temporal y da una estimación más fiable del rendimiento real del modelo.

### Monte Carlo para probabilidades de campeón

```bash
python run_monte_carlo.py
```

Por defecto ejecuta **1000 simulaciones estocásticas** de los partidos restantes y estima:

- Probabilidad de cada equipo de ganar La Liga.
- Probabilidad de entrar en Champions League (top 4).
- Probabilidad de descender (últimos 3).
- Distribución de puntos finales esperados (media, desviación, mínimo, máximo).

También genera un gráfico de barras con las probabilidades de campeón.

Parámetros opcionales:

```bash
# Cambiar el número de simulaciones
python run_monte_carlo.py --sims 2000

# Simular desde una fecha específica (por ejemplo, 1 de marzo de 2026)
python run_monte_carlo.py --start-date 2026-03-01
```

> **Nota:** si la temporada ya está finalizada, las probabilidades serán 0% o 100% porque no quedan partidos por simular. Usa `--start-date` para analizar escenarios intermedios.

### Obtener clasificación y proyectar campeón

```bash
python predict_champion_latest.py
```

Este script intenta obtener la clasificación actual y proyecta los puntos finales de cada equipo según su ritmo de puntuación.

### Actualizar datos y proyectar clasificación

```bash
python update_and_predict.py
```

Este script descarga datos actualizados, calcula la clasificación actual y genera una proyección final.

### Configurar RapidAPI de forma opcional

```bash
export RAPIDAPI_KEY="tu_clave"
```

En Windows PowerShell:

```powershell
$env:RAPIDAPI_KEY="tu_clave"
```

---

## Salidas generadas

El proyecto puede generar:

```text
data/raw/SP1.csv                                                # Datos descargados automáticamente
data/processed/laliga_processed.csv                             # Dataset limpio y con features
data/processed/projected_standings.csv                          # Proyección lineal de clasificación
data/processed/simulated_standings.csv                          # Clasificación simulada con ML
data/processed/simulated_matches.csv                            # Partidos simulados con probabilidades
data/processed/monte_carlo_champion_probabilities.csv           # Probabilidades de campeón
data/processed/monte_carlo_probabilities.png                    # Gráfico de probabilidades de campeón
models/laliga_predictor.pkl                                     # Mejor modelo entrenado
```

Estos archivos pueden no aparecer en el repositorio porque dependen de la ejecución local y están ignorados por Git.

---

## Limitaciones

- La predicción de campeón combina simulación con el modelo de ML, Monte Carlo y proyección lineal; la simulación asigna goles simplificados a los resultados predichos.
- El modelo de resultados de partido alcanza una accuracy realista (~45-55%) porque no utiliza datos del propio partido y se evalúa con validación temporal. Predecir resultados de fútbol es inherentemente difícil.
- Las probabilidades de Monte Carlo dependen de la calidad del modelo y del número de simulaciones; con más simulaciones se reduce la varianza de las estimaciones.
- La disponibilidad de datos externos depende de las fuentes usadas y de posibles cambios en sus páginas o APIs.
- El scraping puede dejar de funcionar si ESPN o BBC modifican la estructura HTML.
- Los datos de `football-data.org` pueden tener un ligero retraso respecto a la jornada actual.

---

## Aprendizajes principales

Este proyecto permite trabajar conceptos prácticos de análisis de datos y Machine Learning:

- Limpieza y normalización de datasets deportivos.
- Feature engineering aplicado a fútbol sin data leakage.
- Clasificación multiclase de resultados de partidos.
- Entrenamiento, comparación y selección de modelos supervisados.
- Validación temporal y walk-forward validation para series deportivas.
- Simulación de temporadas y Monte Carlo para estimar probabilidades.
- Uso de datos externos mediante descarga, API o scraping.
- Proyección de clasificaciones deportivas.
- Organización de un pipeline de datos en Python.

---

## Autor

**Joaquín Cruz Lorenzo**  
GitHub: [@juakincruzz](https://github.com/juakincruzz)
