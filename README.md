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
- requests
- BeautifulSoup4
- matplotlib
- seaborn
- Jupyter

---

## Componentes principales

| Componente | Descripción |
|---|---|
| `run_pipeline.py` | Ejecuta el pipeline completo: carga, limpieza, generación de características, entrenamiento y evaluación. |
| `scripts/load_data.py` | Carga CSVs históricos, normaliza columnas, limpia datos y guarda datasets procesados. |
| `scripts/preprocess.py` | Genera características estadísticas de equipos y partidos. |
| `models/predictor.py` | Implementa modelos de clasificación con Random Forest, Gradient Boosting y XGBoost. |
| `scripts/fetch_updated_data.py` | Obtiene datos actualizados desde football-data.org y, opcionalmente, RapidAPI. |
| `scripts/scraper_laliga.py` | Obtiene clasificación desde ESPN/BBC y usa datos manuales como respaldo. |
| `predict_champion_latest.py` | Muestra clasificación actual y proyecta campeón según puntos por partido. |
| `update_and_predict.py` | Actualiza datos, calcula clasificación y proyecta el resultado final. |

---

## Flujo de trabajo

```text
Datos CSV / fuentes externas
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

El módulo de preprocesamiento crea características a partir de estadísticas históricas y de partido, entre ellas:

- Victorias del equipo local y visitante.
- Promedio de goles a favor.
- Promedio defensivo.
- Puntos totales acumulados.
- Diferencia de puntos.
- Diferencia de victorias.
- Diferencia de promedio de goles.
- Diferencia defensiva.
- Efectividad de tiro.
- Total de goles del partido.
- xG simplificado a partir de tiros a puerta.

---

## Datos

El proyecto trabaja con datos en formato CSV dentro de `data/raw/`.

Archivos esperados por el cargador:

```text
data/raw/SP1.csv
data/raw/football_matches.csv
```

También puede descargar datos actualizados desde:

- `football-data.org`, sin clave de API.
- RapidAPI / API-Football, si se configura `RAPIDAPI_KEY`.
- ESPN o BBC Sport para obtener clasificación actual mediante scraping.

Los CSVs grandes y modelos entrenados no se versionan necesariamente en Git, ya que están ignorados en `.gitignore`.

---

## Estructura del proyecto

```text
LM_LaLiga/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── predict_champion_latest.py
├── update_and_predict.py
│
├── scripts/
│   ├── load_data.py
│   ├── preprocess.py
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

### Ejecutar el pipeline de Machine Learning

```bash
python run_pipeline.py
```

Este script realiza:

1. Carga de datos.
2. Limpieza.
3. Feature engineering.
4. División train/test.
5. Entrenamiento con XGBoost.
6. Evaluación del modelo.
7. Guardado del modelo entrenado.

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
data/processed/laliga_processed.csv
models/laliga_predictor.pkl
data/raw/laliga_updated.csv
```

Estos archivos pueden no aparecer en el repositorio porque dependen de la ejecución local y pueden estar ignorados por Git.

---

## Limitaciones

- La predicción de campeón es una proyección basada en puntos por partido, no un modelo supervisado entrenado directamente para predecir campeones.
- La disponibilidad de datos externos depende de las fuentes usadas y de posibles cambios en sus páginas o APIs.
- El scraping puede dejar de funcionar si ESPN o BBC modifican la estructura HTML.
- Algunos datos locales esperados deben añadirse manualmente en `data/raw/` si no se descargan desde las fuentes externas.
- El xG usado es una aproximación simplificada basada en tiros a puerta.

---

## Aprendizajes principales

Este proyecto permite trabajar conceptos prácticos de análisis de datos y Machine Learning:

- Limpieza y normalización de datasets deportivos.
- Feature engineering aplicado a fútbol.
- Clasificación multiclase de resultados de partidos.
- Entrenamiento y evaluación de modelos supervisados.
- Uso de datos externos mediante descarga, API o scraping.
- Proyección de clasificaciones deportivas.
- Organización de un pipeline de datos en Python.

---

## Autor

**Joaquín Cruz Lorenzo**  
GitHub: [@juakincruzz](https://github.com/juakincruzz)
