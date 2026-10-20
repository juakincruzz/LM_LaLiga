# LM_LaLiga — Predicción y simulación de La Liga

![Python](https://img.shields.io/badge/Python-3.14-blue)
![FastAPI](https://img.shields.io/badge/Web-FastAPI-009688)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-red)

Aplicación web con estética retro MS-DOS/VGA que descarga los resultados de **La Liga**,
entrena un modelo de resultados de partido y simula el resto de la temporada (o proyecta
la siguiente). El proyecto se centra solo en la liga española.

---

## Qué hace

La portada muestra, para quien solo quiere consultar:

- **Próxima jornada** — cada partido con las probabilidades de 1/X/2 y el marcador más
  probable según el modelo.
- **Cara a cara** — elige dos equipos y verás el pronóstico, los 5 marcadores más
  probables, los goles esperados, la forma reciente y los precedentes.
- **Clasificación** real de la temporada, con la forma de cada equipo.
- **Proyección final** simulada, con probabilidades de título y descenso.
- **Historial de aciertos** jornada a jornada: qué partidos acertó el modelo (sin haber
  visto nunca esos resultados) frente a las casas de apuestas.
- **Ficha de cada equipo** (pulsando su nombre) — posición y próximo partido, estado de
  forma, puntos fuertes y débiles frente al resto de la liga, posible 11 y mejores
  jugadores (datos de Understat) y palmarés.

Y las acciones para mantenerla al día (cada una con un atajo de teclado, 1–4):

- **Actualizar datos** — descarga los resultados (football-data.co.uk), el calendario
  (openfootball) y las alineaciones (Understat) de la temporada en curso y los vuelca en
  SQLite. La primera vez tarda unos minutos por las alineaciones.
- **Entrenar modelo** — entrena un XGBoost con todas las temporadas desde la 14/15 y
  reporta sus métricas **fuera de muestra** (walk-forward sobre las 4 últimas
  temporadas), junto a las del mercado de apuestas y otras referencias.
- **Simular temporada** — Monte Carlo de los partidos que quedan por jugar, usando el
  modelo entrenado. Solo tiene sentido si la temporada no ha terminado.
- **Proyectar la próxima temporada** — fuera de temporada: ratings de ataque/defensa a
  partir de la temporada acabada, ascendidos configurados, Monte Carlo con goles
  Poisson. No usa el modelo de ML (no hay partidos que predecir todavía).

## Sobre la precisión del modelo — léase antes de esperar un 90 %

Un modelo de resultados de fútbol entrenado solo con estadística previa al partido (sin
alineaciones, sin lesiones, sin cuotas en tiempo real) tiene un techo real en torno al
**53-55 %** de accuracy — que es aproximadamente lo que ya aciertan las casas de
apuestas siguiendo a su favorito. El modelo actual acierta un **52,6 %** con una log-loss
de **0,980**, frente a 54,6 % y 0,961 del mercado (media de 22/23 a 25/26, cada temporada
predicha con un modelo entrenado solo con las anteriores). La portada muestra esas
métricas **junto a sus referencias** (siempre gana el local, clase mayoritaria, mercado
de apuestas, no saber nada) precisamente para que un 52 % se lea como "está donde se
espera que esté un modelo así", no como "está roto". Si alguna vez ves un número por
encima del 90 %, sospecha: probablemente se está midiendo sobre datos que el modelo ya
ha visto.

---

## Arranque rápido

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m app.seed                 # crea el esquema y registra La Liga
python run_web.py                  # http://127.0.0.1:8000
```

Desde ahí: pulsa **Actualizar datos** (tecla 1) y luego **Entrenar modelo** (tecla 2).
Ambos se ejecutan en segundo plano y la página se actualiza sola mientras corren.

### Desde la terminal, sin levantar el servidor

`manage.py` llama a los mismos servicios que usa la web — no reimplementa nada:

```bash
python manage.py update
python manage.py train
python manage.py simulate --sims 500 --seed 42
python manage.py project
python manage.py evaluate     # última evaluación guardada, sin reentrenar
```

### Tests

```bash
pytest                 # suite completa (los lentos, marcados `slow`, se excluyen)
pytest -m slow         # el test de integración de red real (descarga + entrena)
```

---

## Arquitectura

```
core/       librería compartida: LeagueState (features/estado/clasificación),
            carga y normalización de CSV, evaluación, goles Poisson, errores
app/        web FastAPI + Jinja2 + HTMX (tema retro en app/static) — routers → services → core
models/     LeaguePredictor (XGBoost/RandomForest/GradientBoosting/LightGBM)
scripts/    orquestación: descarga genérica, simulador de temporada
manage.py   CLI fino sobre app/services — sin lógica propia
config/     leagues.yaml: La Liga, temporada actual, ascendidos
tests/      pytest + fixtures de datos reales congeladas (5 temporadas de SP1)
experimentos/  estudio de fiabilidad del modelo, reproducible
```

`core.state.LeagueState` es la única implementación de las features, el estado por
equipo y la clasificación — tanto el entrenamiento por lotes como la simulación
partido-a-partido recorren el mismo objeto, así que no hay dos copias que puedan
desalinearse. Las features son un **Elo** y una **forma continua** (medias exponenciales
de goles y tiros a puerta) que no se reinician en verano: así el modelo ya sabe algo en
la jornada 1.

El target de resultado es `{-1: visitante, 0: empate, 1: local}` en todo el código;
`LeaguePredictor` lo codifica a `{0,1,2}` internamente para los cuatro modelos por
igual. Las métricas son walk-forward (`core/evaluation.py`); el modelo que se guarda
entrena con todo lo jugado, temporada actual incluida.

Cada acción (actualizar datos, entrenar, simular, proyectar) es un job que corre en segundo plano, serializado con un lock — nada permite
lanzar dos entrenamientos de XGBoost a la vez peleándose por los núcleos. El progreso se
sondea desde el navegador cada 2 segundos vía HTMX.

Documentación más detallada, pensada para quien vaya a tocar el código, en
[CLAUDE.md](CLAUDE.md).

---

## Fuente de datos

- [football-data.co.uk](https://www.football-data.co.uk/) — resultados históricos y
  cuotas, sin clave de API. La Liga es `SP1` en su nomenclatura; ese código se conserva
  como clave en la BD, en `data/raw/SP1/` y en el nombre del modelo.
- [openfootball](https://github.com/openfootball/football.json) — calendario con jornadas.
- [Understat](https://understat.com/) — alineaciones y datos de jugadores (endpoints no
  oficiales; ver `scripts/understat.py`).

## Limitaciones conocidas

- El modelo no usa alineaciones, lesiones ni datos en tiempo real — solo estadística
  acumulada previa al partido. Ver la nota de precisión más arriba.
- `config/leagues.yaml → next_season.promoted_teams` viene con nombres de relleno
  (`"Ascendido 1"`, …) que hay que sustituir por los ascendidos reales (con los nombres de
  football-data.co.uk): hasta entonces la proyección se niega a ejecutarse. Solo se puede
  proyectar con la temporada actual terminada.

Las mejoras estudiadas y su prioridad están en [docs/ROADMAP.md](docs/ROADMAP.md).

## Autor

**Joaquín Cruz Lorenzo**
GitHub: [@juakincruzz](https://github.com/juakincruzz)
