# Ficha de equipo y escudos retro — diseño

Fecha: 2026-10-19 · Estado: pendiente de aprobación

## Intención

Hoy la aplicación solo habla de la liga en conjunto: clasificación simulada, predicciones de jornada y
de partido. El usuario quiere poder **entrar en un equipo** y ver cómo está, quién lo sostiene y qué
11 puede sacar en el próximo partido, con el mismo aire de juego de PC de los 90 que el resto de la
web. También quiere que los equipos se reconozcan de un vistazo con un escudo pixelado en todos los
sitios donde aparecen.

El proyecto es personal de momento, pero **se quiere publicar en el futuro**. Eso decide dos cosas:
los escudos son diseño propio (no los oficiales, que son marcas de los clubes) y la fuente de datos de
jugadores queda aislada para poder cambiarla.

## Objetivos

1. **Escudos propios** en estilo pixel EGA para cada equipo, en todos los sitios donde sale un equipo.
2. **Ficha de equipo** en `/equipo/{nombre}` con: estado de forma, puntos fuertes y débiles, posible 11
   del próximo partido, mejores jugadores y palmarés.
3. **Datos de jugadores** desde Understat, temporada actual.

## No objetivos

- Cuotas o cualquier cosa de apuestas en la ficha. (Las dos menciones a casas de apuestas que ya
  existen en la portada y en las predicciones **se quedan como están**: el usuario no ha pedido
  quitarlas.)
- Lesionados: ninguna fuente gratuita los da.
- Histórico de temporadas, cara a cara y lista completa de partidos del equipo (propuestos y
  descartados por ahora).
- Volver a introducir Transfermarkt o el análisis de fichajes.

## Anti-objetivos (fallaría aunque funcione)

- Usar imágenes de escudos oficiales.
- Que algo de la ficha parezca una cuota o un dato de apuestas. En el primer boceto, la línea
  "Últimos 5: 2,0 pts/partido · Temporada: 1,8 ▲ al alza" se leyó como una cuota: **el estado de
  forma son solo las 5 fichas V/E/D**, sin cifras de puntos por partido ni flechas.
- Que la ficha se rompa (500, ventana vacía sin explicación) porque Understat no responde o porque es
  pretemporada.
- Que la ficha haga peticiones a Understat al cargarse: debe leer de caché.
- Duplicar la lógica de clasificación: la posición del equipo sale de `core.state.LeagueState`.

## Decisiones del usuario

- Escudos **estilo B**: silueta común para todos, colores del club, iniciales en tamaño grande.
  Prefería el oficial pixelado, pero eligió B pensando en publicar.
- Cada club con **su propio patrón** dentro de la silueta (rayas, mitades, franja…), para que no se
  parezcan todos. Propuesto por Claude, aprobado por el usuario.
- Fuente de jugadores: **Understat** (sin API oficial; se leen sus endpoints JSON).
- Contenido de la ficha: estado de forma, puntos fuertes y débiles, posible 11, mejores jugadores
  (goles, asistencias, etc.) y **palmarés**.
- Estado de forma = **solo las 5 fichas de colores**.

## Diseño

### 1. Escudos

**Datos.** En [core/equipos.py](../../../core/equipos.py), junto a `NOMBRES_VISIBLES`, un diccionario
por nombre de football-data.co.uk con: dos o tres colores de la paleta EGA (los de
`app/static/css/retro.css`), un patrón y las iniciales (2–3 letras). Patrones mínimos: `liso`,
`rayas_v`, `mitades`, `franja_h`, `diagonal`, `cuartos`. Debe cubrir al menos a todos los equipos de
las temporadas que hay en `data/raw/SP1/` desde 21/22 (los que puede pintar la aplicación), y a los
de 26/27 (Understat los lista hoy: incluye Deportivo, Málaga y Racing). Los colores y patrones son una
interpretación libre de la equipación, no una copia del escudo.

**Dibujo.** Una función Python genera un SVG en una rejilla de 16×16 con la silueta del boceto (recto
arriba, en punta abajo, contorno negro) y `shape-rendering="crispEdges"`. Se expone a Jinja como
global (`escudo(nombre, tam)`) y se inserta **inline** en las plantillas, para que las iniciales
puedan usar la fuente "Press Start 2P" de la página (un SVG cargado por `<img>` no ve las fuentes de
la página). Las iniciales solo se dibujan a partir de 48 px, sobre una franja oscura central.

**Sin datos.** Un equipo que no esté en el diccionario recibe un escudo gris con iniciales derivadas
del nombre. Nunca una imagen rota ni una excepción.

**Dónde.** Clasificación (`partials/standings_table.html`) a 16 px; predicciones de jornada y partido
(`partials/jornada.html`, `partials/prediccion.html`) a 32 px; cabecera de la ficha a 64–80 px. En
todos esos sitios, nombre y escudo enlazan a la ficha.

### 2. Datos de Understat

**Endpoints** (comprobados el 2026-10-19, responden JSON comprimido con gzip a una petición con
`X-Requested-With: XMLHttpRequest` y un User-Agent de navegador):

- `https://understat.com/getLeagueData/La_liga/{año}` → `teams` (historial por partido con xG, xGA,
  presión), `players` (por jugador: partidos, minutos, goles, asistencias, xG, xA, tiros, pases clave,
  amarillas, rojas, posición, `team_title`) y `dates` (partidos con `id`, `isResult`, equipos).
  El año es el de inicio: 26/27 → `2026`.
- `https://understat.com/getMatchData/{id}` → `rosters.h` / `rosters.a`: por jugador, posición,
  minutos, goles, asistencias, tarjetas, `roster_in`/`roster_out` (cambios) y `positionOrder`.

**Un solo módulo** lee Understat (descarga, caché y traducción de nombres). Nada más en el proyecto
conoce sus URLs ni su formato; así se puede cambiar por una API de pago el día que se publique.

**Nombres.** Understat usa sus propios nombres ("Atletico Madrid", "Real Betis", "Athletic Club",
"Celta Vigo", "Deportivo La Coruna", "Racing Santander"…). Se añade a `core/equipos.py` un
diccionario `UNDERSTAT` → nombre de football-data, igual que `OPENFOOTBALL`, y un nombre desconocido
falla con `EquipoDesconocidoError` dentro de la descarga de Understat.

**Caché y momento de descarga.** Understat se descarga dentro del job `update` existente
(`update_league_data`), después de football-data y del calendario, y se guarda bajo
`data/raw/SP1/understat/` (gitignorado como el resto de `data/`). Los partidos ya jugados no
cambian, así que su `getMatchData` se descarga una sola vez y se reutiliza; solo se piden los nuevos,
con una pausa corta entre peticiones. **Si Understat falla, la actualización sigue siendo un éxito** y
lo dice en su `mensaje`, igual que hoy con el calendario. La ficha solo lee la caché.

### 3. Ficha de equipo

**Ruta.** `GET /equipo/{nombre}` con el nombre de football-data (URL-codificado: `Ath%20Madrid`).
Un router nuevo, plantilla nueva que extiende `base.html`. Equipos válidos: los que aparecen en
`matches` de la temporada actual (jugados o programados). Si la temporada actual no tiene ninguna fila
(`SIN_DATOS`), los de la última temporada con datos *(decisión de Claude)*. Cualquier otro nombre →
página 404 existente. Los nombres provisionales de ascendidos ("Ascendido 1"…) de la proyección no
llevan enlace.

**Ventanas, en este orden** (ver boceto aprobado en `.superpowers/brainstorm/`, `ficha.html`, menos la
línea de puntos por partido):

1. **Cabecera.** Escudo grande, nombre visible, posición real, puntos, V/E/D, GF/GC (clasificación de
   los partidos jugados de la temporada actual, calculada con `LeagueState`) y próximo partido
   (rival, casa/fuera, jornada, fecha) desde el calendario. Sin partidos jugados, la cabecera muestra
   solo escudo, nombre y próximo partido si lo hay. Los desempates son los de `LeagueState` (puntos,
   diferencia, goles a favor), no el cara a cara oficial de La Liga: con empates a puntos la posición
   puede diferir de la tabla oficial. Se acepta y no se toca `LeagueState`, porque cambiaría también
   las tablas del simulador *(decisión de Claude)*.
2. **Estado de forma.** Los últimos 5 partidos de liga jugados: ficha de color (verde V, gris E, rojo
   D) con rival abreviado y marcador. El más reciente a la derecha. **Nada más en esta ventana.**
3. **Puntos fuertes y débiles.** Unas 12 medidas por partido del equipo comparadas con las de los 20
   equipos de la temporada actual. Se muestran las 3 en las que el equipo es mejor que la liga y las
   3 en las que es peor, por distancia relativa a la media (p. ej. desviaciones típicas), cada una
   con una frase en castellano, el dato del equipo, la media de la liga y una barra con la media
   marcada. Cada medida sabe si "más" es mejor o peor (goles encajados: menos es mejor).
   Medidas candidatas: goles marcados y encajados (total, en casa, fuera), tiros, % de tiros a puerta,
   conversión (goles / tiros a puerta), goles encajados en segundas partes (sale del marcador al
   descanso), córners, faltas, tarjetas; y xG a favor, xG en contra y goles − xG ("marca más/menos
   de lo que genera"). Todas salen de los CSV de football-data a través de `core.dataset`; no hace
   falta tocar el esquema de la BD. **El xG de equipo también sale de football-data**: su CSV de
   26/27 trae `HxG`/`AxG` (las temporadas anteriores no), que hay que añadir a `COLUMN_MAP`. Así esas
   medidas no dependen de Understat. Si una temporada no trae esas columnas, las medidas xG se omiten
   de la selección sin error *(decisión de Claude)*. Understat solo aporta los datos de jugadores. Los
   CSV de los tests no tienen xG: los tests de esta parte necesitan una fixture que sí lo tenga o
   añadirle las columnas. Las columnas nuevas no pueden entrar en `FEATURE_COLS` ni alterar
   `tests/test_feature_parity.py`.
4. **Posible 11 contra {rival}.** Sobre un campo verde pixelado, 11 jugadores con una camiseta de los
   colores del escudo. Selección: los jugadores con más titularidades en los últimos 5 partidos de
   liga del equipo (desempate: titular en el partido más reciente, luego minutos), con un portero
   obligatorio. Posición en el campo: la que tuvo cada uno en su titularidad más reciente. Junto al
   nombre, entre paréntesis, titularidades en esos 5 partidos. Aviso ⚠ si el jugador lleva un número
   de amarillas en liga que deja a una de la sanción (en La Liga, ciclo de 5) y otro aviso si fue
   expulsado en el último partido. Pie: "Estimado con los titulares de los últimos 5 partidos. No
   incluye lesionados."
5. **Mejores jugadores.** Top 3 del equipo en goles, asistencias, xG, pases clave, minutos y
   tarjetas, temporada actual. Un jugador que cambió de club a mitad de temporada aparece en Understat
   con varios equipos en `team_title`: solo cuenta lo jugado con este.
6. **Palmarés.** Desde `config/palmares.yaml` (nuevo, editado a mano), por nombre de football-data:
   número de títulos y temporada del último, por competición — Liga, Copa del Rey, Supercopa de
   España, Copa de Europa/Champions, UEFA/Europa League, Conference League, Recopa, Supercopa de
   Europa. Se muestran solo las competiciones con títulos, cada una con un trofeo pixelado amarillo.
   Equipo presente con lista vacía → "Sin títulos oficiales todavía". Equipo ausente del fichero →
   "Palmarés no disponible". Rellenarlo para los equipos de 26/27 como mínimo, con datos comprobados
   en Wikipedia.
   **Regla de conteo** *(decisión de Claude)*: solo títulos que reconocen oficialmente la RFEF
   (nacionales) y la UEFA (europeos), con los recuentos de la Wikipedia en español. Los títulos
   discutidos no se suman a otra competición: la Copa de Ferias y la Copa Eva Duarte van en filas
   propias ("Copa de Ferias", "Copa Eva Duarte"), y la Copa de la Coronación de 1902 no cuenta. El
   fichero lleva un comentario arriba con esta regla y la fecha de la última revisión.

**Estados de la temporada** (`season_state`):

- `SIN_DATOS` / `PRETEMPORADA`: ventanas de partidos y jugadores con un mensaje ("Aún no ha jugado
  esta temporada"); escudo, cabecera y palmarés sí se ven.
- `FINALIZADA`: sin próximo partido; la ventana del 11 se titula "11 más habitual" en lugar de
  "contra {rival}".
- Caché de Understat ausente o incompleta: las ventanas 4 y 5 muestran "Datos de jugadores no
  disponibles: pulsa ACTUALIZAR"; el resto de la ficha funciona.

## Restricciones

- Convenciones del repositorio (CLAUDE.md): textos y comentarios en castellano, identificadores en
  inglés; `league_code` string; plantillas compartidas desde `app/templating.py`; lo que HTMX
  sustituya va en `partials/`; nada de frameworks CSS, npm ni build. Estilo en `retro.css`.
- Press Start 2P aplasta las mayúsculas con tilde: úsala solo para nombres sin tilde en mayúscula,
  cifras e iniciales; títulos de ventana en VT323.
- Los tests no pueden tocar la red: Understat se simula en `tests/conftest.py` igual que football-data,
  con JSON de muestra en `tests/fixtures/` (hay que negar ese patrón en `.gitignore`, que hoy solo
  permite `tests/fixtures/*.csv`).
- Ningún cálculo de la ficha va en `manage.py` ni duplica fórmulas de `LeagueState`.

## Queda en manos de quien lo construya

- Nombres de módulos, funciones y ficheros de caché, y la ubicación del módulo de Understat.
- La lista final de medidas de fuertes/débiles y la normalización exacta.
- Cómo distinguir titulares en los `rosters` (comprobarlo con datos reales: `positionOrder`,
  `roster_in`, posición `Sub`).
- Las coordenadas del campo para cada posición de Understat y si se muestra una etiqueta de sistema
  (4-2-3-1…).
- Los colores y patrones concretos de cada club.
- Si la ficha carga completa o por partes con HTMX.
