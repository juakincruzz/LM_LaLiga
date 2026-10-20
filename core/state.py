"""Estado incremental de una liga: única fuente de verdad.

Antes esto existía dos veces: `FeatureEngineer` lo recorría por lotes escribiendo
en columnas de un DataFrame, y `SeasonSimulator` lo llevaba partido a partido
escribiendo en un diccionario. Las 35 fórmulas estaban transcritas en los dos
sitios, así que añadir una feature en uno solo hacía que el modelo recibiera en
silencio un vector desalineado.

Aquí hay una sola implementación de cada cosa:
  - `apply_match`  actualiza el estado con un partido
  - `features`     genera el vector previo al partido
  - `standings`    calcula la clasificación

Las features son un Elo y una forma continua (medias exponenciales de goles y
tiros a puerta), y ninguna de las dos se reinicia en verano. Las 35 anteriores
(acumulados de la temporada, rachas, descanso, H2H) empezaban cada temporada a
cero: hasta la jornada 6 el modelo no sabía más que las frecuencias de
resultados. Ver docs/ROADMAP.md, §2.

Invariante que no se puede romper: `features()` solo puede leer partidos
anteriores. Nunca añadas a `features()` nada derivado del partido en curso.
"""
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

# Puntos que reparte cada resultado, desde el punto de vista del local.
PUNTOS = {"H": (3, 0), "A": (0, 3), "D": (1, 1)}

# Elo: el clásico con multiplicador por diferencia de goles. Parámetros del
# estudio de fiabilidad (experimentos/fiabilidad_modelo.py).
ELO_INICIAL = 1500.0
ELO_ASCENDIDO = 1430.0  # quien llega después de la primera temporada viene de Segunda
ELO_K = 20.0
ELO_VENTAJA_LOCAL = 60.0
ELO_REGRESION = 0.33  # en cada cambio de temporada, un tercio del camino hacia la media

# Forma continua: medias exponenciales, sin reinicio de temporada. Un equipo
# sin historial arranca algo peor que la media de la liga (suele ser un
# ascendido).
EW_ALFA = 0.12
EW_INICIAL = {"goles_favor": 1.2, "goles_contra": 1.5, "tiros_puerta_favor": 3.8, "tiros_puerta_contra": 4.6}

# Ventaja de campo en goles: la forma continua no distingue casa y fuera, así
# que el reparto se aplica después. En La Liga de 14/15 a 21/22 (sin la 20/21),
# 1,525 goles del local y 1,133 del visitante por partido.
FACTOR_GOLES_LOCAL = 1.148
FACTOR_GOLES_VISITANTE = 0.852


def _media(valor: float, n: int) -> float:
    """Promedio evitando la división por cero."""
    return valor / max(n, 1)


def _falta(valor) -> bool:
    return valor is None or pd.isna(valor)


def _ew(team: "TeamState", medida: str, a_favor: float, en_contra: float) -> None:
    """Un paso de la media exponencial de una medida a favor y en contra."""
    team.ew[f"{medida}_favor"] += EW_ALFA * (a_favor - team.ew[f"{medida}_favor"])
    team.ew[f"{medida}_contra"] += EW_ALFA * (en_contra - team.ew[f"{medida}_contra"])


def _forma(resultados: list[int], n: int) -> float:
    """Puntos por partido en los últimos n encuentros."""
    if not resultados:
        return 0.0
    recientes = resultados[-n:]
    return sum(recientes) / len(recientes)


@dataclass
class SideState:
    """Acumulador de un equipo. Se usa tres veces: total, como local y como
    visitante. Ser un dataclass y no un dict hace que un nombre mal escrito
    sea un AttributeError en vez de un cero silencioso."""

    matches: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    results: list[int] = field(default_factory=list)

    def record(self, goals_for: int, goals_against: int, points: int) -> None:
        self.matches += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.points += points
        self.results.append(points)
        if points == 3:
            self.wins += 1
        elif points == 1:
            self.draws += 1
        else:
            self.losses += 1

    @property
    def win_rate(self) -> float:
        return _media(self.wins, self.matches)

    @property
    def points_per_game(self) -> float:
        return _media(self.points, self.matches)

    @property
    def goals_for_avg(self) -> float:
        return _media(self.goals_for, self.matches)

    @property
    def goals_against_avg(self) -> float:
        return _media(self.goals_against, self.matches)

    def form(self, n: int) -> float:
        return _forma(self.results, n)


@dataclass
class TeamState:
    """Estado de un equipo: totales y desglose local/visitante de la
    temporada (clasificación, goles esperados, ficha), más el Elo y la forma
    continua, que no se reinician (son las features del modelo)."""

    total: SideState = field(default_factory=SideState)
    home: SideState = field(default_factory=SideState)
    away: SideState = field(default_factory=SideState)
    elo: float = ELO_INICIAL
    ew: dict[str, float] = field(default_factory=lambda: dict(EW_INICIAL))

    def reset_season(self) -> None:
        """Los acumuladores se vacían al cambiar de temporada: un equipo no
        empieza la siguiente con los puntos de la anterior. El Elo y la forma
        continua no se tocan aquí."""
        self.total = SideState()
        self.home = SideState()
        self.away = SideState()


class LeagueState:
    """Estado de todos los equipos de una liga en un instante dado."""

    def __init__(self) -> None:
        self._teams: dict[str, TeamState] = {}
        # Historial H2H por par ORDENADO (local, visitante), en puntos del
        # local. No es una feature: es para mostrar los precedentes.
        self.h2h_records: dict[tuple[str, str], list[int]] = {}
        # Hasta el primer cambio de temporada, todos los equipos son de la
        # liga desde el principio; los que aparecen después son ascendidos.
        self._primera_temporada = True

    def team(self, name: str) -> TeamState:
        if name not in self._teams:
            elo = ELO_INICIAL if self._primera_temporada else ELO_ASCENDIDO
            self._teams[name] = TeamState(elo=elo)
        return self._teams[name]

    @property
    def teams(self) -> dict[str, TeamState]:
        return self._teams

    def snapshot(self) -> "LeagueState":
        """Copia independiente del estado.

        Es lo que permite que el Monte Carlo reconstruya los partidos jugados
        una sola vez y clone el resultado en cada simulación, en lugar de
        reproducir los 380 partidos 500 veces.
        """
        return deepcopy(self)

    def start_new_season(self) -> None:
        """Cambio de temporada: se resetean los acumuladores (la tabla) y el
        Elo regresa un tercio hacia la media, porque las plantillas cambian en
        verano. La forma continua y el H2H se conservan."""
        if self._teams:
            media = sum(t.elo for t in self._teams.values()) / len(self._teams)
            for team in self._teams.values():
                team.elo += ELO_REGRESION * (media - team.elo)
        for team in self._teams.values():
            team.reset_season()
        self._primera_temporada = False

    def keep_only(self, teams) -> None:
        """Deja solo los equipos indicados (los de la temporada en curso).

        Al construir el estado con varias temporadas, los descendidos siguen
        en `_teams` con los acumuladores a cero y aparecerían en la
        clasificación. El H2H se conserva entero: es historial, no tabla.
        """
        teams = set(teams)
        self._teams = {n: t for n, t in self._teams.items() if n in teams}

    def apply_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        result: str,
        match_date=None,
        home_shots_on_target: float | None = None,
        away_shots_on_target: float | None = None,
    ) -> None:
        """Incorpora un partido ya resuelto al estado.

        Los tiros a puerta son opcionales: un partido simulado solo tiene
        goles, y entonces la forma de tiros no se mueve. `match_date` ya no se
        usa (se quitó la feature de días de descanso); se acepta para no
        cambiar a quien llama.
        """
        home_points, away_points = PUNTOS[result]

        local = self.team(home_team)
        visitante = self.team(away_team)

        local.total.record(home_goals, away_goals, home_points)
        local.home.record(home_goals, away_goals, home_points)

        visitante.total.record(away_goals, home_goals, away_points)
        visitante.away.record(away_goals, home_goals, away_points)

        self.h2h_records.setdefault((home_team, away_team), []).append(home_points)

        # Elo: lo que se esperaba del local (con su ventaja) frente a lo que pasó
        esperado = 1 / (1 + 10 ** (-(local.elo + ELO_VENTAJA_LOCAL - visitante.elo) / 400))
        real = {"H": 1.0, "D": 0.5, "A": 0.0}[result]
        dg = abs(home_goals - away_goals)
        multiplicador = 1.0 if dg <= 1 else (1.5 if dg == 2 else (11 + dg) / 8)
        delta = ELO_K * multiplicador * (real - esperado)
        local.elo += delta
        visitante.elo -= delta

        # Forma continua
        _ew(local, "goles", home_goals, away_goals)
        _ew(visitante, "goles", away_goals, home_goals)
        if not (_falta(home_shots_on_target) or _falta(away_shots_on_target)):
            _ew(local, "tiros_puerta", home_shots_on_target, away_shots_on_target)
            _ew(visitante, "tiros_puerta", away_shots_on_target, home_shots_on_target)

    def features(self, home_team: str, away_team: str, current_date=None) -> dict[str, float]:
        """Vector de features previo al partido. Única implementación.

        `current_date` ya no interviene (se acepta por compatibilidad)."""
        local = self.team(home_team)
        visitante = self.team(away_team)
        lew, vew = local.ew, visitante.ew

        return {
            "Home_Elo": local.elo,
            "Away_Elo": visitante.elo,
            "Elo_Diff": local.elo - visitante.elo,
            "Home_EW_Goals_For": lew["goles_favor"],
            "Home_EW_Goals_Against": lew["goles_contra"],
            "Home_EW_SoT_For": lew["tiros_puerta_favor"],
            "Home_EW_SoT_Against": lew["tiros_puerta_contra"],
            "Away_EW_Goals_For": vew["goles_favor"],
            "Away_EW_Goals_Against": vew["goles_contra"],
            "Away_EW_SoT_For": vew["tiros_puerta_favor"],
            "Away_EW_SoT_Against": vew["tiros_puerta_contra"],
            "EW_Goal_Diff": (lew["goles_favor"] - lew["goles_contra"])
            - (vew["goles_favor"] - vew["goles_contra"]),
            "EW_SoT_Diff": (lew["tiros_puerta_favor"] - lew["tiros_puerta_contra"])
            - (vew["tiros_puerta_favor"] - vew["tiros_puerta_contra"]),
        }

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Goles esperados (λ) para los marcadores Poisson.

        Media del ataque de un equipo y la defensa del rival, con la forma
        continua (medias exponenciales que no se reinician en verano), y
        repartida con la ventaja de campo. Antes usaba las medias de la
        temporada y era neutral hasta el 3.er partido: con datos reales de
        22/23 a 26/27, esto predice mejor los goles (log-verosimilitud −2,833
        frente a −2,884 por partido; en las jornadas 1–6, −2,931 frente a
        −3,098) y el marcador una vez decidido el resultado (−1,830 frente a
        −1,857).
        """
        local, visitante = self.team(home_team).ew, self.team(away_team).ew
        lambda_home = (local["goles_favor"] + visitante["goles_contra"]) / 2 * FACTOR_GOLES_LOCAL
        lambda_away = (visitante["goles_favor"] + local["goles_contra"]) / 2 * FACTOR_GOLES_VISITANTE
        return max(lambda_home, 0.1), max(lambda_away, 0.1)

    def standings(self) -> pd.DataFrame:
        """Clasificación ordenada por puntos, diferencia de goles y goles a favor."""
        filas = [
            {
                "Team": nombre,
                "Games": t.total.matches,
                "Wins": t.total.wins,
                "Draws": t.total.draws,
                "Losses": t.total.losses,
                "GF": t.total.goals_for,
                "GA": t.total.goals_against,
                "Points": t.total.points,
            }
            for nombre, t in self._teams.items()
        ]

        tabla = pd.DataFrame(filas)
        if tabla.empty:
            return tabla

        tabla["goal_diff"] = tabla["GF"] - tabla["GA"]
        tabla = tabla.sort_values(["Points", "goal_diff", "GF"], ascending=False)
        tabla["Position"] = range(1, len(tabla) + 1)
        return tabla.reset_index(drop=True)
