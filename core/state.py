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

Invariante que no se puede romper: `features()` solo puede leer partidos
anteriores. Nunca añadas a `features()` nada derivado del partido en curso.
"""
from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

# Puntos que reparte cada resultado, desde el punto de vista del local.
PUNTOS = {"H": (3, 0), "A": (0, 3), "D": (1, 1)}

# Valores neutrales cuando todavía no hay historial.
DIAS_DESCANSO_POR_DEFECTO = 7
H2H_PUNTOS_NEUTRO = 1.5
H2H_VICTORIAS_NEUTRO = 0.33

# Goles esperados por equipo cuando aún no hay suficiente historial local/
# visitante propio (promedio típico de una liga europea competitiva).
GOLES_LIGA_NEUTRO = 1.35
PARTIDOS_MINIMOS_PARA_MEDIA_PROPIA = 3


def _media(valor: float, n: int) -> float:
    """Promedio evitando la división por cero."""
    return valor / max(n, 1)


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
    """Estado de un equipo: totales más el desglose local/visitante."""

    total: SideState = field(default_factory=SideState)
    home: SideState = field(default_factory=SideState)
    away: SideState = field(default_factory=SideState)

    def reset_season(self) -> None:
        """Los acumuladores se vacían al cambiar de temporada: un equipo no
        empieza la siguiente con los puntos de la anterior."""
        self.total = SideState()
        self.home = SideState()
        self.away = SideState()


class LeagueState:
    """Estado de todos los equipos de una liga en un instante dado."""

    def __init__(self) -> None:
        self._teams: dict[str, TeamState] = {}
        self.last_match_date: dict[str, pd.Timestamp] = {}
        # Historial H2H por par ORDENADO (local, visitante), en puntos del local
        self.h2h_records: dict[tuple[str, str], list[int]] = {}

    def team(self, name: str) -> TeamState:
        if name not in self._teams:
            self._teams[name] = TeamState()
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
        """Cambio de temporada: se resetean los acumuladores y las fechas, pero
        NO el H2H, que es justo lo que hace informativa esa feature cuando se
        entrena con varias temporadas."""
        for team in self._teams.values():
            team.reset_season()
        self.last_match_date.clear()

    def apply_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        result: str,
        match_date,
    ) -> None:
        """Incorpora un partido ya resuelto al estado."""
        home_points, away_points = PUNTOS[result]

        local = self.team(home_team)
        visitante = self.team(away_team)

        local.total.record(home_goals, away_goals, home_points)
        local.home.record(home_goals, away_goals, home_points)

        visitante.total.record(away_goals, home_goals, away_points)
        visitante.away.record(away_goals, home_goals, away_points)

        self.last_match_date[home_team] = match_date
        self.last_match_date[away_team] = match_date
        self.h2h_records.setdefault((home_team, away_team), []).append(home_points)

    def _dias_descanso(self, team: str, current_date) -> int:
        anterior = self.last_match_date.get(team)
        if anterior is None:
            return DIAS_DESCANSO_POR_DEFECTO
        return (current_date - anterior).days

    def features(self, home_team: str, away_team: str, current_date) -> dict[str, float]:
        """Vector de features previo al partido. Única implementación."""
        local = self.team(home_team)
        visitante = self.team(away_team)

        f: dict[str, float] = {}

        # Estadísticas generales acumuladas
        f["Home_Matches"] = local.total.matches
        f["Away_Matches"] = visitante.total.matches
        f["Home_Win_Rate"] = local.total.win_rate
        f["Away_Win_Rate"] = visitante.total.win_rate
        f["Home_Points_Per_Game"] = local.total.points_per_game
        f["Away_Points_Per_Game"] = visitante.total.points_per_game
        f["Home_Goals_For_Avg"] = local.total.goals_for_avg
        f["Away_Goals_For_Avg"] = visitante.total.goals_for_avg
        f["Home_Goals_Against_Avg"] = local.total.goals_against_avg
        f["Away_Goals_Against_Avg"] = visitante.total.goals_against_avg

        # Forma reciente
        f["Home_Form_3"] = local.total.form(3)
        f["Away_Form_3"] = visitante.total.form(3)
        f["Home_Form_5"] = local.total.form(5)
        f["Away_Form_5"] = visitante.total.form(5)

        # Rendimiento específico como local / como visitante
        f["Home_Home_Win_Rate"] = local.home.win_rate
        f["Away_Away_Win_Rate"] = visitante.away.win_rate
        f["Home_Home_Points_Per_Game"] = local.home.points_per_game
        f["Away_Away_Points_Per_Game"] = visitante.away.points_per_game
        f["Home_Home_Goals_For_Avg"] = local.home.goals_for_avg
        f["Away_Away_Goals_For_Avg"] = visitante.away.goals_for_avg
        f["Home_Home_Goals_Against_Avg"] = local.home.goals_against_avg
        f["Away_Away_Goals_Against_Avg"] = visitante.away.goals_against_avg
        f["Home_Home_Form_3"] = local.home.form(3)
        f["Away_Away_Form_3"] = visitante.away.form(3)

        # Diferencias y factor campo
        f["Points_Diff"] = f["Home_Points_Per_Game"] - f["Away_Points_Per_Game"]
        f["Win_Rate_Diff"] = f["Home_Win_Rate"] - f["Away_Win_Rate"]
        f["Goals_For_Diff"] = f["Home_Goals_For_Avg"] - f["Away_Goals_For_Avg"]
        f["Goals_Against_Diff"] = f["Home_Goals_Against_Avg"] - f["Away_Goals_Against_Avg"]
        f["Form_5_Diff"] = f["Home_Form_5"] - f["Away_Form_5"]
        f["Home_Advantage"] = f["Home_Home_Points_Per_Game"] - f["Home_Points_Per_Game"]

        # Descanso
        f["Home_Days_Rest"] = self._dias_descanso(home_team, current_date)
        f["Away_Days_Rest"] = self._dias_descanso(away_team, current_date)

        # Enfrentamientos directos (par ordenado)
        h2h = self.h2h_records.get((home_team, away_team), [])
        f["H2H_Matches"] = len(h2h)
        f["H2H_Home_Points_Per_Game"] = sum(h2h) / len(h2h) if h2h else H2H_PUNTOS_NEUTRO
        f["H2H_Home_Win_Rate"] = (
            sum(1 for r in h2h if r == 3) / len(h2h) if h2h else H2H_VICTORIAS_NEUTRO
        )

        return f

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Goles esperados (λ) para muestrear un marcador Poisson.

        Mezcla el ataque de un equipo con la defensa del rival — la misma idea
        de `next_season_service._build_ratings`, pero con las medias propias en
        vez de ratings relativos a la liga, porque aquí sí hay historial
        partido a partido disponible. Neutral mientras un equipo no tenga
        partidos suficientes para que su propia media no sea puro ruido (los
        primeros partidos de una temporada, o un ascendido sin historial).
        """
        local = self.team(home_team).home
        visitante = self.team(away_team).away

        ataque_local = (
            local.goals_for_avg if local.matches >= PARTIDOS_MINIMOS_PARA_MEDIA_PROPIA
            else GOLES_LIGA_NEUTRO
        )
        defensa_local = (
            local.goals_against_avg if local.matches >= PARTIDOS_MINIMOS_PARA_MEDIA_PROPIA
            else GOLES_LIGA_NEUTRO
        )
        ataque_visitante = (
            visitante.goals_for_avg if visitante.matches >= PARTIDOS_MINIMOS_PARA_MEDIA_PROPIA
            else GOLES_LIGA_NEUTRO
        )
        defensa_visitante = (
            visitante.goals_against_avg if visitante.matches >= PARTIDOS_MINIMOS_PARA_MEDIA_PROPIA
            else GOLES_LIGA_NEUTRO
        )

        lambda_home = max((ataque_local + defensa_visitante) / 2, 0.1)
        lambda_away = max((ataque_visitante + defensa_local) / 2, 0.1)
        return lambda_home, lambda_away

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
