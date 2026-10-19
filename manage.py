#!/usr/bin/env python3
"""CLI fino sobre los mismos servicios que usa el dashboard.

No contiene lógica propia: cada subcomando abre una conexión y llama a
`app/services/*`, exactamente lo que hace `app/routers/jobs.py` desde la web.
Esa es la regla que hay que respetar al tocar este fichero — en el momento en
que tenga su propia clasificación o su propio cálculo de features, vuelven las
copias divergentes que la fase 3 se encargó de eliminar.

    python manage.py update
    python manage.py train
    python manage.py simulate --sims 500 --seed 42
    python manage.py project
    python manage.py evaluate
"""
import argparse
import asyncio
import json

import aiosqlite

from app.config import LEAGUE_CODE
from app.database import DB_PATH, init_db
from core.errors import LMLaLigaError


async def cmd_update(db: aiosqlite.Connection, args: argparse.Namespace) -> None:
    from app.services.data_service import update_league_data

    resumen = await update_league_data(db, args.league, season=args.season)
    print(f"{args.league}: {resumen['partidos']} partidos ({resumen['jugados']} jugados)")
    for aviso in ("aviso", "aviso_jugadores"):
        if aviso in resumen:
            print(f"  aviso: {resumen[aviso]}")


async def cmd_train(db: aiosqlite.Connection, args: argparse.Namespace) -> None:
    from app.services.ml_service import train_league_model

    run_id, accuracy = await train_league_model(db, args.league)

    cursor = await db.execute("SELECT * FROM model_runs WHERE id=?", (run_id,))
    run = dict(await cursor.fetchone())
    print(f"{args.league}: precisión fuera de muestra {accuracy:.1%}")
    print(f"  baseline siempre-local:    {run['baseline_home']:.1%}")
    print(f"  baseline clase mayoritaria: {run['baseline_majority']:.1%}")
    if run["baseline_bookmaker"] is not None:
        print(f"  baseline casas de apuestas: {run['baseline_bookmaker']:.1%}")
    print(f"  log-loss {run['logloss']:.3f} · Brier {run['brier']:.3f}")
    print(f"  {run['n_train']} train / {run['n_test']} test ({run['test_period']})")
    print(f"  temporadas: {run['seasons_used']}")


async def cmd_simulate(db: aiosqlite.Connection, args: argparse.Namespace) -> None:
    from app.services.simulation_service import run_simulation

    standings = await run_simulation(db, args.league, sim_count=args.sims, seed=args.seed)
    _print_standings(standings)


async def cmd_project(db: aiosqlite.Connection, args: argparse.Namespace) -> None:
    from app.services.next_season_service import project_next_season

    standings = await project_next_season(db, args.league, sim_count=args.sims, seed=args.seed)
    _print_standings(standings)


async def cmd_evaluate(db: aiosqlite.Connection, args: argparse.Namespace) -> None:
    cursor = await db.execute(
        """
        SELECT * FROM model_runs
         WHERE league_code = ? AND status = 'completed' AND baseline_home IS NOT NULL
         ORDER BY created_at DESC LIMIT 1
        """,
        (args.league,),
    )
    run = await cursor.fetchone()
    if not run:
        print(f"{args.league}: no hay ningún entrenamiento con métricas honestas. Ejecuta 'train' primero.")
        return
    print(json.dumps(dict(run), indent=2, ensure_ascii=False))


def _print_standings(standings: list[dict]) -> None:
    print(f"{'Pos':<4}{'Equipo':<25}{'Pts':<6}{'Título':<9}{'Descenso':<9}")
    for i, team in enumerate(standings, 1):
        titulo = f"{team['champion_prob']:.1%}" if team.get("champion_prob") is not None else "-"
        descenso = f"{team['relegation_prob']:.1%}" if team.get("relegation_prob") is not None else "-"
        print(f"{i:<4}{team['name']:<25}{team['points']:<6}{titulo:<9}{descenso:<9}")


COMMANDS = {
    "update": cmd_update,
    "train": cmd_train,
    "simulate": cmd_simulate,
    "project": cmd_project,
    "evaluate": cmd_evaluate,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    for nombre in ("update", "train", "simulate", "project", "evaluate"):
        # Solo hay una liga: `league` sigue existiendo en args porque los
        # servicios lo reciben, pero ya no es un parámetro de la línea de comandos.
        sub.add_parser(nombre).set_defaults(league=LEAGUE_CODE)

    sub.choices["update"].add_argument("--season", default=None, help="Por defecto, la temporada actual")
    for nombre in ("simulate", "project"):
        p = sub.choices[nombre]
        p.add_argument("--sims", type=int, default=500)
        p.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    await init_db()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            await COMMANDS[args.command](db, args)
        except LMLaLigaError as e:
            # Errores de dominio (temporada finalizada, sin modelo, liga
            # desconocida...): un mensaje, no una traza — la misma excepción
            # que en la web se traduce a job_status.log.
            raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
