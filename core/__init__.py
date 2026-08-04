"""Núcleo compartido: carga de datos, estado de liga, features y simulación.

Este paquete es librería, no puntos de entrada. Lo importan tanto la capa web
(`app/`) como el CLI. La regla que lo justifica: cada cosa se implementa una
sola vez aquí, y el resto la llama.
"""
