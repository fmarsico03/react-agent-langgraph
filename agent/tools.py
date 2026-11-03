"""Herramientas que simulan consultas a una base de datos de clientes y pedidos.

Los datos viven en ``data/pedidos_db.json`` para poder inspeccionarlos y modificarlos
sin tocar codigo. Cada tool devuelve un diccionario JSON-serializable; en caso de no
encontrar el registro pedido, devuelve ``{"error": "..."}`` en lugar de lanzar una
excepcion, para que el LLM pueda decidir como reaccionar (reintentar con otro id o
pedirle aclaraciones al usuario) en vez de romper el grafo.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pedidos_db.json"


def _load_db() -> dict[str, Any]:
    with _DB_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_DB: dict[str, Any] = _load_db()


@tool
def buscar_cliente_pedidos(cliente_id: int) -> dict[str, Any]:
    """Busca un cliente por su ID y devuelve sus datos junto con los IDs de todos sus pedidos.

    Usa esta herramienta SIEMPRE que el usuario pregunte por la cantidad de pedidos de un
    cliente, su historial de compras, o cuando necesites saber que pedidos existen para un
    cliente antes de poder calcular totales, fechas o detalles especificos (en ese caso,
    el siguiente paso suele ser llamar a `obtener_detalle_pedido` una vez por cada
    pedido_id devuelto).

    Args:
        cliente_id: Identificador numerico del cliente (por ejemplo: 102).

    Returns:
        Si el cliente existe: un diccionario con las claves "cliente_id" (int), "nombre"
        (str) y "pedido_ids" (lista de int, uno por cada pedido de ese cliente).
        Si el cliente NO existe: un diccionario con la unica clave "error" (str)
        describiendo el problema. En ese caso no inventes un nombre ni pedidos: pedile
        aclaracion al usuario o verifica si el ID fue escrito correctamente antes de
        reintentar.
    """
    cliente = _DB["clientes"].get(str(cliente_id))
    if cliente is None:
        return {"error": f"No existe ningun cliente con id {cliente_id} en la base de datos."}
    return {
        "cliente_id": cliente_id,
        "nombre": cliente["nombre"],
        "pedido_ids": cliente["pedido_ids"],
    }


@tool
def obtener_detalle_pedido(pedido_id: int) -> dict[str, Any]:
    """Obtiene el detalle completo de un pedido puntual, dado su ID numerico.

    Usa esta herramienta para saber la fecha, el total en pesos y los items comprados de
    UN pedido especifico. Como solo trabaja con un pedido a la vez, si necesitas el total
    gastado por un cliente o cual fue su pedido mas reciente, primero obtene la lista de
    pedido_ids con `buscar_cliente_pedidos` y despues llama a esta herramienta una vez por
    cada id (puede ser varias veces en la misma respuesta).

    Args:
        pedido_id: Identificador numerico del pedido (por ejemplo: 501).

    Returns:
        Si el pedido existe: un diccionario con "pedido_id" (int), "cliente_id" (int),
        "fecha" (str, formato YYYY-MM-DD), "total" (numero) e "items" (lista de str).
        Si el pedido NO existe: un diccionario con la unica clave "error" (str). No
        inventes un total ni una fecha: pedi aclaracion o revisa si el id es correcto
        antes de reintentar.
    """
    pedido = _DB["pedidos"].get(str(pedido_id))
    if pedido is None:
        return {"error": f"No existe ningun pedido con id {pedido_id} en la base de datos."}
    return {"pedido_id": pedido_id, **pedido}
