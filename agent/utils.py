"""Utilidades compartidas entre los scripts que muestran respuestas del agente."""
from __future__ import annotations

from typing import Any


def extract_text(content: Any) -> str:
    """Normaliza el 'content' de un mensaje a texto plano.

    En langchain-core moderno, 'content' puede ser un string o una lista de bloques
    (ej: [{"type": "text", "text": "..."}]) cuando el modelo devuelve texto junto con
    metadata extra. Esta funcion se queda solo con el texto, para que la salida sea
    legible.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "".join(parts)
    return str(content)
