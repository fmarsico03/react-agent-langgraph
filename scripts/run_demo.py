"""Genera la traza de ejecucion de ejemplo pedida por la consigna.

Corre dos escenarios contra el agente real (LLM + herramientas + persistencia):

1. "razonamiento_multi_paso": se le pregunta por los pedidos del cliente 102 y su total
   (obliga a llamar `buscar_cliente_pedidos` y luego `obtener_detalle_pedido` varias
   veces), y despues, en el MISMO thread_id, se pregunta "y el ultimo?" para probar que
   el agente recuerda el contexto gracias al checkpointer.
2. "ciclo_de_retorno_error": se pregunta por un cliente que no existe, para mostrar que
   el agente recibe el {"error": ...} de la herramienta y responde pidiendo aclaracion
   en vez de inventar datos.

El resultado se guarda en examples/trace_example.json (estructurado) y
examples/trace_example.log (legible).

Uso (desde la raiz del repo, con el venv activado y GOOGLE_API_KEY seteada):

    python -m scripts.run_demo
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.config import RECURSION_LIMIT
from agent.graph import compiled_graph

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def message_to_dict(message: BaseMessage) -> dict[str, Any]:
    data: dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = [
            {"name": tc["name"], "args": tc["args"], "id": tc.get("id")} for tc in tool_calls
        ]
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        data["tool_call_id"] = tool_call_id
    name = getattr(message, "name", None)
    if name:
        data["name"] = name
    return data


async def _ask(graph, thread_id: str, text: str) -> Any:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    return await graph.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)


async def _full_trace(graph, thread_id: str) -> list[dict[str, Any]]:
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [message_to_dict(m) for m in state.values["messages"]]


async def run_scenario_multi_paso(graph) -> dict[str, Any]:
    thread_id = f"demo-cliente-102-{uuid.uuid4().hex[:8]}"
    turnos = []
    for pregunta in [
        "Cuantos pedidos tuvo el cliente 102 y cual fue el total?",
        "Y el ultimo, de cuando fue y cuanto salio?",
    ]:
        result = await _ask(graph, thread_id, pregunta)
        turnos.append({"pregunta": pregunta, "respuesta_final": result["messages"][-1].content})
    return {
        "thread_id": thread_id,
        "turnos": turnos,
        "trace_completa": await _full_trace(graph, thread_id),
    }


async def run_scenario_error(graph) -> dict[str, Any]:
    thread_id = f"demo-cliente-inexistente-{uuid.uuid4().hex[:8]}"
    pregunta = "Cuantos pedidos tuvo el cliente 999?"
    result = await _ask(graph, thread_id, pregunta)
    return {
        "thread_id": thread_id,
        "turnos": [{"pregunta": pregunta, "respuesta_final": result["messages"][-1].content}],
        "trace_completa": await _full_trace(graph, thread_id),
    }


async def main() -> None:
    EXAMPLES_DIR.mkdir(exist_ok=True)

    async with compiled_graph() as graph:
        escenario_multi_paso = await run_scenario_multi_paso(graph)
        escenario_error = await run_scenario_error(graph)

    output = {
        "generado_en_utc": datetime.now(timezone.utc).isoformat(),
        "escenarios": {
            "razonamiento_multi_paso": escenario_multi_paso,
            "ciclo_de_retorno_error": escenario_error,
        },
    }

    json_path = EXAMPLES_DIR / "trace_example.json"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    log_path = EXAMPLES_DIR / "trace_example.log"
    with log_path.open("w", encoding="utf-8") as fh:
        for nombre_escenario, escenario in output["escenarios"].items():
            fh.write(f"=== Escenario: {nombre_escenario} (thread_id={escenario['thread_id']}) ===\n")
            for msg in escenario["trace_completa"]:
                extra = f" tool_calls={msg['tool_calls']}" if "tool_calls" in msg else ""
                nombre = f" name={msg['name']}" if "name" in msg else ""
                fh.write(f"[{msg['type']}]{nombre}{extra} {msg.get('content', '')}\n")
            fh.write("\n")

    print(f"Trace guardada en:\n  {json_path}\n  {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
