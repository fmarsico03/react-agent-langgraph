"""Prueba end-to-end del grafo (ciclo ReAct + SqliteSaver) sin API key.

Se reemplaza el LLM por un modelo falso con respuestas guionadas que imita lo que hace
Gemini en la traza real: primero pide `buscar_cliente_pedidos`, despues pide
`obtener_detalle_pedido` para cada pedido y finalmente responde. Las herramientas, el
ToolNode, `tools_condition` y el checkpointer son los reales.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from agent.graph import compiled_graph


class ScriptedLLM:
    """LLM falso: devuelve respuestas en orden y registra lo que recibio en cada llamada."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self._responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []

    def as_runnable(self) -> RunnableLambda[LanguageModelInput, BaseMessage]:
        async def _invoke(messages: list[BaseMessage]) -> BaseMessage:
            self.calls.append(list(messages))
            return next(self._responses)

        return RunnableLambda(_invoke)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}


def test_ciclo_multi_paso_y_memoria_por_thread_id(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("buscar_cliente_pedidos", {"cliente_id": 102}, "c1")]),
            AIMessage(
                content="",
                tool_calls=[
                    _tool_call("obtener_detalle_pedido", {"pedido_id": pid}, f"c{pid}")
                    for pid in (501, 502, 503)
                ],
            ),
            AIMessage(content="El cliente 102 tuvo 3 pedidos por un total de $14.500."),
            AIMessage(content="El ultimo fue el pedido 503, del 2026-09-30, por $5.000."),
        ]
    )
    db_path = str(tmp_path / "checkpoints.sqlite")

    async def run() -> tuple[list[BaseMessage], list[BaseMessage]]:
        async with compiled_graph(db_path, llm=llm.as_runnable()) as graph:
            primero = await graph.ainvoke(
                {"messages": [HumanMessage(content="Cuantos pedidos tuvo el cliente 102 y cual fue el total?")]},
                config=_config("t-102"),
            )
        # Se reabre el checkpointer (como si se reiniciara el proceso) con el mismo thread_id.
        async with compiled_graph(db_path, llm=llm.as_runnable()) as graph:
            segundo = await graph.ainvoke(
                {"messages": [HumanMessage(content="Y el ultimo?")]}, config=_config("t-102")
            )
        return primero["messages"], segundo["messages"]

    primero, segundo = asyncio.run(run())

    tool_msgs = [m for m in primero if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 4  # 1 busqueda + 3 detalles: la herramienta se invoca >= 2 veces
    assert '"total": 4500' in tool_msgs[2].content
    assert "14.500" in primero[-1].content

    # Memoria: el segundo turno arranca con todo el historial persistido del primero.
    assert segundo[: len(primero)] == primero
    assert "503" in segundo[-1].content
    ultima_llamada = llm.calls[-1]
    assert any(isinstance(m, ToolMessage) and "2026-09-30" in m.content for m in ultima_llamada)


def test_error_de_herramienta_vuelve_al_agente(tmp_path: Path) -> None:
    llm = ScriptedLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("buscar_cliente_pedidos", {"cliente_id": 999}, "c1")]),
            AIMessage(content="No encontre el cliente 999. Podes confirmar el ID?"),
        ]
    )

    async def run() -> list[BaseMessage]:
        async with compiled_graph(str(tmp_path / "cp.sqlite"), llm=llm.as_runnable()) as graph:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content="Cuantos pedidos tuvo el cliente 999?")]},
                config=_config("t-999"),
            )
        return result["messages"]

    messages = asyncio.run(run())

    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert "error" in tool_msg.content
    # El error no corta el grafo: vuelve al nodo agent, que ve el error y pide aclaracion.
    assert any(isinstance(m, ToolMessage) and "error" in m.content for m in llm.calls[-1])
    assert "confirmar" in messages[-1].content
