"""CLI interactivo para charlar con el agente, con memoria persistente por thread_id.

Uso (desde la raiz del repo, con el venv activado):

    python -m scripts.chat

Si no se indica un thread_id se genera uno nuevo. Usando el mismo thread_id en dos
corridas distintas (incluso reiniciando el proceso), el agente recupera el historial
desde el checkpointer de SQLite y recuerda la conversacion previa.
"""
from __future__ import annotations

import asyncio
import uuid

from langchain_core.messages import HumanMessage

from agent.config import RECURSION_LIMIT
from agent.graph import compiled_graph


async def main() -> None:
    thread_id = input("thread_id (enter para generar uno nuevo): ").strip() or str(uuid.uuid4())
    print(f"Usando thread_id={thread_id!r}. Escribi 'salir' para terminar.\n")

    async with compiled_graph() as graph:
        while True:
            user_input = input("Vos: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"salir", "exit", "quit"}:
                break

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            }
            result = await graph.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            print("Agente:", result["messages"][-1].content, "\n")


if __name__ == "__main__":
    asyncio.run(main())
