"""Definicion del grafo de razonamiento ciclico (ReAct) con persistencia.

El flujo es:

    START -> agent --(tools_condition)--> tools -> agent -> ... -> END

El nodo "agent" llama al LLM (con las herramientas "bindeadas"). Si el LLM decide
invocar una herramienta, `tools_condition` enruta al nodo "tools"; el resultado de la
herramienta vuelve como ToolMessage al nodo "agent", que decide si necesita llamar a
otra herramienta mas, reintentar, pedir aclaraciones, o ya puede responder (en cuyo
caso `tools_condition` enruta a END). Esta decision es siempre del LLM: el grafo no
tiene ningun `if/else` manual que elija la herramienta.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import GEMINI_MODEL, GOOGLE_API_KEY, SQLITE_DB_PATH
from agent.tools import buscar_cliente_pedidos, obtener_detalle_pedido

TOOLS = [buscar_cliente_pedidos, obtener_detalle_pedido]

SYSTEM_PROMPT = SystemMessage(
    content=(
        "Sos un asistente que responde preguntas sobre clientes y pedidos de una tienda, "
        "usando EXCLUSIVAMENTE las herramientas disponibles para obtener datos reales. "
        "Nunca inventes numeros de pedido, fechas ni totales que no vengan de una herramienta.\n\n"
        "Si una herramienta devuelve un campo 'error': no la vuelvas a llamar con exactamente "
        "los mismos parametros. Primero fijate si fue un error de tipeo o de dato que vos mismo "
        "puedas corregir y reintenta UNA vez con el valor corregido; si no podes resolverlo, "
        "dejá de llamar herramientas y explicale al usuario el problema puntual, pidiendole que "
        "confirme o corrija el dato.\n\n"
        "Cuando ya tengas todos los datos necesarios para responder, respondé de forma clara y "
        "breve, citando los numeros concretos obtenidos de las herramientas."
    )
)


@lru_cache(maxsize=1)
def _get_llm() -> BaseChatModel:
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en el entorno. Copia .env.example a .env y completa tu key."
        )
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
        timeout=30,
        max_retries=2,
    )
    return llm.bind_tools(TOOLS)


async def _call_model(state: MessagesState) -> dict:
    llm = _get_llm()
    messages = [SYSTEM_PROMPT, *state["messages"]]
    response = await llm.ainvoke(messages)
    return {"messages": [response]}


def build_graph_builder() -> StateGraph:
    """Construye (sin compilar) el StateGraph. Util para tests que no necesitan persistencia."""
    builder = StateGraph(MessagesState)
    builder.add_node("agent", _call_model)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder


@asynccontextmanager
async def compiled_graph(db_path: str = SQLITE_DB_PATH) -> AsyncIterator[CompiledStateGraph]:
    """Context manager que entrega el grafo compilado con checkpointer SQLite async.

    Uso:
        async with compiled_graph() as graph:
            await graph.ainvoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})
    """
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph_builder().compile(checkpointer=checkpointer)
        yield graph
