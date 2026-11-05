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
from typing import AsyncIterator, Awaitable, Callable

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import GEMINI_MODEL, GOOGLE_API_KEY, MAX_HISTORY_MESSAGES, SQLITE_DB_PATH
from agent.tools import buscar_cliente_pedidos, obtener_detalle_pedido

TOOLS = [buscar_cliente_pedidos, obtener_detalle_pedido]

ToolBoundLLM = Runnable[LanguageModelInput, BaseMessage]


class AgentState(MessagesState):
    """Estado del agente: hereda la lista `messages` (con reducer `add_messages`) de
    MessagesState. Se define como subclase para poder sumar campos propios a futuro sin
    tocar el resto del grafo."""


AgentNode = Callable[[AgentState], Awaitable[dict[str, list[BaseMessage]]]]

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
def _get_llm() -> ToolBoundLLM:
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


def trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Recorta el historial que se le manda al LLM a los ultimos MAX_HISTORY_MESSAGES.

    El estado persistido sigue guardando todo; solo se acota la ventana de contexto para
    evitar el "estado sucio" (costos y contexto creciendo sin limite). `start_on="human"`
    garantiza que el recorte nunca deje un ToolMessage huerfano sin su AIMessage con el
    tool_call correspondiente. Si el turno actual es mas largo que la ventana (muchas tool
    calls), se conserva igual completo desde la ultima pregunta del usuario.
    """
    trimmed = trim_messages(
        messages,
        max_tokens=MAX_HISTORY_MESSAGES,
        token_counter=len,
        strategy="last",
        start_on="human",
    )
    if trimmed:
        return trimmed
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0
    )
    return messages[last_human:]


def _make_call_model(llm: ToolBoundLLM | None) -> AgentNode:
    async def _call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        model = llm if llm is not None else _get_llm()
        messages = [SYSTEM_PROMPT, *trim_history(state["messages"])]
        response = await model.ainvoke(messages)
        return {"messages": [response]}

    return _call_model


def build_graph_builder(llm: ToolBoundLLM | None = None) -> StateGraph:
    """Construye (sin compilar) el StateGraph.

    `llm` permite inyectar un modelo ya "bindeado" (por ejemplo uno falso en tests); si es
    None se usa Gemini configurado desde el entorno.
    """
    builder = StateGraph(AgentState)
    builder.add_node("agent", _make_call_model(llm))
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder


@asynccontextmanager
async def compiled_graph(
    db_path: str = SQLITE_DB_PATH, llm: ToolBoundLLM | None = None
) -> AsyncIterator[CompiledStateGraph]:
    """Context manager que entrega el grafo compilado con checkpointer SQLite async.

    Uso:
        async with compiled_graph() as graph:
            await graph.ainvoke({"messages": [...]}, config={"configurable": {"thread_id": "abc"}})
    """
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph_builder(llm).compile(checkpointer=checkpointer)
        yield graph
