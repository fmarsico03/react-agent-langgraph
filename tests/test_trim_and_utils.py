from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

from agent.config import MAX_HISTORY_MESSAGES
from agent.graph import AgentState, trim_history
from agent.utils import extract_text


def test_agent_state_hereda_de_messages_state() -> None:
    # TypedDict no soporta issubclass; se valida que herede el campo con su reducer.
    assert MessagesState in AgentState.__orig_bases__
    assert AgentState.__annotations__["messages"] == MessagesState.__annotations__["messages"]


def test_trim_history_no_toca_historiales_cortos() -> None:
    messages = [HumanMessage(content="hola"), AIMessage(content="hola!")]
    assert trim_history(messages) == messages


def test_trim_history_acota_y_arranca_en_un_mensaje_humano() -> None:
    messages = []
    for i in range(MAX_HISTORY_MESSAGES):
        messages += [
            HumanMessage(content=f"pregunta {i}"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": f"c{i}"}]),
            ToolMessage(content="{}", tool_call_id=f"c{i}"),
            AIMessage(content=f"respuesta {i}"),
        ]
    trimmed = trim_history(messages)
    assert len(trimmed) <= MAX_HISTORY_MESSAGES
    assert isinstance(trimmed[0], HumanMessage)
    assert trimmed[-1] == messages[-1]


def test_trim_history_conserva_turno_actual_aunque_sea_largo() -> None:
    largo = MAX_HISTORY_MESSAGES + 5
    messages = [HumanMessage(content="vieja"), AIMessage(content="ok"), HumanMessage(content="actual")]
    messages += [ToolMessage(content="{}", tool_call_id=f"c{i}") for i in range(largo)]
    trimmed = trim_history(messages)
    assert trimmed[0].content == "actual"
    assert len(trimmed) == largo + 1


def test_extract_text_normaliza_bloques() -> None:
    assert extract_text("hola") == "hola"
    assert extract_text([{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}]) == "hola"
