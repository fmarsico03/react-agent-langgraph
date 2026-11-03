from agent.graph import build_graph_builder


def test_el_grafo_tiene_los_nodos_esperados() -> None:
    compiled = build_graph_builder().compile()
    nodos = compiled.get_graph().nodes
    assert "agent" in nodos
    assert "tools" in nodos


def test_el_grafo_compila_sin_checkpointer_para_tests() -> None:
    # No deberia requerir API key ni checkpointer solo para compilar/inspeccionar la estructura.
    builder = build_graph_builder()
    compiled = builder.compile()
    assert compiled is not None
