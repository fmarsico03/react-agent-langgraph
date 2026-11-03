from agent.tools import buscar_cliente_pedidos, obtener_detalle_pedido


def test_buscar_cliente_pedidos_cliente_existente() -> None:
    resultado = buscar_cliente_pedidos.invoke({"cliente_id": 102})
    assert resultado["nombre"] == "Carlos Fernandez"
    assert resultado["pedido_ids"] == [501, 502, 503]


def test_buscar_cliente_pedidos_cliente_inexistente() -> None:
    resultado = buscar_cliente_pedidos.invoke({"cliente_id": 999})
    assert "error" in resultado


def test_obtener_detalle_pedido_pedido_existente() -> None:
    resultado = obtener_detalle_pedido.invoke({"pedido_id": 501})
    assert resultado["total"] == 5000
    assert resultado["cliente_id"] == 102


def test_obtener_detalle_pedido_pedido_inexistente() -> None:
    resultado = obtener_detalle_pedido.invoke({"pedido_id": 123456})
    assert "error" in resultado


def test_total_cliente_102_coincide_con_la_consigna() -> None:
    cliente = buscar_cliente_pedidos.invoke({"cliente_id": 102})
    total = sum(
        obtener_detalle_pedido.invoke({"pedido_id": pid})["total"] for pid in cliente["pedido_ids"]
    )
    assert len(cliente["pedido_ids"]) == 3
    assert total == 14500
