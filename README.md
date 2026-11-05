# react-agent-langgraph

Agente de razonamiento ciclico (ReAct) construido con [LangGraph](https://langchain-ai.github.io/langgraph/),
con herramientas propias, memoria persistente por `thread_id` (checkpointer SQLite async)
y un ciclo de reintento/aclaracion cuando una herramienta devuelve un error.

## Herramientas

Simulan una consulta a una base de datos de pedidos (`data/pedidos_db.json`):

- `buscar_cliente_pedidos(cliente_id: int)`: devuelve el nombre del cliente y la lista de
  IDs de sus pedidos, o `{"error": "..."}` si el cliente no existe.
- `obtener_detalle_pedido(pedido_id: int)`: devuelve fecha, total e items de un pedido
  puntual, o `{"error": "..."}` si no existe.

Para responder "cuantos pedidos tuvo el cliente 102 y cual fue el total", el agente
necesita llamar primero a `buscar_cliente_pedidos` y despues a `obtener_detalle_pedido`
**una vez por cada pedido** encontrado (razonamiento multi-paso, con 2 o mas invocaciones
de herramientas en una sola respuesta).

### Ciclo de retorno (manejo de errores)

El `SystemMessage` del agente (en `agent/graph.py`) le instruye explicitamente que, si una
herramienta devuelve `error`, no debe inventar datos: tiene que intentar corregir el
parametro una vez, o si no puede, pedirle aclaracion al usuario. Esto se puede ver en
`examples/trace_example.json`, escenario `ciclo_de_retorno_error` (pregunta sobre un
cliente inexistente).

## Requisitos

- Python 3.12+
- Una API key de Google AI Studio (Gemini) con soporte de function calling.

## Instalacion

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env y completar GOOGLE_API_KEY
```
## Uso

### Chat interactivo con memoria

```bash
python -m scripts.chat
```

Te pide un `thread_id` (o genera uno random). Mientras uses el mismo `thread_id`, el
agente recuerda todo lo anterior en esa sesion, incluso si reiniciaste el proceso (queda
persistido en `checkpoints.sqlite`).

### Regenerar la traza de ejemplo

```bash
python -m scripts.run_demo
```

Corre dos escenarios reales contra el LLM y escribe
`examples/trace_example.json` y `examples/trace_example.log`.

### Tests

```bash
python -m pytest
```

## Decisiones de diseño

- Se eligio **Gemini** (`langchain-google-genai`) como proveedor de LLM. La libreria
  soporta `bind_tools()` igual que las integraciones de OpenAI/Anthropic sugeridas por la
  consigna, asi que el resto de la arquitectura (grafo, checkpointer, tools) es
  independiente del proveedor — cambiar de LLM implica tocar solo `agent/graph.py`.
- Se usa `AsyncSqliteSaver` (variante async de `SqliteSaver`) porque todo el grafo corre
  sobre `asyncio` (`ainvoke`, nodos `async def`).
- No hay ningun router manual (`if/else`) que decida que herramienta llamar: esa decision
  es 100% del LLM, guiada por los docstrings de cada `@tool`.
