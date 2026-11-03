# react-agent-langgraph

Agente de razonamiento ciclico (ReAct) construido con [LangGraph](https://langchain-ai.github.io/langgraph/),
con herramientas propias, memoria persistente por `thread_id` (checkpointer SQLite async)
y un ciclo de reintento/aclaracion cuando una herramienta devuelve un error.

Pre-entrega 5 — AI Engineer (Coderhouse).

## Arquitectura

```
START -> agent --(tools_condition)--> tools -> agent -> ... -> END
                 \_______________________________________/
                    (vuelve a "agent" hasta que el LLM no
                     pida mas tool calls)
```

- **Estado**: `MessagesState` de LangGraph (lista de mensajes con reducer `operator.add`
  incorporado). No hace falta un esquema custom porque toda la informacion relevante
  (preguntas, tool calls, resultados de tools, respuestas) vive en el historial de
  mensajes.
- **Nodo `agent`** (`agent/graph.py`): llama al LLM (Gemini, vía `langchain-google-genai`)
  con las herramientas "bindeadas" (`llm.bind_tools([...])`). El LLM decide por si mismo,
  en base al prompt del usuario y a la descripcion de cada herramienta, si necesita
  llamar a una tool o si ya puede responder. No hay ningun `if/else` manual eligiendo la
  herramienta.
- **Nodo `tools`**: un `ToolNode` de LangGraph que ejecuta la/las tool call(s) pedidas por
  el LLM y devuelve `ToolMessage`s.
- **Arista condicional**: `tools_condition` (provisto por `langgraph.prebuilt`) decide si
  ir a `tools` o a `END` segun si el ultimo mensaje del LLM trae `tool_calls` o no.
- **Persistencia**: `AsyncSqliteSaver` (de `langgraph-checkpoint-sqlite`) guarda el estado
  completo de cada `thread_id` en un archivo SQLite local (`checkpoints.sqlite`, ignorado
  por git). Usando el mismo `thread_id` en invocaciones distintas, el agente recupera el
  historial y "recuerda" la conversacion.
- **Limite de recursion**: toda invocacion al grafo pasa `recursion_limit` (default `10`,
  configurable por `.env`) para evitar loops infinitos de llamadas a herramientas.

## Herramientas (`agent/tools.py`)

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
# Editar .env y completar GOOGLE_API_KEY
```

### macOS / Linux (bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env y completar GOOGLE_API_KEY
```

**Importante**: `.env` esta en `.gitignore`. Nunca commitees tu API key; solo se lee desde
variables de entorno (`python-dotenv` carga `.env` en desarrollo local).

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

Corre dos escenarios reales contra el LLM (ver mas abajo) y escribe
`examples/trace_example.json` y `examples/trace_example.log`.

### Tests

```bash
python -m pytest
```

Los tests (`tests/`) son unitarios y no requieren API key: validan las herramientas
contra `data/pedidos_db.json` y que el grafo compile con los nodos esperados.

## Ejemplo de traza (`examples/trace_example.json`)

Contiene dos escenarios generados con una corrida real del agente:

1. **`razonamiento_multi_paso`**: mismo `thread_id`, dos preguntas.
   - *"Cuantos pedidos tuvo el cliente 102 y cual fue el total?"* → el agente llama
     `buscar_cliente_pedidos(102)`, obtiene los pedidos `[501, 502, 503]`, y llama
     `obtener_detalle_pedido` tres veces (una por pedido) para sumar los totales →
     responde con la cantidad de pedidos y el total ($14.500).
   - *"Y el ultimo, de cuando fue y cuanto salio?"* → gracias a la memoria persistida en
     el mismo `thread_id`, el agente ya sabe de que cliente se habla y responde sin
     necesitar que el usuario repita el ID.
2. **`ciclo_de_retorno_error`**: *"Cuantos pedidos tuvo el cliente 999?"* → la herramienta
   devuelve `{"error": "..."}` porque el cliente no existe, y el agente responde
   pidiendole al usuario que confirme el ID en lugar de inventar un numero.

## Decisiones de diseno

- Se eligio **Gemini** (`langchain-google-genai`) como proveedor de LLM. La libreria
  soporta `bind_tools()` igual que las integraciones de OpenAI/Anthropic sugeridas por la
  consigna, asi que el resto de la arquitectura (grafo, checkpointer, tools) es
  independiente del proveedor — cambiar de LLM implica tocar solo `agent/graph.py`.
- Se usa `AsyncSqliteSaver` (variante async de `SqliteSaver`) porque todo el grafo corre
  sobre `asyncio` (`ainvoke`, nodos `async def`).
- No hay ningun router manual (`if/else`) que decida que herramienta llamar: esa decision
  es 100% del LLM, guiada por los docstrings de cada `@tool`.
