# Angel — a deliberate Gemini agent

A terminal and browser agent built on Google's **Gemini** free tier. It does not
just call a model in a loop: every turn is **planned**, executed with tools,
then **reviewed against that plan** before you see it — and sent back for
another pass if the review fails.

```
  you ask
     |
     v
  [ PLAN ]  goal, steps, success criteria
     |
     v
  [ ACT ] --- wants tools? --yes--> [ OBSERVE ] --> back to ACT
     |                               run in parallel,
     no                              cache repeats,
     |                               turn errors into hints
     v
  draft answer
     |
     v
  [ CRITIQUE ]  checked against the criteria and the evidence
     |
     +-- revise --> back to ACT, with the concrete fixes attached
     |
     +-- accept --> final answer to you
```

Everything the agent does is emitted as a typed event stream, so you can watch
it work rather than guess:

```
you> How much would a 7% annual return turn 2,400 into after 30 years, and is
     that better than the current best UK savings rate?

  [plan] Compare a modelled 7% return against today's best UK savings rate  (complex)
      1. Calculate the compounded value of 2400 at 7% over 30 years  (tool: calculate)
      2. Find the current best UK savings rate  (tool: web_search)
      3. Compare the two  (tool: none)
      done when: shows the compounded figure, cites a dated source for the rate
  [tool] calculate(expression=2400 * (1 + 0.07) ** 30)
      -> [0.0s] 2400 * (1 + 0.07) ** 30 = 18269.41210
  [tool] web_search(query=best UK savings rate)
      -> [1.2s] Quick answer: The best easy-access rate is currently...
  [draft written, reviewing it...]
  [review] revising - pass 1
      issue: the savings rate is quoted without the date it was published
      next:  search for the rate with its publication date
  [tool] web_search(query=best UK easy access savings rate September 2026)
      -> [1.1s] Title: Best savings rates, updated 12 September 2026...
  [review] accepted (confidence 92%)
  [3 steps, 3 tool calls, 6.4s, 1 revision(s), 512 thinking tokens]

angel> £2,400 compounding at 7% a year becomes £18,269.41 after 30 years...
```

## What makes the loop deliberate

| Phase | What it adds |
|---|---|
| **Plan** | Decomposes the task into steps and **checkable success criteria**. Trivial questions are labelled `direct` and skip the machinery entirely. |
| **Act** | The model chooses tools; your code routes and runs them. |
| **Observe** | Independent calls run **in parallel**, repeats are **cached within the turn**, and failures become *instructions the model can act on* — so a bad argument is repaired on the next step instead of ending the turn. |
| **Critique** | A second pass re-reads the draft against the criteria and the evidence actually gathered, and can send it back with concrete fixes. |
| **Budget** | Steps, tool calls, reflection rounds and wall-clock time are capped independently. Running out produces a real answer that says what went unverified — never a stub. |

Plus, throughout:

- **Extended thinking** — Gemini 2.5's reasoning budget is on by default, and
  thought summaries appear in the trace.
- **Sub-agents** — `delegate` hands a noisy subtask to a focused agent with its
  own empty memory and a narrowed toolbox. Only the findings come back, so the
  parent's context stays clean. Depth-capped, and its tool calls are charged to
  the parent's budget.
- **Memory that survives trimming** — old turns are *summarised*, not deleted,
  and cuts only ever land on safe boundaries so a tool call is never separated
  from its result. Notes the agent saves itself outlive compaction entirely.

## The tools

| Tool | What it is for |
|---|---|
| `web_search` | Live results via Tavily. |
| `fetch_page` | Reads one page properly, for when the snippets are too thin. |
| `calculate` | Exact arithmetic — 40-odd maths functions, parsed not `eval`'d. |
| `current_datetime` | So date reasoning is not anchored to the training cutoff. |
| `search_documents` | Semantic search over a PDF you loaded. Appears only once one is. |
| `remember` / `recall` | The agent's own scratchpad. |
| `delegate` | Spawns a focused sub-agent. |

The toolbox is assembled from what is actually configured: no Tavily key means
no web tools, and the system prompt is built from the live registry — so the
agent is never told about a tool it does not have.

## Setup

You need **Python 3.10+**, a **Gemini API key**
([aistudio.google.com](https://aistudio.google.com/) → "Get API key", free, no
card) and optionally a **Tavily key** ([app.tavily.com](https://app.tavily.com/),
free tier) for web search.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env              # then put your keys in it
```

Your `.env` needs at minimum:

```
GEMINI_API_KEY=your-real-gemini-key
TAVILY_API_KEY=tvly-your-real-key
```

No spaces around `=`, no quotes, and the file must be named exactly `.env`.

## Run

```bash
python chatbot.py          # terminal, with the live trace
streamlit run app.py       # browser
streamlit run rag_app.py   # browser, focused on one PDF
```

### Terminal commands

| Command | Action |
|---|---|
| `/help` | list the commands |
| `/clear` | forget the conversation (documents stay loaded) |
| `/history` | print the conversation so far |
| `/tools` | list the tools the agent currently has |
| `/notes` | show what the agent has chosen to remember |
| `/stats` | model, token usage and session state |
| `/load <file.pdf>` | index a PDF so the agent can search it |
| `/plan auto\|always\|never` | how often to plan before acting |
| `/trace`, `/thoughts` | toggle the live trace and thinking summaries |
| `/quit` | leave |

## Using it as a library

```python
from gemini_agent import AgentSession, ConsoleRenderer

session = AgentSession.create()
result = session.ask("Compare the two cheapest options and recommend one.",
                     sink=ConsoleRenderer())

print(result.answer)
print(result.plan.render())        # what it set out to do
print(result.critiques)            # what it caught in its own draft
print(result.steps, result.tool_calls, result.elapsed)
```

### Adding a tool

Write a function. The decorator reads its signature and docstring and builds
the Gemini schema, the validation and the routing:

```python
@session.registry.tool(describe={"city": "The city to look up"})
def timezone(city: str) -> str:
    """Get the timezone of a city."""
    return lookup(city)
```

Declare a parameter named `ctx` to reach shared services (the search client,
the document index, the agent's memory, sub-agents) — it is stripped from the
schema the model sees.

## Configuration

Everything is tunable from `.env`; see `.env.example` for the full list.

| Variable | Default | What it does |
|---|---|---|
| `GEMINI_MODEL` | `gemini-2.5-flash` | `-flash-lite` is cheaper, `-pro` reasons harder |
| `AGENT_THINKING_BUDGET` | `-1` | `-1` dynamic, `0` off, or a token cap |
| `AGENT_PLANNING` | `auto` | `auto`, `always`, `never` |
| `AGENT_MAX_STEPS` | `12` | model turns per user turn |
| `AGENT_MAX_TOOL_CALLS` | `24` | tool executions per user turn |
| `AGENT_MAX_REFLECTIONS` | `2` | how often the critic may reject a draft |
| `AGENT_WALL_CLOCK_SECONDS` | `180` | hard time limit per user turn |

## Layout

```
chatbot.py            terminal front end
app.py                browser front end
rag_app.py            browser front end, focused on one PDF
gemini_agent/
  loop.py             the agent: plan, act, observe, critique, revise
  reasoning.py        plan and critique — schemas, parsing, prompts
  llm.py              Gemini transport: retries, fallback, structured output
  registry.py         functions -> tool schemas, validation, dispatch
  memory.py           transcript, compaction, notes
  budget.py           step / tool / reflection / time limits
  events.py           the typed event stream
  render.py           terminal and Streamlit renderers
  session.py          everything wired together
  prompts.py          system prompt built from the live registry
  config.py           one dataclass, read from the environment
  tools/              calculator, web, clock, notes, documents, delegation
  rag/                chunking, embedding, vector store, PDF extraction
tests/                223 tests, no API key and no network needed
```

## Tests

`LLMClient` is a Protocol, so the whole agent runs against a scripted fake
model. The tests assert on loop behaviour — that a tool failure is repaired,
that a critique drives a second pass, that a budget stop still answers, that
compaction never strands a tool result:

```bash
pip install -r requirements-dev.txt
pytest
```

## Notes and limits

- Gemini's free tier has daily limits but needs no card. Planning and review
  each cost an extra call per turn, so `AGENT_PLANNING=never` and
  `AGENT_MAX_REFLECTIONS=0` give you the cheapest (and shallowest) loop.
- The vector store is in-memory NumPy and resets when you quit. Past a few
  thousand chunks, swap `DocumentIndex` for a real vector database — the
  interface is deliberately the shape you would keep.
- `.env` holds secrets and is already in `.gitignore`. Keep it that way.
- The agentic loop here is provider-agnostic in shape. Model → tool request →
  run tool → feed back → critique → repeat maps directly onto Claude's tool-use
  API; only `llm.py` and the schema builder in `registry.py` would change.
