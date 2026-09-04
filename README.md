# sales-agent-evals

An observable, evaluated OpenAI function-calling agent over a store-sales
dataset. Ask it a natural-language question and it routes to one of three tools
— a DuckDB SQL lookup, an LLM analysis step, and a chart-code generator — while
every step is traced in [Arize Phoenix](https://github.com/Arize-ai/phoenix)
and scored by LLM-as-judge and code evaluators.

This is a portfolio build that turns four notebooks from DeepLearning.AI's
*Evaluating AI Agents* course into a proper Python package.

## Architecture

```
question ─▶ router (OpenAI function calling) ─▶ tool ─▶ … ─▶ answer
                     │
                     ├─ lookup_sales_data      NL → SQL → DuckDB over parquet
                     ├─ analyze_sales_data     LLM interprets the data
                     └─ generate_visualization LLM writes matplotlib code
```

Every stage emits a Phoenix span (`agent` → `router_call` → `tool`), so the
whole run is replayable in the Phoenix UI. Evaluators then query those spans and
attach scores.

| Module | Role |
|--------|------|
| `config.py` | Env-driven settings; builds the OpenAI client from `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL` |
| `tracing.py` | Phoenix setup via `register(auto_instrument=True)`, with a `launch_app()` fallback and a no-op tracer default |
| `tools.py` | The three tools + their OpenAI schema |
| `agent.py` | The instrumented router loop |
| `evals.py` | LLM-judge and code evaluators over traced spans |
| `experiments.py` | The convergence experiment |
| `cli.py` | `sales-agent ask | eval | experiment` |

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # then set OPENAI_API_KEY
```

The agent talks to any OpenAI-compatible endpoint. Set `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `OPENAI_MODEL` in your local `.env` (git-ignored). The
code reads these blindly — it has no knowledge of any specific provider.

## Usage

```bash
sales-agent ask "Which stores did the best in 2021?"
sales-agent eval            # run all evaluators over traced spans
sales-agent eval sql        # or just one: tool-calling | sql | clarity | code
sales-agent experiment      # run the convergence experiment
```

Or work through `notebooks/demo.ipynb` for the same flow, cell by cell.

## Observability

The first run with no reachable collector launches a local Phoenix app in-process
and prints its URL. To use a standalone Phoenix instead, run one and point
`PHOENIX_COLLECTOR_ENDPOINT` at it.

## Tests

```bash
pytest
```

The suite runs without an API key: it covers config, the no-op tracer, real
DuckDB SQL execution, the runnable-code check, and CLI parsing. The LLM-judge
evaluators and the router loop are integration-only (they need a key and a live
Phoenix) and are not part of the no-key suite.

## Notes

- **SQL is LLM-generated and executed directly** against the parquet. That is
  acceptable here — a local, read-only file with no sensitive data — but the
  same pattern against a real database would be a SQL-injection risk and would
  need parameterisation or a query allowlist.
- **Phoenix version.** The course used `arize-phoenix` 4.x, which does not
  install on Python 3.13. This package targets the current 20.x line, whose
  evals and experiments APIs differ substantially from the course
  (`llm_classify` / `OpenAIModel` / `px.Client()` are gone, replaced by
  `create_classifier` / `evaluate_dataframe` and `phoenix.client`).

## License

MIT — see [LICENSE](LICENSE).
