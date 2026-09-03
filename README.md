# sales-agent-evals

An observable, evaluated OpenAI function-calling agent over a store-sales
dataset. The agent answers natural-language questions by routing to three
tools — a DuckDB SQL lookup, a data-analysis step, and a chart-code
generator — with every step traced in Arize Phoenix and scored by
LLM-as-judge and code evaluators.

Built by turning four notebooks from DeepLearning.AI's "Evaluating AI Agents"
course into a proper Python package.

## Status

Scaffold in progress. See the package under `src/sales_agent/`.

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env   # then fill in OPENAI_API_KEY
```

The agent talks to any OpenAI-compatible endpoint; set `OPENAI_BASE_URL`,
`OPENAI_API_KEY`, and `OPENAI_MODEL` in your local `.env` (never committed).

## License

MIT — see [LICENSE](LICENSE).
