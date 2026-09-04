"""No-key tests: exercise the deterministic layers without hitting any LLM.

These cover config loading, the no-op tracer, real DuckDB SQL execution, the
runnable-code check, and CLI parsing. LLM-dependent paths (the router loop and
the LLM-judge evaluators) are integration-only and require a key + Phoenix, so
they are intentionally not covered here.
"""

from __future__ import annotations

import pytest


def test_get_openai_client_raises_without_key(monkeypatch):
    from sales_agent import config

    config.get_settings.cache_clear()
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        config.get_openai_client()
    config.get_settings.cache_clear()


def test_settings_defaults_to_public_openai(monkeypatch):
    from sales_agent import config

    config.get_settings.cache_clear()
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    settings = config.Settings(_env_file=None)
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.openai_model == "gpt-4o-mini"


def test_noop_tracer_decorators_are_transparent():
    from sales_agent.tracing import get_tracer

    tracer = get_tracer()

    @tracer.tool()
    def doubled(x):
        return x * 2

    @tracer.chain()
    def incremented(x):
        return x + 1

    assert doubled(3) == 6
    assert incremented(3) == 4
    with tracer.start_as_current_span("s", openinference_span_kind="chain") as span:
        span.set_input("in")
        span.set_output(value="out")


def test_lookup_sales_data_executes_real_sql(monkeypatch):
    from sales_agent import tools

    monkeypatch.setattr(
        tools,
        "generate_sql_query",
        lambda prompt, columns, table_name: (
            f"SELECT Store_Number, SUM(Total_Sale_Value) AS total "
            f"FROM {table_name} GROUP BY Store_Number ORDER BY total DESC LIMIT 1"
        ),
    )
    result = tools.lookup_sales_data("top store by revenue")
    assert "Store_Number" in result
    assert "total" in result


def test_lookup_sales_data_returns_error_string_on_bad_sql(monkeypatch):
    from sales_agent import tools

    monkeypatch.setattr(
        tools,
        "generate_sql_query",
        lambda prompt, columns, table_name: "SELECT * FROM no_such_table",
    )
    result = tools.lookup_sales_data("nonsense")
    assert result.startswith("Error accessing data:")


@pytest.mark.parametrize(
    "code,expected",
    [
        ("x = [1, 2, 3]\nsum(x)", True),
        ("```python\ny = 2 + 2\n```", True),
        ("this is not valid python !!!", False),
    ],
)
def test_code_is_runnable(code, expected):
    from sales_agent.evals import code_is_runnable

    assert code_is_runnable(code) is expected


def test_cli_parser_dispatches_subcommands():
    from sales_agent.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["ask", "hello"])
    assert args.question == "hello"
    assert args.func.__name__ == "_cmd_ask"

    args = parser.parse_args(["eval", "sql"])
    assert args.which == "sql"

    args = parser.parse_args(["experiment"])
    assert args.command == "experiment"


def test_cli_requires_a_subcommand():
    from sales_agent.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([])
