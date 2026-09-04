"""Evals: score the agent's traced spans with LLM-as-judge and code checks.

Uses the Phoenix 20.x / phoenix-evals 3.x API: build a
``ClassificationEvaluator`` via ``create_classifier``, then run it over a span
dataframe with ``evaluate_dataframe``.

Four evaluators:
  * router tool-calling correctness (LLM judge)
  * SQL-generation correctness (LLM judge)
  * response clarity (LLM judge)
  * generated chart code is runnable (pure-Python, no LLM)
"""

from __future__ import annotations

import os

import pandas as pd
from phoenix.client import Client
from phoenix.evals import (
    LLM,
    create_classifier,
    evaluate_dataframe,
)
from phoenix.trace.dsl import SpanQuery

from .config import get_settings

TOOL_CALLING_JUDGE_PROMPT = """
You are an evaluation assistant judging whether an AI agent selected the correct
tool and parameters for a user's question.

[Question]: {question}
[Tool call the agent made]: {tool_call}

Respond with a single word, "correct" or "incorrect". "correct" means the tool
choice and its parameters appropriately address the question.
"""

SQL_EVAL_GEN_PROMPT = """
You are tasked with determining if the SQL generated appropriately answers a
given instruction, taking into account its generated query.

- [Instruction]: {question}
- [Reference Query]: {query_gen}

Assume the db exists and columns are appropriately named. Respond with a single
word: "correct" or "incorrect".
"""

CLARITY_LLM_JUDGE_PROMPT = """
You will be presented with a query and an answer. Evaluate the clarity of the
answer in addressing the query. A clear response is precise, coherent, and
directly addresses the query.

Query: {query}
Answer: {response}

Respond with a single word: "clear" or "unclear".
"""


def _judge_llm() -> LLM:
    settings = get_settings()
    # The phoenix-evals LLM wrapper validates credentials from the environment
    # at construction, before it forwards sync_client_kwargs, so mirror the
    # settings into os.environ for this process.
    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)
    return LLM(
        provider="openai",
        model=settings.openai_model,
        sync_client_kwargs={
            "base_url": settings.openai_base_url,
            "api_key": settings.openai_api_key,
        },
    )


def _client() -> Client:
    return Client(base_url=get_settings().phoenix_collector_endpoint)


def evaluate_tool_calling(project_name: str | None = None) -> pd.DataFrame:
    project = project_name or get_settings().phoenix_project_name
    query = (
        SpanQuery()
        .where("span_kind == 'LLM'")
        .select(question="input.value", tool_call="llm.tools")
    )
    df = _client().spans.get_spans_dataframe(query=query, project_name=project)
    df = df.dropna(subset=["tool_call"])
    if df.empty:
        return df

    evaluator = create_classifier(
        name="tool_calling",
        prompt_template=TOOL_CALLING_JUDGE_PROMPT,
        llm=_judge_llm(),
        choices={"correct": 1.0, "incorrect": 0.0},
    )
    results = evaluate_dataframe(df, [evaluator])
    _log(project, "Tool Calling Eval", df, results, "tool_calling")
    return results


def evaluate_sql_generation(project_name: str | None = None) -> pd.DataFrame:
    project = project_name or get_settings().phoenix_project_name
    query = (
        SpanQuery()
        .where("span_kind == 'LLM'")
        .select(query_gen="llm.output_messages", question="input.value")
    )
    df = _client().spans.get_spans_dataframe(query=query, project_name=project)
    df = df[
        df["question"].str.contains(
            "Generate an SQL query based on a prompt.", na=False
        )
    ]
    if df.empty:
        return df

    evaluator = create_classifier(
        name="sql_generation",
        prompt_template=SQL_EVAL_GEN_PROMPT,
        llm=_judge_llm(),
        choices={"correct": 1.0, "incorrect": 0.0},
    )
    results = evaluate_dataframe(df, [evaluator])
    _log(project, "SQL Gen Eval", df, results, "sql_generation")
    return results


def evaluate_response_clarity(project_name: str | None = None) -> pd.DataFrame:
    project = project_name or get_settings().phoenix_project_name
    query = (
        SpanQuery()
        .where("span_kind == 'AGENT'")
        .select(response="output.value", query="input.value")
    )
    df = _client().spans.get_spans_dataframe(query=query, project_name=project)
    if df.empty:
        return df

    evaluator = create_classifier(
        name="clarity",
        prompt_template=CLARITY_LLM_JUDGE_PROMPT,
        llm=_judge_llm(),
        choices={"clear": 1.0, "unclear": 0.0},
    )
    results = evaluate_dataframe(df, [evaluator])
    _log(project, "Response Clarity", df, results, "clarity")
    return results


def code_is_runnable(code: str) -> bool:
    """Return True if the generated chart code executes without error."""
    code = code.strip().replace("```python", "").replace("```", "")
    try:
        exec(code, {"__name__": "__eval__"})
        return True
    except Exception:
        return False


def evaluate_generated_code(project_name: str | None = None) -> pd.DataFrame:
    project = project_name or get_settings().phoenix_project_name
    query = (
        SpanQuery()
        .where("name == 'generate_visualization'")
        .select(generated_code="output.value")
    )
    df = _client().spans.get_spans_dataframe(query=query, project_name=project)
    if df.empty:
        return df

    df["label"] = (
        df["generated_code"]
        .apply(code_is_runnable)
        .map({True: "runnable", False: "not_runnable"})
    )
    df["score"] = df["label"].map({"runnable": 1, "not_runnable": 0})
    _client().spans.log_span_annotations_dataframe(
        dataframe=df, annotation_name="Runnable Code Eval", annotator_kind="CODE"
    )
    return df


def _log(
    project: str,
    eval_name: str,
    source_df: pd.DataFrame,
    results: pd.DataFrame,
    evaluator_name: str,
) -> None:
    """Log LLM-judge results back to Phoenix as span evaluations.

    ``evaluate_dataframe`` returns the verdict in a ``{evaluator}_score`` column
    whose cells are dicts: ``{'score', 'label', 'explanation', ...}``.
    ``log_span_annotations_dataframe`` needs those three fields as flat columns.
    """
    frame = results.copy()
    verdict = frame[f"{evaluator_name}_score"]
    frame["score"] = verdict.apply(lambda v: v.get("score"))
    frame["label"] = verdict.apply(lambda v: v.get("label"))
    frame["explanation"] = verdict.apply(lambda v: v.get("explanation"))
    Client(
        base_url=get_settings().phoenix_collector_endpoint
    ).spans.log_span_annotations_dataframe(
        dataframe=frame, annotation_name=eval_name, annotator_kind="LLM"
    )
