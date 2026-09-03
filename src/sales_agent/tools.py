"""Tools: the three capabilities the agent's router can call.

1. ``lookup_sales_data`` — translate a natural-language prompt into SQL (via an
   LLM) and run it against the sales parquet using DuckDB.
2. ``analyze_sales_data`` — have an LLM interpret a data string and answer a
   question about it.
3. ``generate_visualization`` — produce a chart config, then matplotlib code.

Tracing is fetched at call time via ``get_tracer()`` so the same functions run
traced (after ``setup_tracing()``) or untraced (tests) with no code change.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, Field

from .config import get_openai_client, get_settings
from .tracing import get_tracer

TABLE_NAME = "sales"

SQL_GENERATION_PROMPT = """
Generate an SQL query based on a prompt. Do not reply with anything besides the SQL query.
The prompt is: {prompt}

The available columns are: {columns}
The table name is: {table_name}
"""

DATA_ANALYSIS_PROMPT = """
Analyze the following data: {data}
Your job is to answer the following question: {prompt}
"""

CHART_CONFIGURATION_PROMPT = """
Generate a chart configuration based on this data: {data}
The goal is to show: {visualization_goal}
"""

CREATE_CHART_PROMPT = """
Write python code to create a chart based on the following configuration.
Only return the code, no other text.
config: {config}
"""


def generate_sql_query(prompt: str, columns: list, table_name: str) -> str:
    client = get_openai_client()
    formatted_prompt = SQL_GENERATION_PROMPT.format(
        prompt=prompt, columns=columns, table_name=table_name
    )
    response = client.chat.completions.create(
        model=get_settings().openai_model,
        messages=[{"role": "user", "content": formatted_prompt}],
    )
    return response.choices[0].message.content


def lookup_sales_data(prompt: str) -> str:
    """Look up sales data by translating the prompt to SQL and running it.

    The SQL is LLM-generated and executed directly. This is acceptable only
    because the target is a local, read-only parquet with no sensitive data;
    against a real database this pattern would be an injection risk.
    """
    tracer = get_tracer()

    @tracer.tool()
    def _run(prompt: str) -> str:
        try:
            df = pd.read_parquet(get_settings().data_file_path)
            duckdb.sql(
                f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM df"
            )

            sql_query = generate_sql_query(prompt, list(df.columns), TABLE_NAME)
            sql_query = sql_query.strip().replace("```sql", "").replace("```", "")

            with tracer.start_as_current_span(
                "execute_sql_query", openinference_span_kind="chain"
            ) as span:
                span.set_input(sql_query)
                result = duckdb.sql(sql_query).df()
                span.set_output(value=str(result))
                span.set_status(StatusCode.OK)

            return result.to_string()
        except Exception as e:
            return f"Error accessing data: {str(e)}"

    return _run(prompt)


def analyze_sales_data(prompt: str, data: str) -> str:
    tracer = get_tracer()

    @tracer.tool()
    def _run(prompt: str, data: str) -> str:
        client = get_openai_client()
        formatted_prompt = DATA_ANALYSIS_PROMPT.format(data=data, prompt=prompt)
        response = client.chat.completions.create(
            model=get_settings().openai_model,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        analysis = response.choices[0].message.content
        return analysis if analysis else "No analysis could be generated"

    return _run(prompt, data)


class VisualizationConfig(BaseModel):
    chart_type: str = Field(..., description="Type of chart to generate")
    x_axis: str = Field(..., description="Name of the x-axis column")
    y_axis: str = Field(..., description="Name of the y-axis column")
    title: str = Field(..., description="Title of the chart")


def extract_chart_config(data: str, visualization_goal: str) -> dict:
    tracer = get_tracer()

    @tracer.chain()
    def _run(data: str, visualization_goal: str) -> dict:
        client = get_openai_client()
        formatted_prompt = CHART_CONFIGURATION_PROMPT.format(
            data=data, visualization_goal=visualization_goal
        )
        response = client.beta.chat.completions.parse(
            model=get_settings().openai_model,
            messages=[{"role": "user", "content": formatted_prompt}],
            response_format=VisualizationConfig,
        )
        try:
            content = response.choices[0].message.parsed
            return {
                "chart_type": content.chart_type,
                "x_axis": content.x_axis,
                "y_axis": content.y_axis,
                "title": content.title,
                "data": data,
            }
        except Exception:
            return {
                "chart_type": "line",
                "x_axis": "date",
                "y_axis": "value",
                "title": visualization_goal,
                "data": data,
            }

    return _run(data, visualization_goal)


def create_chart(config: dict) -> str:
    tracer = get_tracer()

    @tracer.chain()
    def _run(config: dict) -> str:
        client = get_openai_client()
        formatted_prompt = CREATE_CHART_PROMPT.format(config=config)
        response = client.chat.completions.create(
            model=get_settings().openai_model,
            messages=[{"role": "user", "content": formatted_prompt}],
        )
        code = response.choices[0].message.content
        return code.replace("```python", "").replace("```", "").strip()

    return _run(config)


def generate_visualization(data: str, visualization_goal: str) -> str:
    tracer = get_tracer()

    @tracer.tool()
    def _run(data: str, visualization_goal: str) -> str:
        config = extract_chart_config(data, visualization_goal)
        return create_chart(config)

    return _run(data, visualization_goal)


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "lookup_sales_data",
            "description": "Look up data from Store Sales Price Elasticity Promotions dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The unchanged prompt that the user provided.",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_data",
            "description": "Analyze sales data to extract insights",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The lookup_sales_data tool's output.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The unchanged prompt that the user provided.",
                    },
                },
                "required": ["data", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_visualization",
            "description": "Generate Python code to create data visualizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "The lookup_sales_data tool's output.",
                    },
                    "visualization_goal": {
                        "type": "string",
                        "description": "The goal of the visualization.",
                    },
                },
                "required": ["data", "visualization_goal"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "lookup_sales_data": lookup_sales_data,
    "analyze_sales_data": analyze_sales_data,
    "generate_visualization": generate_visualization,
}
