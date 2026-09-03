"""Agent: the function-calling router loop that orchestrates the tools.

The router asks the model what to do; if it returns tool calls, we execute
them, append the results, and loop until the model produces a final answer.
Spans (agent -> router_call -> tool) are created around each stage; when
tracing is not set up these are no-ops.
"""

from __future__ import annotations

import json

from opentelemetry.trace import StatusCode

from .config import get_openai_client, get_settings
from .tracing import get_tracer
from .tools import TOOLS_SCHEMA, TOOL_IMPLEMENTATIONS

SYSTEM_PROMPT = (
    "You are a helpful assistant that can answer questions about the "
    "Store Sales Price Elasticity Promotions dataset."
)


def handle_tool_calls(tool_calls, messages):
    tracer = get_tracer()

    @tracer.chain()
    def _run(tool_calls, messages):
        for tool_call in tool_calls:
            function = TOOL_IMPLEMENTATIONS[tool_call.function.name]
            function_args = json.loads(tool_call.function.arguments)
            result = function(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tool_call.id,
                }
            )
        return messages

    return _run(tool_calls, messages)


def run_agent(messages) -> str:
    client = get_openai_client()
    model = get_settings().openai_model
    tracer = get_tracer()

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not any(
        isinstance(m, dict) and m.get("role") == "system" for m in messages
    ):
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    while True:
        with tracer.start_as_current_span(
            "router_call", openinference_span_kind="chain"
        ) as span:
            span.set_input(value=messages)
            response = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS_SCHEMA
            )
            messages.append(response.choices[0].message.model_dump())
            tool_calls = response.choices[0].message.tool_calls
            span.set_status(StatusCode.OK)

            if tool_calls:
                messages = handle_tool_calls(tool_calls, messages)
                span.set_output(value=tool_calls)
            else:
                final = response.choices[0].message.content
                span.set_output(value=final)
                return final


def start_main_span(messages) -> str:
    tracer = get_tracer()
    with tracer.start_as_current_span(
        "AgentRun", openinference_span_kind="agent"
    ) as span:
        span.set_input(value=messages)
        result = run_agent(messages)
        span.set_output(value=result)
        span.set_status(StatusCode.OK)
        return result


def ask(question: str) -> str:
    """Convenience entry point: run the agent on a single question string."""
    return start_main_span([{"role": "user", "content": question}])
