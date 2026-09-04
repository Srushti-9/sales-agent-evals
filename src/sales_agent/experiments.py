"""Experiments: the convergence experiment over the agent.

Convergence measures how consistently the agent reaches an answer in a similar
number of steps across many paraphrases of the same question. We upload the
paraphrases as a Phoenix dataset, run the agent on each (recording the message
trajectory length), then score each run against the shortest observed path.

Uses the ``phoenix.client.experiments`` API.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from phoenix.client import Client

from .agent import run_agent_with_messages
from .config import get_settings

CONVERGENCE_QUESTIONS = [
    "What was the average quantity sold per transaction?",
    "What is the mean number of items per sale?",
    "Calculate the typical quantity per transaction",
    "What's the mean transaction size in terms of quantity?",
    "On average, how many items were purchased per transaction?",
    "What is the average basket size per sale?",
    "Calculate the mean number of products per purchase",
    "What's the typical number of units per order?",
    "What is the average number of products bought per purchase?",
    "Tell me the mean quantity of items in a typical transaction",
    "How many items does a customer buy on average per transaction?",
    "What's the usual number of units in each sale?",
    "What is the typical amount of products per transaction?",
    "Show the mean number of items customers purchase per visit",
    "What's the average quantity of units per shopping trip?",
    "How many products do customers typically buy in one transaction?",
    "What is the standard basket size in terms of quantity?",
]


def _format_message_steps(messages: list) -> str:
    steps = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            steps.append(f"User: {message.get('content')}")
        elif role == "system":
            steps.append("System: Provided context")
        elif role == "assistant":
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    steps.append(f"Assistant: Called tool '{tc['function']['name']}'")
            else:
                steps.append(f"Assistant: {message.get('content')}")
        elif role == "tool":
            steps.append(f"Tool response: {message.get('content')}")
    return "\n".join(steps)


def _run_agent_task(example) -> dict:
    question = example.input.get("question")
    _, messages = run_agent_with_messages(
        [{"role": "user", "content": question}]
    )
    return {"path_length": len(messages), "messages": _format_message_steps(messages)}


def _client() -> Client:
    return Client(base_url=get_settings().phoenix_collector_endpoint)


def upload_convergence_dataset():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    df = pd.DataFrame({"question": CONVERGENCE_QUESTIONS})
    return _client().datasets.create_dataset(
        name=f"convergence_questions-{now}",
        dataframe=df,
        input_keys=["question"],
    )


def run_convergence_experiment():
    """Run the convergence experiment and score paths against the shortest one."""
    dataset = upload_convergence_dataset()

    from phoenix.client.experiments import evaluate_experiment, run_experiment

    experiment = run_experiment(
        dataset=dataset,
        task=_run_agent_task,
        experiment_name="Convergence Eval",
        experiment_description="Evaluating the convergence of the agent",
    )

    outputs = [run.get("output") for run in experiment["task_runs"]]
    lengths = [
        o.get("path_length")
        for o in outputs
        if o and o.get("path_length") is not None
    ]
    optimal_path_length = min(lengths) if lengths else 0

    def evaluate_path_length(output: dict) -> float:
        length = output.get("path_length") if output else None
        return optimal_path_length / float(length) if length else 0.0

    return evaluate_experiment(
        experiment=experiment, evaluators=[evaluate_path_length]
    )
