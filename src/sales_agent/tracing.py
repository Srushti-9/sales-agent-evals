"""Tracing: configure Arize Phoenix + OpenTelemetry for the agent.

The course notebooks assumed a Phoenix server was already running and got its
endpoint from a course-provided helper. Here we detect whether a collector is
reachable at ``PHOENIX_COLLECTOR_ENDPOINT``; if not, we launch a local Phoenix
app in-process (``px.launch_app()``). Either way ``register(auto_instrument=True)``
wires up the OpenAI instrumentation, so LLM calls are traced automatically.
"""

from __future__ import annotations

import contextlib
import urllib.error
import urllib.request

import phoenix as px
from phoenix.otel import register

from .config import get_settings


class _NoOpSpan:
    def set_input(self, *args, **kwargs) -> None: ...
    def set_output(self, *args, **kwargs) -> None: ...
    def set_status(self, *args, **kwargs) -> None: ...


class _NoOpTracer:
    """Stand-in tracer so the agent runs untraced (e.g. in tests).

    Mirrors the subset of the Phoenix tracer API the agent uses: the
    ``@tracer.tool()`` / ``@tracer.chain()`` decorators and the
    ``start_as_current_span`` context manager.
    """

    def tool(self, *d_args, **d_kwargs):
        def wrap(fn):
            return fn

        return wrap(d_args[0]) if d_args and callable(d_args[0]) else wrap

    chain = tool

    @contextlib.contextmanager
    def start_as_current_span(self, *args, **kwargs):
        yield _NoOpSpan()


_tracer = _NoOpTracer()


def get_tracer():
    return _tracer


def _collector_reachable(endpoint: str, timeout: float = 1.0) -> bool:
    try:
        urllib.request.urlopen(endpoint, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except OSError:
        return False


def setup_tracing():
    """Register a Phoenix tracer provider and return (tracer, session_url).

    ``session_url`` is the local Phoenix UI URL when we launched the app
    ourselves, otherwise ``None``.
    """
    settings = get_settings()
    endpoint = settings.phoenix_collector_endpoint.rstrip("/")

    session_url: str | None = None
    if not _collector_reachable(endpoint):
        session = px.launch_app()
        session_url = session.url
        endpoint = session.url.rstrip("/")

    tracer_provider = register(
        project_name=settings.phoenix_project_name,
        endpoint=f"{endpoint}/v1/traces",
        auto_instrument=True,
    )
    global _tracer
    _tracer = tracer_provider.get_tracer(__name__)
    return _tracer, session_url
