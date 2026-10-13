"""
Token accounting and LangSmith tracing for the Gemini reads.

Every extractor asks Gemini to read a page image and answer in JSON. The reply
carries a usage_metadata block with the exact token counts Google billed, and
until now it was discarded — leaving the per-document cost impossible to
attribute without reading it back off the AI Studio dashboard days later.

Tracing is OPT-IN and degrades to nothing: with LANGSMITH_TRACING unset the
decorator is the identity function and no network call is made, so a missing
key or an unreachable LangSmith cannot fail a document.
"""

import json
import os
from typing import Any, Dict, Optional


def _export_settings_to_environment() -> None:
    """
    Copy the LangSmith values from Settings into os.environ.

    Nothing in this app calls load_dotenv(): pydantic-settings reads .env into
    the Settings object and leaves the process environment untouched. The
    LangSmith SDK only looks at os.environ, so without this the key is present
    in .env, visible on settings, and invisible to the tracer — traces silently
    never appear, which is exactly how this was first noticed.
    """
    from app.core.config import settings

    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGSMITH_ENDPOINT",
        "GEMINI_PRICES",
        "GEMINI_PRICE_CURRENCY",
    ):
        value = getattr(settings, name, None)

        if value and not os.getenv(name):
            os.environ[name] = str(value)


_export_settings_to_environment()


def tracing_enabled() -> bool:
    return (
        os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


try:
    from langsmith import traceable as _traceable
except ImportError:            # SDK absent: tracing is simply unavailable.
    _traceable = None


def traced(name: str, run_type: str = "llm"):
    """Decorator that traces to LangSmith when configured, else does nothing."""
    def decorate(function):
        if _traceable is None or not tracing_enabled():
            return function

        return _traceable(name=name, run_type=run_type)(function)

    return decorate


def traced_generate(call, model: str) -> Dict[str, Any]:
    """
    Run ONE model call as a traced LLM run that LangSmith can cost.

    LangSmith fills its Cost & Tokens dashboards from two things on a run:
    a `usage_metadata` block in the outputs, and `ls_model_name` / `ls_provider`
    in the metadata telling it WHICH model's price list to apply. A traced
    function that merely returns text records the call but reports zero tokens
    and no cost — the dashboards stay empty, which is exactly what they did.

    `call` is a zero-argument callable returning the google-genai response, so
    this wrapper stays independent of how each extractor builds its request.
    """
    def _invoke():
        response = call()

        usage = usage_from_response(response)

        return {
            "text": getattr(response, "text", None),
            # The key LangSmith reads. Its expected field names are
            # input_tokens / output_tokens / total_tokens, which is the shape
            # usage_from_response already produces.
            "usage_metadata": usage,
            "_response": response,
        }

    if _traceable is None or not tracing_enabled():
        return _invoke()

    traced_invoke = _traceable(
        name="gemini.generate",
        run_type="llm",
    )(_invoke)

    return traced_invoke(
        langsmith_extra={
            "metadata": {
                "ls_provider": "google_genai",
                "ls_model_name": model,
            }
        }
    )


def usage_from_response(response) -> Dict[str, Optional[int]]:
    """Pull Google's own token counts off a generate_content response."""
    meta = getattr(response, "usage_metadata", None)

    if meta is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}

    prompt = getattr(meta, "prompt_token_count", None) or 0
    candidates = getattr(meta, "candidates_token_count", None) or 0
    total = getattr(meta, "total_token_count", None) or 0

    # A thinking model reasons before answering, and those tokens are BILLED as
    # output while appearing in neither prompt_token_count nor
    # candidates_token_count — this SDK version exposes no field for them. On a
    # real RTC read, gemini-3.5-flash reported 5,668 in + 1,430 out against a
    # total of 10,467: 3,369 tokens, more than the visible answer, invisible to
    # any cost worked out from the two named counts. flash-lite does not think,
    # and its three numbers reconcile exactly.
    #
    # Recover them from the total so the reported cost is the billed cost.
    reasoning = max(0, total - prompt - candidates) if total else 0

    return {
        "input_tokens": prompt or None,
        "output_tokens": candidates or None,
        "reasoning_tokens": reasoning or None,
        # What output actually costs: the answer plus the thinking behind it.
        "billable_output_tokens": (candidates + reasoning) or None,
        "total_tokens": total or None,
    }


def _price_table() -> Dict[str, Dict[str, float]]:
    """
    Per-MILLION-token prices, as {model: {"input": x, "output": y}}.

    Read from GEMINI_PRICES in the environment so the rates live with the
    deployment rather than in the source: Google changes them, and a stale
    number in code reports a confident wrong cost. With nothing configured the
    token counts are still reported and cost is simply left null.
    """
    raw = os.getenv("GEMINI_PRICES", "").strip()

    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print("GEMINI_PRICES is not valid JSON; per-call cost will be null")
        return {}


def cost_of(model: str, usage: Dict[str, Optional[int]]) -> Optional[float]:
    prices = _price_table().get(model)

    if not prices or usage.get("input_tokens") is None:
        return None

    # Bill the thinking tokens too — they are output as far as the invoice is
    # concerned, and on a thinking model they can exceed the visible answer.
    billable_output = (
        usage.get("billable_output_tokens")
        or usage.get("output_tokens")
        or 0
    )

    return (
        (usage["input_tokens"] or 0) / 1_000_000 * prices.get("input", 0)
        + billable_output / 1_000_000 * prices.get("output", 0)
    )


def accumulate(calls: list) -> Dict[str, Any]:
    """Roll per-call usage into the per-DOCUMENT figure, which is what the
    spend cap is actually consumed by — an RTC costs two Gemini calls."""
    known_cost = [c["cost"] for c in calls if c.get("cost") is not None]

    return {
        "calls": calls,
        "call_count": len(calls),
        "input_tokens": sum(c["usage"].get("input_tokens") or 0 for c in calls),
        "output_tokens": sum(c["usage"].get("output_tokens") or 0 for c in calls),
        "reasoning_tokens": sum(c["usage"].get("reasoning_tokens") or 0 for c in calls),
        "total_tokens": sum(c["usage"].get("total_tokens") or 0 for c in calls),
        "cost": round(sum(known_cost), 6) if known_cost else None,
        "cost_currency": os.getenv("GEMINI_PRICE_CURRENCY", "USD"),
    }
