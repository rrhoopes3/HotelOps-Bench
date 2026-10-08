"""HTTP adapters for xAI Chat Completions, Anthropic Messages, and OpenAI Responses. Stdlib only."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from typing import Any

XAI_URL = "https://api.x.ai/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
# gpt-6-luna rejects function tools + reasoning_effort on /v1/chat/completions
# ("use /v1/responses or set reasoning_effort to 'none'"), so OpenAI goes via Responses.
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def _post_json(url: str, headers: dict[str, str], body: dict, timeout: float) -> tuple[int, dict | str, float]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    ctx = ssl.create_default_context()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            raw = resp.read().decode("utf-8")
            elapsed = time.perf_counter() - t0
            try:
                return resp.status, json.loads(raw), elapsed
            except json.JSONDecodeError:
                return resp.status, raw, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - t0
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return 0, {"error": str(e)}, elapsed


def _redact(text: str) -> str:
    out = text
    for needle in ("sk-ant-", "sk-proj-", "sk-svcacct-", "xai-", "Bearer "):
        if needle in out:
            i = out.find(needle)
            out = out[:i] + needle + "[redacted]"
    return out[:800]


def openai_to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function") or t
        out.append(
            {
                "name": fn["name"],
                "description": fn.get("description") or "",
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def openai_to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    out: list[dict] = []
    pending_tool_results: list[dict] = []

    def flush_tools():
        nonlocal pending_tool_results
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            if m.get("content"):
                system_parts.append(str(m["content"]))
            continue
        if role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "unknown",
                    "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")),
                }
            )
            continue
        flush_tools()
        if role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
            continue
        if role == "assistant":
            content: list[dict] = []
            if m.get("_anthropic_content"):
                content = list(m["_anthropic_content"])
            else:
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args else {}
                        except json.JSONDecodeError:
                            args = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id") or "toolu_unknown",
                            "name": fn.get("name") or tc.get("name"),
                            "input": args or {},
                        }
                    )
            out.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
    flush_tools()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def parse_openai_message(msg: dict) -> dict:
    calls = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        parsed = args
        if isinstance(args, str):
            try:
                parsed = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                parsed = {"_unparsed": args}
        calls.append(
            {
                "id": tc.get("id"),
                "name": fn.get("name"),
                "arguments": parsed if isinstance(parsed, dict) else {"_value": parsed},
                "arguments_raw": args,
            }
        )
    return {
        "role": "assistant",
        "text": msg.get("content") or "",
        "tool_calls": calls,
        "stop_reason": None,
        "refusal": bool(msg.get("refusal")),
        "raw_message": msg,
    }


def parse_anthropic_message(body: dict) -> dict:
    calls = []
    texts = []
    content = body.get("content") or []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            texts.append(block.get("text") or "")
        elif btype == "tool_use":
            calls.append(
                {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("input") if isinstance(block.get("input"), dict) else {},
                    "arguments_raw": block.get("input"),
                }
            )
    stop = body.get("stop_reason")
    return {
        "role": "assistant",
        "text": "\n".join(texts),
        "tool_calls": calls,
        "stop_reason": stop,
        "refusal": stop == "refusal",
        "raw_message": body,
        "_anthropic_content": content,
    }


def usage_from_openai(body: dict) -> dict:
    u = body.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    prompt_details = u.get("prompt_tokens_details") or {}
    return {
        "input": int(u.get("prompt_tokens") or 0),
        "output": int(u.get("completion_tokens") or 0),
        "reasoning": int(details.get("reasoning_tokens") or u.get("reasoning_tokens") or 0),
        "cache_read": int(prompt_details.get("cached_tokens") or 0),
        "cache_write": 0,
    }


def usage_from_anthropic(body: dict) -> dict:
    u = body.get("usage") or {}
    # Anthropic's input_tokens excludes cache reads and writes; OpenAI's includes its cached tokens. Normalize
    # "input" to the total prompt size so tokens and cost compare across providers (cost_usd subtracts both).
    return {
        "input": int(u.get("input_tokens") or 0)
        + int(u.get("cache_read_input_tokens") or 0)
        + int(u.get("cache_creation_input_tokens") or 0),
        "output": int(u.get("output_tokens") or 0),
        "reasoning": 0,
        "cache_read": int(u.get("cache_read_input_tokens") or 0),
        "cache_write": int(u.get("cache_creation_input_tokens") or 0),
    }


def usage_from_responses(body: dict) -> dict:
    u = body.get("usage") or {}
    in_d = u.get("input_tokens_details") or {}
    out_d = u.get("output_tokens_details") or {}
    return {
        "input": int(u.get("input_tokens") or 0),
        "output": int(u.get("output_tokens") or 0),
        "reasoning": int(out_d.get("reasoning_tokens") or 0),
        "cache_read": int(in_d.get("cached_tokens") or 0),
        "cache_write": int(in_d.get("cache_write_tokens") or 0),
    }


def openai_to_responses_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function") or t
        out.append(
            {
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description") or "",
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                "strict": False,
            }
        )
    return out


def openai_to_responses_input(messages: list[dict]) -> list[dict]:
    """Chat-shaped history -> Responses input items.

    An assistant turn this adapter produced carries its raw output items
    (reasoning with encrypted_content + function_call) in _responses_output;
    those go back verbatim so the model keeps its reasoning across tool turns.
    Frozen case history has plain tool_calls and becomes function_call items.
    """
    items: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            items.append({"role": "developer", "content": m.get("content") or ""})
        elif role == "assistant":
            if m.get("_responses_output"):
                items.extend(m["_responses_output"])
                continue
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tc.get("id") or "call",
                        "name": fn.get("name"),
                        "arguments": args if isinstance(args, str) else json.dumps(args or {}),
                    }
                )
        elif role == "tool":
            content = m.get("content")
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id") or "call",
                    "output": content if isinstance(content, str) else json.dumps(content),
                }
            )
        else:
            items.append({"role": role or "user", "content": m.get("content") or ""})
    return items


def parse_responses_output(body: dict) -> dict:
    calls = []
    texts = []
    refusal = False
    output = body.get("output") or []
    for item in output:
        itype = item.get("type")
        if itype == "function_call":
            args = item.get("arguments")
            parsed = args
            if isinstance(args, str):
                try:
                    parsed = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    parsed = {"_unparsed": args}
            calls.append(
                {
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "arguments": parsed if isinstance(parsed, dict) else {"_value": parsed},
                    "arguments_raw": args,
                }
            )
        elif itype == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text") or "")
                elif part.get("type") == "refusal":
                    refusal = True
                    texts.append(part.get("refusal") or "")
    return {
        "role": "assistant",
        "text": "\n".join(texts),
        "tool_calls": calls,
        "stop_reason": body.get("status"),
        "refusal": refusal,
        "raw_message": body,
        "_responses_output": output,
        "served_tier": body.get("service_tier"),
    }


def retrying_post(url, headers, body, timeout, retries=4) -> tuple[int, Any, float]:
    delay = 2.0
    last = (0, {"error": "no_attempt"}, 0.0)
    for i in range(retries):
        status, parsed, elapsed = _post_json(url, headers, body, timeout)
        last = (status, parsed, elapsed)
        if status == 200:
            return last
        if status in (429, 500, 502, 503, 529) or status == 0:
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        return last
    return last


class GrokAdapter:
    def __init__(self, api_key: str, model: str, effort: str, timeout: float = 180.0):
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict], tools: list[dict] | None) -> dict:
        payload_messages = []
        if system:
            payload_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.get("role") == "system":
                continue
            if m.get("role") == "assistant":
                item = {
                    "role": "assistant",
                    "content": m.get("content"),
                }
                if m.get("tool_calls"):
                    item["tool_calls"] = m["tool_calls"]
                payload_messages.append(item)
            elif m.get("role") == "tool":
                payload_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": m.get("tool_call_id"),
                        "content": m.get("content") if isinstance(m.get("content"), str) else json.dumps(m.get("content")),
                    }
                )
            else:
                payload_messages.append({"role": m.get("role"), "content": m.get("content")})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "reasoning_effort": self.effort,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        headers = {"Authorization": "Bearer " + self.api_key}
        status, parsed, elapsed = retrying_post(XAI_URL, headers, body, self.timeout)
        if status != 200 or not isinstance(parsed, dict):
            return {
                "ok": False,
                "error": _redact(str(parsed)),
                "http_status": status,
                "elapsed_s": elapsed,
                "usage": {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0},
                "message": None,
            }
        choice = (parsed.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        parsed_msg = parse_openai_message(msg)
        parsed_msg["stop_reason"] = choice.get("finish_reason")
        return {
            "ok": True,
            "error": None,
            "http_status": status,
            "elapsed_s": elapsed,
            "usage": usage_from_openai(parsed),
            "message": parsed_msg,
            "raw": parsed,
        }


class AnthropicAdapter:
    def __init__(self, api_key: str, model: str, effort: str, timeout: float = 180.0, max_tokens: int = 8192,
                 cache: bool = False):
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.max_tokens = max_tokens
        # Opt-in prompt caching (Anthropic does not cache unless asked; OpenAI and xAI cache automatically).
        # One explicit breakpoint on the system prompt (tools render before it, so they are covered) plus
        # top-level automatic caching for the growing conversation tail. Default TTL (5 min).
        self.cache = cache

    def complete(self, system: str, messages: list[dict], tools: list[dict] | None) -> dict:
        sys_extra, anth_messages = openai_to_anthropic_messages(messages)
        sys_text = system or ""
        if sys_extra:
            sys_text = (sys_text + "\n\n" + sys_extra).strip()
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anth_messages or [{"role": "user", "content": "(empty)"}],
            "output_config": {"effort": self.effort},
        }
        if sys_text:
            if self.cache:
                body["system"] = [{"type": "text", "text": sys_text, "cache_control": {"type": "ephemeral"}}]
            else:
                body["system"] = sys_text
        if self.cache:
            body["cache_control"] = {"type": "ephemeral"}
        if tools:
            body["tools"] = openai_to_anthropic_tools(tools)
            body["tool_choice"] = {"type": "auto"}

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }
        status, parsed, elapsed = retrying_post(ANTHROPIC_URL, headers, body, self.timeout)
        if status != 200 or not isinstance(parsed, dict) or parsed.get("type") == "error":
            err = parsed
            if isinstance(parsed, dict) and parsed.get("error"):
                err = parsed["error"]
            return {
                "ok": False,
                "error": _redact(str(err)),
                "http_status": status,
                "elapsed_s": elapsed,
                "usage": {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0},
                "message": None,
            }
        parsed_msg = parse_anthropic_message(parsed)
        return {
            "ok": True,
            "error": None,
            "http_status": status,
            "elapsed_s": elapsed,
            "usage": usage_from_anthropic(parsed),
            "message": parsed_msg,
            "raw": parsed,
        }


class OpenAIAdapter:
    """OpenAI Responses API. store=false; encrypted reasoning is round-tripped."""

    def __init__(self, api_key: str, model: str, effort: str, timeout: float = 180.0, service_tier: str | None = None):
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout
        # "fast" = OpenAI's Fast tier (Priority was renamed Fast on 2026-07-30; "priority" is served as fast).
        self.service_tier = service_tier

    def complete(self, system: str, messages: list[dict], tools: list[dict] | None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "input": openai_to_responses_input(messages) or [{"role": "user", "content": "(empty)"}],
            "reasoning": {"effort": self.effort},
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if self.service_tier:
            body["service_tier"] = self.service_tier
        if system:
            body["instructions"] = system
        if tools:
            body["tools"] = openai_to_responses_tools(tools)
            body["tool_choice"] = "auto"

        headers = {"Authorization": "Bearer " + self.api_key}
        status, parsed, elapsed = retrying_post(OPENAI_RESPONSES_URL, headers, body, self.timeout)
        if status != 200 or not isinstance(parsed, dict) or parsed.get("error"):
            err = parsed
            if isinstance(parsed, dict) and parsed.get("error"):
                err = parsed["error"]
            return {
                "ok": False,
                "error": _redact(str(err)),
                "http_status": status,
                "elapsed_s": elapsed,
                "usage": {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0},
                "message": None,
            }
        parsed_msg = parse_responses_output(parsed)
        return {
            "ok": True,
            "error": None,
            "http_status": status,
            "elapsed_s": elapsed,
            "usage": usage_from_responses(parsed),
            "message": parsed_msg,
            "raw": parsed,
        }
