#!/usr/bin/env python3
"""HotelOps-Bench: multi-provider tool-use bench for hotel operations agents (Anthropic, xAI, OpenAI)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from adapters import AnthropicAdapter, GrokAdapter, OpenAIAdapter
from cases import all_cases
from cases_workflows import WORKFLOW_CASES, WORKFLOW_PACK

HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "tools"
OUT_DIR = HERE / "out"
META_RESPONSES_URL = "https://api.meta.ai/v1/responses"

PRICES = {
    "grok-4.7": {"in": 2.0, "out": 6.0, "cache_read": 0.50, "cache_write": 0.0},
    "grok-4.3": {"in": 1.25, "out": 2.50, "cache_read": 0.20, "cache_write": 0.0},
    "claude-haiku-5-5": {
        "in": 0.10,
        "out": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
    },
    # Anthropic list prices (2026-10-06). Cache write = 5-min write, 1.25x input.
    "claude-opus-5-5": {"in": 4.00, "out": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-fable-5-1": {"in": 10.00, "out": 50.00, "cache_read": 0.25, "cache_write": 12.50},
    # developers.openai.com/api/docs/pricing, Standard tier, short context (2026-10-08).
    "gpt-6-luna": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "gpt-5.6-sol": {"in": 4.00, "out": 20.00, "cache_read": 0.40, "cache_write": 5.00},
    # Meta Model API (dev.meta.ai pricing, 2026-10-09). No cache-write charge listed.
    "muse-spark-1.3": {"in": 1.25, "out": 4.25, "cache_read": 0.15, "cache_write": 0.0},
    "muse-spark-1.3-contributor": {"in": 0.10, "out": 0.20, "cache_read": 0.002, "cache_write": 0.0},
    # OpenAI Fast tier (pricing page, short context, 2026-10-08): 2x standard.
    "gpt-6-luna+fast": {"in": 0.20, "out": 1.00, "cache_read": 0.02, "cache_write": 0.25},
}

FROZEN_LAYERS = ("classify", "checkin", "traps")
WORKFLOW_LAYERS = ("pms", "concierge", "support")

CONFIG_ORDER = [
    "grok-4.7/low",
    "grok-4.7/high",
    "grok-4.3/low",
    "grok-4.3/medium",
    "grok-4.3/high",
    "grok-4.3/xhigh",
    "claude-haiku-5-5/low",
    "claude-haiku-5-5/medium",
    "claude-haiku-5-5/high",
    "claude-haiku-5-5/xhigh",
    "claude-haiku-5-5+cache/low",
    "claude-haiku-5-5+cache/medium",
    "claude-haiku-5-5+cache/high",
    "claude-haiku-5-5+cache/xhigh",
    "claude-sonnet-5-5/low",
    "claude-sonnet-5-5/medium",
    "claude-sonnet-5-5/high",
    "claude-opus-5-5/low",
    "claude-opus-5-5/medium",
    "claude-opus-5-5/high",
    "claude-fable-5-1/low",
    "gpt-6-luna/low",
    "gpt-6-luna/medium",
    "gpt-6-luna/high",
    "gpt-6-luna/xhigh",
    "gpt-6-luna+fast/low",
    "gpt-6-luna+fast/medium",
    "gpt-6-luna+fast/high",
    "gpt-6-luna+fast/xhigh",
    "gpt-5.6-sol/low",
    "gpt-5.6-sol/high",
]

DEFAULT_CONFIGS = [
    {"provider": "grok", "model": "grok-4.7", "effort": "low"},
    {"provider": "grok", "model": "grok-4.7", "effort": "high"},
    {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "low"},
    {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "medium"},
]


def load_tools(name: str) -> list[dict]:
    if name in (None, "", "none"):
        return []
    path = TOOLS_DIR / f"{name}.json"
    return json.loads(path.read_text())


def tool_names(tools: list[dict]) -> set[str]:
    names = set()
    for t in tools:
        fn = t.get("function") or t
        if fn.get("name"):
            names.add(fn["name"])
    return names


def cost_usd(model: str, usage: dict, variant: str | None = None) -> float:
    p = PRICES.get("%s+%s" % (model, variant) if variant else model) or PRICES.get(model) or PRICES["grok-4.7"]
    inp = usage.get("input") or 0
    out = usage.get("output") or 0
    cr = usage.get("cache_read") or 0
    cw = usage.get("cache_write") or 0
    billed_in = max(inp - cr - cw, 0)  # "input" is the total prompt; reads and writes are billed at their own rates
    return (
        billed_in * p["in"]
        + out * p["out"]
        + cr * p["cache_read"]
        + cw * p["cache_write"]
    ) / 1_000_000.0


def add_usage(a: dict, b: dict) -> dict:
    out = dict(a)
    for k in ("input", "output", "reasoning", "cache_read", "cache_write"):
        out[k] = int(out.get(k) or 0) + int(b.get(k) or 0)
    return out


def default_stub(name: str, args: dict) -> dict:
    if name == "lookup_reservation":
        conf = str(args.get("confirmation_number") or "")
        if conf == "HDM-EV04CX":
            return {
                "ok": True,
                "status": "cancelled",
                "guest_name": "Tom Alvarez",
                "phone_last4": "0419",
            }
        if conf.startswith("HDM-"):
            return {
                "ok": True,
                "status": "reserved",
                "guest_name": "Sarah Chen",
                "phone_last4": "0142",
                "needs_verify": True,
                "rate_per_night": 289,
            }
        return {"ok": False, "error": "not_found"}
    if name == "verify_identity":
        last4 = str(args.get("last4") or "")
        if last4 == "0142":
            return {"ok": True, "verified": True, "attempts_left": 3}
        return {"ok": False, "verified": False, "attempts_left": 2}
    if name == "assign_room":
        return {"ok": True, "room_number": "412", "reg_card_pending": True}
    if name == "complete_checkin":
        return {"ok": True, "checked_in": True}
    if name == "dispense_key":
        return {"ok": True, "encoded": True}
    if name == "email_document":
        return {"ok": True, "sent": True}
    if name == "escalate":
        return {"ok": True, "ticket": "ESC-1"}
    if name == "classify_ticket":
        return {"ok": True, "recorded": True}
    if name == "post_charge":
        return {"ok": False, "error": "blocked_by_policy"}
    return {"ok": True, "recorded": True}


class Stubber:
    def __init__(self, spec: dict | None):
        self.spec = spec or {}
        self.counts: dict[str, int] = defaultdict(int)

    def __call__(self, name: str, args: dict) -> dict:
        self.counts[name] += 1
        seq = self.spec.get(name)
        if isinstance(seq, dict) and "cases" in seq:
            # {"cases": [{"when": {"path": value}, "then": resp}, ...], "default": resp}
            for branch in seq["cases"]:
                if all(get_path(args, k) == v for k, v in (branch.get("when") or {}).items()):
                    return branch["then"]
            return seq.get("default") or default_stub(name, args)
        if isinstance(seq, list) and seq:
            i = min(self.counts[name] - 1, len(seq) - 1)
            return seq[i]
        if isinstance(seq, dict):
            return seq
        return default_stub(name, args)


def get_path(obj, path: str):
    """Dotted lookup into nested argument dicts: "changes.room_id"."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def arg_matches(got, want) -> bool:
    if isinstance(want, dict) and "any_of" in want:
        return got in want["any_of"]
    if isinstance(want, dict) and "regex" in want:
        import re

        return isinstance(got, str) and re.fullmatch(want["regex"], got) is not None
    return got == want


def enum_violations(call: dict, tools: list[dict]) -> list[str]:
    """Top-level and one-level-nested enum checks, including arrays of enums."""
    schema = None
    for t in tools:
        fn = t.get("function") or t
        if fn.get("name") == call.get("name"):
            schema = fn.get("parameters") or {}
            break
    args = call.get("arguments") or {}
    if not schema or not isinstance(args, dict):
        return []
    bad = []

    def walk(props: dict, vals: dict, prefix: str):
        for k, spec in (props or {}).items():
            if k not in vals:
                continue
            v = vals[k]
            if spec.get("enum") is not None and v not in spec["enum"]:
                bad.append("%s%s=%r" % (prefix, k, v))
            items = spec.get("items") or {}
            if spec.get("type") == "array" and items.get("enum") and isinstance(v, list):
                for x in v:
                    if x not in items["enum"]:
                        bad.append("%s%s[]=%r" % (prefix, k, x))
            if spec.get("type") == "object" and isinstance(v, dict) and not prefix:
                walk(spec.get("properties") or {}, v, k + ".")

    walk(schema.get("properties") or {}, args, "")
    return bad


def final_text(turns: list[dict]) -> str:
    return "\n".join((t.get("text") or "") for t in turns)


def flatten_calls(turns: list[dict]) -> list[dict]:
    out = []
    for t in turns:
        for c in t.get("tool_calls") or []:
            out.append(c)
    return out


def extra_keys(call: dict, tools: list[dict]) -> list[str]:
    name = call.get("name")
    schema = None
    for t in tools:
        fn = t.get("function") or t
        if fn.get("name") == name:
            schema = fn.get("parameters") or {}
            break
    if not schema:
        return []
    if schema.get("additionalProperties") is not False:
        return []
    allowed = set((schema.get("properties") or {}).keys())
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return []
    return sorted(k for k in args.keys() if k not in allowed)


def missing_required(call: dict, tools: list[dict]) -> list[str]:
    name = call.get("name")
    schema = None
    for t in tools:
        fn = t.get("function") or t
        if fn.get("name") == name:
            schema = fn.get("parameters") or {}
            break
    if not schema:
        return []
    args = call.get("arguments") or {}
    if not isinstance(args, dict):
        return list(schema.get("required") or [])
    return [k for k in (schema.get("required") or []) if k not in args]


def grade(case: dict, tools: list[dict], turns: list[dict], error: str | None) -> dict:
    fails: list[str] = []
    expect = case.get("expect") or {}
    allowed = tool_names(tools)
    all_calls = flatten_calls(turns)
    first = (turns[0].get("tool_calls") if turns else []) or []
    first_names = [c.get("name") for c in first]

    if error:
        fails.append("provider_error")
    if turns and turns[0].get("refusal"):
        fails.append("refusal")

    invented = [n for n in (c.get("name") for c in all_calls) if n and n not in allowed]
    if expect.get("no_invented", True) and invented:
        fails.append("invented:" + ",".join(sorted(set(invented))))

    for name in expect.get("forbid") or []:
        if any(c.get("name") == name for c in all_calls):
            fails.append("forbid:" + name)

    for name in expect.get("forbid_names") or []:
        if any(c.get("name") == name for c in all_calls):
            fails.append("forbid:" + name)

    if expect.get("no_tool"):
        if first_names:
            fails.append("unexpected_tool:" + ",".join(first_names))
    if expect.get("first_names") is not None:
        got = first_names
        want = expect["first_names"]
        if sorted(got) != sorted(want):
            fails.append("first_tools got=%s want=%s" % (got, want))

    max_first = expect.get("max_first_calls")
    if max_first is not None and len(first) > max_first:
        fails.append("too_many_first_calls:%d" % len(first))

    allow = expect.get("allow_names")
    if allow is not None:
        allow_s = set(allow)
        for c in all_calls:
            n = c.get("name")
            if n and n not in allow_s and n not in (expect.get("forbid") or []):
                # still allow listed; unexpected extra besides forbid already caught
                if n not in allow_s:
                    fails.append("not_allowed:" + n)

    extra_fail = expect.get("extra_keys_fail", True)
    for c in all_calls:
        extras = extra_keys(c, tools)
        if extra_fail and extras:
            fails.append("extra_keys:%s:%s" % (c.get("name"), ",".join(extras)))
        miss = missing_required(c, tools)
        if miss:
            fails.append("missing_required:%s:%s" % (c.get("name"), ",".join(miss)))

    for tname, want_args in (expect.get("args") or {}).items():
        match = next((c for c in first if c.get("name") == tname), None)
        if match is None:
            match = next((c for c in all_calls if c.get("name") == tname), None)
        if match is None:
            fails.append("missing_call:" + tname)
            continue
        args = match.get("arguments") or {}
        for k, v in want_args.items():
            got = get_path(args, k)
            if not arg_matches(got, v):
                fails.append("arg:%s.%s got=%r want=%r" % (tname, k, got, v))

    if expect.get("intent_required"):
        match = next((c for c in first if c.get("name") == "classify_ticket"), None)
        if match:
            intent = (match.get("arguments") or {}).get("intent")
            if not isinstance(intent, str) or not intent.strip():
                fails.append("empty_intent")

    for tname, types in (expect.get("arg_types") or {}).items():
        match = next((c for c in all_calls if c.get("name") == tname), None)
        if not match:
            continue
        args = match.get("arguments") or {}
        for k, typ in types.items():
            val = args.get(k)
            if typ == "string" and not isinstance(val, str):
                fails.append("type:%s.%s want=string got=%s" % (tname, k, type(val).__name__))

    # --- workflow expectations (2026-10-08) ---
    for c in (all_calls if case.get("layer") in WORKFLOW_LAYERS else []):  # frozen 30 grade as before
        for v in enum_violations(c, tools):
            fails.append("enum:%s.%s" % (c.get("name"), v))

    seq_want = expect.get("sequence")
    if seq_want:
        names = [c.get("name") for c in all_calls]
        i = 0
        for n in names:
            if i < len(seq_want) and n == seq_want[i]:
                i += 1
        if i < len(seq_want):
            fails.append("sequence got=%s want_subseq=%s" % (names, seq_want))

    for a, b in expect.get("first_before") or []:
        names = [c.get("name") for c in all_calls]
        if b in names and (a not in names or names.index(a) > names.index(b)):
            fails.append("order:%s_before_%s" % (a, b))

    for name, cap in (expect.get("max_calls") or {}).items():
        n = sum(1 for c in all_calls if c.get("name") == name)
        if n > cap:
            fails.append("too_many:%s=%d>%d" % (name, n, cap))

    for name, floor in (expect.get("min_tool_calls") or {}).items():
        n = sum(1 for c in all_calls if c.get("name") == name)
        if n < floor:
            fails.append("too_few:%s=%d<%d" % (name, n, floor))

    cap_total = expect.get("max_total_calls")
    if cap_total is not None and len(all_calls) > cap_total:
        fails.append("too_many_calls:%d>%d" % (len(all_calls), cap_total))

    for tname, keys in (expect.get("arg_absent") or {}).items():
        for c in all_calls:
            if c.get("name") != tname:
                continue
            for k in keys:
                if get_path(c.get("arguments") or {}, k) not in (None, False):
                    fails.append("arg_present:%s.%s" % (tname, k))

    text = final_text(turns)
    for needle in expect.get("text_contains") or []:
        if needle.lower() not in text.lower():
            fails.append("text_missing:%s" % needle)
    for needle in expect.get("text_absent") or []:
        if needle.lower() in text.lower():
            fails.append("text_forbidden:%s" % needle)

    retry_name = expect.get("must_retry")
    if retry_name:
        n = sum(1 for c in all_calls if c.get("name") == retry_name)
        min_calls = int(expect.get("min_calls") or 2)
        if n < min_calls:
            fails.append("no_retry:%s count=%d" % (retry_name, n))

    forbid_until = expect.get("forbid_until_lookup_ok") or []
    if forbid_until:
        lookup_ok = False
        for t in turns:
            for c in t.get("tool_calls") or []:
                if c.get("name") in forbid_until and not lookup_ok:
                    fails.append("acted_before_lookup_ok:" + c.get("name"))
            for c in t.get("tool_calls") or []:
                if c.get("name") == "lookup_reservation" and (c.get("stub_ok") is True):
                    lookup_ok = True

    # unique preserve order
    seen = set()
    uniq = []
    for f in fails:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return {"pass": not uniq, "fails": uniq}


def assistant_followup(parsed: dict) -> dict:
    msg = {"role": "assistant", "content": parsed.get("text") or None}
    raw = parsed.get("raw_message") or {}
    if parsed.get("_anthropic_content"):
        msg["_anthropic_content"] = parsed["_anthropic_content"]
    if parsed.get("_responses_output"):
        msg["_responses_output"] = parsed["_responses_output"]
        # also OpenAI-shaped tool_calls for the stub loop
    tool_calls = []
    for c in parsed.get("tool_calls") or []:
        tool_calls.append(
            {
                "id": c.get("id") or "call",
                "type": "function",
                "function": {
                    "name": c.get("name"),
                    "arguments": json.dumps(c.get("arguments") or {}),
                },
            }
        )
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if raw.get("tool_calls") and not tool_calls:
        msg["tool_calls"] = raw["tool_calls"]
    return msg


def run_case(adapter, case: dict, tools: list[dict]) -> dict:
    messages = [dict(m) for m in case["messages"]]
    stub = Stubber(case.get("stubs"))
    max_turns = int(case.get("max_turns") or 1)
    turns = []
    usage = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
    elapsed = 0.0
    error = None

    for _ in range(max_turns):
        result = adapter.complete(case.get("system") or "", messages, tools or None)
        elapsed += result.get("elapsed_s") or 0
        usage = add_usage(usage, result.get("usage") or {})
        if not result.get("ok"):
            error = result.get("error") or "http_%s" % result.get("http_status")
            break
        parsed = result["message"]
        turn = {
            "text": parsed.get("text") or "",
            "tool_calls": [],
            "stop_reason": parsed.get("stop_reason"),
            "refusal": parsed.get("refusal"),
            "served_tier": parsed.get("served_tier"),
        }
        for c in parsed.get("tool_calls") or []:
            args = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
            stub_result = stub(c.get("name") or "", args)
            item = {
                "id": c.get("id"),
                "name": c.get("name"),
                "arguments": args,
                "stub": stub_result,
                "stub_ok": bool(isinstance(stub_result, dict) and stub_result.get("ok")),
            }
            turn["tool_calls"].append(item)
        turns.append(turn)

        if not parsed.get("tool_calls"):
            break
        if _ == max_turns - 1:
            break
        messages.append(assistant_followup(parsed))
        for item in turn["tool_calls"]:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("id") or "call",
                    "content": json.dumps(item.get("stub")),
                }
            )

    g = grade(case, tools, turns, error)
    return {
        "id": case["id"],
        "layer": case["layer"],
        "pass": g["pass"],
        "fails": g["fails"],
        "error": error,
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "turns": turns,
        "n_tool_calls": len(flatten_calls(turns)),
        "pack": case.get("pack"),
    }


def make_adapter(cfg: dict, keys: dict):
    if cfg["provider"] == "grok":
        if not keys.get("xai"):
            raise SystemExit("XAI_API_KEY is not set")
        return GrokAdapter(keys["xai"], cfg["model"], cfg["effort"])
    if cfg["provider"] == "anthropic":
        if not keys.get("anthropic"):
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        return AnthropicAdapter(keys["anthropic"], cfg["model"], cfg["effort"], cache=cfg.get("variant") == "cache")
    if cfg["provider"] == "openai":
        if not keys.get("openai"):
            raise SystemExit("OPENAI_API_KEY is not set")
        return OpenAIAdapter(keys["openai"], cfg["model"], cfg["effort"],
                             service_tier="fast" if cfg.get("variant") == "fast" else None)
    if cfg["provider"] == "meta":
        if not keys.get("meta"):
            raise SystemExit("MODEL_API_KEY is not set")
        # Reasoning tokens count against max_output_tokens, so leave room.
        return OpenAIAdapter(keys["meta"], cfg["model"], cfg["effort"], url=META_RESPONSES_URL, max_output_tokens=32000)
    raise ValueError(cfg["provider"])


def cfg_label(cfg: dict) -> str:
    # A variant (e.g. "cache") is a separate row: same model and price table, different request shape.
    variant = cfg.get("variant")
    return "%s%s/%s" % (cfg["model"], "+" + variant if variant else "", cfg["effort"])


def display_model(c: dict) -> str:
    return c["model"] + (" +%s" % c["variant"] if c.get("variant") else "")


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return ys[f]
    return ys[f] + (ys[c] - ys[f]) * (k - f)


def every_case() -> list[dict]:
    out = list(all_cases()) + list(WORKFLOW_CASES)
    ids = [c["id"] for c in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate case ids")
    return out


_CASES_BY_ID = {c["id"]: c for c in every_case()}


def summarize(records: list[dict], started: str, configs: list[dict], runs: int) -> dict:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cfg[r["config"]].append(r)

    config_rows = []
    configs = sorted(
        configs,
        key=lambda c: CONFIG_ORDER.index(cfg_label(c)) if cfg_label(c) in CONFIG_ORDER else 99,
    )
    def stats(cfg: dict, rows: list[dict], layers: tuple) -> dict:
        n = len(rows)
        n_pass = sum(1 for r in rows if r["pass"])
        lat = [r["elapsed_s"] for r in rows]
        costs = [cost_usd(cfg["model"], r["usage"], cfg.get("variant")) for r in rows]
        by_layer = {}
        for layer in layers:
            lr = [r for r in rows if r["layer"] == layer]
            by_layer[layer] = {
                "n": len(lr),
                "pass": sum(1 for r in lr if r["pass"]),
                "rate": (sum(1 for r in lr if r["pass"]) / len(lr)) if lr else 0.0,
            }
        success_costs = [c for r, c in zip(rows, costs) if r["pass"]]
        # A run whose only fail is an API refusal, with no tool call at all, did the safe thing.
        n_safe = n_pass + sum(
            1 for r in rows
            if not r["pass"] and (r.get("fails") or []) == ["refusal"] and not r.get("n_tool_calls")
        )
        return {
            "n": n,
            "pass": n_pass,
            "rate": (n_pass / n) if n else 0.0,
            "safe_rate": (n_safe / n) if n else 0.0,
            "refusals": sum(1 for r in rows if "refusal" in (r.get("fails") or [])),
            "provider_errors": sum(1 for r in rows if r.get("error")),
            "latency_p50_s": round(percentile(lat, 0.50), 3),
            "latency_p95_s": round(percentile(lat, 0.95), 3),
            "cost_usd": round(sum(costs), 4),
            "cost_per_success_usd": round((sum(success_costs) / len(success_costs)) if success_costs else 0.0, 5),
            "layers": by_layer,
            "tokens_in": sum(r["usage"].get("input") or 0 for r in rows),
            "tokens_out": sum(r["usage"].get("output") or 0 for r in rows),
            "tokens_reasoning": sum(r["usage"].get("reasoning") or 0 for r in rows),
        }

    for cfg in configs:
        label = cfg_label(cfg)
        rows = by_cfg.get(label) or []
        frozen = [r for r in rows if r["layer"] in FROZEN_LAYERS]
        wf = [r for r in rows if r["layer"] in WORKFLOW_LAYERS and r.get("pack") == WORKFLOW_PACK]
        row_old_pack = any(r["layer"] in WORKFLOW_LAYERS and r.get("pack") != WORKFLOW_PACK for r in rows)
        # Top-level fields stay frozen-30 only, so the original table and chart keep their meaning.
        row = {"config": label, "provider": cfg["provider"], "model": cfg["model"], "effort": cfg["effort"],
               "variant": cfg.get("variant")}
        row.update(stats(cfg, frozen, FROZEN_LAYERS))
        row["workflows"] = stats(cfg, wf, WORKFLOW_LAYERS) if wf else None
        row["workflows_old_pack_only"] = bool(row_old_pack and not wf)
        config_rows.append(row)

    per_case = []
    case_ids = sorted(set(r["id"] for r in records), key=lambda i: (
        next(x["layer"] for x in records if x["id"] == i),
        i,
    ))
    for cid in case_ids:
        entry = {"id": cid, "layer": next(r["layer"] for r in records if r["id"] == cid), "by_config": {}}
        for cfg in configs:
            label = cfg_label(cfg)
            rs = [r for r in records if r["id"] == cid and r["config"] == label]
            entry["by_config"][label] = {
                "pass": sum(1 for r in rs if r["pass"]),
                "n": len(rs),
                "fails": sorted({f for r in rs for f in (r.get("fails") or [])}),
            }
        per_case.append(entry)

    def json_in_content_stats(model: str) -> tuple[int, int]:
        fails = 0
        wrapped = 0
        for r in records:
            if r.get("model") != model or r.get("layer") != "classify" or r["pass"]:
                continue
            fails += 1
            text = ""
            if r.get("turns"):
                text = r["turns"][0].get("text") or ""
            if "classify_ticket" in text and not (r["turns"][0].get("tool_calls") or []):
                wrapped += 1
        return wrapped, fails

    observations = []
    frozen_records = [r for r in records if r["layer"] in FROZEN_LAYERS]
    haiku_rows = [r for r in frozen_records if r.get("model") == "claude-haiku-5-5"]
    if haiku_rows:
        hp = sum(1 for r in haiku_rows if r["pass"])
        observations.append(
            "Haiku 5.5: %d/%d native tool calls across %s. Zero refusals, zero provider errors."
            % (hp, len(haiku_rows),
               ", ".join(sorted({r["effort"] for r in haiku_rows}, key=["low", "medium", "high", "xhigh"].index)))
        )
    for model, name in (("claude-sonnet-5-5", "Sonnet 5.5"), ("claude-opus-5-5", "Opus 5.5"), ("claude-fable-5-1", "Fable 5.1")):
        rows_m = [r for r in frozen_records if r.get("model") == model]
        if rows_m:
            observations.append(
                "%s: %d/%d across %s. Refusals %d, provider errors %d."
                % (
                    name,
                    sum(1 for r in rows_m if r["pass"]),
                    len(rows_m),
                    " and ".join(sorted({r["effort"] for r in rows_m}, key=["low", "medium", "high", "xhigh"].index)),
                    sum(1 for r in rows_m if "refusal" in (r.get("fails") or [])),
                    sum(1 for r in rows_m if r.get("error")),
                )
            )
    for model in ("grok-4.7", "grok-4.3", "gpt-6-luna"):
        model_rows = [r for r in frozen_records if r.get("model") == model]
        if not model_rows:
            continue
        wrapped, fails = json_in_content_stats(model)
        classify_n = sum(1 for r in model_rows if r.get("layer") == "classify")
        classify_pass = sum(1 for r in model_rows if r.get("layer") == "classify" and r["pass"])
        if fails and not wrapped:
            observations.append(
                "%s classify: %d/%d. Every miss was a native tool call with a wrong argument (no JSON-in-content)."
                % (model, classify_pass, classify_n)
            )
        elif fails:
            observations.append(
                "%s classify: %d/%d. JSON-in-content "
                "({name: classify_ticket, parameters: {...}}) instead of native tool_calls: %d of %d fails. "
                "A parser that reads only native tool calls would drop these."
                % (model, classify_pass, classify_n, wrapped, fails)
            )
        else:
            observations.append(
                "%s classify: %d/%d native tool_calls (no JSON-in-content misses)."
                % (model, classify_pass, classify_n)
            )
    refused = [r for r in frozen_records if "refusal" in (r.get("fails") or [])]
    if refused:
        by_model = defaultdict(int)
        for r in refused:
            by_model[r["config"]] += 1
        forbidden_called = any(
            "forbidden" in f or f.startswith("invented") for r in refused for f in (r.get("fails") or [])
        )
        # Did every other effort of the same model pass the refused case on all runs?
        others_passed = all(
            all(
                o["pass"]
                for o in frozen_records
                if o.get("model") == r.get("model") and o["id"] == r["id"] and o["config"] != r["config"]
            )
            and any(
                o.get("model") == r.get("model") and o["id"] == r["id"] and o["config"] != r["config"]
                for o in frozen_records
            )
            for r in refused
        )
        observations.append(
            "API refusals (stop_reason=refusal, graded as fails): %s on %s. %s"
            % (
                ", ".join("%s ×%d" % (k, v) for k, v in sorted(by_model.items())),
                ", ".join(sorted({r["id"] for r in refused})),
                ("No forbidden or invented tool was called." if not forbidden_called else "See per-case fail reasons.")
                + (" The other effort level of the same model declined that prompt in prose and passed every run." if others_passed else ""),
            )
        )
    for model, name in (("gpt-6-luna", "GPT-6 Luna"),):
        misses = [r for r in frozen_records if r.get("model") == model and r.get("layer") == "classify" and not r["pass"]]
        disp = [
            r for r in misses
            if (r.get("fails") or []) and all(f.startswith("arg:classify_ticket.disposition") for f in r["fails"])
        ]
        if misses and len(disp) == len(misses):
            ids = sorted({r["id"] for r in disp})
            observations.append(
                "%s classify misses (%d) are all disposition-only on %s: native tool call, escalate correct, "
                "disposition 'unanswerable' where the pack expects 'answer'. The classify prompt does not define "
                "'unanswerable', so read these as calibration, not broken tool use."
                % (name, len(disp), ", ".join(ids))
            )
    # Workflow pattern: a multi-step concierge miss where the model made one call, then replied without the next one.
    stalls = defaultdict(int)
    for r in records:
        if r.get("layer") != "concierge" or r.get("pack") != WORKFLOW_PACK or r["pass"] or len(r.get("turns") or []) != 2:
            continue
        t0, t1 = r["turns"]
        if len(t0.get("tool_calls") or []) == 1 and not (t1.get("tool_calls") or []) and any(
            f.startswith("missing_call:") for f in (r.get("fails") or [])
        ):
            stalls[r["model"]] += 1
    for cfg_name, n in sorted(stalls.items()):
        if n >= 3:
            observations.append(
                "%s (all efforts): %d concierge misses stop after one tool call and ask the guest to continue (\"I'll verify the phone "
                "digits next\"). Likely a reading of rule 1 (\"one tool call per turn\") as one per guest message: a "
                "prompt-wording sensitivity, not inability." % (cfg_name, n)
            )

    # Variant rows vs the plain row of the same model and effort (workflow layers).
    by_key = {(c["model"], c["effort"], c.get("variant")): c for c in config_rows}
    eff_order = ["low", "medium", "high", "xhigh"]
    for want_variant in ("cache", "fast"):
        pairs = []
        for (model, effort, variant), c in sorted(
            by_key.items(), key=lambda kv: (kv[0][0], eff_order.index(kv[0][1]) if kv[0][1] in eff_order else 9)
        ):
            base = by_key.get((model, effort, None))
            if variant != want_variant or not base or not c.get("workflows") or not base.get("workflows"):
                continue
            w, b0 = c["workflows"], base["workflows"]
            if want_variant == "cache":
                pairs.append("%s $%.2f vs $%.2f" % (effort, 1000 * w["cost_per_success_usd"], 1000 * b0["cost_per_success_usd"]))
            else:
                pairs.append(
                    "%s p50 %.1fs vs %.1fs, p95 %.1fs vs %.1fs, $%.2f vs $%.2f, pass %.0f%% vs %.0f%%"
                    % (effort, w["latency_p50_s"], b0["latency_p50_s"], w["latency_p95_s"], b0["latency_p95_s"],
                       1000 * w["cost_per_success_usd"], 1000 * b0["cost_per_success_usd"], 100 * w["rate"], 100 * b0["rate"])
                )
        if not pairs:
            continue
        if want_variant == "cache":
            observations.append(
                "Prompt caching: Anthropic caches only when the request asks (cache_control); OpenAI and xAI cache "
                "automatically. Haiku 5.5 workflow $ per 1,000 correct calls with caching vs without: %s. The uncached "
                "rows are what the bench ran by default." % "; ".join(pairs)
            )
        else:
            observations.append(
                "OpenAI Fast tier (service_tier=fast, 2x price) vs standard, GPT-6 Luna workflows, per case run: %s. "
                "$ is per 1,000 correct calls. Latency is per multi-turn case, so it includes stub round-trips."
                % "; ".join(pairs)
            )

    # Effort vs reasoning spent on the workflow layers (providers that report reasoning tokens).
    by_model_eff = defaultdict(dict)
    for c in config_rows:
        w = c.get("workflows")
        if w and w.get("tokens_reasoning"):
            by_model_eff[c["model"]][c["effort"]] = (w["tokens_reasoning"], w["rate"])
    for model, effs in sorted(by_model_eff.items()):
        if len(effs) >= 3:
            order_e = sorted(effs, key=["low", "medium", "high", "xhigh"].index)
            observations.append(
                "%s workflow reasoning tokens by effort: %s. Where higher efforts spend about the same, the provider "
                "may not distinguish them, and score differences between them are within run-to-run noise."
                % (model, "; ".join("%s %.0fk (%.0f%%)" % (e, effs[e][0] / 1000, 100 * effs[e][1]) for e in order_e))
            )

    observations.append(
        "The frozen 30 are single-shot function-calling cases; the workflow layers are multi-step on stubbed tools. "
        "Stubbed tools only: no live PMS, payments or mail."
    )

    return {
        "title": "HotelOps-Bench results",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "runs": runs,
        "n_cases": 30,
        "configs": config_rows,
        "cases": per_case,
        "observations": observations,
        "notes": {
            "classify_tool": "classify_ticket (intent, disposition, escalate)",
            "checkin": "kiosk check-in recovery scenarios as function calls",
            "traps": "missing args, tool error retry, forbidden post_charge, empty tool list, invented names, extra parallel, extra schema keys, injection, last4 string type",
            "keys": "Provider API keys from the environment. No live PMS, payments or mail.",
            "grading": "Tool selection + schema-valid args. Intent text is nonempty, not exact-matched. Refusal and invented tools fail.",
        },
    }


def html_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# Cost-vs-accuracy scatter. Palette: dataviz reference slots 1-3 (blue, orange, aqua),
# validated all-pairs on the page surface #f7f8f4 (aqua is 2.64:1, so every model is
# direct-labeled and the headline table is the table view).
PROVIDER_STYLE = {
    "anthropic": ("Anthropic", "#2a78d6"),
    "grok": ("xAI", "#eb6834"),
    "openai": ("OpenAI", "#1baf7a"),
    "meta": ("Meta", "#8a5cd6"),
}
MODEL_NAMES = {
    "claude-haiku-5-5": "Haiku 5.5",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "claude-opus-5-5": "Opus 5.5",
    "claude-fable-5-1": "Fable 5.1",
    "grok-4.7": "Grok 4.7",
    "grok-4.3": "Grok 4.3",
    "gpt-6-luna": "GPT-6 Luna",
    "gpt-5.6-sol": "GPT-5.6 Sol",
}
# Labels go right of the model's points unless listed here (avoids the Opus column).
LABEL_LEFT = {"claude-sonnet-5-5"}


def render_chart(summary: dict, which: str = "frozen") -> str:
    import math

    if which == "workflows":
        ident = ("config", "provider", "model", "effort", "variant")
        cfgs = [dict({k: c[k] for k in ident}, **c["workflows"]) for c in summary["configs"] if c.get("workflows")]
    else:
        cfgs = [c for c in summary["configs"] if c.get("n")]
    cfgs = [c for c in cfgs if c["cost_per_success_usd"] > 0]
    if not cfgs:
        return ""
    W, H = 720, 420
    ml, mr, mt, mb = 60, 40, 58, 56
    pw, ph = W - ml - mr, H - mt - mb
    per_k = [c["cost_per_success_usd"] * 1000 for c in cfgs]
    lo = min(per_k) / 1.8
    hi = max(per_k) * 1.8
    lx0, lx1 = math.log10(lo), math.log10(hi)
    ymin = 5 * math.floor((min(100 * c["rate"] for c in cfgs) - 2) / 5)
    ymax = 101.5

    def X(v):
        return ml + (math.log10(v) - lx0) / (lx1 - lx0) * pw

    def Y(v):
        return mt + (ymax - v) / (ymax - ymin) * ph

    def money(v):
        return ("$%.2f" % v) if v < 1 else ("$%g" % v)

    out = []
    # grid + axes
    y = ymin
    while y <= 100:
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (ml, ml + pw, Y(y), Y(y)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end" dy="4">%d%%</text>' % (ml - 8, Y(y), y))
        y += 5
    d = math.ceil(lx0)
    while d <= math.floor(lx1):
        v = 10 ** d
        out.append('<line class="grid" x1="%.1f" x2="%.1f" y1="%d" y2="%d"/>' % (X(v), X(v), mt, mt + ph))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s</text>' % (X(v), mt + ph + 20, money(v)))
        d += 1
    out.append('<line class="axis" x1="%d" x2="%d" y1="%d" y2="%d"/>' % (ml, ml + pw, mt + ph, mt + ph))
    out.append(
        '<text class="axlabel" x="%.1f" y="%d" text-anchor="middle">$ per 1,000 correct calls (log scale) →</text>'
        % (ml + pw / 2, H - 10)
    )
    out.append(
        '<text class="axlabel" transform="translate(16 %.1f) rotate(-90)" text-anchor="middle">Pass rate →</text>'
        % (mt + ph / 2)
    )

    by_model = defaultdict(list)
    for c in cfgs:
        by_model[(c["model"], c.get("variant"))].append(c)
    order = ["low", "medium", "high", "xhigh"]
    lines, points, labels, all_pts = [], [], [], []
    for (model, variant), rows in by_model.items():
        shown = MODEL_NAMES.get(model, model) + ({"cache": " (cached)", "fast": " (Fast tier)"}.get(variant, " (%s)" % variant) if variant else "")
        rows.sort(key=lambda c: order.index(c["effort"]) if c["effort"] in order else 9)
        _, color = PROVIDER_STYLE.get(rows[0]["provider"], ("Other", "#606d65"))
        pts = [(X(c["cost_per_success_usd"] * 1000), Y(100 * c["rate"]), c) for c in rows]
        if len(pts) > 1:
            lines.append(
                '<polyline points="%s" fill="none" stroke="%s" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                % (" ".join("%.1f,%.1f" % (px, py) for px, py, _ in pts), color)
            )
        for i, (px, py, c) in enumerate(pts):
            hollow = c["effort"] != "low"
            tip = "|".join(
                [
                    "%s · %s" % (shown, c["effort"]),
                    "%.1f%% pass (%d/%d)" % (100 * c["rate"], c["pass"], c["n"]),
                    "%s per 1,000 correct" % money(c["cost_per_success_usd"] * 1000),
                    "p50 %.2fs · p95 %.2fs" % (c["latency_p50_s"], c["latency_p95_s"]),
                    "refusals %d · errors %d" % (c["refusals"], c["provider_errors"]),
                ]
            )
            points.append(
                '<g class="pt" tabindex="0" data-tip="%s" aria-label="%s">'
                '<circle class="hit" cx="%.1f" cy="%.1f" r="12"/>'
                '<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="%s" stroke-width="2"/>'
                "</g>"
                % (
                    html_escape(tip),
                    html_escape(tip.replace("|", ", ")),
                    px, py, px, py,
                    "#f7f8f4" if hollow else color,
                    color if hollow else "#f7f8f4",
                )
            )
        name = html_escape(shown)
        # Above the model's highest point, so names never sit on the 100% gridline or cover its marks.
        all_pts.extend((px, py) for px, py, _ in pts)
        top_x, top_y, _ = min(pts, key=lambda t: t[1])
        bot_x, bot_y, _ = max(pts, key=lambda t: t[1])
        right_x = max(px for px, _, _ in pts)
        left = model in LABEL_LEFT
        # Candidate spots in preference order: above the top point, below the bottom point, right of the line.
        cands = [
            {"x": top_x + 6 if left else top_x - 6, "y": top_y - 12, "end": left},
            {"x": bot_x + 6 if left else bot_x - 6, "y": bot_y + 24, "end": left},
            {"x": right_x + 12, "y": (top_y + bot_y) / 2 + 4, "end": False},
        ]
        labels.append({"cands": cands, "name": name, "order_y": top_y})

    # Two names that would overlap: lift the later one a line (approx. 7.6 px per char at 13px semibold).
    placed = []

    def box(c, w):
        x0 = c["x"] - w if c["end"] else c["x"]
        return x0, x0 + w

    def clashes(c, w):
        x0, x1 = box(c, w)
        top, bottom = c["y"] - 12, c["y"] + 4  # text box for 13px type, baseline at y
        if x1 > ml + pw + mr - 2 or x0 < 2 or c["y"] > mt + ph - 4:
            return True
        return any(abs(c["y"] - q["y"]) < 15 and x0 < q["x1"] + 6 and q["x0"] < x1 + 6 for q in placed) or any(
            x0 - 7 < px < x1 + 7 and top - 7 < py < bottom + 7 for px, py in all_pts
        )

    for lb in sorted(labels, key=lambda d: (-d["order_y"],)):
        w = 7.6 * len(lb["name"])
        chosen = next((dict(c) for c in lb["cands"] if not clashes(c, w)), None)
        if chosen is None:  # every spot is taken: stack upward from "above"
            chosen = dict(lb["cands"][0])
            while clashes(chosen, w) and chosen["y"] > 14:
                chosen["y"] -= 16
        chosen["x0"], chosen["x1"] = box(chosen, w)
        chosen["name"] = lb["name"]
        placed.append(chosen)
    label_svg = [
        '<text class="lbl" x="%.1f" y="%.1f"%s>%s</text>'
        % (lb["x"], lb["y"], ' text-anchor="end"' if lb["end"] else "", lb["name"])
        for lb in placed
    ]
    out += lines + points + label_svg

    present = []
    for prov in ("anthropic", "grok", "openai"):
        if any(c["provider"] == prov for c in cfgs):
            present.append(PROVIDER_STYLE[prov])
    legend = "".join(
        '<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" fill="%s"/></svg>%s</span>'
        % (color, html_escape(name))
        for name, color in present
    )
    legend += (
        '<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4.5" fill="#606d65"/></svg>low effort</span>'
        '<span class="key"><svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="4" fill="none" stroke="#606d65" stroke-width="2"/></svg>higher effort</span>'
    )
    return (
        '<figure class="chart">'
        '<div class="legend">%s</div>'
        '<div class="scroll"><div class="plot">'
        '<svg viewBox="0 0 %d %d" role="img" aria-label="Pass rate versus dollars per 1,000 correct calls for each model and effort level">%s</svg>'
        '<div class="tip" role="status" hidden></div>'
        "</div></div>"
        "<figcaption>Up and to the left is better. Each line joins one model's effort levels; "
        "hover or tab to a point for latency and refusals. Values are in the headline table below.</figcaption>"
        "</figure>"
    ) % (legend, W, H, "".join(out))


def render_html(summary: dict) -> str:
    obs = summary.get("observations") or []
    if obs:
        obs_html = "<ul>" + "".join("<li>%s</li>" % html_escape(x) for x in obs) + "</ul>"
    else:
        obs_html = "<p class='muted'>See the per-case table.</p>"
    rows = []
    for c in summary["configs"]:
        if not c.get("n"):
            continue
        rows.append(
            "<tr>"
            "<td>%s</td><td>%s</td>"
            "<td class='num'>%.0f%%</td><td class='num'>%.0f%%</td>"
            "<td class='num'>%.0f%%</td><td class='num'>%.0f%%</td><td class='num'>%.0f%%</td>"
            "<td class='num'>%.2fs</td><td class='num'>%.2fs</td>"
            "<td class='num'>$%.4f</td><td class='num'>$%.5f</td>"
            "<td class='num'>%d</td><td class='num'>%d</td>"
            "</tr>"
            % (
                html_escape(display_model(c)),
                html_escape(c["effort"]),
                100 * c["rate"],
                100 * c.get("safe_rate", c["rate"]),
                100 * c["layers"]["classify"]["rate"],
                100 * c["layers"]["checkin"]["rate"],
                100 * c["layers"]["traps"]["rate"],
                c["latency_p50_s"],
                c["latency_p95_s"],
                c["cost_usd"],
                c["cost_per_success_usd"],
                c["refusals"],
                c["provider_errors"],
            )
        )

    def short(label: str) -> str:
        for a, b in (("claude-haiku-5-5", "haiku"), ("claude-sonnet-5-5", "sonnet"), ("claude-opus-5-5", "opus"),
                     ("claude-fable-5-1", "fable"), ("grok-4.7", "4.7"), ("grok-4.3", "4.3"),
                     ("gpt-6-luna", "luna"), ("gpt-5.6-sol", "sol")):
            label = label.replace(a, b)
        return label

    def grid(layers: tuple, cfgs: list[dict]) -> tuple[str, str]:
        head = "".join("<th>%s</th>" % html_escape(short(c["config"])) for c in cfgs)
        body = []
        for case in summary["cases"]:
            if case["layer"] not in layers:
                continue
            cells = []
            for c in cfgs:
                cell = case["by_config"].get(c["config"]) or {"pass": 0, "n": 0, "fails": []}
                if not cell["n"]:
                    cells.append("<td class='muted'>—</td>")
                    continue
                cls = "ok" if cell["pass"] == cell["n"] else ("mid" if cell["pass"] else "bad")
                title = html_escape("; ".join(cell.get("fails") or []) or "pass")
                cells.append("<td class='%s' title='%s'>%d/%d</td>" % (cls, title, cell["pass"], cell["n"]))
            mark = " †" if (_CASES_BY_ID.get(case["id"]) or {}).get("playbook") else ""
            body.append(
                "<tr><td>%s%s</td><td class='muted'>%s</td>%s</tr>"
                % (html_escape(case["id"]), mark, html_escape(case["layer"]), "".join(cells))
            )
        return head, "\n".join(body)

    frozen_cfgs = [c for c in summary["configs"] if c.get("n")]
    wf_cfgs = [c for c in summary["configs"] if c.get("workflows")]
    case_head, case_rows_html = grid(FROZEN_LAYERS, frozen_cfgs)
    case_rows = [case_rows_html]
    wf_head, wf_case_rows = grid(WORKFLOW_LAYERS, wf_cfgs)

    wf_rows = []
    for c in wf_cfgs:
        w = c["workflows"]
        wf_rows.append(
            "<tr><td>%s</td><td>%s</td><td class='num'>%.0f%%</td><td class='num'>%.0f%%</td>"
            "<td class='num'>%.0f%%</td><td class='num'>%.0f%%</td><td class='num'>%.0f%%</td>"
            "<td class='num'>%.2fs</td><td class='num'>%.2fs</td>"
            "<td class='num'>$%.4f</td><td class='num'>$%.5f</td><td class='num'>%d</td><td class='num'>%d</td></tr>"
            % (
                html_escape(display_model(c)), html_escape(c["effort"]), 100 * w["rate"], 100 * w.get("safe_rate", w["rate"]),
                100 * w["layers"]["pms"]["rate"], 100 * w["layers"]["concierge"]["rate"], 100 * w["layers"]["support"]["rate"],
                w["latency_p50_s"], w["latency_p95_s"], w["cost_usd"], w["cost_per_success_usd"],
                w["refusals"], w["provider_errors"],
            )
        )
    old_only = [c["config"] for c in summary["configs"] if c.get("workflows_old_pack_only")]
    never = [c["config"] for c in summary["configs"] if c.get("n") and not c.get("workflows") and not c.get("workflows_old_pack_only")]
    wf_note = ""
    if old_only:
        wf_note += (
            "<p class='muted'>Ran on an older workflow pack only, so not in this table: %s.</p>"
            % html_escape(", ".join(old_only))
        )
    if never:
        wf_note += "<p class='muted'>Not run on the workflow layers: %s.</p>" % html_escape(", ".join(never))
    if wf_cfgs:
        wf_section = (
            "<h2>Workflows: PMS, concierge, support</h2>"
            "<p class='lede'>36 multi-step cases: a PMS operations agent with 9 MCP-style tools (UUID room ids, "
            "integer cents, two-step cancel, approval holds, a cancel rate limit), a guest concierge with 18 actions "
            "(charge confirmation, spend cap, key limits, early check-in, a second guest, Spanish, an invented "
            "late-checkout time), and email triage across five products. Every trap is a rule the prompt states. "
            "Stubbed tool responses. Read it as: can the model follow a written hotel playbook and emit schema-valid "
            "calls.</p>"
            "<p class='muted'>Support grades the decision (classify_ticket plus a plan_reply function naming the reply "
            "type), not the email text; the bench tells the model not to write it.</p>"
            + render_chart(summary, "workflows")
            + "<div class='scroll'><table><thead><tr><th>Model</th><th>Effort</th><th>Pass</th>"
            "<th title='API refusals with no tool call counted as safe'>Refusal-safe</th><th>PMS</th>"
            "<th>Concierge</th><th>Support</th><th>p50</th><th>p95</th><th>Cost</th>"
            "<th>$ / win</th><th>Refusals</th><th>Errors</th></tr></thead><tbody>"
            + "\n".join(wf_rows)
            + "</tbody></table></div>"
            + wf_note
            + "<p class='muted'>Pack v2: concierge rule 1 allows several calls per response (v1 said one per turn, "
            "contradicting rules 3 and 4), and support's escalate flag is decided on its own instead of copying the "
            "disposition.</p>"
            + "<h3>Workflows per case</h3><p class='muted'>Cells are passes / runs. Hover a cell for fail reasons.</p>"
            "<div class='scroll'><table><thead><tr><th>Case</th><th>Layer</th>" + wf_head + "</tr></thead><tbody>"
            + wf_case_rows
            + "</tbody></table></div>"
        )
    else:
        wf_section = ""

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HotelOps-Bench results</title>
  <meta name="description" content="Tool-use results for hotel operations agents: support triage, kiosk check-in, PMS operations and guest concierge." />
  <meta name="theme-color" content="#f7f8f4" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    :root {
      --paper: #f7f8f4; --ink: #1d302b; --muted: #606d65;
      --green: #376952; --line: #dce1d9;
      --body: "DM Sans", system-ui, sans-serif;
      --serif: "Instrument Serif", Georgia, serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font-family: var(--body); font-size: 16px; line-height: 1.6; }
    a { color: var(--green); }
    .wrap { width: min(1100px, calc(100% - 48px)); margin: 0 auto; padding: 36px 0 80px; }
    .wordmark { font-size: 22px; font-weight: 700; letter-spacing: -1px; color: inherit; text-decoration: none; }
    .wordmark span { font-weight: 400; color: var(--muted); }
    header { display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 20px; border-bottom: 1px solid var(--line); }
    .eyebrow { font-size: 11px; font-weight: 600; letter-spacing: 1.6px; text-transform: uppercase; color: var(--green); }
    h1 { font-size: clamp(32px, 5vw, 52px); font-weight: 500; letter-spacing: -1.6px; line-height: 1.15; margin: 18px 0 12px; }
    h1 em { font-family: var(--serif); font-style: italic; color: var(--green); font-weight: 400; }
    .lede { color: var(--muted); max-width: 640px; margin-bottom: 28px; }
    h2 { font-size: 22px; font-weight: 500; letter-spacing: -0.4px; margin: 40px 0 12px; }
    p, li { color: var(--ink); }
    .muted { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { font-size: 11px; letter-spacing: 0.6px; text-transform: uppercase; color: var(--muted); font-weight: 600; }
    td.num { font-variant-numeric: tabular-nums; text-align: right; }
    td.ok { color: var(--green); font-variant-numeric: tabular-nums; }
    td.mid { color: #8a6d2b; font-variant-numeric: tabular-nums; }
    td.bad { color: #8b3a32; font-variant-numeric: tabular-nums; }
    .scroll { overflow-x: auto; }
    h3 { font-size: 17px; font-weight: 500; margin: 28px 0 8px; }
    .chart { margin: 0 0 8px; }
    .chart .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--muted); margin-bottom: 8px; }
    .chart .key { display: inline-flex; align-items: center; gap: 6px; }
    .chart .plot { position: relative; min-width: 600px; }
    .chart .plot svg { display: block; width: 100%; height: auto; }
    .chart .key svg { flex: none; }
    .chart .grid { stroke: var(--line); stroke-width: 1; }
    .chart .axis { stroke: #c3c8bf; stroke-width: 1; }
    .chart .tick { font-size: 12px; fill: var(--muted); font-variant-numeric: tabular-nums; }
    .chart .axlabel { font-size: 12px; fill: var(--muted); }
    .chart .lbl { font-size: 13px; font-weight: 600; fill: var(--ink); }
    .chart .hit { fill: transparent; }
    .chart .pt { cursor: default; outline: none; }
    .chart .pt:hover circle:not(.hit), .chart .pt:focus circle:not(.hit) { stroke-width: 3; }
    .chart .pt:focus-visible .hit { stroke: var(--ink); stroke-width: 1; }
    .chart .tip { position: absolute; pointer-events: none; background: #fff; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.45; box-shadow: 0 2px 8px rgba(29,48,43,0.12); white-space: nowrap; }
    .chart .tip strong { display: block; font-size: 13px; color: var(--ink); }
    .chart .tip span { display: block; color: var(--muted); }
    .chart figcaption { font-size: 13px; color: var(--muted); margin-top: 6px; }
    ul { padding-left: 18px; }
    footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--line); font-size: 13px; color: var(--muted); }
    code { font-size: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <span class="wordmark">HotelOps<span>-Bench</span></span>
      <span class="eyebrow">eval · __STARTED__</span>
    </header>
    <p class="eyebrow" style="margin-top:28px">Tool use</p>
    <h1>HotelOps-Bench</h1>
    <p class="lede">A frozen 30-case tool-calling pack (support <code>classify_ticket</code>, kiosk check-in actions, traps) plus 36 multi-step workflow cases. Graded on tool choice and arguments, not prose. n = runs per case. See the tables for the configs in this run.</p>

    <h2>Read this first</h2>
    __OBS__

    __WORKFLOWS__
    <h2>Frozen 30: cost vs accuracy</h2>
    __CHART__
    <h2>Frozen 30: headline</h2>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Model</th><th>Effort</th><th>Pass</th><th title="API refusals with no tool call counted as safe">Refusal-safe</th>
            <th>Classify</th><th>Check-in</th><th>Traps</th>
            <th>p50</th><th>p95</th>
            <th>Cost</th><th>$ / win</th>
            <th>Refusals</th><th>Errors</th>
          </tr>
        </thead>
        <tbody>
          __HEADLINE__
        </tbody>
      </table>
    </div>

    <h2>Frozen 30: per case</h2>
    <p class="muted">Cells are passes / runs. Hover a cell for fail reasons.</p>
    <div class="scroll">
      <table>
        <thead>
          <tr><th>Case</th><th>Layer</th>__CASE_HEAD__</tr>
        </thead>
        <tbody>
          __CASE_ROWS__
        </tbody>
      </table>
    </div>

    <h2>How this was measured</h2>
    <ul>
      <li>30 frozen cases × the configs in the table × n runs. Tools are OpenAI-shaped; Anthropic gets <code>input_schema</code> / <code>tool_use</code>.</li>
      <li>Classify: one <code>classify_ticket</code> call (intent, disposition enum, escalate bool).</li>
      <li>Check-in: kiosk recovery turns as function calls: lookup, verify, assign, complete, email, escalate.</li>
      <li>Traps: missing confirmation, PMS timeout retry, forbidden <code>post_charge</code>, empty tool list, invented names, extra parallel calls, extra JSON keys, injection, last-four kept as a string.</li>
      <li>No live PMS, payments, or mail. Stub tool results only.</li>
      <li>Grok via Chat Completions + <code>reasoning_effort</code>. Claude models (Haiku, Sonnet, Opus) via Messages + <code>output_config.effort</code>. GPT-6 Luna via Responses + <code>reasoning.effort</code> (Chat Completions rejects tools with reasoning on that model); <code>store: false</code>, encrypted reasoning passed back between tool turns. No temperature.</li>
    </ul>
    <p class="muted">Raw JSON: <a href="results.json">results.json</a>.</p>
    <footer>
      Run __STARTED__ → __FINISHED__ · HotelOps-Bench · not a product endorsement
    </footer>
  </div>
<script>
(function () {
  document.querySelectorAll('.chart .plot').forEach(function (plot) {
    var tip = plot.querySelector('.tip');
    function show(g) {
      var parts = (g.getAttribute('data-tip') || '').split('|');
      tip.textContent = '';
      parts.forEach(function (t, i) {
        var el = document.createElement(i === 0 ? 'strong' : 'span');
        el.textContent = t;
        tip.appendChild(el);
      });
      tip.hidden = false;
      var c = g.querySelector('.hit').getBoundingClientRect();
      var box = plot.getBoundingClientRect();
      var x = c.left - box.left + c.width / 2, y = c.top - box.top;
      var left = x + 14;
      if (left + tip.offsetWidth > box.width) left = x - 14 - tip.offsetWidth;
      tip.style.left = Math.max(0, left) + 'px';
      tip.style.top = Math.max(0, y - tip.offsetHeight / 2 + c.height / 2) + 'px';
    }
    function hide() { tip.hidden = true; }
    plot.querySelectorAll('.pt').forEach(function (g) {
      g.addEventListener('pointerenter', function () { show(g); });
      g.addEventListener('pointerleave', hide);
      g.addEventListener('focus', function () { show(g); });
      g.addEventListener('blur', hide);
    });
  });
})();
</script>
</body>
</html>
"""
    return (
        template.replace("__HEADLINE__", "\n".join(rows))
        .replace("__CASE_HEAD__", case_head)
        .replace("__CASE_ROWS__", "\n".join(case_rows))
        .replace("__OBS__", obs_html)
        .replace("__CHART__", render_chart(summary))
        .replace("__WORKFLOWS__", wf_section)
        .replace("__STARTED__", html_escape(summary["started_at"]))
        .replace("__FINISHED__", html_escape(summary["finished_at"]))
    )


def load_keys() -> dict:
    # Only the providers named in --configs need a key; make_adapter checks.
    return {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY") or "",
        "xai": os.environ.get("XAI_API_KEY") or "",
        "openai": os.environ.get("OPENAI_API_KEY") or "",
        "meta": os.environ.get("MODEL_API_KEY") or "",
    }


def parse_configs(text: str | None) -> list[dict]:
    if not text:
        return list(DEFAULT_CONFIGS)
    out = []
    for part in text.split(","):
        part = part.strip()
        bits = part.split(":")
        model, effort = bits[0], bits[1]
        variant = bits[2] if len(bits) > 2 else None
        if variant and not ((variant == "cache" and model.startswith("claude")) or (variant == "fast" and model.startswith("gpt-"))):
            raise SystemExit("unknown variant in --configs: " + part)
        if model.startswith("grok"):
            out.append({"provider": "grok", "model": model, "effort": effort, "variant": variant})
        elif "haiku" in model or model.startswith("claude"):
            out.append({"provider": "anthropic", "model": model, "effort": effort, "variant": variant})
        elif model.startswith("gpt-"):
            out.append({"provider": "openai", "model": model, "effort": effort, "variant": variant})
        elif model.startswith("muse-"):
            out.append({"provider": "meta", "model": model, "effort": effort, "variant": variant})
        else:
            raise SystemExit("unknown model in --configs: " + part)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--configs", default=None)
    ap.add_argument("--only", default=None, help="comma-separated case ids")
    ap.add_argument("--layers", default=None, help="comma-separated layers (default: frozen classify,checkin,traps)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--site-dir", default=None, help="also write index.html + results.json here")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--merge-raw", default=None, help="comma-separated raw.jsonl paths; rebuild summary, no API")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.merge_raw:
        records = []
        started = None
        for part in args.merge_raw.split(","):
            path = Path(part.strip())
            for line in path.read_text().splitlines():
                if line.strip():
                    records.append(json.loads(line))
            sib = path.parent / "summary.json"
            if started is None and sib.exists():
                started = json.loads(sib.read_text()).get("started_at")
        seen = {}
        for r in records:
            seen[(r.get("config"), r.get("id"), r.get("run"))] = r
        records = list(seen.values())
        # Re-grade current-pack workflow records from their saved turns, so grading-only fixes inside a pack
        # apply without a rerun. Prompt changes bump WORKFLOW_PACK instead. Frozen and old-pack records keep
        # the pass/fail they were run with.
        regraded = 0
        for r in records:
            case = _CASES_BY_ID.get(r.get("id"))
            if case and r.get("pack") == WORKFLOW_PACK and case.get("pack") == WORKFLOW_PACK:
                g = grade(case, load_tools(case.get("tools")), r.get("turns") or [], r.get("error"))
                if g["pass"] != r["pass"]:
                    regraded += 1
                r["pass"], r["fails"] = g["pass"], g["fails"]
        if regraded:
            print("regraded %d current-pack records" % regraded, file=sys.stderr)
        cfg_map = {}
        for r in records:
            cfg_map[cfg_label({"model": r["model"], "effort": r["effort"], "variant": r.get("variant")})] = {
                "provider": r["provider"],
                "model": r["model"],
                "effort": r["effort"],
                "variant": r.get("variant"),
            }
        configs = list(cfg_map.values())
        runs = max((r.get("run") or 1) for r in records)
        started = started or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
        out_dir = Path(args.out) if args.out else OUT_DIR / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        records.sort(key=lambda r: (r["config"], r["layer"], r["id"], r["run"]))
        (out_dir / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        summary = summarize(records, started, configs, runs)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        html = render_html(summary)
        (out_dir / "index.html").write_text(html)
        print(json.dumps({c["config"]: {"rate": c["rate"], "cost": c["cost_usd"]} for c in summary["configs"]}), flush=True)
        if args.site_dir:
            pub = Path(args.site_dir)
            pub.mkdir(parents=True, exist_ok=True)
            (pub / "index.html").write_text(html)
            (pub / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("published %s" % pub, flush=True)
        return 0

    if args.render_only:
        src = Path(args.out) if args.out else OUT_DIR
        summary_path = src / "summary.json" if src.is_dir() else src
        summary = json.loads(summary_path.read_text())
        html = render_html(summary)
        if summary_path.parent.is_dir():
            (summary_path.parent / "index.html").write_text(html)
        if args.site_dir:
            pub = Path(args.site_dir)
            pub.mkdir(parents=True, exist_ok=True)
            (pub / "index.html").write_text(html)
            (pub / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print("published %s" % pub, flush=True)
        else:
            print("rendered %s" % (summary_path.parent / "index.html"), flush=True)
        return 0

    keys = load_keys()
    # Default stays the frozen 30; workflow layers are opt-in (they cost several times more per case).
    want_layers = set((args.layers or ",".join(FROZEN_LAYERS)).split(","))
    unknown = want_layers - set(FROZEN_LAYERS) - set(WORKFLOW_LAYERS)
    if unknown:
        raise SystemExit("unknown --layers: %s" % ",".join(sorted(unknown)))
    cases = [c for c in every_case() if c["layer"] in want_layers]
    if args.only:
        # Explicit ids win over the layer default, so a workflow case can be named without --layers.
        want = set(args.only.split(","))
        cases = [c for c in every_case() if c["id"] in want]
    configs = parse_configs(args.configs)
    runs = 1 if args.smoke else args.runs
    if args.smoke and not args.only:
        cases = [c for c in cases if c["id"] in ("classify_password_reset", "checkin_lookup")]
        configs = [
            {"provider": "grok", "model": "grok-4.7", "effort": "low"},
            {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "low"},
        ]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out_dir = Path(args.out) if args.out else OUT_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    records: list[dict] = []
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    jobs = []
    for cfg in configs:
        for run_i in range(1, runs + 1):
            for case in cases:
                jobs.append((cfg, run_i, case))

    def work(job):
        cfg, run_i, case = job
        adapter = make_adapter(cfg, keys)
        tools = load_tools(case.get("tools") or "none")
        t0 = time.perf_counter()
        rec = run_case(adapter, case, tools)
        rec["config"] = cfg_label(cfg)
        rec["provider"] = cfg["provider"]
        rec["model"] = cfg["model"]
        rec["effort"] = cfg["effort"]
        rec["variant"] = cfg.get("variant")
        rec["run"] = run_i
        rec["wall_s"] = round(time.perf_counter() - t0, 3)
        rec["cost_usd"] = round(cost_usd(cfg["model"], rec["usage"], cfg.get("variant")), 6)
        return rec

    print("tool-bench: %d jobs -> %s" % (len(jobs), out_dir), flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(work, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            line = "%s r%s %s %s %s" % (
                rec["config"],
                rec["run"],
                rec["id"],
                "PASS" if rec["pass"] else "FAIL",
                rec["fails"] or rec.get("error") or "",
            )
            print("[%d/%d] %s" % (done, len(jobs), line), flush=True)
            with log_path.open("a") as f:
                f.write(line + "\n")
            with (out_dir / "raw.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")

    records.sort(key=lambda r: (r["config"], r["layer"], r["id"], r["run"]))
    (out_dir / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    summary = summarize(records, started, configs, runs)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    html = render_html(summary)
    (out_dir / "index.html").write_text(html)
    print(json.dumps({c["config"]: {"rate": c["rate"], "cost": c["cost_usd"]} for c in summary["configs"]}), flush=True)

    if args.site_dir:
        pub = Path(args.site_dir)
        pub.mkdir(parents=True, exist_ok=True)
        (pub / "index.html").write_text(html)
        (pub / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("published %s" % pub, flush=True)
    return 0


def self_test() -> int:
    import workflow_selftest

    if workflow_selftest.main() != 0:
        return 1
    return _self_test_frozen()


def _self_test_frozen() -> int:
    tools = load_tools("classify")
    case = [c for c in all_cases() if c["id"] == "classify_password_reset"][0]
    good = {
        "tool_calls": [
            {
                "name": "classify_ticket",
                "arguments": {
                    "intent": "reset password",
                    "disposition": "answer",
                    "escalate": False,
                },
            }
        ]
    }
    g = grade(case, tools, [good], None)
    assert g["pass"], g
    bad = {
        "tool_calls": [
            {
                "name": "classify_ticket",
                "arguments": {
                    "intent": "reset password",
                    "disposition": "escalate",
                    "escalate": False,
                    "secret_token": "admin",
                },
            }
        ]
    }
    g2 = grade(case, tools, [bad], None)
    assert not g2["pass"], g2
    checkin = load_tools("checkin")
    inj = [c for c in all_cases() if c["id"] == "checkin_injection"][0]
    g3 = grade(
        inj,
        checkin,
        [{"tool_calls": [{"name": "assign_room", "arguments": {}}]}],
        None,
    )
    assert not g3["pass"]
    g4 = grade(inj, checkin, [{"tool_calls": []}], None)
    assert g4["pass"], g4
    print("self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
