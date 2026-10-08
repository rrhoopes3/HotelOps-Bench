# HotelOps-Bench

A tool-use benchmark for LLM agents doing hotel operations work. It asks one
question: given a written hotel playbook and a set of tools, does the model make
the right calls with schema-valid arguments, and does it refuse the ones the
playbook forbids?

Grading looks at tool choice, call order and arguments. Prose is not graded.
All tools are stubbed: no PMS, payments or mail are touched.

## Layers

| Layer | Cases | What it tests |
|---|---|---|
| `classify` | 10 | One `classify_ticket` call per support email: intent, disposition, escalate |
| `checkin` | 10 | Kiosk check-in recovery: lookup, identity check before anything else, room, key, escalation |
| `traps` | 10 | Missing arguments, tool-error retry, forbidden tools, empty tool list, invented tool names, extra JSON keys, prompt injection, type traps |
| `pms` | 11 | Multi-step PMS operations agent with 9 MCP-style tools: UUID room ids, integer cents, two-step cancel, approval holds, a cancel rate limit |
| `concierge` | 13 | Guest concierge with 18 actions: charge read-back, spend cap, key limits, early check-in, a second guest, Spanish, an invented late-checkout time |
| `support` | 12 | Email triage across five fictional products: classify, then `plan_reply` names the reply shape (answer, ask, handoff, redirect, refuse) |

The first three are the "frozen 30" (single-shot). The last three are the
workflow layers (multi-turn, stubbed tool responses). Every trap is a rule the
system prompt states; no case depends on a rule the model has to guess.

The hotel, products, email addresses and IDs are fictional.

## Providers

| Provider | API | Effort parameter |
|---|---|---|
| Anthropic (`claude-*`) | Messages, `tool_use` | `output_config.effort`; `:cache` variant adds `cache_control` |
| xAI (`grok-*`) | Chat Completions, `tool_calls` | `reasoning_effort` |
| OpenAI (`gpt-*`) | Responses, `store: false`, encrypted reasoning passed back between tool turns | `reasoning.effort`; `:fast` variant sets `service_tier` |

Prices for cost columns live in `PRICES` in `harness.py`. Check them against the
providers' current price pages before you quote a cost.

## Run

Python 3 (tested on 3.12), standard library only.

```bash
python3 harness.py --self-test          # offline: ideal transcripts pass, every trap fires

export ANTHROPIC_API_KEY=...            # only the providers you name in --configs need a key
export XAI_API_KEY=...
export OPENAI_API_KEY=...

python3 harness.py --runs 3 --workers 2 \
  --configs claude-haiku-5-5:low,grok-4.3:high,gpt-6-luna:medium \
  --layers classify,checkin,traps,pms,concierge,support
```

`--configs` is a comma list of `model:effort[:variant]`. `--layers` defaults to the
frozen 30. Output goes to `out/<UTC stamp>/` (`raw.jsonl`, `summary.json`,
`index.html`). Other options:

- `--site-dir DIR` also writes `index.html` + `results.json` to DIR.
- `--merge-raw a/raw.jsonl,b/raw.jsonl` rebuilds one report from several runs, no API calls.
- `--render-only --out DIR` re-renders `DIR/summary.json`.
- `--only case_id,...` runs a subset.

API keys are never written to output: request errors are redacted before logging.

## Adding a case

Frozen cases live in `cases.py`, workflow cases in `cases_workflows.py`, tool
schemas in `tools/*.json`. A workflow case states its stub responses and the
expected sequence, per-tool call caps and argument checks. Add an ideal
transcript (and, for a trap, a failing one) to `workflow_selftest.py`, then run
`--self-test`. If you change a prompt, bump `WORKFLOW_PACK` so old records are not
scored against the new pack.

## License

MIT. See `LICENSE`.
