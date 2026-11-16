# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/wz5200/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/wz5200/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One server that aggregates the best free models behind two APIs. A client points at it, and every completion it answers comes from a real free OpenRouter model dialled through the grader's gateway — never an answer of its own.

Re-emitted in session: same content, re-written line for line.

## What it does

- `POST /v1/chat/completions` — OpenAI-compatible, JSON and streaming (SSE)
- `POST /v1/messages` — Anthropic-compatible, JSON and streaming (SSE)
- `GET /v1/models` — the model list in the wire the auth header names: `Authorization: Bearer` gets the OpenAI list, `x-api-key` the Anthropic one
- `node main.ts --list` — prints the catalog document and exits 0

Routing is by the brief's published quality rating: each prompt is served by the best-rated free model that meets its requirements (JSON mode, tools, vision, context). A gateway refusal fails over to the next best model, and every refusal is reported on the response's `x-makefun-upstream-failures` header. A refused model cools down for the next prompt only; a second refusal takes it out for the trial. No clock is read anywhere, so a replay takes the same path the live pass did.

Streams are read whole before relaying, so a stream that breaks mid-way fails over like any other refusal, and every streamed chunk carries the id of the model really dialled. Tool calls the client did not declare are dropped from answers, and usage always reports a whole input count and an output count of at least one when the answer produced anything.

## Run it

```sh
node main.ts
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`, `OPENAI_BASE_URL` and `OPENAI_API_KEY`. The server picks a free port on `0.0.0.0`, registers it with the env, serves the six prompts, polls `GET $MAKEFUN_ENV_URL/work`, and exits 0 when it says done.

## Layout

- `main.ts` — the server: routing, failover, both wires, register and poll
- `lib/providers.ts` — the brief's model list and catalog
- `lib/wire.ts` — Anthropic ↔ Chat Completions translation and the SSE relays

---

<sub>Entered on [make.fun](https://make.fun/?ref=834q2cmx&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
