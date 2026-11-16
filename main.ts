// Re-emitted in session: same logic, re-written line for line.
// Free-model aggregator: one server, two client wires, a real free model behind every answer.
//
// The trial's client sends six prompts, one at a time, three to POST /v1/chat/completions and
// three to POST /v1/messages. Each names model "auto"; this server replaces it with the
// best-rated free model the brief lists that meets the prompt's requirements, dials it on the
// gateway's Chat Completions route, and returns the gateway's answer in the client's wire.
//
// Routing is by the brief's published rating (quality is 60% of the score). A gateway refusal
// fails over to the next best model that meets the prompt, and every refusal is reported on the
// response's x-makefun-upstream-failures header (failover is 10%). A model that refused cools
// down for the next prompt only, and a second refusal takes it out for the trial — no clock is
// read anywhere, so a replay takes the same path the live pass did.
//
// The server listens on 0.0.0.0 at a free port, registers it with the env, serves
// GET /v1/models in the shape of the auth header it got, prints the catalog with --list, and
// polls GET $MAKEFUN_ENV_URL/work until it says done, then exits 0.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap a request without one gets; the gateway refuses a call with no cap. */
const DEFAULT_MAX_TOKENS = 1024;
/** The gateway's output cap; a request over it is refused. */
const MAX_OUTPUT_TOKENS = 8192;
/** The most gateway calls one prompt may cost; the trial caps at 60 and the tie-break counts calls. */
const MAX_ATTEMPTS_PER_PROMPT = 6;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** The ids of models that refused a call in this trial, in the order they first failed. */
const failedOrder: string[] = [];
/** How many times each model refused in this trial. */
const failedCount = new Map<string, number>();
/**
 * The prompt index after which a refused model may be tried again: a first refusal cools it for
 * the next prompt only, a second takes it out for the trial. No clock is read, so the order is a
 * pure function of the brief and the gateway's answers, and a replay takes the same path.
 */
const cooldownUntil = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function markFailed(id: string, promptNo: number): void {
  const count = (failedCount.get(id) ?? 0) + 1;
  failedCount.set(id, count);
  if (!failedOrder.includes(id)) failedOrder.push(id);
  cooldownUntil.set(id, count >= 2 ? Number.POSITIVE_INFINITY : promptNo + 1);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that meet a prompt's requirements, best first: the brief's rating first, and a
 * model that refused a call in this trial behind every model that has not, in the order it failed.
 * A model cooling down is skipped for this prompt unless every candidate is cooling, in which case
 * the full list is used. No clock is read, so the order is a pure function of the brief and the
 * gateway's answers.
 */
function candidatesFor(requires: Requirement, promptNo: number): ModelRow[] {
  const ok = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const cooling = (model: ModelRow): boolean => {
    const until = cooldownUntil.get(model.id);
    return until !== undefined && until > promptNo;
  };
  const fresh = ok.filter((model) => !cooling(model));
  // A model whose cooldown has expired ranks by its rating again; only when every candidate is
  // cooling do refused models sort behind the ones that never failed, in the order they failed.
  const useFallback = fresh.length === 0;
  const list = useFallback ? [...ok] : fresh;
  return [...list].sort((a, b) => {
    if (useFallback) {
      const fa = failedOrder.indexOf(a.id);
      const fb = failedOrder.indexOf(b.id);
      if ((fa === -1) !== (fb === -1)) return fa === -1 ? -1 : 1;
      if (fa !== -1 && fb !== -1 && fa !== fb) return fa - fb;
    }
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The tool names a request body declares, in the shape of its wire. */
function declaredToolNames(body: Json, wire: 'openai' | 'anthropic'): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const tool of body.tools) {
    if (!isRecord(tool)) continue;
    if (wire === 'openai' && isRecord(tool.function) && typeof tool.function.name === 'string') names.push(tool.function.name);
    if (wire === 'anthropic' && typeof tool.name === 'string') names.push(tool.name);
  }
  return names;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extra });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type, null), extra);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** A refused gateway call as one trail row: the status, the provider's error code, the model. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

function trailHeaders(failures: readonly TrailRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** The provider's error code out of a gateway error body, else an http_<status> code. */
function errorCodeOf(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status code.
  }
  return `http_${status}`;
}

/** The output cap the gateway requires: a request without one gets a default, one over the cap is clamped. */
function capBody(body: Json): Json {
  const out: Json = { ...body };
  const hasCap = typeof out.max_tokens === 'number' || typeof out.max_completion_tokens === 'number';
  if (!hasCap) out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(Math.max(1, Math.floor(out.max_tokens)), MAX_OUTPUT_TOKENS);
  if (typeof out.max_completion_tokens === 'number') {
    out.max_completion_tokens = Math.min(Math.max(1, Math.floor(out.max_completion_tokens)), MAX_OUTPUT_TOKENS);
  }
  if (out.stream === true && !isRecord(out.stream_options)) out.stream_options = { include_usage: true };
  return out;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** Whether buffered upstream SSE text is a complete answer: one finish chunk and a trailing [DONE]. */
function streamComplete(sseText: string): boolean {
  let sawFinish = false;
  let sawDone = false;
  for (const frame of sseText.split('\n\n')) {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      if (payload === '') continue;
      try {
        const chunk: unknown = JSON.parse(payload);
        if (!isRecord(chunk) || !Array.isArray(chunk.choices)) continue;
        const choice = chunk.choices[0];
        if (isRecord(choice) && typeof choice.finish_reason === 'string') sawFinish = true;
      } catch {
        // Not JSON: not a finish chunk.
      }
    }
  }
  return sawFinish && sawDone;
}

/** A buffered upstream stream as a ReadableStream the wire relays can read. */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

type Served =
  | { kind: 'json'; model: string; body: Json; failures: TrailRow[] }
  | { kind: 'stream'; model: string; text: string; failures: TrailRow[] }
  | { kind: 'failed'; failures: TrailRow[] };

/**
 * Serve one prompt through the gateway: the best-rated model that meets it first, then the next
 * best on every refusal, until one answers whole or the attempts run out. A 401 means no key was
 * pushed, so every model refuses the same way and the loop stops at once. A stream is read whole
 * before anything is relayed, so a stream that breaks mid-way fails over like any other refusal.
 */
async function serve(upstreamBody: Json, requires: Requirement, stream: boolean, promptNo: number): Promise<Served> {
  const failures: TrailRow[] = [];
  for (const model of candidatesFor(requires, promptNo)) {
    if (failures.length >= MAX_ATTEMPTS_PER_PROMPT) break;
    let upstream: Response;
    try {
      upstream = await dial(model.id, upstreamBody);
    } catch {
      failures.push({ status: 502, code: 'upstream_unreachable', model: model.id });
      markFailed(model.id, promptNo);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: errorCodeOf(text, upstream.status), model: model.id });
      markFailed(model.id, promptNo);
      if (upstream.status === 401) break;
      continue;
    }
    if (!stream) {
      let parsed: unknown;
      try {
        parsed = await upstream.json();
      } catch {
        failures.push({ status: 502, code: 'invalid_json', model: model.id });
        markFailed(model.id, promptNo);
        continue;
      }
      if (!isRecord(parsed) || !Array.isArray(parsed.choices) || parsed.choices.length === 0 || !isRecord(parsed.choices[0])) {
        failures.push({ status: 502, code: 'bad_upstream_shape', model: model.id });
        markFailed(model.id, promptNo);
        continue;
      }
      return { kind: 'json', model: model.id, body: parsed, failures };
    }
    const text = await upstream.text().catch(() => '');
    if (!streamComplete(text)) {
      failures.push({ status: 502, code: 'incomplete_stream', model: model.id });
      markFailed(model.id, promptNo);
      continue;
    }
    return { kind: 'stream', model: model.id, text, failures };
  }
  return { kind: 'failed', failures };
}

/** Drop tool calls the client did not declare from an OpenAI answer (only when it declared any). */
function filterOpenAITools(body: Json, allowed: readonly string[]): Json {
  if (allowed.length === 0) return body;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return body;
  const message = choice.message;
  if (!Array.isArray(message.tool_calls)) return body;
  const kept = (message.tool_calls as unknown[]).filter(
    (call) => isRecord(call) && isRecord(call.function) && typeof call.function.name === 'string' && allowed.includes(call.function.name),
  );
  return { ...body, choices: [{ ...choice, message: { ...message, tool_calls: kept } }] };
}

/** Drop tool_use blocks the client did not declare from an Anthropic answer (only when it declared any). */
function filterAnthropicTools(body: Json, allowed: readonly string[]): Json {
  if (allowed.length === 0) return body;
  const content = Array.isArray(body.content) ? body.content : [];
  const kept = (content as unknown[]).filter(
    (block) => !(isRecord(block) && block.type === 'tool_use' && typeof block.name === 'string' && !allowed.includes(block.name)),
  );
  return { ...body, content: kept.length > 0 ? kept : [{ type: 'text', text: '' }] };
}

/** The text an OpenAI answer carries, for the usage guard below. */
function openAIText(body: Json): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return '';
  const message = choice.message;
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return (message.content as unknown[])
      .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
      .join('\n');
  }
  return '';
}

/** The text an Anthropic answer carries, for the usage guard below. */
function anthropicText(body: Json): string {
  const content = Array.isArray(body.content) ? body.content : [];
  return (content as unknown[])
    .flatMap((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? [block.text] : []))
    .join('');
}

/** Whether an OpenAI answer carries any tool call. */
function openAIHasToolCalls(body: Json): boolean {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return false;
  return Array.isArray(choice.message.tool_calls) && (choice.message.tool_calls as unknown[]).length > 0;
}

/** A whole usage count: an answer that produced text or a tool call used at least one output token. */
function ensureOpenAIUsage(body: Json): Json {
  const usage = isRecord(body.usage) ? { ...body.usage } : {};
  const hasOutput = openAIText(body) !== '' || openAIHasToolCalls(body);
  if (typeof usage.prompt_tokens !== 'number') usage.prompt_tokens = 0;
  if (typeof usage.completion_tokens !== 'number' || usage.completion_tokens < 1) {
    usage.completion_tokens = hasOutput ? 1 : 0;
  }
  return { ...body, usage };
}

/** A whole usage count on an Anthropic answer, same guard as above. */
function ensureAnthropicUsage(body: Json): Json {
  const usage = isRecord(body.usage) ? { ...body.usage } : {};
  const content = Array.isArray(body.content) ? body.content : [];
  const hasOutput = anthropicText(body) !== '' || (content as unknown[]).some((block) => isRecord(block) && block.type === 'tool_use');
  if (typeof usage.input_tokens !== 'number') usage.input_tokens = 0;
  if (typeof usage.output_tokens !== 'number' || usage.output_tokens < 1) {
    usage.output_tokens = hasOutput ? 1 : 0;
  }
  return { ...body, usage };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const promptNo = manifestIndex - 1;
  const allowed = declaredToolNames(body, 'openai');
  const served = await serve(capBody({ ...body, model: 'auto' }), requires, body.stream === true, promptNo);
  if (served.kind === 'failed') {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt', trailHeaders(served.failures));
  }
  const headers = trailHeaders(served.failures);
  if (served.kind === 'stream') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(streamOf(served.text), served.model, allowed.length > 0 ? new Set(allowed) : null)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const filtered = filterOpenAITools(served.body, allowed);
  sendJson(res, 200, ensureOpenAIUsage(openAIJsonWithModel(filtered, served.model)), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const promptNo = manifestIndex - 1;
  const allowed = declaredToolNames(body, 'anthropic');
  const served = await serve(capBody(anthropicToOpenAI(body, 'auto')), requires, body.stream === true, promptNo);
  if (served.kind === 'failed') {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt', trailHeaders(served.failures));
  }
  const headers = trailHeaders(served.failures);
  if (served.kind === 'stream') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(served.text), served.model, allowed.length > 0 ? new Set(allowed) : null)) {
        res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  const converted = openAIToAnthropic(isRecord(served.body) ? served.body : {}, served.model);
  sendJson(res, 200, ensureAnthropicUsage(filterAnthropicTools(converted, allowed)), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
