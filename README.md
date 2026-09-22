# lab-agent

**Argus**, the team's server-diagnosis agent.

It puts a container into an OOM restart loop on purpose, notices it through Prometheus, asks a
language model what to do — and then **stops and waits for a human** before doing anything that
writes. Every component runs in a container. **The host needs nothing installed but Docker.**

This repository is the human gate, the audit trail and durable
checkpointing are all in.

---

# ▶ Deploying on the VM — start here

Seven steps. Copy one block at a time; each says what you should see.

### 1. Get the code and prove it arrived intact

```bash
git clone <REPO-URL> lab-agent
cd lab-agent
docker compose build
docker compose run --rm agent python -m pytest -q
```

**Expect:** `147 passed`.
If the number is lower, stop — the problem is the checkout, not the VM, and nothing after this
will make sense.

> The first build takes 3–6 minutes on one vCPU: it downloads about 100 MB of wheels. Later
> builds take seconds.

### 2. Create your configuration

```bash
cp .env.example .env
```

Do not edit it yet. It already says `LLM_PROVIDER=mock` — free, offline, and it exercises
everything except the model.

### 3. Start the lab

```bash
docker compose up -d
docker compose ps
```

**Expect:** `lab-victim`, `mvp-exporter` and `mvp-prometheus` all `Up`.

Wait about 40 seconds. The victim has to die at least once before there is an incident to find.

### 4. Check that the agent will be able to see the incident

```bash
docker compose run --rm agent python -m agent.wait_for_metrics
```

**Expect:** `fresh samples ready after N check(s)`.

This asks the *same adapter the agent uses*, so if it is satisfied, the agent will be. If it keeps
waiting, go to [DEPLOY.md](DEPLOY.md) §9 — every symptom we have actually hit is listed there.

### 5. One free investigation

```bash
docker compose run --rm agent python -m agent.run
```

**Expect** a trace like this:

```
lab-agent | lab-victim | adapters: PrometheusMetrics, MockLLM, TfidfDocs, SqliteMemory
--------------------------------------------------------------------------------
DETECT   RB-002 | lab-victim was OOM-killed and restarted ...
RECALL   0 prior RB-002 incident(s) on lab-victim in 24h | 4 runbook chunk(s)
REASON   [0/5] use_tool get_container_stats ...
ACT      get_container_stats -> lab-victim: running=True oom_killed=True ...
REASON   [1/5] conclude: lab-victim is in an OOM restart loop ...
RECORD   concluded | saved to episodic memory
--------------------------------------------------------------------------------
VERDICT  ... (confidence 0.9)
```

`DETECT`, `ACT` and `RECORD` are **real** even in mock mode — real Prometheus numbers, a real
Docker call, a real SQLite write, a real audit line. Only the two `REASON` lines are a script.
Everything except the model is now proven working on your VM.

Run it a second time: it should say `1 prior RB-002 incident(s)`. That is episodic memory
surviving on a Docker volume.

### 6. The human gate — the thing Stage 5 added

This is the demo worth showing. It is also free and offline.

```bash
docker compose run --rm agent python -m agent.run --rehearse
```

**Expect** the run to stop, not finish:

```
PAUSED - a write is waiting for a human
  tool       : restart_container({'name': 'lab-victim'})
  confidence : 0.9
  because    : Memory is at the limit; restart to clear the immediate pressure.

  Nothing has been done. The run is on disk and will wait.
```

**The container has now exited.** The run is not held open in memory by a process waiting on
`input()` — it is a row in a SQLite file on the `agent-data` volume. Check if you like:
`docker compose ps` shows no agent running. Go for lunch. Reboot the VM.

Then answer, in a completely new container:

```bash
docker compose run --rm agent python -m agent.run --approve
```

or refuse it:

```bash
docker compose run --rm agent python -m agent.run --deny
```

On `--approve`, `lab-victim` is really restarted. On `--deny`, the agent is told it was refused
and reasons on from there. Either way, read what was written down:

```bash
docker compose run --rm agent cat /data/audit.jsonl
```

**Expect** an `attempting` record written **before** the action, carrying the approval, the
model's confidence and its reasoning — then a `completed` or `denied` record after it. The order
matters: an action that crashed halfway would still have left the first line.

> **Why `--rehearse` and not just a normal run?** Since Stage 4 gave it the runbook, a real model
> usually *declines* to restart — RB-002 says a restart is not a fix. That is the right behaviour
> and a useless demo. `--rehearse` scripts the request so the gate can be shown on demand.
> Everything else in that run is real: real metrics, the real allowlist, the real tool, the real
> audit file, the real checkpoint.

If you want the same story end to end with no containers at all:

```bash
docker compose run --rm agent python -m agent.gate_demo
```

That runs the same incident twice — approved once, denied once — and prints both audit trails.

### 7. Point it at your Qwen

No new code is needed. The adapter speaks `POST /v1/chat/completions`, which is what Ollama,
vLLM, llama.cpp and LM Studio all serve. Edit `.env`:

```
LLM_PROVIDER=openai
LLM_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen2.5:7b-instruct
LLM_REQUIRE_AUTH=false
LLM_RESPONSE_FORMAT=json_object
LLM_TIMEOUT=180
LLM_TEMPERATURE=0
```

Test the **model on its own** before involving the agent:

```bash
docker compose run --rm agent python -m agent.probe
```

One request, printed raw: the config (key masked), the prompt size, the model's exact reply, and
whether Pydantic accepted it. Fix anything wrong here first — it is far easier to read than a
failed investigation.

Then:

```bash
docker compose run --rm agent python -m agent.run
```

### The four settings that matter for a local model

| Setting | Why it is not optional |
|---|---|
| `LLM_BASE_URL` | must end in `/v1`, and must be reachable **from inside a container**. `localhost` there means *the agent container itself* — the single most common way this fails. Use `host.docker.internal` for a model on the VM host, or the machine's IP for one elsewhere. |
| `LLM_REQUIRE_AUTH=false` | the adapter refuses to start without a credential by default — for a paid API a missing key is a typo that should stop the run immediately. A local model has none, so the exemption must be stated. |
| `LLM_RESPONSE_FORMAT=json_object` | asks **your server** for JSON. **Qwen3 emits `<think>…</think>` before every answer**, which fails validation every time and burns the whole 5-step budget. If your server rejects the field, remove this line and disable thinking at the model end instead (`/no_think`). |
| `LLM_TIMEOUT=180` | the 60 s default is generous for a hosted API and tight for a 7B model on CPU. |

If Ollama runs on the VM itself, it must listen beyond loopback for the container to reach it:
`OLLAMA_HOST=0.0.0.0:11434`.

### Stopping

```bash
docker compose down        # stop everything; memory, audit trail and any paused run survive
docker compose down -v     # also delete the volume: memory, audit trail and paused runs are gone
```

Please run `docker compose down` when you finish for the day. `lab-victim` restarts forever by
design, and there is no reason to leave it cycling on a shared 1-vCPU VM overnight.

---

# Two things to know before you run it

**This mounts the Docker socket.** `mvp-exporter` and `mvp-agent` both get
`/var/run/docker.sock`, which is root-equivalent on the host. From the moment it is mounted, two
things stand between a model's suggestion and the whole VM: the allowlist in
`agent/tools/registry.py` (*writes may only target `lab-*`*) and the human gate in `agent/graph.py`.
Both are enforced in Python, never asked for in a prompt. The agent cannot touch `prometheus`,
`cadvisor` or `node-exporter`: those names do not start with `lab-`, and the dispatcher refuses
them — and refuses them *loudly*, with a `refused` line in the audit trail.

**The agent state lives on a volume, and that is not decoration.** Three files sit on
`agent-data`: episodic memory, the audit trail, and the checkpoints. The last one is what makes
the gate work at all — a paused run is a row in that file, and `docker compose run --rm` destroys
everything else the container touched. Point it at the wrong place and the pause disappears
silently: the next command reports *"Nothing is waiting for an answer"*, which reads like nothing
went wrong.

### It will not collide with the monitoring already on the VM

| Already there | This project | Why they coexist |
|---|---|---|
| container `prometheus` | `mvp-prometheus` | container names are unique per host |
| port 9090 | **9091**, on `127.0.0.1` | different host port; inside its container it is still 9090 |
| ports 8080, 9100 | **9101**, on `127.0.0.1` | unused before this |
| their compose project | project `labmvp` | separate network and volumes |

Resting footprint is roughly 350 MiB of RAM and almost no CPU. `lab-victim` is capped at 128 MiB
by its cgroup, so its OOM kills cannot reach anything else on the machine.

---

# What the project is

## In one paragraph

A container called `lab-victim` allocates 10 MB a second against a 128 MiB limit. After about 13
seconds the kernel OOM-kills it, and its restart policy brings it straight back — an **RB-002
container restart loop**. Our exporter publishes that container's state as Prometheus metrics;
Prometheus scrapes them; a pure-function detector reads them and decides an incident exists; a
language model is then asked what to do, given the runbook and the history of previous incidents.
Every tool call passes an allowlist, every write waits for a human, and everything the agent
intends to do is written down before it happens.

## The six moving parts

| Part | Where | What it proves |
|---|---|---|
| The incident | `lab-victim` container | a real OOM loop, contained to 128 MiB |
| Observation | `agent/exporter.py` → Prometheus | metrics are *pulled*, and staleness is a lie you must check for |
| Detection | `agent/detectors.py` | deciding *whether* something is wrong is not a model's job |
| Reasoning | `agent/graph.py` + an LLM | the loop, the step bound, and the trust boundary |
| Safety | `agent/tools/registry.py` | a model's recommendation is not a permission |
| The gate | `interrupt()` + `agent/checkpoint.py` | a pause that outlives the process that made it |

## Layout

| Path | What it is |
|---|---|
| `agent/ports.py` | the four interfaces (`Protocol`) the agent reaches the world through |
| `agent/adapters/` | one implementation per port, chosen in `run.py` and nowhere else |
| `agent/detectors.py` | pure functions: metrics in, an incident or `None` out |
| `agent/graph.py` | the LangGraph wiring: nodes, edges, the step bound, and the gate |
| `agent/models.py` | the Pydantic model the LLM's JSON has to satisfy |
| `agent/tools/registry.py` | `check()` decides, `dispatch()` does — the safety boundary |
| `agent/audit.py` | one JSON line per intended action, written **before** it is attempted |
| `agent/checkpoint.py` | where a paused run is kept, so it survives the process |
| `agent/exporter.py` | reads Docker, publishes Prometheus gauges on :9101 |
| `agent/probe.py` | one deliberate model call, printed raw. First contact with a new provider |
| `agent/gate_demo.py` | approve and deny, side by side, with both audit trails |
| `agent/corpus_experiment.py` | Stage 4's demonstration that retrieval fails *silently* |
| `runbooks/` | procedural memory. RB-002 is the one the agent retrieves |
| `prometheus/` | the scrape config |
| `tests/` | 147 tests, offline and free |

## The rules this code exists to demonstrate

- **A model recommendation is not a permission.** Pydantic checks that a response is well-formed;
  the allowlist decides whether the action is permitted; a human decides whether it happens.
  Three layers, three jobs, never merged.
- **Write tools may only target `lab-*` containers**, enforced in the dispatcher, never in a
  prompt. There is a test asserting that `restart_container(name="postgres")` is refused.
- **The step limit lives in the graph router**, so a model cannot talk its way past it.
- **Audit before action.** `audit.write()` runs before `dispatch()`, so an action that crashes
  mid-flight still leaves a record of what was about to happen and who allowed it.
- **Detectors are pure functions.** No I/O, no network, no clock.
- **Swapping an adapter must not require editing** `graph.py`, `detectors.py` or any test. If it
  does, the port is wrong — fix the port.

## Developing without Docker

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The tests need no Docker, no network and no API key. `python -m agent.run` does need the stack up,
and falls back to laptop defaults (`localhost:9090`, `./lab_agent.db`, `./audit.jsonl`,
`./lab_checkpoints.db`) for anything not set in the environment.
