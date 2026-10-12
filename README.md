# InferenceMonitor — an autopilot for your AI engine

## What it is

When you run an AI model yourself instead of calling someone else's API, you own a new problem: traffic spikes. A surge of users can overwhelm the engine — answers slow to a crawl, requests pile up, things start failing. Most teams handle this with static settings and hope, or by paying for hardware that sits idle "just in case."

InferenceMonitor is a small watchdog program that sits next to your AI engine, checks its vital signs every two seconds, and automatically protects it when demand surges. Think of it as a thermostat for AI: when things heat up, it eases off gracefully instead of melting down — then returns to full quality when the surge passes. No one gets paged at 3am.

## What it does

- **Watches the engine's health** — memory pressure, waiting requests, answer speed, and how often repeated work gets reused from cache.
- **Decides, using simple fixed rules,** whether the engine is OK, under pressure, or recovering. Three bad readings in a row means pressure; five good ones means recovery. No guesswork.
- **Sheds load without restarting** — under pressure it automatically shortens answers and trims stale conversation history, so the engine does less work per request and drains the backlog.
- **Restarts itself as a last resort** — only if you explicitly allow it, it can restart its own deployment when gentler measures aren't enough.
- **Explains itself out loud** — every decision is narrated in one plain-English sentence by the AI model itself, shown in the live dashboard log.
- **Simulates traffic surges** — the dashboard has Low / Medium / High demand buttons (or a full sequence that runs all three back-to-back). Pick one, watch the governor react in real time, and get an automatic report card when it finishes: peak memory, peak queue, how many times pressure tripped, and concrete suggestions for next time.
- **Tests a real cost saver** — the bundled Prefix-Cache Pilot measures whether rewording prompts (putting the repeated document before the question) makes first answers arrive faster by reusing cached computation instead of redoing it.

## The one design rule that matters

The AI model never makes operational decisions. Boring, predictable Python code makes every call — the model only narrates what was decided, for the audience. That's what makes the system trustworthy: you can read the rules, and the rules don't improvise.

## Tech stack

- **Python 3** — the whole app. The dashboard is plain HTML and JavaScript served directly by the program; no web framework.
- **vLLM** — the open-source engine that actually runs the AI model and serves answers.
- **Qwen3-4B** — the open AI model used both for answering users and for narrating the governor's decisions.
- **Kubernetes** — where the engine lives; the watchdog talks to it when the restart option is enabled.
- **requests** — the single outside Python library, used for plain HTTP calls.

That's it. No database, no build step, no frontend toolchain.

## What's in the box

- `governor.py` — the watchdog: control loop, live dashboard (`:8081`), and request-shaping proxy (`:8082`)
- `cachepilot.py` — the prefix-cache experiment (also runs from a dashboard button)
- `loadgen.py` — a standalone script for generating load from the terminal
- `demo-script.md` — the 60-second guided demo
- `requirements.txt` — just `requests`

## Run it

You need a vLLM engine reachable at `http://localhost:8000` (in the workshop this is a `kubectl port-forward` to the cluster) and Python 3:

```
cd ~/MidwestProject/autopilot
pip install -r requirements.txt
python3 -m py_compile governor.py cachepilot.py loadgen.py && echo SYNTAX-OK
python3 governor.py
```

Open http://localhost:8081 for the dashboard. Press **High demand**, watch the state flip to PRESSURE and the governor narrate itself, then read the report card when the simulation ends.

To let the governor restart its own engine as a last resort:

```
python3 governor.py --allow-restart
```

## Demo

See `demo-script.md` for the 60-second version: calm, surge, response, recovery.

## If something looks wrong

- Dashboard says "metrics error": the port-forward to the engine closed. Restart it.
- A tile shows "-": your vLLM version names its metrics differently. Open `/api/status` on the dashboard port — it lists every metric name it found — and add the missing names to `METRIC_CANDIDATES` in `governor.py`.
- CachePilot reports no cache counters: prefix caching may be switched off where your engine runs. The speed comparison still runs; treat its verdict as the honest result.
