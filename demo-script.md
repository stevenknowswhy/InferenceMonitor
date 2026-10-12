# Demo script (60 seconds)

One-liner for the judges: "Most agents decide what to say. Mine decides how it
should run -- and checks whether the change made inference better."

## Setup (before the audience watches)

1. Port-forward running: `KUBECONFIG=~/MidwestProject/s01.yaml kubectl port-forward svc/vllm 8000:8000`
2. `cd ~/MidwestProject/autopilot && pip install -r requirements.txt`
3. `python3 governor.py` (add `--allow-restart` only if you verified RBAC)
4. Dashboard open at http://localhost:8081 on the projector.

## The 60 seconds

**0:00-0:10 -- Calm.** "This is my agent's engine, live. KV cache low, queue
empty, state NORMAL. Every number here is scraped from my own vLLM's
Prometheus endpoint -- the same telemetry a rented API hides from you."

**0:10-0:30 -- Crisis.** Hit **Start flood** on the dashboard. "I'm flooding my
own engine. Watch the queue climb and the state flip to PRESSURE." Point at
the badge change and the decision log: the agent announces terse mode *in its
own words* -- the model narrates, deterministic Python decided.

**0:30-0:45 -- CachePilot inside.** Hit **Run CachePilot**. "While it's under
load, here's the second actuator: my agent noticed repeated prefill work and
restructured its prompts to reuse the engine's prefix cache." Read the TTFT
before/after and the hit-counter deltas off the screen.

**0:45-0:60 -- Recovery + receipts.** Hit **Stop flood**. "Load's off. Watch it
stand down to NORMAL on its own -- full answers restored. Every beat you saw
was a measured number, not a slide." (If `--allow-restart` is on and pressure
sustained: "...and when terse mode wasn't enough, it restarted its own
deployment. That's the closer.")

## Fallbacks (if anything hiccups)

- Flood doesn't move the gauges: raise concurrency (`python3 loadgen.py
  --concurrency 16 --duration 120` in a terminal) -- the dashboard button is
  just a convenience.
- Restart forbidden/slow: skip it live. The diagnosis + terse-mode + CachePilot
  beats are a complete story without any restart.
- Metrics names differ: show `latest.found` in `/api/status` and narrate the
  auto-discovery ("it adapts to whatever this vLLM version exposes").
