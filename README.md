# Autopilot Governor (+ Prefix-Cache Pilot inside)

An agent that owns its inference: it watches its own vLLM engine's telemetry,
classifies pressure with deterministic rules, actuates without restarts first,
and can restart its own Deployment as the closer. The model never makes
operational decisions -- Python does; the model only narrates.

## Layout

- `governor.py` -- control loop + live dashboard (:8081) + optional proxy (:8082)
- `cachepilot.py` -- standalone prefix-cache A/B demo (also a dashboard button)
- `loadgen.py` -- flood script to create demo pressure
- `demo-script.md` -- the 60-second demo narration

## Quickstart (on your Mac)

Prereqs: the `kubectl port-forward` to the workshop cluster running
(`KUBECONFIG=~/MidwestProject/s01.yaml kubectl port-forward svc/vllm 8000:8000`),
and Python 3.

```
cd ~/MidwestProject/autopilot
pip install -r requirements.txt
python3 -m py_compile governor.py cachepilot.py loadgen.py && echo SYNTAX-OK
python3 governor.py
```

Open http://localhost:8081 -- the dashboard. It shows engine state, KV cache,
queue, TTFT, tokens/s, prefix-cache hit rate, and the decision log.

With the restart closer enabled:

```
python3 governor.py --allow-restart
```

This lets the governor run `kubectl rollout restart deployment/vllm -n workshop-s12`
using `~/MidwestProject/s01.yaml` when terse mode isn't enough. Test that your
token can do it first: `KUBECONFIG=~/MidwestProject/s01.yaml kubectl -n workshop-s12 auth can-i rollout/restart deployment`.

## The pieces

**Control loop** (every 2s): scrape `/metrics` -> resolve metric names across
vLLM versions -> compute pressure (KV >= 85% or queue >= 4, 3 samples to enter,
5 healthy samples to exit) -> actuate.

**Actuators, in order:**
1. Terse mode (no restart): via the proxy (:8082), answers get a 3-sentence
   system instruction, max_tokens capped at 128, stale history compacted.
   Point Open WebUI at `http://localhost:8082/v1` to route chat through it
   (non-streaming in this version).
2. Librarian compaction: same proxy path, history truncated with a note.
3. Prefix-Cache Pilot: dashboard button runs the A/B (question-first vs
document-first layouts) and prints TTFT + cache-counter deltas.
4. Restart closer (opt-in): `kubectl rollout restart` on your own Deployment,
   then a cooldown while it re-reads metrics.

**Narration:** every decision is explained in one sentence by Qwen3-4B itself,
shown in the dashboard log. If the model call fails, the action still happens
and is logged -- narration never blocks control.

## Demo in 60 seconds

See `demo-script.md`.

## Troubleshooting

- Dashboard shows "metrics error": the port-forward window closed. Restart it.
- Tiles show "-": your vLLM version uses different metric names. The dashboard's
  `/api/status` JSON lists every name it found under `latest.found` -- compare
  with `curl localhost:8000/metrics | grep ^vllm:` and extend
  `METRIC_CANDIDATES` in `governor.py`.
- CachePilot says "no prefix-cache counters": prefix caching may be off in the
  workshop Deployment. The TTFT comparison still runs; treat it as the honest
  result.
- `kubectl rollout restart` forbidden: your student token lacks RBAC. Demo the
diagnosis + terse-mode beats only -- still a complete story.
