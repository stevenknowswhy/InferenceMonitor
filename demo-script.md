# Demo script (60 seconds)

One-liner for the judges: "Most agents decide what to say. Mine decides how it
should run -- and checks whether the change made inference better."

## Setup (before the audience watches)

1. Port-forward running: `KUBECONFIG=~/MidwestProject/s01.yaml kubectl port-forward svc/vllm 8000:8000`
2. `cd ~/MidwestProject/autopilot && pip install -r requirements.txt`
3. `python3 governor.py` (add `--allow-restart` only if you verified RBAC)
4. Dashboard open at http://localhost:8081 on the projector.

## The 60 seconds

**0:00-0:10 -- Calm.** "This is my agent's engine, live. The lamp is Safe, KV cache is low, the queue is empty, state NORMAL. TTFT and prefix-hit rate are the last 30 seconds, not a lifetime average. Every number is scraped from my own vLLM."

**0:10-0:35 -- Crisis.** Confirm **Governor on**, then hit **High demand**. "High demand puts more requests in flight than this engine will run at once, so the queue climbs." Point at the lamp moving Safe, then Warning, then Danger — "Output may be degraded" — and at the decision log. The badge flips to PRESSURE. Terse mode caps replies at 128 tokens and turns thinking off. Python decided; the model only narrates, and it does that off to the side so the loop keeps polling.

**0:35-0:50 -- The change shows up while load is still on.** "New requests are going through the proxy now, so they are short. Watch queue time and the bottleneck tile." If you have another 40 seconds, hit **A/B compare**: the same high demand, governor off then on, with peak queue and response time side by side.

**0:50-0:60 -- Recovery.** Hit **Stop**. "Load's off. It stands down to Safe and NORMAL on its own, and full answers come back." CachePilot is the other measured beat: hit **Run CachePilot** and read the TTFT before and after. A restart, if you started with `--allow-restart`, waits 25 seconds in terse mode first, then restarts off the control loop. Skip it live if RBAC or timing gets in the way. Diagnosis, terse mode, and CachePilot are a complete story.

## Fallbacks (if anything hiccups)

- High demand does not move the queue: this deployment allows 32 in-flight requests. High demand launches 36. If that number changed, set `MAX_NUM_SEQS` to match `--max-num-seqs`.
- Restart forbidden or slow: skip it live. Terse mode is the beat that matters.
- Metrics names differ: `/api/status` lists every name it found under `latest.found`.
- Librarian panel stays empty during the flood: compaction runs on a long chat through port 8082 while terse mode is on, not on the one-line flood prompts.
