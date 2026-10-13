#!/usr/bin/env python3
"""Autopilot Governor -- an agent that owns its inference.

Sidecar control loop for a self-hosted vLLM engine:
  1. Scrapes /metrics (auto-discovers metric names across vLLM versions).
  2. Classifies pressure with deterministic rules + hysteresis.
     (The 4B model never makes operational decisions -- Python does.
     The model only narrates.)
  3. Actuates without restarts first:
       - terse mode (shorter answers, capped max_tokens, thinking off) via the proxy
       - librarian compaction (summarize stale history, then check the facts) via the proxy
       - Prefix-Cache Pilot (restructure prompts for cache reuse)
     ...then kubectl rollout restart as the closer (opt-in: --allow-restart).
  4. Narrates every decision through the model itself.
  5. Serves a live dashboard and an optional OpenAI-compatible proxy.

The Prefix-Cache Pilot lives inside as actuator #3 (dashboard button +
shared telemetry) and as a standalone demo: cachepilot.py.

Run:
    pip install -r requirements.txt
    python3 governor.py                       # dashboard :8081, proxy :8082
    python3 governor.py --allow-restart       # enables the restart closer
"""

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import requests
except ImportError:
    sys.exit("Missing dependency: pip install -r requirements.txt")

# ----------------------------------------------------------------------------
# Config (overridable via env)
# ----------------------------------------------------------------------------
VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")
KUBECONFIG = os.environ.get("KUBECONFIG", os.path.expanduser("~/MidwestProject/s01.yaml"))
NAMESPACE = os.environ.get("NAMESPACE", "workshop-s12")
DEPLOYMENT = os.environ.get("VLLM_DEPLOYMENT", "vllm")

POLL_SECS = 2.0
KV_PRESSURE = 0.85
QUEUE_PRESSURE = 4
ENTER_AFTER = 3
EXIT_AFTER = 5
RECOVERY_COOLDOWN = 90
TERSE_WINDOW = 25
WINDOW_SECS = 30.0
COMPARE_SECONDS = 20
# Scheduler slots on this deployment (--max-num-seqs). Used only for the
# live safety lamp, so medium demand can show Warning before the engine
# actually crosses the protection line.
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "32"))
HEALTH_WARN = 0.30
HEALTH_WARN_EXIT = 0.18
HEALTH_SCALE = 1.15
HEALTH_RANK = {"": 0, "safe": 1, "warning": 2, "danger": 3}

METRIC_CANDIDATES = {
    "waiting":        ["vllm:num_requests_waiting"],
    "running":        ["vllm:num_requests_running"],
    "kv":             ["vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc",
                         "vllm:gpu_cache_usage_factor"],
    "ttft_sum":       ["vllm:time_to_first_token_seconds_sum"],
    "ttft_cnt":       ["vllm:time_to_first_token_seconds_count"],
    "prompt_tok":     ["vllm:prompt_tokens_total"],
    "gen_tok":        ["vllm:generation_tokens_total"],
    "prefix_hits":    ["vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits_total"],
    "prefix_queries": ["vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries_total"],
    "prefix_rate":    ["vllm:gpu_prefix_cache_hit_rate"],
    "queue_sum":      ["vllm:request_queue_time_seconds_sum"],
    "queue_cnt":      ["vllm:request_queue_time_seconds_count"],
    "prefill_sum":    ["vllm:request_prefill_time_seconds_sum"],
    "prefill_cnt":    ["vllm:request_prefill_time_seconds_count"],
    "decode_sum":     ["vllm:request_decode_time_seconds_sum"],
    "decode_cnt":     ["vllm:request_decode_time_seconds_count"],
    "preemptions":    ["vllm:num_preemptions_total"],
}

FACT_RE = re.compile(r"\$[\d,]+(?:\.\d+)?|\b\d{4}\b|\b\d+(?:\.\d+)?%?")


def restart_ready(allow, used, now, pressure_since, cooldown_until, under_pressure):
    """Restart only after terse mode has had a real window and pressure is still there."""
    return bool(
        allow and not used and under_pressure and pressure_since
        and now >= pressure_since + TERSE_WINDOW
        and now >= cooldown_until
    )


def classify_bottleneck(waiting, kv, queue_avg, prefill_avg, decode_avg, preemptions_delta):
    """Name the limiter from the signals this vLLM build actually publishes."""
    if waiting >= QUEUE_PRESSURE or (
        waiting > 0 and isinstance(queue_avg, float) and queue_avg >= 0.5
    ):
        return "queue-bound"
    if kv >= KV_PRESSURE or (preemptions_delta or 0) > 0:
        return "KV-bound"
    if isinstance(prefill_avg, float) and isinstance(decode_avg, float):
        if prefill_avg > decode_avg * 1.5 and prefill_avg >= 0.05:
            return "prefill-bound"
        if decode_avg > prefill_avg * 1.5 and decode_avg >= 0.05:
            return "decode-bound"
    return "clear"


def retained_facts(text, summary):
    """Numbers and dates from dropped turns, and how many the summary still contains."""
    facts = []
    for fact in FACT_RE.findall(text or ""):
        if fact not in facts:
            facts.append(fact)
    facts = facts[:12]
    kept = sum(1 for fact in facts if fact in (summary or ""))
    return facts, kept


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
class MetricsScraper:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.resolved = {}
        self._hist = []

    @staticmethod
    def _parse(text):
        vals = {}
        for line in text.splitlines():
            if not line.startswith("vllm:"):
                continue
            m = re.match(r"^(vllm:[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+\-]+)", line)
            if m:
                vals[m.group(1)] = float(m.group(2))
        return vals

    def _resolve(self, vals):
        for canon, cands in METRIC_CANDIDATES.items():
            if canon not in self.resolved:
                for c in cands:
                    if c in vals:
                        self.resolved[canon] = c
                        break

    def snapshot(self):
        r = requests.get(self.base + "/metrics", timeout=10)
        r.raise_for_status()
        vals = self._parse(r.text)
        self._resolve(vals)
        now = time.time()

        def get(canon, default=0.0):
            name = self.resolved.get(canon)
            return vals.get(name, default) if name else default

        snap = {"ts": now, "found": dict(self.resolved)}
        snap["waiting"] = get("waiting")
        snap["running"] = get("running")
        kv = get("kv")
        snap["kv"] = kv / 100.0 if kv > 1.5 else kv
        point = {
            "ts": now,
            "ttft_sum": get("ttft_sum"), "ttft_cnt": get("ttft_cnt"),
            "queue_sum": get("queue_sum"), "queue_cnt": get("queue_cnt"),
            "prefill_sum": get("prefill_sum"), "prefill_cnt": get("prefill_cnt"),
            "decode_sum": get("decode_sum"), "decode_cnt": get("decode_cnt"),
            "hits": get("prefix_hits"), "queries": get("prefix_queries"),
            "preemptions": get("preemptions"),
            "prompt_tok": get("prompt_tok"), "gen_tok": get("gen_tok"),
        }
        prev = self._hist[-1] if self._hist else None
        self._hist.append(point)
        cutoff = now - WINDOW_SECS
        while len(self._hist) > 2 and self._hist[0]["ts"] < cutoff:
            self._hist.pop(0)
        old = self._hist[0] if len(self._hist) > 1 else None

        def avg(sum_k, cnt_k):
            if not old or point["ts"] <= old["ts"]:
                return None
            dc = point[cnt_k] - old[cnt_k]
            ds = point[sum_k] - old[sum_k]
            if dc > 0 and ds >= 0:
                return ds / dc
            return None

        snap["ttft_avg"] = avg("ttft_sum", "ttft_cnt")
        snap["queue_avg"] = avg("queue_sum", "queue_cnt")
        snap["prefill_avg"] = avg("prefill_sum", "prefill_cnt")
        snap["decode_avg"] = avg("decode_sum", "decode_cnt")
        snap["tok_per_s"] = None
        if prev and now > prev["ts"]:
            d = (point["prompt_tok"] + point["gen_tok"]
                 - prev["prompt_tok"] - prev["gen_tok"])
            if d >= 0:
                snap["tok_per_s"] = d / (now - prev["ts"])
        snap["prefix_hit_rate"] = None
        if old and point["queries"] > old["queries"]:
            dh = point["hits"] - old["hits"]
            dq = point["queries"] - old["queries"]
            if dq > 0 and dh >= 0:
                snap["prefix_hit_rate"] = dh / dq
        snap["preemptions_delta"] = (
            max(0.0, point["preemptions"] - old["preemptions"]) if old else 0.0)
        snap["bottleneck"] = classify_bottleneck(
            snap["waiting"], snap["kv"], snap["queue_avg"],
            snap["prefill_avg"], snap["decode_avg"], snap["preemptions_delta"])
        return snap


# ----------------------------------------------------------------------------
# Live safety lamp: safe -> warning -> danger (output may be degraded)
# ----------------------------------------------------------------------------
def assess_health(snap, state, terse, previous, exit_streak):
    """Map one metrics sample onto safe, warning, or danger.

    Danger means replies are being shortened, the engine is restarting, or a
    protection threshold has already been crossed. Warning is elevated load
    short of that line. The lamp steps up on the first sample and steps down
    one band at a time, after two calmer samples, so a gap between flood waves
    does not flicker.
    """
    kv = float((snap or {}).get("kv") or 0)
    waiting = float((snap or {}).get("waiting") or 0)
    running = float((snap or {}).get("running") or 0)
    kv_score = kv / KV_PRESSURE if KV_PRESSURE else 0.0
    queue_score = waiting / float(QUEUE_PRESSURE) if QUEUE_PRESSURE else 0.0
    slot_score = running / float(MAX_NUM_SEQS) if MAX_NUM_SEQS else 0.0
    score = max(kv_score, queue_score, slot_score)
    driver = max(
        (("KV cache", kv_score), ("queue", queue_score), ("in-flight requests", slot_score)),
        key=lambda item: item[1],
    )[0]
    tripped = (
        state in ("PRESSURE", "RECOVERY")
        or terse
        or kv >= KV_PRESSURE
        or waiting >= QUEUE_PRESSURE
    )
    if tripped:
        raw = "danger"
    elif score >= HEALTH_WARN or (
        previous in ("warning", "danger") and score >= HEALTH_WARN_EXIT
    ):
        raw = "warning"
    else:
        raw = "safe"

    if previous and HEALTH_RANK[raw] < HEALTH_RANK.get(previous, 0):
        exit_streak += 1
        if exit_streak >= 2:
            stepped = HEALTH_RANK[previous] - 1
            level = next(name for name, rank in HEALTH_RANK.items() if rank == stepped)
            if HEALTH_RANK[raw] >= HEALTH_RANK[level]:
                level = raw
            exit_streak = 0
        else:
            level = previous
    else:
        level = raw
        exit_streak = 0

    if level == "danger":
        display_score = max(score, 1.0)
    elif level == "warning":
        display_score = max(score, HEALTH_WARN)
    else:
        display_score = score
    if state == "RECOVERY":
        message = "Danger. The engine is restarting, so output may be unavailable."
    elif level == "danger":
        message = "Danger. Output may be degraded."
    elif level == "warning":
        message = "Warning. Load is rising. Answers are still full length."
    else:
        message = "Safe. The engine has headroom and answers are full length."
    detail = (
        f"KV {kv * 100:.0f}% · queue {waiting:.0f} · "
        f"{running:.0f}/{MAX_NUM_SEQS} in flight · "
        f"{min(display_score, HEALTH_SCALE) / HEALTH_SCALE * 100:.0f}% of the way to degraded output"
        f" · driven by {driver}"
    )
    return {
        "level": level,
        "score": round(min(display_score, HEALTH_SCALE), 3),
        "message": message,
        "detail": detail,
        "warn_at": HEALTH_WARN,
        "danger_at": 1.0,
        "scale": HEALTH_SCALE,
        "exit_streak": exit_streak,
    }


# ----------------------------------------------------------------------------
# Governor: deterministic state machine (the model never decides -- it narrates)
# ----------------------------------------------------------------------------
class Governor:
    def __init__(self, allow_restart=False):
        self.scraper = MetricsScraper(VLLM_BASE)
        self.allow_restart = allow_restart
        self.lock = threading.Lock()
        self.state = "NORMAL"
        self.pressure_streak = 0
        self.healthy_streak = 0
        self.cooldown_until = 0
        self.pressure_since = 0
        self.restart_used = False
        self.governed = True
        self.proxy_base = "http://127.0.0.1:8082"
        self.compare_running = False
        self.last_compare = None
        self.last_compaction = None
        self.policy = {"terse": False}
        self.log = []
        self.activity = []
        self.flood_preset = None
        self.flood_stats = None
        self.last_summary = None
        self.sequence_running = False
        self.sequence_abort = False
        self.sequence_step = None
        self.last_sequence = None
        self.latest = {}
        self.flood_on = False
        self.live_responses = set()
        self.hardware = {"ok": False, "error": "Reading the Akamai GPU…"}
        self.metrics_ok = True
        self.metrics_error = ""
        self.health_level = ""
        self.health_exit_streak = 0
        self.health = {
            "level": "",
            "score": 0,
            "message": "Waiting for live metrics.",
            "detail": "",
            "warn_at": HEALTH_WARN,
            "danger_at": 1.0,
            "scale": HEALTH_SCALE,
        }

    def _log(self, action, detail, say=""):
        entry = {"t": datetime.now().strftime("%H:%M:%S"),
                 "action": action, "detail": detail, "say": say}
        with self.lock:
            self.log.append(entry)
            self.log = self.log[-50:]
        print(f"[{entry['t']}] {action}: {detail}", flush=True)
        if say:
            print(f"   says: {say}", flush=True)
        return entry

    def track_response(self, resp):
        with self.lock:
            self.live_responses.add(resp)

    def untrack_response(self, resp):
        with self.lock:
            self.live_responses.discard(resp)

    def abort_live(self):
        """Close every flood connection so vLLM stops generating those replies."""
        with self.lock:
            live = list(self.live_responses)
        closed = 0
        for resp in live:
            if _close_response(resp):
                closed += 1
        return closed

    def _activity(self, msg):
        with self.lock:
            self.activity.append({"t": datetime.now().strftime("%H:%M:%S"), "msg": msg})
            self.activity = self.activity[-40:]

    def narrate(self, trigger, action):
        prompt = ("You are the Autopilot Governor, an agent that owns its inference engine. "
                  f"In ONE sentence for a live demo audience, explain this: {trigger} "
                  f"So you decided: {action}. Be concrete and confident.")
        try:
            r = requests.post(VLLM_BASE + "/v1/chat/completions", timeout=20, json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60, "temperature": 0.3,
                "chat_template_kwargs": {"enable_thinking": False}})
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"(narrator offline) {action}."

    def _narrate_async(self, entry, trigger, action):
        """Narration never blocks the control loop."""
        def run():
            say = self.narrate(trigger, action)
            entry["say"] = say
            if say:
                print(f"   says: {say}", flush=True)
        threading.Thread(target=run, daemon=True).start()

    def tick(self):
        try:
            snap = self.scraper.snapshot()
            self.metrics_ok = True
        except Exception as e:
            self.metrics_ok = False
            self.metrics_error = str(e)[:200]
            self._publish_metrics_down()
            return
        with self.lock:
            self.latest = snap
        kv = snap.get("kv") or 0
        waiting = snap.get("waiting") or 0
        if self.flood_on and self.flood_stats:
            fs = self.flood_stats
            fs["peak_kv"] = max(fs["peak_kv"], kv * 100)
            fs["peak_queue"] = max(fs["peak_queue"], waiting)
            fs["peak_running"] = max(fs["peak_running"], snap.get("running") or 0)
            for key, dest in (("tok_per_s", "tok_s_samples"),
                               ("ttft_avg", "ttft_samples"),
                               ("queue_avg", "queue_samples")):
                val = snap.get(key)
                if isinstance(val, (int, float)):
                    fs.setdefault(dest, []).append(val)
            if self.state == "PRESSURE":
                fs["pressure_seconds"] += POLL_SECS
        pressure = kv >= KV_PRESSURE or waiting >= QUEUE_PRESSURE
        now = time.time()

        if self.state == "NORMAL":
            if pressure:
                self.pressure_streak += 1
                self.healthy_streak = 0
                if self.pressure_streak >= ENTER_AFTER:
                    self._enter_pressure(snap)
            else:
                self.pressure_streak = 0
        elif self.state == "PRESSURE":
            if pressure:
                self.healthy_streak = 0
                if restart_ready(self.allow_restart, self.restart_used, now,
                                 self.pressure_since, self.cooldown_until, True):
                    self._do_restart(snap)
            else:
                self.healthy_streak += 1
                self.pressure_streak = 0
                if self.healthy_streak >= EXIT_AFTER:
                    self._exit_pressure(snap)
        elif self.state == "RECOVERY":
            if now >= self.cooldown_until and not pressure:
                self._exit_pressure(snap)
        self._publish_health(snap)

    def _publish_health(self, snap):
        view = assess_health(
            snap, self.state, bool(self.policy.get("terse")),
            self.health_level, self.health_exit_streak)
        self.health_exit_streak = view.pop("exit_streak")
        self.health_level = view["level"]
        with self.lock:
            self.health = view

    def _publish_metrics_down(self):
        # A missing scrape must not paint Safe. Hold Danger if that was the last real reading.
        if self.health_level == "danger":
            level, score = "danger", 1.0
        else:
            level = "warning"
            score = max(self.health.get("score") or 0, HEALTH_WARN)
        self.health_level = level
        view = {
            "level": level,
            "score": round(score, 3),
            "message": "Metrics are down, so this safety reading may be stale.",
            "detail": self.metrics_error,
            "warn_at": HEALTH_WARN,
            "danger_at": 1.0,
            "scale": HEALTH_SCALE,
        }
        with self.lock:
            self.health = view

    def _enter_pressure(self, snap):
        self.state = "PRESSURE"
        if self.flood_on and self.flood_stats:
            self.flood_stats["pressure_entries"] += 1
            if not self.policy["terse"]:
                self.flood_stats["terse_activations"] += 1
        self.policy["terse"] = True
        self.pressure_since = time.time()
        detail = (f"KV {snap['kv']*100:.0f}% (thresh {KV_PRESSURE*100:.0f}), "
                  f"queue {snap['waiting']:.0f} (thresh {QUEUE_PRESSURE}). "
                  "Engaging TERSE mode: short answers, 128 token cap, thinking off. "
                  f"Restart stays armed for {TERSE_WINDOW:.0f}s.")
        entry = self._log("ENTER_PRESSURE", detail)
        self._narrate_async(
            entry,
            f"engine pressure: KV cache {snap['kv']*100:.0f}%, {snap['waiting']:.0f} requests waiting",
            "I'm switching to terse mode -- shorter answers, capped tokens, thinking off -- "
            "to protect latency. No restart yet.")

    def _exit_pressure(self, snap):
        self.state = "NORMAL"
        self.policy["terse"] = False
        self.restart_used = False
        self.pressure_since = 0
        ttft = snap.get("ttft_avg")
        detail = (f"Pressure cleared (KV {snap['kv']*100:.0f}%, queue {snap['waiting']:.0f}). "
                  "Terse mode off. Full answers restored."
                  + (f" TTFT 30s avg {ttft:.2f}s." if ttft else ""))
        entry = self._log("RECOVERED", detail)
        self._narrate_async(entry, "engine pressure cleared",
                            "Standing down -- full answers restored. The intervention worked.")

    def _do_restart(self, snap):
        self.restart_used = True
        self.state = "RECOVERY"
        self.cooldown_until = time.time() + RECOVERY_COOLDOWN
        entry = self._log(
            "RESTART",
            f"Terse mode held for {TERSE_WINDOW:.0f}s and pressure is still high. "
            "Restarting the deployment off the control loop.")
        self._narrate_async(
            entry,
            f"terse mode wasn't enough (KV {snap['kv']*100:.0f}%)",
            "I'm restarting my own engine now -- back in about a minute. "
            "This is what owning your inference means.")
        threading.Thread(target=self._restart_thread, daemon=True).start()

    def _restart_thread(self):
        ok, out = self._kubectl_restart()
        self._log("RESTART_DONE" if ok else "RESTART_FAILED", out[-300:])

    def _kubectl_restart(self):
        cmd = ["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE,
               "rollout", "restart", f"deployment/{DEPLOYMENT}"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if p.returncode != 0:
                return False, (p.stdout + p.stderr)[-300:]
            w = subprocess.run(
                ["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE,
                 "rollout", "status", f"deployment/{DEPLOYMENT}", "--timeout=180s"],
                capture_output=True, text=True, timeout=200)
            return w.returncode == 0, (w.stdout + w.stderr)[-300:]
        except Exception as e:
            return False, str(e)[:300]


# ----------------------------------------------------------------------------
# Flood generator (in-process, toggled from the dashboard)
# ----------------------------------------------------------------------------
FLOOD_PRESETS = {
    # preset: (workers, max_tokens, duration_seconds)
    # High sits 4 above --max-num-seqs so the engine queue can cross 4 and
    # PRESSURE is reachable on this GPU. KV cache alone never fills here.
    "low": (4, 150, 30),
    "medium": (10, 300, 45),
    "high": (MAX_NUM_SEQS + 4, 320, 40),
}
FLOOD_PROMPTS = [
    "Write a detailed 400-word essay on the history of container orchestration, covering Borg, Omega, Kubernetes, and Nomad with dates and key design decisions.",
    "Explain in depth how the transformer attention mechanism works, including Q/K/V projections, softmax scaling, multi-head attention, and KV caching, with equations.",
    "Give a comprehensive overview of GPU memory hierarchies: registers, shared memory, L1/L2 cache, HBM, and how they affect LLM inference throughput.",
    "Describe the full lifecycle of a Kubernetes pod from scheduling to termination, including probes, init containers, and graceful shutdown. Be thorough.",
]


def fresh_stats(preset, workers, mtok, dur, governed):
    return {
        "preset": preset, "workers": workers, "max_tokens": mtok, "duration": dur,
        "governed": bool(governed), "started": time.time(),
        "requests": 0, "errors": 0, "waves": 0, "finished": 0,
        "peak_kv": 0.0, "peak_queue": 0, "peak_running": 0,
        "pressure_entries": 0, "pressure_seconds": 0,
        "terse_activations": 0, "tok_s_samples": [],
        "ttft_samples": [], "queue_samples": [], "latency_samples": [],
        "cancelled": 0,
    }


def _close_response(resp):
    """Drop the TCP connection behind a requests response, from any thread."""
    if resp is None:
        return False
    try:
        raw = getattr(resp, "raw", None)
        conn = getattr(raw, "_connection", None) if raw is not None else None
        sock = getattr(conn, "sock", None) if conn is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        resp.close()
    except Exception:
        pass
    return True


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 2) if xs else None


def _flood_hit(gov, prompt, n, max_tokens):
    if not gov.flood_on:
        return
    t0 = time.time()
    ok = False
    err = ""
    cancelled = False
    resp = None
    try:
        base = gov.proxy_base if gov.governed else VLLM_BASE
        resp = requests.post(
            base + "/v1/chat/completions",
            timeout=180,
            headers={"X-Governor-Flood": "1"},
            stream=True,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            })
        gov.track_response(resp)
        if not gov.flood_on:
            cancelled = True
        else:
            for _chunk in resp.iter_content(chunk_size=4096):
                if not gov.flood_on:
                    cancelled = True
                    break
            if not cancelled:
                ok = resp.status_code < 400
                if not ok:
                    err = f"HTTP {resp.status_code}"
    except Exception as e:
        if not gov.flood_on:
            cancelled = True
        else:
            err = str(e)[:70]
    finally:
        if resp is not None:
            gov.untrack_response(resp)
            _close_response(resp)
    elapsed = time.time() - t0
    with gov.lock:
        fs = gov.flood_stats
        if not fs:
            return
        fs["finished"] = fs.get("finished", 0) + 1
        if cancelled:
            fs["cancelled"] = fs.get("cancelled", 0) + 1
        elif ok:
            fs["requests"] += 1
            fs["latency_samples"].append(elapsed)
        else:
            fs["errors"] += 1
        finished = fs["finished"]
    if not cancelled and (finished == 1 or finished % 8 == 0):
        route = "proxy" if gov.governed else "direct"
        tail = f" ({err})" if err else f" last {elapsed:.1f}s"
        gov._activity(f"{finished} finished via {route}{tail}")


def flood_worker(gov, workers, max_tokens, duration):
    """Keep `workers` requests in flight until the deadline.

    New requests start as old ones finish, so once terse mode flips, the
    next governed request is already capped. A join-the-whole-wave loop
    would spend the flood on the uncapped first batch.
    """
    deadline = time.time() + duration
    token = object()
    slot = {"n": 0, "inflight": 0, "token": token}
    slock = threading.Lock()

    def refill():
        while gov.flood_on and time.time() < deadline:
            with slock:
                if slot["token"] is not token or slot["inflight"] >= workers:
                    return
                slot["inflight"] += 1
                slot["n"] += 1
                n = slot["n"]
            with gov.lock:
                if gov.flood_stats:
                    gov.flood_stats["waves"] = n
            prompt = FLOOD_PROMPTS[n % len(FLOOD_PROMPTS)]

            def run(n=n, prompt=prompt):
                try:
                    _flood_hit(gov, prompt, n, max_tokens)
                finally:
                    with slock:
                        slot["inflight"] -= 1
                        alive = slot["token"] is token
                    if alive and gov.flood_on and time.time() < deadline:
                        refill()

            threading.Thread(target=run, daemon=True).start()

    route = "governor proxy" if gov.governed else "engine directly"
    gov._activity(f"keeping {workers} requests in flight via the {route}")
    refill()
    while gov.flood_on and time.time() < deadline:
        time.sleep(0.4)
    timed_out = gov.flood_on
    with slock:
        slot["token"] = None
    if timed_out:
        gov.flood_on = False
        wait_until = time.time() + 8
        while time.time() < wait_until:
            with slock:
                if slot["inflight"] == 0:
                    break
            time.sleep(0.2)
        if gov.sequence_running or gov.compare_running:
            gov._activity(f"{gov.flood_preset} run finished")
        else:
            gov._activity("flood finished -- generating report")
            build_flood_summary(gov)


def _summarize_after_stop(gov):
    """Let cancelled requests record themselves, then write the report once."""
    time.sleep(0.8)
    if gov.flood_on or gov.sequence_running or gov.compare_running:
        return
    build_flood_summary(gov)


def build_flood_summary(gov, quiet=False):
    """Called once when a flood stops: turn tracked stats into an improvement report.
    quiet=True skips the dashboard write + narration (used by the sequence runner)."""
    fs = gov.flood_stats
    if not fs:
        return None, []
    dur = time.time() - fs["started"]
    samples = fs["tok_s_samples"]
    avg_ts = round(sum(samples) / len(samples), 1) if samples else None
    lats = fs.get("latency_samples") or []
    avg_lat = _mean(lats)
    avg_ttft = _mean(fs.get("ttft_samples") or [])
    avg_queue = _mean(fs.get("queue_samples") or [])
    sugg = []
    routed = "through the governor proxy" if fs.get("governed") else "direct to the engine"
    if fs["pressure_entries"] == 0:
        sugg.append(
            f"PRESSURE never triggered on '{fs['preset']}' demand "
            f"(peak KV {fs['peak_kv']:.0f}%, queue {round(fs['peak_queue'])}, {routed}). "
            "High demand is the preset sized past this engine's 32 in-flight slots.")
    else:
        sugg.append(
            f"Governor entered PRESSURE {fs['pressure_entries']}x for ~{fs['pressure_seconds']}s "
            f"and engaged terse mode {fs['terse_activations']}x. Peak KV {fs['peak_kv']:.0f}%, "
            f"queue {round(fs['peak_queue'])}. Route: {routed}.")
        if fs["peak_kv"] >= 95:
            sugg.append("KV cache nearly saturated. This is where requests start failing outright.")
        if fs["pressure_seconds"] > 120:
            sugg.append("Recovery was slow. Check whether terse mode is draining the queue.")
    if fs.get("cancelled"):
        sugg.append(f"Stop closed {fs['cancelled']} request(s) that were still generating.")
    if fs["errors"]:
        sugg.append(f"{fs['errors']} requests errored. The engine was over-saturated, not just slow.")
    if fs.get("governed"):
        sugg.append("Governed requests after PRESSURE are capped at 128 tokens with thinking off, "
                    "so later arrivals finish sooner while the flood is still running.")
    else:
        sugg.append("This run bypassed the proxy, so terse mode did not shape these requests.")
    summary = {
        "preset": fs["preset"], "duration_s": round(dur),
        "governed": bool(fs.get("governed")),
        "requests": fs["requests"], "errors": fs["errors"], "waves": fs["waves"],
        "peak_kv": round(fs["peak_kv"]), "peak_queue": round(fs["peak_queue"]),
        "peak_running": fs["peak_running"],
        "avg_tok_s": avg_ts if avg_ts is not None else "n/a",
        "avg_latency": avg_lat if avg_lat is not None else "n/a",
        "avg_ttft": avg_ttft if avg_ttft is not None else "n/a",
        "avg_queue": avg_queue if avg_queue is not None else "n/a",
        "pressure_entries": fs["pressure_entries"],
        "pressure_seconds": fs["pressure_seconds"],
        "terse_activations": fs["terse_activations"],
        "suggestions": sugg,
    }
    if not quiet:
        gov.last_summary = summary
        gov._activity("flood report ready -- see Flood report panel")
        def narrate_report():
            say = gov.narrate(
                "flood report",
                f"Preset {fs['preset']} ({routed}): {fs['requests']} requests in {dur:.0f}s, "
                f"peak KV {fs['peak_kv']:.0f}%, queue {round(fs['peak_queue'])}, "
                f"pressure entries {fs['pressure_entries']}. " + " ".join(sugg[:2]))
            if say:
                summary["narration"] = say
                print(f"   says: {say}", flush=True)
        threading.Thread(target=narrate_report, daemon=True).start()
    return summary, sugg


def sequence_worker(gov):
    """Run low -> medium -> high back-to-back, then a combined comparison report."""
    gov.sequence_running = True
    gov.sequence_abort = False
    runs = []
    try:
        for preset in ("low", "medium", "high"):
            if gov.sequence_abort:
                gov._activity("sequence aborted by user")
                break
            workers, mtok, dur = FLOOD_PRESETS[preset]
            gov.flood_preset = preset
            gov.sequence_step = preset
            gov.flood_stats = fresh_stats(preset, workers, mtok, dur, gov.governed)
            gov.flood_on = True
            gov._activity(f"sequence: {preset} demand started ({dur}s)")
            flood_worker(gov, workers, mtok, dur)
            summary, _sugg = build_flood_summary(gov, quiet=True)
            if summary:
                runs.append(summary)
                gov.last_summary = summary
            time.sleep(3)
    finally:
        gov.flood_on = False
        gov.sequence_running = False
        gov.sequence_abort = False
        gov.sequence_step = None
    if runs:
        gov.last_sequence = {"runs": runs, "narration": None}
        gov._activity("sequence complete: low -> medium -> high -- see Flood report")
        say = gov.narrate(
            "full demand sequence",
            "Ran low, medium, then high demand back to back. " +
            " ".join(f"{r['preset']}: peak KV {r['peak_kv']}%, "
                       f"{r['pressure_entries']} pressure entries, "
                       f"{r['requests']} requests" for r in runs) + ". " +
            "The governor engaged terse mode automatically as pressure rose.")
        if say:
            gov.last_sequence["narration"] = say


def compare_worker(gov, preset):
    """Same preset, governor off then on, so the report is a measured A/B."""
    prev = gov.governed
    gov.compare_running = True
    gov.sequence_abort = False
    sides = []
    try:
        for governed, label in ((False, "off"), (True, "on")):
            if gov.sequence_abort:
                gov._activity("compare aborted")
                break
            workers, mtok, _dur = FLOOD_PRESETS[preset]
            dur = COMPARE_SECONDS
            gov.governed = governed
            gov.flood_preset = preset
            gov.flood_stats = fresh_stats(preset, workers, mtok, dur, governed)
            gov.flood_on = True
            gov._activity(f"compare: governor {label}, {preset} demand, {dur}s")
            flood_worker(gov, workers, mtok, dur)
            summary, _sugg = build_flood_summary(gov, quiet=True)
            if summary:
                summary["label"] = f"governor {label}"
                sides.append(summary)
            time.sleep(2)
    finally:
        gov.governed = prev
        gov.flood_on = False
        gov.compare_running = False
        gov.sequence_abort = False
    verdict = ""
    if len(sides) == 2:
        off, on = sides
        bits = []
        if on["peak_queue"] != off["peak_queue"]:
            bits.append(f"Peak queue {off['peak_queue']} with the governor off, {on['peak_queue']} with it on.")
        if (isinstance(off.get("avg_latency"), (int, float))
                and isinstance(on.get("avg_latency"), (int, float))):
            bits.append(f"Average response {off['avg_latency']}s off, {on['avg_latency']}s on.")
        if on["pressure_entries"] or off["pressure_entries"]:
            bits.append(
                f"PRESSURE entries: {off['pressure_entries']} off, {on['pressure_entries']} on.")
        verdict = " ".join(bits) or "Both sides finished. Compare the table."
    gov.last_compare = {"preset": preset, "sides": sides, "verdict": verdict, "narration": None}
    gov._activity("compare complete -- see Flood report")
    if verdict:
        def narrate():
            say = gov.narrate("governor off versus on", verdict)
            if say and gov.last_compare:
                gov.last_compare["narration"] = say
        threading.Thread(target=narrate, daemon=True).start()


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
DASH_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Autopilot Governor</title>
<style>
body{font-family:-apple-system,Helvetica,Arial,sans-serif;background:#0f1419;color:#e6e6e6;margin:0;padding:20px}
h1{font-size:20px;margin:0 0 12px}
.badge{display:inline-block;padding:4px 12px;border-radius:12px;font-weight:700}
.NORMAL{background:#1d7a3a}.PRESSURE{background:#b3540a}.RECOVERY{background:#2b6cb0}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}
.tile{background:#1a2230;border-radius:8px;padding:12px 16px;min-width:120px}
.tile .v{font-size:24px;font-weight:700}.tile .l{font-size:12px;color:#9aa4b2}
#log{background:#0a0e14;border-radius:8px;padding:12px;height:260px;overflow-y:auto;font-family:monospace;font-size:12px}
#log div{margin-bottom:6px;border-bottom:1px solid #1c2532;padding-bottom:6px}
button{background:#2b6cb0;color:#fff;border:0;border-radius:6px;padding:8px 14px;margin-right:8px;cursor:pointer;font-size:14px}
button.warn{background:#b3540a}
.warnline{color:#f0b429}
pre{background:#0a0e14;padding:12px;border-radius:8px;white-space:pre-wrap;font-size:12px}
.say{color:#7ee2a8}
#activity{background:#0a0e14;border-radius:8px;padding:12px;height:180px;overflow-y:auto;font-family:monospace;font-size:12px;color:#8fd0ff}
#activity div{animation:fadein .4s}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
.fbadge{background:#b3540a;color:#fff;font-size:12px;padding:3px 10px;border-radius:10px;animation:pulse 1.2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.cursor{animation:pulse 1s infinite;color:#8fd0ff}
.pills{display:flex;gap:8px;margin:12px 0;flex-wrap:wrap}
.pill{background:#1c2530;border:1px solid #2c3a4a;color:#cfe3ff;padding:8px 18px;border-radius:20px;cursor:pointer;font-size:14px}
.pill:hover{border-color:#4d6a8a}
.pill.active{background:#b3540a;border-color:#b3540a;color:#fff;font-weight:bold}
.pill.live{background:#1d7a3a;border-color:#1d7a3a;color:#fff;font-weight:bold}
#report{background:#0a0e14;border-radius:8px;padding:12px;font-size:13px}
#report ul{margin:8px 0;padding-left:20px}
#report li{margin:4px 0}
#report table{border-collapse:collapse;margin:8px 0}
#report td,#report th{border:1px solid #2c3a4a;padding:4px 10px;font-size:12px;text-align:left}
#report th{color:#9aa4b2;font-weight:600}
.muted{color:#7d8ea3}
.health-head{display:flex;align-items:center;gap:8px;margin:16px 0 8px}
.health-head h2{font-size:13px;margin:0;letter-spacing:.08em;text-transform:uppercase;color:#9aa4b2}
.live-dot{width:8px;height:8px;border-radius:50%;background:#3ddc84;box-shadow:0 0 8px #3ddc84;animation:pulse 1.2s infinite;display:inline-block}
.modes{display:flex;gap:10px;align-items:stretch}
.mode{flex:1;background:#1a2230;border:2px solid #243044;border-radius:10px;padding:12px 14px;opacity:.4;transition:opacity .35s,border-color .35s,box-shadow .35s}
.mode .k{font-size:11px;letter-spacing:.14em;color:#9aa4b2}
.mode .n{font-size:22px;font-weight:800;margin-top:2px}
.mode .s{font-size:13px;color:#c5d0dc;margin-top:4px;min-height:2.2em}
.mode.safe .n{color:#3ddc84}
.mode.warning .n{color:#f0c14a}
.mode.danger .n{color:#ff6b6b}
.mode.on{opacity:1}
.mode.safe.on{border-color:#1d7a3a;box-shadow:0 0 18px rgba(29,122,58,.35)}
.mode.warning.on{border-color:#d4a017;box-shadow:0 0 18px rgba(212,160,23,.28)}
.mode.danger.on{border-color:#c53030;box-shadow:0 0 18px rgba(197,48,48,.4)}
.meter{position:relative;margin:12px 0 4px;height:20px}
.meter .track{height:14px;border-radius:8px}
.meter .needle{position:absolute;top:-3px;width:3px;height:20px;background:#fff;border-radius:2px;box-shadow:0 0 6px #fff;transform:translateX(-1px);transition:left .45s ease}
#healthmsg{margin:8px 0 0;font-size:15px;font-weight:700}
#healthmsg.safe{color:#7ee2a8}
#healthmsg.warning{color:#f0c14a}
#healthmsg.danger{color:#ff8a8a}
#healthdetail{font-size:12px;color:#7d8ea3;margin-top:2px}
.hw{background:#1a2230;border-radius:10px;padding:14px 16px;margin:14px 0 4px}
.hw-head{display:flex;flex-direction:column;align-items:flex-start;gap:4px;margin-bottom:8px}
.hw-head h2{font-size:13px;margin:0;letter-spacing:.08em;text-transform:uppercase;color:#9aa4b2}
#hwname{font-size:15px;font-weight:650;color:#e6e6e6}
.hbar{display:grid;grid-template-columns:72px minmax(0,1fr) max-content;align-items:center;gap:10px;margin:8px 0}
.hbar .hl{font-size:12px;color:#9aa4b2}
.htrack{background:#0a0e14;border-radius:6px;height:12px;overflow:hidden}
.hfill{height:100%;width:0;border-radius:6px;transition:width .6s ease}
#hwmem{background:#4c8dff}
#hwcomp{background:#3ddc84}
#hwpow{background:#f0c14a}
.ht{font-size:12px;color:#c5d0dc;text-align:right;white-space:nowrap}
@media (max-width:700px){.hbar{grid-template-columns:64px 1fr}.ht{grid-column:2}}
</style></head>
<body>
<h1>Autopilot Governor <span id="state" class="badge NORMAL">...</span></h1>
<div class="health-head"><span class="live-dot"></span><h2>Service health, live</h2></div>
<div class="modes" id="modes">
 <div class="mode safe" data-level="safe"><div class="k">STEP 1</div><div class="n">Safe</div><div class="s">Full answers</div></div>
 <div class="mode warning" data-level="warning"><div class="k">STEP 2</div><div class="n">Warning</div><div class="s">Load rising</div></div>
 <div class="mode danger" data-level="danger"><div class="k">STEP 3</div><div class="n">Danger</div><div class="s">Output may be degraded</div></div>
</div>
<div class="meter"><div class="track" id="healthtrack"></div><div class="needle" id="healthneedle"></div></div>
<div id="healthmsg">Waiting for live metrics.</div>
<div id="healthdetail"></div>
<div class="hw">
 <div class="hw-head"><h2>Akamai GPU</h2><span id="hwname">Reading the card…</span></div>
 <div class="hbar"><div class="hl">VRAM</div><div class="htrack"><div class="hfill" id="hwmem"></div></div><div class="ht" id="hwmemt">-</div></div>
 <div class="hbar"><div class="hl">Compute</div><div class="htrack"><div class="hfill" id="hwcomp"></div></div><div class="ht" id="hwcompt">-</div></div>
 <div class="hbar"><div class="hl">Power</div><div class="htrack"><div class="hfill" id="hwpow"></div></div><div class="ht" id="hwpowt">-</div></div>
 <div id="hwnote" class="muted"></div>
</div>
<div class="tiles">
 <div class="tile"><div class="v" id="kv">-</div><div class="l">KV cache</div></div>
 <div class="tile"><div class="v" id="waiting">-</div><div class="l">queued</div></div>
 <div class="tile"><div class="v" id="running">-</div><div class="l">running</div></div>
 <div class="tile"><div class="v" id="ttft">-</div><div class="l">TTFT 30s</div></div>
 <div class="tile"><div class="v" id="qtime">-</div><div class="l">queue 30s</div></div>
 <div class="tile"><div class="v" id="tps">-</div><div class="l">tok/s</div></div>
 <div class="tile"><div class="v" id="hitrate">-</div><div class="l">prefix hit 30s</div></div>
 <div class="tile"><div class="v" id="bound">-</div><div class="l">bottleneck</div></div>
</div>
<div class="pills">
 <button class="pill" data-preset="low" onclick="startFlood('low')">Low demand <span class="muted">30s</span></button>
 <button class="pill" data-preset="medium" onclick="startFlood('medium')">Medium demand <span class="muted">45s</span></button>
 <button class="pill" data-preset="high" onclick="startFlood('high')">High demand <span class="muted">40s</span></button>
 <button class="pill stop" onclick="stopFlood()">Stop</button>
 <button class="pill live" id="govtoggle" onclick="toggleGoverned()">Governor on</button>
 <button class="pill" onclick="startCompare()">A/B compare <span class="muted">20s × 2</span></button>
 <button class="pill" onclick="startSequence()">&#9654; Full sequence <span class="muted">~2 min</span></button>
 <button class="pill" onclick="cachepilot()">Run CachePilot</button>
 <span id="mstate" style="margin-left:12px;color:#9aa4b2"></span>
</div>
<div id="floodprog" style="display:none;margin:4px 0 12px">
 <div style="background:#1c2530;border-radius:6px;height:10px;overflow:hidden">
  <div id="floodbar" style="height:100%;width:0%;background:#b3540a;transition:width .5s"></div>
 </div>
 <div id="floodtime" class="muted" style="font-size:12px;margin-top:4px"></div>
</div>
<h3>Live activity <span id="floodbadge" class="fbadge" style="display:none">&#9679; FLOOD ACTIVE</span></h3>
<div id="activity"><div>idle -- pick a demand pill above</div></div>
<h3>Flood report</h3><div id="report"><div class="muted">run a flood, then stop it -- the report lands here</div></div>
<h3>Librarian</h3><div id="librarian"><div class="muted">long chats compacted under pressure land here, with facts kept</div></div>
<h3>Decision log</h3><div id="log"></div>
<h3>CachePilot result</h3><pre id="cp">not run yet</pre>
<script>
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
async function refresh(){
  try{
    const r = await fetch('/api/status'); const s = await r.json();
    const st = document.getElementById('state');
    st.textContent = s.state; st.className = 'badge ' + s.state;
    const h = s.health || {level:'', score:0, message:'Waiting for live metrics.', detail:'', warn_at:0.3, danger_at:1, scale:1.15};
    document.querySelectorAll('.mode').forEach(el => {
      el.classList.toggle('on', !!h.level && el.dataset.level === h.level);
    });
    const warn = h.warn_at || 0.3, danger = h.danger_at || 1, scale = h.scale || 1.15;
    const score = h.score || 0;
    let pos;
    if (score <= warn) pos = (score / warn) * 33.3;
    else if (score <= danger) pos = 33.3 + ((score - warn) / (danger - warn)) * 33.3;
    else pos = 66.6 + ((score - danger) / Math.max(0.01, scale - danger)) * 33.4;
    pos = Math.max(0, Math.min(100, pos));
    document.getElementById('healthtrack').style.background =
      'linear-gradient(90deg,#1d7a3a 0 33.3%,#d4a017 33.3% 66.6%,#c53030 66.6% 100%)';
    document.getElementById('healthneedle').style.left = pos + '%';
    const msg = document.getElementById('healthmsg');
    msg.textContent = h.message || '';
    msg.className = h.level || '';
    document.getElementById('healthdetail').textContent = h.detail || '';
    const hw = s.hardware || {};
    const bar = (id, pct) => { document.getElementById(id).style.width = Math.max(0, Math.min(100, pct || 0)) + '%'; };
    if (hw.ok) {
      document.getElementById('hwname').textContent = hw.name + ' · 1 GPU assigned';
      bar('hwmem', hw.mem_pct);
      document.getElementById('hwmemt').textContent = hw.mem_used_gib + ' / ' + hw.mem_total_gib + ' GiB';
      bar('hwcomp', hw.compute_pct);
      document.getElementById('hwcompt').textContent = hw.compute_pct + '%';
      bar('hwpow', hw.power_pct);
      document.getElementById('hwpowt').textContent = hw.power_w + ' / ' + hw.power_limit_w + ' W';
      const kv = (s.latest && s.latest.kv != null) ? (s.latest.kv * 100).toFixed(0) + '% of the KV blocks' : 'KV cache idle';
      document.getElementById('hwnote').textContent = 'Temp ' + hw.temp_c + '°C. ' + kv + '. VRAM stays occupied while the model is loaded; compute and power rise when the card is working.';
    } else {
      document.getElementById('hwname').textContent = hw.error || 'Reading the card…';
      bar('hwmem', 0); bar('hwcomp', 0); bar('hwpow', 0);
      document.getElementById('hwmemt').textContent = '-';
      document.getElementById('hwcompt').textContent = '-';
      document.getElementById('hwpowt').textContent = '-';
      document.getElementById('hwnote').textContent = '';
    }
    const L = s.latest || {};
    const pct = x => x==null ? '-' : (x*100).toFixed(0)+'%';
    document.getElementById('kv').textContent = pct(L.kv);
    document.getElementById('waiting').textContent = L.waiting ?? '-';
    document.getElementById('running').textContent = L.running ?? '-';
    document.getElementById('ttft').textContent = L.ttft_avg==null?'-':L.ttft_avg.toFixed(2);
    document.getElementById('qtime').textContent = L.queue_avg==null?'-':L.queue_avg.toFixed(2);
    document.getElementById('tps').textContent = L.tok_per_s==null?'-':L.tok_per_s.toFixed(0);
    document.getElementById('hitrate').textContent = L.prefix_hit_rate==null?'-':(L.prefix_hit_rate*100).toFixed(1)+'%';
    document.getElementById('bound').textContent = L.bottleneck || '-';
    const govb = document.getElementById('govtoggle');
    govb.textContent = s.governed ? 'Governor on' : 'Governor off';
    govb.classList.toggle('live', !!s.governed);
    document.getElementById('mstate').textContent = s.metrics_ok ? '' : ('metrics error: '+(s.metrics_error||''));
    document.getElementById('log').innerHTML = (s.log||[]).slice().reverse().map(e=>
      '<div><span class="warnline">'+esc(e.t)+'</span> <b>'+esc(e.action)+'</b> &mdash; '+esc(e.detail)+''+(e.say?'<br><span class="say">&ldquo;'+esc(e.say)+'&rdquo;</span>':'')+'</div>').join('');
    const stopb = document.querySelector('.pill.stop');
    if (stopb && !s.flood_on && !s.sequence_running && !s.compare_running) stopb.textContent = 'Stop';
    document.getElementById('activity').innerHTML =
      (s.activity||[]).slice().reverse().map(a=>'<div>['+esc(a.t)+'] '+esc(a.msg)+'</div>').join('')
      + (s.flood_on ? '<div class="cursor">&#9613;</div>' : '');
    document.querySelectorAll('.pill[data-preset]').forEach(b=>{
      b.classList.toggle('active', s.flood_on && s.flood_preset===b.dataset.preset);
    });
    const fprog = document.getElementById('floodprog');
    if (s.flood_on && s.flood_progress){
      fprog.style.display = 'block';
      const fp = s.flood_progress;
      document.getElementById('floodbar').style.width =
        Math.min(100, fp.elapsed / fp.total * 100) + '%';
      const flabel = s.compare_running ? 'A/B compare'
                   : s.sequence_running ? ('sequence \u00b7 ' + (s.sequence_step || '') + ' demand')
                   : ((s.flood_preset || '') + ' demand' + (s.governed ? ' \u00b7 governor on' : ' \u00b7 direct'));
      document.getElementById('floodtime').textContent = flabel + ': ' + fp.remaining + 's remaining';
    } else {
      fprog.style.display = 'none';
    }
    const cmp = s.compare;
    const seq = s.sequence;
    const rep = s.summary;
    const rdiv = document.getElementById('report');
    const lib = s.compaction;
    const ldiv = document.getElementById('librarian');
    if (lib && lib.t){
      const facts = lib.facts_total ? (lib.facts_kept + '/' + lib.facts_total + ' numeric facts kept') : 'no numeric facts in the dropped turns';
      const kvb = lib.kv_before==null ? '-' : lib.kv_before + '%';
      const kva = lib.kv_after==null ? '-' : lib.kv_after + '%';
      ldiv.innerHTML = '<div><b>'+esc(lib.t)+'</b> compacted '+lib.before_turns+' turns down to '+lib.after_turns+'</div>'
        +'<div>KV before <b>'+kvb+'</b> &middot; KV after <b>'+kva+'</b> &middot; '+esc(facts)+'</div>'
        +'<div class="muted">'+esc(lib.summary||'')+'</div>';
    }
    if (cmp && cmp.sides && cmp.sides.length && !s.compare_running){
      const rows = cmp.sides.map(x=>'<tr><td><b>'+(x.governed?'governor on':'governor off')+'</b></td><td>'+x.requests+'</td><td>'+x.errors+'</td><td>'+x.peak_queue+'</td><td>'+x.peak_kv+'%</td><td>'+x.avg_latency+'</td><td>'+x.avg_ttft+'</td><td>'+x.pressure_entries+'</td></tr>').join('');
      rdiv.innerHTML =
        '<div><b>Same '+esc(cmp.preset)+' demand, governor off then on</b></div>'
        +'<table><tr><th>route</th><th>ok</th><th>errors</th><th>peak queue</th><th>peak KV</th><th>avg s</th><th>TTFT</th><th>PRESSURE</th></tr>'+rows+'</table>'
        +(cmp.verdict?'<div>'+esc(cmp.verdict)+'</div>':'')
        +(cmp.narration?'<div><span class="say">&ldquo;'+esc(cmp.narration)+'&rdquo;</span></div>':'');
    } else if (seq && seq.runs && seq.runs.length && !s.sequence_running){
      const rows = seq.runs.map(x=>'<tr><td><b>'+esc(x.preset)+'</b></td><td>'+x.requests+'</td><td>'+x.peak_kv+'%</td><td>'+x.peak_queue+'</td><td>'+x.pressure_entries+'</td><td>'+x.terse_activations+'</td></tr>').join('');
      rdiv.innerHTML =
        '<div><b>Full sequence complete: low &rarr; medium &rarr; high</b></div>'
        +'<table><tr><th>demand</th><th>requests</th><th>peak KV</th><th>peak queue</th><th>PRESSURE</th><th>terse</th></tr>'+rows+'</table>'
        +'<div class="muted">Read it left to right: the engine under rising load, and where the governor stepped in.</div>'
        +(seq.narration?'<div><span class="say">&ldquo;'+esc(seq.narration)+'&rdquo;</span></div>':'');
    } else if (rep && rep.preset){
      rdiv.innerHTML =
        '<div><b>'+esc(rep.preset)+' demand</b> &mdash; '+rep.requests+' ok, '+rep.errors+' errors, '+rep.waves+' launched in '+rep.duration_s+'s'
        +' &middot; '+(rep.governed?'governor on':'direct')+'</div>'
        +'<div>peak KV <b>'+rep.peak_kv+'%</b> &middot; peak queue <b>'+rep.peak_queue+'</b> &middot; peak running <b>'+rep.peak_running+'</b> &middot; avg response <b>'+rep.avg_latency+'</b>s &middot; TTFT <b>'+rep.avg_ttft+'</b>s &middot; queue time <b>'+rep.avg_queue+'</b>s</div>'
        +'<div>PRESSURE entries: <b>'+rep.pressure_entries+'</b> (~'+Math.round(rep.pressure_seconds)+'s) &middot; terse activations: <b>'+rep.terse_activations+'</b></div>'
        +'<ul>'+rep.suggestions.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul>'
        +(rep.narration?'<div><span class="say">&ldquo;'+esc(rep.narration)+'&rdquo;</span></div>':'');
    }
  }catch(e){}
}
async function startFlood(p){
  await fetch('/api/flood/start?preset='+p,{method:'POST'});
}
async function stopFlood(){
  const b = document.querySelector('.pill.stop');
  if (b) b.textContent = 'Stopping\u2026';
  await fetch('/api/flood/stop',{method:'POST'});
}
async function toggleGoverned(){
  const on = document.getElementById('govtoggle').textContent.indexOf('off') >= 0;
  await fetch('/api/governed?on='+(on?'1':'0'),{method:'POST'});
}
async function startCompare(){
  await fetch('/api/compare/start?preset=high',{method:'POST'});
}
async function startSequence(){
  await fetch('/api/sequence/start',{method:'POST'});
}
async function cachepilot(){
  document.getElementById('cp').textContent = 'running...';
  const r = await fetch('/api/cachepilot/run',{method:'POST'});
  const j = await r.json();
  document.getElementById('cp').textContent = j.text || JSON.stringify(j);
}
setInterval(refresh, 1000); refresh();
</script></body></html>"""


class DashHandler(BaseHTTPRequestHandler):
    gov = None

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(DASH_HTML.encode(), "text/html")
        elif self.path == "/api/status":
            g = self.gov
            with g.lock:
                fp = None
                if g.flood_on and g.flood_stats:
                    el = time.time() - g.flood_stats["started"]
                    tot = g.flood_stats["duration"]
                    fp = {"elapsed": round(el, 1), "total": tot,
                          "remaining": max(0, int(tot - el))}
                payload = {"state": g.state, "latest": g.latest,
                           "policy": g.policy, "log": g.log, "activity": g.activity,
                           "metrics_ok": g.metrics_ok, "metrics_error": g.metrics_error,
                           "health": g.health, "governed": g.governed,
                           "hardware": g.hardware,
                           "compaction": g.last_compaction,
                           "flood_on": g.flood_on, "flood_preset": g.flood_preset,
                           "flood_progress": fp, "summary": g.last_summary,
                           "sequence_running": g.sequence_running,
                           "sequence_step": g.sequence_step,
                           "sequence": g.last_sequence,
                           "compare_running": g.compare_running,
                           "compare": g.last_compare}
            self._send(json.dumps(payload).encode(), "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        g = self.gov
        if self.path.startswith("/api/flood/start"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            preset = qs.get("preset", ["medium"])[0]
            workers, mtok, dur = FLOOD_PRESETS.get(preset, FLOOD_PRESETS["medium"])
            if not g.flood_on and not g.compare_running:
                g.last_sequence = None
                g.last_compare = None
                g.flood_preset = preset
                g.flood_stats = fresh_stats(preset, workers, mtok, dur, g.governed)
                g.flood_on = True
                route = "proxy" if g.governed else "direct"
                g._activity(f"flood started: {preset} demand ({workers} in flight, {route})")
                threading.Thread(target=flood_worker, args=(g, workers, mtok, dur),
                                 daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        elif self.path == "/api/flood/stop":
            running_job = g.sequence_running or g.compare_running
            g.sequence_abort = True
            g.flood_on = False
            closed = g.abort_live()
            g._activity(f"stop: flood off, closed {closed} live request(s)")
            if running_job:
                g._activity("stop: winding down the sequence or compare")
            else:
                threading.Thread(target=_summarize_after_stop, args=(g,), daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/api/governed"):
            if g.flood_on or g.compare_running or g.sequence_running:
                self._send(b'{"ok":false}', "application/json")
            else:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                g.governed = qs.get("on", ["1"])[0] != "0"
                g._activity("governor routing " + ("on" if g.governed else "off"))
                self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/api/compare/start"):
            if not g.compare_running and not g.flood_on and not g.sequence_running:
                qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                preset = qs.get("preset", ["high"])[0]
                if preset not in FLOOD_PRESETS:
                    preset = "high"
                g.last_compare = None
                g.sequence_abort = False
                threading.Thread(target=compare_worker, args=(g, preset), daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        elif self.path == "/api/sequence/start":
            if not g.sequence_running and not g.flood_on and not g.compare_running:
                g.sequence_abort = False
                g.last_sequence = None
                threading.Thread(target=sequence_worker, args=(g,), daemon=True).start()
            self._send(b'{"ok":true}', "application/json")
        elif self.path == "/api/cachepilot/run":
            here = os.path.dirname(os.path.abspath(__file__))
            try:
                p = subprocess.run([sys.executable, os.path.join(here, "cachepilot.py"), "--json"],
                                   capture_output=True, text=True, timeout=300)
                out = p.stdout.strip() or p.stderr.strip()
                try:
                    self._send(json.dumps({"text": json.loads(out).get("text", out)}).encode(),
                               "application/json")
                except Exception:
                    self._send(json.dumps({"text": out[-4000:]}).encode(), "application/json")
            except Exception as e:
                self._send(json.dumps({"text": f"cachepilot failed: {e}"}).encode(),
                           "application/json")
        else:
            self._send(b"not found", "text/plain", 404)

    def log_message(self, *a):
        pass


# ----------------------------------------------------------------------------
# Optional OpenAI-compatible proxy (applies the active policy)
# ----------------------------------------------------------------------------
def librarian_compact(gov, msgs):
    """Summarize older turns and record how many numeric facts survived."""
    if len(msgs) <= 8:
        return msgs, None
    head = [m for m in msgs if m.get("role") == "system"]
    rest = [m for m in msgs if m.get("role") != "system"]
    if len(rest) <= 6:
        return msgs, None
    older, recent = rest[:-4], rest[-4:]
    older_text = "\n".join(
        f"{m.get('role', 'user')}: {m.get('content', '')}" for m in older)[:4000]
    with gov.lock:
        kv_before = (gov.latest or {}).get("kv")
    summary = ""
    try:
        r = requests.post(VLLM_BASE + "/v1/chat/completions", timeout=25, json={
            "model": MODEL,
            "messages": [{"role": "user", "content":
                          "Summarize the conversation below in under 80 words. "
                          "Keep every number, date, and dollar amount exactly.\n\n"
                          + older_text}],
            "max_tokens": 120, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}})
        summary = r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        summary = ""
    facts, kept = retained_facts(older_text, summary)
    if not summary:
        summary = "[governor compacted older history; the summary call did not return]"
        kept = 0
    event = {
        "t": datetime.now().strftime("%H:%M:%S"),
        "before_turns": len(rest),
        "after_turns": len(recent) + (1 if summary else 0),
        "facts_total": len(facts),
        "facts_kept": kept,
        "kv_before": None if kv_before is None else round(kv_before * 100),
        "kv_after": None,
        "summary": summary[:400],
    }
    note = {"role": "system", "content": "[governor memory] " + summary}
    return head + [note] + recent, event


def apply_terse(gov, body):
    body = dict(body)
    if body.get("max_tokens"):
        body["max_tokens"] = min(int(body["max_tokens"]), 128)
    else:
        body["max_tokens"] = 128
    kwargs = dict(body.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = False
    body["chat_template_kwargs"] = kwargs
    msgs = list(body.get("messages", []))
    msgs, event = librarian_compact(gov, msgs)
    if event is not None:
        with gov.lock:
            gov.last_compaction = event
    sys_msg = {"role": "system",
               "content": "[GOVERNOR TERSE MODE] Answer in 3 sentences or fewer. Be direct."}
    body["messages"] = [sys_msg] + msgs
    return body


class ProxyHandler(BaseHTTPRequestHandler):
    gov = None
    protocol_version = "HTTP/1.1"

    def _send_bytes(self, body, ctype, code=200):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _note_kv_after(self):
        comp = self.gov.last_compaction
        if not comp or comp.get("kv_after") is not None:
            return
        with self.gov.lock:
            kv = (self.gov.latest or {}).get("kv")
            if kv is not None and self.gov.last_compaction is comp:
                comp["kv_after"] = round(kv * 100)

    def _proxy(self, method):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        is_flood = self.headers.get("X-Governor-Flood") == "1"
        with self.gov.lock:
            terse = self.gov.policy.get("terse")
        if method == "POST" and terse and isinstance(body, dict):
            body = apply_terse(self.gov, body)
        if is_flood and isinstance(body, dict):
            body = dict(body)
            body["stream"] = True
        stream = bool(isinstance(body, dict) and body.get("stream")) and method == "POST" and not is_flood
        try:
            if method == "GET":
                r = requests.get(VLLM_BASE + self.path, timeout=30)
            else:
                r = requests.post(VLLM_BASE + self.path, json=body, timeout=300, stream=stream or is_flood)
                if (r.status_code >= 400 and isinstance(body, dict)
                        and body.get("chat_template_kwargs")):
                    fallback = dict(body)
                    fallback.pop("chat_template_kwargs", None)
                    r.close()
                    r = requests.post(VLLM_BASE + self.path, json=fallback, timeout=300,
                                      stream=stream or is_flood)
        except Exception as e:
            self._send_bytes(json.dumps({"error": str(e)}), "application/json", 502)
            return
        if is_flood:
            self.gov.track_response(r)
            stopped = False
            try:
                for _chunk in r.iter_content(chunk_size=4096):
                    if not self.gov.flood_on:
                        stopped = True
                        break
            except Exception:
                stopped = not self.gov.flood_on
            finally:
                self.gov.untrack_response(r)
                _close_response(r)
            try:
                if stopped or not self.gov.flood_on:
                    self._send_bytes(b'{"error":"stopped"}', "application/json", 499)
                else:
                    code = 200 if r.status_code < 400 else r.status_code
                    self._send_bytes(b'{"ok":true}', "application/json", code)
            except Exception:
                pass
            return
        if stream:
            self.send_response(r.status_code)
            self.send_header("Content-Type", r.headers.get("Content-Type", "text/event-stream"))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for chunk in r.iter_content(chunk_size=None):
                    if not chunk:
                        continue
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except Exception:
                pass
            finally:
                r.close()
                self._note_kv_after()
            return
        self._note_kv_after()
        self._send_bytes(r.content, "application/json", r.status_code)

    def do_GET(self):
        self._proxy("GET")

    def do_POST(self):
        self._proxy("POST")

    def log_message(self, *a):
        pass


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def _gpu_num(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


def discover_vllm_pod():
    proc = subprocess.run(
        ["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE,
         "get", "pods", "-l", "app=vllm",
         "--field-selector=status.phase=Running",
         "-o", "jsonpath={.items[0].metadata.name}"],
        capture_output=True, text=True, timeout=15)
    name = (proc.stdout or "").strip()
    if proc.returncode != 0 or not name:
        raise RuntimeError((proc.stderr or "no running vllm pod").strip()[:200])
    return name


def read_gpu(pod):
    proc = subprocess.run(
        ["kubectl", "--kubeconfig", KUBECONFIG, "-n", NAMESPACE,
         "exec", pod, "-c", "vllm", "--",
         "nvidia-smi",
         "--query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "nvidia-smi failed").strip()[:200])
    row = next(csv.reader([proc.stdout.strip()]))
    if len(row) < 7:
        raise RuntimeError("unexpected nvidia-smi row")
    name, used, total, util, power, limit, temp = [part.strip() for part in row[:7]]
    used_mib, total_mib = _gpu_num(used), _gpu_num(total)
    power_w, limit_w = _gpu_num(power), _gpu_num(limit)
    util_pct, temp_c = _gpu_num(util), _gpu_num(temp)
    if not total_mib:
        raise RuntimeError("GPU memory total missing")
    return {
        "ok": True,
        "name": name,
        "mem_used_gib": round(used_mib / 1024, 1),
        "mem_total_gib": round(total_mib / 1024, 1),
        "mem_pct": round(used_mib / total_mib * 100, 1),
        "compute_pct": round(util_pct or 0, 1),
        "power_w": round(power_w or 0, 1),
        "power_limit_w": round(limit_w or 0, 1),
        "power_pct": round((power_w or 0) / limit_w * 100, 1) if limit_w else 0,
        "temp_c": round(temp_c or 0),
    }


def gpu_poll_loop(gov):
    pod = None
    while True:
        try:
            if not pod:
                pod = discover_vllm_pod()
            sample = read_gpu(pod)
        except Exception as e:
            pod = None
            sample = {"ok": False, "error": str(e)[:200]}
        with gov.lock:
            gov.hardware = sample
        time.sleep(4)


def main():
    ap = argparse.ArgumentParser(description="Autopilot Governor")
    ap.add_argument("--dashboard-port", type=int, default=8081)
    ap.add_argument("--proxy-port", type=int, default=8082)
    ap.add_argument("--allow-restart", action="store_true",
                    help="enable the kubectl rollout restart closer")
    args = ap.parse_args()

    gov = Governor(allow_restart=args.allow_restart)
    gov.proxy_base = f"http://127.0.0.1:{args.proxy_port}"
    threading.Thread(target=gpu_poll_loop, args=(gov,), daemon=True).start()
    DashHandler.gov = gov
    ProxyHandler.gov = gov

    dash = ThreadingHTTPServer(("127.0.0.1", args.dashboard_port), DashHandler)
    threading.Thread(target=dash.serve_forever, daemon=True).start()
    proxy = ThreadingHTTPServer(("127.0.0.1", args.proxy_port), ProxyHandler)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()

    print(f"dashboard : http://localhost:{args.dashboard_port}", flush=True)
    print(f"proxy     : http://localhost:{args.proxy_port}/v1  (model {MODEL})", flush=True)
    print(f"watching  : {VLLM_BASE}   namespace {NAMESPACE}   restart actuator: "
          f"{'ON' if args.allow_restart else 'off'}", flush=True)
    try:
        while True:
            gov.tick()
            time.sleep(POLL_SECS)
    except KeyboardInterrupt:
        print("\ngovernor stopped")


if __name__ == "__main__":
    main()
