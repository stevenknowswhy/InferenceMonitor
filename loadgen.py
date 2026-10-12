#!/usr/bin/env python3
"""Flood the engine with concurrent long-prompt requests (demo pressure).

Usage:
    python3 loadgen.py --concurrency 12 --duration 120
"""

import argparse
import os
import threading
import time

import requests

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")

PROMPTS = [
    "Write a detailed 400-word essay on the history of container orchestration, covering Borg, Omega, Kubernetes, and Nomad with dates and key design decisions.",
    "Explain in depth how the transformer attention mechanism works, including Q/K/V projections, softmax scaling, multi-head attention, and KV caching, with equations.",
    "Give a comprehensive overview of GPU memory hierarchies: registers, shared memory, L1/L2 cache, HBM, and how they affect LLM inference throughput.",
    "Describe the full lifecycle of a Kubernetes pod from scheduling to termination, including probes, init containers, and graceful shutdown. Be thorough.",
]


def hit(prompt):
    try:
        requests.post(VLLM_BASE + "/v1/chat/completions", timeout=180, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300, "temperature": 0.7})
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--duration", type=int, default=90,
                    help="seconds to keep flooding")
    args = ap.parse_args()
    print(f"flooding {VLLM_BASE} with {args.concurrency} concurrent requests "
          f"for {args.duration}s ...", flush=True)
    end = time.time() + args.duration
    wave = 0
    while time.time() < end:
        ts = []
        for i in range(args.concurrency):
            t = threading.Thread(target=hit, args=(PROMPTS[(wave + i) % len(PROMPTS)],),
                                 daemon=True)
            t.start()
            ts.append(t)
        for t in ts:
            t.join(timeout=120)
        wave += 1
        time.sleep(1)
    print("flood done", flush=True)


if __name__ == "__main__":
    main()
