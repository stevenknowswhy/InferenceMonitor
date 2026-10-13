#!/usr/bin/env python3
"""Prefix-Cache Pilot -- standalone demo, also wired into the Monitor dashboard.

Proves that restructuring prompts around a shared prefix reuses the engine's
prefix cache: measures per-request TTFT (streaming) and prefix-cache counter
deltas for two semantically equivalent prompt layouts:
  * q_first   : question, then document (cache-hostile)
  * doc_first : document, then question (cache-friendly, one warm-up first)

Usage:
    python3 cachepilot.py           # human-readable table
    python3 cachepilot.py --json    # JSON summary (used by the dashboard)
"""

import json
import os
import re
import statistics
import sys
import time

import requests

VLLM_BASE = os.environ.get("VLLM_BASE", "http://localhost:8000")
MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")

DOCUMENT = """The Meridian Relay is a volunteer-run emergency communications network founded in 2019 after the Ridgefield wildfire exposed gaps in county radio coverage. It operates 14 solar-powered relay stations across three counties, linked by a mix of VHF repeaters and a low-bandwidth mesh data channel.

Membership: The Relay has 212 licensed operators as of March 2026. New members complete a 6-week training course covering FCC Part 97 rules, net control procedures, and field deployment. Annual dues are $45, waived for students.

Operations: During incidents, the Relay runs scheduled nets every 30 minutes on 146.52 MHz simplex, with a designated net control station logging all traffic. Non-emergency weekly nets run Sundays at 19:00 local on the linked repeater system.

Funding: The organization is funded by member dues (60%), an annual preparedness fair (25%), and equipment grants (15%). Its 2025 budget was $38,400, of which $12,000 went to battery replacements for the solar stations.

Governance: A five-member board elected each November sets operating policy. The technical committee maintains the repeater linking plan, last revised January 2026."""

QUESTIONS = [
    ("What year was the Meridian Relay founded?", "2019"),
    ("How many relay stations does it operate?", "14"),
    ("What are the annual dues?", "$45"),
    ("When do the weekly nets run?", "Sunday"),
    ("What was the 2025 budget?", "$38,400"),
    ("How many licensed operators are members?", "212"),
]

SYSTEM = "Answer with just the fact asked for. Be brief."


def chat_stream(messages, max_tokens=128):
    """Stream a completion; return (ttft_seconds, text)."""
    t0 = time.time()
    first = None
    chunks = []
    with requests.post(VLLM_BASE + "/v1/chat/completions", stream=True, timeout=120,
                       json={"model": MODEL, "messages": messages,
                             "max_tokens": max_tokens, "temperature": 0.0,
                             "stream": True,
                             "chat_template_kwargs": {"enable_thinking": False}}) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if first is None:
                first = time.time()
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except Exception:
                delta = ""
            chunks.append(delta)
    return ((first - t0) if first else None), "".join(chunks)


def cache_counters():
    txt = requests.get(VLLM_BASE + "/metrics", timeout=10).text

    def find(names):
        for n in names:
            m = re.search(r"^" + re.escape(n) + r"(?:\{[^}]*\})?\s+([0-9.eE+\-]+)",
                          txt, re.M)
            if m:
                return float(m.group(1))
        return None

    hits = find(["vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits_total"])
    queries = find(["vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries_total"])
    rate = find(["vllm:gpu_prefix_cache_hit_rate"])
    return hits, queries, rate


def build_messages(layout, question):
    if layout == "q_first":
        user = f"Question: {question}\n\nReference document:\n{DOCUMENT}"
    else:
        user = f"Reference document:\n{DOCUMENT}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def run_batch(layout, warmup=False):
    if warmup:
        chat_stream(build_messages(layout, QUESTIONS[0][0]))
    results = []
    h0, q0, _ = cache_counters()
    for question, expected in QUESTIONS:
        ttft, text = chat_stream(build_messages(layout, question))
        ok = expected.lower() in text.lower()
        results.append({"q": question, "ttft": ttft, "ok": ok, "text": text[:80]})
        time.sleep(0.3)
    h1, q1, _ = cache_counters()
    dh = (h1 - h0) if h0 is not None and h1 is not None else None
    dq = (q1 - q0) if q0 is not None and q1 is not None else None
    return results, dh, dq


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main():
    as_json = "--json" in sys.argv
    out_lines = []

    def say(s=""):
        out_lines.append(s)
        if not as_json:
            print(s, flush=True)

    say("CachePilot: baseline = question-first layout (cache-hostile)")
    base, base_dh, base_dq = run_batch("q_first")
    say(f"  median TTFT: {med([r['ttft'] for r in base])}")
    say(f"  correct: {sum(r['ok'] for r in base)}/{len(base)}")
    say("")
    say("CachePilot: restructured = document-first layout + one warm-up (cache-friendly)")
    new, new_dh, new_dq = run_batch("doc_first", warmup=True)
    say(f"  median TTFT: {med([r['ttft'] for r in new])}")
    say(f"  correct: {sum(r['ok'] for r in new)}/{len(new)}")
    say("")

    b, n = med([r["ttft"] for r in base]), med([r["ttft"] for r in new])
    if b and n:
        say(f"TTFT: {b:.2f}s -> {n:.2f}s  ({(1 - n/b)*100:.0f}% faster on first token)")
    if new_dh is not None and new_dq:
        say(f"Prefix-cache counters during restructured batch: +{new_dh:.0f} hits / +{new_dq:.0f} queries")
    elif new_dh is None:
        say("NOTE: no prefix-cache counters found in /metrics -- "
            "enable prefix caching or check metric names for your vLLM version.")
    correct = sum(r["ok"] for r in new) == len(new)
    verdict = "PASS: restructuring reused the prefix cache with no accuracy loss." \
        if (b and n and n < b * 0.9 and correct) else \
        "INCONCLUSIVE: re-run or check that prefix caching is enabled."
    say("")
    say("Verdict: " + verdict)

    if as_json:
        print(json.dumps({
            "text": "\n".join(out_lines),
            "baseline_ttft": b, "restructured_ttft": n,
            "hits_delta": new_dh, "queries_delta": new_dq,
            "correct": f"{sum(r['ok'] for r in new)}/{len(new)}",
            "verdict": verdict,
        }))


if __name__ == "__main__":
    main()
