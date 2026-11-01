"""Comprehensive evaluation and latency benchmarking script for RIO."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.engine import RIOEngine
from ml.threads import load_conversation_threads

ROOT = Path(__file__).resolve().parent.parent


def benchmark_engine(
    sample_size: int = 100,
    thread_csv: Path | str | None = None,
) -> dict[str, Any]:
    """Run end-to-end performance and latency benchmarks."""
    print("Initializing RIO Engine...")
    engine = RIOEngine()
    engine.load_models()

    # 1. Phishing & Intent Real-Time Latency Benchmark
    test_emails = [
        ("Account Review Required", "Please sign in to verify your account immediately at http://secure.example.com", "Security Desk <alert@domain.com>"),
        ("Project Meeting", "Could we schedule a planning meeting next Tuesday?", "Alex <alex@company.com>"),
        ("Invoice Payment", "Please confirm that the invoice payment was received.", "Billing <finance@supplier.org>"),
        ("Document Access", "Please grant access to the research folder before 5pm.", "Dev <dev@corp.internal>"),
    ]

    latencies_ms: list[float] = []
    print(f"Benchmarking real-time inference latency across {sample_size} runs...")
    for i in range(sample_size):
        subject, body, sender = test_emails[i % len(test_emails)]
        t0 = time.perf_counter()
        res = engine.analyze_raw_email(subject=subject, body=body, sender=sender)
        latencies_ms.append(res["execution_time_ms"])

    latency_stats = {
        "mean_ms": float(np.mean(latencies_ms)),
        "median_ms": float(np.median(latencies_ms)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
    }
    print(f"Latency: Mean={latency_stats['mean_ms']:.2f}ms, P95={latency_stats['p95_ms']:.2f}ms")

    # 2. Conversation Thread Dataset Inspection
    thread_stats: dict[str, Any] = {"status": "skipped"}
    target_csv = Path(thread_csv) if thread_csv else (ROOT / "data/raw/email_thread_details.csv")
    if not target_csv.exists():
        dl_fallback = Path("C:/Users/LENOVO/Downloads/email_thread_details.csv")
        if dl_fallback.exists():
            target_csv = dl_fallback

    if target_csv.exists():
        print(f"Loading conversation thread dataset from {target_csv}...")
        threads = load_conversation_threads(target_csv, max_threads=100, min_turns=2)
        turn_counts = [t.turn_count for t in threads.values()]
        thread_stats = {
            "status": "loaded",
            "threads_sampled": len(threads),
            "total_turns": sum(turn_counts),
            "mean_turns_per_thread": float(np.mean(turn_counts)) if turn_counts else 0.0,
            "max_turns_in_thread": int(np.max(turn_counts)) if turn_counts else 0,
        }
        print(f"Threads loaded: {len(threads)}, Total turns: {thread_stats['total_turns']}")

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "realtime_latency": latency_stats,
        "threads_summary": thread_stats,
    }

    report_path = ROOT / "reports/evaluation_summary.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark summary written to {report_path}")
    return summary


if __name__ == "__main__":
    benchmark_engine()
