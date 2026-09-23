"""Run independent reviews from one frozen job; --dry-run stays offline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

from review_protocol import PROMPT, SCHEMA
from review_runner_appserver import (BASE_INSTRUCTIONS, EFFORT, EXE, MODEL,
    input_metadata, load_job, read_rate_limits, run_review, save)


def signature(items):
    raw = json.dumps(input_metadata(items), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_quota(limits):
    if limits.get("ordinaryUsageAllowed") is False:
        raise RuntimeError("Account allowance exhausted; no review was started")
    # Prefer the named Codex bucket; other buckets may belong to other models.
    buckets = limits.get("rateLimitsByLimitId") or {}
    bucket = buckets.get("codex") or limits.get("rateLimits")
    observed_window = False
    if bucket:
        for name in ("primary", "secondary"):
            value = (bucket.get(name) or {}).get("usedPercent")
            observed_window = observed_window or value is not None
            if value is not None and value >= 100:
                raise RuntimeError("Account allowance exhausted; no review was started")
    if not observed_window and limits.get("ordinaryUsageAllowed") is not True:
        raise RuntimeError("Account allowance unavailable; no review was started")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--executable", type=Path, default=EXE)
    parser.add_argument("--rounds", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    job = load_job(args.job.resolve())
    if (job.get("model") != MODEL or job.get("reasoning") != EFFORT
            or job.get("system_prompt") != BASE_INSTRUCTIONS
            or job.get("developer_prompt") != BASE_INSTRUCTIONS
            or job["outputSchema"] != SCHEMA
            or job["input"][0] != {"type": "text", "text": PROMPT}):
        raise ValueError("Job does not match the frozen historical review protocol")
    frozen = signature(job["input"])
    if args.dry_run:
        print(json.dumps({"passed": True, "input_signature": frozen,
                          "items": len(job["input"]), "rounds": args.rounds, "model_calls": 0}))
        return 0
    if not args.output:
        parser.error("--output is required for real reviews")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    save(out / "input_freeze.json", {"signature": frozen, "rounds": args.rounds,
                                     "paper_id": job.get("paper_id"), "model": MODEL, "reasoning": EFFORT})
    threads, scores, reviews = set(), [], []
    try:
        for number in range(1, args.rounds + 1):
            name = f"round_{number:02d}"
            if signature(job["input"]) != frozen:
                raise RuntimeError("Input images changed after the run was frozen")
            quota = read_rate_limits(out, args.executable)
            save(out / f"quota_before_{name}.json", quota)
            check_quota(quota)
            metrics = run_review(job["input"], SCHEMA, out / name, run_id=name,
                                 paper_id=job.get("paper_id", "paper"), executable=args.executable)
            if metrics["status"] != "completed" or metrics.get("tool_call_count") != 0:
                raise RuntimeError(f"{name} failed or was quarantined; inspect its metrics; no automatic retry")
            if metrics["thread_id"] in threads:
                raise RuntimeError("Repeated thread ID; review independence failed")
            threads.add(metrics["thread_id"])
            response = json.loads((out / name / "final.json").read_text(encoding="utf-8"))
            scores.append(response["score"])
            reviews.append({"round": number, "response": response})
        result = {"status": "completed", "scores": scores, "mean_score": statistics.mean(scores),
                  "sample_standard_deviation": statistics.stdev(scores) if len(scores) > 1 else None,
                  "independent_threads": len(threads), "reviews": reviews}
        save(out / "summary.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "reviews"}))
        return 0
    except Exception as error:
        save(out / "status.json", {"status": "stopped", "completed_rounds": len(reviews), "error": str(error)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
