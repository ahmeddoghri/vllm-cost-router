"""Reporting cost and routing quality side by side.

Cost reduction on its own is not a result. Routing every request to the cheap
tier cuts spend 50% and is obviously wrong, and no cost graph will tell you so.
The pairing that means something is savings alongside the misroute rate: the
fraction of genuinely hard requests that got sent to the small model.

    python -m app.routereport
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Dict

from . import routebench as rb
from .router import complexity_score as v1_score
from .router_v2 import complexity_score as v2_score

#: Each router with the threshold it is actually operated at. The two scorers
#: are on different scales, so comparing them at one shared threshold would
#: measure the scales rather than the routers.
ROUTERS: Dict[str, tuple[Callable[[str], float], float]] = {
    "keyword (v1)": (lambda p: v1_score(p, 256), 0.50),
    "structural (v2)": (lambda p: v2_score(p, 256), 0.10),
}


def always_large(_: str) -> float:
    return 1.0


def always_small(_: str) -> float:
    return 0.0


def build_report() -> dict:
    """Score every router on every split."""
    report: dict = {"routers": {}}
    for name, (scorer, threshold) in ROUTERS.items():
        report["routers"][name] = {
            "threshold": threshold,
            "splits": rb.evaluate(scorer, threshold),
        }

    # Degenerate policies, as reference points. Always-large never misroutes
    # and never saves; always-small saves the most and misroutes everything.
    report["baselines"] = {
        "always_large": rb.evaluate(always_large, 0.5)["overall"],
        "always_small": rb.evaluate(always_small, 0.5)["overall"],
    }
    return report


def format_report(report: dict) -> str:
    lines = [
        "Routing quality on the labeled corpus",
        "=" * 78,
        "",
        f"{'router':<18}{'split':<14}{'acc':>7}{'F1':>7}{'misroute':>11}{'overspend':>11}",
        "-" * 78,
    ]
    for name, data in report["routers"].items():
        for split_name in ("easy", "adversarial", "holdout", "overall"):
            row = data["splits"][split_name]
            label = name if split_name == "easy" else ""
            lines.append(
                f"{label:<18}{split_name:<14}"
                f"{row['accuracy']:>7.3f}{row['f1']:>7.3f}"
                f"{row['misroute_rate']:>11.1%}{row['overspend_rate']:>11.1%}"
            )
        lines.append("")

    baselines = report["baselines"]
    lines += [
        "-" * 78,
        f"{'always-large':<18}{'overall':<14}"
        f"{baselines['always_large']['accuracy']:>7.3f}"
        f"{baselines['always_large']['f1']:>7.3f}"
        f"{baselines['always_large']['misroute_rate']:>11.1%}"
        f"{baselines['always_large']['overspend_rate']:>11.1%}",
        f"{'always-small':<18}{'overall':<14}"
        f"{baselines['always_small']['accuracy']:>7.3f}"
        f"{baselines['always_small']['f1']:>7.3f}"
        f"{baselines['always_small']['misroute_rate']:>11.1%}"
        f"{baselines['always_small']['overspend_rate']:>11.1%}",
        "",
        "misroute  = genuinely complex requests sent to the cheap tier (quality loss)",
        "overspend = simple requests sent to the expensive tier (wasted money)",
        "",
        "The holdout split was written after the v2 weights were frozen and has never",
        "been tuned against. It is the only out-of-sample row here.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="also write raw results here")
    parser.add_argument(
        "--show-misroutes", action="store_true",
        help="print the complex prompts each router sends to the cheap tier",
    )
    args = parser.parse_args()

    report = build_report()
    print(format_report(report))

    if args.show_misroutes:
        for name, (scorer, threshold) in ROUTERS.items():
            misses = rb.misrouted_examples(scorer, threshold, limit=4)
            print(f"\n{name} misroutes ({len(misses)} shown):")
            for prompt, score in misses:
                print(f"  {score:.3f}  {prompt[:68]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
