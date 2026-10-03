"""Command line interface for the offline experiment evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import ManifestError, evaluate, load_json, validate_trial, write_receipt
from .paired import (
    evaluate_campaign,
    evaluate_suite,
    validate_campaign,
    validate_confirmation,
    write_campaign_receipt,
)


EXIT_CONFIRMATION_REQUIRED = 5


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate or evaluate one offline trial",
        epilog=(
            "Exit 0 means validated or finally accepted; 2 rejected; 3 invalid; "
            "4 needs the one screen extension; 5 passed screening and requires confirmation."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--baseline", type=Path, required=True)
        command.add_argument("--trial", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--receipts", type=Path, required=True)
    for name in ("paired-validate", "paired-evaluate", "paired-confirm"):
        command = sub.add_parser(name)
        command.add_argument("--campaign", type=Path, required=True)
        if name != "paired-validate":
            command.add_argument("--receipts", type=Path, required=True)
        if name == "paired-confirm":
            command.add_argument("--screen-receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command.startswith("paired-"):
            campaign = load_json(args.campaign)
            if args.command == "paired-confirm":
                screen_receipt = load_json(args.screen_receipt)
                validate_confirmation(campaign, screen_receipt)
                result = evaluate_suite(campaign, screen_receipt)
                path = write_campaign_receipt(args.receipts, result)
                print(json.dumps({"receipt": str(path), **result}, sort_keys=True))
                return 0 if result["verdict"] == "keep" else 2
            validate_campaign(campaign)
            if args.command == "paired-validate":
                print("VALID")
                return 0
            result = evaluate_campaign(campaign)
            path = write_campaign_receipt(args.receipts, result)
            print(json.dumps({"receipt": str(path), **result}, sort_keys=True))
            # A passed screen is intentionally nonzero: only paired-confirm
            # may return zero for a finally accepted optimization.
            if result["verdict"] == "screen_pass":
                return EXIT_CONFIRMATION_REQUIRED
            return 4 if result["verdict"] == "extend_once" else 2
        baseline = load_json(args.baseline)
        trial = load_json(args.trial)
        validate_trial(trial, baseline)
        if args.command == "validate":
            print("VALID")
            return 0
        result = evaluate(baseline, trial)
        if result["verdict"] == "keep":
            result = {
                **result,
                "final_acceptance": False,
                "confirmation_required": True,
            }
        path = write_receipt(args.receipts, result)
        print(json.dumps({"receipt": str(path), **result}, sort_keys=True))
        return EXIT_CONFIRMATION_REQUIRED if result["verdict"] == "keep" else 2
    except ManifestError as exc:
        print(f"INVALID: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
