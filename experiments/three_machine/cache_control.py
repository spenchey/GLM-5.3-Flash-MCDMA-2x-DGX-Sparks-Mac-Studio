"""Orderly control for the Spark-prefill to Mac-decode cache service."""

from __future__ import annotations

import argparse
import json
from typing import Any

from experiments.three_machine.cache_handoff import Request, encode_request, unpack_ack
from experiments.three_machine.mailbox import ClientMailbox


def shutdown(mailbox: Any, timeout: float) -> dict[str, object]:
    """Ask both Spark ranks to exit through the verified MCDMA mailbox."""

    answer = mailbox.call(encode_request(Request(operation="shutdown")), timeout)
    transfer_id = unpack_ack(answer, "shutdown")
    if transfer_id is not None:
        raise RuntimeError("shutdown acknowledgement unexpectedly named a cache transfer")
    return {"action": "shutdown", "acknowledged": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shutdown", choices=("shutdown",))
    parser.add_argument("--mailbox", default="p06cfr")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    with ClientMailbox(args.mailbox, ready_timeout_s=args.timeout) as mailbox:
        result = shutdown(mailbox, args.timeout)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
