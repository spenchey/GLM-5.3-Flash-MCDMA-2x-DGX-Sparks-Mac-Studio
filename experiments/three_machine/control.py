"""Orderly control messages for the live three-machine inference stream."""

from __future__ import annotations

import argparse
import json
import secrets
import time

from experiments.three_machine import protocol
from experiments.three_machine.mailbox import ClientMailbox


def _exchange(mailbox: ClientMailbox, frame: protocol.Frame, timeout: float) -> protocol.Frame:
    answer = protocol.unpack(mailbox.call(protocol.pack(frame), timeout), now_ns=time.time_ns())
    if answer.kind is protocol.Kind.ERROR:
        raise RuntimeError(answer.payload.decode("utf-8", errors="replace"))
    if (answer.kind, answer.request_id, answer.step, answer.committed, answer.generation) != (
            protocol.Kind.ACK, frame.request_id, frame.step, frame.committed, frame.generation):
        raise protocol.ProtocolError("control acknowledgement identity differs")
    return answer


def run(mailbox_name: str, action: str, timeout: float) -> dict:
    request_id = secrets.randbits(63) or 1
    with ClientMailbox(mailbox_name, ready_timeout_s=timeout) as mailbox:
        reset = protocol.control(kind=protocol.Kind.RESET, request_id=request_id, step=0, committed=0,
                                 generation=mailbox.generation, timeout_s=timeout)
        reset_answer = _exchange(mailbox, reset, timeout)
        if (reset_answer.step, reset_answer.committed, reset_answer.generation) != (0, 0, mailbox.generation):
            raise protocol.ProtocolError("reset did not return a clean position")
        if action == "shutdown":
            stop = protocol.control(kind=protocol.Kind.SHUTDOWN, request_id=request_id, step=0, committed=0,
                                    generation=mailbox.generation, timeout_s=timeout)
            answer = _exchange(mailbox, stop, timeout)
        else:
            answer = reset_answer
    return {"action": action, "generation": answer.generation, "step": answer.step,
            "committed": answer.committed, "acknowledged": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reset", "shutdown"))
    parser.add_argument("--mailbox", default="tfglm53")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    print(json.dumps(run(args.mailbox, args.action, args.timeout), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
