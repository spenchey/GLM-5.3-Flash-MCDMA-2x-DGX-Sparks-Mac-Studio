"""One bounded real-link proof for the TensorFold/MCDMA frame and mailbox code."""

from __future__ import annotations

import argparse
import hashlib
import json
import time

from experiments.three_machine import protocol
from experiments.three_machine.mailbox import ClientMailbox, ServiceMailbox

PROOF_TOKEN = 154_321


def payload(rows: int) -> bytes:
    size = rows * protocol.ROW_BYTES
    return bytes(((offset * 131) ^ (offset >> 7) ^ rows) & 0xFF for offset in range(size))


def service(name: str, socket_path: str, timeout_s: float) -> dict:
    with ServiceMailbox(name, socket_path=socket_path) as mailbox:
        request = mailbox.next_request(timeout_s)
        if request is None:
            raise TimeoutError("no activation frame arrived")
        sequence, raw = request
        frame = protocol.unpack(raw)
        if frame.kind is not protocol.Kind.ACTIVATION:
            raise protocol.ProtocolError("real-link proof expected an activation")
        answer = protocol.token(
            request_id=frame.request_id,
            step=frame.step,
            committed=frame.committed + frame.rows,
            generation=frame.generation,
            tokens=[PROOF_TOKEN],
            timeout_s=timeout_s,
        )
        mailbox.reply(sequence, protocol.pack(answer))
        return {
            "role": "service",
            "request_id": frame.request_id,
            "rows": frame.rows,
            "payload_sha256": hashlib.sha256(frame.payload).hexdigest(),
            "reply_token": PROOF_TOKEN,
        }


def client(name: str, rows: int, timeout_s: float) -> dict:
    activation_bytes = payload(rows)
    with ClientMailbox(name, ready_timeout_s=timeout_s) as mailbox:
        frame = protocol.activation(
            request_id=0x54464D43444D41,
            step=0,
            committed=0,
            generation=mailbox.generation,
            rows=rows,
            payload=activation_bytes,
            prompt=True,
            timeout_s=timeout_s,
        )
        reply = protocol.unpack(mailbox.call(protocol.pack(frame), timeout_s), now_ns=time.time_ns())
        if (reply.request_id, reply.step, reply.generation) != (frame.request_id, frame.step, frame.generation):
            raise protocol.ProtocolError("reply identity differs from request")
        tokens = protocol.token_values(reply)
        if tokens != [PROOF_TOKEN] or reply.committed != rows:
            raise protocol.ProtocolError("reply token or committed position differs")
        return {
            "role": "client",
            "request_id": frame.request_id,
            "rows": rows,
            "bytes": len(activation_bytes),
            "payload_sha256": hashlib.sha256(activation_bytes).hexdigest(),
            "reply_token": tokens[0],
            "generation": mailbox.generation,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("client", "service"))
    parser.add_argument("--name", default="tfmcdma")
    parser.add_argument("--socket", default="/tmp/mcdma-rpcd.tfmcdma.sock")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    result = service(args.name, args.socket, args.timeout) if args.role == "service" else client(
        args.name, args.rows, args.timeout
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
