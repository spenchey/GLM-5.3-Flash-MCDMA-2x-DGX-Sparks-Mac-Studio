"""Serve a verified TensorFold prompt cache over MCDMA without using a GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from experiments.three_machine.cache_handoff import (
    HandoffError,
    StoredTensorArchive,
    decode_request,
    pack_ack,
    pack_error,
    pack_manifest,
    prompt_digest,
    reply_frame,
)
from experiments.three_machine.mailbox import ServiceMailbox


def serve(archive: StoredTensorArchive, mailbox_name: str, socket_path: str, timeout: float) -> dict:
    active = False
    served = 0
    with ServiceMailbox(mailbox_name, socket_path=socket_path) as mailbox:
        while True:
            incoming = mailbox.next_request(timeout)
            if incoming is None:
                continue
            sequence, raw = incoming
            try:
                request = decode_request(raw)
                if request.operation == "prepare":
                    archive.manifest.require_identity(
                        model_id=request.model_id or "",
                        model_revision=request.model_revision or "",
                        prompt_sha256=prompt_digest(request.tokens),
                        cached_tokens=len(request.tokens) - 1,
                    )
                    active = True
                    mailbox.reply(sequence, pack_manifest(archive.manifest))
                elif request.operation == "chunk":
                    if not active or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("requested cache transfer is not active")
                    mailbox.reply(sequence, archive.chunk(
                        request.tensor or "", request.offset, request.length
                    ))
                elif request.operation == "frame":
                    if not active or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("requested cache transfer is not active")
                    reply_frame(mailbox, sequence, archive, request.offset, request.length)
                elif request.operation == "release":
                    if not active or request.transfer_id != archive.manifest.transfer_id:
                        raise HandoffError("released cache transfer is not active")
                    active = False
                    served += 1
                    mailbox.reply(sequence, pack_ack("release", archive.manifest.transfer_id))
                elif request.operation == "shutdown":
                    reply = pack_ack("shutdown")
                    mailbox.reply(sequence, reply)
                    mailbox.wait_reply_accepted(sequence, len(reply), min(timeout, 30.0))
                    return {"served": served, "transfer_id": archive.manifest.transfer_id}
                else:
                    raise HandoffError(f"unsupported cache operation {request.operation!r}")
            except Exception as exc:
                print(json.dumps({
                    "event": "glm_cache_replay_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }, sort_keys=True), file=sys.stderr, flush=True)
                mailbox.reply(sequence, pack_error(exc))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--mailbox", default="p06cfr")
    parser.add_argument("--socket", default="/tmp/mcdma-rpcd.p06cfr.sock")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    archive = StoredTensorArchive(args.bundle)
    try:
        result = serve(archive, args.mailbox, args.socket, args.timeout)
    finally:
        archive.close()
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
