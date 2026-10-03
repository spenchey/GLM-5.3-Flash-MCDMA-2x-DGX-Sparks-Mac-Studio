from experiments.three_machine.cache_control import shutdown
from experiments.three_machine.cache_handoff import pack_ack


class Mailbox:
    def __init__(self, answer: bytes):
        self.answer = answer
        self.request = b""
        self.timeout = 0.0

    def call(self, request: bytes, timeout: float) -> bytes:
        self.request = request
        self.timeout = timeout
        return self.answer


def test_shutdown_requires_matching_acknowledgement():
    mailbox = Mailbox(pack_ack("shutdown"))
    assert shutdown(mailbox, 7.0) == {"action": "shutdown", "acknowledged": True}
    assert b'"operation":"shutdown"' in mailbox.request
    assert mailbox.timeout == 7.0
