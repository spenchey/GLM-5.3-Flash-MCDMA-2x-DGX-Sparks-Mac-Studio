import pytest

from experiments.three_machine import protocol
from experiments.three_machine.control import _exchange


class Mailbox:
    def __init__(self, reply):
        self.reply = protocol.pack(reply)

    def call(self, _payload, _timeout):
        return self.reply


def frame(generation=7):
    return protocol.control(kind=protocol.Kind.RESET, request_id=11, step=0, committed=0,
                            generation=generation)


def test_control_exchange_accepts_only_the_exact_ack_identity():
    expected = frame()
    reply = protocol.control(kind=protocol.Kind.ACK, request_id=11, step=0, committed=0, generation=7)
    assert _exchange(Mailbox(reply), expected, 1) == reply


@pytest.mark.parametrize("field,value", [("request_id", 12), ("step", 1), ("committed", 1), ("generation", 8)])
def test_control_exchange_rejects_changed_identity(field, value):
    expected = frame()
    values = {"request_id": 11, "step": 0, "committed": 0, "generation": 7}
    values[field] = value
    reply = protocol.control(kind=protocol.Kind.ACK, **values)
    with pytest.raises(protocol.ProtocolError, match="identity differs"):
        _exchange(Mailbox(reply), expected, 1)
