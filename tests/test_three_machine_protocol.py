import struct
import time

import pytest

from experiments.three_machine import protocol
from experiments.three_machine.frame_echo import PROOF_TOKEN, payload


def activation_frame(rows=2):
    return protocol.activation(
        request_id=7,
        step=3,
        committed=11,
        generation=5,
        rows=rows,
        payload=bytes((i * 17) & 0xFF for i in range(rows * protocol.ROW_BYTES)),
        prompt=True,
    )


def test_activation_round_trip():
    frame = activation_frame()
    assert protocol.unpack(protocol.pack(frame)) == frame
    assert protocol.HEADER_BYTES == 104


def test_token_round_trip():
    frame = protocol.token(request_id=9, step=4, committed=14, generation=2, tokens=[3, 19, 154879])
    decoded = protocol.unpack(protocol.pack(frame))
    assert protocol.token_values(decoded) == [3, 19, 154879]


def test_token_bound_window_and_partial_result_round_trip():
    tokens = [10, 11, 12]
    activations = bytes((i * 13) & 0xFF for i in range(len(tokens) * protocol.ROW_BYTES))
    frame = protocol.window_activation(
        request_id=7, step=3, committed=11, generation=5, tokens=tokens,
        payload=activations, prompt=True, final=True,
    )
    decoded = protocol.unpack(protocol.pack(frame))
    assert protocol.window_activation_values(decoded) == (tokens, activations)
    assert decoded.flags == protocol.Flags.PROMPT | protocol.Flags.FINAL

    result = protocol.window_result(
        request_id=7, step=3, committed=13, generation=5,
        accepted=[101, 102], drafts=[201, 202, 203],
    )
    decoded_result = protocol.unpack(protocol.pack(result))
    assert protocol.window_result_values(decoded_result) == ([101, 102], [201, 202, 203])
    assert protocol.checked_window_result(
        decoded_result, request_id=7, step=3, generation=5,
        base_committed=11, input_rows=3, prompt=False,
    ) == ([101, 102], [201, 202, 203], 2)


def test_token_bound_window_stays_inside_fixed_mailbox_payload():
    tokens = list(range(protocol.WINDOW_MAX_ROWS))
    activations = b"\0" * (len(tokens) * protocol.ROW_BYTES)
    frame = protocol.window_activation(
        request_id=1, step=0, committed=0, generation=1, tokens=tokens,
        payload=activations, prompt=True, final=True,
    )
    assert protocol.WINDOW_MAX_ROWS == 64
    assert len(frame.payload) <= protocol.MAX_PAYLOAD
    with pytest.raises(protocol.ProtocolError, match="1..64"):
        protocol.window_activation(
            request_id=1, step=0, committed=0, generation=1,
            tokens=list(range(65)), payload=b"\0" * (65 * protocol.ROW_BYTES), prompt=True,
        )


def test_token_window_and_verified_hidden_round_trip():
    frame = protocol.token_window(
        request_id=7,
        step=3,
        committed=11,
        generation=5,
        tokens=[10, 11, 12],
        prompt=False,
    )
    decoded = protocol.unpack(protocol.pack(frame))
    assert protocol.token_window_values(decoded) == [10, 11, 12]

    hidden = bytes((i * 7) & 0xFF for i in range(2 * protocol.ROW_BYTES))
    result = protocol.verified_hidden(
        request_id=7,
        step=3,
        committed=13,
        generation=5,
        accepted=[101, 102],
        hidden=hidden,
        rows=2,
        prompt=False,
    )
    decoded_result = protocol.unpack(protocol.pack(result))
    assert protocol.verified_hidden_values(decoded_result) == ([101, 102], hidden)
    assert protocol.checked_verified_hidden(
        decoded_result,
        request_id=7,
        step=3,
        generation=5,
        base_committed=11,
        input_rows=3,
        prompt=False,
        final=False,
    ) == ([101, 102], hidden, 2)


def test_window_result_rejects_invalid_commit_relationships():
    result = protocol.window_result(
        request_id=7, step=3, committed=12, generation=5, accepted=[101, 102], drafts=[]
    )
    with pytest.raises(protocol.ProtocolError, match="one input row per accepted"):
        protocol.checked_window_result(
            result, request_id=7, step=3, generation=5,
            base_committed=11, input_rows=3, prompt=False,
        )
    with pytest.raises(protocol.ProtocolError, match="keep every input row"):
        protocol.checked_window_result(
            result, request_id=7, step=3, generation=5,
            base_committed=11, input_rows=3, prompt=True,
        )


def test_checked_token_values_require_exact_identity_and_count():
    frame = protocol.token(request_id=9, step=4, committed=14, generation=2, tokens=[19])
    assert protocol.checked_token_values(
        frame, request_id=9, step=4, committed=14, generation=2
    ) == [19]
    with pytest.raises(protocol.ProtocolError, match="identity differs"):
        protocol.checked_token_values(frame, request_id=8, step=4, committed=14, generation=2)
    with pytest.raises(protocol.ProtocolError, match="exactly 2"):
        protocol.checked_token_values(
            frame, request_id=9, step=4, committed=14, generation=2, count=2
        )


@pytest.mark.parametrize(
    "change", ["payload", "trailing", "truncated", "kind", "reserved", "request", "deadline"]
)
def test_corruption_fails(change):
    raw = bytearray(protocol.pack(activation_frame(1)))
    if change == "payload":
        raw[-1] ^= 1
    elif change == "trailing":
        raw += b"x"
    elif change == "truncated":
        raw.pop()
    elif change == "kind":
        struct.pack_into(">H", raw, 10, 65535)
    elif change == "request":
        raw[16] ^= 1
    elif change == "deadline":
        raw[64] ^= 1
    else:
        struct.pack_into(">H", raw, 58, 1)
    with pytest.raises(protocol.ProtocolError):
        protocol.unpack(raw)


def test_shape_and_deadline_fail_closed():
    frame = activation_frame(1)
    bad = protocol.Frame(frame.kind, frame.request_id, frame.step, frame.committed, frame.generation, 2,
                         frame.width, frame.dtype, frame.payload, frame.deadline_ns, frame.flags)
    with pytest.raises(protocol.ProtocolError):
        protocol.pack(bad)
    stale = protocol.Frame(frame.kind, frame.request_id, frame.step, frame.committed, frame.generation, frame.rows,
                           frame.width, frame.dtype, frame.payload, time.time_ns() - 1, frame.flags)
    with pytest.raises(protocol.ProtocolError):
        protocol.pack(stale)
    zero = protocol.Frame(
        frame.kind,
        frame.request_id,
        frame.step,
        frame.committed,
        frame.generation,
        frame.rows,
        frame.width,
        frame.dtype,
        frame.payload,
        0,
        frame.flags,
    )
    with pytest.raises(protocol.ProtocolError, match="nonzero"):
        protocol.pack(zero)


def test_activation_rejects_unknown_flags():
    frame = activation_frame(1)
    bad = protocol.Frame(
        frame.kind,
        frame.request_id,
        frame.step,
        frame.committed,
        frame.generation,
        frame.rows,
        frame.width,
        frame.dtype,
        frame.payload,
        frame.deadline_ns,
        protocol.Flags(int(frame.flags) | 4),
    )
    with pytest.raises(protocol.ProtocolError, match="unsupported flag"):
        protocol.pack(bad)


def test_real_link_fixture_is_full_prefill_window_and_deterministic():
    first = payload(protocol.MAX_ROWS)
    assert len(first) == protocol.ACTIVATION_MAX_PAYLOAD == 2 << 20
    assert protocol.MAX_PAYLOAD == protocol.MAX_ROWS * protocol.WINDOW_ROW_BYTES
    assert first == payload(protocol.MAX_ROWS)
    assert 0 <= PROOF_TOKEN < 154_880


def test_control_and_error_round_trip():
    ack = protocol.control(kind=protocol.Kind.ACK, request_id=7, step=1, committed=3, generation=2)
    assert protocol.unpack(protocol.pack(ack)).kind is protocol.Kind.ACK
    failure = protocol.error(request_id=7, step=1, committed=3, generation=2, message="bad frame")
    assert protocol.unpack(protocol.pack(failure)).payload == b"bad frame"


def test_control_rejects_payload_kind():
    with pytest.raises(protocol.ProtocolError):
        protocol.control(kind=protocol.Kind.TOKEN, request_id=1, step=0, committed=0, generation=1)
