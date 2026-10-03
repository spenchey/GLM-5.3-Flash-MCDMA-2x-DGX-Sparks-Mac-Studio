import pytest

from experiments.three_machine import protocol
from experiments.three_machine.transaction import Session


def activation(*, request=9, step=0, committed=0, generation=3, value=1):
    return protocol.activation(request_id=request, step=step, committed=committed, generation=generation, rows=1,
                               payload=bytes([value]) * protocol.ROW_BYTES, prompt=step == 0)


def test_two_phase_commit_and_exact_retry():
    session = Session(3)
    frame = activation()
    assert session.classify_activation(frame) is None
    reply = session.propose(frame, [17])
    assert session.classify_activation(frame) == reply
    pending = session.validate_ack(protocol.control(kind=protocol.Kind.ACK, request_id=9, step=0, committed=1,
                                                    generation=3))
    assert session.committed == 0
    session.commit(pending)
    assert (session.step, session.committed, session.pending) == (1, 1, None)
    with pytest.raises(protocol.ProtocolError, match="no pending"):
        session.validate_ack(protocol.control(
            kind=protocol.Kind.ACK,
            request_id=9,
            step=0,
            committed=1,
            generation=3,
        ))
    with pytest.raises(protocol.ProtocolError, match="out of order"):
        session.classify_activation(frame)


def test_verification_window_partially_commits_only_after_exact_ack():
    session = Session(3)
    frame = protocol.window_activation(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        tokens=[100, 200, 201, 202],
        payload=b"\0" * (4 * protocol.ROW_BYTES),
        prompt=False,
    )
    assert session.classify_activation(frame) is None
    reply = session.propose_window(frame, [101, 102], [203, 204], keep=2)
    assert protocol.window_result_values(reply) == ([101, 102], [203, 204])
    assert reply.committed == 2
    assert session.classify_activation(frame) == reply
    pending = session.validate_ack(protocol.control(
        kind=protocol.Kind.ACK, request_id=9, step=0, committed=2, generation=3
    ))
    assert (pending.rows, pending.keep, session.committed) == (4, 2, 0)
    session.commit(pending)
    assert (session.step, session.committed, session.pending) == (1, 2, None)


def test_window_proposal_rejects_result_that_disagrees_with_keep():
    session = Session(3)
    frame = protocol.window_activation(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        tokens=[100, 200, 201],
        payload=b"\0" * (3 * protocol.ROW_BYTES),
        prompt=False,
    )
    session.classify_activation(frame)
    with pytest.raises(protocol.ProtocolError, match="one accepted token per kept row"):
        session.propose_window(frame, [101], [203], keep=2)


def test_token_window_proposes_verified_hidden_with_two_phase_commit():
    session = Session(3)
    frame = protocol.token_window(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        tokens=[100, 200, 201],
        prompt=False,
    )
    assert session.classify_activation(frame) is None
    hidden = b"\0" * (2 * protocol.ROW_BYTES)
    reply = session.propose_hidden(frame, [101, 102], hidden, keep=2)
    assert protocol.verified_hidden_values(reply) == ([101, 102], hidden)
    assert reply.committed == 2
    assert session.classify_activation(frame) == reply
    pending = session.validate_ack(protocol.control(
        kind=protocol.Kind.ACK, request_id=9, step=0, committed=2, generation=3
    ))
    assert session.committed == 0
    session.commit(pending)
    assert (session.step, session.committed) == (1, 2)


def test_prompt_hidden_samples_only_on_final_chunk():
    session = Session(3)
    intermediate = protocol.token_window(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        tokens=[10, 11],
        prompt=True,
    )
    session.classify_activation(intermediate)
    hidden = b"\0" * (2 * protocol.ROW_BYTES)
    reply = session.propose_hidden(intermediate, [], hidden, keep=2)
    assert protocol.verified_hidden_values(reply) == ([], hidden)

    invalid = Session(3)
    final = protocol.token_window(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        tokens=[10, 11],
        prompt=True,
        final=True,
    )
    invalid.classify_activation(final)
    with pytest.raises(protocol.ProtocolError, match="sample only at the final prompt"):
        invalid.propose_hidden(final, [], hidden, keep=2)


def test_changed_retry_and_out_of_order_frames_fail_closed():
    session = Session(3)
    first = activation()
    session.classify_activation(first)
    session.propose(first, [17])
    with pytest.raises(protocol.ProtocolError, match="different activation"):
        session.classify_activation(activation(value=2))
    with pytest.raises(protocol.ProtocolError, match="does not match"):
        session.validate_ack(protocol.control(kind=protocol.Kind.ACK, request_id=9, step=0, committed=2,
                                              generation=3))


def test_retry_must_keep_prompt_mode_and_gets_a_fresh_valid_deadline():
    session = Session(3)
    first = activation()
    session.classify_activation(first)
    reply = session.propose(first, [17])
    retry = protocol.activation(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        rows=1,
        payload=first.payload,
        prompt=True,
        timeout_s=60,
    )
    cached = session.classify_activation(retry)
    assert cached is not None
    assert cached.payload == reply.payload
    assert cached.deadline_ns == retry.deadline_ns
    decode_retry = protocol.activation(
        request_id=9,
        step=0,
        committed=0,
        generation=3,
        rows=1,
        payload=first.payload,
        prompt=False,
    )
    with pytest.raises(protocol.ProtocolError, match="different activation"):
        session.classify_activation(decode_retry)


def test_reset_allows_a_new_request_but_rejects_replay_of_the_reset_request():
    session = Session(3)
    frame = activation()
    session.classify_activation(frame)
    session.propose(frame, [17])
    session.reset()
    assert session.classify_activation(activation(request=10)) is None
    session.reset()
    with pytest.raises(protocol.ProtocolError, match="reset inference stream"):
        session.classify_activation(frame)
    with pytest.raises(protocol.ProtocolError, match="different MCDMA"):
        session.classify_activation(activation(request=10, generation=4))


def test_reset_and_shutdown_validate_generation_request_and_position():
    session = Session(3)
    frame = activation()
    session.classify_activation(frame)
    session.propose(frame, [17])
    session.validate_reset(protocol.control(
        kind=protocol.Kind.RESET, request_id=9, step=0, committed=0, generation=3
    ))
    with pytest.raises(protocol.ProtocolError, match="another inference stream"):
        session.validate_reset(protocol.control(
            kind=protocol.Kind.RESET, request_id=10, step=0, committed=0, generation=3
        ))
    with pytest.raises(protocol.ProtocolError, match="different MCDMA"):
        session.validate_reset(protocol.control(
            kind=protocol.Kind.RESET, request_id=9, step=0, committed=0, generation=4
        ))
    with pytest.raises(protocol.ProtocolError, match="unacknowledged"):
        session.validate_shutdown(protocol.control(
            kind=protocol.Kind.SHUTDOWN, request_id=9, step=0, committed=0, generation=3
        ))
    session.reset()
    with pytest.raises(protocol.ProtocolError, match="position differs"):
        session.validate_shutdown(protocol.control(
            kind=protocol.Kind.SHUTDOWN, request_id=10, step=1, committed=0, generation=3
        ))
    session.validate_shutdown(protocol.control(
        kind=protocol.Kind.SHUTDOWN, request_id=10, step=0, committed=0, generation=3
    ))
