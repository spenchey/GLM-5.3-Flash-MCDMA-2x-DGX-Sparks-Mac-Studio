import pytest

from experiments.three_machine import cuda_partial, protocol


class Done(Exception):
    pass


class FakeMailbox:
    def __init__(self, packed):
        self.packed = packed
        self.calls = 0
        self.generation = None
        self.replies = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def next_request(self, _timeout):
        self.calls += 1
        if self.calls == 1:
            return 19, self.packed
        raise Done

    def bind_generation(self, generation):
        if self.generation is None:
            self.generation = generation
        elif self.generation != generation:
            raise RuntimeError("changed generation")

    def reply(self, sequence, payload):
        self.replies.append((sequence, protocol.unpack(payload)))


class SequenceMailbox(FakeMailbox):
    def __init__(self, packed_frames):
        super().__init__(b"")
        self.packed_frames = list(packed_frames)

    def next_request(self, _timeout):
        if not self.packed_frames:
            raise Done
        self.calls += 1
        return self.calls, self.packed_frames.pop(0)


class FakeSession:
    def __init__(self, generation):
        self.generation = generation
        self.request_id = 9
        self.step = 1
        self.committed = 2050

    def classify_activation(self, _frame):
        return None


def test_verified_window_advances_before_mtp_draft_and_ack_does_not_double_commit(monkeypatch):
    events = []
    engine = cuda_partial.PartialCUDA.__new__(cuda_partial.PartialCUDA)
    engine.w = object()
    engine.state = type("State", (), {"pos": 20})()
    engine.pending = None
    engine.draft_count = 3
    engine.mtp_confidence = 0.35
    buffer = object()
    hidden = object()

    def fake_commit(weights, state, value, rows, keep):
        assert weights is engine.w
        assert value is buffer
        events.append(("commit", state.pos, rows, keep))
        state.pos += keep

    def fake_draft(value, draft_hidden, accepted):
        assert value is engine
        assert draft_hidden is hidden
        assert accepted == [42, 43]
        assert engine.pending.state_advanced is True
        events.append(("draft", engine.state.pos))
        return [44, 45]

    monkeypatch.setattr(cuda_partial, "_tensorfold_commit", fake_commit)
    monkeypatch.setattr(cuda_partial, "_tensorfold_draft", fake_draft)

    engine._advance_window(buffer, 3, 2)
    assert cuda_partial._tensorfold_draft(engine, hidden, [42, 43]) == [44, 45]
    engine.commit(2)

    assert events == [("commit", 20, 3, 2), ("draft", 22)]
    assert engine.pending is None
    assert engine.state.pos == 22


def test_legacy_activation_still_commits_on_ack(monkeypatch):
    calls = []
    engine = cuda_partial.PartialCUDA.__new__(cuda_partial.PartialCUDA)
    engine.w = object()
    engine.state = object()
    buffer = object()
    engine.pending = cuda_partial.PendingCUDA(buffer, 4, 4, False)
    monkeypatch.setattr(
        cuda_partial,
        "_tensorfold_commit",
        lambda weights, state, value, rows, keep: calls.append(
            (weights, state, value, rows, keep)
        ),
    )

    engine.commit(4)

    assert calls == [(engine.w, engine.state, buffer, 4, 4)]
    assert engine.pending is None


def test_rank_zero_rejects_context_overflow_before_collective_action(monkeypatch):
    frame = protocol.activation(
        request_id=9,
        step=1,
        committed=2050,
        generation=7,
        rows=2,
        payload=b"x" * (2 * protocol.ROW_BYTES),
        prompt=False,
    )
    mailbox = FakeMailbox(protocol.pack(frame))
    controls = []
    monkeypatch.setattr(cuda_partial, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_partial, "Session", FakeSession)
    monkeypatch.setattr(
        cuda_partial,
        "_control",
        lambda _comm, _rank, values: controls.append(values),
    )
    engine = type("Engine", (), {"comm": object(), "context_capacity": 2051})()

    with pytest.raises(Done):
        cuda_partial.rank_zero(engine, "test", "/tmp/test.sock", 1)

    assert controls == [(cuda_partial.ACTION_NOOP, 0, 0, 0, 0, 0)]
    assert mailbox.replies[0][0] == 19
    assert mailbox.replies[0][1].kind is protocol.Kind.ERROR
    assert b"supports 2051" in mailbox.replies[0][1].payload


def test_rank_zero_reports_precollective_errors(monkeypatch, capsys):
    frame = protocol.activation(
        request_id=9,
        step=1,
        committed=2050,
        generation=7,
        rows=2,
        payload=b"x" * (2 * protocol.ROW_BYTES),
        prompt=False,
    )
    mailbox = FakeMailbox(protocol.pack(frame))
    monkeypatch.setattr(cuda_partial, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_partial, "Session", FakeSession)
    monkeypatch.setattr(cuda_partial, "_control", lambda *_args, **_kwargs: None)
    engine = type("Engine", (), {"comm": object(), "context_capacity": 2051})()

    with pytest.raises(Done):
        cuda_partial.rank_zero(engine, "test", "/tmp/test.sock", 1)

    captured = capsys.readouterr()
    assert '"event":"three_machine_error"' in captured.err
    assert '"error_type":"ProtocolError"' in captured.err


def test_rank_zero_completes_activation_ack_reset_and_shutdown(monkeypatch, capsys):
    generation = 7
    request_id = 9
    frames = [
        protocol.activation(
            request_id=request_id,
            step=0,
            committed=0,
            generation=generation,
            rows=1,
            payload=b"x" * protocol.ROW_BYTES,
            prompt=True,
        ),
        protocol.control(
            kind=protocol.Kind.ACK,
            request_id=request_id,
            step=0,
            committed=1,
            generation=generation,
            timeout_s=1,
        ),
        protocol.control(
            kind=protocol.Kind.RESET,
            request_id=request_id,
            step=1,
            committed=1,
            generation=generation,
            timeout_s=1,
        ),
        protocol.control(
            kind=protocol.Kind.SHUTDOWN,
            request_id=request_id,
            step=0,
            committed=0,
            generation=generation,
            timeout_s=1,
        ),
    ]
    mailbox = SequenceMailbox(protocol.pack(frame) for frame in frames)
    controls = []
    monkeypatch.setattr(cuda_partial, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(
        cuda_partial,
        "_control",
        lambda _comm, _rank, values: controls.append(values),
    )
    activation = object()
    monkeypatch.setattr(
        cuda_partial,
        "_activation",
        lambda _comm, _rank, _payload, _rows: activation,
    )
    monkeypatch.setattr(cuda_partial, "_rank_memory", lambda: {
        "cuda_allocated_bytes": 1,
        "cuda_reserved_bytes": 2,
        "process_rss_bytes": 3,
    })

    class Engine:
        comm = object()
        context_capacity = 2051
        phase_timing = False
        last_run_metrics = {"gpu_total_s": 0.01}

        def __init__(self):
            self.commits = 0
            self.resets = 0

        def run(self, value, rows):
            assert value is activation
            assert rows == 1
            return 42

        def commit(self, keep=None):
            assert keep == 1
            self.commits += 1

        def reset(self):
            self.resets += 1

    engine = Engine()
    result = cuda_partial.rank_zero(engine, "test", "/tmp/test.sock", 1)

    assert result == {"rank": 0, "committed_windows": 1, "committed_tokens": 0}
    assert engine.commits == 1
    assert engine.resets == 1
    assert controls == [
        (cuda_partial.ACTION_ACTIVATE, 1, int(protocol.Flags.PROMPT), 0, 0, request_id),
        (cuda_partial.ACTION_ACK, 1, 1, 0, 1, request_id),
        (cuda_partial.ACTION_RESET, 0, 0, 0, 0, request_id),
        (cuda_partial.ACTION_SHUTDOWN, 0, 0, 0, 0, request_id),
    ]
    assert [frame.kind for _, frame in mailbox.replies] == [
        protocol.Kind.TOKEN,
        protocol.Kind.ACK,
        protocol.Kind.ACK,
        protocol.Kind.ACK,
    ]
    output = capsys.readouterr().out
    assert '"prompt":true' in output
    assert '"event":"three_machine_rank0_commit"' in output
    assert '"cuda_allocated_bytes":1' in output


def test_rank_one_reports_every_committed_window(monkeypatch, capsys):
    actions = iter([
        (cuda_partial.ACTION_ACTIVATE, 1, int(protocol.Flags.PROMPT), 0, 0, 9),
        (cuda_partial.ACTION_ACK, 1, 1, 0, 1, 9),
        (cuda_partial.ACTION_RESET, 0, 0, 0, 0, 9),
        (cuda_partial.ACTION_NOOP, 0, 0, 0, 0, 0),
        (cuda_partial.ACTION_DUPLICATE, 0, 0, 0, 0, 9),
        (cuda_partial.ACTION_SHUTDOWN, 0, 0, 0, 0, 9),
    ])
    monkeypatch.setattr(cuda_partial, "_control", lambda _comm, _rank: next(actions))
    activation = object()
    monkeypatch.setattr(cuda_partial, "_activation", lambda *_args: activation)
    monkeypatch.setattr(cuda_partial, "_rank_memory", lambda: {
        "cuda_allocated_bytes": 1,
        "cuda_reserved_bytes": 2,
        "process_rss_bytes": 3,
    })

    class Engine:
        comm = object()

        def __init__(self):
            self.commits = 0
            self.resets = 0
            self.state = type("State", (), {"pos": 0})()

        def run(self, value, rows):
            assert value is activation
            assert rows == 1

        def commit(self, keep=None):
            assert keep == 1
            self.commits += 1
            self.state.pos += 1

        def reset(self):
            self.resets += 1
            self.state.pos = 0

    engine = Engine()
    result = cuda_partial.rank_one(engine)

    assert result == {"rank": 1, "committed_windows": 1, "committed_tokens": 0}
    assert engine.commits == 1
    assert engine.resets == 1
    output = capsys.readouterr().out
    assert '"event":"three_machine_rank1_commit"' in output
    assert '"request_id":9' in output
    assert '"committed_windows":1' in output
    assert '"cuda_reserved_bytes":2' in output


def test_rank_zero_returns_verified_tokens_and_next_mtp_window(monkeypatch):
    frame = protocol.window_activation(
        request_id=17,
        step=0,
        committed=0,
        generation=7,
        tokens=[10, 11],
        payload=b"x" * (2 * protocol.ROW_BYTES),
        prompt=True,
        final=True,
        timeout_s=1,
    )
    mailbox = FakeMailbox(protocol.pack(frame))
    controls = []
    activation = object()
    monkeypatch.setattr(cuda_partial, "ServiceMailbox", lambda *_args, **_kwargs: mailbox)
    monkeypatch.setattr(cuda_partial, "_control", lambda _comm, _rank, values: controls.append(values))
    monkeypatch.setattr(cuda_partial, "_activation", lambda *_args: activation)
    monkeypatch.setattr(cuda_partial, "_token_ids", lambda *_args: [10, 11])

    class Engine:
        comm = object()
        context_capacity = 2051
        phase_timing = False
        last_run_metrics = {"verified_drafts": 0, "accepted_drafts": 0, "next_drafts": 2}

        def run_window(self, value, tokens, *, prompt, final):
            assert value is activation
            assert tokens == [10, 11]
            assert prompt is True
            assert final is True
            return cuda_partial.WindowRun([42], [43, 44], 2)

    with pytest.raises(Done):
        cuda_partial.rank_zero(Engine(), "test", "/tmp/test.sock", 1)

    assert controls == [
        (cuda_partial.ACTION_WINDOW_ACTIVATE, 2, int(protocol.Flags.PROMPT | protocol.Flags.FINAL), 0, 0, 17),
    ]
    reply = mailbox.replies[0][1]
    assert reply.kind is protocol.Kind.WINDOW_RESULT
    assert reply.committed == 2
    assert protocol.window_result_values(reply) == ([42], [43, 44])


def test_rank_one_runs_token_bound_window_and_partial_commit(monkeypatch):
    actions = iter([
        (cuda_partial.ACTION_WINDOW_ACTIVATE, 3, int(protocol.Flags.DECODE), 4, 20, 9),
        (cuda_partial.ACTION_ACK, 3, 2, 4, 22, 9),
        (cuda_partial.ACTION_SHUTDOWN, 0, 0, 0, 0, 9),
    ])
    activation = object()
    monkeypatch.setattr(cuda_partial, "_control", lambda _comm, _rank: next(actions))
    monkeypatch.setattr(cuda_partial, "_activation", lambda *_args: activation)
    monkeypatch.setattr(cuda_partial, "_token_ids", lambda *_args: [50, 51, 52])
    monkeypatch.setattr(cuda_partial, "_rank_memory", lambda: {
        "cuda_allocated_bytes": 1,
        "cuda_reserved_bytes": 2,
        "process_rss_bytes": 3,
    })

    class Engine:
        comm = object()

        def __init__(self):
            self.keep = None
            self.state = type("State", (), {"pos": 22})()

        def run_window(self, value, tokens, *, prompt, final):
            assert value is activation
            assert tokens == [50, 51, 52]
            assert prompt is False
            assert final is False
            return cuda_partial.WindowRun([51, 52], [53], 2)

        def commit(self, keep):
            self.keep = keep

    engine = Engine()
    result = cuda_partial.rank_one(engine)

    assert engine.keep == 2
    assert result == {"rank": 1, "committed_windows": 1, "committed_tokens": 22}
